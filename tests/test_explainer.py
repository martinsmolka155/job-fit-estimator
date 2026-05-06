"""Tests for Explainer — LLM-based career advisor.

IMPORTANT: All tests use pytest-mock for LLMProvider — NO real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.explainer import MAX_RETRIES, Explainer
from src.schemas import (
    Explanation,
    Recommendation,
    Resume,
    SalaryEstimate,
    ScoreComponent,
    SeniorityScore,
)


def _make_recommendation(impact_pct: float) -> Recommendation:
    return Recommendation(
        title="Test recommendation",
        why_it_matters="Test reason",
        estimated_salary_impact_pct=impact_pct,
        timeframe_months=6,
        first_action="Test action",
    )


def _make_explanation(impacts: list[float]) -> Explanation:
    return Explanation(
        summary="Test summary",
        strengths=["Strength 1", "Strength 2"],
        gaps=["Gap 1", "Gap 2"],
        recommendations=[_make_recommendation(i) for i in impacts],
    )


def _make_score() -> SeniorityScore:
    return SeniorityScore(
        total=65.0,
        components=[ScoreComponent(name="Test", score=65.0, weight=1.0, reasoning="test")],
        confidence=0.9,
    )


def _make_salary() -> SalaryEstimate:
    return SalaryEstimate(
        currency="CZK",
        low=90000,
        mid=110000,
        high=130000,
        reasoning="Test reasoning",
        assumptions=["senior band", "Praha"],
    )


def _make_resume() -> Resume:
    return Resume(full_name="Test User", location="Praha", raw_text_length=1000)


def _make_mock_llm(explanations: list[Explanation]) -> MagicMock:
    """Mock LLM that returns explanations in sequence."""
    llm = MagicMock()
    call_count = 0

    def side_effect(*args: object, **kwargs: object) -> tuple[Explanation, dict]:
        nonlocal call_count
        idx = min(call_count, len(explanations) - 1)
        result = explanations[idx]
        call_count += 1
        return result, {
            "cost_usd": 0.01,
            "input_tokens": 500,
            "output_tokens": 300,
            "duration_s": 1.5,
            "model": "gpt-4o-mini",
        }

    llm.extract_structured.side_effect = side_effect
    return llm


class TestExplainerSuccess:
    def test_succeeds_on_first_attempt_with_34pct_impact(self) -> None:
        """Impact 12+10+12=34% → succeeds on first attempt, retries=0."""
        explanation = _make_explanation([12.0, 10.0, 12.0])
        llm = _make_mock_llm([explanation])
        explainer = Explainer(llm)

        result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert isinstance(result, Explanation)
        assert len(result.recommendations) == 3
        total_impact = sum(r.estimated_salary_impact_pct for r in result.recommendations)
        assert total_impact >= 30.0
        assert meta["retries"] == 0
        assert llm.extract_structured.call_count == 1

    def test_max_retries_constant_is_3(self) -> None:
        """MAX_RETRIES module constant must be exactly 3."""
        assert MAX_RETRIES == 3


class TestExplainerRetry:
    def test_low_impact_triggers_retry_then_warning(self) -> None:
        """Impact 5+5+5=15% → all 3 retries fail, returns with warning in meta."""
        low_impact = _make_explanation([5.0, 5.0, 5.0])
        llm = _make_mock_llm([low_impact, low_impact, low_impact])
        explainer = Explainer(llm)

        result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert isinstance(result, Explanation)
        assert meta.get("warning") == "below_30pct_target"
        assert meta.get("retries") == MAX_RETRIES
        # Must have been called exactly MAX_RETRIES times
        assert llm.extract_structured.call_count == MAX_RETRIES

    def test_no_fourth_attempt_after_three_failures(self) -> None:
        """Verifies hardcap: exactly MAX_RETRIES=3 attempts, never 4."""
        low_impact = _make_explanation([5.0, 5.0, 5.0])
        llm = _make_mock_llm([low_impact] * 10)  # plenty of low impact responses
        explainer = Explainer(llm)

        explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert llm.extract_structured.call_count == MAX_RETRIES, (
            f"Expected exactly {MAX_RETRIES} calls, got {llm.extract_structured.call_count}"
        )

    def test_succeeds_on_second_attempt(self) -> None:
        """First attempt fails (15% impact), second succeeds (34% impact)."""
        low_impact = _make_explanation([5.0, 5.0, 5.0])
        good_impact = _make_explanation([12.0, 10.0, 12.0])
        llm = _make_mock_llm([low_impact, good_impact])
        explainer = Explainer(llm)

        result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert meta.get("warning") is None, "Should succeed — no warning expected"
        assert meta["retries"] == 1  # 0-indexed, so 1 means second attempt succeeded
        assert llm.extract_structured.call_count == 2


class TestExplainerEdgeCases:
    def test_retry_prompt_includes_feedback_message(self) -> None:
        """Second attempt prompt must include feedback about failed impact."""
        low_impact = _make_explanation([5.0, 5.0, 5.0])
        good_impact = _make_explanation([12.0, 10.0, 12.0])
        llm = _make_mock_llm([low_impact, good_impact])
        explainer = Explainer(llm)

        explainer.explain(_make_resume(), _make_score(), _make_salary())

        # Second call (index 1) should have feedback in the prompt
        second_call_args = llm.extract_structured.call_args_list[1]
        second_prompt = second_call_args.args[0]
        assert "PREVIOUS ATTEMPT FAILED" in second_prompt
        assert "30" in second_prompt


class TestExplainerErrorAndWarningExclusivity:
    """Verify that error='llm_failed' and warning='below_30pct_target' are
    mutually exclusive in meta — they describe two different failure modes
    and stacking them confuses downstream metrics.
    """

    def test_all_calls_failing_sets_error_only(self) -> None:
        """When every LLM attempt raises, meta must carry error but NOT warning."""
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.extract_structured.side_effect = RuntimeError("simulated outage")
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert meta.get("error") == "llm_failed"
        assert "warning" not in meta, (
            f"Expected no 'warning' key when LLM totally failed, got meta={meta}"
        )

    def test_low_quality_responses_set_warning_only(self) -> None:
        """When LLM returned 3 low-impact recs every time, meta must carry warning but NOT error."""
        low_impact = _make_explanation([5.0, 5.0, 5.0])
        llm = _make_mock_llm([low_impact, low_impact, low_impact])
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert meta.get("warning") == "below_30pct_target"
        assert "error" not in meta, (
            f"Expected no 'error' key when LLM responded but quality was low, got meta={meta}"
        )
