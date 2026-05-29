"""Tests for Explainer — LLM-based career advisor.

IMPORTANT: All tests use pytest-mock for LLMProvider — NO real API calls.

Contract changes (Fix 4 / Fix 5):
- Recommendation no longer has estimated_salary_impact_pct.
- It now has impact_tier (Literal["medium","high","transformative"]) and
  impact_range_pct (ImpactRange with low_pct / high_pct).
- The ≥30% sum constraint and its retry-escalation logic have been removed.
- Retry now only happens on LLM call failure or wrong recommendation count.
- Cost is ACCUMULATED across retries (Fix 5).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.explainer import MAX_RETRIES, Explainer
from src.schemas import (
    Explanation,
    ImpactRange,
    Recommendation,
    Resume,
    SalaryEstimate,
    ScoreComponent,
    SeniorityScore,
)


def _make_recommendation(
    impact_tier: str = "medium",
    low_pct: float = 8.0,
    high_pct: float = 15.0,
) -> Recommendation:
    return Recommendation(
        title="Test recommendation",
        why_it_matters="Test reason",
        impact_tier=impact_tier,  # type: ignore[arg-type]
        impact_range_pct=ImpactRange(low_pct=low_pct, high_pct=high_pct),
        timeframe_months=6,
        first_action="Test action",
    )


def _make_explanation(n_recs: int = 3) -> Explanation:
    return Explanation(
        summary="Test summary",
        strengths=["Strength 1", "Strength 2"],
        gaps=["Gap 1", "Gap 2"],
        recommendations=[_make_recommendation() for _ in range(n_recs)],
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
    def test_succeeds_on_first_attempt_with_3_recs(self) -> None:
        """Exactly 3 recommendations → succeeds on first attempt, retries=0."""
        explanation = _make_explanation(n_recs=3)
        llm = _make_mock_llm([explanation])
        explainer = Explainer(llm)

        result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert isinstance(result, Explanation)
        assert len(result.recommendations) == 3
        assert meta["retries"] == 0
        assert llm.extract_structured.call_count == 1

    def test_max_retries_constant_is_3(self) -> None:
        """MAX_RETRIES module constant must be exactly 3."""
        assert MAX_RETRIES == 3

    def test_recommendation_has_impact_tier_and_range(self) -> None:
        """Each recommendation must have impact_tier and impact_range_pct (new shape)."""
        explanation = _make_explanation(n_recs=3)
        llm = _make_mock_llm([explanation])
        explainer = Explainer(llm)

        result, _ = explainer.explain(_make_resume(), _make_score(), _make_salary())

        for rec in result.recommendations:
            assert rec.impact_tier in ("medium", "high", "transformative")
            assert isinstance(rec.impact_range_pct, ImpactRange)
            assert rec.impact_range_pct.low_pct <= rec.impact_range_pct.high_pct

    def test_recommendation_has_no_estimated_salary_impact_pct(self) -> None:
        """Recommendation must NOT have the old estimated_salary_impact_pct field."""
        rec = _make_recommendation()
        assert not hasattr(rec, "estimated_salary_impact_pct"), (
            "estimated_salary_impact_pct was removed — use impact_tier + impact_range_pct"
        )


class TestExplainerRetry:
    def test_llm_error_triggers_retry_and_error_in_meta(self) -> None:
        """All 3 LLM calls raise → returns fallback with error in meta."""
        llm = MagicMock()
        llm.extract_structured.side_effect = RuntimeError("simulated outage")
        explainer = Explainer(llm)

        result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert isinstance(result, Explanation)
        assert meta.get("error") == "llm_failed"
        assert meta.get("retries") == MAX_RETRIES
        assert llm.extract_structured.call_count == MAX_RETRIES

    def test_no_fourth_attempt_after_three_failures(self) -> None:
        """Verifies hardcap: exactly MAX_RETRIES=3 attempts, never 4."""
        llm = MagicMock()
        llm.extract_structured.side_effect = RuntimeError("down")
        explainer = Explainer(llm)

        explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert llm.extract_structured.call_count == MAX_RETRIES, (
            f"Expected exactly {MAX_RETRIES} calls, got {llm.extract_structured.call_count}"
        )

    def test_succeeds_on_second_attempt_after_error(self) -> None:
        """First attempt raises, second returns valid explanation."""
        good = _make_explanation(n_recs=3)
        llm = MagicMock()
        call_count = 0

        def side_effect(*args: object, **kwargs: object) -> tuple[Explanation, dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            return good, {
                "cost_usd": 0.01,
                "input_tokens": 500,
                "output_tokens": 300,
                "duration_s": 1.5,
                "model": "gpt-4o-mini",
            }

        llm.extract_structured.side_effect = side_effect
        explainer = Explainer(llm)

        result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert len(result.recommendations) == 3
        assert meta.get("error") is None, "Should succeed — no error expected"
        assert llm.extract_structured.call_count == 2


class TestExplainerNoBelowThresholdRetry:
    """Verify the ≥30% sum constraint has been removed.

    Previously, a low-impact result triggered a retry with a feedback message.
    Now, any result with exactly 3 recommendations is accepted on first attempt.
    """

    def test_low_range_recs_accepted_on_first_attempt(self) -> None:
        """Recommendations with small impact_range_pct are accepted without retry."""
        # Under the old contract, impact_tier=medium/low_pct=5 each would have been
        # considered "below threshold" and triggered a retry.  Now it must pass.
        low_range_explanation = Explanation(
            summary="Test summary",
            strengths=["Strength 1"],
            gaps=["Gap 1"],
            recommendations=[
                _make_recommendation(impact_tier="medium", low_pct=3.0, high_pct=8.0)
                for _ in range(3)
            ],
        )
        llm = _make_mock_llm([low_range_explanation])
        explainer = Explainer(llm)

        _result, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        # Must succeed on first attempt — no retry for low impact
        assert llm.extract_structured.call_count == 1
        assert meta["retries"] == 0
        assert "warning" not in meta
        assert "error" not in meta

    def test_no_feedback_message_injected_on_retry(self) -> None:
        """On retry (after error), prompt must NOT contain the old ≥30% feedback message."""
        good = _make_explanation(n_recs=3)
        llm = MagicMock()
        call_count = 0

        def side_effect(*args: object, **kwargs: object) -> tuple[Explanation, dict]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")
            return good, {
                "cost_usd": 0.01,
                "input_tokens": 500,
                "output_tokens": 300,
                "duration_s": 1.5,
                "model": "gpt-4o-mini",
            }

        llm.extract_structured.side_effect = side_effect
        explainer = Explainer(llm)
        explainer.explain(_make_resume(), _make_score(), _make_salary())

        # Check the second call's prompt does NOT contain the old constraint message
        second_call_kwargs = llm.extract_structured.call_args_list[1]
        prompt_arg = second_call_kwargs.args[0] if second_call_kwargs.args else ""
        assert "PREVIOUS ATTEMPT FAILED" not in prompt_arg
        assert "30%" not in prompt_arg
        assert "estimated_salary_impact_pct" not in prompt_arg


class TestExplainerErrorAndWarningExclusivity:
    """Verify that error='llm_failed' is set only when all LLM calls failed."""

    def test_all_calls_failing_sets_error_only(self) -> None:
        """When every LLM attempt raises, meta must carry error but NOT warning."""
        llm = MagicMock()
        llm.extract_structured.side_effect = RuntimeError("simulated outage")
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert meta.get("error") == "llm_failed"
        assert "warning" not in meta, (
            f"Expected no 'warning' key when LLM totally failed, got meta={meta}"
        )

    def test_valid_response_has_no_error_key(self) -> None:
        """When LLM returns valid 3-rec explanation, meta must NOT have error key."""
        good = _make_explanation(n_recs=3)
        llm = _make_mock_llm([good])
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert "error" not in meta, (
            f"Expected no 'error' key for successful result, got meta={meta}"
        )


class TestExplainerCostAccumulation:
    """Fix 5: retry attempt costs must be ACCUMULATED, not overwritten."""

    def test_cost_accumulated_across_failed_retries(self) -> None:
        """Each failed LLM call contributes to the total cost in meta."""
        from src.llm_provider import LLMProviderError

        # Each call raises with a cost of 0.004
        per_call_cost = 0.004
        llm = MagicMock()
        llm.extract_structured.side_effect = LLMProviderError(
            "refused",
            cost_usd=per_call_cost,
            meta={"input_tokens": 1000, "output_tokens": 50, "model": "gpt-4o-mini"},
        )
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        expected_total = per_call_cost * MAX_RETRIES
        assert meta["cost_usd"] == pytest.approx(expected_total, abs=1e-6), (
            f"Total cost should be {expected_total:.6f} "
            f"({MAX_RETRIES} × {per_call_cost}), got {meta.get('cost_usd')}"
        )

    def test_cost_accumulated_across_successful_retries(self) -> None:
        """Cost from failed attempts + successful attempt is summed in meta."""
        from src.llm_provider import LLMProviderError

        per_failed_cost = 0.003
        success_cost = 0.01
        good = _make_explanation(n_recs=3)

        llm = MagicMock()
        call_count = 0

        def side_effect(*args: object, **kwargs: object) -> tuple[Explanation, dict]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise LLMProviderError(
                    "refused",
                    cost_usd=per_failed_cost,
                    meta={"input_tokens": 800, "output_tokens": 30, "model": "gpt-4o-mini"},
                )
            return good, {
                "cost_usd": success_cost,
                "input_tokens": 500,
                "output_tokens": 300,
                "duration_s": 1.5,
                "model": "gpt-4o-mini",
            }

        llm.extract_structured.side_effect = side_effect
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        expected_total = per_failed_cost * 2 + success_cost
        assert meta["cost_usd"] == pytest.approx(expected_total, abs=1e-6), (
            f"Accumulated cost should be {expected_total:.6f}, got {meta.get('cost_usd')}"
        )

    def test_partial_cost_preserved_on_provider_error(self) -> None:
        """When LLM raises LLMProviderError, partial cost must appear in meta."""
        from src.llm_provider import LLMProviderError

        llm = MagicMock()
        llm.extract_structured.side_effect = LLMProviderError(
            "refused",
            cost_usd=0.0042,
            meta={"input_tokens": 1500, "output_tokens": 50, "model": "gpt-4o-mini"},
        )
        explainer = Explainer(llm)

        _explanation, meta = explainer.explain(_make_resume(), _make_score(), _make_salary())

        assert meta.get("error") == "llm_failed"
        # 3 retries × 0.0042
        assert meta.get("cost_usd") == pytest.approx(0.0042 * MAX_RETRIES), (
            f"Accumulated cost from retries must propagate, got meta={meta}"
        )


class TestExplainerPromptIsolation:
    """Fix 1: explainer must use system+user message separation, not concatenation."""

    def test_extract_structured_called_with_system_prompt(self) -> None:
        """extract_structured must be called with system_prompt kwarg (not concatenated)."""
        good = _make_explanation(n_recs=3)
        llm = MagicMock()
        llm.extract_structured.return_value = (good, {"cost_usd": 0.01, "model": "gpt-4o-mini"})
        explainer = Explainer(llm)

        explainer.explain(_make_resume(), _make_score(), _make_salary())

        call_kwargs = llm.extract_structured.call_args
        assert call_kwargs.kwargs.get("system_prompt") is not None, (
            "extract_structured must be called with system_prompt kwarg for prompt isolation"
        )

    def test_user_message_contains_candidate_profile_tags(self) -> None:
        """User message (prompt arg) must be wrapped in <CANDIDATE_PROFILE> tags."""
        good = _make_explanation(n_recs=3)
        llm = MagicMock()
        llm.extract_structured.return_value = (good, {"cost_usd": 0.01, "model": "gpt-4o-mini"})
        explainer = Explainer(llm)

        explainer.explain(_make_resume(), _make_score(), _make_salary())

        call_args = llm.extract_structured.call_args
        user_prompt: str = call_args.args[0]
        assert "<CANDIDATE_PROFILE>" in user_prompt
        assert "</CANDIDATE_PROFILE>" in user_prompt


class TestFallbackRecommendationShape:
    """Fallback recommendations must match the new Recommendation shape."""

    def test_fallback_has_impact_tier(self) -> None:
        from src.explainer import _fallback_recommendation

        rec = _fallback_recommendation(0)
        assert hasattr(rec, "impact_tier")
        assert rec.impact_tier in ("medium", "high", "transformative")

    def test_fallback_has_impact_range_pct(self) -> None:
        from src.explainer import _fallback_recommendation

        rec = _fallback_recommendation(0)
        assert hasattr(rec, "impact_range_pct")
        assert isinstance(rec.impact_range_pct, ImpactRange)
        assert rec.impact_range_pct.low_pct <= rec.impact_range_pct.high_pct

    def test_fallback_has_no_estimated_salary_impact_pct(self) -> None:
        from src.explainer import _fallback_recommendation

        rec = _fallback_recommendation(0)
        assert not hasattr(rec, "estimated_salary_impact_pct")
