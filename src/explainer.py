"""LLM-based career advisor / explainer.

Produces Explanation with exactly 3 Recommendations (sum impact >= 30%).
MAX_RETRIES = 3 — hardcap, no infinite loop.
Prompt lives in prompts/explainer_system.txt — no inline prompts here.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from src.llm_provider import LLMProvider
from src.paths import EXPLAINER_PROMPT_PATH as _PROMPT_PATH
from src.schemas import Explanation, Resume, SalaryEstimate, SeniorityScore

logger = logging.getLogger(__name__)

MAX_RETRIES = 3  # hardcap — no infinite loop
_MIN_TOTAL_IMPACT_PCT = 30.0


def _total_experience_years(resume: Resume) -> int:
    """Total years of experience with overlapping intervals merged.

    Parallel roles (e.g. main HPP + side freelance covering the same period)
    must not double-count. "Ongoing" roles (end_year is None) are evaluated
    against the current calendar year so the count stays correct over time.
    """
    current_year = datetime.now(UTC).year
    intervals = sorted(
        (exp.start_year, exp.end_year or current_year)
        for exp in resume.experiences
        if (exp.end_year or current_year) > exp.start_year
    )
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


class Explainer:
    """Generates career analysis with 3 recommendations using LLM.

    Retries up to MAX_RETRIES if:
    - LLM returns fewer than 3 recommendations
    - Total estimated_salary_impact_pct is below 30%

    After MAX_RETRIES, returns last result with warning in meta.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self._system_prompt: str | None = None

    def _load_prompt(self) -> str:
        """Load explainer system prompt from file, caching after first read."""
        if self._system_prompt is None:
            if not _PROMPT_PATH.exists():
                raise FileNotFoundError(
                    f"Explainer prompt not found at {_PROMPT_PATH}. "
                    "Ensure prompts/explainer_system.txt exists in the project root."
                )
            self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._system_prompt

    def _build_prompt(self, resume: Resume, score: SeniorityScore, salary: SalaryEstimate) -> str:
        """Build the full prompt including candidate data."""
        system = self._load_prompt()
        candidate_summary = (
            f"Name: {resume.full_name or 'Unknown'}\n"
            f"Location: {resume.location or 'Not specified'}\n"
            f"Total experience: {_total_experience_years(resume)} years\n"
            f"Skills: {', '.join(s.name for s in resume.skills[:20])}\n"
            f"Most recent role: {resume.experiences[0].role_title if resume.experiences else 'N/A'}\n"
            f"Seniority score: {score.total:.1f}/100 (confidence: {score.confidence:.0%})\n"
            f"Salary estimate: {salary.low:,}–{salary.high:,} CZK/month\n"
            f"Salary assumptions: {', '.join(salary.assumptions[:3])}\n"
        )
        return f"{system}\n\n{candidate_summary}"

    def explain(
        self,
        resume: Resume,
        score: SeniorityScore,
        salary: SalaryEstimate,
    ) -> tuple[Explanation, dict[str, Any]]:
        """Generate Explanation with exactly 3 recommendations summing to >= 30% impact.

        Retries up to MAX_RETRIES=3 times if quality check fails.
        After exhausting retries, returns last result with warning in meta.

        Args:
            resume: Parsed candidate Resume.
            score: SeniorityScore from rule-based scorer.
            salary: SalaryEstimate from estimator.

        Returns:
            (Explanation, meta) where meta includes retries count and optional warning.
        """
        base_prompt = self._build_prompt(resume, score, salary)
        last_explanation: Explanation | None = None
        last_meta: dict[str, Any] = {}

        for attempt in range(MAX_RETRIES):
            prompt = base_prompt
            if attempt > 0:
                prompt += (
                    f"\n\nPREVIOUS ATTEMPT FAILED: total salary impact was below {_MIN_TOTAL_IMPACT_PCT}%. "
                    f"You MUST provide 3 recommendations whose estimated_salary_impact_pct "
                    f"sum to AT LEAST {_MIN_TOTAL_IMPACT_PCT}. Focus on high-impact areas: "
                    f"specialization, certifications, role transitions."
                )

            try:
                raw_explanation, meta = self.llm.extract_structured(
                    prompt, Explanation, max_tokens=4096
                )
                explanation = cast("Explanation", raw_explanation)
            except Exception:
                logger.exception("LLM call failed on explainer attempt %d", attempt + 1)
                last_meta = {"cost_usd": 0.0, "error": "llm_failed", "retries": attempt + 1}
                continue

            last_explanation = explanation
            last_meta = {**meta, "retries": attempt}

            total_impact = sum(r.estimated_salary_impact_pct for r in explanation.recommendations)

            if total_impact >= _MIN_TOTAL_IMPACT_PCT and len(explanation.recommendations) == 3:
                logger.info(
                    "Explainer succeeded on attempt %d — total impact: %.1f%%",
                    attempt + 1,
                    total_impact,
                )
                return explanation, last_meta

            logger.warning(
                "Explainer attempt %d/%d failed quality check: "
                "%d recs, total impact %.1f%% (need >= %.1f%%)",
                attempt + 1,
                MAX_RETRIES,
                len(explanation.recommendations),
                total_impact,
                _MIN_TOTAL_IMPACT_PCT,
            )

        # Exhausted all retries — return last result with warning
        logger.warning(
            "Explainer failed to produce %.1f%%+ recommendations after %d retries",
            _MIN_TOTAL_IMPACT_PCT,
            MAX_RETRIES,
        )

        # Distinguish two failure modes:
        #   (a) every LLM call raised → last_explanation is None → "llm_failed"
        #   (b) calls succeeded but never met the quality bar → "below_30pct_target"
        # The two are mutually exclusive; stacking both used to confuse metrics.
        all_calls_failed = last_explanation is None

        if all_calls_failed:
            from pydantic import ValidationError

            try:
                last_explanation = Explanation(
                    summary="Analysis unavailable — LLM call failed.",
                    strengths=["Unable to generate analysis"],
                    gaps=["Unable to generate analysis"],
                    recommendations=[_fallback_recommendation(i) for i in range(3)],
                )
            except ValidationError as exc:
                raise RuntimeError("Explainer: LLM failed and fallback creation failed") from exc

        last_meta["retries"] = MAX_RETRIES
        if not all_calls_failed:
            # Quality threshold never met — surface the dedicated warning.
            last_meta["warning"] = "below_30pct_target"
        # If all_calls_failed, last_meta already carries error="llm_failed" from
        # the last attempt; do not also set warning="below_30pct_target".
        return last_explanation, last_meta


def _fallback_recommendation(index: int) -> Any:
    """Minimal fallback recommendation when all LLM calls fail."""
    from src.schemas import Recommendation

    titles = [
        "Consult a career advisor",
        "Review your CV structure",
        "Expand your skill portfolio",
    ]
    return Recommendation(
        title=titles[index],
        why_it_matters="LLM analysis unavailable — this is a placeholder recommendation.",
        estimated_salary_impact_pct=10.0,
        timeframe_months=6,
        first_action="Contact your career advisor or retry the analysis.",
    )
