"""LLM-based career advisor / explainer.

Produces Explanation with exactly 3 Recommendations.
Impact is conveyed as an honest tier + range, never as an artificial point sum.
MAX_RETRIES = 3 — hardcap, no infinite loop.
Prompt lives in prompts/explainer_system.txt — no inline prompts here.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from src.llm_provider import LLMProvider, LLMProviderError
from src.paths import EXPLAINER_PROMPT_PATH as _PROMPT_PATH
from src.schemas import Explanation, Resume, SalaryEstimate, SeniorityScore

logger = logging.getLogger(__name__)

MAX_RETRIES = 3  # hardcap — no infinite loop


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
    """Generates career analysis with exactly 3 recommendations using LLM.

    Retries up to MAX_RETRIES if the LLM call raises or returns fewer than
    3 recommendations.  There is no longer a ≥30% impact-sum requirement —
    that constraint pressured the model to fabricate inflated percentages.
    Impact is now expressed as impact_tier + impact_range_pct (low..high).

    After MAX_RETRIES, returns last result (or fallback) with error in meta.
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

    def _build_user_message(
        self, resume: Resume, score: SeniorityScore, salary: SalaryEstimate
    ) -> str:
        """Build the user-message candidate summary, isolated from the system prompt.

        The candidate summary is derived from the parsed Resume — which in turn was
        extracted from untrusted CV text.  Wrapping it in <CANDIDATE_PROFILE> tags
        makes it explicit to the model that this is structured data, not instructions.
        """
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
        return f"<CANDIDATE_PROFILE>\n{candidate_summary}</CANDIDATE_PROFILE>"

    def explain(
        self,
        resume: Resume,
        score: SeniorityScore,
        salary: SalaryEstimate,
    ) -> tuple[Explanation, dict[str, Any]]:
        """Generate Explanation with exactly 3 recommendations.

        Retries up to MAX_RETRIES=3 times on LLM failure (raises or returns
        fewer than 3 recommendations).  The ≥30% impact-sum constraint has been
        removed — impact is now expressed as honest tier + range, not inflated
        point estimates.

        Cost is accumulated across all attempts so meta.cost_usd reflects the
        total spend even when retries are needed.

        Args:
            resume: Parsed candidate Resume.
            score: SeniorityScore from rule-based scorer.
            salary: SalaryEstimate from estimator.

        Returns:
            (Explanation, meta) where meta includes retries count, accumulated
            cost_usd, and error key on total failure.
        """
        system_prompt = self._load_prompt()
        user_message = self._build_user_message(resume, score, salary)
        last_explanation: Explanation | None = None
        last_meta: dict[str, Any] = {}
        accumulated_cost: float = 0.0

        for attempt in range(MAX_RETRIES):
            try:
                raw_explanation, meta = self.llm.extract_structured(
                    user_message,
                    Explanation,
                    max_tokens=4096,
                    system_prompt=system_prompt,
                )
                explanation = cast("Explanation", raw_explanation)
            except LLMProviderError as exc:
                logger.exception("LLM call failed on explainer attempt %d", attempt + 1)
                # Accumulate partial cost — each retry burns real tokens.
                accumulated_cost += exc.cost_usd
                last_meta = {
                    **exc.meta,
                    "cost_usd": accumulated_cost,
                    "error": "llm_failed",
                    "retries": attempt + 1,
                }
                continue
            except Exception:
                logger.exception("LLM call failed on explainer attempt %d", attempt + 1)
                last_meta = {
                    "cost_usd": accumulated_cost,
                    "error": "llm_failed",
                    "retries": attempt + 1,
                }
                continue

            # Accumulate cost across attempts so total spend is always visible.
            attempt_cost: float = meta.get("cost_usd", 0.0)
            accumulated_cost += attempt_cost
            last_explanation = explanation
            last_meta = {**meta, "cost_usd": accumulated_cost, "retries": attempt}

            if len(explanation.recommendations) == 3:
                logger.info(
                    "Explainer succeeded on attempt %d",
                    attempt + 1,
                )
                return explanation, last_meta

            logger.warning(
                "Explainer attempt %d/%d: got %d recommendations (need 3)",
                attempt + 1,
                MAX_RETRIES,
                len(explanation.recommendations),
            )

        # Exhausted all retries.
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
        last_meta["cost_usd"] = accumulated_cost
        # Only set error when every call raised — not when the model returned a
        # structurally valid but incomplete result.
        if all_calls_failed and "error" not in last_meta:
            last_meta["error"] = "llm_failed"
        return last_explanation, last_meta


def _fallback_recommendation(index: int) -> Any:
    """Minimal fallback recommendation when all LLM calls fail.

    Uses the new impact_tier + impact_range_pct shape (no fabricated point %).
    """
    from src.schemas import ImpactRange, Recommendation

    titles = [
        "Consult a career advisor",
        "Review your CV structure",
        "Expand your skill portfolio",
    ]
    return Recommendation(
        title=titles[index],
        why_it_matters="LLM analysis unavailable — this is a placeholder recommendation.",
        impact_tier="medium",
        impact_range_pct=ImpactRange(low_pct=5.0, high_pct=15.0),
        timeframe_months=6,
        first_action="Contact your career advisor or retry the analysis.",
    )
