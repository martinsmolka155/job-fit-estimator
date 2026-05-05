"""Rule-based seniority scorer.

Produces SeniorityScore (0-100) from Resume via 5 weighted components.
Anti-skill-inflation: cap 5 skills per category + inflation penalty (30+ skills, <5 with depth).
Contractor exception: no job-hopping penalty for profiles with any is_contractor=True experience.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from src.schemas import Experience, Resume, ScoreComponent, SeniorityScore


def _current_year() -> int:
    """Return the current calendar year (UTC).

    Computed per-call rather than captured at import time so a long-running
    process doesn't freeze the "ongoing" reference point at startup.
    """
    return datetime.now(UTC).year


# Component weights — MUST sum to 1.0
_WEIGHT_EXPERIENCE = 0.40
_WEIGHT_SKILLS = 0.25
_WEIGHT_PROGRESSION = 0.15
_WEIGHT_EDUCATION = 0.10
_WEIGHT_DOMAIN = 0.10

# Education degree scores
_DEGREE_SCORES = {
    "phd": 100,
    "master": 75,
    "bachelor": 50,
    "other": 25,
    "none": 0,
}

# Education field keywords per occupation_family.
# Bonus applied only when education field matches the candidate's occupation_family.
_RELEVANT_FIELDS_BY_FAMILY: dict[str, frozenset[str]] = {
    "it": frozenset(
        {
            "computer science",
            "software",
            "informatics",
            "information systems",
            "data science",
            "machine learning",
            "artificial intelligence",
            "cybersecurity",
            "networking",
            "informatika",
            "kybernetika",
        }
    ),
    "healthcare": frozenset(
        {
            "medicine",
            "nursing",
            "pharmacy",
            "physiotherapy",
            "dentistry",
            "public health",
            "medicína",
            "ošetřovatelství",
            "farmacie",
        }
    ),
    "legal": frozenset(
        {
            "law",
            "legal",
            "jurisprudence",
            "právo",
            "právní",
            "finance",
            "accounting",
            "economics",
            "business administration",
            "účetnictví",
            "ekonomie",
        }
    ),
    "education": frozenset(
        {
            "pedagogy",
            "education",
            "teaching",
            "pedagogika",
            "učitelství",
            "didactics",
            "special education",
        }
    ),
    "trades": frozenset(
        {
            "engineering",
            "mechanical",
            "civil",
            "electrical",
            "construction",
            "strojírenství",
            "stavebnictví",
            "elektrotechnika",
        }
    ),
    "services": frozenset(
        {
            "hospitality",
            "tourism",
            "marketing",
            "gastronomy",
            "catering",
            "hotelnictví",
            "turismus",
            "gastronomie",
        }
    ),
    "other": frozenset(),  # no bonus for catch-all
}

# Seniority level ordering for progression scoring
_SENIORITY_ORDER = {"junior": 0, "medior": 1, "senior": 2, "lead": 3, "principal": 4}


class ExperienceComponent:
    """Score based on total years of experience. Weight: 0.40."""

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.experiences:
            return ScoreComponent(
                name="ExperienceComponent",
                score=0.0,
                weight=_WEIGHT_EXPERIENCE,
                reasoning="No experience entries found",
            )

        # Merge overlapping intervals before summing — parallel roles
        # (e.g. main HPP + side freelance covering the same period) must not
        # double-count toward total years of experience.
        current = _current_year()
        intervals: list[tuple[int, int]] = sorted(
            (exp.start_year, exp.end_year or current)
            for exp in resume.experiences
            if (exp.end_year or current) > exp.start_year
        )
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_years = max(0, sum(end - start for start, end in merged))

        base_score = min(100.0, total_years * 8.0)  # 12.5 years → 100

        has_senior_role = any(
            exp.seniority_level in ("senior", "lead", "principal") for exp in resume.experiences
        )
        bonus = 10.0 if has_senior_role else 0.0
        final_score = min(100.0, base_score + bonus)

        reasoning = f"{total_years} total years of experience → base score {base_score:.0f}" + (
            " | +10 senior/lead role bonus" if has_senior_role else ""
        )

        return ScoreComponent(
            name="ExperienceComponent",
            score=final_score,
            weight=_WEIGHT_EXPERIENCE,
            reasoning=reasoning,
        )


class SkillComponent:
    """Score based on skills breadth and depth. Weight: 0.25.

    Anti-inflation: cap 5 skills per category, penalize resumes with 30+ skills but <5 with depth.
    """

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.skills:
            return ScoreComponent(
                name="SkillComponent",
                score=0.0,
                weight=_WEIGHT_SKILLS,
                reasoning="No skills found",
            )

        # Cap 5 per category to prevent inflation
        per_category = Counter(s.category for s in resume.skills)
        capped_count = sum(min(5, n) for n in per_category.values())

        base_score = min(70.0, capped_count * 4.0)  # base capped at 70

        # Quality bonus from deep skills
        deep_skills = sum(1 for s in resume.skills if s.depth in ("advanced", "expert"))
        depth_bonus = min(20.0, deep_skills * 3.0)

        total = base_score + depth_bonus
        reasoning_suffix = ""

        # Inflation penalty: >25 skills but <5 with depth
        if len(resume.skills) > 25 and deep_skills < 5:
            total *= 0.7
            reasoning_suffix = " | INFLATION PENALTY (30+ skills, <5 with depth)"

        final_score = min(100.0, total)

        reasoning = (
            f"Capped skill count: {capped_count} (from {len(resume.skills)} raw) → "
            f"base {base_score:.0f} + depth bonus {depth_bonus:.0f}" + reasoning_suffix
        )

        return ScoreComponent(
            name="SkillComponent",
            score=final_score,
            weight=_WEIGHT_SKILLS,
            reasoning=reasoning,
        )


class ProgressionComponent:
    """Score based on career progression. Weight: 0.15.

    Contractor exception: no job-hopping penalty if any experience has is_contractor=True.
    """

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.experiences:
            return ScoreComponent(
                name="ProgressionComponent",
                score=0.0,
                weight=_WEIGHT_PROGRESSION,
                reasoning="No experience entries found",
            )

        # Sort by start_year ascending
        sorted_exps = sorted(resume.experiences, key=lambda e: e.start_year)

        # Count upward progressions in seniority level
        progressions = 0
        prev_level = -1
        for exp in sorted_exps:
            if exp.seniority_level is not None:
                current_level = _SENIORITY_ORDER.get(exp.seniority_level, -1)
                if current_level > prev_level and prev_level >= 0:
                    progressions += 1
                prev_level = max(prev_level, current_level)

        score = min(100.0, progressions * 30.0)
        reasoning = f"{progressions} upward seniority progressions → score {score:.0f}"

        # Contractor exception check
        is_contractor_profile = any(exp.is_contractor for exp in resume.experiences)

        if is_contractor_profile:
            reasoning += " | Contractor profile — no hopping penalty applied"
        else:
            # Job-hopping penalty: 5+ roles in the last 3 years
            if len(resume.experiences) >= 1:
                max_year = max(e.start_year for e in resume.experiences)
                recent = [e for e in resume.experiences if e.start_year >= max_year - 3]
                if len(recent) >= 5:
                    score = max(0.0, score - 20.0)
                    reasoning += f" | Penalty: {len(recent)} roles in 3 years (job-hopping)"

        return ScoreComponent(
            name="ProgressionComponent",
            score=score,
            weight=_WEIGHT_PROGRESSION,
            reasoning=reasoning,
        )


class EducationComponent:
    """Score based on highest education degree. Weight: 0.10."""

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.educations:
            return ScoreComponent(
                name="EducationComponent",
                score=0.0,
                weight=_WEIGHT_EDUCATION,
                reasoning="No education entries found",
            )

        # Take the highest degree
        best_edu = max(
            resume.educations,
            key=lambda e: _DEGREE_SCORES.get(e.degree, 0),
        )
        base_score = float(_DEGREE_SCORES.get(best_edu.degree, 0))

        # Determine primary occupation_family from the candidate's current role
        # (ongoing jobs first, then most recent end_year, then start_year).
        primary_family: str = "other"

        def _recency_key(exp: Experience) -> tuple[int, int, int]:
            is_current = 1 if exp.end_year is None else 0
            return (is_current, exp.end_year or 0, exp.start_year or 0)

        for exp in sorted(resume.experiences, key=_recency_key, reverse=True):
            if exp.occupation_family is not None:
                primary_family = exp.occupation_family
                break

        relevant_keywords = _RELEVANT_FIELDS_BY_FAMILY.get(primary_family, frozenset())
        field_lower = (best_edu.field or "").lower()
        is_relevant = bool(relevant_keywords) and any(kw in field_lower for kw in relevant_keywords)
        bonus = 10.0 if is_relevant else 0.0
        final_score = min(100.0, base_score + bonus)

        reasoning = f"Highest degree: {best_edu.degree} → {base_score:.0f}" + (
            f" | +10 relevant field ({best_edu.field})" if is_relevant else ""
        )

        return ScoreComponent(
            name="EducationComponent",
            score=final_score,
            weight=_WEIGHT_EDUCATION,
            reasoning=reasoning,
        )


class DomainExpertiseComponent:
    """Score based on domain focus (fewer domains = higher focus = higher score). Weight: 0.10."""

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.experiences:
            return ScoreComponent(
                name="DomainExpertiseComponent",
                score=50.0,  # Neutral if no experience
                weight=_WEIGHT_DOMAIN,
                reasoning="No experience entries — neutral score",
            )

        # Use occupation_family (ISCO-based) as domain diversity measure
        families = {exp.occupation_family for exp in resume.experiences if exp.occupation_family}
        num_domains = max(1, len(families))

        score = max(0.0, 100.0 - (num_domains - 1) * 15.0)

        reasoning = (
            f"{num_domains} distinct occupation families → focus score {score:.0f} "
            f"(fewer domains = higher score)"
        )

        return ScoreComponent(
            name="DomainExpertiseComponent",
            score=score,
            weight=_WEIGHT_DOMAIN,
            reasoning=reasoning,
        )


def score_resume(resume: Resume) -> SeniorityScore:
    """Compute SeniorityScore from Resume using 5 weighted components.

    Component weights: Experience 0.40, Skills 0.25, Progression 0.15,
    Education 0.10, DomainExpertise 0.10. Sum = 1.0.
    """
    components = [
        ExperienceComponent.compute(resume),
        SkillComponent.compute(resume),
        ProgressionComponent.compute(resume),
        EducationComponent.compute(resume),
        DomainExpertiseComponent.compute(resume),
    ]
    total = sum(c.score * c.weight for c in components)
    return SeniorityScore(total=round(total, 2), components=components, confidence=1.0)
