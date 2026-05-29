"""Rule-based seniority scorer.

Produces SeniorityScore (0-100) from Resume via 5 weighted components.
Anti-skill-inflation: count up to 12 distinct skills on a curve + penalty for
>30 buzzwords with no depth signal and no years_used corroboration.
Contractor exception: no job-hopping penalty for profiles with any is_contractor=True.
Progression: infers advancement also from total experience + role breadth, not
solely from explicit "Senior/Lead" title words.
Domain: 2 domains is NOT a hard penalty — cross-domain expertise is often a premium.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.schemas import Experience, Resume, ScoreComponent, SeniorityScore

# TODO(review): All heuristic constants below were set by design reasoning,
# not statistical calibration against a labeled CV dataset. Once ground-truth
# seniority labels (N ≥ 200 CVs) are available, re-fit thresholds empirically
# (e.g. isotonic regression on component scores vs. salary bands).


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

# Czech display labels for each component — shown verbatim to end users.
_LABEL_EXPERIENCE = "Pracovní zkušenosti"
_LABEL_SKILLS = "Dovednosti"
_LABEL_PROGRESSION = "Postup v kariéře"
_LABEL_EDUCATION = "Vzdělání"
_LABEL_DOMAIN = "Zaměření oboru"

# Czech degree labels for education reasoning
_DEGREE_CZ: dict[str, str] = {
    "phd": "doktorát (PhD)",
    "master": "magisterský titul (Mgr./Ing.)",
    "bachelor": "bakalářský titul (Bc.)",
    "high_school": "středoškolské s maturitou (příp. VOŠ / DiS.)",
    "other": "výuční list nebo jiné vzdělání bez maturity",
    "none": "neuvedeno",
}

# Education degree scores
_DEGREE_SCORES = {
    "phd": 100,
    "master": 75,
    "bachelor": 50,
    "high_school": 40,  # maturita / SŠ / VOŠ — a real qualification, not "other"
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
                name=_LABEL_EXPERIENCE,
                score=0.0,
                weight=_WEIGHT_EXPERIENCE,
                reasoning="V CV nejsou žádné pracovní zkušenosti.",
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

        # Build Czech reasoning
        years_text = f"{total_years} {'rok' if total_years == 1 else 'roky' if 2 <= total_years <= 4 else 'let'} praxe"
        if has_senior_role:
            reasoning = f"{years_text}, z toho část na seniorní nebo vedoucí roli."
        else:
            reasoning = f"{years_text} celkem."

        return ScoreComponent(
            name=_LABEL_EXPERIENCE,
            score=final_score,
            weight=_WEIGHT_EXPERIENCE,
            reasoning=reasoning,
        )


class SkillComponent:
    """Score based on skills breadth and depth. Weight: 0.25.

    Redesign rationale vs. v1:
    - Old: cap 5/category × few LLM categories → collapses to base 20, depth tag never fires.
    - New: count up to 12 DISTINCT skills across all categories on a curve (score reaches
      base 70 at 12 skills). "Depth" is corroborated: explicit advanced/expert tag OR
      years_used >= 3 for a skill both count. This fires more reliably on real CVs.
    - Anti-inflation guard: >30 skills with no depth OR years corroboration applies 0.7×.

    TODO(review): The curve (12 skills → 70) and depth-bonus constants need calibration
    against a labeled dataset to verify discrimination between junior/senior populations.
    """

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.skills:
            return ScoreComponent(
                name=_LABEL_SKILLS,
                score=0.0,
                weight=_WEIGHT_SKILLS,
                reasoning="V CV nejsou uvedeny žádné dovednosti.",
            )

        total_skills = len(resume.skills)

        # Count effective (deduplicated) skills — cap at 12 to prevent buzzword inflation.
        # Using distinct names (case-insensitive) across all categories.
        distinct_names = {s.name.lower() for s in resume.skills}
        effective_count = min(12, len(distinct_names))

        # Curve: linear up to 70 at 12 skills (≈5.83 per skill)
        base_score = min(70.0, effective_count * (70.0 / 12.0))

        # Depth: skill counts as "deep" if it has explicit advanced/expert tag
        # OR years_used >= 3 (corroboration from experience duration).
        deep_skills = sum(
            1
            for s in resume.skills
            if s.depth in ("advanced", "expert") or (s.years_used is not None and s.years_used >= 3)
        )
        # Modest bonus — max 20 points at 5+ deep skills
        depth_bonus = min(20.0, deep_skills * 4.0)

        total = base_score + depth_bonus
        inflation_applied = False

        # Inflation guard: >30 skills but fewer than 3 have any depth signal
        if total_skills > 30 and deep_skills < 3:
            total *= 0.7
            inflation_applied = True

        final_score = min(100.0, total)

        # Czech reasoning — no math, just facts
        skill_word = (
            "dovednost"
            if effective_count == 1
            else "dovednosti"
            if 2 <= effective_count <= 4
            else "dovedností"
        )
        if effective_count == total_skills:
            breadth_text = f"{effective_count} relevantních {skill_word}"
        else:
            breadth_text = f"{effective_count} relevantních {skill_word} (z {total_skills} celkem)"

        if deep_skills > 0:
            depth_text = (
                f"; u {deep_skills} z nich je doložena pokročilá úroveň nebo dlouholetá praxe"
            )
        else:
            depth_text = "; v CV chybí signál o pokročilé úrovni"

        reasoning = breadth_text + depth_text + "."
        if inflation_applied:
            reasoning += " (Pozor: příliš mnoho dovedností bez doložené hloubky — možná inflace.)"

        return ScoreComponent(
            name=_LABEL_SKILLS,
            score=final_score,
            weight=_WEIGHT_SKILLS,
            reasoning=reasoning,
        )


class ProgressionComponent:
    """Score based on career progression. Weight: 0.15.

    Redesign rationale vs. v1:
    - Old: needed explicit "Senior/Lead" word in seniority_level → many real seniors get 0.
    - New: infer progression also from total years + number of distinct roles over time.
      "Experienced breadth" path: 8+ years across 3+ distinct employers also signals
      seniority advancement even without an explicit seniority_level tag.
    - Contractor exception and job-hopping penalty preserved.

    TODO(review): The breadth/years thresholds (8 years, 3 roles) need validation
    against real data — they are currently set by heuristic reasoning.
    """

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.experiences:
            return ScoreComponent(
                name=_LABEL_PROGRESSION,
                score=0.0,
                weight=_WEIGHT_PROGRESSION,
                reasoning="V CV nejsou žádné pracovní zkušenosti.",
            )

        # Sort by start_year ascending
        sorted_exps = sorted(resume.experiences, key=lambda e: e.start_year)

        # --- Path A: explicit seniority progression ---
        progressions = 0
        prev_level = -1
        for exp in sorted_exps:
            if exp.seniority_level is not None:
                current_level = _SENIORITY_ORDER.get(exp.seniority_level, -1)
                if current_level > prev_level and prev_level >= 0:
                    progressions += 1
                # max() keeps the bar at the highest level reached so far —
                # a temporary downward step (e.g. senior → contractor stint
                # described as "medior") doesn't reset and let a later return
                # to senior count as a fresh progression.
                prev_level = max(prev_level, current_level)

        explicit_score = min(100.0, progressions * 35.0)

        # --- Path B: inferred progression from breadth + tenure ---
        # A candidate with 8+ de-duplicated years across 3+ distinct employers
        # very likely progressed in responsibility even if titles don't say so.
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

        distinct_employers = len({exp.company for exp in resume.experiences})
        has_management = any(exp.is_management for exp in resume.experiences)

        inferred_score = 0.0
        if total_years >= 8 and distinct_employers >= 3:
            # Long, broad career without explicit progression labels — credit partial score
            inferred_score = 40.0
        elif total_years >= 5 and distinct_employers >= 2:
            inferred_score = 20.0

        if has_management:
            inferred_score = min(100.0, inferred_score + 20.0)

        # Take the better of explicit vs. inferred
        score = max(explicit_score, inferred_score)

        # --- Build Czech reasoning ---
        if progressions > 0:
            progression_text = (
                f"{'Jeden posun' if progressions == 1 else f'{progressions} posuny'} "
                f"na vyšší senioritu během kariéry."
            )
        elif inferred_score > 0:
            progression_text = (
                f"Bez explicitního titulového postupu, ale {total_years} let praxe "
                f"u {distinct_employers} zaměstnavatelů signalizuje rostoucí odpovědnost."
            )
        else:
            progression_text = "Bez jasného postupu na vyšší roli."

        if has_management and progressions == 0 and inferred_score <= 20.0:
            progression_text += " Role zahrnovala management."

        reasoning = progression_text

        # Contractor exception check
        is_contractor_profile = any(exp.is_contractor for exp in resume.experiences)

        if is_contractor_profile:
            reasoning += (
                " Contractor profil — penalizace za časté střídání zaměstnání se neuplatňuje."
            )
        else:
            # Job-hopping penalty: 5+ roles in the last 3 years
            if len(resume.experiences) >= 1:
                max_year = max(e.start_year for e in resume.experiences)
                recent = [e for e in resume.experiences if e.start_year >= max_year - 3]
                if len(recent) >= 5:
                    score = max(0.0, score - 20.0)
                    reasoning += f" (Penalizace: {len(recent)} rolí za 3 roky — časté střídání.)"

        return ScoreComponent(
            name=_LABEL_PROGRESSION,
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
                name=_LABEL_EDUCATION,
                score=0.0,
                weight=_WEIGHT_EDUCATION,
                reasoning="V CV není uvedeno vzdělání.",
            )

        # Take the highest degree
        best_edu = max(
            resume.educations,
            key=lambda e: _DEGREE_SCORES.get(e.degree, 0),
        )
        base_score = float(_DEGREE_SCORES.get(best_edu.degree, 0))

        # Ensure vocational/other degrees have a meaningful floor —
        # for blue-collar and trades roles a degree is genuinely not required,
        # so the component should not drag the total down unfairly.
        if best_edu.degree in ("other", "none"):
            base_score = max(base_score, 20.0)

        # Determine primary occupation_family from the candidate's current role.
        # Ongoing roles (end_year=None) outrank finished ones, then we prefer
        # the *longer-running* role on tie-break — a 6-year ongoing main job
        # should beat a 1-year-old side gig that started later. Negating
        # start_year inside the tuple flips the tie-break direction without
        # breaking the descending sort applied to the outer fields.
        primary_family: str = "other"

        def _recency_key(exp: Experience) -> tuple[int, int, int]:
            is_current = 1 if exp.end_year is None else 0
            return (is_current, exp.end_year or 0, -(exp.start_year or 0))

        for exp in sorted(resume.experiences, key=_recency_key, reverse=True):
            if exp.occupation_family is not None:
                primary_family = exp.occupation_family
                break

        relevant_keywords = _RELEVANT_FIELDS_BY_FAMILY.get(primary_family, frozenset())
        field_lower = (best_edu.field or "").lower()
        is_relevant = bool(relevant_keywords) and any(kw in field_lower for kw in relevant_keywords)
        bonus = 10.0 if is_relevant else 0.0
        final_score = min(100.0, base_score + bonus)

        degree_label = _DEGREE_CZ.get(best_edu.degree, best_edu.degree)
        reasoning = f"Nejvyšší vzdělání: {degree_label}."
        if is_relevant and best_edu.field:
            reasoning += f" Obor ({best_edu.field}) odpovídá zaměření kariéry — bonus za relevanci."

        return ScoreComponent(
            name=_LABEL_EDUCATION,
            score=final_score,
            weight=_WEIGHT_EDUCATION,
            reasoning=reasoning,
        )


class DomainExpertiseComponent:
    """Score based on domain focus. Weight: 0.10.

    Redesign rationale vs. v1:
    - Old: "fewer domains = higher score" with steep 15-point drop per domain.
           This unfairly penalised valuable cross-domain profiles (health-tech, fintech).
    - New: 1 domain = strong focus (100); 2 domains = moderate cross-domain (75, not 70);
           Each additional domain beyond 2 has a smaller 10-point drop.
           The reasoning no longer implies cross-domain is negative.

    TODO(review): The per-domain penalty (10 pts beyond 2 domains) and the
    cross-domain floor were set heuristically. Calibrate against labeled data.
    """

    @staticmethod
    def compute(resume: Resume) -> ScoreComponent:
        if not resume.experiences:
            return ScoreComponent(
                name=_LABEL_DOMAIN,
                score=50.0,  # Neutral if no experience
                weight=_WEIGHT_DOMAIN,
                reasoning="Žádné pracovní zkušenosti — hodnocení oboru nelze provést.",
            )

        # Use occupation_family (ISCO-based) as domain diversity measure
        families = {exp.occupation_family for exp in resume.experiences if exp.occupation_family}
        num_domains = max(1, len(families))

        if num_domains == 1:
            score = 100.0
        elif num_domains == 2:
            # Cross-domain is often a premium — don't penalise heavily
            score = 75.0
        else:
            # Beyond 2 domains: modest additional penalty
            score = max(0.0, 75.0 - (num_domains - 2) * 10.0)

        if num_domains == 1:
            reasoning = "Soustředěné zaměření na jeden obor."
        elif num_domains == 2:
            reasoning = "Zkušenost ve dvou oborech — mezioborový profil může být výhodou."
        else:
            reasoning = f"Zkušenost napříč {num_domains} obory — velmi různorodý profil."

        return ScoreComponent(
            name=_LABEL_DOMAIN,
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
