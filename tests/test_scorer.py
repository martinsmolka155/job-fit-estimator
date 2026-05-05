"""Tests for rule-based seniority scorer.

6 synthetic Resume profiles covering the full scoring range.
Contractor exception and inflation penalty are explicitly verified.
"""

from __future__ import annotations

import pytest

from src.schemas import Education, Experience, Resume, Skill
from src.scorer import score_resume


def _make_skill(name: str, category: str = "language", depth: str | None = None) -> Skill:
    return Skill(name=name, category=category, depth=depth)  # type: ignore[arg-type]


def _make_exp(
    company: str,
    start: int,
    end: int | None,
    occupation_family: str | None = "it",
    seniority: str | None = "medior",
    is_contractor: bool = False,
    is_management: bool = False,
) -> Experience:
    return Experience(
        role_title=f"{seniority or 'Developer'} at {company}",
        occupation_family=occupation_family,  # type: ignore[arg-type]
        company=company,
        start_year=start,
        end_year=end,
        seniority_level=seniority,  # type: ignore[arg-type]
        is_contractor=is_contractor,
        is_management=is_management,
    )


class TestJuniorProfile:
    """Junior: 1 year, BSc, 5 skills → expected total 25-45."""

    def test_score_in_range(self) -> None:
        resume = Resume(
            full_name="Jana Nováková",
            educations=[
                Education(degree="bachelor", field="computer science", year_completed=2024)
            ],
            experiences=[_make_exp("Startup s.r.o.", 2024, None, seniority="junior")],
            skills=[_make_skill(s) for s in ["Python", "SQL", "Git", "Docker", "FastAPI"]],
            raw_text_length=800,
        )
        score = score_resume(resume)
        assert 25 <= score.total <= 45, f"Junior score out of range: {score.total}"
        assert score.confidence == 1.0


class TestMediorProfile:
    """Medior: 4 years, MSc, 12 skills across categories → expected total 40-65."""

    def test_score_in_range(self) -> None:
        # Spread skills across multiple categories to get higher capped count
        skills = [
            _make_skill("Python", "language"),
            _make_skill("JavaScript", "language"),
            _make_skill("SQL", "language"),
            _make_skill("FastAPI", "framework"),
            _make_skill("React", "framework"),
            _make_skill("Django", "framework"),
            _make_skill("Docker", "tool"),
            _make_skill("Git", "tool"),
            _make_skill("Kubernetes", "tool"),
            _make_skill("Communication", "soft"),
            _make_skill("Agile", "soft"),
            _make_skill("Backend", "domain"),
        ]
        resume = Resume(
            full_name="Tomáš Procházka",
            educations=[Education(degree="master", field="computer science", year_completed=2021)],
            experiences=[
                _make_exp("Firma A", 2021, 2023, seniority="junior"),
                _make_exp("Firma B", 2023, None, seniority="medior"),
            ],
            skills=skills,
            raw_text_length=1200,
        )
        score = score_resume(resume)
        assert 40 <= score.total <= 65, f"Medior score out of range: {score.total}"


class TestSeniorProfile:
    """Senior: 8 years, MSc, 18 skills with AI → expected total 75-92."""

    def test_score_in_range(self) -> None:
        skills = [_make_skill(f"Skill{i}", depth="advanced") for i in range(6)]
        skills += [_make_skill("PyTorch", category="framework", depth="expert")]
        skills += [_make_skill("LLM", category="tool")]
        skills += [_make_skill(f"Other{i}") for i in range(10)]

        resume = Resume(
            full_name="Alena Kratochvílová",
            educations=[Education(degree="master", field="machine learning", year_completed=2016)],
            experiences=[
                _make_exp("BigCorp", 2016, 2019, seniority="junior"),
                _make_exp("TechCo", 2019, 2022, seniority="medior"),
                _make_exp("AI Startup", 2022, None, occupation_family="it", seniority="senior"),
            ],
            skills=skills,
            raw_text_length=2000,
        )
        score = score_resume(resume)
        assert 75 <= score.total <= 92, f"Senior score out of range: {score.total}"


class TestPrincipalProfile:
    """Principal: 15 years, PhD, 25 skills, lead roles → expected total 80-100."""

    def test_score_in_range(self) -> None:
        # Spread across all categories to maximize capped count (5 per category × 5 = 25 capped)
        categories = ["language", "framework", "tool", "soft", "domain"]
        skills = [
            _make_skill(f"Skill_{cat}_{i}", cat, depth="expert")
            for cat in categories
            for i in range(5)
        ]

        resume = Resume(
            full_name="Radek Novotný",
            educations=[Education(degree="phd", field="computer science", year_completed=2010)],
            experiences=[
                _make_exp("University", 2010, 2012, seniority="junior"),
                _make_exp("Corp A", 2012, 2016, seniority="medior"),
                _make_exp("Corp B", 2016, 2019, seniority="senior"),
                _make_exp("BigTech", 2019, None, seniority="lead", is_management=True),
            ],
            skills=skills,
            raw_text_length=3000,
        )
        score = score_resume(resume)
        assert 80 <= score.total <= 100, f"Principal score out of range: {score.total}"


class TestContractorProfile:
    """Contractor: 10 years, 8 different freelance roles → expected 55-85, NO job-hop penalty."""

    def test_score_in_range_no_penalty(self) -> None:
        # Spread skills across categories for better capped count
        categories = ["language", "framework", "tool", "soft", "domain"]
        skills = [
            _make_skill(f"Skill_{cat}_{i}", cat, depth="advanced")
            for cat in categories
            for i in range(3)
        ]

        resume = Resume(
            full_name="Monika Blahová",
            educations=[
                Education(degree="bachelor", field="software engineering", year_completed=2014)
            ],
            experiences=[
                _make_exp(f"Client{i}", 2014 + i, 2015 + i, seniority="senior", is_contractor=True)
                for i in range(8)
            ],
            skills=skills,
            raw_text_length=2500,
        )
        score = score_resume(resume)
        assert 55 <= score.total <= 85, f"Contractor score out of range: {score.total}"

        # Verify: no hopping penalty in progression reasoning
        progression = next(c for c in score.components if c.name == "ProgressionComponent")
        assert "Contractor profile — no hopping penalty applied" in progression.reasoning, (
            f"Expected contractor exception in reasoning, got: {progression.reasoning}"
        )
        assert (
            "Penalty" not in progression.reasoning or "no hopping penalty" in progression.reasoning
        )


class TestInflationCheaterProfile:
    """Inflation cheater: 2 years, 35 skills with no depth → expected < 50 (penalty applied)."""

    def test_score_below_50_with_penalty(self) -> None:
        # 35 skills, no depth — triggers inflation penalty
        skills = [_make_skill(f"Skill{i}") for i in range(35)]  # depth=None for all

        resume = Resume(
            full_name="Pavel Inflator",
            educations=[Education(degree="bachelor", year_completed=2023)],
            experiences=[
                _make_exp("Startup", 2023, None, seniority="junior"),
            ],
            skills=skills,
            raw_text_length=600,
        )
        score = score_resume(resume)
        assert score.total < 50, f"Inflation cheater score should be < 50, got: {score.total}"

        # Verify inflation penalty was applied in SkillComponent reasoning
        skill_component = next(c for c in score.components if c.name == "SkillComponent")
        assert "INFLATION PENALTY" in skill_component.reasoning, (
            f"Expected INFLATION PENALTY in reasoning, got: {skill_component.reasoning}"
        )

    def test_weights_sum_to_one(self) -> None:
        """Component weights must sum exactly to 1.0."""
        from src.scorer import (
            _WEIGHT_DOMAIN,
            _WEIGHT_EDUCATION,
            _WEIGHT_EXPERIENCE,
            _WEIGHT_PROGRESSION,
            _WEIGHT_SKILLS,
        )

        total = (
            _WEIGHT_EXPERIENCE
            + _WEIGHT_SKILLS
            + _WEIGHT_PROGRESSION
            + _WEIGHT_EDUCATION
            + _WEIGHT_DOMAIN
        )
        assert total == pytest.approx(1.0), f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------------------
# Bug 4: Education relevance per occupation_family (tests D)
# ---------------------------------------------------------------------------


class TestEducationRelevanceByFamily:
    """Bug 4 regression: education field bonus must be family-scoped, not global."""

    def test_education_cs_relevant_for_it(self) -> None:
        """occupation_family='it' + field='Computer Science' must yield +10 relevant field bonus."""
        resume = Resume(
            full_name="Test Candidate",
            educations=[
                Education(degree="bachelor", field="Computer Science", year_completed=2020)
            ],
            experiences=[
                _make_exp("TechCorp", 2020, None, occupation_family="it", seniority="medior")
            ],
            raw_text_length=1000,
        )
        score = score_resume(resume)
        edu = next(c for c in score.components if c.name == "EducationComponent")
        assert "+10 relevant field" in edu.reasoning, (
            f"Expected '+10 relevant field' in reasoning for IT/CS, got: {edu.reasoning}"
        )
        # bachelor base = 50, +10 bonus = 60
        assert edu.score == pytest.approx(60.0), f"Expected score 60.0 (50+10), got {edu.score}"

    def test_education_sw_engineering_irrelevant_for_other_family(self) -> None:
        """Bug 4 regression: occupation_family='other' must NOT grant IT-field bonus.

        'Software Engineering' is in the 'it' family keyword set. When the candidate's
        primary_family is 'other' (e.g. sales/general roles), no bonus must be applied.
        """
        resume = Resume(
            full_name="Sales Candidate",
            educations=[
                Education(degree="bachelor", field="Software Engineering", year_completed=2015)
            ],
            experiences=[
                # occupation_family="other" — simulates a sales/admin role
                _make_exp("SalesCo", 2015, None, occupation_family="other", seniority="medior")
            ],
            raw_text_length=1000,
        )
        score = score_resume(resume)
        edu = next(c for c in score.components if c.name == "EducationComponent")
        assert "+10 relevant field" not in edu.reasoning, (
            f"Bug 4 regression: 'other' family must not grant IT field bonus. Got: {edu.reasoning}"
        )
        # bachelor base = 50, no bonus = 50
        assert edu.score == pytest.approx(50.0), f"Expected score 50.0 (no bonus), got {edu.score}"

    def test_education_nursing_relevant_for_healthcare(self) -> None:
        """occupation_family='healthcare' + field='Nursing' must yield +10 relevant field bonus."""
        resume = Resume(
            full_name="Nurse Candidate",
            educations=[Education(degree="bachelor", field="Nursing", year_completed=2018)],
            experiences=[
                _make_exp(
                    "Hospital", 2018, None, occupation_family="healthcare", seniority="medior"
                )
            ],
            raw_text_length=1000,
        )
        score = score_resume(resume)
        edu = next(c for c in score.components if c.name == "EducationComponent")
        assert "+10 relevant field" in edu.reasoning, (
            f"Expected healthcare+Nursing bonus in reasoning, got: {edu.reasoning}"
        )
        # bachelor base = 50, +10 bonus = 60
        assert edu.score == pytest.approx(60.0), f"Expected score 60.0 (50+10), got {edu.score}"

    def test_education_no_experiences_defaults_other(self) -> None:
        """Resume with no experiences → primary_family='other' → no relevant field bonus."""
        resume = Resume(
            full_name="No Experience Candidate",
            educations=[
                Education(degree="bachelor", field="Computer Science", year_completed=2024)
            ],
            # Empty experiences — scorer defaults primary_family to "other"
            experiences=[],
            raw_text_length=500,
        )
        score = score_resume(resume)
        edu = next(c for c in score.components if c.name == "EducationComponent")
        # "other" family has frozenset() — no relevant keywords — so no bonus
        assert "+10 relevant field" not in edu.reasoning, (
            f"No experiences → family='other' → must not grant bonus. Got: {edu.reasoning}"
        )
        # bachelor base = 50, no bonus = 50
        assert edu.score == pytest.approx(50.0), f"Expected score 50.0 (no bonus), got {edu.score}"

    def test_education_law_relevant_for_legal(self) -> None:
        """occupation_family='legal' + field='Law' must yield +10 relevant field bonus."""
        resume = Resume(
            full_name="Lawyer Candidate",
            educations=[Education(degree="master", field="Law", year_completed=2016)],
            experiences=[
                _make_exp("LawFirm", 2016, None, occupation_family="legal", seniority="senior")
            ],
            raw_text_length=1200,
        )
        score = score_resume(resume)
        edu = next(c for c in score.components if c.name == "EducationComponent")
        assert "+10 relevant field" in edu.reasoning, (
            f"Expected legal+Law bonus in reasoning, got: {edu.reasoning}"
        )
        # master base = 75, +10 bonus = 85
        assert edu.score == pytest.approx(85.0), f"Expected score 85.0 (75+10), got {edu.score}"


class TestExperienceOverlapDedup:
    """Verify ExperienceComponent merges overlapping intervals before summing.

    Parallel roles (e.g. main HPP 2020-2024 + side freelance 2021-2023) must
    not double-count toward total years of experience.
    """

    def test_two_fully_overlapping_roles_count_once(self) -> None:
        """Two roles 2020-2024 must count as 4 years, not 8."""
        resume = Resume(
            full_name="Parallel Worker",
            educations=[Education(degree="bachelor", year_completed=2019)],
            experiences=[
                _make_exp("MainCorp", 2020, 2024, seniority="senior"),
                _make_exp("SideClient", 2020, 2024, is_contractor=True, seniority="senior"),
            ],
            raw_text_length=1000,
        )
        score = score_resume(resume)
        exp_comp = next(c for c in score.components if c.name == "ExperienceComponent")
        # 4 years × 8 = 32 base + 10 senior bonus = 42 (not 64+10 from 8 years)
        assert "4 total years" in exp_comp.reasoning, (
            f"Expected 4 deduped years, got: {exp_comp.reasoning}"
        )

    def test_partial_overlap_uses_union_length(self) -> None:
        """2018-2022 + 2020-2024 must count as 6 years (union), not 8."""
        resume = Resume(
            full_name="Partial Overlap",
            educations=[Education(degree="bachelor", year_completed=2017)],
            experiences=[
                _make_exp("First", 2018, 2022, seniority="medior"),
                _make_exp("Second", 2020, 2024, seniority="senior"),
            ],
            raw_text_length=1000,
        )
        score = score_resume(resume)
        exp_comp = next(c for c in score.components if c.name == "ExperienceComponent")
        assert "6 total years" in exp_comp.reasoning, (
            f"Expected 6 union years, got: {exp_comp.reasoning}"
        )

    def test_disjoint_intervals_sum_normally(self) -> None:
        """Non-overlapping 2014-2018 + 2020-2024 must count as 8 years total."""
        resume = Resume(
            full_name="Sequential",
            educations=[Education(degree="bachelor", year_completed=2013)],
            experiences=[
                _make_exp("First", 2014, 2018, seniority="medior"),
                _make_exp("Second", 2020, 2024, seniority="senior"),
            ],
            raw_text_length=1000,
        )
        score = score_resume(resume)
        exp_comp = next(c for c in score.components if c.name == "ExperienceComponent")
        assert "8 total years" in exp_comp.reasoning, (
            f"Expected 8 disjoint years, got: {exp_comp.reasoning}"
        )
