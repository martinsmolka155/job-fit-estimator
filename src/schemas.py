"""Pydantic v2 data models — single source of truth for all data contracts.

All Pydantic models MUST be defined here. No other module may define BaseModel subclasses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Education(BaseModel):
    """Academic education entry from CV."""

    degree: Literal["bachelor", "master", "phd", "other", "none"] = Field(
        description="Highest degree level achieved"
    )
    field: str | None = Field(default=None, description="Field of study (e.g. Computer Science)")
    institution: str | None = Field(default=None, description="University or school name")
    year_completed: int | None = Field(default=None, description="Year the degree was completed")


class Experience(BaseModel):
    """Work experience entry from CV."""

    role_title: str = Field(description="Original role title text from CV")
    isco_code: str | None = Field(
        default=None,
        pattern=r"^[1-9][0-9]{3}$",
        description="CZ-ISCO-08 4-digit occupational code, e.g. '2512' (Software developers). Major group must be 1-9.",
    )
    occupation_family: (
        Literal["it", "healthcare", "legal", "education", "trades", "services", "other"] | None
    ) = Field(
        default=None,
        description="Broad occupation family for scoring weight selection",
    )
    company: str = Field(description="Company or client name")
    start_year: int = Field(description="Year the role started")
    end_year: int | None = Field(
        default=None, description="Year the role ended; None means currently active"
    )
    description: str | None = Field(
        default=None, description="Role responsibilities and achievements"
    )
    seniority_level: Literal["junior", "medior", "senior", "lead", "principal"] | None = Field(
        default=None, description="Seniority level parsed from role title"
    )
    is_management: bool = Field(default=False, description="True if the role had management scope")
    is_contractor: bool = Field(
        default=False,
        description="True if freelance/contractor/consultant — detected from title or description",
    )


class Skill(BaseModel):
    """Technical or soft skill extracted from CV."""

    name: str = Field(description="Skill name as written in CV")
    category: Literal["language", "framework", "tool", "soft", "domain"] = Field(
        description="Skill category for scoring"
    )
    years_used: int | None = Field(
        default=None, description="Years of practical experience with this skill"
    )
    depth: Literal["basic", "intermediate", "advanced", "expert"] | None = Field(
        default=None, description="Self-assessed proficiency depth; None if unclear"
    )


class Language(BaseModel):
    """Spoken language with proficiency level."""

    name: str = Field(description="Language name, e.g. 'Czech', 'English'")
    level: str | None = Field(
        default=None,
        description="Proficiency level, e.g. 'native', 'C1', 'B2'",
    )


class Resume(BaseModel):
    """Fully parsed CV — output of the LLM parser."""

    full_name: str | None = Field(default=None, description="Candidate full name")
    location: str | None = Field(default=None, description="City or region (e.g. Praha, Brno)")
    email: str | None = Field(default=None, description="Contact email address")
    phone: str | None = Field(default=None, description="Contact phone number")
    languages: list[Language] = Field(
        default_factory=list,
        description="List of spoken languages with proficiency levels",
    )
    educations: list[Education] = Field(
        default_factory=list, description="Education history, most recent first"
    )
    experiences: list[Experience] = Field(
        default_factory=list, description="Work experience history, most recent first"
    )
    skills: list[Skill] = Field(
        default_factory=list, description="All extracted skills (deduplicated)"
    )
    raw_text_length: int = Field(
        default=0, description="Character count of source text used for parsing"
    )


class ValidationFlag(BaseModel):
    """Hallucination guard flag — a discrepancy found between Resume and raw text."""

    field_path: str = Field(
        description="Dotted path to the flagged field, e.g. 'experiences.0.company'"
    )
    issue: str = Field(description="Human-readable description of the discrepancy")
    severity: Literal["info", "warning", "error"] = Field(
        description="info: minor mismatch; warning: likely hallucination; error: strong signal"
    )


class ScoreComponent(BaseModel):
    """Single component of the seniority score."""

    name: str = Field(description="Component name (e.g. 'ExperienceComponent')")
    score: float = Field(description="Component score 0-100")
    weight: float = Field(description="Weight in final total (all weights sum to 1.0)")
    reasoning: str = Field(description="Human-readable explanation of this component's score")


class SeniorityScore(BaseModel):
    """Aggregated seniority score produced by the rule-based scorer."""

    total: float = Field(description="Weighted total score 0-100")
    components: list[ScoreComponent] = Field(description="Breakdown by scoring component")
    confidence: float = Field(
        description="Confidence 0-1, reduced by validation flags from hallucination guard"
    )


class SalaryEstimate(BaseModel):
    """Salary range estimate in CZK (monthly gross)."""

    currency: Literal["CZK", "EUR"] = Field(description="Currency of the estimate")
    low: int = Field(description="Lower bound of salary range")
    mid: int = Field(description="Mid-point / most likely salary")
    high: int = Field(description="Upper bound of salary range")
    reasoning: str = Field(description="Explanation of how estimate was derived")
    assumptions: list[str] = Field(
        description="Assumptions made (role type, location multiplier, etc.)"
    )


class Recommendation(BaseModel):
    """Single actionable career recommendation."""

    title: str = Field(description="Short actionable title, e.g. 'Master Kubernetes in production'")
    why_it_matters: str = Field(description="1-2 sentences on why this recommendation, not others")
    estimated_salary_impact_pct: float = Field(
        ge=0,
        le=100,
        description="Realistic salary uplift estimate in percent (5-15 typical per recommendation)",
    )
    timeframe_months: int = Field(
        ge=1,
        le=60,
        description="Expected months to achieve this recommendation",
    )
    first_action: str = Field(description="Concrete step the candidate can take tomorrow morning")


class Explanation(BaseModel):
    """LLM-generated career analysis with exactly 3 recommendations."""

    summary: str = Field(description="2-3 sentence neutral candidate summary")
    strengths: list[str] = Field(description="3-5 specific strengths (never generic)")
    gaps: list[str] = Field(description="3-5 actionable gaps (never 'more experience')")
    recommendations: list[Recommendation] = Field(
        description="Exactly 3 recommendations whose estimated_salary_impact_pct sum >= 30%"
    )

    @field_validator("recommendations")
    @classmethod
    def must_have_three_recommendations(cls, v: list[Recommendation]) -> list[Recommendation]:
        if len(v) != 3:
            raise ValueError(f"Explanation must have exactly 3 recommendations, got {len(v)}")
        return v


class ExtractedDocument(BaseModel):
    """Raw text extraction result from PDF or DOCX."""

    text: str = Field(description="Extracted raw text content")
    source_format: Literal["pdf", "docx"] = Field(description="Input file format")
    extraction_method: Literal["pymupdf", "docx", "scanned_unsupported"] = Field(
        description="Method used; scanned_unsupported means OCR would be needed"
    )
    page_count: int = Field(description="Number of pages in the source document")
    char_count: int = Field(description="Character count of extracted text")
    is_scanned: bool = Field(
        default=False,
        description="True if document appears to be a scanned image (OCR not supported)",
    )


class SalaryBand(BaseModel):
    """Single seniority band: low/mid/high CZK monthly gross."""

    low: int = Field(description="Lower bound CZK monthly gross")
    mid: int = Field(description="Median CZK monthly gross")
    high: int = Field(description="Upper bound CZK monthly gross")


class SalaryData(BaseModel):
    """Researched or cached salary band data for one role."""

    role_name: str = Field(description="Human-readable role name (e.g. 'Backend Developer')")
    role_slug: str = Field(description="Cache-key-safe slug (e.g. 'backend_developer')")
    researched_at: datetime = Field(description="When research was performed")
    ttl_days: int = Field(default=90, description="Cache TTL in days")
    junior: SalaryBand | None = Field(default=None, description="Junior band; null if no data")
    medior: SalaryBand | None = Field(default=None, description="Medior band; null if no data")
    senior: SalaryBand | None = Field(default=None, description="Senior band; null if no data")
    lead: SalaryBand | None = Field(default=None, description="Lead band; null if no data")
    principal: SalaryBand | None = Field(
        default=None, description="Principal band; null if no data"
    )
    isco_code: str | None = Field(
        default=None,
        pattern=r"^[1-9][0-9]{3}$",
        description="CZ-ISCO-08 4-digit code used for lookup",
    )
    isco_match_level: Literal["4digit", "3digit", "2digit", "1digit", "family_fallback"] = Field(
        description="Precision of ISCO match achieved",
    )
    dataset_age_months: int = Field(
        ge=0,
        description="Months since ISPV dataset publication",
    )
    inflation_factor_applied: float = Field(
        ge=1.0,
        description="Multiplier applied to raw ISPV figures (>=1.0; deflation case is theoretical)",
    )
    source_dataset_version: str = Field(
        description="ISPV dataset version identifier, e.g. 'ISPV-2025-annual', for reproducibility",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="high = 3+ sources agree within ±10%, medium = 2 sources or ±20%, low = 1 source or wide disagreement",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Concerns noted during extraction (low source count, USD/CZK mismatch, stale data, etc.)",
    )


class PipelineResult(BaseModel):
    """Full output of the end-to-end CV analysis pipeline."""

    resume: Resume = Field(description="Structured Resume extracted by LLM parser")
    validation_flags: list[ValidationFlag] = Field(
        description="Hallucination guard flags from validator"
    )
    score: SeniorityScore = Field(description="Rule-based seniority score")
    salary: SalaryEstimate = Field(description="Heuristic salary estimate")
    explanation: Explanation = Field(description="LLM-generated career analysis")
    meta: dict = Field(  # type: ignore[type-arg]
        default_factory=dict,
        description="Pipeline metadata: cost_usd, duration_s, model, per-step timings",
    )
