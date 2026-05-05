"""Tests for hallucination guard / Resume validator.

Key test: normalization ensures "Česká spořitelna" in CV matches "Ceska Sporitelna" in Resume.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.llm_provider import LLMProvider
from src.schemas import Education, Experience, Resume, Skill, ValidationFlag
from src.validator import (
    EmbeddingValidator,
    ResumeValidator,
    _cosine_similarity,  # pyright: ignore[reportPrivateUsage]
    _normalize,  # pyright: ignore[reportPrivateUsage]
    compute_confidence,
)


class _MockProvider(LLMProvider):
    """Minimal mock LLMProvider for EmbeddingValidator tests — no real API calls."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings

    def extract_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, dict[str, object]]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, object]]:
        vecs = self.embeddings[: len(texts)]
        return vecs, {"cost_usd": 0.0, "input_tokens": len(texts), "model": "mock"}


class TestCosineSimlarity:
    def test_identical_vectors_return_one(self) -> None:
        """Identical non-zero vectors must have cosine similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        result = _cosine_similarity(v, v)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Orthogonal vectors (dot product = 0) must have cosine similarity 0.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self) -> None:
        """Zero vector must return 0.0 without NaN or ZeroDivisionError."""
        zero = [0.0, 0.0, 0.0]
        other = [1.0, 2.0, 3.0]
        assert _cosine_similarity(zero, other) == 0.0
        assert _cosine_similarity(other, zero) == 0.0

    def test_length_mismatch_raises(self) -> None:
        """Mismatched vector lengths must raise ValueError."""
        with pytest.raises(ValueError, match="Vector length mismatch"):
            _cosine_similarity([1.0, 2.0], [1.0])


class TestEmbeddingValidator:
    def _make_error_flag(self, field: str = "experiences.0.company") -> ValidationFlag:
        return ValidationFlag(
            field_path=field,
            issue="Company 'Czech Technical University' not found in raw text",
            severity="error",
        )

    def _make_warning_flag(self, field: str = "experiences.0.company") -> ValidationFlag:
        return ValidationFlag(
            field_path=field,
            issue="Company 'CVUT' not found in raw text",
            severity="warning",
        )

    def _make_info_flag(self) -> ValidationFlag:
        return ValidationFlag(
            field_path="educations.0.year_completed",
            issue="Graduation year 2020 not found in raw text",
            severity="info",
        )

    def test_no_op_on_info_flags_only(self) -> None:
        """When all flags are severity='info', upgrade_flags returns them unchanged."""
        provider = _MockProvider(embeddings=[])
        validator = EmbeddingValidator(provider)
        flags = [self._make_info_flag()]
        result_flags, meta = validator.upgrade_flags(flags, "some raw text")
        assert result_flags == flags
        assert meta["embed_cost_usd"] == 0.0

    def test_emits_info_flag_for_high_similarity(self) -> None:
        """When max cosine similarity > threshold, a new 'info' flag must be appended."""
        # Two chunks + one flag text — all embedded as parallel vectors (sim = 1.0)
        raw_text = "A" * 200 + "B" * 200  # two 200-char chunks
        flag = self._make_error_flag()

        # chunk_0, chunk_1, flag_text — all identical → sim = 1.0 > 0.85
        identical_vec = [1.0, 0.0, 0.0]
        provider = _MockProvider(embeddings=[identical_vec, identical_vec, identical_vec])
        validator = EmbeddingValidator(provider, threshold=0.85)

        result_flags, meta = validator.upgrade_flags([flag], raw_text)

        assert len(result_flags) == 2
        info_flags = [f for f in result_flags if f.severity == "info"]
        assert len(info_flags) == 1
        assert "Fuzzy match found" in info_flags[0].issue
        assert "0.85" in info_flags[0].issue
        assert "embed_cost_usd" in meta

    def test_no_emit_for_low_similarity(self) -> None:
        """When max cosine similarity <= threshold, no new flag is emitted."""
        raw_text = "X" * 200  # one chunk
        flag = self._make_warning_flag()

        # chunk perpendicular to flag_text → sim = 0.0
        provider = _MockProvider(
            embeddings=[
                [1.0, 0.0],  # chunk
                [0.0, 1.0],  # flag text
            ]
        )
        validator = EmbeddingValidator(provider, threshold=0.85)

        result_flags, _meta = validator.upgrade_flags([flag], raw_text)

        assert len(result_flags) == 1
        assert result_flags[0].severity == "warning"

    def test_original_error_flag_preserved(self) -> None:
        """Original error flag must NOT be removed — only new info flags are appended."""
        raw_text = "A" * 200
        flag = self._make_error_flag()
        identical_vec = [1.0, 0.0]
        provider = _MockProvider(embeddings=[identical_vec, identical_vec])
        validator = EmbeddingValidator(provider, threshold=0.5)

        result_flags, _meta = validator.upgrade_flags([flag], raw_text)

        error_flags = [f for f in result_flags if f.severity == "error"]
        assert len(error_flags) == 1
        assert error_flags[0].field_path == flag.field_path

    def test_empty_flags_returns_empty(self) -> None:
        """Empty flags list returns empty list without calling provider."""
        provider = _MockProvider(embeddings=[])
        validator = EmbeddingValidator(provider)
        result_flags, meta = validator.upgrade_flags([], "some text")
        assert result_flags == []
        assert meta["embed_cost_usd"] == 0.0

    def test_empty_raw_text_returns_original_flags(self) -> None:
        """Empty raw_text returns original flags unchanged."""
        provider = _MockProvider(embeddings=[])
        validator = EmbeddingValidator(provider)
        flag = self._make_error_flag()
        result_flags, meta = validator.upgrade_flags([flag], "")
        assert result_flags == [flag]
        assert meta["embed_cost_usd"] == 0.0

    def test_field_path_preserved_in_info_flag(self) -> None:
        """Emitted info flag must have the same field_path as its source flag."""
        raw_text = "A" * 200
        flag = self._make_warning_flag(field="skills.3.name")
        identical_vec = [0.5, 0.5]
        provider = _MockProvider(embeddings=[identical_vec, identical_vec])
        validator = EmbeddingValidator(provider, threshold=0.5)

        result_flags, _meta = validator.upgrade_flags([flag], raw_text)

        info_flags = [f for f in result_flags if f.severity == "info"]
        assert info_flags[0].field_path == "skills.3.name"


class TestNormalize:
    def test_lowercase_and_accents(self) -> None:
        """_normalize should lowercase and strip Czech diacritics."""
        assert _normalize("Česká spořitelna") == "ceska sporitelna"
        assert _normalize("PŘÍLIŠ ŽLUŤOUČKÝ KŮŇ") == "prilis zlutoucky kun"
        assert _normalize("  Multiple   Spaces  ") == "multiple spaces"

    def test_accent_matching(self) -> None:
        """Normalized 'Česká spořitelna' == normalized 'Ceska Sporitelna'."""
        assert _normalize("Česká spořitelna") == _normalize("Ceska Sporitelna")


class TestResumeValidator:
    def _make_raw_text(self) -> str:
        return (
            "Jan Novák\n"
            "jan@example.cz | +420 123 456 789 | Praha\n\n"
            "EXPERIENCE\n"
            "Software Developer\n"
            "Česká spořitelna, Praha\n"
            "2020 - 2024\n\n"
            "EDUCATION\n"
            "BSc Computer Science\n"
            "CVUT FIT, 2020\n\n"
            "SKILLS\n"
            "Python, SQL, Git, Docker"
        )

    def _make_valid_resume(self) -> Resume:
        return Resume(
            full_name="Jan Novák",
            email="jan@example.cz",
            location="Praha",
            experiences=[
                Experience(
                    role_title="Software Developer",
                    isco_code="2512",
                    occupation_family="it",
                    company="Česká spořitelna",
                    start_year=2020,
                    end_year=2024,
                )
            ],
            educations=[Education(degree="bachelor", institution="CVUT FIT", year_completed=2020)],
            skills=[
                Skill(name="Python", category="language"),
                Skill(name="SQL", category="language"),
            ],
            raw_text_length=500,
        )

    def test_valid_resume_produces_no_flags(self) -> None:
        """A Resume that exactly matches the raw text should produce 0 flags."""
        validator = ResumeValidator()
        resume = self._make_valid_resume()
        raw_text = self._make_raw_text()

        flags = validator.validate(resume, raw_text)

        assert flags == []

    def test_normalization_prevents_false_positive(self) -> None:
        """Resume with 'Ceska Sporitelna' (no accents) should match CV with 'Česká spořitelna'."""
        validator = ResumeValidator()
        raw_text = self._make_raw_text()

        # Resume has non-accented version — normalization should bridge the gap
        resume = self._make_valid_resume()
        resume = resume.model_copy(
            update={
                "experiences": [
                    resume.experiences[0].model_copy(update={"company": "Ceska Sporitelna"})
                ]
            }
        )

        flags = validator.validate(resume, raw_text)

        # There should be NO flag for company — normalization makes them match
        company_flags = [f for f in flags if "company" in f.field_path]
        assert company_flags == [], f"Unexpected company flag: {company_flags}"

    def test_hallucinated_company_produces_warning(self) -> None:
        """A company not in raw text should produce a warning flag."""
        validator = ResumeValidator()
        raw_text = self._make_raw_text()

        resume = self._make_valid_resume()
        resume = resume.model_copy(
            update={
                "experiences": [
                    resume.experiences[0].model_copy(
                        update={"company": "Completely Invented Corp Ltd"}
                    )
                ]
            }
        )

        flags = validator.validate(resume, raw_text)

        company_flags = [f for f in flags if "company" in f.field_path]
        assert len(company_flags) == 1
        assert company_flags[0].severity == "warning"

    def test_hallucinated_email_produces_error(self) -> None:
        """An email not in raw text should produce an ERROR flag (strong hallucination signal)."""
        validator = ResumeValidator()
        raw_text = self._make_raw_text()

        resume = self._make_valid_resume()
        resume = resume.model_copy(update={"email": "hallucinated@nowhere.com"})

        flags = validator.validate(resume, raw_text)

        email_flags = [f for f in flags if f.field_path == "email"]
        assert len(email_flags) == 1
        assert email_flags[0].severity == "error"

    def test_excessive_experience_years_produces_warning(self) -> None:
        """Total experience years >> CV length should produce a warning."""
        validator = ResumeValidator()

        # Short raw text (50 chars) but resume claims 30 years of experience
        raw_text = "Jan Novak jan@example.cz 2020 Acme Python SQL Git"
        resume = Resume(
            full_name="Jan Novak",
            email="jan@example.cz",
            experiences=[
                Experience(
                    role_title="Developer",
                    company="Acme",
                    start_year=1994,
                    end_year=2024,
                )
            ],
            raw_text_length=len(raw_text),
        )

        flags = validator.validate(resume, raw_text)

        experience_flags = [f for f in flags if f.field_path == "experiences"]
        assert len(experience_flags) == 1
        assert experience_flags[0].severity == "warning"


class TestComputeConfidence:
    def test_no_flags_is_full_confidence(self) -> None:
        assert compute_confidence([]) == 1.0

    def test_one_error_reduces_by_20_pct(self) -> None:
        from src.schemas import ValidationFlag

        flags = [ValidationFlag(field_path="email", issue="not found", severity="error")]
        assert compute_confidence(flags) == pytest.approx(0.8)

    def test_one_warning_reduces_by_5_pct(self) -> None:
        from src.schemas import ValidationFlag

        flags = [ValidationFlag(field_path="company", issue="not found", severity="warning")]
        assert compute_confidence(flags) == pytest.approx(0.95)

    def test_minimum_is_zero(self) -> None:
        from src.schemas import ValidationFlag

        flags = [
            ValidationFlag(field_path=f"x.{i}", issue="not found", severity="error")
            for i in range(10)
        ]
        assert compute_confidence(flags) == 0.0
