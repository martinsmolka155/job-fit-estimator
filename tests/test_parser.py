"""Tests for CV parser.

IMPORTANT: All tests use pytest-mock for LLMProvider — NO real API calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.parser import ResumeParser
from src.schemas import Education, Experience, Resume, Skill


def _make_mock_llm(resume: Resume) -> MagicMock:
    """Build a mock LLMProvider that returns the given Resume."""
    llm = MagicMock()
    llm.extract_structured.return_value = (
        resume,
        {
            "cost_usd": 0.005,
            "input_tokens": 500,
            "output_tokens": 200,
            "duration_s": 1.2,
            "model": "gpt-4o-mini",
        },
    )
    return llm


def _make_full_resume() -> Resume:
    """Minimal valid Resume for testing."""
    return Resume(
        full_name="Jan Novák",
        location="Praha",
        email="jan@example.cz",
        experiences=[
            Experience(
                role_title="Software Developer",
                isco_code="2512",
                occupation_family="it",
                company="Acme Corp",
                start_year=2022,
                end_year=2024,
                seniority_level="medior",
            )
        ],
        educations=[Education(degree="bachelor", institution="CVUT FIT", year_completed=2022)],
        skills=[
            Skill(name="Python", category="language"),
            Skill(name="FastAPI", category="framework"),
        ],
        raw_text_length=0,  # will be overwritten by parser
    )


class TestResumeParser:
    def test_valid_cv_returns_resume_with_required_fields(self, tmp_path: Path) -> None:
        """A valid CV text should produce a Resume with populated fields."""
        resume = _make_full_resume()
        llm = _make_mock_llm(resume)
        parser = ResumeParser(llm)

        cv_text = "Jan Novák\njan@example.cz | Praha\nSoftware Developer at Acme Corp 2022-2024\nPython FastAPI"
        result, meta = parser.parse(cv_text)

        assert isinstance(result, Resume)
        assert result.full_name == "Jan Novák"
        assert result.email == "jan@example.cz"
        assert len(result.experiences) == 1
        assert result.raw_text_length == len(cv_text)
        assert meta["cost_usd"] > 0

    def test_czech_cv_has_occupation_family(self, tmp_path: Path) -> None:
        """Czech CV should produce an Experience with isco_code + occupation_family set."""
        resume = Resume(
            full_name="Petr Svoboda",
            experiences=[
                Experience(
                    role_title="Vývojář softwaru",
                    isco_code="2512",
                    occupation_family="it",
                    company="Česká firma s.r.o.",
                    start_year=2021,
                    seniority_level="senior",
                )
            ],
            raw_text_length=0,
        )
        llm = _make_mock_llm(resume)
        parser = ResumeParser(llm)

        cv_text = (
            "Petr Svoboda\nSenior Vývojář softwaru\nČeská firma s.r.o. 2021-dosud\nPython, Docker"
        )
        result, _ = parser.parse(cv_text)

        assert result.experiences[0].occupation_family == "it"
        assert result.experiences[0].isco_code == "2512"

    def test_freelance_cv_sets_is_contractor_true(self) -> None:
        """CV with freelance/OSVČ keywords should produce is_contractor=True."""
        resume = Resume(
            full_name="Marie Horáková",
            experiences=[
                Experience(
                    role_title="Freelance Backend Developer",
                    occupation_family="it",
                    company="Self-employed",
                    start_year=2020,
                    is_contractor=True,  # Parser should detect this
                    seniority_level="senior",
                )
            ],
            raw_text_length=0,
        )
        llm = _make_mock_llm(resume)
        parser = ResumeParser(llm)

        cv_text = "Marie Horáková\nFreelance Backend Developer\nOSVČ 2020-dosud\nPython, FastAPI, PostgreSQL"
        result, _ = parser.parse(cv_text)

        assert result.experiences[0].is_contractor is True

    def test_empty_text_returns_empty_resume(self) -> None:
        """Empty text should return Resume with empty lists and None fields."""
        empty_resume = Resume(raw_text_length=0)
        llm = _make_mock_llm(empty_resume)
        parser = ResumeParser(llm)

        result, meta = parser.parse("")

        assert isinstance(result, Resume)
        assert result.full_name is None
        assert result.experiences == []
        assert result.skills == []
        assert result.educations == []
        assert result.raw_text_length == 0

    def test_long_text_gets_truncated(self) -> None:
        """Text longer than 30k chars should be truncated with meta['truncated']=True."""
        resume = _make_full_resume()
        llm = _make_mock_llm(resume)
        parser = ResumeParser(llm)

        # Create text longer than 30k chars
        long_text = "Python developer " * 2000  # ~34k chars
        assert len(long_text) > 30_000

        result, meta = parser.parse(long_text)

        assert meta.get("truncated") is True
        # raw_text_length should be truncated length (30k), not original
        assert result.raw_text_length == 30_000

    def test_llm_failure_returns_empty_resume_with_error_meta(self) -> None:
        """If LLM fails, parser should return empty Resume with error in meta."""
        llm = MagicMock()
        llm.extract_structured.side_effect = RuntimeError("API down")
        parser = ResumeParser(llm)

        result, meta = parser.parse("some CV text")

        assert isinstance(result, Resume)
        assert meta.get("error") == "llm_failed"


def test_parser_preserves_cost_on_llm_provider_error() -> None:
    """Parser must propagate the partial cost from LLMProviderError so a
    refusal / validation failure after token spend doesn't show as $0.
    """
    from unittest.mock import MagicMock

    from src.llm_provider import LLMProviderError
    from src.parser import ResumeParser

    llm = MagicMock()
    llm.extract_structured.side_effect = LLMProviderError(
        "refused",
        cost_usd=0.0017,
        meta={"input_tokens": 800, "output_tokens": 30, "model": "gpt-4o-mini"},
    )
    parser = ResumeParser(llm)

    _resume, meta = parser.parse("dummy CV text")

    assert meta["error"] == "llm_failed"
    assert meta["cost_usd"] == pytest.approx(0.0017)
    # Token detail from the failed call surfaces too — operator can investigate.
    assert meta["input_tokens"] == 800
