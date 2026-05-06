"""Golden-set ISCO/occupation_family eval for the LLM parser.

Requires a live OpenAI API call — run manually as part of the eval suite.
Costs approximately $0.01-0.05 per CV (gpt-4o-mini); 3-CV set ≈ $0.05-0.15 total.

Golden set:
  cv_it_developer.txt    → isco_code="2512", occupation_family="it"
  cv_healthcare_nurse.txt → isco_code=None (TBD: 3221 vs 2221), occupation_family="healthcare"
  cv_trades_cook.txt     → isco_code="5120", occupation_family="services"

Run manually:
  uv run pytest tests/test_parser_isco_eval.py -v --no-header -rN

  Or with live API:
  OPENAI_API_KEY=sk-... uv run pytest tests/test_parser_isco_eval.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import settings
from src.llm_provider import OpenAIProvider
from src.parser import ResumeParser

FIXTURES_DIR = Path(__file__).parent / "fixtures"

GOLDEN_SET = [
    ("cv_it_developer.txt", "2512", "it"),
    # CZ-ISCO-08 classification for "Sestra specialistka ARIP" (DiS. + postgrad ARIP):
    #   2221 = Nursing and midwifery professionals (registrovaná sestra s vyšší kvalifikací)
    #   3221 = Nursing associate professionals (praktická sestra bez specializace)
    # Diploma + ARIP specialization pushes this into the 2221 professional category.
    # Live LLM eval (gpt-4o-mini, 2026-05-05) returned 2221 — accepted as correct.
    ("cv_healthcare_nurse.txt", "2221", "healthcare"),
    # ISCO 5120 (kuchaři) — cook/chef category.
    # Note: ISCO 5120 maps to "services" via _ISCO_FAMILY (major group 5).
    ("cv_trades_cook.txt", "5120", "services"),
    # ISCO 3322 (Obchodní zástupci / Sales representatives) — B2B sales role.
    # Acceptable alternatives: 3323 (Zásobovací agenti), 3339 (Business services agents).
    # Fail condition: 2411, 2421 (financial analysts/lawyers) = Bug 2 regression.
    ("cv_sales_repr.txt", "3322", "other"),
]

_SKIP_REASON = (
    "golden-set eval: requires live OpenAI API — set OPENAI_API_KEY env var and run manually. "
    "Skip marker intentional to avoid CI cost; remove to run."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.parametrize("cv_file,expected_isco,expected_family", GOLDEN_SET)
def test_parser_assigns_correct_isco(
    cv_file: str,
    expected_isco: str | None,
    expected_family: str,
) -> None:
    """Verify LLM parser assigns expected ISCO code and occupation_family.

    Uses real LLM (OpenAI gpt-4o-mini) with settings from .env.

    Acceptance criterion:
      - occupation_family must match exactly for all 3 fixtures
      - isco_code must match where expected_isco is not None
      - At least one experience must be parsed from each CV
    """
    cv_path = FIXTURES_DIR / cv_file
    assert cv_path.exists(), f"Fixture not found: {cv_path}"

    # Use API key from env or settings
    api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key
    assert api_key, "OPENAI_API_KEY must be set to run golden-set eval"

    cv_text = cv_path.read_text(encoding="utf-8")

    provider = OpenAIProvider(model=settings.parser_model, api_key=api_key)
    parser = ResumeParser(llm=provider)

    resume, meta = parser.parse(cv_text)

    assert resume.experiences, f"No experiences parsed from {cv_file}"

    # Pick the most recent experience (highest start_year)
    latest = max(resume.experiences, key=lambda e: e.start_year)

    assert latest.occupation_family == expected_family, (
        f"{cv_file}: expected occupation_family={expected_family!r}, "
        f"got {latest.occupation_family!r} (role_title={latest.role_title!r})"
    )

    if expected_isco is not None:
        assert latest.isco_code == expected_isco, (
            f"{cv_file}: expected isco_code={expected_isco!r}, "
            f"got {latest.isco_code!r} (role_title={latest.role_title!r})\n"
            f"  occupation_family={latest.occupation_family!r}\n"
            f"  cost_usd={meta.get('cost_usd', 'n/a')}"
        )
