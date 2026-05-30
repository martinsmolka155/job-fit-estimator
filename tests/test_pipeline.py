"""Integration tests for the Pipeline orchestrator.

The full pipeline talks to OpenAI and ISPV; these tests stub both so the
exception-propagation contract and cost-record persistence behaviour can
be exercised without real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.pipeline import Pipeline, PipelineInfrastructureError
from src.salary_ispv import ISPVDataMissingError
from src.schemas import Resume


@pytest.fixture
def fake_pdf(tmp_path: Path) -> Path:
    """Create a tiny digital PDF good enough for the extractor."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "fake_cv.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setFont("Helvetica", 11)
    text = c.beginText(50, 750)
    for line in (
        "Jan Novák",
        "jan@example.cz | Praha",
        "Senior Software Engineer at TestCorp, 2018 - Present",
        "Skills: Python, SQL, Docker",
        "BSc Computer Science, ČVUT, 2017",
    ):
        text.textLine(line)
    c.drawText(text)
    c.save()
    return path


def _build_pipeline_with_mocks(
    parser_side_effect: Exception | None = None,
) -> Pipeline:
    """Construct a Pipeline whose parser is replaced by a controllable mock.

    All other modules use real implementations.
    """
    pipeline = Pipeline.__new__(Pipeline)  # bypass __init__
    pipeline._parser_llm = MagicMock(model="mock-parser")  # type: ignore[attr-defined]
    pipeline._explainer_llm = MagicMock(model="mock-explainer")  # type: ignore[attr-defined]

    # Parser stub — either succeeds with empty Resume + error meta (mimics
    # the swallow path), or raises directly.
    parser_mock = MagicMock()
    if parser_side_effect is not None:

        def _raise(text: str) -> tuple[Resume, dict[str, Any]]:
            raise parser_side_effect

        parser_mock.parse.side_effect = _raise
    else:
        # Mimic ResumeParser's swallow path: empty Resume + error in meta.
        parser_mock.parse.return_value = (
            Resume(raw_text_length=10),
            {"cost_usd": 0.0023, "error": "llm_failed"},
        )
    pipeline.parser = parser_mock

    # Other components — minimal mocks; we won't reach them.
    pipeline.validator = MagicMock()
    pipeline.explainer = MagicMock()
    pipeline.estimator = MagicMock()
    pipeline._embedding_validator = None  # type: ignore[attr-defined]
    return pipeline


def test_parser_infra_failure_raises_pipeline_infrastructure_error(
    fake_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the parser swallows an LLM error and returns parse_meta['error'],
    the pipeline must raise PipelineInfrastructureError instead of letting
    the empty Resume cascade into a misleading MissingISCOError downstream.
    """
    cost_log = tmp_path / "cost_log.jsonl"
    monkeypatch.setattr(
        "src.pipeline.record_run", lambda r: cost_log.write_text(json.dumps(r.__dict__) + "\n")
    )
    # Patch reserve() so the test does not need a real budget log; also patch
    # _release_reservation so the finally block is a no-op in this test context.
    monkeypatch.setattr("src.pipeline.reserve", lambda run_id, cost: None)
    monkeypatch.setattr("src.pipeline._release_reservation", lambda run_id: None)

    pipeline = _build_pipeline_with_mocks()

    with pytest.raises(PipelineInfrastructureError, match="Parser LLM call failed"):
        pipeline.run(fake_pdf)


def test_failed_run_persists_cost_record(
    fake_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cost record must be written even when the pipeline raises after parser
    has already spent tokens. Otherwise DAILY_API_BUDGET_USD under-counts.
    """
    cost_log = tmp_path / "cost_log.jsonl"
    persisted: list[dict[str, Any]] = []

    def _capture(record: Any) -> None:
        persisted.append(record.__dict__)
        cost_log.write_text(json.dumps(record.__dict__) + "\n")

    monkeypatch.setattr("src.pipeline.record_run", _capture)
    # Patch reserve() / _release_reservation so the test does not need a real
    # budget log and the finally block remains a no-op.
    monkeypatch.setattr("src.pipeline.reserve", lambda run_id, cost: None)
    monkeypatch.setattr("src.pipeline._release_reservation", lambda run_id: None)

    pipeline = _build_pipeline_with_mocks()

    with pytest.raises(PipelineInfrastructureError):
        pipeline.run(fake_pdf)

    assert len(persisted) == 1, "Cost record must be written exactly once on failure"
    record = persisted[0]
    assert record["error"] == "PipelineInfrastructureError"
    assert record["parse_cost_usd"] == pytest.approx(0.0023), (
        "Parser tokens consumed before failure must be reflected in the cost log"
    )


def test_extract_failure_still_persists_cost_record(
    fake_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even an exception in the extract step (before any LLM call) must
    produce a cost record so downstream metrics see the failed run.
    """
    persisted: list[dict[str, Any]] = []
    monkeypatch.setattr("src.pipeline.record_run", lambda r: persisted.append(r.__dict__))
    # Patch reserve() / _release_reservation so the test does not need a real
    # budget log and the finally block remains a no-op.
    monkeypatch.setattr("src.pipeline.reserve", lambda run_id, cost: None)
    monkeypatch.setattr("src.pipeline._release_reservation", lambda run_id: None)
    monkeypatch.setattr(
        "src.pipeline.extract_text", MagicMock(side_effect=ISPVDataMissingError("simulated"))
    )

    pipeline = _build_pipeline_with_mocks()

    with pytest.raises(ISPVDataMissingError):
        pipeline.run(fake_pdf)

    assert len(persisted) == 1
    assert persisted[0]["error"] == "ISPVDataMissingError"
    assert persisted[0]["parse_cost_usd"] == 0.0


def test_paths_anchored_to_project_root() -> None:
    """src.paths constants must point to absolute locations under the repo,
    not relative paths that depend on the current working directory.
    """
    from src.paths import (
        COST_LOG_PATH,
        EXPLAINER_PROMPT_PATH,
        INFLATION_FACTORS_PATH,
        ISPV_XLSX_PATH,
        PARSER_PROMPT_PATH,
        PROJECT_ROOT,
    )

    for path in (
        PARSER_PROMPT_PATH,
        EXPLAINER_PROMPT_PATH,
        ISPV_XLSX_PATH,
        INFLATION_FACTORS_PATH,
        COST_LOG_PATH,
    ):
        assert path.is_absolute(), f"{path} must be absolute"
        assert PROJECT_ROOT in path.parents, f"{path} must live under {PROJECT_ROOT}"


def test_pipeline_init_wires_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise Pipeline.__init__ end-to-end with mocked LLM provider.

    Earlier integration tests bypassed __init__ via Pipeline.__new__, which
    meant provider wiring, ISPV loading, and inflation-factor merging were
    completely untested. This test covers the construction path.
    """
    fake_llm = MagicMock(model="gpt-4o-mini")

    def _fake_get_provider(name: str, model: str) -> MagicMock:
        return MagicMock(model=model)

    monkeypatch.setattr("src.pipeline.get_provider", _fake_get_provider)

    pipeline = Pipeline("openai", parser_model="gpt-4o-mini", explainer_model="gpt-4o-mini")

    # Required components present and wired.
    assert pipeline.parser is not None
    assert pipeline.validator is not None
    assert pipeline.explainer is not None
    assert pipeline.estimator is not None
    # Embedding validator stays None unless explicitly enabled.
    assert pipeline._embedding_validator is None  # type: ignore[attr-defined]
    # ISPV loader is wired iff data/ispv_2025.xlsx exists in this checkout.
    # Either way, the estimator must hold a reference (None or loaded).
    assert hasattr(pipeline.estimator, "_loader")  # noqa: SLF001
    # Both LLMs were resolved through get_provider.
    assert pipeline._parser_llm.model == "gpt-4o-mini"  # type: ignore[attr-defined]
    assert pipeline._explainer_llm.model == "gpt-4o-mini"  # type: ignore[attr-defined]
    # fake_llm reference held to prevent unused-variable warning.
    assert fake_llm is not None
