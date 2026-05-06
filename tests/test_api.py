"""Tests for the FastAPI HTTP entry point.

Cover the error-mapping contract — each pipeline failure mode must produce
the documented HTTP status, and uploaded temp files must be cleaned up
even when the Pipeline constructor itself raises.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import api
from src.cost_tracker import BudgetExceededError
from src.salary_ispv import (
    ISPVDataMissingError,
    MissingISCOError,
    NonCZLocationError,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def _upload(client: TestClient, suffix: str = ".pdf") -> httpx.Response:
    return client.post(
        "/analyze",
        files={"cv": (f"sample{suffix}", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unsupported_extension_returns_415(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        files={"cv": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_budget_exceeded_returns_429(client: TestClient) -> None:
    """BudgetExceededError must be surfaced as 429 — operational state, not 500."""
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = BudgetExceededError("over budget")
        response = _upload(client)

    assert response.status_code == 429
    assert "over budget" in response.json()["detail"]


def test_ispv_data_missing_returns_503(client: TestClient) -> None:
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = ISPVDataMissingError("ISPV not loaded")
        response = _upload(client)
    assert response.status_code == 503


def test_non_cz_location_returns_422(client: TestClient) -> None:
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = NonCZLocationError("Berlin")
        response = _upload(client)
    assert response.status_code == 422


def test_missing_isco_returns_422(client: TestClient) -> None:
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = MissingISCOError("no ISCO")
        response = _upload(client)
    assert response.status_code == 422


def test_pipeline_init_failure_cleans_up_tempfile(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Pipeline() constructor raises, the uploaded temp file must NOT leak.

    Regression for the prior ordering where Pipeline(...) lived OUTSIDE the
    try/finally block and bypassed both the HTTP error mapping and the
    temp-file cleanup.
    """
    leaked_paths: list[Path] = []
    real_named_tempfile = api.tempfile.NamedTemporaryFile

    def _tracking_tempfile(*args: object, **kwargs: object) -> object:
        handle = real_named_tempfile(*args, **kwargs)  # type: ignore[arg-type]
        leaked_paths.append(Path(handle.name))
        return handle

    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", _tracking_tempfile)

    with patch.object(api, "Pipeline", side_effect=RuntimeError("boom")):
        response = _upload(client)

    assert response.status_code == 500
    # Temp file must be cleaned up even though Pipeline() raised before run().
    assert leaked_paths, "tempfile fixture not exercised"
    for path in leaked_paths:
        assert not path.exists(), f"Tempfile leaked on Pipeline init failure: {path}"
