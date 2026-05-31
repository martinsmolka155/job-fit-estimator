"""Tests for the FastAPI HTTP entry point.

Cover the error-mapping contract — each pipeline failure mode must produce
the documented HTTP status, and uploaded temp files must be cleaned up
even when the Pipeline constructor itself raises.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

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
    # Reset the process-local rate limiter so per-test upload counts do not
    # accumulate across the session (all TestClient requests share one IP key).
    api.rate_limiter.reset()
    return TestClient(api.app)


def _upload(client: TestClient, suffix: str = ".pdf") -> httpx.Response:
    return client.post(
        "/analyze",
        files={"cv": (f"sample{suffix}", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
    )


def _make_pipeline_result(**meta_overrides: Any) -> MagicMock:
    """Build a mock PipelineResult whose model_dump returns a minimal valid dict."""
    result = MagicMock()
    result.validation_flags = []
    result.score = MagicMock(confidence=0.9)
    result.meta = {"salary_data": None, **meta_overrides}
    result.model_dump.return_value = {"score": {"total": 80}}
    return result


def test_honeypot_filled_returns_400_without_pipeline(client: TestClient) -> None:
    """A bot filling the hidden company_url field is rejected before the pipeline."""
    with patch("api._build_pipeline") as build:
        response = client.post(
            "/analyze",
            files={"cv": ("cv.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
            data={"company_url": "http://spam.example"},
        )
    assert response.status_code == 400
    build.assert_not_called()  # never reached the cost-bearing pipeline


def test_honeypot_empty_passes_through(client: TestClient) -> None:
    """Empty honeypot (real user) does not block the request."""
    response = client.post(
        "/analyze",
        files={"cv": ("cv.txt", io.BytesIO(b"x"), "text/plain")},
        data={"company_url": ""},
    )
    # Rejected for the unsupported extension, NOT the honeypot — proves it passed.
    assert response.status_code == 415


def test_turnstile_enabled_missing_token_returns_403(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Turnstile is configured, a request without a token is refused."""
    monkeypatch.setattr(api.settings, "turnstile_secret_key", "test-secret")
    response = client.post(
        "/analyze",
        files={"cv": ("cv.pdf", io.BytesIO(b"%PDF-1.4 fake"), "application/pdf")},
        data={"cf_turnstile_response": ""},
    )
    assert response.status_code == 403


def test_turnstile_disabled_skips_verification(client: TestClient) -> None:
    """With no secret key, Turnstile is a no-op (default state)."""
    assert api.settings.turnstile_secret_key == ""
    response = _upload(client, ".docx")  # rejected later for fake content, not 403
    assert response.status_code != 403


def test_index_returns_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_privacy_returns_html(client: TestClient) -> None:
    response = client.get("/privacy")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


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


# ---------------------------------------------------------------------------
# Fix 1: error leakage — budget/spent values must NOT appear in HTTP responses
# ---------------------------------------------------------------------------


def test_budget_exceeded_returns_429_with_generic_message(client: TestClient) -> None:
    """BudgetExceededError must be surfaced as 429 with a generic message.

    The original detail=str(e) leaked "spent=4.98 budget=5.00" internals.
    After the fix the detail must be a generic Czech string with no financial
    figures or 'budget'/'spent' tokens.
    """
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = BudgetExceededError(
            "spent=4.98 budget=5.00 exceeded"
        )
        response = _upload(client)

    assert response.status_code == 429
    detail = response.json()["detail"]
    # Must NOT leak budget figures
    assert "spent" not in detail.lower(), f"Leaks 'spent' in detail: {detail!r}"
    assert "budget" not in detail.lower(), f"Leaks 'budget' in detail: {detail!r}"
    assert "4.98" not in detail, f"Leaks numeric figure in detail: {detail!r}"
    # Must be a non-empty user-facing message
    assert len(detail) > 10


def test_generic_exception_returns_500_without_internals(client: TestClient) -> None:
    """Unexpected pipeline exceptions must return 500 with a safe generic message.

    The original detail=f"Pipeline failed: {e}" leaked exception text.
    """
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = RuntimeError(
            "db connection string: postgres://user:secret@host/db"
        )
        response = _upload(client)

    assert response.status_code == 500
    detail = response.json()["detail"]
    # Must NOT echo the original exception message
    assert "postgres" not in detail.lower(), f"Leaks exception in detail: {detail!r}"
    assert "secret" not in detail, f"Leaks secret in detail: {detail!r}"
    assert "Pipeline failed" not in detail, f"Old error format leaked: {detail!r}"
    assert len(detail) > 10


def test_infrastructure_error_returns_502_without_internals(client: TestClient) -> None:
    """PipelineInfrastructureError detail must NOT contain prompt paths or LLM details."""
    from src.pipeline import PipelineInfrastructureError

    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = PipelineInfrastructureError(
            "OpenAI outage — prompts/cv_parser_system.txt missing"
        )
        response = _upload(client)

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "prompts/" not in detail, f"Leaks file path in detail: {detail!r}"
    assert "OpenAI outage" not in detail, f"Leaks internal message: {detail!r}"
    assert len(detail) > 10


# ---------------------------------------------------------------------------
# Fix 2: temp-file cleanup when cv.read() raises mid-stream
# ---------------------------------------------------------------------------


def test_tempfile_cleaned_up_when_pipeline_raises(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Pipeline() constructor raises, the uploaded temp file must NOT leak."""
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
    assert leaked_paths, "tempfile fixture not exercised"
    for path in leaked_paths:
        assert not path.exists(), f"Tempfile leaked on Pipeline init failure: {path}"


# ---------------------------------------------------------------------------
# Fix 3: scanned/corrupt PDF → 422 not 500
# ---------------------------------------------------------------------------


def test_scanned_pdf_returns_422(client: TestClient) -> None:
    """A ValueError('Scanned PDF...') from the pipeline must map to 422, not 500."""
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = ValueError(
            "Scanned PDF detected. OCR is not supported in MVP."
        )
        response = _upload(client)

    assert response.status_code == 422
    detail = response.json()["detail"]
    # Must contain a helpful hint about text-based PDF
    assert "sken" in detail.lower() or "text" in detail.lower(), (
        f"Expected hint about scanned PDF in detail: {detail!r}"
    )


def test_page_limit_exceeded_returns_422(client: TestClient) -> None:
    """A ValueError about page limit must map to 422 with a friendly message."""
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.side_effect = ValueError(
            "PDF has 45 pages which exceeds the 30-page limit."
        )
        response = _upload(client)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "stran" in detail.lower() or "pages" in detail.lower(), (
        f"Expected page-limit hint in detail: {detail!r}"
    )


# ---------------------------------------------------------------------------
# Fix 4: warnings surface in /analyze response
# ---------------------------------------------------------------------------


def test_warnings_present_in_analyze_response_for_low_confidence(client: TestClient) -> None:
    """A low-confidence result must include at least one warning in the response."""
    from src.schemas import ValidationFlag

    result = _make_pipeline_result()
    # Inject a hallucination warning flag
    flag = MagicMock(spec=ValidationFlag)
    flag.severity = "warning"
    result.validation_flags = [flag]
    result.score = MagicMock(confidence=0.4)

    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.return_value = result
        response = _upload(client)

    assert response.status_code == 200
    body = response.json()
    assert "warnings" in body, "Response must contain 'warnings' key"
    assert isinstance(body["warnings"], list)
    assert len(body["warnings"]) > 0, "Expected at least one warning for low-confidence result"


def test_warnings_empty_for_nominal_result(client: TestClient) -> None:
    """A clean, high-confidence result must have an empty warnings list."""
    result = _make_pipeline_result()
    result.validation_flags = []
    result.score = MagicMock(confidence=0.95)

    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.return_value = result
        response = _upload(client)

    assert response.status_code == 200
    body = response.json()
    assert "warnings" in body
    assert body["warnings"] == [], f"Expected empty warnings for nominal result: {body['warnings']}"


def test_warnings_salary_family_fallback(client: TestClient) -> None:
    """A family_fallback ISCO match must produce a salary approximation warning."""
    result = _make_pipeline_result(
        salary_data={
            "isco_match_level": "family_fallback",
            "confidence": "medium",
        }
    )

    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.return_value = result
        response = _upload(client)

    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert any("mzda" in w.lower() or "přibližně" in w.lower() for w in warnings), (
        f"Expected salary approximation warning, got: {warnings}"
    )


# ---------------------------------------------------------------------------
# Existing status-code contract tests (unchanged behaviour)
# ---------------------------------------------------------------------------


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


def test_rate_limit_returns_429(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A client over its per-IP quota gets 429 with a Retry-After header."""
    from src.rate_limit import SlidingWindowRateLimiter

    monkeypatch.setattr(
        api, "rate_limiter", SlidingWindowRateLimiter(max_requests=1, window_seconds=3600)
    )
    with patch.object(api, "Pipeline") as mock_pipeline_cls:
        mock_pipeline_cls.return_value.run.return_value = _make_pipeline_result()
        first = _upload(client)
        second = _upload(client)

    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1


def test_oversized_upload_returns_413(client: TestClient) -> None:
    """Uploads exceeding MAX_UPLOAD_BYTES must be rejected with 413."""
    big_payload = b"%PDF-1.4 " + b"x" * (api.MAX_UPLOAD_BYTES + 1024)
    response = client.post(
        "/analyze",
        files={"cv": ("big.pdf", io.BytesIO(big_payload), "application/pdf")},
    )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()
