"""FastAPI HTTP service for the Job Fit Estimator.

Exposes a single endpoint that accepts a CV upload and returns the analysis
as JSON. Hard-fails for non-CZ locations (HTTP 422) and missing ISPV data
(HTTP 503) to surface the same constraints as the Streamlit UI and CLI.

Usage:
    uvicorn api:app --reload
    curl -F "cv=@path/to/cv.pdf" http://localhost:8000/analyze
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from src.config import settings
from src.cost_tracker import BudgetExceededError
from src.logging_config import configure_logging
from src.pipeline import Pipeline, PipelineInfrastructureError
from src.rate_limit import RateLimitExceeded, SlidingWindowRateLimiter
from src.salary_ispv import (
    ISPVDataMissingError,
    ISPVLookupError,
    MissingISCOError,
    NonCZLocationError,
)

logger = logging.getLogger(__name__)

# Hard cap on upload size — CVs are typically < 1 MB; 10 MB leaves headroom
# for image-heavy designer portfolios while bounding memory/disk impact.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks
_WEB_INDEX = Path(__file__).parent / "web" / "index.html"
_WEB_PRIVACY = Path(__file__).parent / "web" / "privacy.html"

# Per-IP rate limiter (process-local). Complements the per-day budget guard in
# src.cost_tracker: the budget caps total daily spend, this caps how fast one
# client can consume it. Rebuilt from settings at import time.
rate_limiter = SlidingWindowRateLimiter(
    max_requests=settings.rate_limit_max_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def _client_ip(request: Request) -> str:
    """Best-effort client IP, trusting the left-most X-Forwarded-For hop.

    The service runs behind a single trusted nginx proxy that sets
    X-Forwarded-For, so the first entry is the real client. Without a trusted
    proxy this header is spoofable; revisit if the topology changes.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


def _collect_warnings(result: Any) -> list[str]:
    """Collect user-visible Czech warning strings from a PipelineResult.

    Surfaces two signal categories that would otherwise be hidden in ``meta``:

    1. Validation flags (hallucination guard) — ``warning`` or ``error`` severity
       indicates the LLM may have invented data not found in the source text.
    2. Salary confidence / ISCO match-level signals buried in ``meta.salary_data``
       and ``meta.salary_source`` — a family-level fallback or "low" confidence
       means the salary estimate is a rough approximation.

    Returns an empty list when all signals look nominal.
    """
    warnings: list[str] = []

    # Validation flags — check for hallucination guard hits
    flags = getattr(result, "validation_flags", None) or []
    has_error = any(f.severity == "error" for f in flags)
    has_warning = any(f.severity == "warning" for f in flags)
    if has_error or has_warning:
        warnings.append("Nižší spolehlivost analýzy — v CV chyběly některé ověřitelné údaje.")

    # Score confidence — low overall confidence from the scorer
    score = getattr(result, "score", None)
    if score is not None and getattr(score, "confidence", 1.0) < 0.5:
        warnings.append("Nižší spolehlivost skóre — v CV chyběly některé klíčové informace.")

    # Salary confidence / ISCO match precision — read from meta
    meta = getattr(result, "meta", {}) or {}
    salary_data = meta.get("salary_data")
    if salary_data:
        match_level = salary_data.get("isco_match_level", "")
        confidence = salary_data.get("confidence", "high")
        if match_level in ("2digit", "1digit", "family_fallback"):
            warnings.append(
                "Mzda je odhadnuta jen přibližně — profese nebyla v datech ISPV nalezena přesně."
            )
        elif confidence == "low":
            warnings.append(
                "Nižší spolehlivost odhadu mzdy — data ISPV pro tuto profesi jsou omezená."
            )

    return warnings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """One-time startup checks and logging configuration.

    configure_logging() is called here (not per-request) so structlog is
    initialised exactly once. Pipeline carries per-run state inside its
    SalaryEstimator (last_salary_* fields), so a single shared instance behind
    concurrent /analyze requests would leak that state across responses. The
    request handler builds a fresh Pipeline per call and offloads .run() to a
    thread, which keeps the event loop responsive without sharing mutable state.
    """
    # Initialise logging before any other startup log messages.
    configure_logging()

    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set; /analyze will return 503 until configured")

    # Fail loudly at startup rather than producing a 500 per request when
    # web/index.html is missing (common after a partial deploy or bad volume mount).
    if not _WEB_INDEX.exists():
        raise RuntimeError(
            f"web/index.html not found at {_WEB_INDEX}. "
            "Ensure the web/ directory is present before starting the server."
        )

    yield


app = FastAPI(
    title="Job Fit & Salary Estimator API",
    description="Analyzes a CV and returns seniority score, salary estimate, and recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)


def _build_pipeline() -> Pipeline:
    """Construct a fresh Pipeline for one request (see lifespan note)."""
    return Pipeline(
        "openai",
        parser_model=settings.parser_model,
        explainer_model=settings.explainer_model,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page web frontend."""
    return HTMLResponse(content=_WEB_INDEX.read_text(encoding="utf-8"))


@app.get("/privacy", response_class=HTMLResponse)
def privacy() -> HTMLResponse:
    """Serve the GDPR privacy policy page."""
    return HTMLResponse(content=_WEB_PRIVACY.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request, cv: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    """Analyze a CV (PDF or DOCX) and return the structured pipeline result."""
    # Rate-limit first: reject abusive clients before touching the upload body
    # or the (cost-bearing) pipeline.
    try:
        rate_limiter.check(_client_ip(request))
    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(e.retry_after_seconds)},
        ) from e

    filename = cv.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}; expected .pdf or .docx",
        )
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    # Stream the upload to a temp file in 64 KB chunks and run the pipeline.
    # The entire block — file creation, streaming write, and pipeline call — is
    # wrapped in a single try/finally so tmp_path is ALWAYS unlinked, even when
    # cv.read() raises mid-stream or Pipeline() raises before run() is reached.
    tmp_path: Path | None = None
    try:
        bytes_written = 0
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await cv.read(_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                tmp.write(chunk)

        # Pipeline.run is fully synchronous (PDF parsing, sync OpenAI client,
        # openpyxl). Running it directly on the event-loop coroutine would block
        # every concurrent request on this worker. asyncio.to_thread offloads it
        # to the default thread pool so the loop stays responsive.
        pipeline = _build_pipeline()
        result = await asyncio.to_thread(pipeline.run, tmp_path)

    except HTTPException:
        # Re-raise 413/415/429/503 unchanged — these are already user-safe.
        raise
    except BudgetExceededError as e:
        # Operational state — surface as 429 so callers can back off.
        # Do NOT include str(e): it leaks "spent=X.XX budget=Y.YY" internals.
        logger.warning("Daily budget exceeded — request rejected")
        raise HTTPException(
            status_code=429,
            detail="Služba je teď přetížená, zkuste to prosím za chvíli.",
        ) from e
    except PipelineInfrastructureError as e:
        # Server-side dependency failed (LLM provider, prompt, etc.) — 502 so
        # operator monitoring picks this up instead of blaming the CV.
        # str(e) can leak prompt file paths, so we log and return a generic message.
        logger.exception("pipeline infrastructure failure")
        raise HTTPException(
            status_code=502,
            detail="Dočasný problém na straně AI poskytovatele.",
        ) from e
    except ISPVDataMissingError as e:
        # Service is not ready — operator must download the ISPV dataset.
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (NonCZLocationError, MissingISCOError) as e:
        # Caller-side problem with the submitted CV — refuse with a 422.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ISPVLookupError as e:
        # Other salary-lookup failures.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        # Scanned/corrupt PDF and page-limit violations raise ValueError in the
        # extractor. These are caller-side problems (bad input), so 422 with a
        # friendly message. Internal detail is logged for diagnostics.
        msg = str(e)
        logger.info("CV rejected (ValueError): %s", msg)
        # Map the two known ValueError messages to friendly Czech strings.
        # Fall back to a generic message for any unexpected ValueError.
        if "Scanned PDF" in msg or "scanned" in msg.lower() or "is_scanned" in msg.lower():
            user_msg = (
                "CV vypadá jako sken/obrázek bez textu — "
                "nahraj prosím PDF nebo DOCX s vybíratelným textem."
            )
        elif "pages" in msg.lower() and "limit" in msg.lower():
            user_msg = (
                "Nahraný soubor má příliš mnoho stránek. "
                "Nahraj prosím pouze své CV (max. 30 stran)."
            )
        else:
            user_msg = "Soubor se nepodařilo zpracovat. Zkontroluj formát a zkus to znovu."
        raise HTTPException(status_code=422, detail=user_msg) from e
    except Exception as e:
        # Unexpected errors — log the real exception, return a safe generic message.
        logger.exception("pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Při analýze nastala chyba. Zkuste to prosím znovu.",
        ) from e
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)

    payload = result.model_dump(mode="json")
    payload["warnings"] = _collect_warnings(result)
    return payload
