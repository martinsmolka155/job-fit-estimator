"""FastAPI HTTP service for the Job Fit Estimator.

Exposes a single endpoint that accepts a CV upload and returns the analysis
as JSON. Hard-fails for non-CZ locations (HTTP 422) and missing ISPV data
(HTTP 503) to surface the same constraints as the Streamlit UI and CLI.

Usage:
    uvicorn api:app --reload
    curl -F "cv=@path/to/cv.pdf" http://localhost:8000/analyze
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import settings
from src.pipeline import Pipeline
from src.salary_ispv import (
    ISPVDataMissingError,
    ISPVLookupError,
    MissingISCOError,
    NonCZLocationError,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Job Fit & Salary Estimator API",
    description="Analyzes a CV and returns seniority score, salary estimate, and recommendations.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(cv: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    """Analyze a CV (PDF or DOCX) and return the structured pipeline result."""
    filename = cv.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in (".pdf", ".docx"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}; expected .pdf or .docx",
        )
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    body = await cv.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        tmp_path = Path(tmp.name)

    pipeline = Pipeline(
        "openai",
        parser_model=settings.parser_model,
        explainer_model=settings.explainer_model,
    )

    try:
        result = pipeline.run(tmp_path)
    except ISPVDataMissingError as e:
        # Service is not ready — operator must download the ISPV dataset.
        raise HTTPException(status_code=503, detail=str(e)) from e
    except (NonCZLocationError, MissingISCOError) as e:
        # Caller-side problem with the submitted CV — refuse with a 422.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ISPVLookupError as e:
        # Other salary-lookup failures.
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("pipeline failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    return result.model_dump(mode="json")
