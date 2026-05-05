"""End-to-end CV analysis pipeline orchestrator.

Chains: extract -> parse -> validate -> score -> estimate -> explain.
Step-level try/except + logger.exception() + meta dict aggregation per step.
Does NOT implement retry or caching — those belong in Explainer (MAX_RETRIES=3) and LLMProvider.
Does NOT modify schemas.py, prompts, or internal logic of other modules.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import settings
from src.cost_tracker import (
    CostRecord,
    check_budget,
    new_run_id,
    record_run,
)
from src.estimator import SalaryEstimator
from src.explainer import Explainer
from src.extractor import UnsupportedFormatError, extract_text
from src.llm_provider import get_provider
from src.logging_config import bind_run_id, configure_logging
from src.parser import ResumeParser
from src.salary_ispv import ISPVLoader, ISPVLookupError
from src.schemas import PipelineResult, Resume, SeniorityScore
from src.scorer import score_resume
from src.validator import EmbeddingValidator, ResumeValidator, compute_confidence

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end CV analysis pipeline.

    Initializes all components and chains them in sequence.
    Each step has its own try/except to allow partial results and per-step timing.
    """

    def __init__(
        self,
        llm_provider_name: str = "openai",
        parser_model: str | None = None,
        explainer_model: str | None = None,
        enable_embedding_validator: bool = False,
    ) -> None:
        # Phase 22 fix: when caller doesn't pass explicit models, read from Settings
        # so make eval / scripts / Streamlit all pick up upgraded defaults
        # (e.g., explainer_model=gpt-5-mini). Explicit None still possible via
        # passing settings.parser_model directly.
        effective_parser_model = parser_model or settings.parser_model
        effective_explainer_model = explainer_model or settings.explainer_model
        logger.info(
            "Initializing pipeline — provider=%s parser_model=%s explainer_model=%s embedding_validator=%s",
            llm_provider_name,
            effective_parser_model,
            effective_explainer_model,
            enable_embedding_validator,
        )
        # Per-stage providers allow different models for parsing vs explanation.
        # Only "openai" is supported as of 2026-05-05 (see ADR morning update).
        parser_llm = get_provider(llm_provider_name, model=effective_parser_model)
        explainer_llm = get_provider(llm_provider_name, model=effective_explainer_model)
        self._parser_llm = parser_llm
        self._explainer_llm = explainer_llm
        self.parser = ResumeParser(parser_llm)
        self.validator = ResumeValidator()
        self.explainer = Explainer(explainer_llm)

        # Wire ISPV salary data source — loads from data/ispv_2025.xlsx if present.
        # Download the file via the /admin page when missing.
        ispv_path = Path("data/ispv_2025.xlsx")
        inflation_path = Path("data/inflation_factors.json")
        inflation_factors: dict[str, float] = {}
        if inflation_path.exists():
            raw = json.loads(inflation_path.read_text(encoding="utf-8"))
            # Strip comment/metadata keys (convention: keys starting with "_")
            inflation_factors = {k: float(v) for k, v in raw.items() if not k.startswith("_")}

        ispv_loader: ISPVLoader | None = None
        if ispv_path.exists():
            ispv_loader = ISPVLoader(xlsx_path=ispv_path, inflation_factors=inflation_factors)
            try:
                ispv_loader.load()
            except Exception:
                logger.exception("ISPV load failed — salary estimation will be unavailable")
                ispv_loader = None
        else:
            logger.warning("ISPV file not found at %s — download via /admin page", ispv_path)

        self.estimator = SalaryEstimator(ispv_loader=ispv_loader)
        # Layer-2 embedding validator — opt-in to avoid embedding cost on every run.
        # When enabled, uses the parser provider (same model config) for embeddings.
        self._embedding_validator: EmbeddingValidator | None = (
            EmbeddingValidator(parser_llm) if enable_embedding_validator else None
        )

    def run(self, file_path: Path) -> PipelineResult:
        """Run the full CV analysis pipeline.

        Args:
            file_path: Path to the CV file (PDF or DOCX).

        Returns:
            PipelineResult with all analysis components and aggregated meta.

        Raises:
            BudgetExceededError: If today's API spend would exceed DAILY_API_BUDGET_USD.
            ValueError: If the CV is a scanned PDF (OCR not supported in MVP).
            UnsupportedFormatError: If the file format is not PDF or DOCX.
        """
        run_id = new_run_id()
        configure_logging()
        bind_run_id(run_id)

        # Pre-flight budget check — rough estimate per CV to catch runaway spend early.
        check_budget(estimated_cost_usd=0.05)

        wall_start = time.monotonic()
        meta: dict[str, Any] = {"steps": {}, "run_id": run_id}
        run_error: str | None = None

        # ── Step 1: Extract ──────────────────────────────────────────────────
        logger.info("Pipeline step 1/6: extract — %s", file_path.name)
        step_start = time.monotonic()
        try:
            doc = extract_text(file_path)
            meta["steps"]["extract"] = {
                "status": "ok",
                "method": doc.extraction_method,
                "chars": doc.char_count,
                "is_scanned": doc.is_scanned,
                "duration_s": time.monotonic() - step_start,
            }
        except UnsupportedFormatError:
            logger.exception("Unsupported file format: %s", file_path.suffix)
            meta["steps"]["extract"] = {
                "status": "error",
                "duration_s": time.monotonic() - step_start,
            }
            raise

        if doc.is_scanned:
            raise ValueError(
                "Scanned PDF detected. OCR is not supported in MVP. "
                "Please provide a text-based (digital) PDF or a DOCX file."
            )

        # ── Step 2: Parse ────────────────────────────────────────────────────
        logger.info("Pipeline step 2/6: parse")
        step_start = time.monotonic()
        try:
            resume, parse_meta = self.parser.parse(doc.text)
            meta["steps"]["parse"] = {
                "status": "ok",
                "cost_usd": parse_meta.get("cost_usd", 0.0),
                "duration_s": time.monotonic() - step_start,
                "truncated": parse_meta.get("truncated", False),
            }
        except Exception:
            logger.exception("Parse step failed")
            resume = Resume(raw_text_length=len(doc.text))
            parse_meta = {"cost_usd": 0.0}
            meta["steps"]["parse"] = {
                "status": "error",
                "duration_s": time.monotonic() - step_start,
            }

        # ── Step 3: Validate ─────────────────────────────────────────────────
        logger.info("Pipeline step 3/6: validate")
        step_start = time.monotonic()
        embed_cost_usd = 0.0
        try:
            flags = self.validator.validate(resume, doc.text)
            # Layer-2: embedding-based upgrade (opt-in, default off).
            # Runs only when enable_embedding_validator=True was passed to __init__.
            if self._embedding_validator is not None and any(
                f.severity in ("error", "warning") for f in flags
            ):
                logger.info("Pipeline step 3/6: validate — running Layer-2 embedding upgrade")
                flags, embed_meta = self._embedding_validator.upgrade_flags(flags, doc.text)
                embed_cost_usd = float(embed_meta.get("embed_cost_usd", 0.0))
            meta["steps"]["validate"] = {
                "status": "ok",
                "flags": len(flags),
                "errors": sum(1 for f in flags if f.severity == "error"),
                "warnings": sum(1 for f in flags if f.severity == "warning"),
                "embed_cost_usd": embed_cost_usd,
                "duration_s": time.monotonic() - step_start,
            }
        except Exception:
            logger.exception("Validate step failed")
            flags = []
            meta["steps"]["validate"] = {
                "status": "error",
                "duration_s": time.monotonic() - step_start,
            }

        # ── Step 4: Score ────────────────────────────────────────────────────
        logger.info("Pipeline step 4/6: score")
        step_start = time.monotonic()
        try:
            score: SeniorityScore = score_resume(resume)
            # Reduce confidence by hallucination guard result
            confidence_reduction = compute_confidence(flags)
            score = score.model_copy(
                update={"confidence": round(score.confidence * confidence_reduction, 3)}
            )
            meta["steps"]["score"] = {
                "status": "ok",
                "total": score.total,
                "confidence": score.confidence,
                "duration_s": time.monotonic() - step_start,
            }
        except Exception:
            logger.exception("Score step failed")
            from src.schemas import ScoreComponent

            score = SeniorityScore(
                total=0.0,
                components=[
                    ScoreComponent(name="Error", score=0.0, weight=1.0, reasoning="Scoring failed")
                ],
                confidence=0.0,
            )
            meta["steps"]["score"] = {
                "status": "error",
                "duration_s": time.monotonic() - step_start,
            }

        # ── Step 5: Estimate salary ──────────────────────────────────────────
        # ISPVLookupError is a hard error — no silent fallback per project decision Q4 2026.
        logger.info("Pipeline step 5/6: estimate salary")
        step_start = time.monotonic()
        try:
            salary = self.estimator.estimate(resume, score)
            # Surface salary source label and resolved SalaryData for UI display.
            salary_source = self.estimator.last_salary_source
            salary_data_obj = self.estimator.last_salary_data
            meta["salary_source"] = salary_source
            meta["salary_data"] = (
                salary_data_obj.model_dump() if salary_data_obj is not None else None
            )
            meta["steps"]["estimate"] = {
                "status": "ok",
                "low": salary.low,
                "mid": salary.mid,
                "high": salary.high,
                "salary_source": salary_source,
                "duration_s": time.monotonic() - step_start,
            }
        except ISPVLookupError:
            # Hard fail — propagate to caller so UI can surface a clear error message.
            logger.exception("Salary estimate failed — ISPV data unavailable or no ISCO code")
            raise

        # ── Step 6: Explain ──────────────────────────────────────────────────
        logger.info("Pipeline step 6/6: explain")
        step_start = time.monotonic()
        try:
            explanation, exp_meta = self.explainer.explain(resume, score, salary)
            meta["steps"]["explain"] = {
                "status": "ok",
                "cost_usd": exp_meta.get("cost_usd", 0.0),
                "retries": exp_meta.get("retries", 0),
                "warning": exp_meta.get("warning"),
                "duration_s": time.monotonic() - step_start,
            }
        except Exception:
            logger.exception("Explain step failed")
            from src.explainer import _fallback_recommendation  # type: ignore[reportPrivateUsage]
            from src.schemas import Explanation

            explanation = Explanation(
                summary="Explanation unavailable due to error.",
                strengths=["Analysis failed"],
                gaps=["Analysis failed"],
                recommendations=[_fallback_recommendation(i) for i in range(3)],
            )
            exp_meta = {"cost_usd": 0.0}
            meta["steps"]["explain"] = {
                "status": "error",
                "duration_s": time.monotonic() - step_start,
            }

        # ── Aggregate ────────────────────────────────────────────────────────
        parse_cost = float(meta["steps"].get("parse", {}).get("cost_usd", 0.0))
        explain_cost = float(meta["steps"].get("explain", {}).get("cost_usd", 0.0))
        # embed_cost_usd is set in the validate block (0.0 when embedding validator is off)
        embed_cost = float(meta["steps"].get("validate", {}).get("embed_cost_usd", 0.0))
        total_cost = parse_cost + explain_cost + embed_cost
        meta["total_cost_usd"] = total_cost
        meta["total_duration_s"] = round(time.monotonic() - wall_start, 2)
        meta["parser_model"] = getattr(self._parser_llm, "model", "unknown")
        meta["explainer_model"] = getattr(self._explainer_llm, "model", "unknown")
        meta["model"] = meta["parser_model"]  # backward-compat alias for UI footer

        logger.info(
            "Pipeline complete — duration=%.2fs cost=$%.4f score=%.1f",
            meta["total_duration_s"],
            meta["total_cost_usd"],
            score.total,
        )

        # Derive run_error from per-step status so the metrics CLI sees real failures.
        # Initial run_error is None; we promote any step's "error" status to the
        # top-level marker (last erroring step wins — they tend to cascade anyway).
        for step_name, step_meta in meta["steps"].items():
            if isinstance(step_meta, dict) and step_meta.get("status") == "error":
                run_error = f"{step_name}_failed"

        # Persist cost record for budget tracking and metrics CLI
        cost_record = CostRecord(
            run_id=run_id,
            timestamp_utc=datetime.now(UTC).isoformat(),
            fixture_or_file=str(file_path),
            total_cost_usd=total_cost,
            parse_cost_usd=parse_cost,
            explain_cost_usd=explain_cost,
            embed_cost_usd=embed_cost,
            duration_s=meta["total_duration_s"],
            score_total=score.total,
            error=run_error,
        )
        record_run(cost_record)

        return PipelineResult(
            resume=resume,
            validation_flags=flags,
            score=score,
            salary=salary,
            explanation=explanation,
            meta=meta,
        )
