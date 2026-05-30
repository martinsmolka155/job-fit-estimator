"""Labeled-eval / accuracy harness for the Job Fit Estimator pipeline.

Loads ground-truth labels from data/eval_labels.json, runs the pipeline on each
labeled CV (both .pdf and .txt), computes accuracy metrics via src/eval_metrics.py,
and prints a rich report. Optionally exports JSON / HTML.

Usage:
  uv run python scripts/eval.py                          # all labeled CVs
  uv run python scripts/eval.py --limit 2               # first 2 entries (cost guard)
  uv run python scripts/eval.py --only nurse            # entries whose cv path contains 'nurse'
  uv run python scripts/eval.py --json out.json         # export predictions + metrics as JSON
  uv run python scripts/eval.py --html out.html         # HTML report
  uv run python scripts/eval.py --limit 2 --html r.html # combined

NOTE: This script calls the OpenAI API. It is intentionally NOT run in the normal
pytest suite — call it manually or as a nightly job.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

# Add project root to sys.path so relative imports work when called as a script.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.cost_tracker import check_budget  # noqa: E402
from src.eval_metrics import (  # noqa: E402
    EvalResults,
    Label,
    Prediction,
    SalaryLabel,
    SalaryPrediction,
    aggregate_metrics,
    format_czk,
    format_pct,
)
from src.pipeline import Pipeline  # noqa: E402
from src.schemas import PipelineResult  # noqa: E402

logger = logging.getLogger(__name__)

_LABELS_PATH = _PROJECT_ROOT / "data" / "eval_labels.json"
_FIXTURES_BASE = _PROJECT_ROOT

# Score sentinel used when the pipeline fails and we still want a row in the table.
_FAILED_SENTINEL = "ERROR"


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def _load_labels(path: Path) -> list[Label]:
    """Load and parse eval_labels.json into Label objects.

    Args:
        path: Path to eval_labels.json.

    Returns:
        List of Label objects, one per entry.

    Raises:
        FileNotFoundError: If the labels file does not exist.
        ValueError: If the JSON is malformed.
    """
    raw_list: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    labels: list[Label] = []
    for entry in raw_list:
        isco = str(entry.get("isco_code", "TODO"))
        seniority = str(entry.get("seniority", "TODO"))
        salary_raw = entry.get("salary_czk_month")
        salary: SalaryLabel | None = None
        if salary_raw is not None:
            salary = SalaryLabel.from_raw(salary_raw)

        salary_source = str(entry.get("salary_source", ""))
        is_real = isco == "TODO" or seniority == "TODO" or "TODO_real" in salary_source

        labels.append(
            Label(
                cv_name=Path(str(entry["cv"])).name,
                isco_code=isco,
                seniority=seniority,
                salary=salary,
                is_real=is_real,
            )
        )
    return labels


# ---------------------------------------------------------------------------
# Running the pipeline on a single CV
# ---------------------------------------------------------------------------


def _run_cv(
    pipeline: Pipeline,
    cv_path: Path,
    label: Label,
    console: Console,
) -> tuple[PipelineResult | None, Prediction]:
    """Run the pipeline on one CV file, handling .txt bypass.

    For .txt files the extractor does not support direct text input, so we read
    the file and feed its content directly to the parser, bypassing PDF extraction.

    Args:
        pipeline: Initialized Pipeline instance.
        cv_path: Absolute path to the CV file.
        label: Ground-truth label for this CV (used only for display context).
        console: Rich console for status messages.

    Returns:
        Tuple of (PipelineResult or None on failure, Prediction).
    """
    console.print(f"  Running: [bold]{cv_path.name}[/bold]", end="")

    try:
        if cv_path.suffix.lower() == ".txt":
            # .txt files: read text directly, inject into the parse step.
            result = _run_txt(pipeline, cv_path)
        else:
            result = pipeline.run(cv_path)

        salary_pred: SalaryPrediction | None = None
        if result.salary:
            salary_pred = SalaryPrediction(
                low=result.salary.low,
                high=result.salary.high,
                mid=result.salary.mid,
            )

        # Extract ISCO from the most recent experience that has one.
        isco_pred: str | None = None
        for exp in result.resume.experiences:
            if exp.isco_code:
                isco_pred = exp.isco_code
                break

        # Derive predicted seniority from the score band.
        from src.estimator import _score_to_seniority  # type: ignore[reportPrivateUsage]

        seniority_pred = _score_to_seniority(result.score.total)  # type: ignore[arg-type]

        w_count = sum(1 for f in result.validation_flags if f.severity in ("warning", "error"))

        pred = Prediction(
            cv_name=cv_path.name,
            isco_code=isco_pred,
            seniority=seniority_pred,  # type: ignore[arg-type]
            salary=salary_pred,
            warnings_emitted=w_count,
        )
        console.print(" [green]OK[/green]")
        return result, pred

    except Exception as exc:
        console.print(f" [red]FAIL: {exc}[/red]")
        logger.exception("Pipeline failed for %s", cv_path.name)
        pred = Prediction(
            cv_name=cv_path.name,
            isco_code=None,
            seniority=None,
            salary=None,
        )
        return None, pred


def _run_txt(pipeline: Pipeline, cv_path: Path) -> PipelineResult:
    """Run pipeline on a .txt CV by bypassing PDF extraction.

    Reads the plain text directly and feeds it through the parser.
    This avoids the UnsupportedFormatError that extract_text() raises for .txt.

    Args:
        pipeline: Initialized Pipeline instance.
        cv_path: Path to the .txt CV file.

    Returns:
        PipelineResult as if the text had come from a PDF.

    Raises:
        Any exception that the pipeline internals raise (parse, score, etc.).
    """
    import time as _time
    from datetime import UTC
    from datetime import datetime as _datetime

    from src.cost_tracker import CostRecord, new_run_id, record_run
    from src.extractor import ExtractedDocument
    from src.logging_config import bind_run_id
    from src.schemas import PipelineResult

    run_id = new_run_id()
    bind_run_id(run_id)

    # Budget guard — same reservation logic as Pipeline.run()
    from src.llm_provider import estimate_run_cost_usd

    reservation = estimate_run_cost_usd(
        parser_model=getattr(pipeline._parser_llm, "model", "gpt-4o-mini"),
        explainer_model=getattr(pipeline._explainer_llm, "model", "gpt-4o-mini"),
    )
    check_budget(estimated_cost_usd=reservation)

    raw_text = cv_path.read_text(encoding="utf-8")
    doc = ExtractedDocument(
        text=raw_text,
        source_format="pdf",  # sentinel — treated as digital text
        extraction_method="pymupdf",
        page_count=1,
        char_count=len(raw_text),
        is_scanned=False,
    )

    wall_start = _time.monotonic()
    meta: dict[str, Any] = {"steps": {}, "run_id": run_id}

    # ── Parse ────────────────────────────────────────────────────────────────
    resume, parse_meta = pipeline.parser.parse(doc.text)
    if parse_meta.get("error"):
        raise RuntimeError(f"Parser failed: {parse_meta['error']}")
    meta["steps"]["parse"] = {"status": "ok", "cost_usd": parse_meta.get("cost_usd", 0.0)}

    # ── Validate ─────────────────────────────────────────────────────────────
    from src.validator import compute_confidence

    flags = pipeline.validator.validate(resume, doc.text)
    meta["steps"]["validate"] = {"status": "ok", "flags": len(flags), "embed_cost_usd": 0.0}

    # ── Score ─────────────────────────────────────────────────────────────────
    from src.scorer import score_resume

    score = score_resume(resume)
    confidence_reduction = compute_confidence(flags)
    score = score.model_copy(
        update={"confidence": round(score.confidence * confidence_reduction, 3)}
    )
    meta["steps"]["score"] = {"status": "ok", "total": score.total}

    # ── Estimate ─────────────────────────────────────────────────────────────
    salary = pipeline.estimator.estimate(resume, score)
    meta["steps"]["estimate"] = {"status": "ok"}

    # ── Explain ──────────────────────────────────────────────────────────────
    explanation, exp_meta = pipeline.explainer.explain(resume, score, salary)
    meta["steps"]["explain"] = {"status": "ok", "cost_usd": exp_meta.get("cost_usd", 0.0)}

    # ── Aggregate ────────────────────────────────────────────────────────────
    parse_cost = float(meta["steps"]["parse"]["cost_usd"])
    explain_cost = float(meta["steps"]["explain"]["cost_usd"])
    total_cost = parse_cost + explain_cost
    meta["total_cost_usd"] = total_cost
    meta["total_duration_s"] = round(_time.monotonic() - wall_start, 2)
    meta["parser_model"] = getattr(pipeline._parser_llm, "model", "unknown")
    meta["explainer_model"] = getattr(pipeline._explainer_llm, "model", "unknown")
    meta["model"] = meta["parser_model"]

    cost_record = CostRecord(
        run_id=run_id,
        timestamp_utc=_datetime.now(UTC).isoformat(),
        fixture_or_file=f"eval_txt{cv_path.suffix}",
        total_cost_usd=total_cost,
        parse_cost_usd=parse_cost,
        explain_cost_usd=explain_cost,
        embed_cost_usd=0.0,
        duration_s=meta["total_duration_s"],
        score_total=score.total,
        error=None,
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


# ---------------------------------------------------------------------------
# Rich report helpers
# ---------------------------------------------------------------------------


def _print_per_cv_table(
    console: Console,
    run_data: list[tuple[Label, PipelineResult | None, Prediction]],
) -> None:
    """Print a per-CV table: predicted vs true ISCO/seniority/salary + pass/fail.

    Args:
        console: Rich console.
        run_data: List of (label, result_or_None, prediction) tuples.
    """
    table = Table(title="Per-CV Predictions vs Ground Truth", show_lines=True)
    table.add_column("CV", style="bold", max_width=22)
    table.add_column("True ISCO", justify="center")
    table.add_column("Pred ISCO", justify="center")
    table.add_column("ISCO", justify="center", width=5)
    table.add_column("True Sr.", justify="center")
    table.add_column("Pred Sr.", justify="center")
    table.add_column("Sr.", justify="center", width=5)
    table.add_column("True Salary", justify="right")
    table.add_column("Pred Salary", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Cost $", justify="right")

    for label, result, pred in run_data:
        if result is None:
            table.add_row(
                label.cv_name,
                label.isco_code,
                "[red]ERR[/red]",
                "[red]✗[/red]",
                label.seniority,
                "[red]ERR[/red]",
                "[red]✗[/red]",
                _fmt_salary_label(label),
                "-",
                "-",
                "-",
            )
            continue

        pred_isco = pred.isco_code or "?"
        true_isco = label.isco_code
        isco_ok = pred_isco == true_isco
        isco_p3 = len(pred_isco) >= 3 and len(true_isco) >= 3 and pred_isco[:3] == true_isco[:3]
        isco_mark = (
            "[green]✓[/green]" if isco_ok else ("[yellow]~[/yellow]" if isco_p3 else "[red]✗[/red]")
        )
        if label.is_real:
            isco_mark = "[dim]?[/dim]"
            pred_isco_display = pred_isco
            true_isco_display = f"[dim]{true_isco}[/dim]"
        else:
            pred_isco_display = (
                f"[red]{pred_isco}[/red]" if not isco_ok else f"[green]{pred_isco}[/green]"
            )
            true_isco_display = true_isco

        pred_sr = pred.seniority or "?"
        true_sr = label.seniority
        sr_ok = pred_sr == true_sr
        from src.eval_metrics import SENIORITY_ORDER

        pred_rank = SENIORITY_ORDER.get(pred_sr, -99)
        true_rank = SENIORITY_ORDER.get(true_sr, -99)
        sr_obo = abs(pred_rank - true_rank) <= 1
        if label.is_real:
            sr_mark = "[dim]?[/dim]"
        else:
            sr_mark = (
                "[green]✓[/green]"
                if sr_ok
                else ("[yellow]±1[/yellow]" if sr_obo else "[red]✗[/red]")
            )

        pred_salary_str = (
            f"{pred.salary.low:,}–{pred.salary.high:,}" if pred.salary else "[dim]n/a[/dim]"
        )
        score_str = f"{result.score.total:.1f}" if result else "-"
        cost_str = f"{result.meta.get('total_cost_usd', 0):.4f}" if result else "-"

        table.add_row(
            label.cv_name,
            true_isco_display,
            pred_isco_display,
            isco_mark,
            true_sr,
            pred_sr,
            sr_mark,
            _fmt_salary_label(label),
            pred_salary_str,
            score_str,
            cost_str,
        )

    console.print(table)


def _fmt_salary_label(label: Label) -> str:
    """Format the true salary label for display.

    Args:
        label: Ground-truth label.

    Returns:
        Formatted string, or 'TODO' / 'n/a' for incomplete labels.
    """
    if label.salary is None:
        return "[dim]n/a[/dim]"
    if label.salary.low == 0 and label.salary.high == 0:
        return "[dim]TODO[/dim]"
    if label.salary.low == label.salary.high:
        return f"{label.salary.low:,}"
    return f"{label.salary.low:,}–{label.salary.high:,}"


def _print_aggregate(console: Console, metrics: EvalResults, total_cost: float) -> None:
    """Print the aggregate metrics summary block.

    Args:
        console: Rich console.
        metrics: Computed EvalResults.
        total_cost: Total API cost in USD across all runs.
    """
    console.rule("[bold]Aggregate Metrics[/bold]")
    console.print(f"  Labeled CVs: {metrics.n_labeled}/{metrics.n_total} (excluding TODO entries)")
    console.print()

    if metrics.n_labeled > 0:
        console.print("[bold]ISCO accuracy[/bold]")
        console.print(f"  Exact match:        {format_pct(metrics.isco_exact_rate)}")
        console.print(f"  3-digit prefix:     {format_pct(metrics.isco_prefix3_rate)}")
        if metrics.isco_mismatches:
            console.print("  Mismatches:")
            for mm in metrics.isco_mismatches:
                prefix_note = " (prefix match)" if mm.prefix3_match else ""
                console.print(
                    f"    {mm.cv_name}: predicted={mm.predicted} true={mm.true}{prefix_note}"
                )
        console.print()

        console.print("[bold]Seniority accuracy[/bold]")
        console.print(f"  Exact match:        {format_pct(metrics.seniority_exact_rate)}")
        console.print(f"  Off-by-one (±1):    {format_pct(metrics.seniority_off_by_one_rate)}")
        console.print()

    if metrics.n_with_salary > 0:
        console.print("[bold]Salary accuracy[/bold]")
        console.print(f"  MAE:                {format_czk(metrics.salary_mae)}")
        console.print(f"  MAPE:               {format_pct(metrics.salary_mape)}")
        console.print(f"  Within ±15%:        {format_pct(metrics.salary_within_15pct_rate)}")
        console.print(f"  True in pred range: {format_pct(metrics.salary_range_coverage_rate)}")
        console.print()

    console.print("[bold]Warnings[/bold]")
    console.print(f"  Total flags:        {metrics.warnings_total}")
    console.print(f"  Avg per CV:         {metrics.warnings_per_cv:.1f}")
    console.print()

    console.print(f"[bold]Total API cost:[/bold] ${total_cost:.4f}")
    console.print()
    console.print(
        "[dim]Note: Accuracy is over the current labeled set only. "
        "Add real labeled CVs to data/eval_labels.json for trustworthy calibration.[/dim]"
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def _build_json_export(
    run_data: list[tuple[Label, PipelineResult | None, Prediction]],
    metrics: EvalResults,
    total_cost: float,
    total_duration: float,
) -> dict[str, Any]:
    """Serialize predictions + metrics to a JSON-serializable dict.

    Args:
        run_data: Per-CV run results.
        metrics: Aggregated metrics.
        total_cost: Total API cost USD.
        total_duration: Total wall-clock seconds.

    Returns:
        Dict ready for json.dumps().
    """
    cvs_out = []
    for label, result, pred in run_data:
        entry: dict[str, Any] = {
            "cv": label.cv_name,
            "true_isco": label.isco_code,
            "true_seniority": label.seniority,
            "predicted_isco": pred.isco_code,
            "predicted_seniority": pred.seniority,
            "predicted_salary_low": pred.salary.low if pred.salary else None,
            "predicted_salary_high": pred.salary.high if pred.salary else None,
            "score": result.score.total if result else None,
            "cost_usd": result.meta.get("total_cost_usd") if result else None,
            "error": result is None,
        }
        cvs_out.append(entry)

    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "total_cost_usd": total_cost,
        "total_duration_s": total_duration,
        "cvs": cvs_out,
        "metrics": {
            "n_total": metrics.n_total,
            "n_labeled": metrics.n_labeled,
            "n_with_salary": metrics.n_with_salary,
            "isco_exact_rate": metrics.isco_exact_rate,
            "isco_prefix3_rate": metrics.isco_prefix3_rate,
            "seniority_exact_rate": metrics.seniority_exact_rate,
            "seniority_off_by_one_rate": metrics.seniority_off_by_one_rate,
            "salary_mae": metrics.salary_mae,
            "salary_mape": metrics.salary_mape,
            "salary_within_15pct_rate": metrics.salary_within_15pct_rate,
            "salary_range_coverage_rate": metrics.salary_range_coverage_rate,
            "warnings_total": metrics.warnings_total,
            "warnings_per_cv": metrics.warnings_per_cv,
        },
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _generate_html(
    run_data: list[tuple[Label, PipelineResult | None, Prediction]],
    metrics: EvalResults,
    total_cost: float,
    total_duration: float,
    output_path: Path,
) -> None:
    """Write a vanilla HTML accuracy report.

    Args:
        run_data: Per-CV run results.
        metrics: Aggregated metrics.
        total_cost: Total API cost USD.
        total_duration: Total wall-clock seconds.
        output_path: Destination HTML file path.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metric_rows = ""
    if metrics.n_labeled > 0:
        metric_rows += (
            f"<tr><td>ISCO exact match</td><td>{format_pct(metrics.isco_exact_rate)}</td></tr>"
        )
        metric_rows += (
            f"<tr><td>ISCO 3-digit prefix</td><td>{format_pct(metrics.isco_prefix3_rate)}</td></tr>"
        )
        metric_rows += (
            f"<tr><td>Seniority exact</td><td>{format_pct(metrics.seniority_exact_rate)}</td></tr>"
        )
        metric_rows += f"<tr><td>Seniority off-by-1</td><td>{format_pct(metrics.seniority_off_by_one_rate)}</td></tr>"
    if metrics.n_with_salary > 0:
        metric_rows += f"<tr><td>Salary MAE</td><td>{format_czk(metrics.salary_mae)}</td></tr>"
        metric_rows += f"<tr><td>Salary MAPE</td><td>{format_pct(metrics.salary_mape)}</td></tr>"
        metric_rows += (
            f"<tr><td>Within ±15%</td><td>{format_pct(metrics.salary_within_15pct_rate)}</td></tr>"
        )
        metric_rows += f"<tr><td>True in pred range</td><td>{format_pct(metrics.salary_range_coverage_rate)}</td></tr>"

    cv_rows = ""
    for label, result, pred in run_data:
        true_isco = label.isco_code
        pred_isco = pred.isco_code or "N/A"
        isco_ok = pred_isco == true_isco
        isco_cls = "match" if isco_ok else "mismatch"
        true_sr = label.seniority
        pred_sr = pred.seniority or "N/A"
        sr_cls = "match" if pred_sr == true_sr else "mismatch"
        true_sal = _fmt_salary_label(label)
        pred_sal = f"{pred.salary.low:,}–{pred.salary.high:,}" if pred.salary else "N/A"
        score_str = f"{result.score.total:.1f}" if result else "ERROR"
        cost_str = f"${result.meta.get('total_cost_usd', 0):.4f}" if result else "-"
        status_cls = "ok" if result else "err"
        cv_rows += (
            f"<tr class='{status_cls}'>"
            f"<td>{label.cv_name}</td>"
            f"<td>{true_isco}</td>"
            f"<td class='{isco_cls}'>{pred_isco}</td>"
            f"<td>{true_sr}</td>"
            f"<td class='{sr_cls}'>{pred_sr}</td>"
            f"<td>{true_sal}</td>"
            f"<td>{pred_sal}</td>"
            f"<td>{score_str}</td>"
            f"<td>{cost_str}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Job Fit Estimator — Accuracy Eval Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1200px; margin: 40px auto; padding: 0 20px;
           color: #333; background: #f9f9f9; }}
    h1 {{ color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 10px; }}
    .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
    .metrics-table, .cv-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    .metrics-table th, .metrics-table td,
    .cv-table th, .cv-table td {{ padding: 8px 12px; border: 1px solid #e0e0e0; text-align: left; }}
    .metrics-table th, .cv-table th {{ background: #f0f4f8; font-weight: 600; }}
    tr.err {{ background: #fff5f5; }}
    .match {{ color: #22863a; font-weight: bold; }}
    .mismatch {{ color: #cb2431; }}
    h2 {{ margin-top: 32px; color: #1a1a2e; }}
    .note {{ font-size: 0.85em; color: #666; margin-top: 16px; }}
  </style>
</head>
<body>
  <h1>Job Fit Estimator — Accuracy Eval Report</h1>
  <div class="meta">
    Generated: {timestamp} |
    Labeled CVs: {metrics.n_labeled}/{metrics.n_total} |
    Total cost: ${total_cost:.4f} |
    Duration: {total_duration:.1f}s
  </div>

  <h2>Aggregate Metrics</h2>
  <table class="metrics-table">
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>{metric_rows}</tbody>
  </table>

  <h2>Per-CV Results</h2>
  <table class="cv-table">
    <thead>
      <tr>
        <th>CV</th><th>True ISCO</th><th>Pred ISCO</th>
        <th>True Sr.</th><th>Pred Sr.</th>
        <th>True Salary</th><th>Pred Salary</th>
        <th>Score</th><th>Cost</th>
      </tr>
    </thead>
    <tbody>{cv_rows}</tbody>
  </table>

  <p class="note">
    ISCO match: green = exact, red = mismatch.<br>
    Accuracy is over the current labeled set only. Add real labeled CVs to
    <code>data/eval_labels.json</code> for trustworthy calibration (~100 CVs target).
  </p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the eval harness."""
    arg_parser = argparse.ArgumentParser(description="Job Fit Estimator accuracy eval harness")
    arg_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N entries (cost guard)",
    )
    arg_parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="SUBSTR",
        help="Run only entries whose cv path contains SUBSTR",
    )
    arg_parser.add_argument("--json", type=Path, default=None, help="Export JSON to this path")
    arg_parser.add_argument("--html", type=Path, default=None, help="Export HTML to this path")
    args = arg_parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    console = Console()

    # ── Load labels ──────────────────────────────────────────────────────────
    if not _LABELS_PATH.exists():
        console.print(f"[red]ERROR: labels file not found: {_LABELS_PATH}[/red]")
        sys.exit(1)

    labels = _load_labels(_LABELS_PATH)

    # ── Apply filters ────────────────────────────────────────────────────────
    # Load raw entries again to get the original cv path for file resolution
    raw_entries: list[dict[str, Any]] = json.loads(_LABELS_PATH.read_text(encoding="utf-8"))
    filtered_entries = raw_entries
    if args.only:
        filtered_entries = [e for e in filtered_entries if args.only in e["cv"]]
    if args.limit:
        filtered_entries = filtered_entries[: args.limit]

    if not filtered_entries:
        console.print("[yellow]No entries match the given filters.[/yellow]")
        sys.exit(0)

    # Rebuild labels list to match filtered entries
    all_labels_by_name = {lb.cv_name: lb for lb in labels}

    console.print(
        f"\n[bold]Job Fit Estimator — Accuracy Eval[/bold]  "
        f"({len(filtered_entries)} CV(s), labels from data/eval_labels.json)\n"
    )

    # ── Initialize pipeline ──────────────────────────────────────────────────
    pipeline = Pipeline()

    # ── Run each CV ──────────────────────────────────────────────────────────
    run_data: list[tuple[Label, PipelineResult | None, Prediction]] = []
    predictions: list[Prediction] = []
    all_labels_used: list[Label] = []

    wall_start = time.monotonic()

    for entry in filtered_entries:
        cv_rel = Path(str(entry["cv"]))
        cv_path = _PROJECT_ROOT / cv_rel
        cv_name = cv_path.name
        label = all_labels_by_name.get(cv_name)
        if label is None:
            console.print(f"  [yellow]SKIP {cv_name}: no label found (name mismatch?)[/yellow]")
            continue

        if not cv_path.exists():
            console.print(f"  [yellow]SKIP {cv_name}: file not found at {cv_path}[/yellow]")
            run_data.append(
                (
                    label,
                    None,
                    Prediction(cv_name=cv_name, isco_code=None, seniority=None, salary=None),
                )
            )
            continue

        result, pred = _run_cv(pipeline, cv_path, label, console)
        run_data.append((label, result, pred))
        predictions.append(pred)
        all_labels_used.append(label)

    total_duration = time.monotonic() - wall_start
    total_cost = sum(r.meta.get("total_cost_usd", 0) for _, r, _ in run_data if r is not None)

    # ── Metrics ──────────────────────────────────────────────────────────────
    # Use ALL labels from the file for metric denominators (not just filtered ones),
    # but only pass predictions that actually ran.
    metrics = aggregate_metrics(predictions, all_labels_used)

    # ── Print report ─────────────────────────────────────────────────────────
    console.print()
    _print_per_cv_table(console, run_data)
    console.print()
    _print_aggregate(console, metrics, total_cost)

    # ── Exports ──────────────────────────────────────────────────────────────
    if args.json:
        payload = _build_json_export(run_data, metrics, total_cost, total_duration)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]JSON export: {args.json}[/green]")

    if args.html:
        _generate_html(run_data, metrics, total_cost, total_duration, args.html)
        console.print(f"[green]HTML report: {args.html}[/green]")


if __name__ == "__main__":
    main()
