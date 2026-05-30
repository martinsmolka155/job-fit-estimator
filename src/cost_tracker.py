"""Per-run cost tracking + daily budget enforcement.

Persistence: append-only JSONL at data/.cost_log.jsonl (gitignored).
Budget: read from env DAILY_API_BUDGET_USD (default 5.00).

Raises BudgetExceededError BEFORE making LLM calls when budget would be
exceeded. Warns (does not raise) when 90%+ of budget consumed.

Concurrency model — reservation protocol
-----------------------------------------
The bare check_budget()+record_run() pattern has a TOCTOU window equal to the
whole pipeline duration: two concurrent requests both read "spent=0" before
either writes its cost record.  To close that window without holding a lock
for the entire pipeline we use a *reservation* protocol:

  1. ``reserve(run_id, estimate)``  — atomically (under ``_budget_lock``):
        • checks that committed_spend + outstanding_reservations + estimate
          does not exceed the daily budget
        • if ok, stores an in-memory reservation keyed by run_id
        • raises BudgetExceededError otherwise (reservation is NOT stored)

  2. ``finalize(run_id, actual_cost, record)``  — atomically (under
     ``_budget_lock``):
        • removes the reservation (frees the slot even on failure/abort)
        • appends the actual CostRecord to the JSONL log

``daily_spent()`` returns committed JSONL actuals + sum of outstanding
in-memory reservations, so any concurrent ``reserve`` call sees the full
picture.

Guarantees
----------
* Within one OS process, two concurrent requests with estimates that
  individually fit but jointly exceed the budget → exactly one passes.
* A failed/aborted run MUST call ``finalize`` (typically in a try/finally)
  so its reservation is released.  The CostRecord for an aborted run is
  written with the actual (possibly partial) cost and an error field.
* Multi-worker deploys (uvicorn --workers N, Gunicorn) need a shared store
  (e.g. Redis atomic increment) for cross-process protection — that is a
  separate item.  Single-worker is the supported configuration.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# COST_LOG_PATH is anchored to the project root in src/paths.py so the cost
# log lands in the right place regardless of the shell's current working
# directory.
from src.paths import COST_LOG_PATH as DEFAULT_COST_LOG_PATH

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_USD = 5.00
WARN_THRESHOLD_PCT = 0.90

# Serialises all budget-mutating operations within one process.
# See module docstring for multi-worker caveat.
_budget_lock = threading.Lock()

# Outstanding reservations: {run_id: estimated_cost_usd}.
# Mutated only while _budget_lock is held.
_reservations: dict[str, float] = {}


class BudgetExceededError(RuntimeError):
    """Raised when an LLM call would exceed the daily budget."""


@dataclass
class CostRecord:
    """One pipeline run's cost+timing record."""

    run_id: str
    timestamp_utc: str
    fixture_or_file: str
    total_cost_usd: float
    parse_cost_usd: float
    explain_cost_usd: float
    embed_cost_usd: float
    duration_s: float
    score_total: float | None = None
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=lambda: {})


def _budget_from_settings() -> float:
    """Return the daily API budget by re-reading Settings.

    Re-instantiates Settings on every call rather than using the module-level
    `settings` singleton — this keeps the budget responsive to monkeypatched
    env vars in tests and to operator changes (e.g. .env edit + restart of
    the worker but not of the importer process).
    """
    from pydantic import ValidationError

    from src.config import Settings

    try:
        return Settings().daily_api_budget_usd
    except ValidationError:
        logger.warning(
            "Invalid DAILY_API_BUDGET_USD; falling back to default $%.2f",
            DEFAULT_BUDGET_USD,
        )
        return DEFAULT_BUDGET_USD


def today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def daily_spent(log_path: Path = DEFAULT_COST_LOG_PATH) -> float:
    """Sum of today's committed actuals (JSONL) + outstanding reservations.

    Both terms are accounted for so that a ``reserve`` call from thread B
    sees the estimated cost that thread A reserved but has not yet finalized.
    Must be called with ``_budget_lock`` held when used inside the reservation
    protocol (``_daily_spent_locked``), or stand-alone for read-only queries.
    """
    committed = _committed_spent(log_path)
    reserved = sum(_reservations.values())
    return committed + reserved


def _committed_spent(log_path: Path) -> float:
    """Sum of total_cost_usd in the JSONL log for today's UTC date only."""
    if not log_path.exists():
        return 0.0
    today = today_utc()
    total = 0.0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = rec.get("timestamp_utc", "")
        if ts.startswith(today):
            total += float(rec.get("total_cost_usd", 0.0))
    return total


def reserve(
    run_id: str,
    estimated_cost_usd: float,
    log_path: Path = DEFAULT_COST_LOG_PATH,
) -> None:
    """Atomically claim a budget slot for *run_id*.

    Checks that committed_spend + outstanding_reservations + estimate does not
    exceed the daily budget.  If the check passes, stores the reservation in
    ``_reservations`` so subsequent concurrent calls see the claimed amount.

    Raises:
        BudgetExceededError: if the projected spend (including all outstanding
            reservations) would exceed the daily budget.

    The caller MUST pair every successful ``reserve`` call with a ``finalize``
    call (in a try/finally) to prevent reservation leaks.
    """
    with _budget_lock:
        budget = _budget_from_settings()
        # daily_spent() here includes reservations from other in-flight runs.
        spent = _committed_spent(log_path) + sum(_reservations.values())
        projected = spent + estimated_cost_usd
        if projected > budget:
            raise BudgetExceededError(
                f"Daily budget would be exceeded: "
                f"spent+reserved={spent:.4f} + estimate={estimated_cost_usd:.4f} "
                f"= {projected:.4f} > budget={budget:.2f} (DAILY_API_BUDGET_USD)"
            )
        if projected > budget * WARN_THRESHOLD_PCT:
            logger.warning(
                "Approaching daily budget: spent+reserved=$%.4f + estimate=$%.4f "
                "/ budget=$%.2f (%.0f%%)",
                spent,
                estimated_cost_usd,
                budget,
                projected / budget * 100,
            )
        _reservations[run_id] = estimated_cost_usd


def finalize(
    run_id: str,
    record: CostRecord,
    log_path: Path = DEFAULT_COST_LOG_PATH,
) -> None:
    """Release the reservation for *run_id* and persist the actual CostRecord.

    Called at the end of a run (success or failure).  Atomically (under
    ``_budget_lock``) removes the in-memory reservation so future ``reserve``
    calls see the freed slot, then appends the actual cost record to the JSONL
    log.

    If *run_id* has no outstanding reservation (e.g. called twice, or after a
    reservation was never made), the function logs a warning and still persists
    the record — it does not raise.
    """
    with _budget_lock:
        if run_id in _reservations:
            del _reservations[run_id]
        else:
            logger.warning(
                "finalize called for run_id=%s which has no outstanding reservation",
                run_id,
            )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _release_reservation(run_id: str) -> None:
    """Release the in-memory reservation for *run_id* without writing a record.

    This is a low-level helper for callers that manage cost-record persistence
    separately (e.g. ``pipeline.py`` which calls ``record_run`` directly via
    ``_persist_cost_record``).  Prefer ``finalize`` when you want the atomic
    release + persist in one call.

    Safe to call even if no reservation exists (logs a debug message, no raise).
    """
    with _budget_lock:
        if run_id in _reservations:
            del _reservations[run_id]
        else:
            logger.debug(
                "_release_reservation called for run_id=%s with no outstanding reservation",
                run_id,
            )


def check_budget(estimated_cost_usd: float, log_path: Path = DEFAULT_COST_LOG_PATH) -> None:
    """Read-only budget check — does NOT create a reservation.

    Acquires ``_budget_lock`` for a consistent read but does not store any
    reservation.  Use ``reserve`` / ``finalize`` instead for the full atomic
    protocol that prevents the TOCTOU window in concurrent pipelines.

    Kept for backwards-compatibility with callers that only need a quick
    pre-flight check without the reservation lifecycle.
    """
    with _budget_lock:
        _check_budget_locked(estimated_cost_usd, log_path)


def _check_budget_locked(estimated_cost_usd: float, log_path: Path) -> None:
    """Inner check — must be called with ``_budget_lock`` held."""
    budget = _budget_from_settings()
    spent = _committed_spent(log_path) + sum(_reservations.values())
    projected = spent + estimated_cost_usd
    if projected > budget:
        raise BudgetExceededError(
            f"Daily budget would be exceeded: spent={spent:.4f} + estimate={estimated_cost_usd:.4f} "
            f"= {projected:.4f} > budget={budget:.2f} (DAILY_API_BUDGET_USD)"
        )
    if projected > budget * WARN_THRESHOLD_PCT:
        logger.warning(
            "Approaching daily budget: spent=$%.4f + estimate=$%.4f / budget=$%.2f (%.0f%%)",
            spent,
            estimated_cost_usd,
            budget,
            projected / budget * 100,
        )


def record_run(record: CostRecord, log_path: Path = DEFAULT_COST_LOG_PATH) -> None:
    """Append a CostRecord to the JSONL log (no budget check, no lock).

    Use ``finalize`` instead when you have an outstanding reservation.
    This function exists for callers that write cost records outside the
    reservation lifecycle (e.g. metrics tooling, backfill scripts).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def atomic_check_and_record(
    estimated_cost_usd: float,
    record: CostRecord,
    log_path: Path = DEFAULT_COST_LOG_PATH,
) -> None:
    """Check the budget and append the cost record in a single locked operation.

    Preferred over separate ``check_budget`` + ``record_run`` calls when the
    caller wants the TOCTOU window fully closed: no second coroutine/thread can
    slip past the check between these two steps.

    Note: this helper is appropriate only when the actual cost is known at check
    time (e.g. after a dry-run or for very cheap fixed-cost operations).  For
    long-running pipelines where actual cost is unknown at check time, use the
    ``reserve`` / ``finalize`` pair instead.
    """
    with _budget_lock:
        _check_budget_locked(estimated_cost_usd, log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
