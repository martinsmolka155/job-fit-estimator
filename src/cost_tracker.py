"""Per-run cost tracking + daily budget enforcement.

Persistence: append-only JSONL at data/.cost_log.jsonl (gitignored).
Budget: read from env DAILY_API_BUDGET_USD (default 5.00).

Raises BudgetExceededError BEFORE making LLM calls when budget would be
exceeded. Warns (does not raise) when 90%+ of budget consumed.
"""

from __future__ import annotations

import json
import logging
import os
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


def _budget_from_env(default: float = DEFAULT_BUDGET_USD) -> float:
    raw = os.environ.get("DAILY_API_BUDGET_USD")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid DAILY_API_BUDGET_USD=%r; using default %.2f", raw, default)
        return default


def today_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def daily_spent(log_path: Path = DEFAULT_COST_LOG_PATH) -> float:
    """Sum of total_cost_usd across all records with today's UTC date."""
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


def check_budget(estimated_cost_usd: float, log_path: Path = DEFAULT_COST_LOG_PATH) -> None:
    """Raise BudgetExceededError if estimated_cost would push today over budget.

    Warn (log only) at 90%+ of budget consumed.
    """
    budget = _budget_from_env()
    spent = daily_spent(log_path)
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
    """Append a CostRecord to the JSONL log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]
