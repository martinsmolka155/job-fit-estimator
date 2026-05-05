"""CLI: summarize last N runs from data/.cost_log.jsonl.

Usage: uv run python scripts/show_metrics.py [--last N] [--today]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cost_tracker import DEFAULT_COST_LOG_PATH, today_utc  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=20, help="Show stats from last N runs.")
    parser.add_argument("--today", action="store_true", help="Restrict to today's UTC date.")
    parser.add_argument("--log", type=Path, default=DEFAULT_COST_LOG_PATH)
    args = parser.parse_args()

    if not args.log.exists():
        print(f"No cost log at {args.log}. Run the pipeline first.")
        return 1

    records: list[dict[str, object]] = []
    today = today_utc()
    for line in args.log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec: dict[str, object] = json.loads(line)
        except json.JSONDecodeError:
            continue
        if args.today and not str(rec.get("timestamp_utc", "")).startswith(today):
            continue
        records.append(rec)
    if not records:
        print("No records match filter.")
        return 1

    records = records[-args.last :]
    total_cost = sum(float(r.get("total_cost_usd", 0.0)) for r in records)  # type: ignore[arg-type]
    durations = [float(r.get("duration_s", 0.0)) for r in records]  # type: ignore[arg-type]
    errors = [r.get("error") for r in records if r.get("error")]
    error_counter: Counter[str] = Counter(str(e) for e in errors)

    print(f"Showing last {len(records)} runs (filter: {'today' if args.today else 'all'})")
    print(f"  Total cost:   ${total_cost:.4f}")
    print(f"  Avg cost:     ${total_cost / len(records):.4f}")
    print(f"  Avg duration: {mean(durations):.2f}s")
    print(f"  Errors:       {len(errors)}/{len(records)}")
    if error_counter:
        print("  Top errors:")
        for err, count in error_counter.most_common(3):
            print(f"    {count}x  {err[:80]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
