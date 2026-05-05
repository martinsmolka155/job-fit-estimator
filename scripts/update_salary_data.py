"""CLI: refresh data/salary_ranges.json from configured source.

Usage:
  uv run python scripts/update_salary_data.py            # uses LocalCSVSource if CSV exists, else fails
  uv run python scripts/update_salary_data.py --static   # rewrites JSON from itself (refresh _meta only)

Exit code 0 on success; non-zero on validation/IO failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so src/ is importable without package install.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.salary_pipeline import (  # noqa: E402
    LocalCSVSource,
    SalaryDataValidationError,
    StaticJSONSource,
    fetch_with_fallback,
    validate_salary_data,
    write_with_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_JSON = Path("data/salary_ranges.json")
DEFAULT_CSV = Path("data/salary_csu_2025.csv")
DEFAULT_WARNING = (
    "Manually curated demo data, supplemented by CSU.gov.cz when CSV is provided. "
    "NOT live market data; do not use for compensation decisions."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh salary data with optional CSU CSV import."
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Refresh _meta only from current JSON (no source change).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to CSU CSV (default: data/salary_csu_2025.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_JSON,
        help="Output JSON path (default: data/salary_ranges.json).",
    )
    args = parser.parse_args()

    static_source = StaticJSONSource(args.out)

    if args.static:
        logger.info("Refreshing _meta from existing JSON only.")
        try:
            data = static_source.fetch()
            validate_salary_data(data)
            write_with_meta(data, args.out, static_source.source_url, DEFAULT_WARNING)
            logger.info("Wrote %s (static refresh).", args.out)
            return 0
        except SalaryDataValidationError as exc:
            logger.error("Validation failed: %s", exc)
            return 1
        except Exception as exc:
            logger.exception("Write failed: %s", exc)
            return 2

    if not args.csv.exists():
        logger.error(
            "CSV not found at %s. Use --static to refresh existing JSON, or provide CSV.",
            args.csv,
        )
        return 1

    csv_source = LocalCSVSource(args.csv)
    try:
        data, source_used = fetch_with_fallback(csv_source, static_source)
        write_with_meta(data, args.out, source_used.source_url, DEFAULT_WARNING)
        logger.info("Wrote %s using %s.", args.out, source_used.source_url)
        return 0
    except SalaryDataValidationError as exc:
        logger.error("Validation failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Update failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
