"""Headless CLI for the Job Fit Estimator.

Runs the analysis pipeline on a single CV file and prints the JSON result to
stdout. Errors go to stderr with a non-zero exit code.

Usage:
    python main.py path/to/cv.pdf
    python main.py path/to/cv.pdf --output result.json
    python main.py path/to/cv.docx --pretty

Requires OPENAI_API_KEY in the environment (or .env file). ISPV salary data
must be present at data/ispv_2025.xlsx — download via the Streamlit sidebar
or place the file there manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import settings
from src.cost_tracker import BudgetExceededError
from src.pipeline import Pipeline, PipelineInfrastructureError
from src.salary_ispv import ISPVLookupError

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a CV (PDF/DOCX) and emit a JSON report.",
    )
    parser.add_argument("cv_path", type=Path, help="Path to the CV file (.pdf or .docx)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (indent=2).",
    )
    args = parser.parse_args()

    if not args.cv_path.exists():
        print(f"error: file not found: {args.cv_path}", file=sys.stderr)
        return 2
    if args.cv_path.suffix.lower() not in (".pdf", ".docx"):
        print(
            f"error: unsupported file type {args.cv_path.suffix!r} (expected .pdf or .docx)",
            file=sys.stderr,
        )
        return 2

    if not settings.openai_api_key:
        print("error: OPENAI_API_KEY is not set (check your .env file)", file=sys.stderr)
        return 3

    pipeline = Pipeline(
        "openai",
        parser_model=settings.parser_model,
        explainer_model=settings.explainer_model,
    )

    try:
        result = pipeline.run(args.cv_path)
    except BudgetExceededError as e:
        print(f"budget: daily API budget exhausted: {e}", file=sys.stderr)
        return 5
    except PipelineInfrastructureError as e:
        print(f"infrastructure: {e}", file=sys.stderr)
        return 6
    except ISPVLookupError as e:
        print(f"error: salary estimate unavailable: {e}", file=sys.stderr)
        return 4
    except Exception as e:
        logger.exception("pipeline failed")
        print(f"error: pipeline failed: {e}", file=sys.stderr)
        return 1

    payload = result.model_dump(mode="json")
    rendered = json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
