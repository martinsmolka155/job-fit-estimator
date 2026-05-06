"""Project-anchored filesystem paths.

Every module that loads a prompt, data file, or appends to the cost log
should pull its path from here. Anchoring to ``__file__`` rather than the
current working directory avoids the "works from project root, mysterious
failures elsewhere" trap when the app is launched from a service supervisor,
a different shell, or imported as a module.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, computed once at import time relative to this file.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: System prompts.
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
PARSER_PROMPT_PATH: Path = PROMPTS_DIR / "cv_parser_system.txt"
EXPLAINER_PROMPT_PATH: Path = PROMPTS_DIR / "explainer_system.txt"

#: Data files.
DATA_DIR: Path = PROJECT_ROOT / "data"
ISPV_XLSX_PATH: Path = DATA_DIR / "ispv_2025.xlsx"
INFLATION_FACTORS_PATH: Path = DATA_DIR / "inflation_factors.json"
COST_LOG_PATH: Path = DATA_DIR / ".cost_log.jsonl"
