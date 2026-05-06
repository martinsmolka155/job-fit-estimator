"""LLM-based CV parser.

Converts raw text into structured Resume Pydantic model via OpenAI Structured Outputs.
Prompt lives in prompts/parser_system.txt — no inline prompts here.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from src.llm_provider import LLMProvider
from src.paths import PARSER_PROMPT_PATH as _PROMPT_PATH
from src.schemas import Resume

logger = logging.getLogger(__name__)

# Maximum characters sent to LLM — longer CVs get truncated.
# _PROMPT_PATH is anchored to the project root so the parser works regardless
# of the shell's current working directory.
_MAX_TEXT_CHARS = 30_000


class ResumeParser:
    """Parses CV text into a structured Resume using an LLM provider.

    Uses tool_use structured output via LLMProvider.extract_structured().
    All prompts are loaded from prompts/parser_system.txt — no inline prompts.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self._system_prompt: str | None = None

    def _load_prompt(self) -> str:
        """Load system prompt from file, caching after first read."""
        if self._system_prompt is None:
            if not _PROMPT_PATH.exists():
                raise FileNotFoundError(
                    f"Parser prompt not found at {_PROMPT_PATH}. "
                    "Ensure prompts/parser_system.txt exists in the project root."
                )
            self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        return self._system_prompt

    def parse(self, text: str) -> tuple[Resume, dict[str, Any]]:
        """Parse raw CV text into a structured Resume.

        Args:
            text: Raw text extracted from the CV document.

        Returns:
            (Resume, meta) where meta contains LLM cost/token info plus
            truncated=True if text was longer than _MAX_TEXT_CHARS.
        """
        meta: dict[str, Any] = {}

        # Truncate overly long CVs
        if len(text) > _MAX_TEXT_CHARS:
            logger.warning(
                "CV text truncated from %d to %d chars for LLM parsing",
                len(text),
                _MAX_TEXT_CHARS,
            )
            text = text[:_MAX_TEXT_CHARS]
            meta["truncated"] = True
        else:
            meta["truncated"] = False

        system_prompt = self._load_prompt()
        full_prompt = f"{system_prompt}\n\n{text}"

        try:
            raw_resume, llm_meta = self.llm.extract_structured(full_prompt, Resume, max_tokens=4096)
        except Exception:
            logger.exception("LLM parsing failed, returning empty Resume")
            resume = Resume(raw_text_length=len(text))
            meta.update({"cost_usd": 0.0, "error": "llm_failed"})
            return resume, meta

        # Post-processing: ensure raw_text_length is set to original length
        # (before truncation if it occurred). model_copy() preserves the concrete type.
        resume = cast("Resume", raw_resume.model_copy(update={"raw_text_length": len(text)}))

        meta.update(llm_meta)
        return resume, meta
