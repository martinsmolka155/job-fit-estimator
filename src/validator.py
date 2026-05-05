"""Hallucination guard / Resume validator.

Cross-checks extracted Resume fields against raw CV text using normalized substring matching.
Uses _normalize() helper (lowercase + remove accents + collapse whitespace) for real-world robustness.

Key insight: LLMs may extract "Ceska Sporitelna" when CV says "Česká spořitelna".
Normalization ensures accent/case differences do not produce false positive flags.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

from src.llm_provider import LLMProvider
from src.schemas import Resume, ValidationFlag

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase + remove diacritics/accents + collapse whitespace.

    Uses NFKD Unicode normalization to decompose accented characters,
    then strips combining characters. This handles Czech (á→a, č→c, ž→z, etc.)
    and other European scripts.
    """
    nfkd = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents).strip()


def _is_in_text(needle: str, haystack: str) -> bool:
    """Return True if needle appears in haystack (after normalization of both).

    Short tokens (≤ 3 chars) require a word-boundary match to avoid false
    positives where a single letter happens to occur inside any longer word
    (e.g. "R" inside "Senior", "Go" inside "Google", "C" inside "Project").
    Longer tokens fall back to plain substring containment, which tolerates
    embeddings such as "PostgreSQL" inside "Senior PostgreSQL DBA".
    """
    needle_n = _normalize(needle)
    haystack_n = _normalize(haystack)
    if not needle_n:
        return True
    if len(needle_n) <= 3:
        # Word-boundary match. re.escape covers special regex chars (e.g. "C++").
        return re.search(rf"(?<!\w){re.escape(needle_n)}(?!\w)", haystack_n) is not None
    return needle_n in haystack_n


class ResumeValidator:
    """Cross-checks Resume fields against the original raw CV text.

    Generates ValidationFlags for fields that appear to be hallucinated
    (i.e., not found in the source document).
    """

    def validate(self, resume: Resume, raw_text: str) -> list[ValidationFlag]:
        """Validate Resume fields against raw CV text.

        Args:
            resume: Parsed Resume to validate.
            raw_text: Original text extracted from the CV document.

        Returns:
            List of ValidationFlag objects. Empty list means no issues found.
        """
        flags: list[ValidationFlag] = []

        # --- Strong signals (errors) ---
        if resume.full_name and not _is_in_text(resume.full_name, raw_text):
            flags.append(
                ValidationFlag(
                    field_path="full_name",
                    issue=f"Name '{resume.full_name}' not found in raw text — likely hallucinated",
                    severity="error",
                )
            )

        if resume.email and not _is_in_text(resume.email, raw_text):
            flags.append(
                ValidationFlag(
                    field_path="email",
                    issue=f"Email '{resume.email}' not found in raw text — likely hallucinated",
                    severity="error",
                )
            )

        # --- Experience checks ---
        for i, exp in enumerate(resume.experiences):
            if not _is_in_text(exp.company, raw_text):
                flags.append(
                    ValidationFlag(
                        field_path=f"experiences.{i}.company",
                        issue=f"Company '{exp.company}' not found in raw text",
                        severity="warning",
                    )
                )

            start_str = str(exp.start_year)
            if not _is_in_text(start_str, raw_text):
                flags.append(
                    ValidationFlag(
                        field_path=f"experiences.{i}.start_year",
                        issue=f"Start year {exp.start_year} not found in raw text",
                        severity="warning",
                    )
                )

            if exp.end_year is not None and not _is_in_text(str(exp.end_year), raw_text):
                flags.append(
                    ValidationFlag(
                        field_path=f"experiences.{i}.end_year",
                        issue=f"End year {exp.end_year} not found in raw text",
                        severity="warning",
                    )
                )

        # --- Education checks ---
        for i, edu in enumerate(resume.educations):
            if edu.institution and not _is_in_text(edu.institution, raw_text):
                flags.append(
                    ValidationFlag(
                        field_path=f"educations.{i}.institution",
                        issue=f"Institution '{edu.institution}' not found in raw text",
                        severity="warning",
                    )
                )

            if edu.year_completed is not None and not _is_in_text(
                str(edu.year_completed), raw_text
            ):
                flags.append(
                    ValidationFlag(
                        field_path=f"educations.{i}.year_completed",
                        issue=f"Graduation year {edu.year_completed} not found in raw text",
                        severity="info",
                    )
                )

        # --- Skill checks ---
        for i, skill in enumerate(resume.skills):
            if not _is_in_text(skill.name, raw_text):
                flags.append(
                    ValidationFlag(
                        field_path=f"skills.{i}.name",
                        issue=f"Skill '{skill.name}' not found in raw text",
                        severity="warning",
                    )
                )

        # --- Cross-check: unrealistic total years ---
        # Merge overlapping intervals so parallel HPP + freelance roles don't
        # double-count, and compute "ongoing" against today's year (not a
        # hard-coded constant that decays as time passes).
        current_year = datetime.now(UTC).year
        intervals = sorted(
            (exp.start_year, exp.end_year or current_year)
            for exp in resume.experiences
            if (exp.end_year or current_year) > exp.start_year
        )
        merged: list[tuple[int, int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        total_years = sum(end - start for start, end in merged)
        # Heuristic: if total experience years > raw_text_length / 100, something's off
        if resume.raw_text_length > 0 and total_years > resume.raw_text_length / 100:
            flags.append(
                ValidationFlag(
                    field_path="experiences",
                    issue=(
                        f"Total experience years ({total_years}) seems high relative to "
                        f"CV length ({resume.raw_text_length} chars)"
                    ),
                    severity="warning",
                )
            )

        if flags:
            logger.warning(
                "Validation produced %d flags (%d errors, %d warnings, %d info)",
                len(flags),
                sum(1 for f in flags if f.severity == "error"),
                sum(1 for f in flags if f.severity == "warning"),
                sum(1 for f in flags if f.severity == "info"),
            )

        return flags


def compute_confidence(flags: list[ValidationFlag]) -> float:
    """Compute confidence score 0-1 based on validation flags.

    Each error reduces confidence by 0.20, each warning by 0.05.
    Clamped to minimum 0.0.
    """
    errors = sum(1 for f in flags if f.severity == "error")
    warnings = sum(1 for f in flags if f.severity == "warning")
    return max(0.0, 1.0 - errors * 0.2 - warnings * 0.05)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. No numpy dep."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingValidator:
    """Layer-2 hallucination detector via embedding cosine similarity.

    For entities flagged as missing/error by ResumeValidator (Layer 1),
    compute cosine similarity between the entity text and chunks of the
    raw CV text. If max similarity > threshold, emit ValidationFlag with
    severity='info' indicating a fuzzy match was found — likely paraphrasing
    rather than hallucination (e.g., 'CVUT' vs 'Czech Technical University').
    """

    # Cosine similarity threshold for "fuzzy match found" decision.
    #
    # Calibrated against real OpenAI text-embedding-3-small (1536-dim) measurements:
    # - EN/CS paraphrase pairs (e.g. "ČVUT" ↔ "Czech Technical University"): ~0.67
    # - Different institutions (e.g. "ČVUT" ↔ "Cambridge"): ~0.41
    # - Same string self-similarity: 1.00
    #
    # Default 0.65 captures legitimate paraphrases while rejecting unrelated entities.
    # Original spec used 0.85 but that was unreachable for legitimate paraphrases —
    # see Phase 19 review for measurement details. Production deployment may need
    # per-domain tuning (institution names vs company names vs job titles have
    # different similarity distributions).
    DEFAULT_THRESHOLD: float = 0.65
    DEFAULT_CHUNK_SIZE: int = 200  # characters per chunk

    def __init__(
        self,
        provider: LLMProvider,
        threshold: float = DEFAULT_THRESHOLD,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.provider = provider
        self.threshold = threshold
        self.chunk_size = chunk_size

    def upgrade_flags(
        self,
        flags: list[ValidationFlag],
        raw_text: str,
    ) -> tuple[list[ValidationFlag], dict[str, Any]]:
        """Re-classify flags using embedding similarity against raw_text chunks.

        For each flag in `flags`:
        - If severity is 'error' or 'warning' (Layer 1 says missing):
          compute cosine sim between flag's issue text and chunked raw_text.
          If max_sim > threshold, append a NEW ValidationFlag (severity='info')
          noting the fuzzy match. Original flag preserved.
        - If severity is 'info': pass through unchanged.

        Returns (new list of flags (original + new info flags), meta) where
        meta contains 'embed_cost_usd'.
        """
        if not flags or not raw_text:
            return list(flags), {"embed_cost_usd": 0.0}

        # Chunk raw_text into fixed-size character windows
        chunks = [
            raw_text[i : i + self.chunk_size] for i in range(0, len(raw_text), self.chunk_size)
        ]
        if not chunks:
            return list(flags), {"embed_cost_usd": 0.0}

        # Only process flags that Layer 1 marked as missing/suspicious
        flag_texts = [f.issue for f in flags if f.severity in ("error", "warning")]
        if not flag_texts:
            return list(flags), {"embed_cost_usd": 0.0}

        # Batch embed: chunks first, then flag issue texts — single API call
        all_texts = chunks + flag_texts
        embeddings, embed_meta = self.provider.embed(all_texts)
        chunk_embeds = embeddings[: len(chunks)]
        flag_embeds = embeddings[len(chunks) :]

        upgraded: list[ValidationFlag] = list(flags)
        flag_idx = 0
        for f in flags:
            if f.severity not in ("error", "warning"):
                continue
            f_vec = flag_embeds[flag_idx]
            flag_idx += 1
            max_sim = max(_cosine_similarity(f_vec, c_vec) for c_vec in chunk_embeds)
            if max_sim > self.threshold:
                upgraded.append(
                    ValidationFlag(
                        field_path=f.field_path,
                        issue=(
                            f"Fuzzy match found in raw text "
                            f"(cosine similarity {max_sim:.3f} > {self.threshold}). "
                            f"Original Layer-1 flag may be paraphrasing, not hallucination."
                        ),
                        severity="info",
                    )
                )
            logger.debug(
                "EmbeddingValidator: field=%s max_sim=%.4f threshold=%.2f upgraded=%s",
                f.field_path,
                max_sim,
                self.threshold,
                max_sim > self.threshold,
            )

        return upgraded, {"embed_cost_usd": embed_meta["cost_usd"]}
