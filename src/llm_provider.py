"""LLM provider abstraction layer.

OpenAIProvider: real implementation using client.beta.chat.completions.parse.
LLMProvider ABC retained as extensibility point for future providers (Ollama, local LLM).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, cast

import openai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pricing per 1M tokens: (input_usd, output_usd)
# OpenAI prices verified at openai.com/pricing as of 2026-05 — verify before merge.
# gpt-5-mini and gpt-5 prices are conservative estimates (Phase 22); exact prices
# not yet confirmed via live check — update when OpenAI publishes official rates.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5-mini": (0.25, 2.00),  # estimated — verify at openai.com/pricing
    "gpt-5": (1.25, 10.00),  # estimated — verify at openai.com/pricing
}

# Typical token usage per pipeline run, derived from observed eval logs:
# parser receives ~5 kCV-text + prompt tokens, emits ~1 k structured output;
# explainer receives ~1 k context + prompt, emits ~0.7 k recommendations.
# Multiplied by a 2.5× safety margin so a single $5 budget admits ~1000 runs
# on gpt-4o-mini and still catches runaway spend on gpt-4o (~$0.05 reserved).
_TYPICAL_PARSE_INPUT_TOKENS = 5_000
_TYPICAL_PARSE_OUTPUT_TOKENS = 1_000
_TYPICAL_EXPLAIN_INPUT_TOKENS = 1_500
_TYPICAL_EXPLAIN_OUTPUT_TOKENS = 700
_RESERVATION_SAFETY_MARGIN = 2.5


def _model_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Lookup model price and compute USD for a given token mix."""
    input_price, output_price = _PRICING.get(model, _PRICING["gpt-4o-mini"])
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def estimate_run_cost_usd(parser_model: str, explainer_model: str) -> float:
    """Best-effort upper-bound cost for one pipeline run.

    Used by the pipeline's pre-flight budget check so the daily cap reserves
    realistic spend, not a flat $0.005 that under-counts gpt-4o by 10×.
    """
    parse = _model_cost_usd(parser_model, _TYPICAL_PARSE_INPUT_TOKENS, _TYPICAL_PARSE_OUTPUT_TOKENS)
    explain = _model_cost_usd(
        explainer_model, _TYPICAL_EXPLAIN_INPUT_TOKENS, _TYPICAL_EXPLAIN_OUTPUT_TOKENS
    )
    return (parse + explain) * _RESERVATION_SAFETY_MARGIN


class LLMProviderError(RuntimeError):
    """Raised when an LLM call returns a refusal, empty parse, or fails validation.

    Carries the partial cost actually incurred (tokens consumed before the
    failure) so the pipeline can still log it against the daily budget.
    Otherwise refused/malformed responses would burn tokens silently and
    DAILY_API_BUDGET_USD would under-count.
    """

    def __init__(
        self,
        message: str,
        *,
        cost_usd: float = 0.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd
        self.meta: dict[str, Any] = meta or {}


class LLMProvider(ABC):
    """Abstract base for LLM providers that support structured extraction."""

    @abstractmethod
    def extract_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_tokens: int = 4096,
        *,
        system_prompt: str | None = None,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Extract structured data from prompt into schema.

        When system_prompt is provided, it is sent as the system message and
        `prompt` becomes the user message.  This separates trusted instructions
        from untrusted user/document content and is the preferred call pattern
        for CV parsing and other untrusted-input flows.

        When system_prompt is None (legacy callers), `prompt` is sent as a
        single user message as before.

        Returns:
            (parsed_model, meta) where meta contains:
            cost_usd, input_tokens, output_tokens, duration_s, model
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        """Embed texts into vectors. Returns (vectors, meta) where meta contains
        'cost_usd', 'input_tokens', 'model'."""


class OpenAIProvider(LLMProvider):
    """OpenAI provider using client.beta.chat.completions.parse for structured output.

    CRITICAL: temperature=0.0 is hardcoded — invariant #10.
    Uses Structured Outputs (response_format=<PydanticModel>) via the beta parse endpoint.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self.model = model
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def _compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute USD cost based on model pricing."""
        input_price, output_price = _PRICING.get(self.model, (0.15, 0.60))
        return (input_tokens * input_price + output_tokens * output_price) / 1_000_000

    def extract_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_tokens: int = 4096,
        *,
        system_prompt: str | None = None,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Call OpenAI beta parse endpoint, retry up to 3 times with backoff.

        Uses response_format=schema for Structured Outputs.
        Respects Retry-After header on RateLimitError.

        When system_prompt is provided, it is sent as role=system and `prompt`
        as role=user — this separates trusted instructions from untrusted document
        content (prompt injection mitigation).
        """
        delays = [1.0, 2.0, 4.0]
        last_exception: Exception | None = None

        # Build message list: system message + user message when system_prompt is
        # given, otherwise a single user message (backward-compatible).
        if system_prompt is not None:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        for attempt, delay in enumerate(delays):
            start = time.monotonic()
            try:
                # gpt-5* family quirks (all surfaced during Phase 22):
                # 1. uses max_completion_tokens (gpt-4o* accepts both)
                # 2. rejects custom temperature with HTTP 400 — model default 1.0 only
                # 3. reasoning_tokens count toward max_completion_tokens; default 4096
                #    gets eaten by reasoning, leaving zero output tokens. Need 16k+
                #    headroom for any non-trivial structured schema.
                # See decisions/2026-05-05-dynamic-salary.md "gpt-5 gotchas" section.
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "response_format": schema,
                }
                if self.model.startswith("gpt-5"):
                    kwargs["max_completion_tokens"] = max(max_tokens, 16000)
                    # temperature omitted (rejected by API)
                else:
                    kwargs["max_completion_tokens"] = max_tokens
                    kwargs["temperature"] = 0.0  # invariant #10 — gpt-4o* only
                response = self.client.beta.chat.completions.parse(**kwargs)
                duration = time.monotonic() - start

                # Compute token cost FIRST — usage is reported even when the
                # response is a refusal or has parsed=None. Cost is then
                # propagated through LLMProviderError on validation failures
                # so DAILY_API_BUDGET_USD reflects what was actually spent.
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost = self._compute_cost(input_tokens, output_tokens)
                partial_meta: dict[str, Any] = {
                    "cost_usd": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_s": duration,
                    "model": self.model,
                    "attempt": attempt,
                }

                # Refusal check — mandatory before model_validate
                msg = response.choices[0].message
                if msg.refusal:
                    raise LLMProviderError(
                        f"OpenAI refused request: {msg.refusal}",
                        cost_usd=cost,
                        meta=partial_meta,
                    )

                parsed = msg.parsed
                if parsed is None:
                    raise LLMProviderError(
                        "OpenAI returned None parsed result — model may have encountered an issue",
                        cost_usd=cost,
                        meta=partial_meta,
                    )

                try:
                    result = schema.model_validate(parsed)
                except Exception as exc:
                    raise LLMProviderError(
                        f"Structured output failed validation: {exc}",
                        cost_usd=cost,
                        meta=partial_meta,
                    ) from exc

                # Warn on empty languages list — helps detect silent schema-generator
                # failures without modifying schemas.py.
                # Access via getattr to avoid pyright complaining about unknown attribute on BaseModel.
                _languages = getattr(result, "languages", None)
                if isinstance(_languages, list) and len(cast(list[object], _languages)) == 0:
                    logger.warning(
                        "OpenAI parse returned empty languages list for schema=%s — "
                        "possible silent schema-generator failure. Check CV source text.",
                        schema.__name__,
                    )

                meta: dict[str, Any] = dict(partial_meta)
                logger.info(
                    "LLM call OK — model=%s tokens=%d+%d cost=$%.4f attempt=%d",
                    self.model,
                    input_tokens,
                    output_tokens,
                    cost,
                    attempt,
                )
                return result, meta

            except openai.RateLimitError as exc:
                last_exception = exc
                logger.exception(
                    "OpenAI rate limit on attempt %d/%d",
                    attempt + 1,
                    len(delays),
                )
                if attempt < len(delays) - 1:
                    # Respect Retry-After header; fall back to fixed delay
                    retry_after = float(exc.response.headers.get("retry-after", delay))
                    time.sleep(retry_after)

            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.APIStatusError,
            ) as exc:
                last_exception = exc
                logger.exception(
                    "LLM call failed on attempt %d/%d: %s",
                    attempt + 1,
                    len(delays),
                    exc,
                )
                if attempt < len(delays) - 1:
                    time.sleep(delay)

            except LLMProviderError:
                # Refusal, None-parsed, and validation failures are permanent —
                # do not retry. Cost has already been attached to the exception.
                raise

            except Exception as exc:
                last_exception = exc
                logger.exception(
                    "LLM call failed on attempt %d/%d: %s",
                    attempt + 1,
                    len(delays),
                    exc,
                )
                if attempt < len(delays) - 1:
                    time.sleep(delay)

        raise RuntimeError(
            f"OpenAIProvider failed after {len(delays)} attempts"
        ) from last_exception

    def embed(self, texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        """Embed texts via text-embedding-3-small.

        Uses one batched call. Returns (vectors, meta) where meta contains
        'cost_usd', 'input_tokens', 'model'.
        Raises on API error — no retry, embeddings rarely transient-fail.
        """
        if not texts:
            return [], {"cost_usd": 0.0, "input_tokens": 0, "model": "text-embedding-3-small"}
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        input_tokens = response.usage.total_tokens if response.usage else 0
        # text-embedding-3-small pricing: $0.02 per 1M tokens
        cost_usd = input_tokens * 0.02 / 1_000_000
        meta: dict[str, Any] = {
            "cost_usd": cost_usd,
            "input_tokens": input_tokens,
            "model": "text-embedding-3-small",
        }
        return vectors, meta


def get_provider(
    name: str = "openai",
    model: str | None = None,
) -> LLMProvider:
    """Factory — return configured LLM provider by name and model.

    Reads OPENAI_API_KEY from Pydantic Settings (.env-aware) and passes explicitly,
    so CLI scripts work without separate dotenv loading.

    Currently only OpenAI is supported. AnthropicProvider was removed 2026-05-05
    (see decisions/2026-05-05-openai-provider.md, "Update 2026-05-05-morning").
    LLMProvider ABC is retained for future provider additions (Ollama, local LLM).
    """
    if name == "openai":
        from src.config import settings

        api_key = settings.openai_api_key or None
        return OpenAIProvider(model=model or "gpt-4o-mini", api_key=api_key)
    if name == "anthropic":
        raise NotImplementedError(
            "AnthropicProvider was removed 2026-05-05. "
            "Only 'openai' is supported. "
            "See decisions/2026-05-05-openai-provider.md (morning update)."
        )
    raise ValueError(f"Unknown provider: {name!r}. Valid options: 'openai'")
