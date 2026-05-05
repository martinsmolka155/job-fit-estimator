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


class LLMProvider(ABC):
    """Abstract base for LLM providers that support structured extraction."""

    @abstractmethod
    def extract_structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_tokens: int = 4096,
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Extract structured data from prompt into schema.

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
    ) -> tuple[BaseModel, dict[str, Any]]:
        """Call OpenAI beta parse endpoint, retry up to 3 times with backoff.

        Uses response_format=schema for Structured Outputs.
        Respects Retry-After header on RateLimitError.
        """
        delays = [1.0, 2.0, 4.0]
        last_exception: Exception | None = None

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
                    "messages": [{"role": "user", "content": prompt}],
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

                # Refusal check — mandatory before model_validate
                msg = response.choices[0].message
                if msg.refusal:
                    raise ValueError(f"OpenAI refused request: {msg.refusal}")

                parsed = msg.parsed
                if parsed is None:
                    raise ValueError(
                        "OpenAI returned None parsed result — model may have encountered an issue"
                    )

                result = schema.model_validate(parsed)

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

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                cost = self._compute_cost(input_tokens, output_tokens)

                meta: dict[str, Any] = {
                    "cost_usd": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "duration_s": duration,
                    "model": self.model,
                    "attempt": attempt,
                }
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

            except ValueError:
                # Refusal and None-parsed are permanent — do not retry.
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
