"""Tests for LLM provider abstraction layer.

IMPORTANT: All tests use pytest-mock — NO real API calls.
Real API calls only in eval harness (Phase 10) and UI smoke test (Phase 11).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from src.llm_provider import (
    LLMProviderError,
    OpenAIProvider,
    estimate_run_cost_usd,
    get_provider,
)


class _SimpleSchema(BaseModel):
    """Minimal schema for testing."""

    value: str


def _make_openai_mock_response(
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    refusal: str | None = None,
) -> MagicMock:
    """Build a mock OpenAI parse response (client.beta.chat.completions.parse shape)."""
    msg = MagicMock()
    msg.refusal = refusal
    msg.parsed = _SimpleSchema(value="extracted_value") if refusal is None else None
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


class TestGetProvider:
    def test_get_openai_provider_default(self) -> None:
        """get_provider() with no args returns OpenAIProvider (single-provider MVP)."""
        with patch("openai.OpenAI"):
            provider = get_provider()
        assert isinstance(provider, OpenAIProvider)

    def test_get_openai_provider_explicit(self) -> None:
        """get_provider('openai') returns OpenAIProvider."""
        with patch("openai.OpenAI"):
            provider = get_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_get_anthropic_raises_not_implemented(self) -> None:
        """get_provider('anthropic') raises NotImplementedError (removed 2026-05-05)."""
        with pytest.raises(NotImplementedError, match="AnthropicProvider was removed"):
            get_provider("anthropic")

    def test_invalid_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid")

    def test_get_provider_with_model_override(self) -> None:
        """get_provider passes model override to OpenAIProvider."""
        with patch("openai.OpenAI"):
            openai_provider = get_provider("openai", model="gpt-4o")
        assert isinstance(openai_provider, OpenAIProvider)
        assert openai_provider.model == "gpt-4o"


class TestOpenAIProvider:
    def test_returns_correct_model_type(self) -> None:
        """extract_structured returns an instance of the requested schema with correct value."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()
        mock_response = _make_openai_mock_response()

        with patch.object(
            provider.client.beta.chat.completions, "parse", return_value=mock_response
        ):
            result, _meta = provider.extract_structured("test prompt", _SimpleSchema)

        assert isinstance(result, _SimpleSchema)
        assert result.value == "extracted_value"

    def test_temperature_zero_passed_to_api(self) -> None:
        """CRITICAL invariant #10: temperature=0.0 must reach the actual parse() call."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()
        mock_response = _make_openai_mock_response()

        with patch.object(
            provider.client.beta.chat.completions,
            "parse",
            return_value=mock_response,
        ) as mock_parse:
            provider.extract_structured("test prompt", _SimpleSchema)

        call_kwargs = mock_parse.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.0, (
            f"temperature must be 0.0, got {call_kwargs.get('temperature')}"
        )

    def test_meta_contains_required_keys(self) -> None:
        """meta dict must contain all required keys with correct values."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()
        mock_response = _make_openai_mock_response(prompt_tokens=1000, completion_tokens=500)

        with patch.object(
            provider.client.beta.chat.completions, "parse", return_value=mock_response
        ):
            _result, meta = provider.extract_structured("test prompt", _SimpleSchema)

        assert "cost_usd" in meta
        assert "input_tokens" in meta
        assert "output_tokens" in meta
        assert "duration_s" in meta
        assert "model" in meta
        assert "attempt" in meta
        assert meta["cost_usd"] > 0
        assert meta["input_tokens"] == 1000
        assert meta["output_tokens"] == 500
        assert meta["model"] == "gpt-4o-mini"

    def test_refusal_raises_value_error(self) -> None:
        """OpenAI refusal must raise ValueError immediately, not be retried."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()
        mock_response = _make_openai_mock_response(refusal="I cannot help with this")

        with (
            patch.object(
                provider.client.beta.chat.completions,
                "parse",
                return_value=mock_response,
            ) as mock_parse,
            pytest.raises(LLMProviderError, match="OpenAI refused"),
        ):
            provider.extract_structured("test prompt", _SimpleSchema)

        assert mock_parse.call_count == 1, (
            f"Refusal must not be retried — expected 1 call, got {mock_parse.call_count}"
        )

    def test_retry_on_rate_limit_reads_retry_after_header(self) -> None:
        """RateLimitError must respect Retry-After header, not fixed delay."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()
        success_response = _make_openai_mock_response()

        rate_limit_response = MagicMock()
        rate_limit_response.headers = {"retry-after": "5"}

        side_effects = [
            openai.RateLimitError(
                message="rate limited",
                response=rate_limit_response,
                body=None,
            ),
            success_response,
        ]

        with (
            patch.object(
                provider.client.beta.chat.completions,
                "parse",
                side_effect=side_effects,
            ) as mock_parse,
            patch("time.sleep") as mock_sleep,
        ):
            result, _meta = provider.extract_structured("test prompt", _SimpleSchema)

        assert isinstance(result, _SimpleSchema)
        assert mock_parse.call_count == 2
        mock_sleep.assert_called_once_with(5.0)

    def test_fails_after_three_attempts(self) -> None:
        """Provider raises RuntimeError after 3 consecutive APIConnectionError failures."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()

        with (
            patch.object(
                provider.client.beta.chat.completions,
                "parse",
                side_effect=openai.APIConnectionError(request=MagicMock()),
            ) as mock_parse,
            patch("time.sleep"),
            pytest.raises(RuntimeError, match="failed after 3 attempts"),
        ):
            provider.extract_structured("test prompt", _SimpleSchema)

        assert mock_parse.call_count == 3

    def test_parsed_none_raises_value_error(self) -> None:
        """msg.parsed=None with no refusal must raise ValueError immediately, not be retried."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider()

        # Build response where refusal is None but parsed is also None
        msg = MagicMock()
        msg.refusal = None
        msg.parsed = None
        choice = MagicMock()
        choice.message = msg
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        mock_response = MagicMock()
        mock_response.choices = [choice]
        mock_response.usage = usage

        with (
            patch.object(
                provider.client.beta.chat.completions,
                "parse",
                return_value=mock_response,
            ) as mock_parse,
            pytest.raises(LLMProviderError, match="None parsed result"),
        ):
            provider.extract_structured("test prompt", _SimpleSchema)

        assert mock_parse.call_count == 1

    def test_refusal_raises_immediately_no_retry(self) -> None:
        """Refusal must propagate at first call, not retry 3x then raise RuntimeError."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        mock_response = _make_openai_mock_response(refusal="I cannot help with this")

        with (
            patch.object(
                provider.client.beta.chat.completions,
                "parse",
                return_value=mock_response,
            ) as mock_parse,
            pytest.raises(LLMProviderError, match="OpenAI refused"),
        ):
            provider.extract_structured("prompt", _SimpleSchema, max_tokens=100)

        assert mock_parse.call_count == 1, (
            f"Refusal must not be retried — expected 1 call, got {mock_parse.call_count}"
        )

    def test_parsed_none_raises_immediately_no_retry(self) -> None:
        """Parsed=None must propagate at first call, not retry."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")

        msg = MagicMock()
        msg.refusal = None
        msg.parsed = None
        choice = MagicMock()
        choice.message = msg
        usage = MagicMock(prompt_tokens=10, completion_tokens=0)
        response = MagicMock(choices=[choice], usage=usage)

        with (
            patch.object(
                provider.client.beta.chat.completions,
                "parse",
                return_value=response,
            ) as mock_parse,
            pytest.raises(LLMProviderError, match="None parsed result"),
        ):
            provider.extract_structured("prompt", _SimpleSchema, max_tokens=100)

        assert mock_parse.call_count == 1


class TestOpenAIProviderEmbed:
    def _make_embeddings_response(
        self, vectors: list[list[float]], total_tokens: int = 100
    ) -> MagicMock:
        """Build a mock openai.embeddings.create response."""
        response = MagicMock()
        response.data = [MagicMock(embedding=vec) for vec in vectors]
        response.usage = MagicMock(total_tokens=total_tokens)
        return response

    def test_embed_returns_correct_shape(self) -> None:
        """embed() returns (vectors, meta) — vectors has one entry per input text."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        vectors = [[float(i)] * 1536 for i in range(3)]
        mock_response = self._make_embeddings_response(vectors)

        with patch.object(provider.client.embeddings, "create", return_value=mock_response):
            result, meta = provider.embed(["text one", "text two", "text three"])

        assert len(result) == 3
        assert all(len(v) == 1536 for v in result)
        assert "cost_usd" in meta
        assert "input_tokens" in meta
        assert "model" in meta

    def test_embed_empty_input_returns_empty_list(self) -> None:
        """embed([]) must return ([], meta) without making any API call."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")

        with patch.object(provider.client.embeddings, "create") as mock_create:
            result, meta = provider.embed([])

        assert result == []
        assert meta["cost_usd"] == 0.0
        mock_create.assert_not_called()

    def test_embed_passes_correct_model(self) -> None:
        """embed() must call embeddings.create with model='text-embedding-3-small'."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        mock_response = self._make_embeddings_response([[0.1, 0.2]])

        with patch.object(
            provider.client.embeddings, "create", return_value=mock_response
        ) as mock_create:
            provider.embed(["hello world"])

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("model") == "text-embedding-3-small"

    def test_embed_cost_computed_correctly(self) -> None:
        """embed() meta['cost_usd'] = total_tokens * 0.02 / 1_000_000."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="test-key")
        mock_response = self._make_embeddings_response([[0.1, 0.2]], total_tokens=1_000_000)

        with patch.object(provider.client.embeddings, "create", return_value=mock_response):
            _result, meta = provider.embed(["hello"])

        assert meta["cost_usd"] == pytest.approx(0.02)


class TestEstimateRunCostUsd:
    """Pre-flight reservation must scale with the configured model pricing."""

    def test_mini_run_cost_under_one_cent(self) -> None:
        """gpt-4o-mini round trip should reserve well under a cent."""
        cost = estimate_run_cost_usd("gpt-4o-mini", "gpt-4o-mini")
        assert 0 < cost < 0.01

    def test_gpt4o_costs_at_least_an_order_of_magnitude_more_than_mini(self) -> None:
        """gpt-4o pricing is ~17× gpt-4o-mini; reservation must reflect that."""
        mini = estimate_run_cost_usd("gpt-4o-mini", "gpt-4o-mini")
        full = estimate_run_cost_usd("gpt-4o", "gpt-4o")
        assert full >= mini * 10

    def test_unknown_model_falls_back_to_mini(self) -> None:
        """Unknown model name must not crash — falls back to gpt-4o-mini pricing."""
        cost = estimate_run_cost_usd("future-model-x", "future-model-y")
        mini_cost = estimate_run_cost_usd("gpt-4o-mini", "gpt-4o-mini")
        assert cost == pytest.approx(mini_cost)
