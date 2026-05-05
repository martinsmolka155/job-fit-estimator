"""Application configuration loaded from environment variables."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    provider: Literal["openai"] = "openai"
    parser_model: str = "gpt-4o-mini"
    explainer_model: str = "gpt-4o-mini"  # Phase 22: reverted from gpt-5-mini after reasoning-token + slowness issues; salary research stays on gpt-5-mini where reasoning helps
    salary_research_model: str = "gpt-5-mini"  # NEW (Phase 22)
    salary_research_fallback: str = "gpt-5"  # NEW (Phase 22)
    embedding_model: str = "text-embedding-3-small"  # already used in OpenAIProvider.embed
    tavily_search_depth: str = "basic"  # NEW (1 credit per call; "advanced" = 2)
    tavily_api_key: str = ""  # NEW (read from TAVILY_API_KEY env)


settings = Settings()
