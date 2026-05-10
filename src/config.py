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
    explainer_model: str = "gpt-4o-mini"
    daily_api_budget_usd: float = 5.00


settings = Settings()
