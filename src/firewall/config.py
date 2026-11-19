"""Runtime settings, read from environment variables and an optional .env file."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_backend: Literal["anthropic", "ollama", "fake"] = "anthropic"

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_timeout_s: float = Field(default=60.0, gt=0)
    system_prompt: str = "You are a helpful assistant. Answer concisely."

    budget_max_calls: int = Field(default=200, ge=0)
    budget_max_tokens: int = Field(default=400_000, ge=0)

    policy_path: str = "config/policy.yaml"

    max_message_chars: int = Field(default=20_000, gt=0)
