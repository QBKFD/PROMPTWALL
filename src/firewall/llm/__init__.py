"""LLM backends and the factory that picks one from settings."""

from __future__ import annotations

from firewall.config import Settings
from firewall.llm.base import LLMClient, LLMError, LLMResponse
from firewall.llm.budget import Budget, BudgetExceeded

__all__ = ["Budget", "BudgetExceeded", "LLMClient", "LLMError", "LLMResponse", "build_llm"]


def build_llm(settings: Settings, budget: Budget | None = None) -> LLMClient:
    if settings.llm_backend == "anthropic":
        from firewall.llm.anthropic_client import AnthropicClient

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicClient(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            effort=settings.anthropic_effort,
            api_key=key,
            timeout_s=settings.llm_timeout_s,
            budget=budget,
        )
    if settings.llm_backend == "ollama":
        from firewall.llm.ollama_client import OllamaClient

        return OllamaClient(
            model=settings.ollama_model,
            max_tokens=settings.llm_max_tokens,
            base_url=settings.ollama_url,
            timeout_s=settings.llm_timeout_s,
            budget=budget,
        )
    from firewall.llm.fake import FakeLLM

    return FakeLLM(max_tokens=settings.llm_max_tokens, budget=budget)
