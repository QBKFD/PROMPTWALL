"""LLM client interface shared by all backends."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from firewall.llm.budget import Budget, estimate_input_tokens


class LLMError(RuntimeError):
    """The backend failed to produce an answer (network, API error, refusal...)."""


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    stop_reason: str | None = None


class LLMClient(ABC):
    """Calls an LLM. Every call goes through the budget, if one is set."""

    backend: str = "base"

    def __init__(self, max_tokens: int, budget: Budget | None = None) -> None:
        self.max_tokens = max_tokens
        self.budget = budget

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse: ...

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> LLMResponse:
        max_tokens = max_tokens or self.max_tokens
        reservation = 0
        if self.budget is not None:
            reservation = self.budget.reserve(estimate_input_tokens(system, user), max_tokens)
        start = time.perf_counter()
        # If this raises, the reservation is kept: a failed call may still be billed.
        response = self._complete(system, user, max_tokens)
        response.latency_ms = (time.perf_counter() - start) * 1000
        if self.budget is not None:
            actual = None
            if response.input_tokens is not None and response.output_tokens is not None:
                actual = response.input_tokens + response.output_tokens
            self.budget.settle(reservation, actual)
        return response
