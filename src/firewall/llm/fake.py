"""Deterministic offline LLM for tests and latency measurement. Never makes network calls."""

from __future__ import annotations

from collections.abc import Callable

from firewall.llm.base import LLMClient, LLMError, LLMResponse
from firewall.llm.budget import Budget


class FakeLLM(LLMClient):
    backend = "fake"

    def __init__(
        self,
        reply: str | Callable[[str, str], str] = "fake answer",
        max_tokens: int = 256,
        budget: Budget | None = None,
        fail: bool = False,
    ) -> None:
        super().__init__(max_tokens=max_tokens, budget=budget)
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, str]] = []  # (system, user) as received

    @property
    def model(self) -> str:
        return "fake"

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        self.calls.append((system, user))
        if self.fail:
            raise LLMError("fake backend configured to fail")
        text = self.reply(system, user) if callable(self.reply) else self.reply
        return LLMResponse(
            text=text,
            model="fake",
            input_tokens=len(system + user) // 4,
            output_tokens=len(text) // 4,
            stop_reason="end_turn",
        )
