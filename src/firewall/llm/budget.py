"""Hard spending cap for LLM calls.

The cap is checked *before* a call is sent, using a worst-case reservation
(estimated input tokens + the call's max output tokens). A call that could push
usage past either limit is refused, so the limits are never exceeded, even if
the provider's token count differs from our estimate in our favour. After the
call, the reservation is replaced by the provider's reported usage.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised when a call would exceed the configured call or token cap."""


def estimate_input_tokens(*texts: str) -> int:
    """Deliberately pessimistic token estimate: 1 token per 2 characters, plus overhead.

    Typical English is ~4 characters per token, so this over-reserves; that is
    the point, because the cap must hold before we know the real count.
    """
    return sum(len(t) for t in texts) // 2 + 64


@dataclass
class BudgetSnapshot:
    calls_used: int
    tokens_used: int
    max_calls: int
    max_tokens: int


class Budget:
    def __init__(self, max_calls: int, max_tokens: int) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self._calls = 0
        self._tokens = 0  # committed (actual) + outstanding reservations
        self._lock = threading.Lock()

    def reserve(self, input_estimate: int, max_output_tokens: int) -> int:
        """Reserve capacity for one call. Returns the reserved token amount."""
        reservation = input_estimate + max_output_tokens
        with self._lock:
            if self._calls + 1 > self.max_calls:
                raise BudgetExceeded(
                    f"call cap reached ({self._calls}/{self.max_calls} calls used)"
                )
            if self._tokens + reservation > self.max_tokens:
                raise BudgetExceeded(
                    f"token cap would be exceeded ({self._tokens} used/reserved + "
                    f"{reservation} for this call > {self.max_tokens})"
                )
            self._calls += 1
            self._tokens += reservation
        return reservation

    def settle(self, reservation: int, actual_tokens: int | None) -> None:
        """Replace a reservation with the actual usage (keep it if usage is unknown)."""
        if actual_tokens is None:
            return
        with self._lock:
            self._tokens += actual_tokens - reservation

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(self._calls, self._tokens, self.max_calls, self.max_tokens)
