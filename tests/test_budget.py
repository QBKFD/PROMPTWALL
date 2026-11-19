import pytest

from firewall.llm.budget import Budget, BudgetExceeded, estimate_input_tokens
from firewall.llm.fake import FakeLLM


def test_call_cap_refuses_before_sending():
    llm = FakeLLM(budget=Budget(max_calls=2, max_tokens=10_000))
    llm.complete("sys", "a")
    llm.complete("sys", "b")
    with pytest.raises(BudgetExceeded, match="call cap"):
        llm.complete("sys", "c")
    assert len(llm.calls) == 2  # the third call never reached the backend


def test_token_cap_uses_worst_case_reservation():
    # reservation = estimate(input) + max_tokens; must fit under the cap
    est = estimate_input_tokens("sys", "hi")
    budget = Budget(max_calls=100, max_tokens=est + 256 - 1)
    llm = FakeLLM(max_tokens=256, budget=budget)
    with pytest.raises(BudgetExceeded, match="token cap"):
        llm.complete("sys", "hi")
    assert llm.calls == []


def test_settle_replaces_reservation_with_actual_usage():
    budget = Budget(max_calls=10, max_tokens=100_000)
    llm = FakeLLM(reply="x" * 40, max_tokens=500, budget=budget)
    llm.complete("system", "user text")
    snap = budget.snapshot()
    expected_actual = len("system" + "user text") // 4 + 40 // 4
    assert snap.calls_used == 1
    assert snap.tokens_used == expected_actual


def test_failed_call_keeps_full_reservation():
    budget = Budget(max_calls=10, max_tokens=100_000)
    llm = FakeLLM(max_tokens=500, budget=budget, fail=True)
    with pytest.raises(Exception, match="fail"):
        llm.complete("s", "u")
    snap = budget.snapshot()
    assert snap.calls_used == 1
    assert snap.tokens_used == estimate_input_tokens("s", "u") + 500


def test_estimate_is_pessimistic_relative_to_typical_english():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    # ~4 chars/token is typical; the estimate must be at least that
    assert estimate_input_tokens(text) >= len(text) / 4
