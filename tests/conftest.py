"""Shared test fixtures. Tests never touch the network: sockets are disabled below."""

from __future__ import annotations

import re
import socket

import pytest

from firewall.checks.base import Check, CheckContext, register
from firewall.policy import CheckRule, ConfiguredCheck
from firewall.schemas import Finding


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


class WordCheck(Check):
    """Finds a fixed word and reports its spans (so it can mask)."""

    supports_mask = True

    def __init__(self, word: str = "secret", label: str = "word", name: str | None = None) -> None:
        self.word, self.label = word, label
        self.name = name or f"word:{word}"

    def run(self, text: str, ctx: CheckContext) -> list[Finding]:
        return [
            Finding(type=self.label, start=m.start(), end=m.end())
            for m in re.finditer(re.escape(self.word), text)
        ]


class FlagCheck(Check):
    """Flags the whole text if a word appears, without spans (like a classifier)."""

    def __init__(self, word: str = "ignore") -> None:
        self.word = word
        self.name = f"flag:{word}"

    def run(self, text: str, ctx: CheckContext) -> list[Finding]:
        return [Finding(type="flag", score=0.9)] if self.word in text else []


class CrashCheck(Check):
    name = "crash"

    def run(self, text: str, ctx: CheckContext) -> list[Finding]:
        raise ValueError("boom")


@register("test_word")
def _word_factory(params: dict) -> Check:
    return WordCheck(**params)


@register("test_flag")
def _flag_factory(params: dict) -> Check:
    return FlagCheck(**params)


def configured(check: Check, action: str, on_error: str = "block") -> ConfiguredCheck:
    return ConfiguredCheck(
        check=check, rule=CheckRule(check=check.name, action=action, on_error=on_error)
    )
