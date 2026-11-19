"""The interface every input or output check implements, plus a name -> factory registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from firewall.schemas import Finding, Stage


@dataclass(frozen=True)
class CheckContext:
    stage: Stage
    user_message: str  # the message as sent to the LLM (i.e. after input masking)


class Check(ABC):
    """A check inspects text and returns findings. It never decides the action;
    the policy does. A check that can locate what it found (character spans)
    sets supports_mask so the policy may mask instead of block."""

    name: str = "check"
    supports_mask: bool = False

    @abstractmethod
    def run(self, text: str, ctx: CheckContext) -> list[Finding]: ...


CheckFactory = Callable[[dict[str, Any]], Check]
_REGISTRY: dict[str, CheckFactory] = {}


def register(name: str) -> Callable[[CheckFactory], CheckFactory]:
    def deco(factory: CheckFactory) -> CheckFactory:
        if name in _REGISTRY:
            raise ValueError(f"check {name!r} registered twice")
        _REGISTRY[name] = factory
        return factory

    return deco


def get_factory(name: str) -> CheckFactory:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"unknown check {name!r}; registered checks: {known}") from None


def registered_names() -> list[str]:
    return sorted(_REGISTRY)
