"""Data shapes for findings, decision records and the HTTP API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    ALLOW = "allow"
    LOG = "log"
    MASK = "mask"
    BLOCK = "block"

    @property
    def severity(self) -> int:
        return _SEVERITY[self]

    @staticmethod
    def strongest(actions: list[Action]) -> Action:
        return max(actions, key=lambda a: a.severity, default=Action.ALLOW)


_SEVERITY = {Action.ALLOW: 0, Action.LOG: 1, Action.MASK: 2, Action.BLOCK: 3}


class Stage(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class Finding(BaseModel):
    """One thing a check found. Never carries the matched value itself."""

    type: str
    start: int | None = None
    end: int | None = None
    score: float | None = None
    detail: str | None = None

    @property
    def has_span(self) -> bool:
        return self.start is not None and self.end is not None


class CheckRecord(BaseModel):
    name: str
    stage: Stage
    ran: bool
    skipped_reason: str | None = None
    configured_action: Action
    triggered: bool = False
    findings: list[Finding] = Field(default_factory=list)
    action_taken: Action = Action.ALLOW
    error: str | None = None
    latency_ms: float = 0.0


class LLMRecord(BaseModel):
    called: bool
    backend: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


class Decision(BaseModel):
    request_id: str
    final_action: Action
    input_action: Action
    output_action: Action
    input_checks: list[CheckRecord]
    output_checks: list[CheckRecord]
    llm: LLMRecord
    total_ms: float
    overhead_ms: float = Field(description="total_ms minus the main LLM call's latency")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    decision: Decision


class ErrorResponse(BaseModel):
    error: str
    request_id: str
