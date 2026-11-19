"""The request flow: input checks -> LLM -> output checks, producing a decision record."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from firewall.checks.base import CheckContext
from firewall.llm.base import LLMClient
from firewall.policy import ConfiguredCheck, Policy
from firewall.schemas import Action, CheckRecord, Decision, Finding, LLMRecord, Stage


@dataclass
class StageResult:
    records: list[CheckRecord]
    action: Action
    text: str  # text after masking (unchanged unless a check masked)


def mask_spans(text: str, findings: list[Finding]) -> str:
    """Replace each finding's span with [TYPE]. Overlapping spans are merged first;
    a merged span takes the type of the finding that starts first."""
    spans = sorted(
        ((f.start, f.end, f.type) for f in findings if f.has_span),
        key=lambda s: (s[0], -s[1]),
    )
    merged: list[list] = []
    for start, end, typ in spans:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, typ])
    out, pos = [], 0
    for start, end, typ in merged:
        out.append(text[pos:start])
        out.append(f"[{typ.upper()}]")
        pos = end
    out.append(text[pos:])
    return "".join(out)


def run_stage(
    checks: list[ConfiguredCheck], text: str, ctx: CheckContext, stop_on_block: bool
) -> StageResult:
    records: list[CheckRecord] = []
    to_mask: list[Finding] = []
    blocked = False

    for cc in checks:
        rule = cc.rule
        if blocked and stop_on_block:
            records.append(
                CheckRecord(
                    name=rule.check,
                    stage=ctx.stage,
                    ran=False,
                    skipped_reason="an earlier check blocked",
                    configured_action=rule.action,
                )
            )
            continue

        start = time.perf_counter()
        error = None
        try:
            findings = cc.check.run(text, ctx)
        except Exception as e:  # a crashing check must not crash the request
            findings = []
            error = f"{type(e).__name__}: {e}"
        latency_ms = (time.perf_counter() - start) * 1000

        if error is not None:
            taken = rule.on_error
        elif not findings:
            taken = Action.ALLOW
        elif rule.action == Action.MASK and not all(f.has_span for f in findings):
            taken = Action.BLOCK  # cannot mask what we cannot locate: fail closed
        else:
            taken = rule.action

        if taken == Action.MASK:
            to_mask.extend(findings)
        if taken == Action.BLOCK:
            blocked = True

        records.append(
            CheckRecord(
                name=rule.check,
                stage=ctx.stage,
                ran=True,
                configured_action=rule.action,
                triggered=bool(findings),
                findings=findings,
                action_taken=taken,
                error=error,
                latency_ms=round(latency_ms, 3),
            )
        )

    action = Action.strongest([r.action_taken for r in records])
    masked = mask_spans(text, to_mask) if to_mask else text
    return StageResult(records=records, action=action, text=masked)


def _skipped(checks: list[ConfiguredCheck], stage: Stage, reason: str) -> list[CheckRecord]:
    return [
        CheckRecord(
            name=cc.rule.check,
            stage=stage,
            ran=False,
            skipped_reason=reason,
            configured_action=cc.rule.action,
        )
        for cc in checks
    ]


@dataclass
class ChatResult:
    answer: str
    decision: Decision


class Firewall:
    def __init__(self, policy: Policy, llm: LLMClient, system_prompt: str) -> None:
        self.policy = policy
        self.llm = llm
        self.system_prompt = system_prompt

    def handle(self, message: str, request_id: str | None = None) -> ChatResult:
        """Run one request. LLMError and BudgetExceeded propagate to the caller."""
        request_id = request_id or uuid.uuid4().hex
        cfg = self.policy.config
        t0 = time.perf_counter()

        inp = run_stage(
            self.policy.input_checks,
            message,
            CheckContext(stage=Stage.INPUT, user_message=message),
            cfg.input.stop_on_block,
        )
        llm_record = LLMRecord(called=False, backend=self.llm.backend, model=self.llm.model)

        if inp.action == Action.BLOCK:
            answer = cfg.input_block_message
            out_records = _skipped(self.policy.output_checks, Stage.OUTPUT, "input was blocked")
            out_action = Action.ALLOW
        else:
            resp = self.llm.complete(self.system_prompt, inp.text)
            llm_record = LLMRecord(
                called=True,
                backend=self.llm.backend,
                model=resp.model,
                latency_ms=round(resp.latency_ms, 3),
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                stop_reason=resp.stop_reason,
            )
            out = run_stage(
                self.policy.output_checks,
                resp.text,
                CheckContext(stage=Stage.OUTPUT, user_message=inp.text),
                cfg.output.stop_on_block,
            )
            out_records, out_action = out.records, out.action
            answer = cfg.output_block_message if out_action == Action.BLOCK else out.text

        total_ms = (time.perf_counter() - t0) * 1000
        decision = Decision(
            request_id=request_id,
            final_action=Action.strongest([inp.action, out_action]),
            input_action=inp.action,
            output_action=out_action,
            input_checks=inp.records,
            output_checks=out_records,
            llm=llm_record,
            total_ms=round(total_ms, 3),
            overhead_ms=round(total_ms - llm_record.latency_ms, 3),
        )
        return ChatResult(answer=answer, decision=decision)
