"""Policy config: which checks run at each stage, and what to do when one triggers.

Example (config/policy.yaml):

    version: 1
    input:
      stop_on_block: false
      checks:
        - check: secrets_regex
          action: block      # block | mask | log
          on_error: block    # action if the check itself crashes (fail closed)
          params: {}
    output:
      stop_on_block: true
      checks: []
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from firewall.checks.base import Check, get_factory
from firewall.schemas import Action


class CheckRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check: str
    enabled: bool = True
    action: Literal[Action.BLOCK, Action.MASK, Action.LOG]
    on_error: Literal[Action.BLOCK, Action.LOG] = Action.BLOCK
    params: dict[str, Any] = Field(default_factory=dict)


class StagePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop_on_block: bool = True
    checks: list[CheckRule] = Field(default_factory=list)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    input: StagePolicy = Field(default_factory=StagePolicy)
    output: StagePolicy = Field(default_factory=StagePolicy)
    input_block_message: str = "Your message was blocked by the safety policy."
    output_block_message: str = "The answer was withheld by the safety policy."


@dataclass
class ConfiguredCheck:
    check: Check
    rule: CheckRule


@dataclass
class Policy:
    config: PolicyConfig
    input_checks: list[ConfiguredCheck] = field(default_factory=list)
    output_checks: list[ConfiguredCheck] = field(default_factory=list)


def _build_stage(stage: StagePolicy, stage_name: str) -> list[ConfiguredCheck]:
    built = []
    for rule in stage.checks:
        if not rule.enabled:
            continue
        check = get_factory(rule.check)(rule.params)
        if rule.action == Action.MASK and not check.supports_mask:
            raise ValueError(
                f"{stage_name} check {rule.check!r} cannot mask (it does not report spans); "
                "use block or log"
            )
        built.append(ConfiguredCheck(check=check, rule=rule))
    return built


def build_policy(config: PolicyConfig) -> Policy:
    return Policy(
        config=config,
        input_checks=_build_stage(config.input, "input"),
        output_checks=_build_stage(config.output, "output"),
    )


def load_policy(path: str | Path) -> Policy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return build_policy(PolicyConfig.model_validate(raw))
