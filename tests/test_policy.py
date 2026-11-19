from pathlib import Path

import pytest
from pydantic import ValidationError

from firewall.policy import load_policy

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path, body: str) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(body)
    return p


def test_shipped_policy_loads():
    policy = load_policy(REPO / "config" / "policy.yaml")
    assert policy.config.version == 1


def test_builds_checks_with_params_and_skips_disabled(tmp_path):
    p = write(
        tmp_path,
        """
input:
  checks:
    - {check: test_word, action: mask, params: {word: pin, label: pin}}
    - {check: test_flag, action: block, enabled: false}
output:
  checks:
    - {check: test_flag, action: log, on_error: log}
""",
    )
    policy = load_policy(p)
    assert [c.check.name for c in policy.input_checks] == ["word:pin"]
    assert policy.output_checks[0].rule.on_error == "log"


def test_unknown_check_is_an_error(tmp_path):
    p = write(tmp_path, "input:\n  checks:\n    - {check: nope, action: block}\n")
    with pytest.raises(ValueError, match="unknown check 'nope'"):
        load_policy(p)


def test_mask_requires_span_support(tmp_path):
    p = write(tmp_path, "input:\n  checks:\n    - {check: test_flag, action: mask}\n")
    with pytest.raises(ValueError, match="cannot mask"):
        load_policy(p)


@pytest.mark.parametrize(
    "body",
    [
        "input:\n  checks:\n    - {check: test_flag, action: allow}\n",  # allow is not an action
        "input:\n  checks:\n    - {check: test_flag, action: block, typo: 1}\n",  # unknown key
        "version: 2\n",
    ],
)
def test_invalid_config_rejected(tmp_path, body):
    with pytest.raises(ValidationError):
        load_policy(write(tmp_path, body))
