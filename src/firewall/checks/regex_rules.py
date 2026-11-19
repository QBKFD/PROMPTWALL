"""Rule-based detection of secrets and PII with regular expressions.

secrets_regex: known credential formats (AWS, GitHub, Anthropic, OpenAI, Slack,
  Google, Stripe, SendGrid, Square, Twilio, JWT, PEM private keys) plus
  "key = value" assignments to secret-named variables.
pii_regex: e-mail addresses, IBANs (country length + ISO 7064 mod-97 checksum),
  phone numbers (digit-count and shape filters to cut false positives).

Both report character spans, so the policy can mask instead of block.
Findings carry the type and span only, never the matched value.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from firewall.checks.base import Check, CheckContext, register
from firewall.schemas import Finding


@dataclass(frozen=True)
class Rule:
    type: str
    pattern: re.Pattern[str]
    group: int = 0  # which regex group is the sensitive span
    # returns the (start, end) to report, relative to the text, or None to reject
    validate: Callable[[str, int, int], tuple[int, int] | None] | None = None


def _resolve_overlaps(cands: list[tuple[int, int, int, str]]) -> list[Finding]:
    """cands: (start, end, priority, type). Keep non-overlapping spans, preferring
    lower priority number (earlier rule), then longer span."""
    chosen: list[tuple[int, int, str]] = []
    for start, end, _, typ in sorted(cands, key=lambda c: (c[2], -(c[1] - c[0]), c[0])):
        if all(end <= s or start >= e for s, e, _ in chosen):
            chosen.append((start, end, typ))
    return [Finding(type=t, start=s, end=e) for s, e, t in sorted(chosen)]


def _find(rules: list[Rule], text: str) -> list[Finding]:
    cands = []
    for prio, rule in enumerate(rules):
        for m in rule.pattern.finditer(text):
            start, end = m.span(rule.group)
            if start < 0 or start == end:
                continue
            if rule.validate is not None:
                span = rule.validate(text, start, end)
                if span is None:
                    continue
                start, end = span
            cands.append((start, end, prio, rule.type))
    return _resolve_overlaps(cands)


# --------------------------------------------------------------------------- secrets

_SECRET_KEYWORDS = (
    r"api[\s_-]?(?:key|schl[üu]ssel|nyckel|sleutel)|apikey|(?:cl[ée]|clave|chiave)\s+(?:d['’])?api|"
    r"secret(?:[_-]?key)?|client[_-]?secret|access[_-]?token|auth[_-]?token|bearer[_-]?token|"
    r"private[_-]?key|token|password|passwd|pwd"
)


def _token_shaped(text: str, start: int, end: int) -> tuple[int, int] | None:
    """An assigned value counts as a secret only if it mixes letters and digits,
    so 'password: required' or 'token = placeholder' do not match."""
    v = text[start:end]
    if any(c.isdigit() for c in v) and any(c.isalpha() for c in v):
        return start, end
    return None


SECRET_RULES: list[Rule] = [
    Rule(
        "private_key",
        re.compile(
            r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----(?:[\s\S]*?-----END (?:[A-Z]+ )*PRIVATE KEY-----)?"
        ),
    ),
    Rule("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    Rule("openai_key", re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}")),
    Rule("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    Rule(
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    ),
    Rule("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    Rule("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    Rule("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    Rule("sendgrid_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,32}\.[A-Za-z0-9_-]{16,64}")),
    Rule("square_token", re.compile(r"\bsq0(?:atp|csp)-[A-Za-z0-9_-]{22,43}")),
    Rule("twilio_key", re.compile(r"\bSK[0-9a-fA-F]{32}\b")),
    Rule("dropbox_token", re.compile(r"\bsl\.[A-Za-z0-9_-]{20,}")),
    Rule("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    # "key = value", "key: value", "api key is value"; the value (group 1) is the secret
    Rule(
        "secret_assignment",
        re.compile(
            rf"(?i)(?<![A-Za-z0-9])(?:{_SECRET_KEYWORDS})\b[\"']?(?:\s*[:=]\s*|\s+is\s+)[\"']?"
            r"([^\s\"',;]{8,})"
        ),
        group=1,
        validate=_token_shaped,
    ),
]


# --------------------------------------------------------------------------- PII

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9](?:[A-Za-z0-9._%+-]{0,62}[A-Za-z0-9_%+-])?"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}\b"
)

# Official IBAN lengths per country (SWIFT IBAN registry).
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22,
    "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24,
    "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SC": 31,
    "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24,
    "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}  # fmt: skip

IBAN_CANDIDATE = re.compile(r"\b([A-Z]{2})(\d{2})((?:[ ]?[A-Z0-9]){10,30})")


def iban_checksum_ok(iban: str) -> bool:
    """ISO 7064 mod 97-10: move the first 4 chars to the end, letters -> 10..35."""
    s = iban[4:] + iban[:4]
    digits = "".join(str(int(c, 36)) for c in s)
    return int(digits) % 97 == 1


def _iban_validator(require_checksum: bool) -> Callable[[str, int, int], tuple[int, int] | None]:
    def validate(text: str, start: int, end: int) -> tuple[int, int] | None:
        country = text[start : start + 2]
        length = IBAN_LENGTHS.get(country)
        if length is None:
            return None
        # walk the match, collecting non-space chars until the country length is reached
        compact, pos = [], start
        while pos < end and len(compact) < length:
            if text[pos] != " ":
                compact.append(text[pos])
            pos += 1
        if len(compact) != length:
            return None
        # the IBAN must not continue as part of a longer alphanumeric token
        if pos < len(text) and text[pos].isalnum() and text[pos - 1] != " ":
            return None
        iban = "".join(compact)
        if require_checksum and not iban_checksum_ok(iban):
            return None
        return start, pos

    return validate


PHONE_CANDIDATE = re.compile(
    r"(?<![\w+\-./:=])"
    r"((?:\+|00)?[\[(]?\d[\d \-.()\[\]]{5,22}\d(?:\s?(?:x|ext\.?)\s?\d{1,5})?)"
    r"(?![\w\-/:]|\.\d)"
)
_REJECT_SHAPES = [
    re.compile(r"\d{4}-\d{1,2}-\d{1,2}"),  # ISO date (possibly followed by a time)
    re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$"),  # dd/mm/yyyy
    re.compile(r"^(?:19|20)\d{6}"),  # yyyymmdd...
    re.compile(r"^\d{6,8}-\d{4}$"),  # Nordic personal / organisation numbers
    re.compile(r"^[1-9]\d{0,2}(?:[ ,.]\d{3})+$"),  # thousands grouping: 1 000 000, 12.500.000
    re.compile(r"^\d{3}-\d{2}-\d{4}$"),  # US social security number
    re.compile(r"^\d{2}-\d{7}$"),  # US employer identification number
    re.compile(r"^(?:19|20)\d{2}\s?-\s?(?:19|20)\d{2}$"),  # year range 2023-2024
    re.compile(r"^\d{4}[ -]\d{4,6}[ -]\d{4,5}(?:[ -]\d{3,4})?$"),  # card-number grouping
]
_DOTTED_PHONE = re.compile(r"^\(?\d{2,4}\)?(?:\.\d{2,4}){2,4}$")  # 555.123.4567
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_SEP = re.compile(r"[ \-.()\[\]]")


def _phone_validator(text: str, start: int, end: int) -> tuple[int, int] | None:
    s = text[start:end]
    # brackets that do not pair up are surrounding punctuation, not part of the number
    if s.count("(") != s.count(")") or s.count("[") != s.count("]"):
        stripped = s.strip("()[]")
        start += s.find(stripped)
        end = start + len(stripped)
        s = stripped
    main = re.split(r"\s?(?:x|ext\.?)\s?(?=\d)", s)[0]
    digits = re.sub(r"\D", "", main)
    if not 7 <= len(digits) <= 15:
        return None
    if any(p.search(main) for p in _REJECT_SHAPES):
        return None
    # dots: only the 555.123.4567 shape (rules out decimals, IPs, coordinates)
    if "." in main and (not _DOTTED_PHONE.match(main) or _IPV4.match(main)):
        return None
    international = main.startswith("+") or re.match(r"00[1-9]", main) is not None
    # a bare digit run (no separators) is too often an account/reference number
    if not international and not _SEP.search(main):
        return None
    return start, end


def pii_rules(entities: list[str], iban_checksum: bool = True) -> list[Rule]:
    available = {
        "email": Rule("email", EMAIL),
        "iban": Rule("iban", IBAN_CANDIDATE, validate=_iban_validator(iban_checksum)),
        "phone": Rule("phone", PHONE_CANDIDATE, group=1, validate=_phone_validator),
    }
    unknown = set(entities) - set(available)
    if unknown:
        raise ValueError(f"unknown PII entities: {sorted(unknown)}")
    # order = priority when spans overlap: email and iban win over phone
    return [available[e] for e in ("email", "iban", "phone") if e in entities]


class SecretsCheck(Check):
    name = "secrets_regex"
    supports_mask = True

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules or SECRET_RULES

    def run(self, text: str, ctx: CheckContext) -> list[Finding]:
        return _find(self.rules, text)


class PIICheck(Check):
    name = "pii_regex"
    supports_mask = True

    def __init__(self, entities: list[str] | None = None, iban_checksum: bool = True) -> None:
        self.rules = pii_rules(entities or ["email", "iban", "phone"], iban_checksum)

    def run(self, text: str, ctx: CheckContext) -> list[Finding]:
        return _find(self.rules, text)


@register("secrets_regex")
def _secrets_factory(params: dict[str, Any]) -> Check:
    if params:
        raise ValueError(f"secrets_regex takes no params, got {sorted(params)}")
    return SecretsCheck()


@register("pii_regex")
def _pii_factory(params: dict[str, Any]) -> Check:
    return PIICheck(**params)
