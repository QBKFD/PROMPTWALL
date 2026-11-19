"""Unit tests for the secrets and PII rules. Fake key-shaped strings are built at
runtime (see eval/synth_secrets.py) so none are stored in the repo."""

import random

import pytest

from eval.synth_secrets import KINDS, fake_secret
from firewall.checks.base import CheckContext
from firewall.checks.regex_rules import PIICheck, SecretsCheck, iban_checksum_ok
from firewall.pipeline import mask_spans
from firewall.schemas import Stage

CTX = CheckContext(stage=Stage.INPUT, user_message="")


def types(check, text):
    return [(f.type, text[f.start : f.end]) for f in check.run(text, CTX)]


@pytest.mark.parametrize("kind", KINDS)
def test_each_secret_format_is_found_and_fully_masked(kind):
    secret = fake_secret(random.Random(kind), kind)
    text = f"debug this for me: {secret} thanks"
    found = SecretsCheck().run(text, CTX)
    assert found, kind
    masked = mask_spans(text, found)
    # the sensitive part never survives masking
    core = secret.split("=", 1)[1] if kind == "password_assignment" else secret
    assert core not in masked
    assert masked.endswith("] thanks")
    prefix = (
        "debug this for me: password=[" if kind == "password_assignment" else "debug this for me: ["
    )
    assert masked.startswith(prefix)


@pytest.mark.parametrize(
    "text",
    [
        "Please reset my password, I forgot it.",
        "The password field is required.",
        "password: required",  # no digit -> not token-shaped
        "Use a token bucket for rate limiting.",
        "request id 550e8400-e29b-41d4-a716-446655440000",
        "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "Skills and SKUs are not keys.",
    ],
)
def test_secret_negatives(text):
    assert SecretsCheck().run(text, CTX) == []


def test_assignment_reports_only_the_value():
    text = "db_password = hunter2Secret9"
    [(typ, value)] = types(SecretsCheck(), text)
    assert typ == "secret_assignment"
    assert value == "hunter2Secret9"


# ----------------------------------------------------------------- PII


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mail me at jane.doe+work@example.co.uk.", "jane.doe+work@example.co.uk"),
        ("<a.b@mail.example.org>", "a.b@mail.example.org"),
    ],
)
def test_email(text, expected):
    assert types(PIICheck(entities=["email"]), text) == [("email", expected)]


def test_email_negatives():
    c = PIICheck(entities=["email"])
    assert c.run("ping @channel about the 2@3 ratio", CTX) == []


def test_iban_checksum():
    # The ISO 13616 standard example IBANs
    assert iban_checksum_ok("GB82WEST12345698765432")
    assert iban_checksum_ok("DE89370400440532013000")
    assert not iban_checksum_ok("GB82WEST12345698765433")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("IBAN: GB82WEST12345698765432.", "GB82WEST12345698765432"),
        ("IBAN DE89 3704 0044 0532 0130 00 please", "DE89 3704 0044 0532 0130 00"),
    ],
)
def test_iban_found_with_and_without_spaces(text, expected):
    assert types(PIICheck(entities=["iban"]), text) == [("iban", expected)]


def test_iban_rejects_bad_checksum_unless_disabled():
    text = "IBAN GB82WEST12345698765433"
    assert PIICheck(entities=["iban"]).run(text, CTX) == []
    assert PIICheck(entities=["iban"], iban_checksum=False).run(text, CTX)


@pytest.mark.parametrize(
    "number",
    ["+44 20 7946 0958", "(212) 555-0147", "+1-800-555-0199", "030 1234567",
     "0049 30 1234567", "555.123.4567", "+33 1 23 45 67 89", "555-0147 ext. 12"],
)  # fmt: skip
def test_phone_positives(number):
    text = f"call me on {number} tomorrow"
    assert types(PIICheck(entities=["phone"]), text) == [("phone", number)]


@pytest.mark.parametrize(
    "text",
    [
        "Meeting on 2024-03-15 10:30",
        "Born 15/03/1985",
        "Total 1 250 000 EUR",
        "Total 12.500.000 EUR",
        "Server at 192.168.10.12",
        "Location 52.520008, 13.404954",
        "Paid 5478000.00 today",
        "SSN 123-45-6789",
        "Org.nr 556894-3452",
        "Card 4111 1111 1111 1111",
        "Order 1234567890",
        "Fiscal year 2023-2024",
    ],
)
def test_phone_negatives(text):
    assert PIICheck(entities=["phone"]).run(text, CTX) == []


def test_iban_wins_over_phone_on_overlap():
    text = "pay to DE89 3704 0044 0532 0130 00"
    assert [t for t, _ in types(PIICheck(), text)] == ["iban"]


def test_pii_mask_end_to_end():
    text = "I'm jane@example.com, phone +44 20 7946 0958, IBAN GB82WEST12345698765432"
    masked = mask_spans(text, PIICheck().run(text, CTX))
    assert masked == "I'm [EMAIL], phone [PHONE], IBAN [IBAN]"


def test_unknown_entity_rejected():
    with pytest.raises(ValueError, match="unknown PII"):
        PIICheck(entities=["ssn"])
