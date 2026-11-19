"""Seeded synthetic set for the secrets rules: fake credentials in real provider
formats inside chat-like sentences, plus hard negatives (UUIDs, hashes, base64,
prose that mentions passwords or tokens).

This set is self-made by the same author as the rules, so it measures format
coverage only. It is a sanity check, not a benchmark.

Key-shaped strings are assembled at runtime so no literal secret-looking string
lives in the repository (keeps secret scanners quiet, and honest).
"""

from __future__ import annotations

import base64
import hashlib
import random
import string
import uuid

from eval.datasets import TextExample

ALNUM = string.ascii_letters + string.digits
UPPER_NUM = string.ascii_uppercase + string.digits
URLSAFE = ALNUM + "-_"
HEX = "0123456789abcdef"


def _r(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _b64url(rng: random.Random, n: int) -> str:
    return base64.urlsafe_b64encode(rng.randbytes(n)).decode().rstrip("=")


def fake_secret(rng: random.Random, kind: str) -> str:
    p = {  # prefixes split so the source never contains a full key-shaped literal
        "aws": "AK" + "IA",
        "gh": "gh" + "p_",
        "ant": "sk-" + "ant-",
        "oai": "sk-" + "proj-",
        "slack": "xo" + "xb-",
        "google": "AI" + "za",
        "stripe": "sk_" + "live_",
        "sg": "S" + "G.",
        "jwt": "ey" + "J",
    }
    if kind == "aws":
        return p["aws"] + _r(rng, UPPER_NUM, 16)
    if kind == "github":
        return p["gh"] + _r(rng, ALNUM, 36)
    if kind == "anthropic":
        return p["ant"] + "api03-" + _r(rng, URLSAFE, 93)
    if kind == "openai":
        return p["oai"] + _r(rng, URLSAFE, 48)
    if kind == "slack":
        return (
            p["slack"]
            + "-".join(_r(rng, string.digits, 12) for _ in range(2))
            + "-"
            + _r(rng, ALNUM, 24)
        )
    if kind == "google":
        return p["google"] + _r(rng, URLSAFE, 35)
    if kind == "stripe":
        return p["stripe"] + _r(rng, ALNUM, 24)
    if kind == "sendgrid":
        return p["sg"] + _r(rng, URLSAFE, 22) + "." + _r(rng, URLSAFE, 43)
    if kind == "jwt":
        head = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').decode().rstrip("=")
        body = (
            base64.urlsafe_b64encode(f'{{"sub":"{rng.randint(1, 10**6)}"}}'.encode())
            .decode()
            .rstrip("=")
        )
        assert head.startswith(p["jwt"]) and body.startswith(p["jwt"])
        return f"{head}.{body}.{_b64url(rng, 32)}"
    if kind == "pem":
        label = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
        end = "-----END " + "RSA PRIVATE" + " KEY-----"
        return f"{label}\n{base64.b64encode(rng.randbytes(48)).decode()}\n{end}"
    if kind == "password_assignment":
        return "password=" + _r(rng, ALNUM, 6) + str(rng.randint(10, 99)) + rng.choice("!#$%")
    raise ValueError(kind)


KINDS = [
    "aws", "github", "anthropic", "openai", "slack", "google",
    "stripe", "sendgrid", "jwt", "pem", "password_assignment",
]  # fmt: skip

POS_TEMPLATES = [
    "Here is my key {s}, why does the request fail?",
    "My config is:\n{s}\nWhat is wrong with it?",
    "Can you debug this? export TOKEN={s}",
    "I pasted the credential by accident: {s}",
    "headers = {{'Authorization': 'Bearer {s}'}}",
    "Use {s} to call the API and summarise the result.",
]

NEG_TEMPLATES = [
    "The request id was {n}, can you look it up?",
    "Commit {n} broke the build.",
    "The file checksum is {n}.",
    "Here is the base64 blob: {n}",
    "My order number is {n}.",
    "Please reset my password, I forgot it.",
    "What is a good password policy for a small company?",
    "The access token expires after one hour; how do I refresh it?",
    "Set the api_key variable before running the script.",
    "The password field is required and must be at least 12 characters.",
    "Use a token bucket for rate limiting.",
    "Version {n} of the library fixed the bug.",
]


def fake_non_secret(rng: random.Random) -> str:
    kind = rng.choice(["uuid", "sha256", "sha1", "b64", "order", "version"])
    if kind == "uuid":
        return str(uuid.UUID(int=rng.getrandbits(128)))
    if kind == "sha256":
        return hashlib.sha256(rng.randbytes(16)).hexdigest()
    if kind == "sha1":
        return hashlib.sha1(rng.randbytes(16)).hexdigest()
    if kind == "b64":
        return base64.b64encode(rng.randbytes(24)).decode()
    if kind == "order":
        return f"ORD-{rng.randint(10**7, 10**8)}"
    return f"{rng.randint(0, 9)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}"


def build(n_pos_per_kind: int = 40, n_neg: int = 440, seed: int = 7) -> list[TextExample]:
    rng = random.Random(seed)
    out = []
    for kind in KINDS:
        for _ in range(n_pos_per_kind):
            out.append(TextExample(rng.choice(POS_TEMPLATES).format(s=fake_secret(rng, kind)), 1))
    for _ in range(n_neg):
        out.append(TextExample(rng.choice(NEG_TEMPLATES).format(n=fake_non_secret(rng)), 0))
    rng.shuffle(out)
    return out
