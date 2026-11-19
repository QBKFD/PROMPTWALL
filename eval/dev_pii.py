"""Development loop for the PII/secret regexes. Uses ONLY a sample of the Gretel
*train* split, never the test split. Prints metrics and example errors.

    uv run python -m eval.dev_pii [--n 5000] [--no-checksum]
"""

from __future__ import annotations

import argparse
import json
import random

from eval.datasets import gretel
from eval.span_eval import score_spans
from firewall.checks.base import CheckContext
from firewall.checks.regex_rules import PIICheck, SecretsCheck
from firewall.schemas import Stage

PII_MAP = {"email": {"email"}, "iban": {"iban"}, "phone": {"phone_number"}}
CTX = CheckContext(stage=Stage.INPUT, user_message="")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-checksum", action="store_true")
    args = ap.parse_args()

    docs = gretel("train")
    docs = random.Random(args.seed).sample(docs, args.n)
    pii = PIICheck(iban_checksum=not args.no_checksum)
    sec = SecretsCheck()

    def detect_pii(text):
        return [(f.start, f.end, f.type) for f in pii.run(text, CTX)]

    def detect_secret(text):
        return [(f.start, f.end, "secret") for f in sec.run(text, CTX)]

    for name, det, tmap in (
        ("PII", detect_pii, PII_MAP),
        ("SECRETS vs gold api_key", detect_secret, {"secret": {"api_key"}}),
    ):
        res = score_spans(docs, det, tmap, n_boot=200)
        print(f"===== {name} (train sample n={len(docs)})")
        for typ, r in res.items():
            if typ == "any_document_level":
                continue
            ex = r.pop("_examples")
            print(
                typ, json.dumps({k: r[k] for k in ("gold_spans", "pred_spans", "overlap", "exact")})
            )
            print("   FP overlaps:", r["false_positives_overlap_gold_label"])
            for e in ex["fp"][:10]:
                print("   FP", repr(e[1]), e[2])
            for e in ex["fn"][:10]:
                print("   FN", repr(e[1]))


if __name__ == "__main__":
    main()
