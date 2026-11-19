"""Evaluate the input controls on held-out test sets and write results/results.json + .md.

    uv run python -m eval.run_eval

Test sets (see data/manifest.yaml for pinned revisions):
  injection classifier: deepset test, jackhhao jailbreak test, Lakera gandalf test
  PII rules:            Gretel synthetic PII finance, test split
  secrets rules:        Gretel test (api_key labels), synthetic self-made set
None of these were used for training, model selection or threshold choice.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from collections import defaultdict

import yaml

from eval import synth_secrets
from eval.datasets import ROOT, deepset, gandalf_test, gretel, jackhhao_test, overlap_with
from eval.metrics import binary_report
from eval.span_eval import score_spans
from firewall.checks.base import CheckContext
from firewall.checks.injection import (
    DEFAULT_MODEL,
    DEFAULT_TRANSFORMER,
    InjectionClassifierCheck,
)
from firewall.checks.regex_rules import PIICheck, SecretsCheck
from firewall.schemas import Stage

RESULTS = ROOT / "results"
CTX = CheckContext(stage=Stage.INPUT, user_message="")
PII_MAP = {"email": {"email"}, "iban": {"iban"}, "phone": {"phone_number"}}


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        return sha + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


TEST_SETS = {
    "deepset_test": "prompt injection vs benign (in-distribution)",
    "jackhhao_jailbreak_test": "jailbreak vs benign (out-of-distribution)",
    "lakera_gandalf_test": "injections only -> recall only (out-of-distribution)",
}


def _load_test_sets() -> dict:
    return {
        "deepset_test": deepset("test"),
        "jackhhao_jailbreak_test": jackhhao_test(),
        "lakera_gandalf_test": gandalf_test(),
    }


def _classifiers() -> dict:
    """The TF-IDF model, and the transformer if it was built (needs the extra)."""
    meta = json.loads((ROOT / DEFAULT_MODEL).with_suffix(".json").read_text())
    out = {
        "tfidf": (
            InjectionClassifierCheck.from_path(ROOT / DEFAULT_MODEL),
            {
                k: meta[k]
                for k in ("model", "selected", "threshold", "cv_f1_at_threshold", "sha256")
            },
        )
    }
    tdir = ROOT / DEFAULT_TRANSFORMER
    if (tdir / "firewall_meta.json").exists():
        from firewall.checks.injection import load_transformer_check

        tmeta = json.loads((tdir / "firewall_meta.json").read_text())
        cv = json.loads((RESULTS / "transformer_cv.json").read_text())
        tmeta["cv_mean_f1_at_0.5"] = cv["cv_mean_f1_at_0.5"]
        tmeta["sha256_weights"] = hashlib.sha256(
            (tdir / "model.safetensors").read_bytes()
        ).hexdigest()
        out["transformer"] = (load_transformer_check(tdir), tmeta)
    return out


def eval_injection(n_boot: int) -> dict:
    train = deepset("train")
    sets = _load_test_sets()
    overlaps = {name: overlap_with(train, data) for name, data in sets.items()}
    out = {"selected": None, "models": {}}
    for model_name, (clf, meta) in _classifiers().items():
        res = {"model": meta}
        for name, data in sets.items():
            exact, near = overlaps[name]
            kept = [e for i, e in enumerate(data) if i not in exact | near]
            scores = clf.score([e.text for e in kept])
            y_true = [e.label for e in kept]
            y_pred = [s >= clf.threshold for s in scores]
            rep = binary_report(y_true, y_pred, n_boot=n_boot)
            rep.update(
                description=TEST_SETS[name],
                n_original=len(data),
                dropped_train_duplicates={"exact": len(exact), "near_jaccard>=0.9": len(near)},
            )
            if rep["negatives"] == 0:
                for k in ("precision", "fpr", "f1"):
                    rep[k] = None
                    rep["ci95"][k] = None
                rep["wilson95"]["precision"] = rep["wilson95"]["fpr"] = None
                rep["note"] = "no benign examples: only recall is meaningful"
            res[name] = rep
        out["models"][model_name] = res
    # pre-registered rule (docs/decisions.md #7), decided on CV before test scores
    tfidf_cv = out["models"]["tfidf"]["model"]["selected"]["cv_f1_at_0.5"]
    cv_path = RESULTS / "transformer_cv.json"
    trans_cv = json.loads(cv_path.read_text())["cv_mean_f1_at_0.5"] if cv_path.exists() else None
    out["cv_f1_at_0.5"] = {"tfidf": tfidf_cv, "transformer": trans_cv}
    out["selected"] = (
        "transformer" if trans_cv is not None and trans_cv >= tfidf_cv + 0.03 else "tfidf"
    )
    out["selection_rule"] = "transformer if CV mean F1 >= TF-IDF CV F1 + 0.03 (decided on CV only)"
    return out


def eval_rules(n_boot: int) -> dict:
    docs = gretel("test")
    pii, sec = PIICheck(), SecretsCheck()

    def detect_pii(text):
        return [(f.start, f.end, f.type) for f in pii.run(text, CTX)]

    def detect_secret(text):
        return [(f.start, f.end, "secret") for f in sec.run(text, CTX)]

    pii_res = score_spans(docs, detect_pii, PII_MAP, n_boot=n_boot)
    # per-language span F1 (overlap) for each entity
    by_lang = defaultdict(list)
    for d in docs:
        by_lang[d.language].append(d)
    per_lang = {}
    for lang, ds in sorted(by_lang.items()):
        r = score_spans(ds, detect_pii, PII_MAP, n_boot=0)
        per_lang[lang] = {
            t: {
                "gold": r[t]["gold_spans"],
                **{k: r[t]["overlap"][k] for k in ("precision", "recall", "f1")},
            }
            for t in PII_MAP
        }
    pii_res["per_language_overlap"] = per_lang

    sec_api = score_spans(docs, detect_secret, {"secret": {"api_key"}}, n_boot=n_boot)
    sec_cred = score_spans(docs, detect_secret, {"secret": {"api_key", "password"}}, n_boot=n_boot)
    synth = synth_secrets.build()
    synth_rep = binary_report(
        [e.label for e in synth], [bool(sec.run(e.text, CTX)) for e in synth], n_boot=n_boot
    )
    synth_rep["note"] = (
        "self-made by the rules' author: format-coverage sanity check, not a benchmark"
    )

    for r in (pii_res, sec_api, sec_cred):
        for v in r.values():
            if isinstance(v, dict):
                v.pop("_examples", None)
    return {
        "dataset": f"gretel_pii test split, n={len(docs)} documents",
        "pii_regex": pii_res,
        "secrets_regex": {
            "gretel_vs_api_key": sec_api["secret"],
            "gretel_vs_api_key_or_password": sec_cred["secret"],
            "synthetic": synth_rep,
        },
    }


def fmt(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def ci(c) -> str:
    return "–" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"


def to_markdown(res: dict) -> str:
    lines = [f"# Evaluation results\n\nGenerated {res['meta']['generated_utc']} from commit "
             f"`{res['meta']['git_commit']}`. 95 % CIs: percentile bootstrap, "
             f"{res['meta']['n_boot']} resamples, seed 0.\n"]  # fmt: skip
    inj = res["injection_classifier"]
    lines += [
        "## Prompt-injection classifiers\n",
        "Both trained only on deepset/prompt-injections train (546 rows) and selected by "
        "5-fold CV on train. Selection rule fixed before any test score was computed: "
        f"{inj['selection_rule']}. Selected: **{inj['selected']}**.\n",
    ]
    for mname, mres in inj["models"].items():
        m = mres["model"]
        if mname == "tfidf":
            lines.append(
                f"- tfidf: TF-IDF {m['selected']['features']} + logistic regression C={m['selected']['C']}, "
                f"threshold {m['threshold']} (CV-tuned), CV F1 {m['cv_f1_at_threshold']}"
            )
        else:
            lines.append(
                f"- transformer: {m['base_model'].split('@')[0]} fine-tuned "
                f"({m['hyperparameters']['epochs']} epochs, lr {m['hyperparameters']['lr']}), "
                f"threshold {m['threshold']} (fixed), CV mean F1 {m['cv_mean_f1_at_0.5']}"
            )
    lines += [
        "\nAll test sets side by side. Precision, recall and FPR show Wilson 95 % intervals "
        "(informative at zero counts); F1 shows the bootstrap 95 % interval.\n",
        "| Test set | Model | n | Precision | Recall | F1 [95 % CI] | FP | FN | FPR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in TEST_SETS:
        for mname, mres in inj["models"].items():
            r = mres[name]
            lines.append(
                f"| {name} ({r['description']}) | {mname} | {r['n']} | {fmt(r['precision'])} "
                f"{ci(r['wilson95']['precision'])} | {fmt(r['recall'])} {ci(r['wilson95']['recall'])} | "
                f"{fmt(r['f1'])} {ci(r['ci95']['f1'])} | {r['false_positives']} | "
                f"{r['false_negatives']} | {fmt(r['fpr'])} {ci(r['wilson95']['fpr'])} |"
            )
    lines.append("\nConfusion matrices (tp / fp / fn / tn):\n")
    for mname, mres in inj["models"].items():
        lines.append(f"- {mname}: " + "; ".join(
            f"{n}: {mres[n]['confusion']['tp']} / {mres[n]['confusion']['fp']} / "
            f"{mres[n]['confusion']['fn']} / {mres[n]['confusion']['tn']}"
            for n in TEST_SETS
        ))  # fmt: skip
    first = next(iter(inj["models"].values()))
    lines.append(
        "\nRows removed because they duplicate a train row: " + "; ".join(
            f"{n}: {first[n]['dropped_train_duplicates']}" for n in TEST_SETS
        )
    )  # fmt: skip

    rules = res["input_rules"]
    lines += [
        f"\n## PII rules ({rules['dataset']})\n",
        "Span level, overlap match (a predicted span counts if it overlaps a gold span of the same type).\n",
        "| Entity | Gold spans | Precision | Recall | F1 [95 % CI] | FP | FN | Exact-match F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in ("email", "iban", "phone"):
        r = rules["pii_regex"][t]
        o = r["overlap"]
        lines.append(
            f"| {t} | {r['gold_spans']} | {fmt(o['precision'])} | {fmt(o['recall'])} | "
            f"{fmt(o['f1'])} {ci(o['ci95']['f1'])} | {o['false_positives']} | "
            f"{o['false_negatives']} | {fmt(r['exact']['f1'])} |"
        )
    d = rules["pii_regex"]["any_document_level"]
    lines.append(
        f"\nDocument level (contains any of email/IBAN/phone): precision {fmt(d['precision'])}, "
        f"recall {fmt(d['recall'])}, F1 {fmt(d['f1'])} {ci(d['ci95']['f1'])}, "
        f"FP {d['false_positives']}, FN {d['false_negatives']} (n={d['n']}).\n"
    )
    lines.append(
        "False positives by the gold label they overlap ('(unlabelled text)' = no gold label):\n"
    )
    for t in ("email", "iban", "phone"):
        lines.append(f"- {t}: {rules['pii_regex'][t]['false_positives_overlap_gold_label']}")

    s = rules["secrets_regex"]
    lines += ["\n## Secrets rules\n", "| Evaluation | Precision | Recall | F1 [95 % CI] | FP | FN |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for name, label in (("gretel_vs_api_key", "Gretel test, gold = api_key spans"),
                        ("gretel_vs_api_key_or_password", "Gretel test, gold = api_key or password")):  # fmt: skip
        o = s[name]["overlap"]
        lines.append(
            f"| {label} ({s[name]['gold_spans']} gold spans) | {fmt(o['precision'])} | "
            f"{fmt(o['recall'])} | {fmt(o['f1'])} {ci(o['ci95']['f1'])} | "
            f"{o['false_positives']} | {o['false_negatives']} |"
        )
    sy = s["synthetic"]
    lines.append(
        f"| Synthetic, self-made (n={sy['n']}; sanity check only) | {fmt(sy['precision'])} | "
        f"{fmt(sy['recall'])} | {fmt(sy['f1'])} | {sy['false_positives']} | {sy['false_negatives']} |"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    args = ap.parse_args()

    manifest = yaml.safe_load((ROOT / "data" / "manifest.yaml").read_text())["datasets"]
    res = {
        "meta": {
            "generated_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "n_boot": args.n_boot,
            "datasets": {k: f"{v['repo']}@{v['revision']}" for k, v in manifest.items()},
            "data_checksums_sha256": json.loads((ROOT / "data" / "checksums.json").read_text()),
            "policy_sha256": hashlib.sha256(
                (ROOT / "config" / "policy.yaml").read_bytes()
            ).hexdigest(),
        },
        "injection_classifier": eval_injection(args.n_boot),
        "input_rules": eval_rules(args.n_boot),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    (RESULTS / "results.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
