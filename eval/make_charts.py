"""Render the README charts and result tables from results/results.json.

    uv run python -m eval.make_charts           # write docs/img/*.svg|png and update README.md
    uv run python -m eval.make_charts --check   # fail if README tables are out of date

Every number shown comes from results/results.json. Nothing is typed by hand;
a metric that is missing from the results file is left out, not estimated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.json"
IMG = ROOT / "docs" / "img"
README = ROOT / "README.md"

MODEL_LABELS = {"tfidf": "TF-IDF + logistic regression", "transformer": "DistilBERT (multilingual)"}
SET_LABELS = {
    "deepset_test": "deepset test",
    "jackhhao_jailbreak_test": "jackhhao jailbreak test",
    "lakera_gandalf_test": "Lakera Gandalf test",
}
# The metric that fits each test set, and which interval to show for it.
SET_METRIC = {
    "deepset_test": ("f1", "ci95", "F1 (higher is better)"),
    "jackhhao_jailbreak_test": ("fpr", "wilson95", "False-positive rate on benign prompts (lower is better)"),
    "lakera_gandalf_test": ("recall", "wilson95", "Recall (injections only; higher is better)"),
}  # fmt: skip
CI_NAMES = {"ci95": "bootstrap", "wilson95": "Wilson"}
METRIC_NAMES = {"precision": "precision", "recall": "recall", "f1": "F1", "fpr": "FPR"}

# Reference palette (validated: categorical slots 1-2 on the light chart surface).
SURFACE, INK, INK_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SERIES = {"tfidf": "#2a78d6", "transformer": "#eb6834"}


def load() -> dict:
    return json.loads(RESULTS.read_text())


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def ci_str(ci) -> str:
    return "n/a" if not ci else f"{ci[0]:.3f} to {ci[1]:.3f}"


# --------------------------------------------------------------------------- tables


def injection_table(res: dict) -> str:
    inj = res["injection_classifier"]
    rows = [
        "| Test set | Model | n | Metric | Value | 95 % CI | CI method |",
        "|---|---|---|---|---|---|---|",
    ]
    for set_name, set_label in SET_LABELS.items():
        for model, mres in inj["models"].items():
            r = mres[set_name]
            for metric, key in (
                ("precision", "wilson95"),
                ("recall", "wilson95"),
                ("f1", "ci95"),
                ("fpr", "wilson95"),
            ):
                if r.get(metric) is None:
                    continue
                rows.append(
                    f"| {set_label} | {MODEL_LABELS[model]} | {r['n']} | {METRIC_NAMES[metric]} "
                    f"| {f3(r[metric])} | {ci_str(r[key].get(metric))} | {CI_NAMES[key]} |"
                )
    return "\n".join(rows)


def pii_table(res: dict) -> str:
    rules = res["input_rules"]
    pii = rules["pii_regex"]
    rows = [
        "| Entity | Gold spans | Metric | Value | 95 % CI (bootstrap over documents) |",
        "|---|---|---|---|---|",
    ]
    for ent in ("email", "iban", "phone"):
        o = pii[ent]["overlap"]
        for metric in ("precision", "recall", "f1"):
            rows.append(
                f"| {ent} | {pii[ent]['gold_spans']} | span {metric} | {f3(o[metric])} | {ci_str(o['ci95'].get(metric))} |"
            )
    d = pii["any_document_level"]
    for metric, key in (("precision", "wilson95"), ("recall", "wilson95"), ("f1", "ci95")):
        rows.append(
            f"| any of the three (document level, n={d['n']}) | {d['positives']} docs | {metric} | {f3(d[metric])} | {ci_str(d[key].get(metric))} ({CI_NAMES[key]}) |"
        )
    return "\n".join(rows)


def secrets_table(res: dict) -> str:
    s = res["input_rules"]["secrets_regex"]
    rows = [
        "| Evaluation | n | Metric | Value | 95 % CI |",
        "|---|---|---|---|---|",
    ]
    for key, label in (
        ("gretel_vs_api_key", "Gretel test, gold = api_key spans"),
        ("gretel_vs_api_key_or_password", "Gretel test, gold = api_key or password spans"),
    ):
        o = s[key]["overlap"]
        for metric in ("precision", "recall", "f1"):
            rows.append(
                f"| {label} | {s[key]['gold_spans']} gold spans | span {metric} | {f3(o[metric])} | {ci_str(o['ci95'].get(metric))} (bootstrap) |"
            )
    sy = s["synthetic"]
    rows.append(
        f"| Synthetic set written by the rules' author (format coverage only, not a benchmark) | {sy['n']} | F1 | {f3(sy['f1'])} | not meaningful |"
    )
    return "\n".join(rows)


def limitations_block(res: dict) -> str:
    inj = res["injection_classifier"]["models"]
    secrets = res["input_rules"]["secrets_regex"]["gretel_vs_api_key"]["overlap"]
    lines = []
    for model in ("tfidf", "transformer"):
        if model not in inj:
            continue
        j = inj[model]["jackhhao_jailbreak_test"]
        fp, neg = j["confusion"]["fp"], j["negatives"]
        lines.append(
            f"- {MODEL_LABELS[model]}: flags {fp} of {neg} benign jackhhao prompts as attacks "
            f"(false-positive rate {f3(j['fpr'])}, Wilson 95 % CI {ci_str(j['wilson95']['fpr'])})."
        )
    lines.append(
        f"- Secrets rules against Gretel `api_key` labels: span F1 {f3(secrets['f1'])} "
        f"(bootstrap 95 % CI {ci_str(secrets['ci95']['f1'])})."
    )
    return "\n".join(lines)


def provenance(res: dict) -> str:
    m = res["meta"]
    return (
        f"Generated from `results/results.json` ({m['generated_utc']}, code commit "
        f"`{m['git_commit'][:7]}`, {m['n_boot']} bootstrap resamples, seed 0)."
    )


def selection_block(res: dict) -> str:
    inj = res["injection_classifier"]
    cv = inj["cv_f1_at_0.5"]
    parts = [f"{MODEL_LABELS[m]} {f3(v)}" for m, v in cv.items() if v is not None]
    return (
        f"Cross-validated F1 at threshold 0.5 on deepset train: {', '.join(parts)}. "
        f"Pre-registered rule: {inj['selection_rule']}. Model chosen by that rule: "
        f"{MODEL_LABELS[inj['selected']]}."
    )


BLOCKS = {
    "provenance": provenance,
    "selection": selection_block,
    "injection-table": injection_table,
    "pii-table": pii_table,
    "secrets-table": secrets_table,
    "limitations": limitations_block,
}


def render_readme(readme: str, res: dict) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(
            rf"(<!-- BEGIN GENERATED: {name} -->\n).*?(<!-- END GENERATED: {name} -->)", re.S
        )
        if not pat.search(readme):
            raise SystemExit(f"README is missing the generated block '{name}'")
        readme = pat.sub(lambda m, fn=fn: m.group(1) + fn(res) + "\n" + m.group(2), readme)
    return readme


# --------------------------------------------------------------------------- charts


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "svg.fonttype": "none",  # keep text as text in the SVG
        "svg.hashsalt": "promptwall",  # deterministic element ids
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })  # fmt: skip
    return plt


def _save(fig, name: str) -> list[Path]:
    IMG.mkdir(parents=True, exist_ok=True)
    out = []
    svg, png = IMG / f"{name}.svg", IMG / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None, "Creator": None})
    fig.savefig(png, bbox_inches="tight", dpi=200, metadata={"Software": None})
    out += [svg, png]
    return out


def _axes_common(ax):
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def chart_injection(res: dict) -> list[Path]:
    plt = _style()
    models = [m for m in ("tfidf", "transformer") if m in res["injection_classifier"]["models"]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    width = 0.36
    for ax, (set_name, (metric, ci_key, ylabel)) in zip(axes, SET_METRIC.items(), strict=True):
        n = None
        for i, model in enumerate(models):
            r = res["injection_classifier"]["models"][model][set_name]
            n = r["n"]
            val, ci = r.get(metric), r[ci_key].get(metric)
            if val is None:
                continue
            x = i * (width + 0.04)
            yerr = [[val - ci[0]], [ci[1] - val]] if ci else None
            ax.bar(x, val, width=width, color=SERIES[model], label=MODEL_LABELS[model])
            ax.errorbar(x, val, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4)
            top = ci[1] if ci else val
            ax.text(x, top + 0.025, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
        _axes_common(ax)
        ax.set_xticks([])
        ax.set_xlim(-0.4, (len(models) - 1) * (width + 0.04) + 0.4)
        ax.set_title(f"{SET_LABELS[set_name]} (n={n})", fontsize=10.5, color=INK, pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=9.5)  # fmt: skip
    fig.text(
        0.5, -0.04,
        "Error bars: 95 % CI (bootstrap for F1, Wilson for FPR and recall). "
        "Both models trained on deepset train only.",
        ha="center", fontsize=8.5, color=INK_2,
    )  # fmt: skip
    fig.tight_layout(w_pad=2.5)
    paths = _save(fig, "injection_classifiers")
    plt.close(fig)
    return paths


def chart_pii(res: dict) -> list[Path]:
    plt = _style()
    pii = res["input_rules"]["pii_regex"]
    ents = ["email", "iban", "phone"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, ent in enumerate(ents):
        o = pii[ent]["overlap"]
        val, ci = o["f1"], o["ci95"].get("f1")
        ax.bar(i, val, width=0.5, color=SERIES["tfidf"])
        if ci:
            ax.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]], fmt="none", ecolor=INK,
                        elinewidth=1.2, capsize=4)  # fmt: skip
        ax.text(i, (ci[1] if ci else val) + 0.025, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)  # fmt: skip
    _axes_common(ax)
    ax.set_xticks(range(len(ents)))
    ax.set_xticklabels([f"{e}\n({pii[e]['gold_spans']} gold spans)" for e in ents], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Span F1, overlap match", fontsize=9)
    n_docs = pii["any_document_level"]["n"]
    ax.set_title(f"PII regex rules, Gretel test split ({n_docs} documents)", fontsize=10.5,
                 color=INK, pad=8)  # fmt: skip
    fig.text(0.5, -0.05, "Error bars: 95 % bootstrap CI, resampling documents.", ha="center",
             fontsize=8.5, color=INK_2)  # fmt: skip
    fig.tight_layout()
    paths = _save(fig, "pii_span_f1")
    plt.close(fig)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify README blocks are current")
    args = ap.parse_args()
    res = load()
    readme = README.read_text()
    updated = render_readme(readme, res)
    if args.check:
        if updated != readme:
            print("README result blocks are out of date; run `uv run python -m eval.make_charts`")
            return 1
        print("README result blocks match results/results.json")
        return 0
    README.write_text(updated)
    written = chart_injection(res) + chart_pii(res)
    for p in written:
        print(f"wrote {p.relative_to(ROOT)}")
    print("updated README.md generated blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
