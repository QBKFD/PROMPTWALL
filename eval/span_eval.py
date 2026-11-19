"""Span- and document-level scoring of a span detector against gold PII labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np

from eval.datasets import SpanExample
from eval.metrics import binary_report

Span = tuple[int, int, str]
_STRIP = " \t\n\r[](){}<>\"'.,;:"


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s] in _STRIP:
        s += 1
    while e > s and text[e - 1] in _STRIP:
        e -= 1
    return s, e


def score_spans(
    docs: list[SpanExample],
    detect: Callable[[str], list[Span]],
    type_map: dict[str, set[str]],
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """type_map: our entity type -> set of gold labels that count as that entity.

    Overlap match: a predicted span is correct if it overlaps a gold span of a
    mapped label; a gold span is found if any predicted span of its type overlaps it.
    Exact match: same, but boundaries must be equal after trimming spaces/brackets/punctuation.
    """
    per_doc = {
        t: [] for t in type_map
    }  # rows: (pred_ok, n_pred, gold_found, n_gold, exact_ok, exact_found)
    doc_true = {t: [] for t in type_map}
    doc_pred = {t: [] for t in type_map}
    fp_overlap = {t: Counter() for t in type_map}
    examples = {t: {"fp": [], "fn": []} for t in type_map}

    for di, doc in enumerate(docs):
        preds = detect(doc.text)
        for typ, labels in type_map.items():
            p = [(s, e) for s, e, t in preds if t == typ]
            g = [(s, e) for s, e, lab in doc.spans if lab in labels]
            others = [(s, e, lab) for s, e, lab in doc.spans if lab not in labels]
            pt = {_trim(doc.text, *x) for x in p}
            gt = {_trim(doc.text, *x) for x in g}
            pred_ok = sum(any(_overlaps(x, y) for y in g) for x in p)
            gold_found = sum(any(_overlaps(y, x) for x in p) for y in g)
            per_doc[typ].append((pred_ok, len(p), gold_found, len(g), len(pt & gt), len(pt & gt)))
            doc_true[typ].append(bool(g))
            doc_pred[typ].append(bool(p))
            for x in p:
                if not any(_overlaps(x, y) for y in g):
                    hit = [lab for s, e, lab in others if _overlaps(x, (s, e))]
                    fp_overlap[typ][hit[0] if hit else "(unlabelled text)"] += 1
                    if len(examples[typ]["fp"]) < 15:
                        examples[typ]["fp"].append((di, doc.text[x[0] : x[1]], hit[:1]))
            for y in g:
                if not any(_overlaps(y, x) for x in p) and len(examples[typ]["fn"]) < 15:
                    examples[typ]["fn"].append((di, doc.text[y[0] : y[1]]))

    rng = np.random.default_rng(seed)
    out = {}
    for typ in type_map:
        a = np.array(per_doc[typ], dtype=float).reshape(-1, 6)
        tot = a.sum(0)
        res = {
            "gold_spans": int(tot[3]),
            "pred_spans": int(tot[1]),
            "overlap": _prf(tot[0], tot[1], tot[2], tot[3]),
            "exact": _prf(tot[4], tot[1], tot[5], tot[3]),
        }
        if n_boot:
            # resample documents in chunks to bound memory: (chunk, n_docs, 6)
            parts = []
            for start in range(0, n_boot, 500):
                idx = rng.integers(0, len(a), size=(min(500, n_boot - start), len(a)))
                parts.append(a[idx].sum(1))
            boot = np.concatenate(parts)  # (B, 6)
            res["overlap"]["ci95"] = _boot_prf(boot[:, 0], boot[:, 1], boot[:, 2], boot[:, 3])
        else:
            res["overlap"]["ci95"] = None
        res["false_positives_overlap_gold_label"] = dict(fp_overlap[typ].most_common())
        res["document_level"] = binary_report(
            doc_true[typ], doc_pred[typ], n_boot=max(n_boot, 1), seed=seed
        )
        res["_examples"] = examples[typ]
        out[typ] = res

    any_true = [any(doc_true[t][i] for t in type_map) for i in range(len(docs))]
    any_pred = [any(doc_pred[t][i] for t in type_map) for i in range(len(docs))]
    out["any_document_level"] = binary_report(any_true, any_pred, n_boot=max(n_boot, 1), seed=seed)
    return out


def _prf(pred_ok, n_pred, gold_found, n_gold) -> dict:
    p = pred_ok / n_pred if n_pred else None
    r = gold_found / n_gold if n_gold else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {
        "precision": None if p is None else round(p, 4),
        "recall": None if r is None else round(r, 4),
        "f1": None if f is None else round(f, 4),
        "false_positives": int(n_pred - pred_ok),
        "false_negatives": int(n_gold - gold_found),
    }


def _boot_prf(pred_ok, n_pred, gold_found, n_gold) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        p = pred_ok / n_pred
        r = gold_found / n_gold
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    out = {}
    for name, v in (("precision", p), ("recall", r), ("f1", f)):
        v = v[~np.isnan(v)]
        out[name] = (
            None
            if len(v) == 0
            else tuple(round(float(q), 4) for q in np.quantile(v, [0.025, 0.975]))
        )
    return out
