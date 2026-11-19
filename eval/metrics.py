"""Binary classification metrics with percentile-bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None if p is None or r is None else 0.0
        return 2 * p * r / (p + r)

    @property
    def fpr(self) -> float | None:
        d = self.fp + self.tn
        return self.fp / d if d else None


def confusion(y_true, y_pred) -> Confusion:
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    return Confusion(
        tp=int((t & p).sum()),
        fp=int((~t & p).sum()),
        fn=int((t & ~p).sum()),
        tn=int((~t & ~p).sum()),
    )


def _metric_arrays(t: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised metrics over bootstrap rows. t, p: (B, n) bool arrays."""
    tp = (t & p).sum(1).astype(float)
    fp = (~t & p).sum(1).astype(float)
    fn = (t & ~p).sum(1).astype(float)
    tn = (~t & ~p).sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        f1 = np.where(np.isnan(prec) | np.isnan(rec), np.nan, f1)
        fpr = fp / (fp + tn)
        acc = (tp + tn) / t.shape[1]
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr, "accuracy": acc}


def bootstrap_ci(
    y_true, y_pred, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> dict[str, tuple[float, float] | None]:
    """Percentile CIs from resampling examples with replacement.
    Resamples where a metric is undefined (e.g. no positives drawn) are dropped
    for that metric."""
    t = np.asarray(y_true, dtype=bool)
    p = np.asarray(y_pred, dtype=bool)
    n = len(t)
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[float, float] | None] = {}
    chunks = []
    for start in range(0, n_boot, 2000):
        b = min(2000, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        chunks.append(_metric_arrays(t[idx], p[idx]))
    for name in chunks[0]:
        vals = np.concatenate([c[name] for c in chunks])
        vals = vals[~np.isnan(vals)]
        if len(vals) < n_boot * 0.5:  # metric undefined in most resamples
            out[name] = None
        else:
            lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
            out[name] = (round(float(lo), 4), round(float(hi), 4))
    return out


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson score interval for a proportion k/n. Unlike the percentile bootstrap,
    it stays informative when k == 0 or k == n (e.g. zero false positives)."""
    if n == 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (round(float(max(0.0, centre - half)), 4), round(float(min(1.0, centre + half)), 4))


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def binary_report(y_true, y_pred, n_boot: int = 10_000, seed: int = 0) -> dict:
    c = confusion(y_true, y_pred)
    n = len(y_true)
    return {
        "n": n,
        "positives": c.tp + c.fn,
        "negatives": c.fp + c.tn,
        "confusion": asdict(c),
        "false_positives": c.fp,
        "false_negatives": c.fn,
        "precision": _r(c.precision),
        "recall": _r(c.recall),
        "f1": _r(c.f1),
        "fpr": _r(c.fpr),
        "accuracy": _r((c.tp + c.tn) / n if n else None),
        "ci95": bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed),
        "wilson95": {
            "precision": wilson(c.tp, c.tp + c.fp),
            "recall": wilson(c.tp, c.tp + c.fn),
            "fpr": wilson(c.fp, c.fp + c.tn),
        },
    }
