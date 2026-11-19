"""Train the TF-IDF + logistic regression prompt-injection classifier.

Uses ONLY deepset/prompt-injections `train`. Model selection (features, C) and the
decision threshold are chosen by stratified 5-fold cross-validation inside train;
the test split is never loaded here.

    uv run python -m training.train_tfidf
Writes models/injection_tfidf.joblib and models/injection_tfidf.json (metadata).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from eval.datasets import deepset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "injection_tfidf.joblib"
SEED = 0


def make_pipeline(features: str, C: float) -> Pipeline:
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True
    )
    feats = {"word": word, "char": char, "word+char": FeatureUnion([("w", word), ("c", char)])}
    return Pipeline([
        ("tfidf", feats[features]),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                   random_state=SEED)),
    ])  # fmt: skip


def oof_probs(texts: list[str], y: np.ndarray, features: str, C: float) -> np.ndarray:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = np.zeros(len(y))
    for tr, va in skf.split(texts, y):
        m = make_pipeline(features, C).fit([texts[i] for i in tr], y[tr])
        probs[va] = m.predict_proba([texts[i] for i in va])[:, 1]
    return probs


def best_threshold(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y, probs >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def main() -> None:
    data = deepset("train")
    texts = [e.text for e in data]
    y = np.array([e.label for e in data])
    print(f"train n={len(y)} positives={int(y.sum())}")

    results = []
    for features, C in itertools.product(["word", "char", "word+char"], [0.3, 1, 3, 10, 30, 100]):
        probs = oof_probs(texts, y, features, C)
        f1_05 = f1_score(y, probs >= 0.5)
        results.append({"features": features, "C": C, "cv_f1_at_0.5": round(f1_05, 4)})
        print(f"  {features:10s} C={C:<5} CV F1@0.5={f1_05:.4f}")

    best = max(results, key=lambda r: r["cv_f1_at_0.5"])
    probs = oof_probs(texts, y, best["features"], best["C"])
    thr, thr_f1 = best_threshold(y, probs)
    print(f"selected {best}; CV-tuned threshold={thr} (CV F1 {thr_f1:.4f})")

    t0 = time.perf_counter()
    model = make_pipeline(best["features"], best["C"]).fit(texts, y)
    train_s = time.perf_counter() - t0

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    meta = {
        "model": "tfidf_logreg",
        "train_data": "deepset/prompt-injections@4f61ecb train split",
        "n_train": len(y),
        "selection": "stratified 5-fold CV on train, max F1 at threshold 0.5",
        "grid": results,
        "selected": best,
        "threshold": thr,
        "threshold_selection": "max out-of-fold F1 on train, grid 0.05..0.95",
        "cv_f1_at_threshold": round(thr_f1, 4),
        "fit_seconds": round(train_s, 2),
        "sklearn_version": sklearn.__version__,
        "sha256": digest,
    }
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT} sha256={digest[:12]}")


if __name__ == "__main__":
    main()
