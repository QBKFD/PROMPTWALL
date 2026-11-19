"""Transformer experiment: 5-fold CV of a fine-tuned small multilingual model on
deepset train, with the same folds as training/train_tfidf.py. Hyperparameters are
fixed up front (no tuning). The pre-registered adoption rule is in docs/decisions.md.

    uv sync --extra transformer
    uv run python -m training.train_transformer           # CV only
    uv run python -m training.train_transformer --final   # also fit on all of train

Needs the optional `transformer` dependency group. CPU is fine but slow.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from eval.datasets import deepset

ROOT = Path(__file__).resolve().parents[1]
BASE = "distilbert/distilbert-base-multilingual-cased"
BASE_REV = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
HP = {"lr": 3e-5, "epochs": 3, "batch_size": 16, "max_len": 256, "weight_decay": 0.01,
      "warmup_frac": 0.1, "seed": 0}  # fmt: skip


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load():
    tok = AutoTokenizer.from_pretrained(BASE, revision=BASE_REV)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, revision=BASE_REV, num_labels=2, use_safetensors=True
    )
    return tok, model


def batches(texts, labels, tok, bs, shuffle, rng):
    idx = list(range(len(texts)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i : i + bs]
        enc = tok([texts[j] for j in b], truncation=True, max_length=HP["max_len"],
                  padding=True, return_tensors="pt")  # fmt: skip
        yield enc, torch.tensor([labels[j] for j in b]) if labels is not None else None


def fit(texts, labels):
    set_seed(HP["seed"])
    tok, model = load()
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    steps = HP["epochs"] * ((len(texts) + HP["batch_size"] - 1) // HP["batch_size"])
    warm = int(steps * HP["warmup_frac"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (steps - s) / max(1, steps - warm)),
    )
    rng = random.Random(HP["seed"])
    model.train()
    for _ in range(HP["epochs"]):
        for enc, y in batches(texts, labels, tok, HP["batch_size"], True, rng):
            loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return tok, model


@torch.no_grad()
def predict(tok, model, texts) -> np.ndarray:
    out = []
    for enc, _ in batches(texts, None, tok, 32, False, None):
        out.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--final-only", action="store_true", help="skip CV, only fit on all of train")
    args = ap.parse_args()
    torch.set_num_threads(8)

    data = deepset("train")
    texts = [e.text for e in data]
    y = np.array([e.label for e in data])
    if args.final_only:
        return save_final(texts, y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # same folds as TF-IDF
    probs = np.zeros(len(y))
    fold_f1, t0 = [], time.perf_counter()
    for k, (tr, va) in enumerate(skf.split(texts, y)):
        tok, model = fit([texts[i] for i in tr], [int(y[i]) for i in tr])
        probs[va] = predict(tok, model, [texts[i] for i in va])
        fold_f1.append(f1_score(y[va], probs[va] >= 0.5))
        print(f"fold {k}: F1@0.5={fold_f1[-1]:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)

    res = {
        "base_model": f"{BASE}@{BASE_REV}",
        "hyperparameters": HP,
        "cv_fold_f1_at_0.5": [round(f, 4) for f in fold_f1],
        "cv_mean_f1_at_0.5": round(float(np.mean(fold_f1)), 4),
        "cv_pooled_f1_at_0.5": round(float(f1_score(y, probs >= 0.5)), 4),
        "cv_seconds": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    out = ROOT / "results" / "transformer_cv.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    np.save(ROOT / "models" / "transformer_oof_probs.npy", probs)

    if args.final:
        save_final(texts, y)


def save_final(texts, y) -> None:
    t0 = time.perf_counter()
    tok, model = fit(texts, [int(v) for v in y])
    dest = ROOT / "models" / "injection_transformer"
    model.save_pretrained(dest, safe_serialization=True)
    tok.save_pretrained(dest)
    meta = {
        "model": "transformer",
        "base_model": f"{BASE}@{BASE_REV}",
        "hyperparameters": HP,
        "train_data": "deepset/prompt-injections@4f61ecb train split",
        "n_train": len(y),
        "threshold": 0.5,
        "threshold_selection": "fixed at 0.5 (not tuned)",
        "fit_seconds": round(time.perf_counter() - t0, 1),
    }
    (dest / "firewall_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved {dest}")


if __name__ == "__main__":
    main()
