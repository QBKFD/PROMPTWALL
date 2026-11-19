"""Loaders for the pinned datasets in data/raw/ (run `python -m eval.download` first)."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


@dataclass
class TextExample:
    text: str
    label: int  # 1 = injection / jailbreak, 0 = benign


@dataclass
class SpanExample:
    text: str
    spans: list[tuple[int, int, str]] = field(default_factory=list)  # (start, end, label)
    language: str = ""


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `uv run python -m eval.download`")
    return path


def deepset(split: str) -> list[TextExample]:
    t = pq.read_table(_require(RAW / "deepset_injections" / f"{split}.parquet"))
    return [
        TextExample(x, int(y))
        for x, y in zip(t["text"].to_pylist(), t["label"].to_pylist(), strict=True)
    ]


def jackhhao_test() -> list[TextExample]:
    with _require(RAW / "jackhhao_jailbreak" / "test.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labels = {"jailbreak": 1, "benign": 0}
    return [TextExample(r["prompt"], labels[r["type"]]) for r in rows]


def gandalf_test() -> list[TextExample]:
    t = pq.read_table(_require(RAW / "lakera_gandalf" / "test.parquet"))
    return [TextExample(x, 1) for x in t["text"].to_pylist()]


def gretel(split: str) -> list[SpanExample]:
    t = pq.read_table(
        _require(RAW / "gretel_pii" / f"{split}.parquet"),
        columns=["generated_text", "pii_spans", "language"],
    )
    out = []
    for text, spans, lang in zip(
        t["generated_text"].to_pylist(),
        t["pii_spans"].to_pylist(),
        t["language"].to_pylist(),
        strict=True,
    ):
        parsed = [
            (s["start"], s["end"], s["label"])
            for s in json.loads(spans)
            if s["end"] > s["start"] and text[s["start"] : s["end"]].strip()
        ]
        out.append(SpanExample(text, parsed, lang))
    return out


# ---------------------------------------------------------------- leakage checks


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()


def _shingles(text: str, n: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i : i + n] for i in range(max(1, len(t) - n + 1))}


def overlap_with(
    train: list[TextExample], test: list[TextExample], near_threshold: float = 0.9
) -> tuple[set[int], set[int]]:
    """Indices of test examples that duplicate a train example: (exact, near).
    exact = identical after normalization; near = char-5-gram Jaccard >= threshold."""
    train_norm = {normalize(e.text) for e in train}
    exact = {i for i, e in enumerate(test) if normalize(e.text) in train_norm}
    train_sh = [_shingles(e.text) for e in train]
    near = set()
    for i, e in enumerate(test):
        if i in exact:
            continue
        sh = _shingles(e.text)
        for tsh in train_sh:
            inter = len(sh & tsh)
            if inter and inter / len(sh | tsh) >= near_threshold:
                near.add(i)
                break
    return exact, near
