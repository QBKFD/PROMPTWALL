"""Prompt-injection classifier check (TF-IDF + logistic regression, trained by
training/train_tfidf.py). Flags the whole message; it reports a score, not spans,
so the policy can block or log but not mask."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

from firewall.checks.base import Check, CheckContext, register
from firewall.schemas import Finding

DEFAULT_MODEL = "models/injection_tfidf.joblib"


class InjectionClassifierCheck(Check):
    name = "injection_classifier"
    supports_mask = False

    def __init__(self, model: Any, threshold: float) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.model = model
        self.threshold = threshold

    def score(self, texts: list[str]) -> list[float]:
        return [float(p) for p in self.model.predict_proba(texts)[:, 1]]

    def run(self, text: str, ctx: CheckContext) -> list[Finding]:
        p = self.score([text])[0]
        if p >= self.threshold:
            return [
                Finding(
                    type="prompt_injection",
                    score=round(p, 4),
                    detail=f"classifier probability {p:.3f} >= threshold {self.threshold}",
                )
            ]
        return []

    @classmethod
    def from_path(
        cls, path: str | Path, threshold: float | None = None
    ) -> InjectionClassifierCheck:
        """Load a model trained by training/train_tfidf.py, verifying its sha256 against
        the metadata file written next to it (we only load artefacts we produced)."""
        path = Path(path)
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"{path} (or its .json metadata) not found; train it with "
                "`uv run python -m training.train_tfidf`"
            )
        meta = json.loads(meta_path.read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != meta.get("sha256"):
            raise ValueError(f"{path} sha256 does not match its metadata; refusing to load")
        return cls(joblib.load(path), threshold if threshold is not None else meta["threshold"])


@register("injection_classifier")
def _factory(params: dict[str, Any]) -> Check:
    unknown = set(params) - {"model_path", "threshold"}
    if unknown:
        raise ValueError(f"injection_classifier: unknown params {sorted(unknown)}")
    return InjectionClassifierCheck.from_path(
        params.get("model_path", DEFAULT_MODEL), params.get("threshold")
    )


# ---------------------------------------------------------------- transformer variant

DEFAULT_TRANSFORMER = "models/injection_transformer"


class TransformerScorer:
    """Wraps a fine-tuned sequence classifier saved by training/train_transformer.py.
    Exposes predict_proba like scikit-learn so the same check class can use it.
    Needs the optional `transformer` extra (torch, transformers)."""

    def __init__(self, path: str | Path, max_len: int = 256, num_threads: int | None = None):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - depends on installed extras
            raise ImportError(
                "injection_transformer needs the optional extra: `uv sync --extra transformer`"
            ) from e
        self._torch = torch
        if num_threads:
            torch.set_num_threads(num_threads)
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True
        ).eval()
        self.max_len = max_len

    def predict_proba(self, texts: list[str]):
        import numpy as np

        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = self.tok(texts[i : i + 32], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt")  # fmt: skip
                p = self._torch.softmax(self.model(**enc).logits, -1)
                out.append(p.numpy())
        return np.concatenate(out)


def load_transformer_check(
    path: str | Path = DEFAULT_TRANSFORMER, threshold: float | None = None
) -> InjectionClassifierCheck:
    path = Path(path)
    meta_path = path / "firewall_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found; train it with "
            "`uv run --extra transformer python -m training.train_transformer --final-only`"
        )
    meta = json.loads(meta_path.read_text())
    scorer = TransformerScorer(path, max_len=meta["hyperparameters"]["max_len"])
    return InjectionClassifierCheck(
        scorer, threshold if threshold is not None else meta["threshold"]
    )


@register("injection_transformer")
def _transformer_factory(params: dict[str, Any]) -> Check:
    unknown = set(params) - {"model_path", "threshold"}
    if unknown:
        raise ValueError(f"injection_transformer: unknown params {sorted(unknown)}")
    return load_transformer_check(
        params.get("model_path", DEFAULT_TRANSFORMER), params.get("threshold")
    )
