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
