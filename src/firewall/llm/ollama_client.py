"""Local model via Ollama's /api/chat endpoint (non-streaming)."""

from __future__ import annotations

import httpx

from firewall.llm.base import LLMClient, LLMError, LLMResponse
from firewall.llm.budget import Budget


class OllamaClient(LLMClient):
    backend = "ollama"

    def __init__(
        self,
        model: str,
        max_tokens: int,
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        budget: Budget | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(max_tokens=max_tokens, budget=budget)
        self._model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    @property
    def model(self) -> str:
        return self._model

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        payload = {
            "model": self._model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"num_predict": max_tokens},
        }
        try:
            r = self._http.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama error {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, ValueError) as e:
            raise LLMError(f"could not get a response from Ollama: {e}") from e

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self._model),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            stop_reason=data.get("done_reason"),
        )
