"""Claude via the official Anthropic SDK."""

from __future__ import annotations

import anthropic

from firewall.llm.base import LLMClient, LLMError, LLMResponse
from firewall.llm.budget import Budget


class AnthropicClient(LLMClient):
    backend = "anthropic"

    def __init__(
        self,
        model: str,
        max_tokens: int,
        effort: str = "low",
        api_key: str | None = None,
        timeout_s: float = 60.0,
        budget: Budget | None = None,
    ) -> None:
        super().__init__(max_tokens=max_tokens, budget=budget)
        self._model = model
        self.effort = effort
        # api_key=None lets the SDK resolve credentials from the environment.
        # max_retries=0: every retry would be a billed call outside the budget.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=0)

    @property
    def model(self) -> str:
        return self._model

    def _complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited by Anthropic API: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"could not reach Anthropic API: {e}") from e

        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason,
        )
