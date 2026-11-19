"""Backend adapters, tested offline: Ollama through a mock HTTP transport,
Anthropic by replacing the SDK's messages.create with a stub."""

from types import SimpleNamespace

import httpx
import pytest

from firewall.config import Settings
from firewall.llm import build_llm
from firewall.llm.anthropic_client import AnthropicClient
from firewall.llm.base import LLMError
from firewall.llm.budget import Budget
from firewall.llm.fake import FakeLLM
from firewall.llm.ollama_client import OllamaClient


def test_ollama_parses_response_and_sends_expected_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["json"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "model": "llama3.2",
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 12,
                "eval_count": 3,
                "done_reason": "stop",
            },
        )

    budget = Budget(10, 10_000)
    llm = OllamaClient("llama3.2", 64, transport=httpx.MockTransport(handler), budget=budget)
    r = llm.complete("sys", "hi")
    assert (r.text, r.input_tokens, r.output_tokens, r.stop_reason) == ("hello", 12, 3, "stop")
    assert seen["path"] == "/api/chat"
    assert '"num_predict":64' in seen["json"].replace(" ", "")
    assert budget.snapshot().tokens_used == 15


def test_ollama_http_error_becomes_llm_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="model not found"))
    llm = OllamaClient("x", 64, transport=transport)
    with pytest.raises(LLMError, match="500"):
        llm.complete("s", "u")


def test_anthropic_request_and_response_mapping(monkeypatch):
    llm = AnthropicClient("claude-haiku-5-5", 128, effort="low", api_key="test-not-a-key")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            model="claude-haiku-5-5",
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="Paris."),
            ],
        )

    monkeypatch.setattr(llm._client.messages, "create", fake_create)
    r = llm.complete("sys", "capital of France?")
    assert r.text == "Paris."
    assert (r.input_tokens, r.output_tokens) == (20, 5)
    assert captured["model"] == "claude-haiku-5-5"
    assert captured["max_tokens"] == 128
    assert captured["system"] == "sys"
    assert captured["output_config"] == {"effort": "low"}
    assert captured["messages"] == [{"role": "user", "content": "capital of France?"}]


def test_anthropic_sdk_has_retries_disabled():
    llm = AnthropicClient("claude-haiku-5-5", 128, api_key="test-not-a-key")
    assert llm._client.max_retries == 0  # retries would be billed outside the budget


def test_factory_selects_backend():
    s = Settings(_env_file=None, llm_backend="fake")
    assert isinstance(build_llm(s), FakeLLM)
    s = Settings(_env_file=None, llm_backend="ollama")
    assert isinstance(build_llm(s), OllamaClient)
    s = Settings(_env_file=None, llm_backend="anthropic", anthropic_api_key="test-not-a-key")
    assert isinstance(build_llm(s), AnthropicClient)
