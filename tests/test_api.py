from fastapi.testclient import TestClient

from conftest import WordCheck, configured
from firewall.api import create_app
from firewall.config import Settings
from firewall.llm.budget import Budget
from firewall.llm.fake import FakeLLM
from firewall.pipeline import Firewall
from firewall.policy import Policy, PolicyConfig


def client_with(llm, input_checks=(), max_chars=20_000):
    policy = Policy(config=PolicyConfig(), input_checks=list(input_checks))
    fw = Firewall(policy, llm, system_prompt="sys")
    settings = Settings(_env_file=None, llm_backend="fake", max_message_chars=max_chars)
    return TestClient(create_app(settings=settings, firewall=fw))


def test_healthz():
    r = client_with(FakeLLM()).get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "fake", "model": "fake"}


def test_chat_returns_answer_and_decision_record():
    c = client_with(FakeLLM(reply="hi there"), [configured(WordCheck("pin", "pin"), "mask")])
    r = c.post("/chat", json={"message": "my pin is 1234"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "hi there"
    d = body["decision"]
    assert d["final_action"] == "mask"
    assert d["llm"]["called"] is True
    assert d["input_checks"][0]["findings"] == [
        {"type": "pin", "start": 3, "end": 6, "score": None, "detail": None}
    ]
    assert set(d) >= {"request_id", "total_ms", "overhead_ms", "output_checks"}


def test_decision_record_never_contains_message_text():
    c = client_with(FakeLLM(), [configured(WordCheck("hunter2", "password", name="pw"), "mask")])
    r = c.post("/chat", json={"message": "password hunter2"})
    assert "hunter2" not in str(r.json()["decision"])


def test_budget_exhausted_returns_429():
    c = client_with(FakeLLM(budget=Budget(max_calls=1, max_tokens=100_000)))
    assert c.post("/chat", json={"message": "a"}).status_code == 200
    r = c.post("/chat", json={"message": "b"})
    assert r.status_code == 429
    assert "spending cap" in r.json()["error"]


def test_llm_failure_returns_502():
    r = client_with(FakeLLM(fail=True)).post("/chat", json={"message": "a"})
    assert r.status_code == 502
    assert r.json()["request_id"]


def test_too_long_returns_413_and_empty_returns_422():
    c = client_with(FakeLLM(), max_chars=10)
    assert c.post("/chat", json={"message": "x" * 11}).status_code == 413
    assert c.post("/chat", json={"message": ""}).status_code == 422
    assert c.post("/chat", json={}).status_code == 422
