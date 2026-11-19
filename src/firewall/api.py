"""HTTP API: POST /chat runs one message through the firewall."""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from firewall.config import Settings
from firewall.llm import Budget, BudgetExceeded, LLMError, build_llm
from firewall.pipeline import Firewall
from firewall.policy import load_policy
from firewall.schemas import ChatRequest, ChatResponse, ErrorResponse

decision_log = logging.getLogger("firewall.decision")


def build_firewall(settings: Settings) -> Firewall:
    budget = Budget(settings.budget_max_calls, settings.budget_max_tokens)
    return Firewall(
        policy=load_policy(settings.policy_path),
        llm=build_llm(settings, budget=budget),
        system_prompt=settings.system_prompt,
    )


def create_app(settings: Settings | None = None, firewall: Firewall | None = None) -> FastAPI:
    settings = settings or Settings()
    fw = firewall or build_firewall(settings)
    app = FastAPI(title="prompt-firewall", version="0.1.0")
    app.state.firewall = fw

    def error(status: int, msg: str, request_id: str) -> JSONResponse:
        body = ErrorResponse(error=msg, request_id=request_id).model_dump()
        return JSONResponse(status_code=status, content=body)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "backend": fw.llm.backend, "model": fw.llm.model}

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            413: {"model": ErrorResponse},
            429: {"model": ErrorResponse},
            502: {"model": ErrorResponse},
        },
    )
    def chat(req: ChatRequest):
        request_id = uuid.uuid4().hex
        if len(req.message) > settings.max_message_chars:
            return error(413, f"message longer than {settings.max_message_chars} chars", request_id)
        try:
            result = fw.handle(req.message, request_id=request_id)
        except BudgetExceeded as e:
            return error(429, f"spending cap reached: {e}", request_id)
        except LLMError as e:
            return error(502, f"LLM backend failed: {e}", request_id)
        # The decision record holds check names, actions and spans, never message text.
        decision_log.info(json.dumps(result.decision.model_dump(mode="json")))
        return ChatResponse(answer=result.answer, decision=result.decision)

    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory firewall.api:app_factory`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app()
