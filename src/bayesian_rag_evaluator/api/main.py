from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")
load_dotenv()

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    SafeAnswerResponse,
)
from bayesian_rag_evaluator.observability import REGISTRY

logger = logging.getLogger("bayesian_rag_evaluator.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="Hallucination Gate — Bayesian RAG/LLM Evaluator",
    version="0.6.0",
    description=(
        "Production grounding gate. Public /v1/answer returns only safe_answer. "
        "Internal /evaluate requires an API key when RAG_EVAL_API_KEY is set."
    ),
)

_evaluator: DiagnosticEvaluator | None = None
_pool = ThreadPoolExecutor(max_workers=int(os.getenv("RAG_EVAL_WORKERS", "8")))


def get_evaluator() -> DiagnosticEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = DiagnosticEvaluator()
    return _evaluator


def _api_key() -> str | None:
    return os.getenv("RAG_EVAL_API_KEY") or None


def _timeout_sec() -> float:
    return float(os.getenv("RAG_EVAL_TIMEOUT_SEC", "12"))


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = _api_key()
    if not expected:
        return
    if x_api_key != expected:
        REGISTRY.record_auth_failure()
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _evaluate_timed(request: EvaluateRequest) -> EvaluateResponse:
    try:
        return _pool.submit(get_evaluator().evaluate, request).result(timeout=_timeout_sec())
    except FuturesTimeout:
        REGISTRY.record_timeout()
        raise HTTPException(status_code=504, detail="Evaluation timed out") from None


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = f"{latency:.1f}"
        ok = response.status_code < 400
        if request.url.path not in {"/health", "/metrics"}:
            REGISTRY.record_request(latency, ok=ok)
        logger.info(
            "request_id=%s path=%s status=%s latency_ms=%.1f",
            request_id,
            request.url.path,
            response.status_code,
            latency,
        )
        return response


app.add_middleware(AccessLogMiddleware)


@app.get("/health")
def health() -> dict[str, str]:
    heuristic = os.getenv("RAG_EVAL_HEURISTIC", "").lower() in {"1", "true", "yes"}
    return {
        "status": "ok",
        "version": "0.6.0",
        "backend": "heuristic" if heuristic else "neural",
    }


@app.get("/metrics")
def metrics() -> dict:
    return REGISTRY.snapshot()


@app.post("/evaluate", response_model=EvaluateResponse, dependencies=[Depends(require_api_key)])
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    result = _evaluate_timed(request)
    REGISTRY.record_request(
        result.latency_ms or 0.0, ok=True, gate_action=result.gate.action.value
    )
    return result


@app.post(
    "/evaluate/batch",
    response_model=list[EvaluateResponse],
    dependencies=[Depends(require_api_key)],
)
def evaluate_batch(requests: list[EvaluateRequest]) -> list[EvaluateResponse]:
    results = [_evaluate_timed(req) for req in requests]
    for result in results:
        REGISTRY.record_request(
            result.latency_ms or 0.0, ok=True, gate_action=result.gate.action.value
        )
    return results


@app.post("/v1/answer", response_model=SafeAnswerResponse, dependencies=[Depends(require_api_key)])
def public_answer(request: EvaluateRequest) -> SafeAnswerResponse:
    """User-facing endpoint. Returns only the gated safe_answer — never the raw model text."""
    result = _evaluate_timed(request)
    REGISTRY.record_request(
        result.latency_ms or 0.0, ok=True, gate_action=result.gate.action.value
    )
    return SafeAnswerResponse(
        request_id=result.request_id or str(uuid.uuid4()),
        safe_answer=result.safe_answer,
        released=result.gate.released,
        latency_ms=result.latency_ms or 0.0,
    )
