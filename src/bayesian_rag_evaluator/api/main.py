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
from fastapi.responses import PlainTextResponse
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
    title="Hallucination Gate",
    version="0.9.4",
    description=(
        "Conservative grounding sidecar. Claim status decides release; BN scores are "
        "diagnostic fusion, not calibrated P(hallucination). Public /v1/answer returns "
        "only safe_answer. Auth is per-key tenant labels in one process — not a SaaS platform."
    ),
)

_evaluator: DiagnosticEvaluator | None = None
_pool = ThreadPoolExecutor(max_workers=int(os.getenv("RAG_EVAL_WORKERS", "8")))
_SKIP_METRICS = {"/health", "/metrics", "/metrics.json"}


def get_evaluator() -> DiagnosticEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = DiagnosticEvaluator()
    return _evaluator


def _timeout_sec() -> float:
    return float(os.getenv("RAG_EVAL_TIMEOUT_SEC", "12"))


def _api_keys() -> dict[str, str]:
    """Map presented secret → tenant id. Shared process; labels only."""
    keys: dict[str, str] = {}
    single = os.getenv("RAG_EVAL_API_KEY") or ""
    if single.strip():
        keys[single.strip()] = os.getenv("RAG_EVAL_TENANT", "default")
    multi = os.getenv("RAG_EVAL_API_KEYS") or ""
    for part in multi.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        tenant, secret = part.split(":", 1)
        tenant, secret = tenant.strip(), secret.strip()
        if tenant and secret:
            keys[secret] = tenant
    return keys


async def require_api_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
) -> str:
    keys = _api_keys()
    if not keys:
        return (x_tenant_id or "default").strip() or "default"
    presented = x_api_key
    if not presented and authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    tenant = keys.get(presented or "")
    if tenant is None:
        REGISTRY.record_auth_failure()
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if x_tenant_id and x_tenant_id.strip() and x_tenant_id.strip() != tenant:
        REGISTRY.record_auth_failure()
        raise HTTPException(status_code=403, detail="Tenant does not match API key")
    return tenant


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
        tenant = request.headers.get("x-tenant-id") or "default"
        response = await call_next(request)
        latency = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        response.headers["x-latency-ms"] = f"{latency:.1f}"
        ok = response.status_code < 400
        if request.url.path not in _SKIP_METRICS:
            REGISTRY.record_request(latency, ok=ok, tenant=tenant)
        logger.info(
            "request_id=%s tenant=%s path=%s status=%s latency_ms=%.1f",
            request_id,
            tenant,
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
        "version": "0.9.4",
        "backend": "heuristic" if heuristic else "neural",
        "release_authority": "claim_status",
        "scores_are_calibrated": "false",
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_prometheus() -> str:
    return REGISTRY.prometheus()


@app.get("/metrics.json")
def metrics_json() -> dict:
    return REGISTRY.snapshot()


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest, tenant: str = Depends(require_api_key)) -> EvaluateResponse:
    result = _evaluate_timed(request)
    REGISTRY.record_request(
        result.latency_ms or 0.0,
        ok=True,
        gate_action=result.gate.action.value,
        tenant=tenant,
        evidence_gap=result.gate.evidence_gap.value,
    )
    return result


@app.post("/evaluate/batch", response_model=list[EvaluateResponse])
def evaluate_batch(
    requests: list[EvaluateRequest], tenant: str = Depends(require_api_key)
) -> list[EvaluateResponse]:
    results = [_evaluate_timed(req) for req in requests]
    for result in results:
        REGISTRY.record_request(
            result.latency_ms or 0.0,
            ok=True,
            gate_action=result.gate.action.value,
            tenant=tenant,
            evidence_gap=result.gate.evidence_gap.value,
        )
    return results


@app.post("/v1/answer", response_model=SafeAnswerResponse)
def public_answer(
    request: EvaluateRequest, tenant: str = Depends(require_api_key)
) -> SafeAnswerResponse:
    """User-facing endpoint. Returns only the gated safe_answer — never the raw model text."""
    result = _evaluate_timed(request)
    REGISTRY.record_request(
        result.latency_ms or 0.0,
        ok=True,
        gate_action=result.gate.action.value,
        tenant=tenant,
        evidence_gap=result.gate.evidence_gap.value,
    )
    return SafeAnswerResponse(
        request_id=result.request_id or str(uuid.uuid4()),
        safe_answer=result.safe_answer,
        released=result.gate.released,
        latency_ms=result.latency_ms or 0.0,
        evidence_gap=result.gate.evidence_gap,
    )
