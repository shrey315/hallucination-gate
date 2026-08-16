"""RAGAS-class RAG evaluation on top of claim-level grounding.

Uses the same embed/NLI stack as the gate. Faithfulness is claim↔chunk
soft-OR (stronger than whole-answer NLI). Context precision/recall are
first-class when labels or ground truth are provided; otherwise honest proxies.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bayesian_rag_evaluator.adapters import normalize_context
from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.claims.policy import is_chunk_aligned
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.evidence.backends import token_coverage
from bayesian_rag_evaluator.metrics.latency import (
    LatencyBudget,
    LatencyReport,
    check_latency_budget,
)
from bayesian_rag_evaluator.metrics.regression import (
    RegressionResult,
    compare_to_baseline,
    save_baseline,
)
from bayesian_rag_evaluator.metrics.retrieval import (
    RetrievalScores,
    aggregate_retrieval,
    score_retrieval,
)
from bayesian_rag_evaluator.models.schemas import (
    ClaimVerdict,
    EvaluateRequest,
    EvaluateResponse,
    ModelType,
)

DEFAULT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "groundedness",
    "hallucination_risk",
    "release_safety",
)


@dataclass
class SampleMetrics:
    """Per-sample quality scores (RAGAS-like + gate posteriors)."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    groundedness: float | None = None
    hallucination_risk: float | None = None
    release_safety: float | None = None
    gate_action: str | None = None
    released: bool | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleResult:
    query: str
    answer: str
    contexts: list[str]
    scores: SampleMetrics
    claims: list[dict[str, Any]] = field(default_factory=list)
    safe_answer: str | None = None
    ground_truth: str | None = None
    latency_ms: float | None = None
    retrieval: RetrievalScores | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
            "scores": self.scores.as_dict(),
            "claims": self.claims,
            "safe_answer": self.safe_answer,
            "latency_ms": self.latency_ms,
            "retrieval": self.retrieval.as_dict() if self.retrieval else None,
        }


@dataclass
class EvalReport:
    n: int
    aggregate: dict[str, float]
    samples: list[SampleResult]
    metrics: list[str]
    retrieval: dict[str, float] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    regression: dict[str, Any] | None = None
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    framework: str = "hallucination-gate"
    compared_to: str = "ragas-like claim-level eval + retrieval + latency"

    def as_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "compared_to": self.compared_to,
            "n": self.n,
            "metrics": self.metrics,
            "aggregate": self.aggregate,
            "retrieval": self.retrieval,
            "latency": self.latency,
            "regression": self.regression,
            "ok": self.ok,
            "failures": self.failures,
            "samples": [s.as_dict() for s in self.samples],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        payload = json.dumps(self.as_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.as_dict(), ensure_ascii=False) + "\n")

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise AssertionError("; ".join(self.failures) or "eval budget/regression failed")


def score_sample_from_response(
    response: EvaluateResponse,
    *,
    contexts: list[str],
    ground_truth: str | None = None,
    relevant_contexts: list[str] | None = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> SampleMetrics:
    """Map one EvaluateResponse into named RAG-eval metrics."""
    wanted = set(metrics)
    notes: list[str] = []
    claims = response.claims
    n_claims = len(claims)
    supported = sum(1 for c in claims if c.status == ClaimVerdict.SUPPORTED)
    contradicted = sum(1 for c in claims if c.status == ClaimVerdict.CONTRADICTED)

    faithfulness: float | None = None
    if "faithfulness" in wanted:
        if n_claims == 0:
            faithfulness = 0.0
            notes.append("faithfulness: no extractable claims → 0")
        elif contradicted:
            # Any contradiction tanks faithfulness (conservative, better firewall).
            faithfulness = max(0.0, supported / n_claims * (1.0 - contradicted / n_claims))
        else:
            faithfulness = supported / n_claims

    answer_relevancy = (
        float(response.evidence.query_relevance) if "answer_relevancy" in wanted else None
    )

    context_precision: float | None = None
    if "context_precision" in wanted:
        context_precision = _context_precision(
            claims, contexts, relevant_contexts=relevant_contexts, notes=notes
        )

    context_recall: float | None = None
    if "context_recall" in wanted:
        context_recall = _context_recall(
            contexts, ground_truth=ground_truth, response=response, notes=notes
        )

    groundedness = float(response.scores.groundedness) if "groundedness" in wanted else None
    hallucination_risk = (
        float(response.scores.hallucination_risk) if "hallucination_risk" in wanted else None
    )
    release_safety = (
        float(response.scores.release_safety) if "release_safety" in wanted else None
    )

    return SampleMetrics(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
        groundedness=groundedness,
        hallucination_risk=hallucination_risk,
        release_safety=release_safety,
        gate_action=response.gate.action.value,
        released=response.gate.released,
        notes=notes,
    )


def _context_precision(
    claims: list[Any],
    contexts: list[str],
    *,
    relevant_contexts: list[str] | None,
    notes: list[str],
) -> float:
    if not contexts:
        notes.append("context_precision: no contexts → 0")
        return 0.0

    if relevant_contexts:
        rel = {_norm(c) for c in relevant_contexts}
        hits = sum(1 for c in contexts if _norm(c) in rel)
        return hits / len(contexts)

    # Proxy: chunk is "precise" if it is aligned/supportive for any claim hit.
    useful = 0
    for i, chunk in enumerate(contexts):
        sid = f"context:{i}"
        chunk_useful = False
        for claim in claims:
            for hit in getattr(claim, "chunk_hits", None) or []:
                if hit.source_id == sid and (
                    hit.status == ClaimVerdict.SUPPORTED
                    or (
                        is_chunk_aligned(hit.coverage, hit.similarity)
                        and hit.support_score >= 0.45
                    )
                ):
                    chunk_useful = True
                    break
            if chunk_useful:
                break
            # Fallback without chunk_hits: lexical coverage of claim in chunk.
            if token_coverage(claim.text, chunk) >= 0.45:
                chunk_useful = True
                break
        if chunk_useful:
            useful += 1
    notes.append("context_precision: claim-aligned proxy (pass relevant_contexts for labeled metric)")
    return useful / len(contexts)


def _context_recall(
    contexts: list[str],
    *,
    ground_truth: str | None,
    response: EvaluateResponse,
    notes: list[str],
) -> float | None:
    if not ground_truth or not ground_truth.strip():
        notes.append("context_recall: skipped (provide ground_truth / reference answer)")
        return None
    if not contexts:
        return 0.0

    facts = extract_claims(ground_truth) or [ground_truth.strip()]
    covered = 0
    blob_parts = list(contexts)
    # Also allow answer-supported facts only when they are context-grounded claims.
    supported_claim_texts = {
        c.text for c in response.claims if c.status == ClaimVerdict.SUPPORTED
    }
    for fact in facts:
        if any(token_coverage(fact, chunk) >= 0.55 for chunk in blob_parts):
            covered += 1
            continue
        if any(token_coverage(fact, s) >= 0.72 for s in supported_claim_texts):
            covered += 1
    return covered / len(facts)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _aggregate(samples: list[SampleResult], metrics: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in metrics:
        values = [
            getattr(s.scores, name)
            for s in samples
            if getattr(s.scores, name, None) is not None
        ]
        if values:
            out[name] = round(sum(values) / len(values), 4)
    released = [s.scores.released for s in samples if s.scores.released is not None]
    if released:
        out["release_rate"] = round(sum(1 for r in released if r) / len(released), 4)
    return out


def normalize_sample(sample: dict[str, Any] | Any) -> dict[str, Any]:
    """Accept dicts or objects with query/answer/contexts fields."""
    if isinstance(sample, dict):
        data = sample
    else:
        data = {
            "query": getattr(sample, "query", ""),
            "answer": getattr(sample, "answer", "") or getattr(sample, "response", ""),
            "contexts": getattr(sample, "contexts", None)
            or getattr(sample, "context", None)
            or getattr(sample, "context_chunks", None),
            "ground_truth": getattr(sample, "ground_truth", None)
            or getattr(sample, "reference", None),
            "relevant_contexts": getattr(sample, "relevant_contexts", None),
            "relevant_indices": getattr(sample, "relevant_indices", None),
            "kb": getattr(sample, "kb", None),
        }
    contexts = normalize_context(
        data.get("contexts")
        or data.get("context")
        or data.get("context_chunks")
        or []
    )
    raw_idx = data.get("relevant_indices") or data.get("qrels") or []
    indices: list[int] = []
    if isinstance(raw_idx, dict):
        indices = [int(k) for k, v in raw_idx.items() if v]
    else:
        indices = [int(i) for i in raw_idx]
    return {
        "query": str(data.get("query") or ""),
        "answer": str(data.get("answer") or data.get("response") or ""),
        "contexts": contexts,
        "ground_truth": data.get("ground_truth") or data.get("reference"),
        "relevant_contexts": normalize_context(data.get("relevant_contexts") or []),
        "relevant_indices": indices,
        "kb": normalize_context(data.get("kb") or data.get("kb_chunks") or []),
    }


class RAGEval:
    """Dataset-level RAG quality system: grounding metrics + retrieval + latency + regression."""

    def __init__(
        self,
        *,
        use_heuristic: bool | None = None,
        embed_model: str | None = None,
        nli_model: str | None = None,
        evaluator: DiagnosticEvaluator | None = None,
        metrics: Sequence[str] = DEFAULT_METRICS,
        latency_budget: LatencyBudget | None = None,
    ) -> None:
        self.metrics = list(metrics)
        self.latency_budget = latency_budget
        self._evaluator = evaluator or DiagnosticEvaluator(
            use_heuristic=use_heuristic,
            embed_model=embed_model,
            nli_model=nli_model,
        )

    def evaluate(
        self,
        samples: Iterable[dict[str, Any] | Any],
        *,
        metrics: Sequence[str] | None = None,
        latency_budget: LatencyBudget | None = None,
        baseline_path: str | Path | None = None,
        save_baseline_path: str | Path | None = None,
        fail_on_regression: bool = False,
        fail_on_latency: bool = True,
    ) -> EvalReport:
        metric_names = list(metrics or self.metrics)
        budget = latency_budget if latency_budget is not None else self.latency_budget
        rows: list[SampleResult] = []
        retrieval_rows: list[RetrievalScores] = []
        for raw in samples:
            item = normalize_sample(raw)
            request = EvaluateRequest(
                query=item["query"] or " ",
                answer=item["answer"] or " ",
                context_chunks=item["contexts"],
                kb_chunks=item["kb"],
                model_type=ModelType.RAG,
                strict=True,
            )
            response = self._evaluator.evaluate(request)
            scores = score_sample_from_response(
                response,
                contexts=item["contexts"],
                ground_truth=item["ground_truth"],
                relevant_contexts=item["relevant_contexts"] or None,
                metrics=metric_names,
            )
            retrieval = score_retrieval(
                item["contexts"],
                relevant_contexts=item["relevant_contexts"] or None,
                relevant_indices=item["relevant_indices"] or None,
            )
            if retrieval.mrr is not None:
                retrieval_rows.append(retrieval)
            rows.append(
                SampleResult(
                    query=item["query"],
                    answer=item["answer"],
                    contexts=item["contexts"],
                    ground_truth=item["ground_truth"],
                    scores=scores,
                    claims=[c.model_dump(mode="json") for c in response.claims],
                    safe_answer=response.safe_answer,
                    latency_ms=response.latency_ms,
                    retrieval=retrieval,
                )
            )

        latencies = [s.latency_ms for s in rows if s.latency_ms is not None]
        latency_report: LatencyReport = check_latency_budget(latencies, budget)
        retrieval_agg = aggregate_retrieval(retrieval_rows)

        failures: list[str] = []
        if fail_on_latency and not latency_report.ok:
            failures.extend(latency_report.failures)

        report = EvalReport(
            n=len(rows),
            aggregate=_aggregate(rows, metric_names),
            samples=rows,
            metrics=metric_names,
            retrieval=retrieval_agg,
            latency=latency_report.as_dict(),
            ok=not failures,
            failures=list(failures),
        )

        if save_baseline_path:
            save_baseline(report, save_baseline_path)

        regression: RegressionResult | None = None
        if baseline_path:
            regression = compare_to_baseline(report, baseline_path)
            report.regression = regression.as_dict()
            if fail_on_regression and not regression.passed:
                report.ok = False
                report.failures = list(report.failures) + list(regression.failures)

        return report

    def evaluate_paths(
        self,
        path: str | Path,
        *,
        metrics: Sequence[str] | None = None,
        latency_budget: LatencyBudget | None = None,
        baseline_path: str | Path | None = None,
        save_baseline_path: str | Path | None = None,
        fail_on_regression: bool = False,
        fail_on_latency: bool = True,
    ) -> EvalReport:
        """Load JSON list or JSONL of samples and evaluate."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            samples = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            samples = data if isinstance(data, list) else data.get("samples", [])
        return self.evaluate(
            samples,
            metrics=metrics,
            latency_budget=latency_budget,
            baseline_path=baseline_path,
            save_baseline_path=save_baseline_path,
            fail_on_regression=fail_on_regression,
            fail_on_latency=fail_on_latency,
        )
