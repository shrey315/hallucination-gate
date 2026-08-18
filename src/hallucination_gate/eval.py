"""Public RAG evaluation API (quality + retrieval + latency + regression)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from bayesian_rag_evaluator.metrics.latency import LatencyBudget, LatencyReport
from bayesian_rag_evaluator.metrics.rag_eval import (
    DEFAULT_METRICS,
    EvalReport,
    RAGEval,
    SampleMetrics,
    SampleResult,
)
from bayesian_rag_evaluator.metrics.regression import (
    RegressionResult,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)
from bayesian_rag_evaluator.metrics.retrieval import RetrievalScores, score_retrieval

__all__ = [
    "DEFAULT_METRICS",
    "EvalReport",
    "RAGEval",
    "SampleMetrics",
    "SampleResult",
    "LatencyBudget",
    "LatencyReport",
    "RegressionResult",
    "RetrievalScores",
    "compare_to_baseline",
    "load_baseline",
    "save_baseline",
    "score_retrieval",
    "evaluate",
]


def evaluate(
    samples: Iterable[dict[str, Any] | Any],
    *,
    use_heuristic: bool | None = None,
    metrics: Sequence[str] | None = None,
    latency_budget: LatencyBudget | None = None,
    baseline_path: str | None = None,
    save_baseline_path: str | None = None,
    fail_on_regression: bool = False,
    fail_on_latency: bool = True,
    **kwargs: Any,
) -> EvalReport:
    """One-shot dataset evaluation with optional latency/regression gates."""
    return RAGEval(
        use_heuristic=use_heuristic,
        latency_budget=latency_budget,
        **kwargs,
    ).evaluate(
        samples,
        metrics=metrics,
        latency_budget=latency_budget,
        baseline_path=baseline_path,
        save_baseline_path=save_baseline_path,
        fail_on_regression=fail_on_regression,
        fail_on_latency=fail_on_latency,
    )
