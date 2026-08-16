"""Public RAG evaluation API (RAGAS-class metrics on claim-level grounding)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from bayesian_rag_evaluator.metrics.rag_eval import (
    DEFAULT_METRICS,
    EvalReport,
    RAGEval,
    SampleMetrics,
    SampleResult,
)

__all__ = [
    "DEFAULT_METRICS",
    "EvalReport",
    "RAGEval",
    "SampleMetrics",
    "SampleResult",
]


def evaluate(
    samples: Iterable[dict[str, Any] | Any],
    *,
    use_heuristic: bool | None = None,
    metrics: Sequence[str] | None = None,
    **kwargs: Any,
) -> EvalReport:
    """One-shot dataset evaluation."""
    return RAGEval(use_heuristic=use_heuristic, **kwargs).evaluate(
        samples, metrics=metrics
    )
