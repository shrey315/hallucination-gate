"""Alias for ``hallucinate_gate``. Prefer::

    from hallucination_gate import HallucinationGate, RAGEval, LatencyBudget
"""

from hallucinate_gate import (
    DEFAULT_METRICS,
    Evidence,
    EvalReport,
    GatedAnswer,
    HallucinationGate,
    ImageEvidence,
    LatencyBudget,
    LatencyReport,
    RAGEval,
    RegressionResult,
    RetrievalScores,
    SampleMetrics,
    SampleResult,
    TableEvidence,
    compare_to_baseline,
    evaluate,
    load_baseline,
    save_baseline,
    score_retrieval,
)

__version__ = "0.8.0"
__all__ = [
    "HallucinationGate",
    "GatedAnswer",
    "Evidence",
    "ImageEvidence",
    "TableEvidence",
    "RAGEval",
    "EvalReport",
    "SampleMetrics",
    "SampleResult",
    "DEFAULT_METRICS",
    "evaluate",
    "LatencyBudget",
    "LatencyReport",
    "RegressionResult",
    "RetrievalScores",
    "compare_to_baseline",
    "load_baseline",
    "save_baseline",
    "score_retrieval",
]

try:
    from hallucinate_gate import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover
    pass
