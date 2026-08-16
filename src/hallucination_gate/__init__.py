"""Alias for ``hallucinate_gate``. Prefer::

    from hallucination_gate import HallucinationGate, RAGEval, LatencyBudget
"""

from hallucinate_gate import (
    BALANCED,
    DEFAULT_METRICS,
    Evidence,
    EvalReport,
    GatedAnswer,
    HallucinationGate,
    ImageEvidence,
    LatencyBudget,
    LatencyReport,
    PolicyProfile,
    RAGEval,
    RegressionResult,
    RetrievalScores,
    STRICT,
    SampleMetrics,
    SampleResult,
    TableEvidence,
    compare_to_baseline,
    evaluate,
    load_baseline,
    resolve_mode,
    resolve_policy,
    save_baseline,
    score_retrieval,
)

__version__ = "0.9.0"
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
    "BALANCED",
    "STRICT",
    "PolicyProfile",
    "resolve_mode",
    "resolve_policy",
]

try:
    from hallucinate_gate import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover
    pass
