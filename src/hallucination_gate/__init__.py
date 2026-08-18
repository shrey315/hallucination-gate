"""Public SDK. Import from here::

    from hallucination_gate import HallucinationGate, RAGEval, LatencyBudget

``bayesian_rag_evaluator`` is the internal engine, not a second product.
``hallucinate_gate`` re-exports this package for older snippets.
"""

from hallucination_gate.evidence import Evidence, ImageEvidence, TableEvidence
from hallucination_gate.eval import (
    DEFAULT_METRICS,
    EvalReport,
    LatencyBudget,
    LatencyReport,
    RAGEval,
    RegressionResult,
    RetrievalScores,
    SampleMetrics,
    SampleResult,
    compare_to_baseline,
    evaluate,
    load_baseline,
    save_baseline,
    score_retrieval,
)
from hallucination_gate.gate import GatedAnswer, HallucinationGate

try:
    from bayesian_rag_evaluator.quality import (
        BALANCED,
        STRICT,
        PolicyProfile,
        resolve_mode,
        resolve_policy,
    )
except Exception:  # pragma: no cover
    BALANCED = STRICT = PolicyProfile = resolve_mode = resolve_policy = None  # type: ignore

__version__ = "0.9.4"
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
    from bayesian_rag_evaluator.evidence.ocr import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover
    pass
