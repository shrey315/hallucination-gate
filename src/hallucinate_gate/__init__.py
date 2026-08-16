"""hallucinate_gate — RAG quality eval + conservative hallucination firewall."""

from hallucinate_gate.evidence import Evidence, ImageEvidence, TableEvidence
from hallucinate_gate.eval import (
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
from hallucinate_gate.gate import GatedAnswer, HallucinationGate

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
    from bayesian_rag_evaluator.evidence.ocr import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover
    pass
