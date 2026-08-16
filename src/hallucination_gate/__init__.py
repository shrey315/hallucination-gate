"""Alias for ``hallucinate_gate``. Prefer::

    from hallucination_gate import HallucinationGate, RAGEval
"""

from hallucinate_gate import (
    DEFAULT_METRICS,
    Evidence,
    EvalReport,
    GatedAnswer,
    HallucinationGate,
    ImageEvidence,
    RAGEval,
    SampleMetrics,
    SampleResult,
    TableEvidence,
    evaluate,
)

__version__ = "0.7.0"
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
]

try:
    from hallucinate_gate import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover
    pass
