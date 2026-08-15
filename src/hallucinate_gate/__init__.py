"""hallucinate_gate — drop-in hallucination firewall for any RAG or fine-tuned model."""

from hallucinate_gate.evidence import Evidence, ImageEvidence, TableEvidence
from hallucinate_gate.gate import GatedAnswer, HallucinationGate

__version__ = "0.6.2"
__all__ = [
    "HallucinationGate",
    "GatedAnswer",
    "Evidence",
    "ImageEvidence",
    "TableEvidence",
]

# Optional OCR helpers (import path stays stable even if engines are missing).
try:
    from bayesian_rag_evaluator.evidence.ocr import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover - never fail package import
    pass
