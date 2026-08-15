"""Alias for ``hallucinate_gate``. Prefer::

    from hallucination_gate import HallucinationGate
"""

from hallucinate_gate import (
    Evidence,
    GatedAnswer,
    HallucinationGate,
    ImageEvidence,
    TableEvidence,
)

__version__ = "0.6.3"
__all__ = [
    "HallucinationGate",
    "GatedAnswer",
    "Evidence",
    "ImageEvidence",
    "TableEvidence",
]

try:
    from hallucinate_gate import OcrResult, ocr_available, ocr_image

    __all__ += ["OcrResult", "ocr_available", "ocr_image"]
except Exception:  # pragma: no cover
    pass
