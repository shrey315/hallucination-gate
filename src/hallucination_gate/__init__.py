"""Alias for ``hallucinate_gate``. Prefer::

    from hallucination_gate import HallucinationGate
"""

from hallucinate_gate import Evidence, GatedAnswer, HallucinationGate, ImageEvidence, TableEvidence

__version__ = "0.4.0"
__all__ = [
    "HallucinationGate",
    "GatedAnswer",
    "Evidence",
    "ImageEvidence",
    "TableEvidence",
]
