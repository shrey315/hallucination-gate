"""hallucinate_gate — drop-in hallucination firewall for any RAG or fine-tuned model."""

from hallucinate_gate.evidence import Evidence, ImageEvidence, TableEvidence
from hallucinate_gate.gate import GatedAnswer, HallucinationGate

__version__ = "0.4.0"
__all__ = [
    "HallucinationGate",
    "GatedAnswer",
    "Evidence",
    "ImageEvidence",
    "TableEvidence",
]
