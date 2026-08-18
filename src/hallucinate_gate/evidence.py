"""Back-compat. Use ``from hallucination_gate.evidence import Evidence``."""

from hallucination_gate.evidence import Evidence, ImageEvidence, TableEvidence

__all__ = ["Evidence", "ImageEvidence", "TableEvidence"]
