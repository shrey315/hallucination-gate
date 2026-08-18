from __future__ import annotations

"""Back-compat shim. Use ``from hallucination_gate import HallucinationGate``."""

from hallucination_gate.gate import GatedAnswer, HallucinationGate

__all__ = ["GatedAnswer", "HallucinationGate"]
