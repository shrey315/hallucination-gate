from __future__ import annotations

"""Back-compat shim. Use ``from hallucinate_gate import HallucinationGate``."""

from hallucinate_gate.gate import GatedAnswer, HallucinationGate

__all__ = ["GatedAnswer", "HallucinationGate"]
