"""Back-compat. Use ``from hallucination_gate.eval import RAGEval``."""

from hallucination_gate.eval import *  # noqa: F403
from hallucination_gate.eval import __all__ as _all

__all__ = list(_all)
