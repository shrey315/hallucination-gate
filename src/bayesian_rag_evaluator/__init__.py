"""Bayesian Network RAG/LLM Diagnostic Evaluator (engine).

Prefer the public library API::

    from hallucination_gate import HallucinationGate

This package intentionally does **not** re-export HallucinationGate here, so
importing engine submodules cannot cycle through the public SDK.
"""

__version__ = "0.9.3"
__all__: list[str] = []
