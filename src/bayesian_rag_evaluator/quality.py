"""Generic quality modes and policy profiles (dataset-agnostic)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

QualityMode = Literal["ci", "quality"]
PolicyName = Literal["strict", "balanced"]


@dataclass(frozen=True)
class PolicyProfile:
    """Tunable claim thresholds. No domain knowledge — numbers only."""

    name: str = "strict"
    min_support_coverage: float = 0.72
    min_support_entail: float = 0.58
    min_lexical_sim: float = 0.50
    min_lexical_entail: float = 0.42
    contradiction_threshold: float = 0.55
    uncertain_support: float = 0.38
    uncertain_coverage: float = 0.40
    # When True, UNCERTAIN-only answers may rewrite instead of abstain if they answer the query.
    allow_uncertain_rewrite: bool = False
    # Keep at most this many query/answer-aligned chunks for scoring (None = all).
    max_aligned_chunks: int | None = None
    # Multi-hop: jointly supported claims. Strict keeps them UNCERTAIN (not a release).
    enable_multihop: bool = True
    allow_inferred_release: bool = False
    # Chunks below this reliability cannot be the sole SUPPORTED citation.
    min_support_reliability: float = 0.45


STRICT = PolicyProfile(name="strict")
BALANCED = PolicyProfile(
    name="balanced",
    min_support_coverage=0.64,
    min_support_entail=0.52,
    min_lexical_sim=0.42,
    min_lexical_entail=0.36,
    contradiction_threshold=0.55,  # keep contradiction bar — false-release lock
    uncertain_support=0.34,
    uncertain_coverage=0.36,
    allow_uncertain_rewrite=True,
    max_aligned_chunks=5,
    enable_multihop=True,
    allow_inferred_release=True,
    min_support_reliability=0.30,
)

PROFILES: dict[str, PolicyProfile] = {
    "strict": STRICT,
    "balanced": BALANCED,
}


def resolve_policy(name: PolicyName | str | PolicyProfile | None) -> PolicyProfile:
    if name is None:
        env = os.getenv("HALLUCINATION_GATE_POLICY", "strict").lower()
        return PROFILES.get(env, STRICT)
    if isinstance(name, PolicyProfile):
        return name
    return PROFILES.get(str(name).lower(), STRICT)


def resolve_mode(mode: QualityMode | str | None = None) -> QualityMode:
    if mode is None:
        raw = os.getenv("HALLUCINATION_GATE_MODE", "").lower()
        if raw in {"ci", "quality"}:
            return raw  # type: ignore[return-value]
        if os.getenv("RAG_EVAL_HEURISTIC", "").lower() in {"1", "true", "yes"}:
            return "ci"
        return "quality"
    m = str(mode).lower()
    if m not in {"ci", "quality"}:
        raise ValueError("mode must be 'ci' or 'quality'")
    return m  # type: ignore[return-value]


def heuristic_for_mode(mode: QualityMode) -> bool:
    """ci → heuristic smoke backends; quality → neural backends."""
    return mode == "ci"
