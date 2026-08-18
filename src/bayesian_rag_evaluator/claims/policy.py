"""Conservative claim-support policy.

Similarity alone is never enough to mark a claim supported. Invented entities
must fail coverage. This is the lock that keeps false-release near zero.
"""

from __future__ import annotations

from bayesian_rag_evaluator.evidence.backends import content_tokens, token_coverage, token_set
from bayesian_rag_evaluator.evidence.multimodal import extract_numbers
from bayesian_rag_evaluator.models.schemas import ClaimVerdict
from bayesian_rag_evaluator.quality import PolicyProfile

# Release requires real overlap with evidence, not topical similarity.
MIN_SUPPORT_COVERAGE = 0.72
MIN_SUPPORT_ENTAIL = 0.58
MIN_LEXICAL_SIM = 0.50
MIN_LEXICAL_ENTAIL = 0.42
CONTRADICTION_THRESHOLD = 0.55
UNCERTAIN_SUPPORT = 0.38
UNCERTAIN_COVERAGE = 0.40
# Neighbor chunks below this are ignored for contradiction vetoes.
ALIGN_COVERAGE = 0.35
ALIGN_SIMILARITY = 0.40


def is_chunk_aligned(coverage: float, similarity: float) -> bool:
    """True when a chunk is topical enough that disagreement can veto a claim."""
    return coverage >= ALIGN_COVERAGE or similarity >= ALIGN_SIMILARITY


def status_reason(
    status: ClaimVerdict,
    *,
    numbers_ok: bool | None,
    literals_ok: bool | None,
    extra_distinctive: int,
    coverage: float,
    contradiction: float,
    logic_flags: list[str] | None = None,
) -> str:
    if numbers_ok is False:
        return "numeric mismatch with this chunk"
    if literals_ok is False:
        return "quoted literal missing from this chunk"
    if logic_flags:
        return "logic mismatch: " + ",".join(logic_flags)
    if status == ClaimVerdict.CONTRADICTED:
        if extra_distinctive >= 2:
            return "claim adds content words absent from this chunk"
        return f"NLI/heuristic contradiction={contradiction:.2f}"
    if status == ClaimVerdict.SUPPORTED:
        return f"coverage={coverage:.2f}"
    if status == ClaimVerdict.UNCERTAIN:
        return f"weak overlap coverage={coverage:.2f}"
    return f"insufficient grounding coverage={coverage:.2f}"


def numbers_agree(claim: str, evidence: str) -> bool | None:
    """True iff every claim number appears in evidence. None if the claim has none."""
    claim_nums = extract_numbers(claim)
    if not claim_nums:
        return None
    evidence_nums = extract_numbers(evidence)
    if not evidence_nums:
        return False
    claim_vals = _to_floats(claim_nums)
    evid_vals = _to_floats(evidence_nums)
    if not claim_vals:
        return None
    for value in claim_vals:
        if not any(abs(value - other) < 1e-6 for other in evid_vals):
            return False
    return True


def _to_floats(nums: list[str]) -> list[float]:
    out: list[float] = []
    for num in nums:
        try:
            out.append(float(num.rstrip("%")))
        except ValueError:
            continue
    return out


def fused_support(entailment: float, similarity: float, coverage: float) -> float:
    """Weighted support. Does not allow similarity to override missing coverage."""
    from bayesian_rag_evaluator.claims.fusion import fused_support as _fused

    return _fused(entailment, similarity, coverage)


def decide_status(
    entailment: float,
    similarity: float,
    coverage: float,
    contradiction: float,
    numbers_ok: bool | None,
    extra_distinctive: int = 0,
    literals_ok: bool | None = None,
    profile: PolicyProfile | None = None,
    logic_flags: list[str] | None = None,
) -> ClaimVerdict:
    from bayesian_rag_evaluator.quality import STRICT

    p = profile or STRICT
    flags = logic_flags or []
    if flags:
        from bayesian_rag_evaluator.claims.logic import logic_penalty

        contradiction = min(1.0, contradiction + logic_penalty(flags))
    if numbers_ok is False or literals_ok is False:
        contradiction = max(contradiction, 0.78)
        entailment = min(entailment, 0.30)

    support = fused_support(entailment, similarity, coverage)

    if contradiction >= p.contradiction_threshold and contradiction >= support:
        return ClaimVerdict.CONTRADICTED

    # Invented extra assertions cannot ride a copied prefix.
    strong_nli = entailment >= p.min_support_entail and coverage >= 0.50
    if extra_distinctive >= 2:
        strong_nli = False
    strong_lex = (
        extra_distinctive == 0
        and coverage >= p.min_support_coverage
        and similarity >= p.min_lexical_sim
        and entailment >= p.min_lexical_entail
    )
    if (strong_nli or strong_lex) and contradiction < p.contradiction_threshold:
        return ClaimVerdict.SUPPORTED

    if extra_distinctive >= 2:
        return ClaimVerdict.UNSUPPORTED

    if support >= p.uncertain_support and coverage >= p.uncertain_coverage:
        return ClaimVerdict.UNCERTAIN
    return ClaimVerdict.UNSUPPORTED


def extra_distinctive_tokens(claim: str, evidence: str) -> set[str]:
    from bayesian_rag_evaluator.claims.special import (
        extra_code_tokens,
        fluent_unattested_justification,
    )
    from bayesian_rag_evaluator.evidence.synonyms import covers_token

    extra: set[str] = set()
    evidence_toks = token_set(evidence)
    for tok in content_tokens(claim):
        if len(tok) < 6:
            continue
        if not covers_token(tok, evidence_toks):
            extra.add(tok)
    extra |= extra_code_tokens(claim, evidence)
    if fluent_unattested_justification(claim, evidence):
        extra.add("fluent_continuation")
        extra.add("unattested_tail")
    return extra


def extra_entity_penalty(claim: str, evidence: str) -> float:
    """Raise contradiction when the claim introduces content words absent from evidence."""
    distinctive = extra_distinctive_tokens(claim, evidence)
    if len(distinctive) >= 2:
        return 0.35
    if len(distinctive) == 1 and token_coverage(claim, evidence) < 0.55:
        return 0.18
    return 0.0


def literals_agree(claim: str, evidence: str) -> bool | None:
    """Quoted / code-like literals in the claim must appear in evidence."""
    import re

    lit = re.findall(r"['\"]([^'\"]{3,})['\"]", claim)
    if not lit:
        return None
    blob = evidence
    return all(item in blob for item in lit)
