"""Conservative claim-support policy.

Similarity alone is never enough to mark a claim supported. Invented entities
must fail coverage. This is the lock that keeps false-release near zero.
"""

from __future__ import annotations

from bayesian_rag_evaluator.evidence.backends import content_tokens, token_coverage, token_set
from bayesian_rag_evaluator.evidence.multimodal import extract_numbers
from bayesian_rag_evaluator.models.schemas import ClaimVerdict

# Release requires real overlap with evidence, not topical similarity.
MIN_SUPPORT_COVERAGE = 0.72
MIN_SUPPORT_ENTAIL = 0.58
MIN_LEXICAL_SIM = 0.50
MIN_LEXICAL_ENTAIL = 0.42
CONTRADICTION_THRESHOLD = 0.55
UNCERTAIN_SUPPORT = 0.38
UNCERTAIN_COVERAGE = 0.40


def numbers_agree(claim: str, evidence: str) -> bool | None:
    """True if claim numbers appear in evidence, False if both have numbers and they clash, None if no claim numbers."""
    claim_nums = extract_numbers(claim)
    if not claim_nums:
        return None
    evidence_nums = extract_numbers(evidence)
    if not evidence_nums:
        return False
    claim_vals = _to_floats(claim_nums)
    evid_vals = _to_floats(evidence_nums)
    for value in claim_vals:
        if any(abs(value - other) < 1e-6 for other in evid_vals):
            return True
    return False


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
    lexical = 0.65 * coverage + 0.35 * similarity
    return max(0.0, min(1.0, 0.55 * entailment + 0.45 * lexical))


def decide_status(
    entailment: float,
    similarity: float,
    coverage: float,
    contradiction: float,
    numbers_ok: bool | None,
    extra_distinctive: int = 0,
    literals_ok: bool | None = None,
) -> ClaimVerdict:
    if numbers_ok is False or literals_ok is False:
        contradiction = max(contradiction, 0.78)
        entailment = min(entailment, 0.30)

    support = fused_support(entailment, similarity, coverage)

    if contradiction >= CONTRADICTION_THRESHOLD and contradiction >= support:
        return ClaimVerdict.CONTRADICTED

    # Invented extra assertions cannot ride a copied prefix.
    strong_nli = entailment >= MIN_SUPPORT_ENTAIL and coverage >= 0.50
    if extra_distinctive >= 2:
        strong_nli = False
    strong_lex = (
        extra_distinctive == 0
        and coverage >= MIN_SUPPORT_COVERAGE
        and similarity >= MIN_LEXICAL_SIM
        and entailment >= MIN_LEXICAL_ENTAIL
    )
    if (strong_nli or strong_lex) and contradiction < CONTRADICTION_THRESHOLD:
        return ClaimVerdict.SUPPORTED

    if extra_distinctive >= 2:
        return ClaimVerdict.UNSUPPORTED

    if support >= UNCERTAIN_SUPPORT and coverage >= UNCERTAIN_COVERAGE:
        return ClaimVerdict.UNCERTAIN
    return ClaimVerdict.UNSUPPORTED


def extra_distinctive_tokens(claim: str, evidence: str) -> set[str]:
    from bayesian_rag_evaluator.evidence.synonyms import covers_token

    extra: set[str] = set()
    evidence_toks = token_set(evidence)
    for tok in content_tokens(claim):
        if len(tok) < 6:
            continue
        if not covers_token(tok, evidence_toks):
            extra.add(tok)
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
