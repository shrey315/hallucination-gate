from __future__ import annotations

import numpy as np

from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.evidence.backends import EmbeddingBackend, NLIBackend
from bayesian_rag_evaluator.models.schemas import (
    ClaimResult,
    ClaimVerdict,
    EvidenceUnit,
    MediaType,
)

SUPPORT_THRESHOLD = 0.48
CONTRADICTION_THRESHOLD = 0.55
UNCERTAIN_BAND = 0.38
DEFAULT_TOP_K = 8


def verify_claims(
    answer: str,
    units: list[EvidenceUnit],
    embedder: EmbeddingBackend,
    nli: NLIBackend,
    top_k: int = DEFAULT_TOP_K,
) -> list[ClaimResult]:
    claims = extract_claims(answer)
    if not claims:
        return []
    if not units:
        return [
            ClaimResult(
                text=claim,
                status=ClaimVerdict.UNSUPPORTED,
                support_score=0.0,
                contradiction_score=0.0,
            )
            for claim in claims
        ]

    unit_texts = [u.content for u in units]
    sim = embedder.similarity_matrix(claims, unit_texts)
    k = min(top_k, len(units))

    pair_index: list[tuple[int, int]] = []
    pairs: list[tuple[str, str]] = []
    for i, claim in enumerate(claims):
        top = np.argsort(sim[i])[-k:]
        for j in top:
            pair_index.append((i, int(j)))
            pairs.append((unit_texts[int(j)], claim))

    nli_rows = nli.predict_batch(pairs) if pairs else []

    from bayesian_rag_evaluator.evidence.multimodal import extract_numbers

    best_support = [0.0] * len(claims)
    best_contradiction = [0.0] * len(claims)
    best_j = [-1] * len(claims)
    for (i, j), row, sim_ij in zip(
        pair_index, nli_rows, [sim[i, j] for i, j in pair_index], strict=True
    ):
        support = max(float(row["entailment"]), 0.85 * float(sim_ij))
        contra = float(row["contradiction"])
        claim_nums = extract_numbers(claims[i])
        unit_nums = extract_numbers(unit_texts[j])
        if claim_nums and unit_nums and not any(n in unit_nums for n in claim_nums):
            contra = max(contra, 0.78)
            support = min(support, 0.35)
        if support > best_support[i]:
            best_support[i] = support
            best_j[i] = j
        if contra > best_contradiction[i]:
            best_contradiction[i] = contra

    results: list[ClaimResult] = []
    for i, claim in enumerate(claims):
        status = _status(best_support[i], best_contradiction[i])
        src = units[best_j[i]] if best_j[i] >= 0 else None
        results.append(
            ClaimResult(
                text=claim,
                status=status,
                support_score=round(best_support[i], 4),
                contradiction_score=round(best_contradiction[i], 4),
                citation=src.content[:280] if src and status == ClaimVerdict.SUPPORTED else None,
                source_id=src.source_id if src and status == ClaimVerdict.SUPPORTED else None,
                modality=src.modality if src else MediaType.TEXT,
            )
        )
    return results


def _status(support: float, contradiction: float) -> ClaimVerdict:
    if contradiction >= CONTRADICTION_THRESHOLD and contradiction > support:
        return ClaimVerdict.CONTRADICTED
    if support >= SUPPORT_THRESHOLD and support > contradiction:
        return ClaimVerdict.SUPPORTED
    if support >= UNCERTAIN_BAND:
        return ClaimVerdict.UNCERTAIN
    return ClaimVerdict.UNSUPPORTED


def unsupported_ratio(claims: list[ClaimResult]) -> float:
    if not claims:
        return 0.0
    bad = sum(
        1
        for c in claims
        if c.status in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED}
    )
    return bad / len(claims)


def max_contradiction(claims: list[ClaimResult]) -> float:
    if not claims:
        return 0.0
    return max(c.contradiction_score for c in claims)
