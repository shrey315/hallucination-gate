from __future__ import annotations

import numpy as np

from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.claims.policy import (
    decide_status,
    extra_distinctive_tokens,
    extra_entity_penalty,
    fused_support,
    literals_agree,
    numbers_agree,
)
from bayesian_rag_evaluator.evidence.backends import (
    EmbeddingBackend,
    NLIBackend,
    token_coverage,
)
from bayesian_rag_evaluator.models.schemas import (
    ClaimResult,
    ClaimVerdict,
    EvidenceUnit,
    MediaType,
)

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

    best_support = [0.0] * len(claims)
    best_contradiction = [0.0] * len(claims)
    best_entail = [0.0] * len(claims)
    best_sim = [0.0] * len(claims)
    best_cov = [0.0] * len(claims)
    best_nums: list[bool | None] = [None] * len(claims)
    best_lits: list[bool | None] = [None] * len(claims)
    best_extra = [0] * len(claims)
    best_j = [-1] * len(claims)

    for (i, j), row, sim_ij in zip(
        pair_index, nli_rows, [sim[i, j] for i, j in pair_index], strict=True
    ):
        coverage = token_coverage(claims[i], unit_texts[j])
        nums_ok = numbers_agree(claims[i], unit_texts[j])
        lits_ok = literals_agree(claims[i], unit_texts[j])
        extra_n = len(extra_distinctive_tokens(claims[i], unit_texts[j]))
        contra = float(row["contradiction"]) + extra_entity_penalty(
            claims[i], unit_texts[j]
        )
        contra = min(1.0, contra)
        entail = float(row["entailment"])
        if nums_ok is False or lits_ok is False:
            contra = max(contra, 0.78)
            entail = min(entail, 0.30)
        support = fused_support(entail, float(sim_ij), coverage)
        if support > best_support[i]:
            best_support[i] = support
            best_entail[i] = entail
            best_sim[i] = float(sim_ij)
            best_cov[i] = coverage
            best_nums[i] = nums_ok
            best_lits[i] = lits_ok
            best_extra[i] = extra_n
            best_j[i] = j
        if contra > best_contradiction[i]:
            best_contradiction[i] = contra

    results: list[ClaimResult] = []
    for i, claim in enumerate(claims):
        status = decide_status(
            best_entail[i],
            best_sim[i],
            best_cov[i],
            best_contradiction[i],
            best_nums[i],
            extra_distinctive=best_extra[i],
            literals_ok=best_lits[i],
        )
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


# Back-compat for tests that imported these names.
SUPPORT_THRESHOLD = MIN_SUPPORT_COVERAGE = 0.72
UNCERTAIN_BAND = 0.38
