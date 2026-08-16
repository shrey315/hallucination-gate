from __future__ import annotations

import numpy as np

from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.claims.policy import (
    ALIGN_COVERAGE,
    ALIGN_SIMILARITY,
    decide_status,
    extra_distinctive_tokens,
    extra_entity_penalty,
    fused_support,
    is_chunk_aligned,
    literals_agree,
    numbers_agree,
    status_reason,
)
from bayesian_rag_evaluator.evidence.backends import (
    EmbeddingBackend,
    NLIBackend,
    token_coverage,
)
from bayesian_rag_evaluator.models.schemas import (
    ChunkHit,
    ClaimResult,
    ClaimVerdict,
    EvidenceUnit,
    MediaType,
)
from bayesian_rag_evaluator.quality import PolicyProfile, STRICT

DEFAULT_TOP_K = 8


def verify_claims(
    answer: str,
    units: list[EvidenceUnit],
    embedder: EmbeddingBackend,
    nli: NLIBackend,
    top_k: int = DEFAULT_TOP_K,
    profile: PolicyProfile | None = None,
) -> list[ClaimResult]:
    """Ground each claim against individual chunks, then soft-OR aggregate.

    A neighbor chunk with unrelated numbers must not veto a claim that another
    chunk fully supports. Contradiction only wins from *aligned* chunks, and
    only when no chunk supports the claim.
    """
    profile = profile or STRICT
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
                reason="No evidence units were provided.",
            )
            for claim in claims
        ]

    unit_texts = [u.content for u in units]
    sim = embedder.similarity_matrix(claims, unit_texts)
    k = min(top_k, len(units))

    pair_index: list[tuple[int, int]] = []
    pairs: list[tuple[str, str]] = []
    for i, _claim in enumerate(claims):
        top = np.argsort(sim[i])[-k:]
        for j in top:
            pair_index.append((i, int(j)))
            pairs.append((unit_texts[int(j)], claims[i]))

    nli_rows = nli.predict_batch(pairs) if pairs else []

    # Per-claim list of chunk-level hits (status decided against that chunk alone).
    hits_by_claim: list[list[ChunkHit]] = [[] for _ in claims]

    for (i, j), row, sim_ij in zip(
        pair_index, nli_rows, [sim[i, j] for i, j in pair_index], strict=True
    ):
        claim = claims[i]
        unit = units[j]
        coverage = token_coverage(claim, unit.content)
        nums_ok = numbers_agree(claim, unit.content)
        lits_ok = literals_agree(claim, unit.content)
        extras = extra_distinctive_tokens(claim, unit.content)
        contra = float(row["contradiction"]) + extra_entity_penalty(claim, unit.content)
        contra = min(1.0, contra)
        entail = float(row["entailment"])
        if nums_ok is False or lits_ok is False:
            contra = max(contra, 0.78)
            entail = min(entail, 0.30)
        support = fused_support(entail, float(sim_ij), coverage)
        status = decide_status(
            entail,
            float(sim_ij),
            coverage,
            contra,
            nums_ok,
            extra_distinctive=len(extras),
            literals_ok=lits_ok,
            profile=profile,
        )
        hits_by_claim[i].append(
            ChunkHit(
                source_id=unit.source_id,
                status=status,
                support_score=round(support, 4),
                contradiction_score=round(contra, 4),
                coverage=round(coverage, 4),
                similarity=round(float(sim_ij), 4),
                entailment=round(entail, 4),
                citation=unit.content[:280],
                modality=unit.modality,
                reason=status_reason(
                    status,
                    numbers_ok=nums_ok,
                    literals_ok=lits_ok,
                    extra_distinctive=len(extras),
                    coverage=coverage,
                    contradiction=contra,
                ),
            )
        )

    results: list[ClaimResult] = []
    for claim, hits in zip(claims, hits_by_claim, strict=True):
        results.append(_aggregate_claim(claim, hits))
    return results


def _aggregate_claim(claim: str, hits: list[ChunkHit]) -> ClaimResult:
    if not hits:
        return ClaimResult(
            text=claim,
            status=ClaimVerdict.UNSUPPORTED,
            support_score=0.0,
            contradiction_score=0.0,
            reason="No candidate evidence chunks.",
            chunk_hits=[],
        )

    # Prefer higher support, then higher coverage (stable citation choice).
    hits_sorted = sorted(
        hits,
        key=lambda h: (h.support_score, h.coverage, h.similarity),
        reverse=True,
    )
    supported = [h for h in hits_sorted if h.status == ClaimVerdict.SUPPORTED]
    if supported:
        best = supported[0]
        return _result_from_hit(
            claim,
            ClaimVerdict.SUPPORTED,
            best,
            hits_sorted,
            reason=f"Supported by {best.source_id or 'chunk'}.",
        )

    # Contradiction only from chunks that actually talk about the claim.
    aligned_contra = [
        h
        for h in hits_sorted
        if h.status == ClaimVerdict.CONTRADICTED
        and is_chunk_aligned(h.coverage, h.similarity)
    ]
    if aligned_contra:
        best = max(
            aligned_contra,
            key=lambda h: (h.contradiction_score, h.coverage, h.similarity),
        )
        detail = best.reason or "aligned chunk disagrees"
        return _result_from_hit(
            claim,
            ClaimVerdict.CONTRADICTED,
            best,
            hits_sorted,
            reason=f"Contradicted by {best.source_id or 'chunk'}: {detail}",
        )

    uncertain = [h for h in hits_sorted if h.status == ClaimVerdict.UNCERTAIN]
    if uncertain:
        best = uncertain[0]
        return _result_from_hit(
            claim,
            ClaimVerdict.UNCERTAIN,
            best,
            hits_sorted,
            reason=f"Partial overlap with {best.source_id or 'chunk'}; not enough to release.",
        )

    best = hits_sorted[0]
    return _result_from_hit(
        claim,
        ClaimVerdict.UNSUPPORTED,
        best,
        hits_sorted,
        reason=(
            f"No aligned support (best chunk {best.source_id or 'n/a'} "
            f"coverage={best.coverage:.2f})."
        ),
    )


def _result_from_hit(
    claim: str,
    status: ClaimVerdict,
    best: ChunkHit,
    all_hits: list[ChunkHit],
    *,
    reason: str,
) -> ClaimResult:
    show_citation = status in {
        ClaimVerdict.SUPPORTED,
        ClaimVerdict.CONTRADICTED,
        ClaimVerdict.UNCERTAIN,
    }
    return ClaimResult(
        text=claim,
        status=status,
        support_score=best.support_score,
        contradiction_score=best.contradiction_score,
        citation=best.citation if show_citation else None,
        source_id=best.source_id if show_citation else None,
        modality=best.modality,
        reason=reason,
        chunk_hits=all_hits,
    )


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
