"""Multi-hop verification: two chunks jointly support a claim neither covers alone.

Conservative: each hop must contribute unique claim tokens, union coverage must
clear the support bar, and extra distinctive tokens on the union stay < 2.
Inferred is tagged separately from extractive; strict policy does not release it.
"""

from __future__ import annotations

from bayesian_rag_evaluator.claims.fusion import fused_support
from bayesian_rag_evaluator.claims.logic import logic_mismatches, logic_penalty
from bayesian_rag_evaluator.claims.policy import (
    extra_distinctive_tokens,
    extra_entity_penalty,
    is_chunk_aligned,
    literals_agree,
    numbers_agree,
)
from bayesian_rag_evaluator.evidence.backends import (
    NLIBackend,
    content_tokens,
    token_coverage,
    token_set,
)
from bayesian_rag_evaluator.evidence.synonyms import covers_token
from bayesian_rag_evaluator.models.schemas import ChunkHit, ClaimVerdict, EvidenceUnit
from bayesian_rag_evaluator.quality import PolicyProfile, STRICT


def unique_hop_tokens(claim: str, left: str, right: str) -> tuple[set[str], set[str]]:
    """Content tokens of the claim covered by only one hop."""
    claim_toks = content_tokens(claim)
    left_toks = token_set(left)
    right_toks = token_set(right)
    only_left: set[str] = set()
    only_right: set[str] = set()
    for tok in claim_toks:
        in_l = covers_token(tok, left_toks)
        in_r = covers_token(tok, right_toks)
        if in_l and not in_r:
            only_left.add(tok)
        elif in_r and not in_l:
            only_right.add(tok)
    return only_left, only_right


def try_multihop(
    claim: str,
    hits: list[ChunkHit],
    units: list[EvidenceUnit],
    nli: NLIBackend,
    profile: PolicyProfile | None = None,
    max_pairs: int = 6,
) -> ChunkHit | None:
    """Return a hop hit if two aligned chunks jointly support the claim."""
    p = profile or STRICT
    if not p.enable_multihop or len(hits) < 2:
        return None
    by_id = {u.source_id: u for u in units if u.source_id}
    ranked = sorted(
        hits,
        key=lambda h: (h.support_score, h.coverage, h.similarity),
        reverse=True,
    )[:6]
    tried = 0
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 :]:
            if tried >= max_pairs:
                return None
            if not a.source_id or not b.source_id or a.source_id == b.source_id:
                continue
            ua, ub = by_id.get(a.source_id), by_id.get(b.source_id)
            if ua is None or ub is None:
                continue
            if not (
                is_chunk_aligned(a.coverage, a.similarity)
                or is_chunk_aligned(b.coverage, b.similarity)
            ):
                continue
            left_only, right_only = unique_hop_tokens(claim, ua.content, ub.content)
            if not left_only or not right_only:
                continue
            tried += 1
            joined = f"{ua.content}\n{ub.content}"
            coverage = token_coverage(claim, joined)
            nums_ok = numbers_agree(claim, joined)
            lits_ok = literals_agree(claim, joined)
            extras = extra_distinctive_tokens(claim, joined)
            if nums_ok is False or lits_ok is False or len(extras) >= 2:
                continue
            flags = logic_mismatches(claim, joined)
            row = nli.predict_batch([(joined, claim)])[0]
            contra = min(
                1.0,
                float(row["contradiction"])
                + extra_entity_penalty(claim, joined)
                + logic_penalty(flags),
            )
            entail = float(row["entailment"])
            sim = max(a.similarity, b.similarity)
            support = fused_support(entail, sim, coverage)
            if contra >= p.contradiction_threshold and contra >= support:
                continue
            strong = (
                entail >= p.min_support_entail and coverage >= 0.50
            ) or (
                coverage >= p.min_support_coverage
                and sim >= p.min_lexical_sim
                and entail >= p.min_lexical_entail
            )
            if not strong:
                continue
            rel = min(ua.reliability, ub.reliability)
            return ChunkHit(
                source_id=f"{a.source_id}+{b.source_id}",
                status=ClaimVerdict.SUPPORTED,
                support_score=round(support, 4),
                contradiction_score=round(contra, 4),
                coverage=round(coverage, 4),
                similarity=round(sim, 4),
                entailment=round(entail, 4),
                citation=joined[:280],
                modality=ua.modality,
                reason=(
                    f"multi-hop {a.source_id} + {b.source_id} "
                    f"(unique hops {len(left_only)}/{len(right_only)})"
                ),
                reliability=rel,
                hop=True,
            )
    return None
