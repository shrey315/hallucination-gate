"""Multi-hop verification: 2–3 chunks jointly support a claim neither covers alone.

Composed = each conjunct is extractively attested on some hop (strict may release).
Inferred = joint NLI support without per-conjunct cover (strict does not release).
"""

from __future__ import annotations

import re
from itertools import combinations

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

_CONJUNCT = re.compile(r"\s+and\s+|,\s+(?:and\s+)?", re.IGNORECASE)


def unique_hop_tokens(claim: str, left: str, right: str) -> tuple[set[str], set[str]]:
    """Content tokens of the claim covered by only one hop."""
    groups = unique_n_hop_tokens(claim, [left, right])
    if groups is None:
        return set(), set()
    return groups[0], groups[1]


def unique_n_hop_tokens(claim: str, texts: list[str]) -> list[set[str]] | None:
    """Each hop must cover at least one claim token the others do not."""
    claim_toks = content_tokens(claim)
    tok_sets = [token_set(t) for t in texts]
    only: list[set[str]] = [set() for _ in texts]
    for tok in claim_toks:
        hits = [i for i, ts in enumerate(tok_sets) if covers_token(tok, ts)]
        if len(hits) == 1:
            only[hits[0]].add(tok)
    if any(not group for group in only):
        return None
    return only


def claim_conjuncts(claim: str) -> list[str]:
    parts = [p.strip(" .") for p in _CONJUNCT.split(claim) if p.strip(" .")]
    return parts if len(parts) >= 2 else []


def is_composed_extractive(claim: str, hop_texts: list[str]) -> bool:
    """True when each conjunct is attested by some hop (AND of extractive facts)."""
    parts = claim_conjuncts(claim)
    if len(parts) < 2 or len(hop_texts) < 2:
        return False
    checked = 0
    for part in parts:
        if len(content_tokens(part)) < 2:
            continue
        checked += 1
        if not any(token_coverage(part, text) >= 0.50 for text in hop_texts):
            return False
    return checked >= 2


def try_multihop(
    claim: str,
    hits: list[ChunkHit],
    units: list[EvidenceUnit],
    nli: NLIBackend,
    profile: PolicyProfile | None = None,
    max_pairs: int = 8,
    query: str = "",
) -> ChunkHit | None:
    """Return a hop hit if 2–3 query-aligned chunks jointly support the claim."""
    p = profile or STRICT
    if not p.enable_multihop or len(hits) < 2:
        return None
    by_id = {u.source_id: u for u in units if u.source_id}
    ranked = sorted(
        hits,
        key=lambda h: (h.support_score, h.coverage, h.similarity),
        reverse=True,
    )[:6]
    sizes = (3, 2) if len(claim_conjuncts(claim)) >= 3 else (2, 3)
    for size in sizes:
        if len(ranked) < size:
            continue
        size_tries = 0
        for combo in combinations(ranked, size):
            if size_tries >= max_pairs:
                break
            ids = [h.source_id for h in combo]
            if len(set(ids)) != size or any(not i for i in ids):
                continue
            hops = [by_id.get(i) for i in ids]
            if any(u is None for u in hops):
                continue
            if not all(
                h.coverage >= 0.15 or h.similarity >= 0.22 or is_chunk_aligned(h.coverage, h.similarity)
                for h in combo
            ):
                continue
            texts = [u.content for u in hops]  # type: ignore[union-attr]
            if query.strip():
                q_union = token_coverage(query, " ".join(texts))
                q_max = max(token_coverage(query, t) for t in texts)
                if q_union < 0.10 and q_max < 0.12:
                    continue
            groups = unique_n_hop_tokens(claim, texts)
            if groups is None:
                continue
            size_tries += 1
            joined = "\n".join(texts)
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
            sim = max(h.similarity for h in combo)
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
            composed = is_composed_extractive(claim, texts)
            if not strong and not composed:
                continue
            if not strong and composed:
                # Extractive conjuncts jointly cover the claim.
                if coverage < 0.50:
                    continue
            rel = min(u.reliability for u in hops)  # type: ignore[union-attr]
            kind = "composed-hop" if composed else "inferred-hop"
            uniq = "/".join(str(len(g)) for g in groups)
            return ChunkHit(
                source_id="+".join(ids),  # type: ignore[arg-type]
                status=ClaimVerdict.SUPPORTED,
                support_score=round(support, 4),
                contradiction_score=round(contra, 4),
                coverage=round(coverage, 4),
                similarity=round(sim, 4),
                entailment=round(entail, 4),
                citation=joined[:280],
                modality=hops[0].modality,  # type: ignore[union-attr]
                reason=f"{kind} {'+'.join(ids)} (unique hops {uniq})",  # type: ignore[arg-type]
                reliability=rel,
                hop=True,
            )
    return None
