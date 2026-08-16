"""Ranked retrieval metrics: hit@k, recall@k, MRR, nDCG@k."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class RetrievalScores:
    hit_at_1: float | None = None
    hit_at_3: float | None = None
    hit_at_5: float | None = None
    hit_at_10: float | None = None
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    mrr: float | None = None
    ndcg_at_5: float | None = None
    ndcg_at_10: float | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "notes" or v}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def relevant_set(
    contexts: Sequence[str],
    *,
    relevant_contexts: Sequence[str] | None = None,
    relevant_indices: Sequence[int] | None = None,
) -> set[int]:
    """Return indices into ``contexts`` that are labeled relevant."""
    out: set[int] = set()
    if relevant_indices:
        for idx in relevant_indices:
            if 0 <= int(idx) < len(contexts):
                out.add(int(idx))
    if relevant_contexts:
        wanted = {_norm(c) for c in relevant_contexts}
        for i, chunk in enumerate(contexts):
            if _norm(chunk) in wanted:
                out.add(i)
    return out


def hit_at_k(relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if any(i < k for i in relevant) else 0.0


def recall_at_k(relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for i in relevant if i < k)
    return hits / len(relevant)


def mrr_score(relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    first = min(relevant)
    return 1.0 / (first + 1)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    total = 0.0
    for i, gain in enumerate(gains[:k]):
        total += gain / math.log2(i + 2)
    return total


def ndcg_at_k(relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    gains = [1.0 if i in relevant else 0.0 for i in range(k)]
    ideal = sorted(gains, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def score_retrieval(
    contexts: Sequence[str],
    *,
    relevant_contexts: Sequence[str] | None = None,
    relevant_indices: Sequence[int] | None = None,
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalScores:
    """Score one ranked context list. Order of ``contexts`` is retrieval rank."""
    notes: list[str] = []
    rel = relevant_set(
        contexts,
        relevant_contexts=relevant_contexts,
        relevant_indices=relevant_indices,
    )
    if not contexts:
        notes.append("retrieval: no contexts")
        return RetrievalScores(notes=notes)
    if not rel:
        notes.append("retrieval: skipped (pass relevant_contexts or relevant_indices)")
        return RetrievalScores(notes=notes)

    by_k = {k: hit_at_k(rel, k) for k in ks}
    return RetrievalScores(
        hit_at_1=by_k.get(1),
        hit_at_3=by_k.get(3),
        hit_at_5=by_k.get(5),
        hit_at_10=by_k.get(10),
        recall_at_5=recall_at_k(rel, 5),
        recall_at_10=recall_at_k(rel, 10),
        mrr=mrr_score(rel),
        ndcg_at_5=ndcg_at_k(rel, 5),
        ndcg_at_10=ndcg_at_k(rel, 10),
        notes=notes,
    )


def aggregate_retrieval(scores: Sequence[RetrievalScores]) -> dict[str, float]:
    keys = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "ndcg_at_10",
    )
    out: dict[str, float] = {}
    for key in keys:
        vals = [getattr(s, key) for s in scores if getattr(s, key) is not None]
        if vals:
            out[key] = round(sum(vals) / len(vals), 4)
    return out
