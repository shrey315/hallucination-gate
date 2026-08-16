"""Dataset-agnostic context alignment: keep chunks that match query/answer."""

from __future__ import annotations

from bayesian_rag_evaluator.evidence.backends import (
    EmbeddingBackend,
    jaccard_similarity,
    token_coverage,
)


def align_contexts(
    query: str,
    answer: str,
    contexts: list[str],
    embedder: EmbeddingBackend | None = None,
    *,
    max_chunks: int | None = 5,
    min_score: float = 0.12,
) -> list[str]:
    """Rerank/filter retrieved chunks by overlap with query and answer.

    Generic: no domain lexicon. Preserves order among ties via original rank.
    """
    if not contexts or max_chunks is None or max_chunks <= 0:
        return list(contexts)
    if len(contexts) <= max_chunks and max_chunks >= len(contexts):
        # Still drop near-zero alignment when possible.
        scored = [_score_chunk(query, answer, c, embedder) for c in contexts]
        if all(s >= min_score for s in scored):
            return list(contexts)

    ranked: list[tuple[float, int, str]] = []
    for i, chunk in enumerate(contexts):
        score = _score_chunk(query, answer, chunk, embedder)
        ranked.append((score, i, chunk))
    ranked.sort(key=lambda t: (t[0], -t[1]), reverse=True)

    kept = [c for score, _, c in ranked if score >= min_score][:max_chunks]
    if not kept:
        # Never empty the evidence bag entirely — fall back to top-scoring raw.
        kept = [c for _, _, c in ranked[:max_chunks]]
    return kept


def _score_chunk(
    query: str,
    answer: str,
    chunk: str,
    embedder: EmbeddingBackend | None,
) -> float:
    lex = 0.55 * max(token_coverage(answer, chunk), token_coverage(query, chunk))
    lex += 0.25 * jaccard_similarity(answer, chunk)
    lex += 0.20 * jaccard_similarity(query, chunk)
    if embedder is None:
        return float(lex)
    try:
        sims = embedder.similarity_matrix([query, answer], [chunk])
        sem = float(max(sims[0, 0], sims[1, 0]))
        return 0.45 * lex + 0.55 * sem
    except Exception:
        return float(lex)
