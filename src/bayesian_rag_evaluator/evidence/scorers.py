from __future__ import annotations

from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.evidence.backends import (
    EmbeddingBackend,
    NLIBackend,
    token_set,
)


def split_claims(text: str) -> list[str]:
    return extract_claims(text)


def _max_similarity(text: str, chunks: list[str], embedder: EmbeddingBackend) -> float:
    if not chunks:
        return 0.0
    return max(embedder.similarity(text, chunk) for chunk in chunks)


def score_query_relevance(query: str, answer: str, embedder: EmbeddingBackend) -> float:
    return embedder.similarity(query, answer)


def score_context_faithfulness(
    answer: str, context_chunks: list[str], embedder: EmbeddingBackend
) -> float:
    if not context_chunks:
        return 0.0
    return _max_similarity(answer, context_chunks, embedder)


def score_entailment(
    answer: str, context_chunks: list[str], nli: NLIBackend
) -> float:
    if not context_chunks:
        return 0.0
    return max(nli.entailment_prob(chunk, answer) for chunk in context_chunks)


def score_retrieval_quality(
    query: str, context_chunks: list[str], embedder: EmbeddingBackend
) -> float:
    if not context_chunks:
        return 0.0
    return _max_similarity(query, context_chunks, embedder)


def score_completeness(query: str, answer: str) -> float:
    query_tokens = token_set(query)
    answer_tokens = token_set(answer)
    if not query_tokens:
        return 0.0
    content_tokens = {t for t in query_tokens if len(t) > 3}
    if not content_tokens:
        content_tokens = query_tokens
    covered = sum(1 for t in content_tokens if t in answer_tokens)
    base = covered / len(content_tokens)
    if len(answer.split()) < 8:
        base *= 0.85
    return max(0.0, min(1.0, base))


def score_contradiction(
    answer: str, context_chunks: list[str], nli: NLIBackend
) -> float:
    if not context_chunks:
        return 0.0
    return max(nli.contradiction_prob(chunk, answer) for chunk in context_chunks)


def score_unsupported_claims(
    answer: str,
    context_chunks: list[str],
    embedder: EmbeddingBackend,
    nli: NLIBackend,
    support_threshold: float = 0.45,
) -> float:
    claims = split_claims(answer)
    if not claims:
        return 0.0
    if not context_chunks:
        return 1.0

    unsupported = 0
    for claim in claims:
        best_entail = max(nli.entailment_prob(c, claim) for c in context_chunks)
        best_sim = max(embedder.similarity(c, claim) for c in context_chunks)
        supported = best_entail >= support_threshold or best_sim >= support_threshold
        if not supported:
            unsupported += 1
    return unsupported / len(claims)
