from __future__ import annotations

import os

from bayesian_rag_evaluator.claims.reliability import apply_source_reliability
from bayesian_rag_evaluator.claims.verifier import (
    max_contradiction,
    unsupported_ratio,
    verify_claims,
)
from bayesian_rag_evaluator.evidence.align import align_contexts
from bayesian_rag_evaluator.evidence.backends import (
    create_embedding_backend,
    create_nli_backend,
)
from bayesian_rag_evaluator.evidence.ingest import enrich_image, load_pdfs
from bayesian_rag_evaluator.evidence.multimodal import (
    build_evidence_store,
    grounding_texts,
    score_numeric_consistency,
    score_visual_grounding,
)
from bayesian_rag_evaluator.evidence.scorers import (
    score_completeness,
    score_context_faithfulness,
    score_entailment,
    score_query_relevance,
    score_retrieval_quality,
)
from bayesian_rag_evaluator.models.schemas import (
    ClaimResult,
    EvidenceScores,
    EvidenceUnit,
    EvaluateRequest,
    ImageInput,
    ModelType,
    TableInput,
)
from bayesian_rag_evaluator.quality import (
    PolicyProfile,
    heuristic_for_mode,
    resolve_mode,
    resolve_policy,
)


class EvidenceExtractor:
    def __init__(
        self,
        use_heuristic: bool | None = None,
        embed_model: str | None = None,
        nli_model: str | None = None,
        mode: str | None = None,
        policy: PolicyProfile | str | None = None,
        align_contexts_flag: bool = True,
    ) -> None:
        resolved_mode = resolve_mode(mode)
        if use_heuristic is None:
            use_heuristic = heuristic_for_mode(resolved_mode)
            # Legacy env still wins if explicitly set without mode.
            if mode is None and os.getenv("RAG_EVAL_HEURISTIC", "").lower() in {
                "1",
                "true",
                "yes",
            }:
                use_heuristic = True
        self.mode = resolved_mode
        self.policy = resolve_policy(policy)
        self.align_contexts_flag = align_contexts_flag
        self._use_heuristic = use_heuristic
        self._embedder = create_embedding_backend(
            use_heuristic, model_name=embed_model, mode=resolved_mode
        )
        self._nli = create_nli_backend(
            use_heuristic, model_name=nli_model, mode=resolved_mode
        )

    def warm(self) -> None:
        """Load models into memory (cuts cold-start latency on later calls)."""
        _ = self._embedder.similarity_matrix(["warmup query"], ["warmup evidence"])
        _ = self._nli.predict_batch([("warmup evidence", "warmup claim")])

    def store_from_request(self, request: EvaluateRequest) -> list[EvidenceUnit]:
        images = [enrich_image(img.model_copy()) for img in request.images]
        documents = list(request.documents)
        if request.pdf_paths:
            documents.extend(load_pdfs(request.pdf_paths))
        context_chunks = list(request.context_chunks)
        if self.align_contexts_flag and context_chunks:
            context_chunks = align_contexts(
                request.query,
                request.answer,
                context_chunks,
                self._embedder,
                max_chunks=self.policy.max_aligned_chunks,
            )
        units = build_evidence_store(
            context_chunks=context_chunks,
            kb_chunks=request.kb_chunks,
            images=images,
            tables=request.tables,
            documents=documents,
            audio_transcripts=request.audio_transcripts,
        )
        return apply_source_reliability(units, request.source_reliability)

    def extract_claims(
        self,
        answer: str,
        units: list[EvidenceUnit],
        query: str = "",
    ) -> list[ClaimResult]:
        from bayesian_rag_evaluator.judge import refine_uncertain_claims

        claims = verify_claims(
            answer,
            units,
            self._embedder,
            self._nli,
            profile=self.policy,
            query=query,
        )
        return refine_uncertain_claims(claims, units)

    def extract(
        self,
        query: str,
        answer: str,
        context_chunks: list[str],
        kb_chunks: list[str],
        model_type: ModelType,
        images: list[ImageInput] | None = None,
        tables: list[TableInput] | None = None,
        documents: list[str] | None = None,
        audio_transcripts: list[str] | None = None,
        claims: list[ClaimResult] | None = None,
        units: list[EvidenceUnit] | None = None,
    ) -> EvidenceScores:
        images = [enrich_image(img.model_copy()) for img in (images or [])]
        units = units or build_evidence_store(
            context_chunks=context_chunks,
            kb_chunks=kb_chunks,
            images=images,
            tables=tables,
            documents=documents,
            audio_transcripts=audio_transcripts,
        )
        grounding_chunks = grounding_texts(units)

        if claims is None:
            claims = verify_claims(
                answer, units, self._embedder, self._nli, query=query
            )

        retrieval_quality = score_retrieval_quality(
            query, grounding_chunks, self._embedder
        )
        if model_type == ModelType.FINE_TUNED and not context_chunks and kb_chunks:
            retrieval_quality = score_retrieval_quality(
                query, kb_chunks, self._embedder
            )

        contradiction = max(
            score_contradiction_from_claims(claims),
            _legacy_contradiction(answer, grounding_chunks, self._nli),
        )

        return EvidenceScores(
            query_relevance=score_query_relevance(query, answer, self._embedder),
            context_faithfulness=score_context_faithfulness(
                answer, grounding_chunks, self._embedder
            ),
            entailment_score=score_entailment(answer, grounding_chunks, self._nli),
            retrieval_quality=retrieval_quality,
            completeness=score_completeness(query, answer),
            contradiction=contradiction,
            unsupported_claims=unsupported_ratio(claims)
            if claims
            else (1.0 if not grounding_chunks else 0.0),
            visual_grounding=score_visual_grounding(
                query, answer, images or [], self._embedder
            ),
            numeric_consistency=score_numeric_consistency(answer, grounding_chunks),
        )


def score_contradiction_from_claims(claims: list[ClaimResult]) -> float:
    return max_contradiction(claims)


def _legacy_contradiction(answer: str, chunks: list[str], nli) -> float:
    if not chunks:
        return 0.0
    return max(nli.contradiction_prob(chunk, answer) for chunk in chunks)
