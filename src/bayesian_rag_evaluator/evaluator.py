from __future__ import annotations

import uuid
from pathlib import Path

from bayesian_rag_evaluator.bn.calibration import load_learned_model
from bayesian_rag_evaluator.bn.discretize import (
    DEFAULT_THRESHOLDS_PATH,
    discretize_evidence,
    load_yaml,
)
from bayesian_rag_evaluator.bn.inference import BayesianInferenceEngine
from bayesian_rag_evaluator.diagnostics.engine import (
    compute_verdict,
    generate_suggestions,
    identify_gaps,
)
from bayesian_rag_evaluator.evidence.extractor import EvidenceExtractor
from bayesian_rag_evaluator.gate.engine import apply_gate
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, EvaluateResponse
from bayesian_rag_evaluator.observability import Timer

DEFAULT_THRESHOLDS = DEFAULT_THRESHOLDS_PATH


class DiagnosticEvaluator:
    def __init__(
        self,
        use_heuristic: bool | None = None,
        structure_path: Path | None = None,
        thresholds_path: Path | None = None,
        learned_model_path: Path | None = None,
        embed_model: str | None = None,
        nli_model: str | None = None,
    ) -> None:
        self._thresholds_path = thresholds_path or DEFAULT_THRESHOLDS
        self._thresholds = load_yaml(self._thresholds_path)
        self._evidence = EvidenceExtractor(
            use_heuristic=use_heuristic,
            embed_model=embed_model,
            nli_model=nli_model,
        )
        model = None
        if learned_model_path and learned_model_path.exists():
            model = load_learned_model(learned_model_path)
        self._inference = BayesianInferenceEngine(
            structure_path=structure_path, model=model
        )

    def evaluate(self, request: EvaluateRequest) -> EvaluateResponse:
        timer = Timer()
        request_id = str(uuid.uuid4())
        units = self._evidence.store_from_request(request)
        claims = self._evidence.extract_claims(request.answer, units)
        scores = self._evidence.extract(
            query=request.query,
            answer=request.answer,
            context_chunks=request.context_chunks,
            kb_chunks=request.kb_chunks,
            model_type=request.model_type,
            images=request.images,
            tables=request.tables,
            documents=request.documents,
            audio_transcripts=request.audio_transcripts,
            claims=claims,
            units=units,
        )
        discretized = discretize_evidence(scores, request.model_type, self._thresholds)
        posteriors = self._inference.infer(discretized)
        gate = apply_gate(
            original_answer=request.answer,
            claims=claims,
            posteriors=posteriors,
            strict=request.strict,
            thresholds_path=self._thresholds_path,
            query=request.query,
        )
        gaps = identify_gaps(discretized, posteriors, self._thresholds_path)
        suggestions = generate_suggestions(
            gaps, discretized, posteriors, request.model_type
        )
        verdict = compute_verdict(posteriors, self._thresholds_path, gate=gate)

        modalities = sorted({u.modality.value for u in units})
        return EvaluateResponse(
            model_type=request.model_type,
            evidence=scores,
            discretized_evidence=discretized,
            scores=posteriors,
            gaps=gaps,
            suggestions=suggestions,
            verdict=verdict,  # type: ignore[arg-type]
            gate=gate,
            safe_answer=gate.safe_answer,
            claims=claims,
            modalities_used=modalities,
            request_id=request_id,
            latency_ms=round(timer.ms(), 2),
        )
