from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    RAG = "rag"
    FINE_TUNED = "fine_tuned"


class MediaType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DOCUMENT = "document"
    AUDIO = "audio"


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class GroundingKind(str, Enum):
    """How a claim relates to evidence. Distinct from release status.

    extractive — one chunk copies/paraphrases the claim
    composed — AND of extractive facts across 2–3 chunks (releasable)
    inferred — joint NLI support that is not per-conjunct extractive
    unsupported — no extractive or composed support
    contradicted — aligned evidence disagrees
    """

    EXTRACTIVE = "extractive"
    COMPOSED = "composed"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class GateAction(str, Enum):
    PASS = "pass"
    REWRITE = "rewrite"
    ABSTAIN = "abstain"


class EvidenceGap(str, Enum):
    """Why the gate blocked or rewrote — retrieval vs generation vs contradiction.

    none — released; evidence was adequate for the claims that shipped
    retrieval — retrieved/KB text is too weakly aligned with the query
    generation — evidence looks adequate; the answer is not supported by it
    mixed — weak retrieval and unsupported claims
    contradiction — aligned evidence disagrees with the answer
    """

    NONE = "none"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    MIXED = "mixed"
    CONTRADICTION = "contradiction"


class ImageInput(BaseModel):
    """Image evidence. Provide caption/OCR for heuristic mode; optional path for CLIP."""

    caption: str = ""
    ocr_text: str = ""
    alt_text: str = ""
    source_id: str | None = None
    path: str | None = None


class TableInput(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str = ""
    source_id: str | None = None


class EvidenceUnit(BaseModel):
    content: str
    modality: MediaType = MediaType.TEXT
    source_id: str | None = None
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)


class ChunkHit(BaseModel):
    """Per-chunk grounding score for one claim (before soft-OR aggregation)."""

    source_id: str | None = None
    status: ClaimVerdict
    support_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    entailment: float = Field(default=0.0, ge=0.0, le=1.0)
    citation: str | None = None
    modality: MediaType = MediaType.TEXT
    reason: str | None = None
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)
    hop: bool = False


class ClaimResult(BaseModel):
    text: str
    status: ClaimVerdict
    support_score: float = Field(..., ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)
    citation: str | None = None
    source_id: str | None = None
    modality: MediaType = MediaType.TEXT
    reason: str | None = None
    chunk_hits: list[ChunkHit] = Field(default_factory=list)
    grounding_kind: GroundingKind = GroundingKind.UNSUPPORTED
    hop_source_ids: list[str] = Field(default_factory=list)
    logic_flags: list[str] = Field(default_factory=list)
    reliability: float = Field(default=1.0, ge=0.0, le=1.0)


class GateResult(BaseModel):
    action: GateAction
    released: bool
    reason: str
    original_answer: str
    safe_answer: str
    claims: list[ClaimResult] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)
    evidence_gap: EvidenceGap = EvidenceGap.NONE
    retrieval_quality: float | None = None
    release_authority: Literal["claim_status"] = "claim_status"
    scores_are_calibrated: bool = False


class EvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    context_chunks: list[str] = Field(default_factory=list)
    kb_chunks: list[str] = Field(default_factory=list)
    images: list[ImageInput] = Field(default_factory=list)
    tables: list[TableInput] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)
    audio_transcripts: list[str] = Field(default_factory=list)
    pdf_paths: list[str] = Field(default_factory=list)
    model_type: ModelType = ModelType.RAG
    strict: bool = Field(
        default=True,
        description="If true, ungrounded or contradicted claims are rewritten or blocked.",
    )
    source_reliability: dict[str, float] = Field(
        default_factory=dict,
        description="Optional reliability in [0,1] keyed by source_id (e.g. context:0).",
    )


class EvidenceScores(BaseModel):
    query_relevance: float = Field(..., ge=0.0, le=1.0)
    context_faithfulness: float = Field(..., ge=0.0, le=1.0)
    entailment_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_quality: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    contradiction: float = Field(..., ge=0.0, le=1.0)
    unsupported_claims: float = Field(..., ge=0.0, le=1.0)
    visual_grounding: float = Field(default=1.0, ge=0.0, le=1.0)
    numeric_consistency: float = Field(default=1.0, ge=0.0, le=1.0)


class DiscretizedEvidence(BaseModel):
    query_relevance: Literal["low", "medium", "high"]
    context_faithfulness: Literal["low", "medium", "high"]
    entailment_score: Literal["low", "medium", "high"]
    retrieval_quality: Literal["low", "medium", "high"]
    completeness: Literal["low", "medium", "high"]
    contradiction: Literal["low", "medium", "high"]
    unsupported_claims: Literal["low", "medium", "high"]
    visual_grounding: Literal["low", "medium", "high"] = "high"
    numeric_consistency: Literal["low", "medium", "high"] = "high"
    model_type: Literal["rag", "fine_tuned"]


class PosteriorScores(BaseModel):
    """Discrete BN fusion scores: P(high)+0.5·P(medium).

    These are **not** calibrated probabilities of truth, hallucination, or
    release risk. They do not decide ``safe_answer`` unless BN veto is on.
    """

    answer_quality: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    hallucination_risk: float = Field(..., ge=0.0, le=1.0)
    retrieval_adequacy: float = Field(..., ge=0.0, le=1.0)
    release_safety: float = Field(default=0.0, ge=0.0, le=1.0)


class GapItem(BaseModel):
    dimension: str
    severity: Literal["low", "medium", "high"]
    driver: str


class EvaluateResponse(BaseModel):
    model_type: ModelType
    evidence: EvidenceScores
    discretized_evidence: DiscretizedEvidence
    scores: PosteriorScores
    gaps: list[GapItem]
    suggestions: list[str]
    verdict: Literal["pass", "needs_improvement", "fail"]
    gate: GateResult
    safe_answer: str
    claims: list[ClaimResult] = Field(default_factory=list)
    modalities_used: list[str] = Field(default_factory=list)
    request_id: str | None = None
    latency_ms: float | None = None
    scores_are_calibrated: bool = False
    release_authority: Literal["claim_status"] = "claim_status"
    bn_role: Literal["diagnostic"] = "diagnostic"


class SafeAnswerResponse(BaseModel):
    """Public response: the only text a user-facing client should display."""

    request_id: str
    safe_answer: str
    released: bool
    latency_ms: float
    evidence_gap: EvidenceGap = EvidenceGap.NONE


class GoldExample(BaseModel):
    query: str
    answer: str
    context_chunks: list[str] = Field(default_factory=list)
    expected_gate: GateAction
    expected_release: bool
    labels: DiscretizedEvidence
    latent_labels: PosteriorScores | None = None
    model_type: ModelType = ModelType.RAG
    source_reliability: dict[str, float] = Field(default_factory=dict)


class LabeledExample(BaseModel):
    """Labeled eval case for CPT learning (Phase 2)."""

    query: str
    answer: str
    context_chunks: list[str] = Field(default_factory=list)
    kb_chunks: list[str] = Field(default_factory=list)
    model_type: ModelType = ModelType.RAG
    labels: DiscretizedEvidence
    latent_labels: PosteriorScores | None = None
