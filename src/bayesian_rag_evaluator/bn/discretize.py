from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from bayesian_rag_evaluator.config_paths import config_file
from bayesian_rag_evaluator.models.schemas import (
    DiscretizedEvidence,
    EvidenceScores,
    ModelType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_THRESHOLDS_PATH = config_file("thresholds.yaml")

LEVELS = ("low", "medium", "high")
LEVEL_TO_INT = {"low": 0, "medium": 1, "high": 2}
INT_TO_LEVEL = {0: "low", 1: "medium", 2: "high"}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def discretize_score(value: float, edges: list[float]) -> str:
    if value < edges[0]:
        return "low"
    if value < edges[1]:
        return "medium"
    return "high"


def discretize_evidence(
    scores: EvidenceScores,
    model_type: ModelType,
    thresholds: dict[str, Any],
) -> DiscretizedEvidence:
    bins = thresholds["bins"]
    return DiscretizedEvidence(
        query_relevance=discretize_score(scores.query_relevance, bins["query_relevance"]),
        context_faithfulness=discretize_score(
            scores.context_faithfulness, bins["context_faithfulness"]
        ),
        entailment_score=discretize_score(scores.entailment_score, bins["entailment_score"]),
        retrieval_quality=discretize_score(
            scores.retrieval_quality, bins["retrieval_quality"]
        ),
        completeness=discretize_score(scores.completeness, bins["completeness"]),
        contradiction=discretize_score(scores.contradiction, bins["contradiction"]),
        unsupported_claims=discretize_score(
            scores.unsupported_claims, bins["unsupported_claims"]
        ),
        visual_grounding=discretize_score(
            scores.visual_grounding, bins.get("visual_grounding", [0.35, 0.65])
        ),
        numeric_consistency=discretize_score(
            scores.numeric_consistency, bins.get("numeric_consistency", [0.50, 0.85])
        ),
        model_type=model_type.value,
    )


def level_to_probability(level: str) -> float:
    """Map discrete level to representative probability for reporting."""
    return {"low": 0.2, "medium": 0.5, "high": 0.85}[level]
