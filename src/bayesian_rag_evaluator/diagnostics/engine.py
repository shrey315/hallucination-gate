from __future__ import annotations

from pathlib import Path
from typing import Any

from bayesian_rag_evaluator.bn.discretize import load_yaml
from bayesian_rag_evaluator.config_paths import config_file
from bayesian_rag_evaluator.models.schemas import (
    DiscretizedEvidence,
    GapItem,
    ModelType,
    PosteriorScores,
)

DEFAULT_STRUCTURE_PATH = config_file("bn_structure.yaml")
DEFAULT_THRESHOLDS_PATH = config_file("thresholds.yaml")

EVIDENCE_NODES = (
    "query_relevance",
    "context_faithfulness",
    "entailment_score",
    "retrieval_quality",
    "completeness",
    "contradiction",
    "unsupported_claims",
    "visual_grounding",
    "numeric_consistency",
)

LATENT_NODES = (
    "groundedness",
    "hallucination_risk",
    "answer_quality",
    "retrieval_adequacy",
    "release_safety",
)

# For these nodes, "high" is bad
INVERSE_NODES = frozenset({"contradiction", "unsupported_claims", "hallucination_risk"})


def _severity_for_level(level: str, inverse: bool = False) -> str:
    if inverse:
        if level == "high":
            return "high"
        if level == "medium":
            return "medium"
        return "low"
    if level == "low":
        return "high"
    if level == "medium":
        return "medium"
    return "low"


def _posterior_severity(score: float, thresholds: dict[str, float], inverse: bool) -> str | None:
    if inverse:
        if score >= thresholds["high"]:
            return "high"
        if score >= thresholds["medium"]:
            return "medium"
        return None
    if score <= thresholds["high"]:
        return "high"
    if score <= thresholds["medium"]:
        return "medium"
    return None


def identify_gaps(
    evidence: DiscretizedEvidence,
    posteriors: PosteriorScores,
    thresholds_path: Path | None = None,
) -> list[GapItem]:
    thresholds_path = thresholds_path or DEFAULT_THRESHOLDS_PATH
    cfg = load_yaml(thresholds_path)
    gap_thresholds = cfg["gap"]
    gaps: list[GapItem] = []

    evidence_map = evidence.model_dump()
    for node in EVIDENCE_NODES:
        level = evidence_map[node]
        inverse = node in INVERSE_NODES
        severity = _severity_for_level(level, inverse=inverse)
        if severity in {"high", "medium"}:
            gaps.append(
                GapItem(
                    dimension=node,
                    severity=severity,  # type: ignore[arg-type]
                    driver=f"{node}={level}",
                )
            )

    posterior_map = posteriors.model_dump()
    for node in LATENT_NODES:
        score = posterior_map[node]
        inverse = node in INVERSE_NODES
        severity = _posterior_severity(score, gap_thresholds, inverse=inverse)
        if severity:
            gaps.append(
                GapItem(
                    dimension=node,
                    severity=severity,  # type: ignore[arg-type]
                    driver=f"P({node})={score:.2f}",
                )
            )

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: severity_rank[g.severity])
    return gaps


def generate_suggestions(
    gaps: list[GapItem],
    evidence: DiscretizedEvidence,
    posteriors: PosteriorScores,
    model_type: ModelType,
    structure_path: Path | None = None,
) -> list[str]:
    structure_path = structure_path or DEFAULT_STRUCTURE_PATH
    cfg = load_yaml(structure_path)
    templates: dict[str, Any] = cfg.get("suggestions", {})
    suggestions: list[str] = []
    seen: set[str] = set()

    def add(key: str, level: str) -> None:
        msg = templates.get(key, {}).get(level)
        if msg and msg not in seen:
            suggestions.append(msg)
            seen.add(msg)

    for gap in gaps:
        if gap.dimension in templates:
            level = gap.driver.split("=")[-1]
            if gap.dimension in INVERSE_NODES:
                if level in {"high", "medium"}:
                    add(gap.dimension, level if level in templates[gap.dimension] else "high")
            else:
                if level in {"low", "medium"}:
                    add(gap.dimension, level)

    # Composite rules
    if (
        evidence.context_faithfulness in {"low", "medium"}
        and evidence.retrieval_quality == "high"
    ):
        add("context_faithfulness", "low")

    if model_type == ModelType.FINE_TUNED and posteriors.groundedness < 0.45:
        msg = templates.get("fine_tuned_kb_mismatch", {}).get("low")
        if msg and msg not in seen:
            suggestions.append(msg)
            seen.add(msg)

    if not suggestions:
        suggestions.append(
            "No critical gaps detected — continue monitoring with labeled eval data for calibration."
        )
    return suggestions


def compute_verdict(
    posteriors: PosteriorScores,
    thresholds_path: Path | None = None,
    gate: Any | None = None,
) -> str:
    thresholds_path = thresholds_path or DEFAULT_THRESHOLDS_PATH
    cfg = load_yaml(thresholds_path)
    v = cfg["verdict"]

    if gate is not None and getattr(gate.action, "value", gate.action) == "abstain":
        return "fail"
    if gate is not None and getattr(gate.action, "value", gate.action) == "rewrite":
        return "needs_improvement"

    if (
        posteriors.answer_quality >= v["pass_quality"]
        and posteriors.groundedness >= v["pass_groundedness"]
        and posteriors.hallucination_risk <= v["max_hallucination_risk"]
        and getattr(posteriors, "release_safety", 1.0) >= v.get("min_release_safety", 0.0)
    ):
        return "pass"
    if posteriors.answer_quality < 0.35 or posteriors.hallucination_risk > 0.65:
        return "fail"
    return "needs_improvement"
