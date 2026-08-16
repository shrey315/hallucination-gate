from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from pgmpy.estimators import BayesianEstimator

from bayesian_rag_evaluator.bn.discretize import load_yaml
from bayesian_rag_evaluator.bn.network import build_network
from bayesian_rag_evaluator.config_paths import config_file
from bayesian_rag_evaluator.models.schemas import LabeledExample

DEFAULT_THRESHOLDS_PATH = config_file("thresholds.yaml")


def load_labeled_examples(path: Path) -> list[LabeledExample]:
    examples: list[LabeledExample] = []
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("examples", [])
        for item in items:
            examples.append(LabeledExample.model_validate(item))
    elif path.suffix in {".jsonl", ".ndjson"}:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                examples.append(LabeledExample.model_validate(json.loads(line)))
    else:
        raise ValueError(f"Unsupported labeled data format: {path.suffix}")
    return examples


def examples_to_dataframe(examples: list[LabeledExample]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for ex in examples:
        row = {
            "query_relevance": ex.labels.query_relevance,
            "context_faithfulness": ex.labels.context_faithfulness,
            "entailment_score": ex.labels.entailment_score,
            "retrieval_quality": ex.labels.retrieval_quality,
            "completeness": ex.labels.completeness,
            "contradiction": ex.labels.contradiction,
            "unsupported_claims": ex.labels.unsupported_claims,
            "visual_grounding": getattr(ex.labels, "visual_grounding", "high") or "high",
            "numeric_consistency": getattr(ex.labels, "numeric_consistency", "high")
            or "high",
            "model_type": ex.labels.model_type,
        }
        if ex.latent_labels:
            row["groundedness"] = _score_to_level(ex.latent_labels.groundedness)
            row["hallucination_risk"] = _score_to_level(ex.latent_labels.hallucination_risk)
            row["answer_quality"] = _score_to_level(ex.latent_labels.answer_quality)
            row["retrieval_adequacy"] = _score_to_level(ex.latent_labels.retrieval_adequacy)
            row["release_safety"] = _score_to_level(
                getattr(ex.latent_labels, "release_safety", 0.5) or 0.5
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _score_to_level(score: float) -> str:
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def learn_cpds_from_data(
    examples: list[LabeledExample],
    structure_path: Path | None = None,
    equivalent_sample_size: int = 5,
) -> Any:
    """Learn CPT parameters from labeled examples (Phase 2).

    Requires latent node labels in examples for full learning; otherwise learns
    evidence priors only from discretized evidence labels.
    """
    df = examples_to_dataframe(examples)
    model = build_network(structure_path)

    if df.empty:
        raise ValueError("Cannot learn CPTs from an empty labeled set")

    estimator = BayesianEstimator(model, df)
    parameters = estimator.get_parameters(
        prior_type="BDeu",
        equivalent_sample_size=max(equivalent_sample_size, 5),
    )
    model.cpds = []
    model.add_cpds(*parameters)
    if not model.check_model():
        raise ValueError("Learned model failed validation")
    return model


def tune_thresholds_from_labels(
    examples: list[LabeledExample],
    thresholds_path: Path | None = None,
) -> dict[str, Any]:
    """Suggest bin edges from labeled continuous scores if present in metadata."""
    thresholds_path = thresholds_path or DEFAULT_THRESHOLDS_PATH
    cfg = load_yaml(thresholds_path)
    # Keep existing bins if examples lack raw scores; return config for manual review.
    scores_by_dim: dict[str, list[float]] = {k: [] for k in cfg["bins"]}

    for ex in examples:
        meta = getattr(ex, "model_extra", {}) or {}
        raw = meta.get("raw_scores") if isinstance(meta, dict) else None
        if raw:
            for dim, val in raw.items():
                if dim in scores_by_dim:
                    scores_by_dim[dim].append(float(val))

    suggested = dict(cfg)
    for dim, values in scores_by_dim.items():
        if len(values) >= 5:
            import numpy as np

            suggested["bins"][dim] = [
                float(np.percentile(values, 33)),
                float(np.percentile(values, 66)),
            ]
    return suggested


def save_learned_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)


def load_learned_model(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)
