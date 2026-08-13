from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from bayesian_rag_evaluator.bn.discretize import LEVELS, load_yaml
from bayesian_rag_evaluator.config_paths import config_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRUCTURE_PATH = config_file("bn_structure.yaml")

THREE_LEVEL = list(LEVELS)
MODEL_TYPE_STATES = ["rag", "fine_tuned"]


def _states_for(node: str) -> list[str]:
    if node == "model_type":
        return MODEL_TYPE_STATES
    return THREE_LEVEL


def _state_names(node: str, parents: list[str]) -> dict[str, list[str]]:
    return {node: _states_for(node), **{p: _states_for(p) for p in parents}}


def _lookup_group(groups: dict[Any, list[float]], key: int) -> list[float]:
    if key in groups:
        return groups[key]
    return groups[str(key)]


def _build_composite_cpt(
    parents: list[str],
    composite_groups: dict[Any, list[float]],
    invert: set[str],
) -> np.ndarray:
    n_cols = 3 ** len(parents)
    values = np.zeros((3, n_cols))
    max_comp = 2 * len(parents)
    for idx in range(n_cols):
        combo: list[int] = []
        rem = idx
        # pgmpy TabularCPD: last evidence variable changes fastest
        for _ in reversed(parents):
            combo.append(rem % 3)
            rem //= 3
        combo.reverse()
        total = 0
        for parent, val in zip(parents, combo, strict=True):
            total += (2 - val) if parent in invert else val
        total = max(0, min(max_comp, total))
        values[:, idx] = _lookup_group(composite_groups, total)
    return values


def build_network(structure_path: Path | None = None) -> DiscreteBayesianNetwork:
    structure_path = structure_path or DEFAULT_STRUCTURE_PATH
    cfg = load_yaml(structure_path)
    model = DiscreteBayesianNetwork(cfg["edges"])
    cpds: list[TabularCPD] = []

    for node, spec in cfg["cpts"].items():
        states = spec.get("states", _states_for(node))
        parents = spec.get("parents", [])

        if "composite_groups" in spec:
            invert = set(spec.get("invert", []))
            values = _build_composite_cpt(parents, spec["composite_groups"], invert)
            cpd = TabularCPD(
                variable=node,
                variable_card=3,
                values=values,
                evidence=parents,
                evidence_card=[3] * len(parents),
                state_names=_state_names(node, parents),
            )
        else:
            raw = spec["values"]
            if parents:
                arr = np.array(raw, dtype=float).T
            else:
                arr = np.array(raw, dtype=float).reshape(-1, 1)

            evidence_card = [len(_states_for(p)) for p in parents]
            cpd = TabularCPD(
                variable=node,
                variable_card=len(states),
                values=arr,
                evidence=parents or None,
                evidence_card=evidence_card or None,
                state_names=_state_names(node, parents),
            )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    if not model.check_model():
        raise ValueError("Bayesian network model validation failed")
    return model


def evidence_dict_from_discretized(disc: Any) -> dict[str, str]:
    return {
        "query_relevance": disc.query_relevance,
        "context_faithfulness": disc.context_faithfulness,
        "entailment_score": disc.entailment_score,
        "retrieval_quality": disc.retrieval_quality,
        "completeness": disc.completeness,
        "contradiction": disc.contradiction,
        "unsupported_claims": disc.unsupported_claims,
        "visual_grounding": getattr(disc, "visual_grounding", "high"),
        "numeric_consistency": getattr(disc, "numeric_consistency", "high"),
        "model_type": disc.model_type,
    }
