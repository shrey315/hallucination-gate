from __future__ import annotations

from pathlib import Path
from typing import Any

from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from bayesian_rag_evaluator.bn.discretize import LEVEL_TO_INT
from bayesian_rag_evaluator.bn.network import build_network, evidence_dict_from_discretized
from bayesian_rag_evaluator.models.schemas import DiscretizedEvidence, PosteriorScores

LATENT_NODES = (
    "groundedness",
    "hallucination_risk",
    "answer_quality",
    "retrieval_adequacy",
    "release_safety",
)


class BayesianInferenceEngine:
    def __init__(
        self,
        structure_path: Path | None = None,
        model: DiscreteBayesianNetwork | None = None,
    ) -> None:
        self._model = model or build_network(structure_path)
        self._inference = VariableElimination(self._model)

    def infer(self, evidence: DiscretizedEvidence) -> PosteriorScores:
        evidence_map = evidence_dict_from_discretized(evidence)
        posteriors: dict[str, float] = {}
        for node in LATENT_NODES:
            factor = self._inference.query(
                variables=[node],
                evidence=evidence_map,
                show_progress=False,
            )
            posteriors[node] = _expected_high_prob(factor)

        return PosteriorScores(
            answer_quality=posteriors["answer_quality"],
            groundedness=posteriors["groundedness"],
            hallucination_risk=posteriors["hallucination_risk"],
            retrieval_adequacy=posteriors["retrieval_adequacy"],
            release_safety=posteriors["release_safety"],
        )


def _expected_high_prob(factor) -> float:
    """Convert factor distribution to scalar: P(high) + 0.5*P(medium)."""
    var = factor.variables[0]
    states = list(factor.state_names[var])
    probs = {s: float(factor.values.flatten()[i]) for i, s in enumerate(states)}
    if "high" in probs:
        return probs.get("high", 0.0) + 0.5 * probs.get("medium", 0.0)
    total = 0.0
    for state, p in probs.items():
        if state in LEVEL_TO_INT:
            total += p * (LEVEL_TO_INT[state] / 2.0)
        else:
            total += p * 0.5
    return min(1.0, max(0.0, total))
