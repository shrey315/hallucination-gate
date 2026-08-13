from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, GoldExample


@dataclass
class GateMetrics:
    n: int
    action_accuracy: float
    precision_release: float
    recall_release: float
    f1_release: float
    false_abstain_rate: float
    false_release_rate: float
    per_action: dict[str, int]
    confusion: dict[str, dict[str, int]]

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "action_accuracy": round(self.action_accuracy, 4),
            "precision_release": round(self.precision_release, 4),
            "recall_release": round(self.recall_release, 4),
            "f1_release": round(self.f1_release, 4),
            "false_abstain_rate": round(self.false_abstain_rate, 4),
            "false_release_rate": round(self.false_release_rate, 4),
            "per_action": self.per_action,
            "confusion": self.confusion,
        }


def evaluate_gold_set(
    examples: list[GoldExample],
    evaluator: DiagnosticEvaluator | None = None,
    max_cases: int | None = None,
) -> GateMetrics:
    """Score the hallucination gate against labeled pass/rewrite/abstain cases."""
    evaluator = evaluator or DiagnosticEvaluator(use_heuristic=True)
    subset = examples[:max_cases] if max_cases else examples

    confusion: dict[str, dict[str, int]] = {
        gold: {"pass": 0, "rewrite": 0, "abstain": 0} for gold in ("pass", "rewrite", "abstain")
    }
    tp = fp = tn = fn = 0
    should_release = 0
    should_block = 0
    false_abstain = 0
    false_release = 0
    correct_action = 0

    for ex in subset:
        result = evaluator.evaluate(
            EvaluateRequest(
                query=ex.query,
                answer=ex.answer,
                context_chunks=ex.context_chunks,
                model_type=ex.model_type,
                strict=True,
            )
        )
        pred = result.gate.action.value
        gold = ex.expected_gate.value
        confusion[gold][pred] += 1
        if pred == gold:
            correct_action += 1

        released = result.gate.released
        if ex.expected_release:
            should_release += 1
            if released:
                tp += 1
            else:
                fn += 1
                false_abstain += 1
        else:
            should_block += 1
            if released:
                fp += 1
                false_release += 1
            else:
                tn += 1

    n = len(subset)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pred_counts = Counter(
        action for row in confusion.values() for action, c in row.items() for _ in range(c)
    )

    return GateMetrics(
        n=n,
        action_accuracy=correct_action / n if n else 0.0,
        precision_release=precision,
        recall_release=recall,
        f1_release=f1,
        false_abstain_rate=false_abstain / should_release if should_release else 0.0,
        false_release_rate=false_release / should_block if should_block else 0.0,
        per_action=dict(pred_counts),
        confusion=confusion,
    )
