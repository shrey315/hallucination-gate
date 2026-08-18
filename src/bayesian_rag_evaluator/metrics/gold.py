from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, GateAction, GoldExample


@dataclass
class GateMetrics:
    n: int
    action_accuracy: float
    precision_release: float
    recall_release: float
    f1_release: float
    false_abstain_rate: float
    over_refusal_rate: float
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
            "over_refusal_rate": round(self.false_abstain_rate, 4),
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
    evaluator = evaluator or DiagnosticEvaluator(
        use_heuristic=True, policy="strict", align_contexts=False
    )
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
                source_reliability=getattr(ex, "source_reliability", None) or {},
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
        over_refusal_rate=false_abstain / should_release if should_release else 0.0,
        false_release_rate=false_release / should_block if should_block else 0.0,
        per_action=dict(pred_counts),
        confusion=confusion,
    )


def load_corpus_examples(path: Path) -> list[GoldExample]:
    """Load labeled {query, answer, contexts, expected_release} JSON/JSONL.

    This is the buy-off path for *your* retrieval DB. Dummy BN labels are
    filled in; only gate action / release are scored.
    """
    from bayesian_rag_evaluator.data_gen.gold import _abstain_labels, _pass_labels, _rewrite_labels

    raw_items = _read_json_or_jsonl(path)
    examples: list[GoldExample] = []
    for item in raw_items:
        release = bool(item.get("expected_release", item.get("release", False)))
        gate_raw = item.get("expected_gate") or item.get("expected_action")
        if gate_raw:
            action = GateAction(str(gate_raw).lower())
        else:
            action = GateAction.PASS if release else GateAction.ABSTAIN
        if action == GateAction.REWRITE:
            labels, latent = _rewrite_labels()
        elif release:
            labels, latent = _pass_labels()
        else:
            labels, latent = _abstain_labels()
        chunks = item.get("context_chunks") or item.get("contexts") or item.get("context") or []
        if isinstance(chunks, str):
            chunks = [chunks]
        examples.append(
            GoldExample(
                query=str(item["query"]),
                answer=str(item["answer"]),
                context_chunks=list(chunks),
                expected_gate=action,
                expected_release=release,
                labels=labels,
                latent_labels=latent,
                source_reliability=dict(item.get("source_reliability") or {}),
            )
        )
    if not examples:
        raise ValueError(f"No labeled examples in {path}")
    return examples


def _read_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix in {".jsonl", ".ndjson"} or (not text.startswith("[") and not text.startswith("{")):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "examples" in data:
        return list(data["examples"])
    if isinstance(data, dict):
        return [data]
    return []
