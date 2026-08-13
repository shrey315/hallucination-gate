from __future__ import annotations

from bayesian_rag_evaluator.data_gen.gold import generate_gold_examples
from bayesian_rag_evaluator.data_gen.heldout import heldout_examples
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.metrics.gold import evaluate_gold_set


def test_heldout_false_release_and_over_refusal():
    """Safety bar: almost no false releases, and we still answer grounded paraphrases."""
    examples = heldout_examples()
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    metrics = evaluate_gold_set(examples, evaluator=evaluator)
    assert metrics.n == len(examples)
    assert metrics.false_release_rate <= 0.01
    assert metrics.false_abstain_rate <= 0.20
    assert metrics.precision_release >= 0.90
    assert metrics.recall_release >= 0.80


def test_heldout_is_not_the_template_gold():
    held_q = {ex.query + ex.answer for ex in heldout_examples()}
    gold_q = {ex.query + ex.answer for ex in generate_gold_examples(n=60, seed=42)}
    overlap = held_q & gold_q
    assert len(overlap) == 0
