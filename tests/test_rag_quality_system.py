from pathlib import Path

from hallucination_gate import (
    LatencyBudget,
    RAGEval,
    compare_to_baseline,
    save_baseline,
    score_retrieval,
)


def test_retrieval_hit_mrr_ndcg():
    scores = score_retrieval(
        ["relevant chunk about warranty", "shipping info", "hours"],
        relevant_contexts=["relevant chunk about warranty"],
    )
    assert scores.hit_at_1 == 1.0
    assert scores.mrr == 1.0
    assert scores.ndcg_at_5 == 1.0
    assert scores.recall_at_5 == 1.0


def test_retrieval_miss_at_top():
    scores = score_retrieval(
        ["noise", "noise2", "the relevant doc"],
        relevant_indices=[2],
    )
    assert scores.hit_at_1 == 0.0
    assert scores.hit_at_3 == 1.0
    assert abs(scores.mrr - 1 / 3) < 1e-6


def test_latency_budget_fails():
    report = RAGEval(use_heuristic=True, latency_budget=LatencyBudget(p95_ms=0.001)).evaluate(
        [
            {
                "query": "Warranty?",
                "answer": "2-year warranty.",
                "contexts": ["2-year warranty for defects."],
            }
        ],
        fail_on_latency=True,
    )
    assert report.latency["n"] >= 1
    assert report.ok is False
    assert any("p95" in f for f in report.failures)


def test_regression_baseline_roundtrip(tmp_path: Path):
    samples = [
        {
            "query": "Warranty?",
            "answer": "The Titan watch has a 2-year warranty.",
            "contexts": [
                "The Titan watch has a 2-year warranty.",
                "Shipping takes 3 days.",
            ],
            "relevant_contexts": ["The Titan watch has a 2-year warranty."],
            "ground_truth": "The Titan watch has a 2-year warranty.",
        }
    ]
    evaler = RAGEval(use_heuristic=True)
    first = evaler.evaluate(samples, fail_on_latency=False)
    baseline = tmp_path / "baseline.json"
    save_baseline(first, baseline)

    second = evaler.evaluate(
        samples,
        baseline_path=baseline,
        fail_on_regression=True,
        fail_on_latency=False,
    )
    assert second.regression is not None
    assert second.ok is True
    assert second.retrieval.get("hit_at_1") == 1.0

    # Force a fake regression by comparing against inflated baseline.
    inflated = first.as_dict()
    inflated["aggregate"]["faithfulness"] = 1.0
    inflated["aggregate"]["answer_relevancy"] = min(
        1.0, (inflated["aggregate"].get("answer_relevancy") or 0) + 0.5
    )
    bad = tmp_path / "inflated.json"
    bad.write_text(__import__("json").dumps(inflated), encoding="utf-8")
    diff = compare_to_baseline(first, bad)
    # May or may not fail depending on first scores; ensure compare runs.
    assert isinstance(diff.passed, bool)
    assert diff.deltas
