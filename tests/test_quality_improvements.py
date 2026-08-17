from bayesian_rag_evaluator.evidence.align import align_contexts
from bayesian_rag_evaluator.quality import heuristic_for_mode, resolve_mode, resolve_policy
from hallucination_gate import HallucinationGate, RAGEval


def test_modes_ci_vs_quality():
    assert resolve_mode("ci") == "ci"
    assert resolve_mode("quality") == "quality"
    assert resolve_mode("quality_plus") == "quality_plus"
    assert heuristic_for_mode("ci") is True
    assert heuristic_for_mode("quality") is False
    assert heuristic_for_mode("quality_plus") is False


def test_align_contexts_keeps_relevant_first():
    chunks = [
        "Shipping takes 3-5 business days within India.",
        "The device has a 2-year warranty covering manufacturing defects.",
        "Store hours are 9 to 5.",
    ]
    kept = align_contexts(
        "What is the warranty?",
        "The device has a 2-year warranty covering manufacturing defects.",
        chunks,
        embedder=None,
        max_chunks=2,
        min_score=0.05,
    )
    assert kept
    assert "warranty" in kept[0].lower()


def test_balanced_policy_still_blocks_hard_contradiction():
    gate = HallucinationGate(use_heuristic=True, policy="balanced")
    result = gate.check(
        "What is the refund policy?",
        "Refunds are never allowed.",
        context=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.released is False


def test_precision_split_labeled_vs_aligned():
    report = RAGEval(use_heuristic=True, policy="balanced").evaluate(
        [
            {
                "query": "Warranty?",
                "answer": "2-year warranty.",
                "contexts": [
                    "2-year warranty for defects.",
                    "Store opens at 9.",
                    "Cafe next door.",
                ],
                "relevant_contexts": ["2-year warranty for defects."],
            }
        ],
        fail_on_latency=False,
    )
    s = report.samples[0].scores
    assert s.context_precision_labeled is not None
    assert abs(s.context_precision_labeled - 1 / 3) < 1e-6
    assert s.context_precision_aligned is not None
    # Aligned precision should be at least as selective as raw top-k mud.
    assert s.context_precision_aligned <= 1.0


def test_gate_warm_smoke():
    gate = HallucinationGate(use_heuristic=True, policy="balanced", warm=True)
    r = gate.check("q", "hello world", context=["hello world"])
    assert isinstance(r.text, str)
