from hallucination_gate import RAGEval, evaluate


def test_rag_eval_faithfulness_on_supported_answer():
    report = RAGEval(use_heuristic=True).evaluate(
        [
            {
                "query": "What is the warranty?",
                "answer": "The Titan watch has a 2-year warranty covering manufacturing defects.",
                "contexts": [
                    "The Titan watch has a 2-year warranty covering manufacturing defects.",
                    "Shipping takes 3-5 business days.",
                ],
                "ground_truth": "The Titan watch has a 2-year warranty covering manufacturing defects.",
            }
        ]
    )
    assert report.n == 1
    assert "faithfulness" in report.aggregate
    assert report.aggregate["faithfulness"] >= 0.9
    assert report.samples[0].scores.released is True
    assert report.samples[0].scores.context_recall is not None


def test_rag_eval_flags_ungrounded():
    report = evaluate(
        [
            {
                "query": "What is the refund policy?",
                "answer": "Refunds are never allowed.",
                "contexts": ["Customers may request a refund within 30 days."],
            }
        ],
        use_heuristic=True,
    )
    assert report.samples[0].scores.released is False
    assert report.aggregate.get("faithfulness", 1.0) < 1.0 or report.samples[0].scores.gate_action == "abstain"


def test_context_precision_with_labels():
    report = RAGEval(use_heuristic=True).evaluate(
        [
            {
                "query": "Warranty?",
                "answer": "2-year warranty.",
                "contexts": [
                    "2-year warranty for defects.",
                    "Store opens at 9.",
                ],
                "relevant_contexts": ["2-year warranty for defects."],
            }
        ]
    )
    # 1 of 2 contexts labeled relevant
    assert abs(report.samples[0].scores.context_precision - 0.5) < 1e-6
