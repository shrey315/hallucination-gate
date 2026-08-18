from __future__ import annotations

from pathlib import Path

from bayesian_rag_evaluator.claims.special import fluent_unattested_justification
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.metrics.gold import evaluate_gold_set, load_corpus_examples
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, ModelType
from hallucination_gate import HallucinationGate


def _eval(**kwargs):
    evaluator = DiagnosticEvaluator(
        use_heuristic=True, policy="strict", align_contexts=False
    )
    return evaluator.evaluate(EvaluateRequest(model_type=ModelType.RAG, **kwargs))


def test_faithful_wrong_chunk_abstains():
    result = _eval(
        query="What is the refund policy?",
        answer="Today's cafeteria menu is tomato soup and grilled cheese.",
        context_chunks=["Today's cafeteria menu is tomato soup and grilled cheese."],
    )
    assert result.gate.released is False
    assert result.gate.action.value == "abstain"
    assert result.gate.evidence_gap.value == "retrieval"


def test_paraphrase_refund_still_releases():
    result = _eval(
        query="What is the refund policy?",
        answer="Buyers are permitted to ask for a refund in the 30-day period after buying.",
        context_chunks=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.gate.released is True


def test_fluent_federal_pivot_does_not_release():
    evidence = "The Titan watch has a 2-year warranty covering manufacturing defects."
    claim = (
        "The Titan watch has a 2-year warranty covering manufacturing defects "
        "as required by federal law worldwide including ISO 9001."
    )
    assert fluent_unattested_justification(claim, evidence) is True
    result = _eval(
        query="What is the warranty?",
        answer=claim,
        context_chunks=[evidence],
    )
    assert result.gate.released is False


def test_shadow_mode_keeps_original_text():
    lie = (
        "Customers may request a refund within 30 days of purchase "
        "under a federal mandate covering all digital goods worldwide."
    )
    evidence = "Customers may request a refund within 30 days of purchase."
    gate = HallucinationGate(use_heuristic=True, policy="strict", shadow=True)
    result = gate.check("What is the refund policy?", lie, context=[evidence])
    assert result.shadow is True
    assert result.text == lie
    assert result.gated_text is not None
    assert result.gated_text != lie
    assert result.released is False
    assert result.action == "abstain"


def test_enforce_mode_replaces_text():
    lie = (
        "Customers may request a refund within 30 days of purchase "
        "under a federal mandate covering all digital goods worldwide."
    )
    gate = HallucinationGate(use_heuristic=True, policy="strict", shadow=False)
    result = gate.check(
        "What is the refund policy?",
        lie,
        context=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.shadow is False
    assert result.text != lie
    assert result.gated_text == result.text


def test_fluent_unattested_tail_without_keyword():
    evidence = "The Titan watch has a 2-year warranty covering manufacturing defects."
    claim = (
        "The Titan watch has a 2-year warranty covering manufacturing defects, "
        "valid in every country on earth without exception."
    )
    assert fluent_unattested_justification(claim, evidence) is True
    result = _eval(query="What is the warranty?", answer=claim, context_chunks=[evidence])
    assert result.gate.released is False


def test_composed_two_hop_releases_under_strict():
    result = _eval(
        query="Where is HQ and who leads the company?",
        answer="Headquarters is in Austin and the CEO is Jordan Lee.",
        context_chunks=[
            "Acme headquarters is in Austin, Texas.",
            "Jordan Lee is the CEO of Acme.",
        ],
    )
    assert result.gate.released is True
    kinds = {c.grounding_kind.value for c in result.claims}
    assert "composed" in kinds or "extractive" in kinds


def test_composed_three_hop_releases_under_strict():
    result = _eval(
        query="Where is HQ, who is CEO, and when was Acme founded?",
        answer="Headquarters is in Austin, the CEO is Jordan Lee, and Acme was founded in 2011.",
        context_chunks=[
            "Acme headquarters is in Austin, Texas.",
            "Jordan Lee is the CEO of Acme.",
            "Acme was founded in 2011.",
        ],
    )
    assert result.gate.released is True


def test_public_sdk_lives_in_hallucination_gate():
    import hallucination_gate.gate as canonical
    import hallucinate_gate.gate as alias

    assert canonical.HallucinationGate is alias.HallucinationGate
    assert canonical.__name__ == "hallucination_gate.gate"


def test_inrepo_corpus_zero_false_release():
    from bayesian_rag_evaluator.data_gen.corpus import inrepo_corpus_examples

    metrics = evaluate_gold_set(inrepo_corpus_examples())
    assert metrics.n >= 40
    assert metrics.false_release_rate == 0.0


def test_load_corpus_and_score(tmp_path: Path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"query":"What is the refund policy?","answer":"Customers may request a refund within 30 days of purchase.","contexts":["Customers may request a refund within 30 days of purchase."],"expected_release":true}\n'
        '{"query":"What is the refund policy?","answer":"Today\'s cafeteria menu is tomato soup and grilled cheese.","contexts":["Today\'s cafeteria menu is tomato soup and grilled cheese."],"expected_release":false}\n',
        encoding="utf-8",
    )
    examples = load_corpus_examples(path)
    assert len(examples) == 2
    metrics = evaluate_gold_set(examples)
    assert metrics.n == 2
    assert metrics.false_release_rate == 0.0
