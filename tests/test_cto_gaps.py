from bayesian_rag_evaluator.claims.special import extra_code_tokens, math_agree
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, ModelType
from bayesian_rag_evaluator.quality import default_models_for_mode
from hallucination_gate import HallucinationGate


def _eval(**kwargs):
    evaluator = DiagnosticEvaluator(
        use_heuristic=True, policy="strict", align_contexts=False
    )
    return evaluator.evaluate(EvaluateRequest(model_type=ModelType.RAG, **kwargs))


def test_bn_is_diagnostic_not_release_authority():
    result = _eval(
        query="What is the refund policy?",
        answer="Customers may request a refund within 30 days of purchase.",
        context_chunks=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.scores_are_calibrated is False
    assert result.release_authority == "claim_status"
    assert result.bn_role == "diagnostic"
    assert result.gate.release_authority == "claim_status"
    assert result.gate.scores_are_calibrated is False
    assert not any(g.driver.startswith("P(") for g in result.gaps)


def test_gated_answer_honesty_fields():
    gate = HallucinationGate(use_heuristic=True, policy="strict")
    result = gate.check(
        "What is the refund policy?",
        "Customers may request a refund within 30 days of purchase.",
        context=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.release_authority == "claim_status"
    assert result.scores_are_calibrated is False
    assert result.evidence_gap == "none"
    assert result.retrieval_quality is not None


def test_wrong_equation_never_releases():
    result = _eval(
        query="What is 2 plus 2?",
        answer="2 + 2 = 5.",
        context_chunks=["The checksum identity is 2 + 2 = 4."],
    )
    assert result.gate.released is False
    assert math_agree("2 + 2 = 5.", "The checksum identity is 2 + 2 = 4.") is False


def test_math_agree_accepts_matching_equation():
    assert math_agree("2 + 2 = 4.", "checksum 2 + 2 = 4") is True
    assert math_agree("no equation here", "2 + 2 = 4") is None


def test_code_identifier_swap_never_releases():
    result = _eval(
        query="How do I scale checkout?",
        answer="Call checkoutService.scaleTo(30).",
        context_chunks=["Call checkoutService.scaleTo(3) during business hours."],
    )
    assert result.gate.released is False
    extras = extra_code_tokens(
        "Call billingService.explode()",
        "Call checkoutService.scaleTo(3)",
    )
    assert extras


def test_offtopic_retrieval_is_labeled_retrieval_gap():
    result = _eval(
        query="What is the refund policy?",
        answer="Customers may request a refund within 30 days of purchase.",
        context_chunks=["Today's cafeteria menu is tomato soup and grilled cheese."],
    )
    assert result.gate.released is False
    assert result.gate.evidence_gap.value == "retrieval"
    assert "retriev" in result.gate.reason.lower()


def test_empty_evidence_is_retrieval_gap():
    result = _eval(
        query="What is the office Wi-Fi password?",
        answer="The guest network password is orchid-42.",
        context_chunks=[],
    )
    assert result.gate.released is False
    assert result.gate.evidence_gap.value == "retrieval"


def test_contradiction_is_not_called_retrieval():
    result = _eval(
        query="What is the refund policy?",
        answer="Refunds are never allowed.",
        context_chunks=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.gate.released is False
    assert result.gate.evidence_gap.value == "contradiction"


def test_quality_plus_model_defaults():
    embed, nli = default_models_for_mode("quality_plus")
    assert "mpnet" in embed
    assert "deberta-v3-base" in nli
    embed_q, nli_q = default_models_for_mode("quality")
    assert "MiniLM" in embed_q
    assert "deberta-v3-small" in nli_q
