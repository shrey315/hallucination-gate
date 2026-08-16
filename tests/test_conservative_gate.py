from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, ModelType


def _eval(**kwargs):
    evaluator = DiagnosticEvaluator(
        use_heuristic=True, policy="strict", align_contexts=False
    )
    return evaluator.evaluate(EvaluateRequest(model_type=ModelType.RAG, **kwargs))


def test_contradiction_never_released():
    result = _eval(
        query="What is the refund policy?",
        answer="Refunds are never allowed under any circumstances.",
        context_chunks=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.gate.released is False
    assert result.gate.action.value == "abstain"
    assert "never allowed" not in result.safe_answer.lower()


def test_wrong_number_never_released():
    result = _eval(
        query="How long is the refund window?",
        answer="The refund window is 365 days.",
        context_chunks=["The refund window is 30 days."],
    )
    assert result.gate.released is False
    assert result.evidence.numeric_consistency < 1.0


def test_entity_swap_never_released():
    result = _eval(
        query="Who supports refunds?",
        answer="The marketing team on Mars handles refunds.",
        context_chunks=["Support for refunds is handled by the billing team."],
    )
    assert result.gate.released is False


def test_off_query_rewrite_abstains():
    result = _eval(
        query="Who founded Acme?",
        answer="Refunds are available within 30 days. Acme was founded by Shakespeare.",
        context_chunks=["Refunds are available within 30 days of purchase."],
    )
    assert result.gate.released is False
    assert result.gate.action.value == "abstain"


def test_paraphrase_is_released():
    result = _eval(
        query="What is the refund policy?",
        answer="A customer can request a refund within 30 days of purchase.",
        context_chunks=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.gate.released is True
    assert result.gate.action.value in {"pass", "rewrite"}


def test_mixed_claims_drop_hallucination():
    result = _eval(
        query="What is the refund policy?",
        answer=(
            "Customers can request a refund within 30 days of purchase. "
            "The CEO lives on Mars."
        ),
        context_chunks=["Customers may request a refund within 30 days of purchase."],
    )
    assert result.gate.action.value in {"rewrite", "abstain"}
    assert "mars" not in result.safe_answer.lower()


def test_empty_evidence_abstains():
    result = _eval(
        query="What is the refund policy?",
        answer="Refunds are available within 30 days.",
        context_chunks=[],
    )
    assert result.gate.released is False
