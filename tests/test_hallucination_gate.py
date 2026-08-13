from bayesian_rag_evaluator.claims.extractor import extract_claims
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.models.schemas import (
    EvaluateRequest,
    ImageInput,
    ModelType,
    TableInput,
)


def test_extract_claims_splits_sentences():
    claims = extract_claims(
        "Refunds are available within 30 days. Shipping is free worldwide."
    )
    assert len(claims) >= 2


def test_gate_abstains_on_contradiction():
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    result = evaluator.evaluate(
        EvaluateRequest(
            query="What is the refund policy?",
            answer="Refunds are never allowed under any circumstances.",
            context_chunks=[
                "Our refund policy allows customers to request a full refund within 30 days of purchase."
            ],
            model_type=ModelType.RAG,
            strict=True,
        )
    )
    assert result.gate.action.value == "abstain"
    assert result.gate.released is False
    assert result.safe_answer != result.gate.original_answer
    assert result.verdict == "fail"


def test_gate_rewrites_mixed_claims():
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    result = evaluator.evaluate(
        EvaluateRequest(
            query="What is the refund policy?",
            answer=(
                "Customers can request a refund within 30 days of purchase. "
                "The CEO lives on Mars and the company was founded in 1492."
            ),
            context_chunks=[
                "Our refund policy allows customers to request a full refund within 30 days of purchase."
            ],
            model_type=ModelType.RAG,
            strict=True,
        )
    )
    assert result.gate.action.value in {"rewrite", "abstain"}
    assert any(c.status.value != "supported" for c in result.claims)


def test_multimodal_image_and_table_grounding():
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    result = evaluator.evaluate(
        EvaluateRequest(
            query="What color is the car in the photo and what is the price?",
            answer="The car is red and the listed price is 25000.",
            images=[ImageInput(caption="A red sedan parked on the street", ocr_text="RED")],
            tables=[
                TableInput(
                    headers=["item", "price"],
                    rows=[["sedan", "25000"]],
                    caption="Vehicle price list",
                )
            ],
            model_type=ModelType.RAG,
        )
    )
    assert "image" in result.modalities_used
    assert "table" in result.modalities_used
    assert result.evidence.visual_grounding > 0.0
    assert result.evidence.numeric_consistency == 1.0


def test_numeric_hallucination_is_flagged():
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    result = evaluator.evaluate(
        EvaluateRequest(
            query="How many days is the refund window?",
            answer="The refund window is 365 days.",
            context_chunks=["Customers may request a refund within 30 days of purchase."],
            model_type=ModelType.RAG,
        )
    )
    assert result.evidence.numeric_consistency < 1.0
    assert result.gate.action.value in {"rewrite", "abstain"}
