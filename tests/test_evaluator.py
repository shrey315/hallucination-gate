import pytest

from bayesian_rag_evaluator.bn.discretize import discretize_evidence, discretize_score, load_yaml
from bayesian_rag_evaluator.bn.inference import BayesianInferenceEngine
from bayesian_rag_evaluator.bn.network import build_network
from bayesian_rag_evaluator.evidence.backends import HeuristicEmbeddingBackend, HeuristicNLIBackend
from bayesian_rag_evaluator.evidence.scorers import (
    score_completeness,
    score_query_relevance,
    score_unsupported_claims,
)
from bayesian_rag_evaluator.models.schemas import (
    DiscretizedEvidence,
    EvidenceScores,
    EvaluateRequest,
    ModelType,
)


@pytest.fixture
def thresholds():
    from bayesian_rag_evaluator.bn.discretize import DEFAULT_THRESHOLDS_PATH

    return load_yaml(DEFAULT_THRESHOLDS_PATH)


def test_discretize_score_bins():
    assert discretize_score(0.1, [0.35, 0.65]) == "low"
    assert discretize_score(0.5, [0.35, 0.65]) == "medium"
    assert discretize_score(0.9, [0.35, 0.65]) == "high"


def test_build_network_valid():
    model = build_network()
    assert model.check_model()


def test_heuristic_relevance():
    embedder = HeuristicEmbeddingBackend()
    score = score_query_relevance(
        "refund policy",
        "customers can request a refund within 30 days",
        embedder,
    )
    assert score > 0.0


def test_completeness():
    score = score_completeness(
        "What is the refund policy for international orders?",
        "The refund policy allows returns within 30 days for international orders.",
    )
    assert score > 0.3


def test_unsupported_claims_no_context():
    embedder = HeuristicEmbeddingBackend()
    nli = HeuristicNLIBackend()
    score = score_unsupported_claims(
        "Refunds are never allowed. Shipping is free worldwide.",
        [],
        embedder,
        nli,
    )
    assert score == 1.0


def test_bn_inference_high_quality_evidence(thresholds):
    engine = BayesianInferenceEngine()
    evidence = DiscretizedEvidence(
        query_relevance="high",
        context_faithfulness="high",
        entailment_score="high",
        retrieval_quality="high",
        completeness="high",
        contradiction="low",
        unsupported_claims="low",
        model_type="rag",
    )
    posteriors = engine.infer(evidence)
    assert posteriors.answer_quality > 0.5
    assert posteriors.hallucination_risk < 0.5


def test_bn_inference_poor_evidence(thresholds):
    engine = BayesianInferenceEngine()
    evidence = DiscretizedEvidence(
        query_relevance="low",
        context_faithfulness="low",
        entailment_score="low",
        retrieval_quality="low",
        completeness="low",
        contradiction="high",
        unsupported_claims="high",
        model_type="rag",
    )
    posteriors = engine.infer(evidence)
    assert posteriors.answer_quality < 0.6
    assert posteriors.hallucination_risk > 0.3


def test_end_to_end_evaluator():
    from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator

    evaluator = DiagnosticEvaluator(use_heuristic=True)
    result = evaluator.evaluate(
        EvaluateRequest(
            query="What is the refund policy?",
            answer="Customers can request a refund within 30 days of purchase.",
            context_chunks=[
                "Our refund policy allows customers to request a full refund within 30 days of purchase."
            ],
            model_type=ModelType.RAG,
        )
    )
    assert result.verdict in {"pass", "needs_improvement", "fail"}
    assert result.scores.answer_quality >= 0.0
    assert len(result.suggestions) >= 1


def test_discretize_evidence(thresholds):
    scores = EvidenceScores(
        query_relevance=0.8,
        context_faithfulness=0.75,
        entailment_score=0.7,
        retrieval_quality=0.85,
        completeness=0.6,
        contradiction=0.1,
        unsupported_claims=0.2,
    )
    disc = discretize_evidence(scores, ModelType.RAG, thresholds)
    assert disc.query_relevance == "high"
    assert disc.contradiction == "low"
