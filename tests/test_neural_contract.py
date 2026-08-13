"""Production-path contract tests. Fake neural backends, real gate policy."""

from __future__ import annotations

import pytest

from bayesian_rag_evaluator.claims.verifier import verify_claims
from bayesian_rag_evaluator.evidence.backends import (
    HeuristicEmbeddingBackend,
    HeuristicNLIBackend,
    create_embedding_backend,
    create_nli_backend,
)
from bayesian_rag_evaluator.gate.engine import apply_gate
from bayesian_rag_evaluator.models.schemas import (
    ClaimVerdict,
    EvidenceUnit,
    GateAction,
    MediaType,
    PosteriorScores,
)


class FakeEmbed(HeuristicEmbeddingBackend):
    """Same contract as a neural embedder; scores come from the heuristic backend."""


class FakeNLI(HeuristicNLIBackend):
    pass


def _units(*texts: str) -> list[EvidenceUnit]:
    return [EvidenceUnit(content=t, modality=MediaType.TEXT) for t in texts]


def test_production_verify_path_blocks_evidence_reuse_lie():
    claims = verify_claims(
        "Customers may request a refund within 30 days of purchase under a federal mandate covering digital goods worldwide.",
        _units("Customers may request a refund within 30 days of purchase."),
        FakeEmbed(),
        FakeNLI(),
    )
    assert claims
    assert all(c.status != ClaimVerdict.SUPPORTED for c in claims) or any(
        c.status in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED} for c in claims
    )


def test_production_verify_path_allows_paraphrase():
    claims = verify_claims(
        "Buyers are permitted to ask for a refund in the 30-day period after buying.",
        _units("Customers may request a refund within 30 days of purchase."),
        FakeEmbed(),
        FakeNLI(),
    )
    assert any(c.status == ClaimVerdict.SUPPORTED for c in claims)


def test_bn_is_not_used_as_release_lock():
    """Uncalibrated BN posteriors must not veto a fully supported answer."""
    from bayesian_rag_evaluator.models.schemas import ClaimResult

    claims = [
        ClaimResult(
            text="Staff get 15 days of paid leave each year.",
            status=ClaimVerdict.SUPPORTED,
            support_score=0.9,
            contradiction_score=0.05,
        )
    ]
    bad_bn = PosteriorScores(
        answer_quality=0.1,
        groundedness=0.1,
        hallucination_risk=0.9,
        retrieval_adequacy=0.1,
        release_safety=0.1,
    )
    gate = apply_gate(
        original_answer=claims[0].text,
        claims=claims,
        posteriors=bad_bn,
        query="How much paid leave do employees get?",
        use_bn_veto=False,
    )
    assert gate.released is True
    assert gate.action == GateAction.PASS


def test_backend_factory_heuristic():
    embed = create_embedding_backend(use_heuristic=True)
    nli = create_nli_backend(use_heuristic=True)
    assert embed.similarity("refund policy", "refund policy") == 1.0
    assert nli.entailment_prob("refunds allowed", "refunds allowed") > 0.5


@pytest.mark.neural
def test_neural_backends_load():
    import os

    if os.getenv("RAG_EVAL_NEURAL", "").lower() not in {"1", "true", "yes"}:
        pytest.skip("set RAG_EVAL_NEURAL=1 to load sentence-transformers models")
    embed = create_embedding_backend(use_heuristic=False)
    nli = create_nli_backend(use_heuristic=False)
    assert embed.similarity("the cat sat", "the cat sat") > 0.5
    row = nli.predict_batch([("refunds are allowed within 30 days", "refunds are allowed")])[0]
    assert "entailment" in row
