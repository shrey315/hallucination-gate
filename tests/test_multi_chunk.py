"""Multi-chunk RAG grounding: neighbors must not veto a supported claim."""

from hallucination_gate import HallucinationGate


def test_multi_chunk_neighbors_do_not_veto_copy_paste():
    """Titan-style failure: warranty claim + shipping/returns neighbors."""
    gate = HallucinationGate(use_heuristic=True)
    answer = "The Titan watch has a 2-year warranty covering manufacturing defects."
    chunks = [
        "The Titan watch has a 2-year warranty covering manufacturing defects.",
        "Shipping takes 3-5 business days within India.",
        "Returns are accepted within 30 days of purchase if unused.",
    ]
    result = gate.check(
        "What is the warranty period?",
        answer,
        context=chunks,
    )
    assert result.released is True, result.reason
    assert result.action in {"pass", "rewrite"}
    assert result.claims
    claim = result.claims[0]
    assert claim["status"] == "supported"
    assert claim["chunk_hits"], "expected per-chunk diagnostics"
    # Unrelated shipping/returns chunks must not be the deciding contradiction.
    assert "contradict" not in result.reason.lower()


def test_aligned_chunk_still_contradicts():
    gate = HallucinationGate(use_heuristic=True)
    result = gate.check(
        "What is the refund policy?",
        "Refunds are never allowed.",
        context=[
            "Customers may request a refund within 30 days of purchase.",
            "Shipping takes 3-5 business days.",
        ],
    )
    assert result.released is False
    assert result.claims
    statuses = {c["status"] for c in result.claims}
    assert "contradicted" in statuses or "uncertain" in statuses
    # Neighbor shipping chunk must not be required; reason should mention claim grounding.
    assert result.reason
    claim = result.claims[0]
    assert claim.get("reason") or claim.get("chunk_hits")


def test_claim_chunk_hits_expose_source_ids():
    gate = HallucinationGate(use_heuristic=True)
    result = gate.check(
        "What is the warranty?",
        "Warranty is 2 years.",
        context=[
            "Warranty is 2 years for manufacturing defects.",
            "Store hours are 9 to 5.",
        ],
    )
    assert result.claims
    hits = result.claims[0].get("chunk_hits") or []
    assert len(hits) >= 1
    assert all("support_score" in h and "status" in h for h in hits)


def test_hallucination_gate_import_has_no_cycle():
    """Public alias must import cleanly without package-root cycle."""
    import importlib

    import bayesian_rag_evaluator

    # Engine package must not pull hallucinate_gate at import time.
    assert not hasattr(bayesian_rag_evaluator, "HallucinationGate")

    mod = importlib.import_module("hallucination_gate")
    assert hasattr(mod, "HallucinationGate")
    gate = mod.HallucinationGate(use_heuristic=True)
    assert gate.check("q", "a", context=["a"]).text
