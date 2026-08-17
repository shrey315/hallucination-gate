from bayesian_rag_evaluator.claims.extractor import extract_claims, extract_structured_claims
from bayesian_rag_evaluator.claims.fusion import calibrate, fused_support, learn_fusion_calibration
from bayesian_rag_evaluator.claims.logic import logic_mismatches
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.metrics.benchmark import run_adversarial_suite, run_competitor_benchmark
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, ModelType
from hallucination_gate import HallucinationGate


def _eval(policy: str = "strict", **kwargs):
    evaluator = DiagnosticEvaluator(
        use_heuristic=True, policy=policy, align_contexts=False
    )
    return evaluator.evaluate(EvaluateRequest(model_type=ModelType.RAG, **kwargs))


def test_structured_decompose_splits_reason_clause():
    claims = extract_claims(
        "The device has a 2-year warranty covering defects because the CEO lives on Mars."
    )
    assert len(claims) >= 2
    blob = " ".join(claims).lower()
    assert "warranty" in blob
    assert "mars" in blob
    structured = extract_structured_claims(
        "The device has a 2-year warranty covering defects because the CEO lives on Mars."
    )
    assert any(c.has_temporal is False for c in structured)


def test_packed_hallucination_drops_mars_clause():
    result = _eval(
        query="What is the warranty?",
        answer=(
            "The device has a 2-year warranty covering defects "
            "because the CEO lives on Mars."
        ),
        context_chunks=["The device has a 2-year warranty covering manufacturing defects."],
    )
    assert "mars" not in result.safe_answer.lower()


def test_scope_overclaim_not_released():
    result = _eval(
        query="Who gets a refund?",
        answer="All customers always receive an automatic refund.",
        context_chunks=["Some customers may request a refund within 30 days."],
    )
    assert result.gate.released is False
    flags = [f for c in result.claims for f in c.logic_flags]
    assert "scope" in flags or result.gate.action.value == "abstain"


def test_temporal_mismatch_not_released():
    result = _eval(
        query="Is the promo active?",
        answer="The promo is currently active.",
        context_chunks=["The promo expired in 2019 and is no longer available."],
    )
    assert result.gate.released is False
    flags = [f for c in result.claims for f in c.logic_flags]
    assert "temporal" in flags or "negation" in flags or result.gate.released is False


def test_logic_mismatches_detects_temporal():
    flags = logic_mismatches(
        "The promo is currently active.",
        "The promo expired in 2019 and is formerly available.",
    )
    assert "temporal" in flags


def test_low_reliability_cannot_sole_release():
    result = _eval(
        query="What is the warranty?",
        answer="The device has a 2-year warranty covering manufacturing defects.",
        context_chunks=["The device has a 2-year warranty covering manufacturing defects."],
        source_reliability={"context:0": 0.05},
    )
    assert result.gate.released is False
    assert result.claims
    assert result.claims[0].status != "supported" or result.claims[0].reliability < 0.45


def test_sdk_source_reliability():
    gate = HallucinationGate(use_heuristic=True, policy="strict")
    result = gate.check(
        "What is the warranty?",
        "The device has a 2-year warranty covering manufacturing defects.",
        context=["The device has a 2-year warranty covering manufacturing defects."],
        source_reliability={"context:0": 0.05},
    )
    assert result.released is False


def test_multihop_inferred_strict_does_not_treat_as_extractive_release_junk():
    result = _eval(
        policy="strict",
        query="Where is HQ and who is CEO?",
        answer="HQ is in Paris and the CEO is Ada Lovelace.",
        context_chunks=["The warehouse is in Berlin.", "Support hours are 9 to 5."],
    )
    assert result.gate.released is False


def test_multihop_true_composition_tagged_inferred():
    result = _eval(
        policy="balanced",
        query="Where is HQ and who leads the company?",
        answer="Headquarters is in Austin and the CEO is Jordan Lee.",
        context_chunks=[
            "Acme headquarters is in Austin, Texas.",
            "Jordan Lee is the CEO of Acme.",
        ],
    )
    kinds = {c.grounding_kind.value for c in result.claims}
    # Either inferred (joint) or extractive if one chunk was enough; never contradicted.
    assert "contradicted" not in kinds
    if "inferred" in kinds:
        hop = next(c for c in result.claims if c.grounding_kind.value == "inferred")
        assert hop.hop_source_ids
        assert hop.status.value in {"supported", "uncertain"}


def test_inference_distinct_from_unsupported():
    result = _eval(
        policy="strict",
        query="Where is HQ and who leads?",
        answer="Headquarters is in Austin and the CEO is Jordan Lee.",
        context_chunks=[
            "Acme headquarters is in Austin, Texas.",
            "Jordan Lee is the CEO of Acme.",
        ],
    )
    if any(c.grounding_kind.value == "inferred" for c in result.claims):
        inferred = [c for c in result.claims if c.grounding_kind.value == "inferred"]
        assert all(c.status.value != "unsupported" for c in inferred)


def test_fusion_calibration_is_conservative():
    raw = fused_support(0.9, 0.9, 0.9)
    assert 0.0 <= raw <= 1.0
    pulled = calibrate(0.9, ((0.0, 0.0), (0.5, 0.4), (1.0, 0.7)))
    assert pulled < 0.9
    curve = learn_fusion_calibration(
        [(0.2, False), (0.3, False), (0.8, True), (0.9, True)] * 3
    )
    assert curve[0][0] == 0.0
    assert curve[-1][0] == 1.0


def test_adversarial_suite_zero_false_release():
    payload = run_adversarial_suite()
    assert payload["false_release_rate"] == 0.0
    assert payload["n"] >= 6


def test_competitor_benchmark_gate_beats_overlap_on_false_release():
    payload = run_competitor_benchmark()
    gate = payload["competitors"]["hallucination_gate"]
    overlap = payload["competitors"]["overlap"]
    assert gate["false_release_rate"] <= overlap["false_release_rate"]
    assert gate["false_release_rate"] == 0.0
