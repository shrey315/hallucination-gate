import json

from bayesian_rag_evaluator.bn.calibration import learn_cpds_from_data, save_learned_model
from bayesian_rag_evaluator.claims.verifier import verify_claims
from bayesian_rag_evaluator.data_gen.gold import generate_gold_examples, gold_to_labeled, write_gold_jsonl
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.evidence.backends import HeuristicEmbeddingBackend, HeuristicNLIBackend
from bayesian_rag_evaluator.evidence.ingest import extract_pdf_text
from bayesian_rag_evaluator.metrics.gold import evaluate_gold_set
from bayesian_rag_evaluator.models.schemas import (
    EvidenceUnit,
    MediaType,
)
from fastapi.testclient import TestClient
from pypdf import PdfWriter


def test_gold_set_has_at_least_1000_examples(tmp_path):
    path = tmp_path / "gold.jsonl"
    count = write_gold_jsonl(path, n=1200, seed=42)
    assert count == 1200
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1200
    row = json.loads(lines[0])
    assert row["expected_gate"] in {"pass", "rewrite", "abstain"}


def test_calibrate_from_gold_examples(tmp_path):
    labeled = gold_to_labeled(generate_gold_examples(n=120, seed=1))
    model = learn_cpds_from_data(labeled)
    assert model.check_model()
    out = tmp_path / "learned.pkl"
    save_learned_model(model, out)
    assert out.exists()


def test_gold_metrics_false_release_is_low():
    examples = generate_gold_examples(n=90, seed=42)
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    metrics = evaluate_gold_set(examples, evaluator=evaluator)
    assert metrics.n == 90
    assert metrics.false_release_rate <= 0.01
    assert metrics.precision_release >= 0.90


def test_batched_claim_verification_matches_single_path():
    embedder = HeuristicEmbeddingBackend()
    nli = HeuristicNLIBackend()
    units = [
        EvidenceUnit(content="Refunds are allowed within 30 days.", modality=MediaType.TEXT),
        EvidenceUnit(content="Shipping takes 5 days.", modality=MediaType.TEXT),
    ]
    claims = verify_claims(
        "Refunds are allowed within 30 days. The CEO lives on Mars.",
        units,
        embedder,
        nli,
        top_k=2,
    )
    assert len(claims) >= 2
    statuses = {c.status.value for c in claims}
    assert "supported" in statuses or "unsupported" in statuses


def test_pdf_text_extraction(tmp_path):
    pdf_path = tmp_path / "policy.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(pdf_path)
    text = extract_pdf_text(pdf_path)
    assert isinstance(text, str)


def test_v1_answer_returns_only_safe_fields(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_HEURISTIC", "1")
    monkeypatch.delenv("RAG_EVAL_API_KEY", raising=False)
    from bayesian_rag_evaluator.api import main as api_main

    api_main._evaluator = None
    client = TestClient(api_main.app)
    payload = {
        "query": "What is the refund policy?",
        "answer": "Refunds are never allowed.",
        "context_chunks": ["Customers may request a refund within 30 days."],
        "model_type": "rag",
    }
    resp = client.post("/v1/answer", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"request_id", "safe_answer", "released", "latency_ms"}
    assert "original_answer" not in data
    assert "Refunds are never allowed" not in data["safe_answer"]


def test_api_key_required_when_configured(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_HEURISTIC", "1")
    monkeypatch.setenv("RAG_EVAL_API_KEY", "secret-key")
    from bayesian_rag_evaluator.api import main as api_main

    api_main._evaluator = None
    client = TestClient(api_main.app)
    payload = {
        "query": "What is Python?",
        "answer": "Python is a programming language.",
        "context_chunks": ["Python is a high-level language."],
    }
    denied = client.post("/v1/answer", json=payload)
    assert denied.status_code == 401
    ok = client.post("/v1/answer", json=payload, headers={"x-api-key": "secret-key"})
    assert ok.status_code == 200


def test_metrics_endpoint(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_HEURISTIC", "1")
    monkeypatch.delenv("RAG_EVAL_API_KEY", raising=False)
    from bayesian_rag_evaluator.api import main as api_main

    client = TestClient(api_main.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "requests_total" in resp.json()
    assert "latency_ms" in resp.json()
