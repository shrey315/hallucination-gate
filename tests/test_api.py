import pytest
from fastapi.testclient import TestClient

from bayesian_rag_evaluator.api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RAG_EVAL_HEURISTIC", "1")
    monkeypatch.delenv("RAG_EVAL_API_KEY", raising=False)
    monkeypatch.delenv("RAG_EVAL_API_KEYS", raising=False)
    from bayesian_rag_evaluator.api import main as api_main

    api_main._evaluator = None
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_evaluate_endpoint(client):
    payload = {
        "query": "What is the refund policy?",
        "answer": "Customers can request a refund within 30 days.",
        "context_chunks": [
            "Our refund policy allows customers to request a full refund within 30 days."
        ],
        "model_type": "rag",
    }
    resp = client.post("/evaluate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data
    assert "gaps" in data
    assert "suggestions" in data


def test_batch_endpoint(client):
    payload = [
        {
            "query": "What is Python?",
            "answer": "Python is a programming language.",
            "context_chunks": ["Python is a high-level programming language."],
            "model_type": "rag",
        }
    ]
    resp = client.post("/evaluate/batch", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
