from hallucinate_gate import Evidence, HallucinationGate


def test_library_import():
    gate = HallucinationGate(use_heuristic=True)
    result = gate.check(
        query="What is the refund policy?",
        answer="Refunds are never allowed.",
        context=["Customers may request a refund within 30 days."],
    )
    assert result.released is False
    assert result.text != result.original


def test_library_accepts_langchain_like_documents():
    class FakeDocument:
        def __init__(self, page_content: str):
            self.page_content = page_content

    gate = HallucinationGate(use_heuristic=True)
    result = gate.check(
        query="What is Python?",
        answer="Python is a programming language.",
        context=[FakeDocument("Python is a high-level programming language.")],
    )
    assert result.action in {"pass", "rewrite", "abstain"}
    assert isinstance(result.text, str)


def test_library_fine_tuned_kb_mode():
    gate = HallucinationGate(mode="fine_tuned", use_heuristic=True)
    result = gate.check(
        query="Who founded the company?",
        answer="The company was founded by Jane Doe in 2010.",
        kb=["Acme Corp was founded by Jane Doe in 2010 in San Francisco."],
    )
    assert result.action in {"pass", "rewrite", "abstain"}


def test_library_multimodal_evidence_helpers():
    gate = HallucinationGate(use_heuristic=True)
    evidence = Evidence.from_image(caption="A red sedan parked on the street", ocr="RED")
    evidence.tables = []
    result = gate.check(
        query="What color is the car?",
        answer="The car is red.",
        evidence=evidence,
    )
    assert isinstance(result.text, str)


def test_protect_decorator():
    gate = HallucinationGate(use_heuristic=True)

    @gate.protect
    def my_rag(query: str):
        return (
            "Refunds are never allowed.",
            ["Customers may request a refund within 30 days."],
        )

    text = my_rag("What is the refund policy?")
    assert "never allowed" not in text.lower() or "withheld" in text.lower() or "grounded" in text.lower()
