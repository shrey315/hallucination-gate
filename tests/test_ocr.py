"""OCR plumbing tests (no system Tesseract required)."""

from pathlib import Path
from unittest.mock import patch

from hallucination_gate import Evidence, HallucinationGate


def test_from_image_auto_ocr_uses_engine(tmp_path: Path):
    img = tmp_path / "card.png"
    img.write_bytes(b"not-a-real-image")

    fake = type(
        "R",
        (),
        {
            "text": "Titan warranty 2 years",
            "engine": "tesseract",
            "confidence": 0.91,
            "ok": True,
        },
    )()

    with patch(
        "bayesian_rag_evaluator.evidence.ocr.ocr_image_detailed",
        return_value=fake,
    ):
        ev = Evidence.from_image(path=str(img))

    assert len(ev.images) == 1
    assert ev.images[0].ocr == "Titan warranty 2 years"
    assert "ocr:tesseract" in ev.images[0].caption


def test_from_image_skips_ocr_when_text_provided():
    with patch("bayesian_rag_evaluator.evidence.ocr.ocr_image_detailed") as mocked:
        ev = Evidence.from_image(path="ignored.jpg", ocr="already extracted")
        mocked.assert_not_called()
    assert ev.images[0].ocr == "already extracted"


def test_gate_grounds_on_ocr_text():
    gate = HallucinationGate(use_heuristic=True)
    ev = Evidence.from_image(ocr="The Titan watch has a 2-year warranty.", auto_ocr=False)
    result = gate.check(
        "What is the warranty?",
        "The Titan watch has a 2-year warranty.",
        evidence=ev,
    )
    assert result.released is True


def test_ocr_available_keys():
    from hallucination_gate import ocr_available

    status = ocr_available()
    assert set(status) >= {"pillow", "tesseract", "easyocr"}
