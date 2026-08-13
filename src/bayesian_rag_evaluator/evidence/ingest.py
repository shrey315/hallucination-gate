from __future__ import annotations

import logging
from pathlib import Path

from bayesian_rag_evaluator.models.schemas import ImageInput

logger = logging.getLogger("bayesian_rag_evaluator.ingest")

_CLIP_MODEL = None


def extract_pdf_text(path: str | Path, max_pages: int = 40) -> str:
    """Extract text from a PDF. Requires pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


def ocr_image(path: str | Path) -> str:
    """OCR an image with pytesseract if installed; otherwise empty string."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.debug("OCR skipped: pytesseract or Pillow not installed")
        return ""
    try:
        return pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as exc:  # tesseract binary missing
        logger.warning("OCR failed for %s: %s", path, exc)
        return ""


def clip_image_text_similarity(image_path: str | Path, text: str) -> float | None:
    """CLIP cosine similarity. Returns None if the model cannot be loaded."""
    global _CLIP_MODEL
    if not text.strip() or not Path(image_path).exists():
        return None
    try:
        if _CLIP_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _CLIP_MODEL = SentenceTransformer("clip-ViT-B-32")
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        img_emb = _CLIP_MODEL.encode([image], convert_to_numpy=True, normalize_embeddings=True)
        txt_emb = _CLIP_MODEL.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        score = float((img_emb @ txt_emb.T)[0, 0])
        return max(0.0, min(1.0, (score + 1.0) / 2.0)) if score < 0 else max(0.0, min(1.0, score))
    except Exception as exc:
        logger.debug("CLIP unavailable: %s", exc)
        return None


def enrich_image(image: ImageInput) -> ImageInput:
    """Fill OCR from disk when a path is provided and OCR text is empty."""
    if image.path and not image.ocr_text:
        image.ocr_text = ocr_image(image.path)
    return image


def load_pdfs(paths: list[str]) -> list[str]:
    docs: list[str] = []
    for path in paths:
        try:
            text = extract_pdf_text(path)
            if text:
                docs.append(text)
        except Exception as exc:
            logger.warning("PDF extract failed for %s: %s", path, exc)
    return docs
