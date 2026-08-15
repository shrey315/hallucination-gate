from __future__ import annotations

import logging
from pathlib import Path

from bayesian_rag_evaluator.evidence.ocr import (
    ocr_available,
    ocr_image,
    ocr_image_detailed,
    ocr_pdf_pages,
)
from bayesian_rag_evaluator.models.schemas import ImageInput

logger = logging.getLogger("bayesian_rag_evaluator.ingest")

_CLIP_MODEL = None


def extract_pdf_text(path: str | Path, max_pages: int = 40) -> str:
    """Extract text from a PDF. Requires pypdf. Falls back to page OCR when empty."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages[:max_pages]:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    joined = "\n\n".join(pages)
    if joined.strip():
        return joined

    # Scanned / image-only PDF → OCR rasterized pages.
    ocr_pages = ocr_pdf_pages(path, max_pages=min(max_pages, 20))
    if ocr_pages:
        logger.info("PDF text empty; used OCR for %s (%d pages)", path, len(ocr_pages))
        return "\n\n".join(ocr_pages)
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


def enrich_image(image: ImageInput, *, force_ocr: bool = False) -> ImageInput:
    """Fill OCR from disk when a path is provided and OCR text is empty."""
    if image.path and (force_ocr or not image.ocr_text):
        detailed = ocr_image_detailed(image.path)
        if detailed.text:
            image.ocr_text = detailed.text
            if not image.caption and detailed.engine != "none":
                conf = (
                    f"{detailed.confidence:.0%}"
                    if detailed.confidence is not None
                    else "n/a"
                )
                image.caption = image.caption or f"[ocr:{detailed.engine} conf={conf}]"
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


__all__ = [
    "clip_image_text_similarity",
    "enrich_image",
    "extract_pdf_text",
    "load_pdfs",
    "ocr_available",
    "ocr_image",
    "ocr_image_detailed",
    "ocr_pdf_pages",
]
