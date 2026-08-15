"""High-tech OCR for images and scanned PDFs.

Engines (first available wins, or fuse when requested):
  1. Tesseract via pytesseract (+ Pillow preprocess)
  2. EasyOCR (optional extra)

Install::

    pip install \"hallucination-gate[ocr]\"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("bayesian_rag_evaluator.ocr")

EngineName = Literal["auto", "tesseract", "easyocr"]

_EASYOCR_READER: Any = None


@dataclass
class OcrResult:
    """Structured OCR output for grounding and diagnostics."""

    text: str
    engine: str
    confidence: float | None = None
    path: str | None = None
    preprocess: list[str] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def ocr_available() -> dict[str, bool]:
    """Which OCR backends can load in this environment."""
    return {
        "pillow": _has_pillow(),
        "tesseract": _has_tesseract(),
        "easyocr": _has_easyocr(),
    }


def ocr_image(
    path: str | Path,
    *,
    engine: EngineName = "auto",
    lang: str = "eng",
    fuse: bool = False,
) -> str:
    """OCR an image path. Returns empty string if no engine is available."""
    return ocr_image_detailed(path, engine=engine, lang=lang, fuse=fuse).text


def ocr_image_detailed(
    path: str | Path,
    *,
    engine: EngineName = "auto",
    lang: str = "eng",
    fuse: bool = False,
) -> OcrResult:
    path = Path(path)
    if not path.exists():
        logger.warning("OCR skipped: path does not exist (%s)", path)
        return OcrResult(text="", engine="none", path=str(path))

    engines = _resolve_engines(engine)
    if not engines:
        logger.debug("OCR skipped: no engine installed (pip install hallucination-gate[ocr])")
        return OcrResult(text="", engine="none", path=str(path))

    results: list[OcrResult] = []
    for name in engines:
        try:
            if name == "tesseract":
                results.append(_ocr_tesseract(path, lang=lang))
            elif name == "easyocr":
                results.append(_ocr_easyocr(path, lang=lang))
        except Exception as exc:
            logger.warning("OCR engine %s failed for %s: %s", name, path, exc)

    if not results:
        return OcrResult(text="", engine="none", path=str(path))

    if fuse and len(results) > 1:
        return _fuse_results(results, path=str(path))

    best = max(results, key=lambda r: (len(r.text), r.confidence or 0.0))
    return best


def ocr_pdf_pages(
    path: str | Path,
    *,
    max_pages: int = 20,
    dpi: int = 200,
    engine: EngineName = "auto",
    lang: str = "eng",
) -> list[str]:
    """Rasterize PDF pages and OCR them (for scanned / image-only PDFs)."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.debug("PDF OCR skipped: pdf2image not installed (hallucination-gate[ocr])")
        return []

    path = Path(path)
    try:
        images = convert_from_path(str(path), dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as exc:
        logger.warning("PDF rasterize failed for %s: %s", path, exc)
        return []

    pages: list[str] = []
    for idx, image in enumerate(images):
        tmp = path.with_suffix(f".ocr-page-{idx}.png")
        try:
            image.save(tmp)
            text = ocr_image(tmp, engine=engine, lang=lang)
            if text.strip():
                pages.append(text.strip())
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    return pages


def preprocess_for_ocr(image: Any) -> tuple[Any, list[str]]:
    """Upscale + contrast + mild denoise for tougher photos / UI screenshots."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    steps: list[str] = []
    img = image.convert("RGB")
    w, h = img.size
    # Upscale small images so Tesseract has more signal.
    if min(w, h) < 900:
        scale = max(2, int(900 / max(1, min(w, h))))
        img = img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)
        steps.append(f"upscale×{scale}")

    gray = ImageOps.grayscale(img)
    steps.append("grayscale")
    gray = ImageOps.autocontrast(gray)
    steps.append("autocontrast")
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    steps.append("denoise")
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    steps.append("sharpen")
    return gray, steps


def _resolve_engines(engine: EngineName) -> list[str]:
    if engine == "tesseract":
        return ["tesseract"] if _has_tesseract() else []
    if engine == "easyocr":
        return ["easyocr"] if _has_easyocr() else []
    ordered: list[str] = []
    if _has_tesseract():
        ordered.append("tesseract")
    if _has_easyocr():
        ordered.append("easyocr")
    return ordered


def _ocr_tesseract(path: Path, *, lang: str) -> OcrResult:
    import pytesseract
    from PIL import Image

    raw = Image.open(path)
    processed, steps = preprocess_for_ocr(raw)
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        processed, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    words: list[str] = []
    confs: list[float] = []
    blocks: list[dict[str, Any]] = []
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        words.append(word)
        confs.append(conf)
        blocks.append(
            {
                "text": word,
                "confidence": conf / 100.0,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            }
        )
    text = pytesseract.image_to_string(processed, lang=lang, config=config).strip()
    if not text and words:
        text = " ".join(words)
    confidence = (sum(confs) / len(confs) / 100.0) if confs else None
    return OcrResult(
        text=text,
        engine="tesseract",
        confidence=confidence,
        path=str(path),
        preprocess=steps,
        blocks=blocks,
    )


def _ocr_easyocr(path: Path, *, lang: str) -> OcrResult:
    global _EASYOCR_READER
    import easyocr
    from PIL import Image
    import numpy as np

    langs = _easyocr_langs(lang)
    if _EASYOCR_READER is None or getattr(_EASYOCR_READER, "_hg_langs", None) != tuple(langs):
        _EASYOCR_READER = easyocr.Reader(langs, verbose=False)
        _EASYOCR_READER._hg_langs = tuple(langs)

    raw = Image.open(path)
    processed, steps = preprocess_for_ocr(raw)
    arr = np.array(processed)
    rows = _EASYOCR_READER.readtext(arr)
    parts: list[str] = []
    confs: list[float] = []
    blocks: list[dict[str, Any]] = []
    for bbox, text, conf in rows:
        text = (text or "").strip()
        if not text:
            continue
        parts.append(text)
        confs.append(float(conf))
        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        blocks.append(
            {
                "text": text,
                "confidence": float(conf),
                "left": int(min(xs)),
                "top": int(min(ys)),
                "width": int(max(xs) - min(xs)),
                "height": int(max(ys) - min(ys)),
            }
        )
    return OcrResult(
        text="\n".join(parts).strip(),
        engine="easyocr",
        confidence=(sum(confs) / len(confs)) if confs else None,
        path=str(path),
        preprocess=steps,
        blocks=blocks,
    )


def _easyocr_langs(lang: str) -> list[str]:
    mapping = {
        "eng": ["en"],
        "en": ["en"],
        "eng+hin": ["en", "hi"],
        "hi": ["hi"],
        "hin": ["hi"],
    }
    if "+" in lang:
        out: list[str] = []
        for part in lang.split("+"):
            out.extend(mapping.get(part, [part[:2]]))
        return out or ["en"]
    return mapping.get(lang, ["en"])


def _fuse_results(results: list[OcrResult], *, path: str) -> OcrResult:
    """Prefer the longest high-confidence transcript; keep both engines in metadata."""
    ranked = sorted(
        results,
        key=lambda r: ((r.confidence or 0.0) * 0.4 + min(1.0, len(r.text) / 400) * 0.6),
        reverse=True,
    )
    best = ranked[0]
    engines = "+".join(sorted({r.engine for r in results if r.text}))
    return OcrResult(
        text=best.text,
        engine=engines or best.engine,
        confidence=best.confidence,
        path=path,
        preprocess=best.preprocess,
        blocks=best.blocks,
    )


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _has_tesseract() -> bool:
    if not _has_pillow():
        return False
    try:
        import pytesseract
        from PIL import Image  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _has_easyocr() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False
