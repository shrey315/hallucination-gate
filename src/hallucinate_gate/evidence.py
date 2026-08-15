from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesian_rag_evaluator.models.schemas import ImageInput, TableInput


@dataclass
class ImageEvidence:
    """Image evidence from any vision / OCR / captioning pipeline."""

    path: str | None = None
    caption: str = ""
    ocr: str = ""
    alt: str = ""
    source_id: str | None = None

    def to_schema(self) -> ImageInput:
        return ImageInput(
            path=self.path,
            caption=self.caption,
            ocr_text=self.ocr,
            alt_text=self.alt,
            source_id=self.source_id,
        )


@dataclass
class TableEvidence:
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""
    source_id: str | None = None

    def to_schema(self) -> TableInput:
        return TableInput(
            headers=self.headers,
            rows=self.rows,
            caption=self.caption,
            source_id=self.source_id,
        )


class Evidence:
    """Bag of grounding evidence. Dataset-agnostic; pass whatever you retrieved."""

    def __init__(
        self,
        *,
        context: Any = None,
        kb: Any = None,
        images: list[ImageEvidence | dict[str, Any] | ImageInput] | None = None,
        tables: list[TableEvidence | dict[str, Any] | TableInput] | None = None,
        documents: Any = None,
        pdfs: list[str] | None = None,
        audio: Any = None,
    ) -> None:
        self.context = context
        self.kb = kb
        self.images = images or []
        self.tables = tables or []
        self.documents = documents
        self.pdfs = pdfs or []
        self.audio = audio

    @classmethod
    def rag(cls, chunks: Any, **extra: Any) -> Evidence:
        return cls(context=chunks, **extra)

    @classmethod
    def fine_tuned(cls, kb: Any, **extra: Any) -> Evidence:
        return cls(kb=kb, **extra)

    @classmethod
    def from_image(
        cls,
        path: str | None = None,
        caption: str = "",
        ocr: str = "",
        *,
        auto_ocr: bool = True,
        ocr_engine: str = "auto",
        ocr_lang: str = "eng",
        **extra: Any,
    ) -> Evidence:
        """Build image evidence. When ``path`` is set and ``ocr`` is empty, runs OCR."""
        ocr_text = ocr
        if auto_ocr and path and not ocr_text:
            from bayesian_rag_evaluator.evidence.ocr import ocr_image_detailed

            result = ocr_image_detailed(path, engine=ocr_engine, lang=ocr_lang)  # type: ignore[arg-type]
            ocr_text = result.text
            if not caption and result.ok:
                conf = f"{result.confidence:.0%}" if result.confidence is not None else "n/a"
                caption = f"[ocr:{result.engine} conf={conf}]"
        return cls(
            images=[ImageEvidence(path=path, caption=caption, ocr=ocr_text)],
            **extra,
        )

    @classmethod
    def from_pdf(cls, *paths: str, **extra: Any) -> Evidence:
        return cls(pdfs=list(paths), **extra)

    @classmethod
    def from_documents(cls, documents: Any, **extra: Any) -> Evidence:
        return cls(documents=documents, **extra)

    @classmethod
    def from_ocr(
        cls,
        text: Any = None,
        *,
        path: str | None = None,
        auto_ocr: bool = True,
        ocr_engine: str = "auto",
        ocr_lang: str = "eng",
        **extra: Any,
    ) -> Evidence:
        """Pass OCR text directly, or OCR an image/PDF path into document evidence."""
        if text is not None and path is None:
            return cls(documents=text, **extra)
        if path and auto_ocr:
            from pathlib import Path

            from bayesian_rag_evaluator.evidence.ocr import ocr_image, ocr_pdf_pages

            p = Path(path)
            if p.suffix.lower() == ".pdf":
                pages = ocr_pdf_pages(p, engine=ocr_engine, lang=ocr_lang)  # type: ignore[arg-type]
                return cls(documents=pages or text or "", **extra)
            extracted = ocr_image(p, engine=ocr_engine, lang=ocr_lang)  # type: ignore[arg-type]
            return cls(documents=extracted or text or "", **extra)
        return cls(documents=text or "", **extra)

    @classmethod
    def from_audio(cls, transcripts: Any, **extra: Any) -> Evidence:
        return cls(audio=transcripts, **extra)

    @classmethod
    def from_table(
        cls,
        headers: list[str],
        rows: list[list[str]],
        caption: str = "",
        **extra: Any,
    ) -> Evidence:
        return cls(tables=[TableEvidence(headers=headers, rows=rows, caption=caption)], **extra)

    def image_inputs(self) -> list[ImageInput]:
        out: list[ImageInput] = []
        for item in self.images:
            if isinstance(item, ImageInput):
                out.append(item)
            elif isinstance(item, ImageEvidence):
                out.append(item.to_schema())
            elif isinstance(item, dict):
                out.append(
                    ImageInput(
                        path=item.get("path"),
                        caption=item.get("caption", ""),
                        ocr_text=item.get("ocr") or item.get("ocr_text") or "",
                        alt_text=item.get("alt") or item.get("alt_text") or "",
                        source_id=item.get("source_id"),
                    )
                )
        return out

    def table_inputs(self) -> list[TableInput]:
        out: list[TableInput] = []
        for item in self.tables:
            if isinstance(item, TableInput):
                out.append(item)
            elif isinstance(item, TableEvidence):
                out.append(item.to_schema())
            elif isinstance(item, dict):
                out.append(TableInput.model_validate(item))
        return out
