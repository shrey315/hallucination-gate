from __future__ import annotations

import re

from bayesian_rag_evaluator.models.schemas import (
    EvidenceUnit,
    ImageInput,
    MediaType,
    TableInput,
)

_NUMBER = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w.])"
)


def flatten_table(table: TableInput) -> str:
    parts: list[str] = []
    if table.caption:
        parts.append(table.caption)
    headers = table.headers
    for row in table.rows:
        if headers and len(headers) == len(row):
            cells = [f"{h}: {v}" for h, v in zip(headers, row, strict=False)]
            parts.append("; ".join(cells))
        else:
            parts.append(" | ".join(str(c) for c in row))
    return " ".join(parts).strip()


def image_to_text(image: ImageInput) -> str:
    return " ".join(
        part for part in (image.caption, image.ocr_text, image.alt_text) if part
    ).strip()


def build_evidence_store(
    context_chunks: list[str],
    kb_chunks: list[str],
    images: list[ImageInput] | None = None,
    tables: list[TableInput] | None = None,
    documents: list[str] | None = None,
    audio_transcripts: list[str] | None = None,
) -> list[EvidenceUnit]:
    """Normalize all modalities into a single grounding store."""
    units: list[EvidenceUnit] = []

    for i, chunk in enumerate(context_chunks):
        if chunk.strip():
            units.append(
                EvidenceUnit(
                    content=chunk.strip(),
                    modality=MediaType.TEXT,
                    source_id=f"context:{i}",
                )
            )
    for i, chunk in enumerate(kb_chunks):
        if chunk.strip():
            units.append(
                EvidenceUnit(
                    content=chunk.strip(),
                    modality=MediaType.TEXT,
                    source_id=f"kb:{i}",
                )
            )
    for i, doc in enumerate(documents or []):
        if doc.strip():
            units.append(
                EvidenceUnit(
                    content=doc.strip(),
                    modality=MediaType.DOCUMENT,
                    source_id=f"document:{i}",
                )
            )
    for i, transcript in enumerate(audio_transcripts or []):
        if transcript.strip():
            units.append(
                EvidenceUnit(
                    content=transcript.strip(),
                    modality=MediaType.AUDIO,
                    source_id=f"audio:{i}",
                )
            )
    for i, table in enumerate(tables or []):
        flat = flatten_table(table)
        if flat:
            units.append(
                EvidenceUnit(
                    content=flat,
                    modality=MediaType.TABLE,
                    source_id=table.source_id or f"table:{i}",
                )
            )
    for i, image in enumerate(images or []):
        text = image_to_text(image)
        if text:
            units.append(
                EvidenceUnit(
                    content=text,
                    modality=MediaType.IMAGE,
                    source_id=image.source_id or f"image:{i}",
                )
            )
    return units


def grounding_texts(units: list[EvidenceUnit]) -> list[str]:
    return [u.content for u in units]


def extract_numbers(text: str) -> list[str]:
    return [m.group(0).replace(",", "") for m in _NUMBER.finditer(text)]


def score_numeric_consistency(answer: str, evidence_texts: list[str]) -> float:
    """Fraction of numeric tokens in the answer that appear in evidence.

    No numbers in the answer → 1.0 (nothing to hallucinate).
    Numbers with no evidence at all → 0.0.
    """
    answer_nums = extract_numbers(answer)
    if not answer_nums:
        return 1.0
    if not evidence_texts:
        return 0.0
    joined = " ".join(evidence_texts).replace(",", "")
    evidence_nums = set(extract_numbers(joined))
    if not evidence_nums:
        return 0.0
    hits = 0
    for num in answer_nums:
        if _number_supported(num, evidence_nums, joined):
            hits += 1
    return hits / len(answer_nums)


def _number_supported(num: str, evidence_nums: set[str], joined: str) -> bool:
    if num in evidence_nums or num in joined:
        return True
    try:
        target = float(num.rstrip("%"))
    except ValueError:
        return False
    for other in evidence_nums:
        try:
            value = float(other.rstrip("%"))
        except ValueError:
            continue
        if abs(target - value) < 1e-6:
            return True
    return False


def score_visual_grounding(
    query: str,
    answer: str,
    images: list[ImageInput],
    embedder,
) -> float:
    """Alignment of query+answer with image evidence. 1.0 if no images."""
    if not images:
        return 1.0
    from bayesian_rag_evaluator.evidence.ingest import clip_image_text_similarity

    scores: list[float] = []
    probe = f"{query} {answer}"
    for img in images:
        text = image_to_text(img)
        text_score = 0.0
        if text:
            text_score = 0.5 * embedder.similarity(query, text) + 0.5 * embedder.similarity(
                answer, text
            )
        clip_score = None
        if img.path:
            clip_score = clip_image_text_similarity(img.path, probe)
        if clip_score is not None and text:
            scores.append(0.55 * clip_score + 0.45 * text_score)
        elif clip_score is not None:
            scores.append(clip_score)
        elif text:
            scores.append(text_score)
        else:
            scores.append(0.0)
    if not scores:
        return 0.0
    return max(0.0, min(1.0, max(scores)))
