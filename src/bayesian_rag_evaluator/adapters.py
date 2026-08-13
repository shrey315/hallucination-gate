from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    return text or None


def _from_mapping(item: dict[str, Any]) -> str | None:
    for key in (
        "page_content",
        "text",
        "content",
        "chunk",
        "document",
        "body",
        "ocr",
        "caption",
        "transcript",
        "source",
    ):
        if key in item and item[key]:
            return _as_text(item[key])
    return None


def _from_object(item: Any) -> str | None:
    for attr in (
        "page_content",
        "text",
        "content",
        "get_content",
        "chunk",
        "document",
        "ocr_text",
        "caption",
    ):
        if not hasattr(item, attr):
            continue
        value = getattr(item, attr)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        text = _as_text(value)
        if text:
            return text
    return _as_text(item)


def normalize_context(context: Any) -> list[str]:
    """Turn any RAG/fine-tune evidence blob into plain text chunks.

    Accepts strings, lists, LangChain Documents, LlamaIndex nodes,
    dicts, tuples, and objects with .text / .page_content / .content.
    Dataset-agnostic: no schema is required.
    """
    if context is None:
        return []
    if isinstance(context, (str, bytes)):
        text = _as_text(context)
        return [text] if text else []
    if isinstance(context, dict):
        if any(k in context for k in ("documents", "contexts", "chunks", "nodes", "sources", "hits")):
            for key in ("documents", "contexts", "chunks", "nodes", "sources", "hits"):
                if key in context:
                    return normalize_context(context[key])
        text = _from_mapping(context)
        return [text] if text else []
    if isinstance(context, (list, tuple, set)):
        chunks: list[str] = []
        for item in context:
            chunks.extend(normalize_context(item))
        return chunks
    text = _from_object(context)
    return [text] if text else []
