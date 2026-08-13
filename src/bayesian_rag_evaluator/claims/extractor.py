from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CONJUNCTION = re.compile(r"\s+(?:and|but|while|whereas)\s+", re.IGNORECASE)
_HEDGE = re.compile(
    r"^(?:i think|i believe|maybe|perhaps|it seems|it appears)\b",
    re.IGNORECASE,
)


def extract_claims(text: str) -> list[str]:
    """Split an answer into atomic factual claims.

    Uses sentence boundaries first, then conjunction splits for long clauses.
    Falls back to the full answer so short replies are never skipped.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = [p.strip() for p in _SENTENCE_SPLIT.split(cleaned) if p.strip()]
    claims: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip(" -•*")
        if not sentence:
            continue
        if _HEDGE.match(sentence) and len(sentence.split()) < 6:
            continue
        if len(sentence) > 80 and _CONJUNCTION.search(sentence):
            parts = [p.strip() for p in _CONJUNCTION.split(sentence) if p.strip()]
            if all(len(p.split()) >= 4 for p in parts):
                claims.extend(_normalize_claim(p) for p in parts)
                continue
        claims.append(_normalize_claim(sentence))

    claims = [c for c in claims if c]
    if not claims:
        return [cleaned]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for claim in claims:
        key = claim.lower()
        if key not in seen:
            seen.add(key)
            unique.append(claim)
    return unique


def _normalize_claim(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    if len(text) < 3:
        return ""
    return text
