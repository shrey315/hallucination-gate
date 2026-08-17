from __future__ import annotations

import re
from dataclasses import dataclass, field

from bayesian_rag_evaluator.evidence.multimodal import extract_numbers

_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?。！？])\s+|"
    r"\n+\s*[-•*]+\s*|"
    r"\n+"
    r"|(?:(?<=\d\.)\s+)(?=[A-ZА-Я])"
)
_CONJUNCTION = re.compile(
    r"\s+(?:and|but|while|whereas|但是|そして)\s+",
    re.IGNORECASE,
)
_REASON = re.compile(
    r"\s+(?:because|therefore|thus|hence|so that|which means)\s+",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"^(?:i think|i believe|maybe|perhaps|it seems|it appears)\b",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_NEGATION = re.compile(
    r"\b(?:not|never|no|none|cannot|without)\b",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r"\b(?:currently|now|today|expired|formerly|previously|until|as of|since)\b|"
    r"\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_SCOPE = re.compile(
    r"\b(?:all|every|always|some|sometimes|none|never|may|might)\b",
    re.IGNORECASE,
)


@dataclass
class StructuredClaim:
    """Atomic claim plus cheap facets used by logic / numeric checks."""

    text: str
    parent: str | None = None
    numbers: list[str] = field(default_factory=list)
    has_negation: bool = False
    has_temporal: bool = False
    has_scope: bool = False


def extract_claims(text: str) -> list[str]:
    """Split an answer into atomic factual claims.

    Sentence boundaries (including CJK), bullets, reason clauses, then
    conjunction splits for long clauses. Falls back to the full answer so
    short replies are never skipped.
    """
    return [c.text for c in extract_structured_claims(text)]


def extract_structured_claims(text: str) -> list[StructuredClaim]:
    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = [p.strip() for p in _SENTENCE_SPLIT.split(cleaned) if p.strip()]
    raw: list[tuple[str, str | None]] = []
    for sentence in sentences:
        sentence = _BULLET.sub("", sentence).strip(" -•*")
        if not sentence:
            continue
        if _HEDGE.match(sentence) and len(sentence.split()) < 6:
            continue
        pieces = _decompose(sentence)
        if len(pieces) == 1:
            raw.append((pieces[0], None))
        else:
            for piece in pieces:
                raw.append((piece, sentence))

    claims: list[StructuredClaim] = []
    seen: set[str] = set()
    for piece, parent in raw:
        normalized = _normalize_claim(piece)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append(_to_structured(normalized, parent))

    if not claims:
        return [_to_structured(cleaned, None)]
    return claims


def _decompose(sentence: str) -> list[str]:
    parts = [p.strip() for p in sentence.split(";") if p.strip()]
    if len(parts) == 1:
        parts = [sentence]
    out: list[str] = []
    for part in parts:
        if _REASON.search(part):
            split = [p.strip() for p in _REASON.split(part) if p.strip()]
            if len(split) >= 2 and all(len(p.split()) >= 4 for p in split):
                out.extend(split)
                continue
        if len(part) > 80 and _CONJUNCTION.search(part):
            conj = [p.strip() for p in _CONJUNCTION.split(part) if p.strip()]
            if all(len(p.split()) >= 4 for p in conj):
                out.extend(conj)
                continue
        out.append(part)
    return out or [sentence]


def _to_structured(text: str, parent: str | None) -> StructuredClaim:
    return StructuredClaim(
        text=text,
        parent=parent,
        numbers=extract_numbers(text),
        has_negation=bool(_NEGATION.search(text)),
        has_temporal=bool(_TEMPORAL.search(text)),
        has_scope=bool(_SCOPE.search(text)),
    )


def _normalize_claim(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?。！？":
        text += "."
    if len(text) < 3:
        return ""
    return text
