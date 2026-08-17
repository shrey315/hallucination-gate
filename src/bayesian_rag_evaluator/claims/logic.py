"""Temporal, negation, and scope checks. Conservative: only fire on clear mismatches."""

from __future__ import annotations

import re

from bayesian_rag_evaluator.evidence.backends import content_tokens, token_set
from bayesian_rag_evaluator.evidence.synonyms import covers_token

UNIVERSAL = frozenset({"all", "every", "always", "everyone", "everything", "invariably"})
EXISTENTIAL = frozenset({"some", "sometimes", "may", "might", "optional", "occasionally"})
NEGATIVE = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "without",
        "cannot",
        "denied",
        "forbidden",
        "disallowed",
        "false",
    }
)
TEMPORAL_PRESENT = frozenset(
    {"currently", "now", "today", "presently", "ongoing", "active"}
)
TEMPORAL_PAST = frozenset(
    {
        "expired",
        "formerly",
        "previously",
        "ended",
        "discontinued",
        "retired",
        "obsolete",
        "historical",
    }
)
_UNTIL_YEAR = re.compile(r"\buntil\s+(?:year\s+)?((?:19|20)\d{2})\b", re.I)
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_AS_OF = re.compile(r"\bas of\s+((?:19|20)\d{2})\b", re.I)


def _topical(claim: str, evidence: str) -> bool:
    ev = token_set(evidence)
    claim_toks = content_tokens(claim)
    if not claim_toks:
        return False
    hits = sum(1 for tok in claim_toks if covers_token(tok, ev))
    return hits / len(claim_toks) >= 0.25


def logic_mismatches(claim: str, evidence: str) -> list[str]:
    """Return flag names when claim polarity/scope/time clashes with this chunk."""
    if not _topical(claim, evidence):
        return []
    flags: list[str] = []
    c = token_set(claim)
    e = token_set(evidence)
    c_neg = bool(c & NEGATIVE)
    e_neg = bool(e & NEGATIVE)
    if c_neg != e_neg and (c & NEGATIVE or e & NEGATIVE):
        # "not all" vs "none" is a scope issue handled below; skip weak hedges.
        if not ({"not"} <= c and (c & UNIVERSAL)):
            flags.append("negation")
    c_univ = bool(c & UNIVERSAL)
    e_exist = bool(e & EXISTENTIAL)
    e_univ = bool(e & UNIVERSAL)
    if c_univ and e_exist and not e_univ:
        flags.append("scope")
    if c & TEMPORAL_PRESENT and e & TEMPORAL_PAST:
        flags.append("temporal")
    until = _UNTIL_YEAR.search(evidence)
    as_of = _AS_OF.search(claim) or _YEAR.search(claim)
    if until and as_of:
        try:
            if int(as_of.group(1)) > int(until.group(1)):
                flags.append("temporal")
        except ValueError:
            pass
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.add(flag)
            out.append(flag)
    return out


def logic_penalty(flags: list[str]) -> float:
    if not flags:
        return 0.0
    if "negation" in flags or "temporal" in flags:
        return 0.40
    if "scope" in flags:
        return 0.28
    return 0.15
