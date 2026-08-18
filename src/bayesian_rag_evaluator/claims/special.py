"""Math and code-shaped checks that a small NLI stack routinely misses.

These sit on the claim lock. They never raise a support score; they can only
force a mismatch (conservative).
"""

from __future__ import annotations

import operator
import re

_EQ = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)
_OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "x": operator.mul,
    "×": operator.mul,
    "/": operator.truediv,
}
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_CALL = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*\(")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]+(?:_[a-z0-9]+)+\b")
_CAMEL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")
_FLUENT_JUSTIFY = re.compile(
    r"\b(?:as required by|pursuant to|under federal|under international|"
    r"which means that|in accordance with|worldwide including|"
    r"as mandated by|it follows that|hence the|as established by|"
    r"according to (?:iso|gdpr|hipaa|sox))\b",
    re.IGNORECASE,
)


def fluent_unattested_justification(claim: str, evidence: str) -> bool:
    """True when a well-supported prefix is followed by an unattested tail.

    This is a continuation check (copy-then-invent), not a closed list of lie
    phrases. Known discourse markers are a fast path; the prefix/tail scan is
    the general solver.
    """
    if _FLUENT_JUSTIFY.search(claim) and _FLUENT_JUSTIFY.search(evidence) is None:
        return True
    return unattested_continuation(claim, evidence)


def unattested_continuation(claim: str, evidence: str) -> bool:
    """True if a covered prefix is followed by two+ unattested content tokens."""
    from bayesian_rag_evaluator.evidence.backends import content_token_seq, token_set
    from bayesian_rag_evaluator.evidence.synonyms import covers_token

    toks = content_token_seq(claim)
    if len(toks) < 6:
        return False
    evid = token_set(evidence)
    covered = [covers_token(tok, evid) for tok in toks]
    for split in range(4, len(toks) - 1):
        prefix_hit = sum(covered[:split]) / split
        if prefix_hit < 0.75:
            continue
        extra = [
            tok
            for tok in toks[split:]
            if len(tok) >= 5 and not covers_token(tok, evid)
        ]
        if len(extra) >= 2:
            return True
    return False


_COMMON = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "http",
        "https",
        "json",
        "true",
        "false",
        "none",
        "null",
        "print",
        "return",
        "class",
        "def",
        "import",
    }
)


def math_agree(claim: str, evidence: str) -> bool | None:
    """None if the claim has no equation; False on a clear arithmetic clash."""
    eqs = list(_EQ.finditer(claim))
    if not eqs:
        return None
    evid_eqs = list(_EQ.finditer(evidence))
    for match in eqs:
        left, op, right, result = (
            float(match.group(1)),
            match.group(2),
            float(match.group(3)),
            float(match.group(4)),
        )
        fn = _OPS.get(op)
        internally_wrong = False
        if fn is not None:
            try:
                internally_wrong = abs(fn(left, right) - result) > 1e-6
            except ZeroDivisionError:
                return False
        evid_same_wrong = any(
            _same_equation(match, other) for other in evid_eqs
        )
        if internally_wrong and not evid_same_wrong:
            return False
        for other in evid_eqs:
            if _same_lhs(match, other) and not _same_equation(match, other):
                return False
    return True


def extra_code_tokens(claim: str, evidence: str) -> set[str]:
    """Identifiers / dotted paths / calls in the claim that evidence lacks."""
    found: set[str] = set()
    found.update(_DOTTED.findall(claim))
    found.update(_SNAKE.findall(claim))
    found.update(_CAMEL.findall(claim))
    found.update(m.group(0).rstrip("(").strip() for m in _CALL.finditer(claim))
    extra: set[str] = set()
    for tok in found:
        if tok.lower() in _COMMON:
            continue
        if tok not in evidence:
            extra.add(tok)
    return extra


def _same_lhs(a: re.Match[str], b: re.Match[str]) -> bool:
    return (
        abs(float(a.group(1)) - float(b.group(1))) < 1e-6
        and a.group(2) == b.group(2)
        and abs(float(a.group(3)) - float(b.group(3))) < 1e-6
    )


def _same_equation(a: re.Match[str], b: re.Match[str]) -> bool:
    return _same_lhs(a, b) and abs(float(a.group(4)) - float(b.group(4))) < 1e-6
