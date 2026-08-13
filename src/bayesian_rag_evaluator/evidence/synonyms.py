"""Small English paraphrase groups. Not a thesaurus — just enough to stop
over-refusing obvious rewordings without letting invented entities through.
"""

from __future__ import annotations

SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"customer", "customers", "buyer", "buyers", "client", "clients", "purchaser", "user", "users"}),
    frozenset({"request", "ask", "apply", "seek"}),
    frozenset({"allow", "allows", "allowed", "permit", "permits", "permitted", "available", "can", "may"}),
    frozenset({"purchase", "purchases", "buy", "buying", "bought", "order", "ordered"}),
    frozenset({"day", "days", "period", "window"}),
    frozenset({"refund", "refunds", "return", "returns", "reimburse", "reimbursement"}),
    frozenset({"employee", "employees", "staff", "worker", "workers"}),
    frozenset({"pto", "leave", "vacation", "timeoff"}),
    frozenset({"limit", "limits", "cap", "capped", "quota", "rate"}),
    frozenset({"connect", "connection", "url", "endpoint"}),
    frozenset({"store", "stored", "keep", "kept", "refrigerat"}),
    frozenset({"support", "handled", "handle", "handles", "billing"}),
)


def synonym_set(token: str) -> set[str]:
    for group in SYNONYM_GROUPS:
        if token in group:
            return set(group)
    return {token}


def covers_token(claim_tok: str, evidence_toks: set[str]) -> bool:
    if claim_tok in evidence_toks:
        return True
    return bool(synonym_set(claim_tok) & evidence_toks)
