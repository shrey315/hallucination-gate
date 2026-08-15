from __future__ import annotations

from pathlib import Path

from bayesian_rag_evaluator.bn.discretize import load_yaml
from bayesian_rag_evaluator.config_paths import config_file
from bayesian_rag_evaluator.evidence.backends import jaccard_similarity
from bayesian_rag_evaluator.evidence.scorers import score_completeness
from bayesian_rag_evaluator.models.schemas import (
    ClaimResult,
    ClaimVerdict,
    GateAction,
    GateResult,
    PosteriorScores,
)

DEFAULT_THRESHOLDS_PATH = config_file("thresholds.yaml")

ABSTAIN_TEXT = (
    "I do not have sufficient grounded evidence to answer this reliably. "
    "The available knowledge base does not support a verifiable response, "
    "so this answer is withheld to avoid hallucination."
)


def apply_gate(
    original_answer: str,
    claims: list[ClaimResult],
    posteriors: PosteriorScores,
    strict: bool = True,
    thresholds_path: Path | None = None,
    query: str = "",
    cite_sources: bool = False,
    use_bn_veto: bool = False,
) -> GateResult:
    cfg = load_yaml(thresholds_path or DEFAULT_THRESHOLDS_PATH)
    gate_cfg = cfg.get("gate", {})
    max_hallucination = float(gate_cfg.get("max_hallucination_risk", 0.22))
    min_groundedness = float(gate_cfg.get("min_groundedness", 0.70))
    min_release_safety = float(gate_cfg.get("min_release_safety", 0.65))
    min_rewrite_complete = float(gate_cfg.get("min_rewrite_completeness", 0.28))
    min_rewrite_overlap = float(gate_cfg.get("min_rewrite_overlap", 0.16))
    if not strict:
        max_hallucination = min(0.55, max_hallucination + 0.15)
        min_groundedness = max(0.40, min_groundedness - 0.12)
        min_release_safety = max(0.35, min_release_safety - 0.12)
        min_rewrite_complete = max(0.12, min_rewrite_complete - 0.10)
        min_rewrite_overlap = max(0.08, min_rewrite_overlap - 0.06)

    contradicted = [c for c in claims if c.status == ClaimVerdict.CONTRADICTED]
    supported = [c for c in claims if c.status == ClaimVerdict.SUPPORTED]
    unsupported = [
        c
        for c in claims
        if c.status in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.UNCERTAIN}
    ]

    if contradicted:
        return _abstain(
            original_answer,
            claims,
            _contradiction_reason(contradicted),
        )

    unsafe_posteriors = False
    if use_bn_veto:
        unsafe_posteriors = (
            posteriors.hallucination_risk > max_hallucination
            or posteriors.groundedness < min_groundedness
            or posteriors.release_safety < min_release_safety
        )

    if not supported and (unsupported or unsafe_posteriors or not claims):
        return _abstain(
            original_answer,
            claims,
            "No claim is grounded in retrieved or knowledge-base evidence.",
        )

    if unsupported or unsafe_posteriors:
        safe = compose_supported(supported, cite_sources=cite_sources)
        if not _answers_query(
            query, safe, min_rewrite_complete, min_rewrite_overlap
        ):
            return _abstain(
                original_answer,
                claims,
                "Grounded fragments remain but they do not answer the user query.",
            )
        return GateResult(
            action=GateAction.REWRITE,
            released=True,
            reason="Ungrounded claims were removed; only evidence-backed statements remain.",
            original_answer=original_answer,
            safe_answer=safe,
            claims=claims,
            dropped_claims=[c.text for c in unsupported],
        )

    return GateResult(
        action=GateAction.PASS,
        released=True,
        reason="All claims are grounded in the knowledge base.",
        original_answer=original_answer,
        safe_answer=original_answer,
        claims=claims,
        dropped_claims=[],
    )


def compose_supported(claims: list[ClaimResult], cite_sources: bool = False) -> str:
    """Join supported claims into readable prose. Citations are opt-in."""
    parts: list[str] = []
    for claim in claims:
        text = claim.text.rstrip()
        if cite_sources and claim.source_id:
            text = f"{text} [{claim.source_id}]"
        parts.append(text)
    return " ".join(parts).strip()


def _answers_query(
    query: str,
    text: str,
    min_complete: float,
    min_overlap: float,
) -> bool:
    if not query.strip() or not text.strip():
        return bool(text.strip())
    complete = score_completeness(query, text)
    overlap = jaccard_similarity(query, text)
    return complete >= min_complete or overlap >= min_overlap


def _abstain(
    original_answer: str, claims: list[ClaimResult], reason: str
) -> GateResult:
    return GateResult(
        action=GateAction.ABSTAIN,
        released=False,
        reason=reason,
        original_answer=original_answer,
        safe_answer=ABSTAIN_TEXT,
        claims=claims,
        dropped_claims=[c.text for c in claims],
    )


def _contradiction_reason(claims: list[ClaimResult]) -> str:
    parts: list[str] = []
    for claim in claims:
        snippet = claim.text.strip()
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        bit = f'"{snippet}"'
        if claim.source_id:
            bit += f" ↔ {claim.source_id}"
        if claim.reason:
            bit += f" ({claim.reason})"
        parts.append(bit)
    return "Contradicted claim(s): " + "; ".join(parts)


def _compose_supported(claims: list[ClaimResult]) -> str:
    """Back-compat alias."""
    return compose_supported(claims, cite_sources=False)
