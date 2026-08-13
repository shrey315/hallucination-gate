from __future__ import annotations

from pathlib import Path

from bayesian_rag_evaluator.bn.discretize import load_yaml
from bayesian_rag_evaluator.config_paths import config_file
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
) -> GateResult:
    cfg = load_yaml(thresholds_path or DEFAULT_THRESHOLDS_PATH)
    gate_cfg = cfg.get("gate", {})
    max_hallucination = float(gate_cfg.get("max_hallucination_risk", 0.28))
    min_groundedness = float(gate_cfg.get("min_groundedness", 0.62))
    min_release_safety = float(gate_cfg.get("min_release_safety", 0.55))
    if not strict:
        max_hallucination = min(0.55, max_hallucination + 0.15)
        min_groundedness = max(0.40, min_groundedness - 0.12)
        min_release_safety = max(0.35, min_release_safety - 0.12)

    contradicted = [c for c in claims if c.status == ClaimVerdict.CONTRADICTED]
    supported = [c for c in claims if c.status == ClaimVerdict.SUPPORTED]
    unsupported = [
        c
        for c in claims
        if c.status in {ClaimVerdict.UNSUPPORTED, ClaimVerdict.UNCERTAIN}
    ]

    if contradicted:
        return GateResult(
            action=GateAction.ABSTAIN,
            released=False,
            reason="One or more claims contradict the knowledge base.",
            original_answer=original_answer,
            safe_answer=ABSTAIN_TEXT,
            claims=claims,
            dropped_claims=[c.text for c in claims],
        )

    unsafe_posteriors = (
        posteriors.hallucination_risk > max_hallucination
        or posteriors.groundedness < min_groundedness
        or posteriors.release_safety < min_release_safety
    )

    if not supported and (unsupported or unsafe_posteriors or not claims):
        return GateResult(
            action=GateAction.ABSTAIN,
            released=False,
            reason="No claim is grounded in retrieved or knowledge-base evidence.",
            original_answer=original_answer,
            safe_answer=ABSTAIN_TEXT,
            claims=claims,
            dropped_claims=[c.text for c in claims],
        )

    if unsupported or unsafe_posteriors:
        safe = _compose_supported(supported)
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


def _compose_supported(claims: list[ClaimResult]) -> str:
    lines: list[str] = []
    for claim in claims:
        text = claim.text.rstrip()
        if claim.source_id:
            text = f"{text} [{claim.source_id}]"
        lines.append(text)
    return " ".join(lines)
