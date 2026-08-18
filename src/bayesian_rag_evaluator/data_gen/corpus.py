"""In-repo labeled corpus: held-out + adversarial + extra FAQ-style traces.

This is still not a customer DB. It is the packaged buy-off set you can re-run
with heuristic or neural backends. Point ``eval-corpus`` at your own JSONL for
production traffic.
"""

from __future__ import annotations

from bayesian_rag_evaluator.data_gen.adversarial import adversarial_cases
from bayesian_rag_evaluator.data_gen.gold import _abstain_labels, _pass_labels, _rewrite_labels
from bayesian_rag_evaluator.data_gen.heldout import heldout_examples
from bayesian_rag_evaluator.models.schemas import GateAction, GoldExample, ModelType


def inrepo_corpus_examples() -> list[GoldExample]:
    out = list(heldout_examples())
    seen = {ex.query + "\n" + ex.answer for ex in out}
    for case in adversarial_cases():
        key = case.query + "\n" + case.answer
        if key in seen:
            continue
        seen.add(key)
        out.append(_from_adversarial(case))
    for ex in _extra_faq_traces():
        key = ex.query + "\n" + ex.answer
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def _from_adversarial(case) -> GoldExample:
    if case.expected_action:
        action = case.expected_action
    elif case.expected_release:
        action = GateAction.PASS
    else:
        action = GateAction.ABSTAIN
    if action == GateAction.REWRITE:
        labels, latent = _rewrite_labels()
    elif case.expected_release:
        labels, latent = _pass_labels()
    else:
        labels, latent = _abstain_labels()
    return GoldExample(
        query=case.query,
        answer=case.answer,
        context_chunks=list(case.contexts),
        expected_gate=action,
        expected_release=case.expected_release,
        labels=labels,
        latent_labels=latent,
        model_type=ModelType.RAG,
        source_reliability=dict(case.source_reliability or {}),
    )


def _extra_faq_traces() -> list[GoldExample]:
    pass_l, pass_z = _pass_labels()
    abs_l, abs_z = _abstain_labels()

    def row(query, answer, chunks, release, action=None) -> GoldExample:
        act = action or (GateAction.PASS if release else GateAction.ABSTAIN)
        labels, latent = (pass_l, pass_z) if release else (abs_l, abs_z)
        if act == GateAction.REWRITE:
            labels, latent = _rewrite_labels()
        return GoldExample(
            query=query,
            answer=answer,
            context_chunks=chunks if isinstance(chunks, list) else [chunks],
            expected_gate=act,
            expected_release=release,
            labels=labels,
            latent_labels=latent,
            model_type=ModelType.RAG,
        )

    return [
        row(
            "What is the payroll cutoff?",
            "Payroll submissions close at 15:00 on the last business day of the month.",
            "Payroll submissions close at 15:00 on the last business day of the month.",
            True,
        ),
        row(
            "What is the payroll cutoff?",
            "Payroll submissions close at 03:00 on the last business day of the month.",
            "Payroll submissions close at 15:00 on the last business day of the month.",
            False,
        ),
        row(
            "How long is encryption key rotation?",
            "Encryption key rotation is every 90 days.",
            "Encryption key rotation is every 90 days.",
            True,
        ),
        row(
            "How long is encryption key rotation?",
            "Encryption key rotation is every 9 days.",
            "Encryption key rotation is every 90 days.",
            False,
        ),
        row(
            "What is the data retention period?",
            "The data retention period is 18 months.",
            "The data retention period is 18 months.",
            True,
        ),
        row(
            "What is the data retention period?",
            "The data retention period is 180 months.",
            "The data retention period is 18 months.",
            False,
        ),
        row(
            "Who owns vendor onboarding?",
            "Procurement owns vendor onboarding.",
            "Vendor onboarding is owned by the procurement team.",
            True,
        ),
        row(
            "Who owns vendor onboarding?",
            "The legal team on Saturn owns vendor onboarding.",
            "Vendor onboarding is owned by the procurement team.",
            False,
        ),
    ]
