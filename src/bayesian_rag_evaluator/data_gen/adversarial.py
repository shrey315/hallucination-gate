"""Adversarial cases that must not false-release, plus multi-hop/inference cases."""

from __future__ import annotations

from dataclasses import dataclass

from bayesian_rag_evaluator.models.schemas import GateAction


@dataclass(frozen=True)
class AdversarialCase:
    name: str
    query: str
    answer: str
    contexts: list[str]
    expected_release: bool
    expected_action: GateAction | None = None
    tag: str = "adversarial"
    source_reliability: dict[str, float] | None = None
    notes: str = ""


def adversarial_cases() -> list[AdversarialCase]:
    return [
        AdversarialCase(
            name="negation_flip",
            query="Are refunds allowed?",
            answer="Refunds are never allowed under any circumstances.",
            contexts=["Customers may request a refund within 30 days of purchase."],
            expected_release=False,
            expected_action=GateAction.ABSTAIN,
            tag="negation",
        ),
        AdversarialCase(
            name="number_swap",
            query="How long is the refund window?",
            answer="The refund window is 365 days.",
            contexts=["The refund window is 30 days."],
            expected_release=False,
            tag="numeric",
        ),
        AdversarialCase(
            name="entity_swap",
            query="Who handles refunds?",
            answer="The marketing team on Mars handles refunds.",
            contexts=["Support for refunds is handled by the billing team."],
            expected_release=False,
            tag="entity",
        ),
        AdversarialCase(
            name="scope_overclaim",
            query="Who gets a refund?",
            answer="All customers always receive an automatic refund.",
            contexts=["Some customers may request a refund within 30 days."],
            expected_release=False,
            tag="scope",
        ),
        AdversarialCase(
            name="temporal_expired",
            query="Is the promo active?",
            answer="The promo is currently active.",
            contexts=["The promo expired in 2019 and is no longer available."],
            expected_release=False,
            tag="temporal",
        ),
        AdversarialCase(
            name="low_reliability_only_source",
            query="What is the warranty?",
            answer="The device has a 2-year warranty covering manufacturing defects.",
            contexts=["The device has a 2-year warranty covering manufacturing defects."],
            expected_release=False,
            tag="reliability",
            source_reliability={"context:0": 0.05},
            notes="Untrusted source cannot be the sole release citation.",
        ),
        AdversarialCase(
            name="fabricated_multihop",
            query="Where is HQ and who is CEO?",
            answer="HQ is in Paris and the CEO is Ada Lovelace.",
            contexts=[
                "The warehouse is in Berlin.",
                "Support hours are 9 to 5.",
            ],
            expected_release=False,
            tag="multihop",
        ),
        AdversarialCase(
            name="copy_paste_must_pass",
            query="What is the warranty?",
            answer="The Titan watch has a 2-year warranty covering manufacturing defects.",
            contexts=[
                "The Titan watch has a 2-year warranty covering manufacturing defects.",
                "Shipping takes 3-5 business days.",
            ],
            expected_release=True,
            tag="control",
        ),
    ]


def inference_cases() -> list[AdversarialCase]:
    """True multi-hop: neither chunk alone covers the composed claim."""
    return [
        AdversarialCase(
            name="true_multihop_hq_ceo",
            query="Where is HQ and who leads the company?",
            answer="Headquarters is in Austin and the CEO is Jordan Lee.",
            contexts=[
                "Acme headquarters is in Austin, Texas.",
                "Jordan Lee is the CEO of Acme.",
            ],
            expected_release=True,
            tag="inferred",
            notes="Balanced may release as inferred; strict should tag inferred and not false-release junk.",
        ),
    ]
