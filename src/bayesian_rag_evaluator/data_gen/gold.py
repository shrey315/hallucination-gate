from __future__ import annotations

import random
from pathlib import Path

from bayesian_rag_evaluator.models.schemas import (
    DiscretizedEvidence,
    GateAction,
    GoldExample,
    LabeledExample,
    ModelType,
    PosteriorScores,
)

FACTS: list[tuple[str, str, str, str]] = [
    (
        "What is the {topic} policy?",
        "Customers may request a {topic} within {n} days of purchase.",
        "Customers can request a {topic} within {n} days of purchase.",
        "{Topic} is never allowed under any circumstances.",
    ),
    (
        "How long is the {topic} window?",
        "The {topic} window is {n} days.",
        "The {topic} window is {n} days.",
        "The {topic} window is {alt} days.",
    ),
    (
        "Is {topic} available?",
        "{Topic} is available for all verified accounts.",
        "{Topic} is available for verified accounts.",
        "{Topic} is never available to anyone.",
    ),
    (
        "What is the {topic} fee?",
        "The {topic} fee is {n} dollars.",
        "The {topic} fee is {n} dollars.",
        "The {topic} fee is {alt} dollars.",
    ),
    (
        "Who supports {topic}?",
        "Support for {topic} is handled by the billing team.",
        "The billing team supports {topic}.",
        "The marketing team on Mars handles {topic}.",
    ),
]

TOPICS = [
    "refund",
    "return",
    "warranty",
    "exchange",
    "cancellation",
    "shipping",
    "upgrade",
    "replacement",
    "credit",
    "discount",
]

# Non-numeric invented tails so mixed answers rewrite instead of numeric-contradict.
HALLUCINATED_TAILS = [
    "The CEO lives on Mars.",
    "All employees are robots.",
    "Headquarters is on the moon.",
    "This policy was signed by Shakespeare.",
    "Every customer is a fictional character.",
]


def _disc(
    *,
    relevance: str,
    faithful: str,
    entail: str,
    retrieval: str,
    complete: str,
    contra: str,
    unsupported: str,
    numeric: str = "high",
) -> DiscretizedEvidence:
    return DiscretizedEvidence(
        query_relevance=relevance,  # type: ignore[arg-type]
        context_faithfulness=faithful,  # type: ignore[arg-type]
        entailment_score=entail,  # type: ignore[arg-type]
        retrieval_quality=retrieval,  # type: ignore[arg-type]
        completeness=complete,  # type: ignore[arg-type]
        contradiction=contra,  # type: ignore[arg-type]
        unsupported_claims=unsupported,  # type: ignore[arg-type]
        numeric_consistency=numeric,  # type: ignore[arg-type]
        model_type="rag",
    )


def _latent(quality: float, ground: float, hall: float, retrieval: float, safety: float) -> PosteriorScores:
    return PosteriorScores(
        answer_quality=quality,
        groundedness=ground,
        hallucination_risk=hall,
        retrieval_adequacy=retrieval,
        release_safety=safety,
    )


def _pass_labels() -> tuple[DiscretizedEvidence, PosteriorScores]:
    return (
        _disc(
            relevance="high",
            faithful="high",
            entail="high",
            retrieval="high",
            complete="high",
            contra="low",
            unsupported="low",
        ),
        _latent(0.88, 0.90, 0.10, 0.85, 0.86),
    )


def _rewrite_labels() -> tuple[DiscretizedEvidence, PosteriorScores]:
    return (
        _disc(
            relevance="medium",
            faithful="medium",
            entail="medium",
            retrieval="high",
            complete="medium",
            contra="low",
            unsupported="medium",
        ),
        _latent(0.55, 0.58, 0.42, 0.80, 0.50),
    )


def _abstain_labels(*, numeric: str = "high") -> tuple[DiscretizedEvidence, PosteriorScores]:
    return (
        _disc(
            relevance="medium",
            faithful="low",
            entail="low",
            retrieval="high",
            complete="medium",
            contra="high",
            unsupported="high",
            numeric=numeric,
        ),
        _latent(0.22, 0.18, 0.82, 0.75, 0.12),
    )


def hard_cases() -> list[GoldExample]:
    """Adversarial and paraphrase cases the gate must get right."""
    pass_l, pass_z = _pass_labels()
    rewrite_l, rewrite_z = _rewrite_labels()
    abs_l, abs_z = _abstain_labels()
    abs_num, abs_num_z = _abstain_labels(numeric="low")
    return [
        GoldExample(
            query="What is the refund policy?",
            answer="A customer can request a refund within 30 days of purchase.",
            context_chunks=["Customers may request a refund within 30 days of purchase."],
            expected_gate=GateAction.PASS,
            expected_release=True,
            labels=pass_l,
            latent_labels=pass_z,
        ),
        GoldExample(
            query="What color is the sedan?",
            answer="The sedan is red.",
            context_chunks=["A red sedan is parked on the street."],
            expected_gate=GateAction.PASS,
            expected_release=True,
            labels=pass_l,
            latent_labels=pass_z,
        ),
        GoldExample(
            query="What is the refund policy?",
            answer="Refunds are never allowed under any circumstances.",
            context_chunks=["Customers may request a refund within 30 days of purchase."],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_l,
            latent_labels=abs_z,
        ),
        GoldExample(
            query="How long is the refund window?",
            answer="The refund window is 365 days.",
            context_chunks=["The refund window is 30 days."],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_num,
            latent_labels=abs_num_z,
        ),
        GoldExample(
            query="Who supports refunds?",
            answer="The marketing team on Mars handles refunds.",
            context_chunks=["Support for refunds is handled by the billing team."],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_l,
            latent_labels=abs_z,
        ),
        GoldExample(
            query="What is the refund policy?",
            answer=(
                "Customers can request a refund within 30 days of purchase. "
                "The CEO lives on Mars."
            ),
            context_chunks=["Customers may request a refund within 30 days of purchase."],
            expected_gate=GateAction.REWRITE,
            expected_release=True,
            labels=rewrite_l,
            latent_labels=rewrite_z,
        ),
        GoldExample(
            query="Who founded Acme?",
            answer="Refunds are available within 30 days. Acme was founded by Shakespeare.",
            context_chunks=["Refunds are available within 30 days of purchase."],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_l,
            latent_labels=abs_z,
        ),
        GoldExample(
            query="What is the exchange fee?",
            answer="The exchange fee is 40 dollars.",
            context_chunks=["The exchange fee is 15 dollars."],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_num,
            latent_labels=abs_num_z,
        ),
        GoldExample(
            query="Is warranty available?",
            answer="Warranty is available for verified accounts.",
            context_chunks=["Warranty is available for all verified accounts."],
            expected_gate=GateAction.PASS,
            expected_release=True,
            labels=pass_l,
            latent_labels=pass_z,
        ),
        GoldExample(
            query="退款政策是什么？",
            answer="客户可在购买后30天内申请退款。",
            context_chunks=["客户可在购买后30天内申请退款。"],
            expected_gate=GateAction.PASS,
            expected_release=True,
            labels=pass_l,
            latent_labels=pass_z,
        ),
        GoldExample(
            query="退款政策是什么？",
            answer="退款永远不允许。",
            context_chunks=["客户可在购买后30天内申请退款。"],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_l,
            latent_labels=abs_z,
        ),
        GoldExample(
            query="What is the shipping policy?",
            answer="Shipping is free worldwide and unicorns deliver overnight.",
            context_chunks=["Standard shipping takes 5 business days."],
            expected_gate=GateAction.ABSTAIN,
            expected_release=False,
            labels=abs_l,
            latent_labels=abs_z,
        ),
    ]


def generate_gold_examples(n: int = 2000, seed: int = 42) -> list[GoldExample]:
    """Deterministic gold set: hard cases first, then balanced pass/rewrite/abstain."""
    rng = random.Random(seed)
    examples = list(hard_cases())
    i = 0
    while len(examples) < n:
        topic = TOPICS[i % len(TOPICS)]
        template = FACTS[i % len(FACTS)]
        n_days = 7 + (i % 50)
        alt = n_days + 17 + (i % 9)
        kind = i % 3
        query = template[0].format(topic=topic, Topic=topic.capitalize())
        context = template[1].format(topic=topic, Topic=topic.capitalize(), n=n_days, alt=alt)
        supported = template[2].format(topic=topic, Topic=topic.capitalize(), n=n_days, alt=alt)
        bad = template[3].format(topic=topic, Topic=topic.capitalize(), n=n_days, alt=alt)
        tail = HALLUCINATED_TAILS[i % len(HALLUCINATED_TAILS)]
        model_type = ModelType.RAG if rng.random() > 0.15 else ModelType.FINE_TUNED

        if kind == 0:
            labels, latent = _pass_labels()
            example = GoldExample(
                query=query,
                answer=supported,
                context_chunks=[context],
                expected_gate=GateAction.PASS,
                expected_release=True,
                model_type=model_type,
                labels=labels,
                latent_labels=latent,
            )
        elif kind == 1:
            labels, latent = _rewrite_labels()
            example = GoldExample(
                query=query,
                answer=f"{supported} {tail}",
                context_chunks=[context],
                expected_gate=GateAction.REWRITE,
                expected_release=True,
                model_type=model_type,
                labels=labels,
                latent_labels=latent,
            )
        else:
            numeric = "low" if "{alt}" in template[3] else "high"
            labels, latent = _abstain_labels(numeric=numeric)
            example = GoldExample(
                query=query,
                answer=bad,
                context_chunks=[context],
                expected_gate=GateAction.ABSTAIN,
                expected_release=False,
                model_type=model_type,
                labels=labels,
                latent_labels=latent,
            )
        examples.append(example)
        i += 1
    return examples[:n]


def gold_to_labeled(examples: list[GoldExample]) -> list[LabeledExample]:
    return [
        LabeledExample(
            query=ex.query,
            answer=ex.answer,
            context_chunks=ex.context_chunks,
            model_type=ex.model_type,
            labels=ex.labels,
            latent_labels=ex.latent_labels,
        )
        for ex in examples
    ]


def write_gold_jsonl(path: Path, n: int = 2000, seed: int = 42) -> int:
    import json

    examples = generate_gold_examples(n=n, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(mode="json"), ensure_ascii=True) + "\n")
    return len(examples)
