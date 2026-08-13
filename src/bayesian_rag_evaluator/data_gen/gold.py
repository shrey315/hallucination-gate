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

HALLUCINATED_TAILS = [
    "The CEO lives on Mars.",
    "The company was founded in 1492.",
    "All employees are robots.",
    "Headquarters is on the moon.",
    "This policy was signed by Shakespeare.",
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


def generate_gold_examples(n: int = 2000, seed: int = 42) -> list[GoldExample]:
    """Deterministic 1k–10k claim-level gold set with pass/rewrite/abstain labels."""
    rng = random.Random(seed)
    examples: list[GoldExample] = []
    i = 0
    while len(examples) < n:
        topic = TOPICS[i % len(TOPICS)]
        template = FACTS[i % len(FACTS)]
        n_days = 7 + (i % 50)
        alt = n_days + 17 + (i % 9)
        kind = i % 3  # 0 pass, 1 rewrite, 2 abstain
        query = template[0].format(topic=topic, Topic=topic.capitalize())
        context = template[1].format(topic=topic, Topic=topic.capitalize(), n=n_days, alt=alt)
        supported = template[2].format(topic=topic, Topic=topic.capitalize(), n=n_days, alt=alt)
        bad = template[3].format(topic=topic, Topic=topic.capitalize(), n=n_days, alt=alt)
        tail = HALLUCINATED_TAILS[i % len(HALLUCINATED_TAILS)]
        model_type = ModelType.RAG if rng.random() > 0.15 else ModelType.FINE_TUNED

        if kind == 0:
            example = GoldExample(
                query=query,
                answer=supported,
                context_chunks=[context],
                expected_gate=GateAction.PASS,
                expected_release=True,
                model_type=model_type,
                labels=_disc(
                    relevance="high",
                    faithful="high",
                    entail="high",
                    retrieval="high",
                    complete="high",
                    contra="low",
                    unsupported="low",
                ),
                latent_labels=_latent(0.88, 0.90, 0.10, 0.85, 0.86),
            )
        elif kind == 1:
            example = GoldExample(
                query=query,
                answer=f"{supported} {tail}",
                context_chunks=[context],
                expected_gate=GateAction.REWRITE,
                expected_release=True,
                model_type=model_type,
                labels=_disc(
                    relevance="medium",
                    faithful="medium",
                    entail="medium",
                    retrieval="high",
                    complete="medium",
                    contra="low",
                    unsupported="medium",
                ),
                latent_labels=_latent(0.55, 0.58, 0.42, 0.80, 0.50),
            )
        else:
            numeric = "low" if "{alt}" in template[3] else "high"
            example = GoldExample(
                query=query,
                answer=bad,
                context_chunks=[context],
                expected_gate=GateAction.ABSTAIN,
                expected_release=False,
                model_type=model_type,
                labels=_disc(
                    relevance="medium",
                    faithful="low",
                    entail="low",
                    retrieval="high",
                    complete="medium",
                    contra="high",
                    unsupported="high",
                    numeric=numeric,
                ),
                latent_labels=_latent(0.22, 0.18, 0.82, 0.75, 0.12),
            )
        examples.append(example)
        i += 1
    return examples


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
