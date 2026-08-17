"""Adversarial gate eval + naive competitor baselines (overlap / cosine-only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from bayesian_rag_evaluator.data_gen.adversarial import (
    AdversarialCase,
    adversarial_cases,
    inference_cases,
)
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.evidence.backends import (
    HeuristicEmbeddingBackend,
    jaccard_similarity,
    token_coverage,
)
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, ModelType

CompetitorName = Literal["hallucination_gate", "overlap", "cosine"]


@dataclass
class BenchmarkRow:
    name: str
    tag: str
    competitor: str
    released: bool
    expected_release: bool
    false_release: bool
    over_refuse: bool
    action: str | None = None
    grounding_kind: str | None = None


@dataclass
class BenchmarkReport:
    n: int
    competitor: str
    false_release_rate: float
    over_refuse_rate: float
    accuracy: float
    rows: list[BenchmarkRow] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _overlap_release(answer: str, contexts: list[str]) -> bool:
    if not contexts:
        return False
    return max(token_coverage(answer, c) for c in contexts) >= 0.50


def _cosine_release(answer: str, contexts: list[str]) -> bool:
    if not contexts:
        return False
    embed = HeuristicEmbeddingBackend()
    sims = embed.similarity_matrix([answer], contexts)
    return float(sims.max()) >= 0.50


def _gate_eval(
    case: AdversarialCase,
    evaluator: DiagnosticEvaluator,
) -> tuple[bool, str, str | None]:
    result = evaluator.evaluate(
        EvaluateRequest(
            query=case.query,
            answer=case.answer,
            context_chunks=case.contexts,
            model_type=ModelType.RAG,
            source_reliability=case.source_reliability or {},
        )
    )
    kind = result.claims[0].grounding_kind.value if result.claims else None
    return result.gate.released, result.gate.action.value, kind


def run_competitor(
    cases: list[AdversarialCase],
    competitor: CompetitorName,
    evaluator: DiagnosticEvaluator | None = None,
) -> BenchmarkReport:
    evaluator = evaluator or DiagnosticEvaluator(
        use_heuristic=True, policy="strict", align_contexts=False
    )
    rows: list[BenchmarkRow] = []
    for case in cases:
        kind = None
        action = None
        if competitor == "hallucination_gate":
            released, action, kind = _gate_eval(case, evaluator)
        elif competitor == "overlap":
            released = _overlap_release(case.answer, case.contexts)
        else:
            released = _cosine_release(case.answer, case.contexts)
        false_release = released and not case.expected_release
        over_refuse = (not released) and case.expected_release
        rows.append(
            BenchmarkRow(
                name=case.name,
                tag=case.tag,
                competitor=competitor,
                released=released,
                expected_release=case.expected_release,
                false_release=false_release,
                over_refuse=over_refuse,
                action=action,
                grounding_kind=kind,
            )
        )
    n = len(rows) or 1
    fr = sum(1 for r in rows if r.false_release) / len(rows) if rows else 0.0
    ore = sum(1 for r in rows if r.over_refuse) / len(rows) if rows else 0.0
    acc = sum(1 for r in rows if r.released == r.expected_release) / n
    return BenchmarkReport(
        n=len(rows),
        competitor=competitor,
        false_release_rate=round(fr, 4),
        over_refuse_rate=round(ore, 4),
        accuracy=round(acc, 4),
        rows=rows,
    )


def run_adversarial_suite(
    evaluator: DiagnosticEvaluator | None = None,
) -> dict[str, Any]:
    """Strict gate on the adversarial set — false-release must stay ~0."""
    cases = adversarial_cases()
    report = run_competitor(cases, "hallucination_gate", evaluator=evaluator)
    return {
        "suite": "adversarial",
        "n": report.n,
        "false_release_rate": report.false_release_rate,
        "over_refuse_rate": report.over_refuse_rate,
        "accuracy": report.accuracy,
        "failures": [r.name for r in report.rows if r.false_release],
        "rows": [asdict(r) for r in report.rows],
    }


def run_competitor_benchmark(
    evaluator: DiagnosticEvaluator | None = None,
) -> dict[str, Any]:
    cases = adversarial_cases()
    out: dict[str, Any] = {"suite": "competitor", "competitors": {}}
    for name in ("hallucination_gate", "overlap", "cosine"):
        report = run_competitor(cases, name, evaluator=evaluator)  # type: ignore[arg-type]
        out["competitors"][name] = {
            "false_release_rate": report.false_release_rate,
            "over_refuse_rate": report.over_refuse_rate,
            "accuracy": report.accuracy,
            "n": report.n,
        }
    return out
