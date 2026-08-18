from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bayesian_rag_evaluator.data_gen.gold import generate_gold_examples, gold_to_labeled, write_gold_jsonl
from bayesian_rag_evaluator.data_gen.heldout import heldout_examples
from bayesian_rag_evaluator.evaluator import DiagnosticEvaluator
from bayesian_rag_evaluator.metrics.gold import evaluate_gold_set
from bayesian_rag_evaluator.models.schemas import EvaluateRequest, ImageInput, ModelType

app = typer.Typer(help="Hallucination-gated Bayesian RAG/LLM evaluator")
console = Console()


@app.command("evaluate")
def evaluate_cmd(
    query: str = typer.Option(..., help="User query"),
    answer: str = typer.Option(..., help="Model answer to evaluate"),
    context: list[str] = typer.Option([], help="Retrieved context chunks"),
    kb: list[str] = typer.Option([], help="Knowledge base chunks"),
    image_caption: list[str] = typer.Option([], help="Image captions / OCR text"),
    pdf: list[str] = typer.Option([], help="PDF paths to extract as evidence"),
    model_type: str = typer.Option("rag", help="rag or fine_tuned"),
    strict: bool = typer.Option(True, help="Block or rewrite ungrounded claims"),
    heuristic: bool = typer.Option(False, help="Use lightweight heuristic scorers"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON response"),
    public: bool = typer.Option(False, help="Print only safe_answer"),
) -> None:
    evaluator = DiagnosticEvaluator(use_heuristic=True if heuristic else None)
    result = evaluator.evaluate(
        EvaluateRequest(
            query=query,
            answer=answer,
            context_chunks=context,
            kb_chunks=kb,
            images=[ImageInput(caption=c) for c in image_caption],
            pdf_paths=pdf,
            model_type=ModelType(model_type),
            strict=strict,
        )
    )
    if public:
        console.print(result.safe_answer)
        return
    if json_output:
        console.print_json(result.model_dump_json())
        return

    table = Table(title="Hallucination Gate")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Verdict", result.verdict)
    table.add_row("Gate", result.gate.action.value)
    table.add_row("Released", str(result.gate.released))
    table.add_row("Latency ms", f"{result.latency_ms:.1f}" if result.latency_ms else "-")
    for key, val in result.scores.model_dump().items():
        table.add_row(key, f"{val:.2f}")
    console.print(table)
    console.print(f"\n[bold]Safe answer[/bold]\n{result.safe_answer}")

    if result.claims:
        console.print("\n[bold]Claims[/bold]")
        for claim in result.claims:
            console.print(
                f"  - [{claim.status.value}] {claim.text} "
                f"(support={claim.support_score:.2f})"
            )

    if result.gaps:
        console.print("\n[bold]Gaps[/bold]")
        for gap in result.gaps:
            console.print(f"  - {gap.dimension} ({gap.severity}): {gap.driver}")

    if result.suggestions:
        console.print("\n[bold]Suggestions[/bold]")
        for s in result.suggestions:
            console.print(f"  - {s}")


@app.command("batch")
def batch_cmd(
    input_path: Path = typer.Argument(..., help="JSON file with list of evaluate requests"),
    output_path: Path = typer.Option(None, help="Optional output JSON path"),
    heuristic: bool = typer.Option(False, help="Use lightweight heuristic scorers"),
) -> None:
    evaluator = DiagnosticEvaluator(use_heuristic=True if heuristic else None)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    requests = [EvaluateRequest.model_validate(item) for item in payload]
    results = [evaluator.evaluate(req).model_dump(mode="json") for req in requests]
    text = json.dumps(results, indent=2)
    if output_path:
        output_path.write_text(text, encoding="utf-8")
        console.print(f"Wrote {len(results)} results to {output_path}")
    else:
        console.print(text)


@app.command("generate-gold")
def generate_gold_cmd(
    output_path: Path = typer.Option(Path("data/gold/claims_2000.jsonl")),
    n: int = typer.Option(2000, help="Number of labeled cases (1000-10000)"),
    seed: int = typer.Option(42),
) -> None:
    count = write_gold_jsonl(output_path, n=n, seed=seed)
    console.print(f"Wrote {count} gold examples to {output_path}")


@app.command("calibrate")
def calibrate_cmd(
    labels_path: Path = typer.Option(
        None, help="Labeled examples JSON/JSONL. Default: generate 2000 gold cases."
    ),
    n: int = typer.Option(2000),
    output_model: Path = typer.Option(Path("config/learned_bn.pkl")),
) -> None:
    if labels_path:
        from bayesian_rag_evaluator.bn.calibration import load_labeled_examples

        examples = load_labeled_examples(labels_path)
    else:
        examples = gold_to_labeled(generate_gold_examples(n=n))
    from bayesian_rag_evaluator.bn.calibration import (
        learn_cpds_from_data,
        save_learned_model,
        tune_thresholds_from_labels,
    )

    model = learn_cpds_from_data(examples)
    thresholds = tune_thresholds_from_labels(examples)
    save_learned_model(model, output_model)
    console.print(
        f"Learned CPTs from {len(examples)} examples. Model valid: {model.check_model()}"
    )
    console.print(f"Suggested threshold bins: {thresholds.get('bins', {})}")
    console.print(f"Saved learned network to {output_model}")


def _gate_evaluator(*, heuristic: bool, neural: bool, mode: str | None = None) -> DiagnosticEvaluator:
    if neural:
        return DiagnosticEvaluator(
            use_heuristic=False, mode=mode or "quality", policy="strict", align_contexts=False
        )
    return DiagnosticEvaluator(
        use_heuristic=True if mode in {None, "ci"} else None,
        mode=mode,
        policy="strict",
        align_contexts=False,
    )


@app.command("eval-heldout")
def eval_heldout_cmd(
    neural: bool = typer.Option(False, help="Use quality (MiniLM+DeBERTa) backends"),
    mode: str | None = typer.Option(None, help="ci | quality | quality_plus"),
) -> None:
    """Report false-release and over-refusal on the held-out domain set."""
    evaluator = _gate_evaluator(heuristic=not neural, neural=neural, mode=mode)
    metrics = evaluate_gold_set(heldout_examples(), evaluator=evaluator)
    payload = metrics.as_dict()
    payload["backend"] = "neural" if neural or (mode and mode != "ci") else "heuristic"
    console.print_json(json.dumps(payload))


@app.command("eval-corpus")
def eval_corpus_cmd(
    path: Path | None = typer.Argument(
        None,
        help="JSON/JSONL of {query, answer, contexts, expected_release}. Omit for the in-repo corpus.",
    ),
    heuristic: bool = typer.Option(True, help="Heuristic backends (CI-friendly)"),
    neural: bool = typer.Option(False, help="Use quality (MiniLM+DeBERTa) backends"),
) -> None:
    """Score false-release / over-refusal on labeled traces (in-repo or your DB)."""
    from bayesian_rag_evaluator.data_gen.corpus import inrepo_corpus_examples
    from bayesian_rag_evaluator.metrics.gold import load_corpus_examples

    if path is None:
        examples = inrepo_corpus_examples()
        source = "inrepo_corpus"
    else:
        examples = load_corpus_examples(path)
        source = str(path)
    evaluator = _gate_evaluator(heuristic=heuristic and not neural, neural=neural)
    metrics = evaluate_gold_set(examples, evaluator=evaluator)
    payload = metrics.as_dict()
    payload["source"] = source
    payload["backend"] = "neural" if neural else "heuristic"
    payload["note"] = (
        "In-repo corpus is packaged FAQ-style labels, not a customer DB. "
        "Pass your own JSONL for production buy-off."
    )
    console.print_json(json.dumps(payload))
    if metrics.false_release_rate > 0:
        raise typer.Exit(code=1)


@app.command("eval-gold")
def eval_gold_cmd(
    n: int = typer.Option(300, help="How many gold cases to score (heuristic, keep modest)"),
    seed: int = typer.Option(42),
) -> None:
    examples = generate_gold_examples(n=n, seed=seed)
    evaluator = DiagnosticEvaluator(use_heuristic=True)
    metrics = evaluate_gold_set(examples, evaluator=evaluator)
    console.print_json(json.dumps(metrics.as_dict()))


@app.command("eval-dataset")
def eval_dataset_cmd(
    path: Path = typer.Argument(..., help="JSON list or JSONL of {query,answer,contexts,...}"),
    heuristic: bool = typer.Option(True, help="Heuristic backends (CI-friendly)"),
    out: Path | None = typer.Option(None, help="Write full report JSON"),
    jsonl_out: Path | None = typer.Option(None, help="Write per-sample JSONL"),
    baseline: Path | None = typer.Option(None, help="Compare against saved baseline JSON"),
    save_baseline: Path | None = typer.Option(None, help="Write baseline JSON from this run"),
    fail_on_regression: bool = typer.Option(False, help="Exit 1 if metrics regress vs baseline"),
    p95_ms: float | None = typer.Option(None, help="Fail if latency p95 exceeds this (ms)"),
    p50_ms: float | None = typer.Option(None, help="Fail if latency p50 exceeds this (ms)"),
    max_ms: float | None = typer.Option(None, help="Fail if any sample exceeds this (ms)"),
) -> None:
    """Full RAG quality report: grounding + retrieval + latency (+ optional regression)."""
    from bayesian_rag_evaluator.metrics.latency import LatencyBudget
    from bayesian_rag_evaluator.metrics.rag_eval import RAGEval

    budget = None
    if p95_ms is not None or p50_ms is not None or max_ms is not None:
        budget = LatencyBudget(p50_ms=p50_ms, p95_ms=p95_ms, max_ms=max_ms)

    report = RAGEval(
        use_heuristic=True if heuristic else None,
        latency_budget=budget,
    ).evaluate_paths(
        path,
        baseline_path=baseline,
        save_baseline_path=save_baseline,
        fail_on_regression=fail_on_regression,
        fail_on_latency=budget is not None,
    )
    table = Table(title=f"RAGEval n={report.n} ok={report.ok}")
    table.add_column("section")
    table.add_column("metric")
    table.add_column("value")
    for key, value in report.aggregate.items():
        table.add_row("quality", key, f"{value:.4f}")
    for key, value in report.retrieval.items():
        table.add_row("retrieval", key, f"{value:.4f}")
    for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms"):
        if key in (report.latency or {}):
            table.add_row("latency", key, f"{report.latency[key]}")
    console.print(table)
    if report.regression:
        console.print_json(json.dumps(report.regression))
    if report.failures:
        console.print("[red]" + "; ".join(report.failures) + "[/red]")
    if out:
        report.to_json(out)
        console.print(f"Wrote {out}")
    if jsonl_out:
        report.to_jsonl(jsonl_out)
        console.print(f"Wrote {jsonl_out}")
    if not report.ok:
        raise typer.Exit(code=1)


@app.command("eval-adversarial")
def eval_adversarial_cmd(
    neural: bool = typer.Option(False, help="Use quality (MiniLM+DeBERTa) backends"),
) -> None:
    """False-release suite: negation, numbers, entities, scope, time, reliability."""
    from bayesian_rag_evaluator.metrics.benchmark import run_adversarial_suite

    evaluator = _gate_evaluator(heuristic=not neural, neural=neural)
    payload = run_adversarial_suite(evaluator=evaluator)
    payload["backend"] = "neural" if neural else "heuristic"
    console.print_json(json.dumps(payload))
    if payload["false_release_rate"] > 0:
        raise typer.Exit(code=1)


@app.command("eval-benchmark")
def eval_benchmark_cmd() -> None:
    """Compare hallucination-gate vs overlap-only and cosine-only baselines."""
    from bayesian_rag_evaluator.metrics.benchmark import run_competitor_benchmark

    console.print_json(json.dumps(run_competitor_benchmark()))


if __name__ == "__main__":
    app()
