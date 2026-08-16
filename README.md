# hallucination-gate

[![PyPI](https://img.shields.io/pypi/v/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![Python](https://img.shields.io/pypi/pyversions/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml)

**RAG quality system + conservative release gate** for RAG and fine-tuned LLMs.

- **Eval:** claim-level faithfulness / relevancy / context metrics
- **Retrieval:** hit@k, recall@k, MRR, nDCG@k
- **Regression:** save baseline → diff → fail CI
- **Latency budget:** p50 / p95 / p99 / max ceilings
- **Gate:** pass / rewrite / abstain for production `safe_answer`

Author: **Shreyas G**.

## Install

```bash
pip install -U hallucination-gate
pip install "hallucination-gate[ocr]"   # optional OCR
```

## RAG eval (RAGAS replacement path)

```python
from hallucination_gate import RAGEval, LatencyBudget

evaler = RAGEval(
    use_heuristic=True,  # CI; omit for neural production eval
    latency_budget=LatencyBudget(p95_ms=1500, max_ms=5000),
)
report = evaler.evaluate(
    [
        {
            "query": "What is the warranty?",
            "answer": "The Titan watch has a 2-year warranty.",
            "contexts": [
                "The Titan watch has a 2-year warranty covering defects.",
                "Shipping takes 3-5 days.",
            ],
            "ground_truth": "2-year warranty for manufacturing defects.",
            "relevant_contexts": [
                "The Titan watch has a 2-year warranty covering defects."
            ],
            # or: "relevant_indices": [0],
        }
    ],
    save_baseline_path="baselines/titan.json",
    # baseline_path="baselines/titan.json",
    # fail_on_regression=True,
)
print(report.aggregate)   # faithfulness, answer_relevancy, ...
print(report.retrieval)   # hit_at_k, mrr, ndcg_at_k, ...
print(report.latency)     # p50/p95/p99 + budget ok
report.raise_if_failed()
```

```bash
hallucination-gate eval-dataset samples.jsonl --out report.json \
  --save-baseline baselines/titan.json --p95-ms 1500

# later in CI:
hallucination-gate eval-dataset samples.jsonl \
  --baseline baselines/titan.json --fail-on-regression --p95-ms 1500
```

| Metric | How this package scores it |
|---|---|
| **faithfulness** | Fraction of answer claims supported by individual chunks (contradictions penalize) |
| **answer_relevancy** | Query↔answer embedding relevance |
| **context_precision** | Labeled `relevant_contexts` if provided; else claim-aligned chunk proxy |
| **context_recall** | Requires `ground_truth` — fraction of reference facts covered by contexts |
| **hit@k / MRR / nDCG** | Ranked retrieval vs `relevant_contexts` or `relevant_indices` |
| **latency budget** | p50/p95/p99/max vs `LatencyBudget` |
| **regression** | Diff aggregates/retrieval/latency vs saved baseline; fail CI on drops |
| **groundedness / hallucination_risk / release_safety** | BN posteriors from the same evidence stack |

**Why this beats typical RAGAS setups for grounding:** claim-level soft-OR against neighbors, retrieval+latency+regression in one report, false-release oriented gate, multimodal/OCR evidence, and a production `safe_answer` path — not only a mean score.

## Production gate

```python
from hallucination_gate import HallucinationGate, Evidence

gate = HallucinationGate()
result = gate.check(query, answer, context=retrieved_docs)
return result.text
```

```python
report = gate.evaluate(samples)  # same backends as the gate
```

## OCR

```python
from hallucination_gate import Evidence, ocr_available

ev = Evidence.from_image(path="warranty_card.jpg")
ev = Evidence.from_ocr(path="scanned_policy.pdf")
```

## Drawbacks (honest)

- **Latency & cost** — neural path adds inference time / GPU·CPU load per sample.
- **Over-refusal** — conservative gate can abstain on good extractive answers.
- **Only as good as evidence** — checks support, not world truth; bad retrieval still hurts.
- **Hard cases** — subtle math/code/reasoning can fool or over-block NLI.
- **Heuristic ≠ quality gate** — `use_heuristic=True` is for CI smoke, not calibrated faithfulness.
- **Ops surface** — HF downloads, torch/sentence-transformers weight, Windows symlink quirks.
- **Not magic** — still needs your domain labels (`relevant_contexts` / `ground_truth`) and human review for hard cases; the stack now covers grounding + retrieval + latency SLOs + regression diffs.

## Eval (gate safety)

```bash
pip install -e ".[dev]"
set RAG_EVAL_HEURISTIC=1
pytest -q -m "not neural"
hallucination-gate eval-heldout
```

## License

MIT © Shreyas G
