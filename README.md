# hallucination-gate

[![PyPI](https://img.shields.io/pypi/v/hallucination-gate.svg?cacheSeconds=300)](https://pypi.org/project/hallucination-gate/)
[![Python](https://img.shields.io/pypi/pyversions/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml)

**RAG quality system + conservative release gate** for RAG and fine-tuned LLMs.

End-to-end system design (pipeline, claim lock, BN, eval, sidecar): **[ARCHITECTURE.md](ARCHITECTURE.md)**.

Visual architecture (download): [SVG](docs/architecture-visual.svg) · [PNG](docs/architecture-visual.png)

- **Eval:** claim-level faithfulness / relevancy / context metrics
- **Retrieval:** hit@k, recall@k, MRR, nDCG@k
- **Regression:** save baseline → diff → fail CI
- **Latency budget:** p50 / p95 / p99 / max ceilings
- **Modes:** `ci` (heuristic smoke) vs `quality` (MiniLM + DeBERTa-small) vs `quality_plus` (mpnet + DeBERTa-base); policies `strict` / `balanced`
- **Gate:** pass / rewrite / abstain for production `safe_answer`
- **Lock upgrades:** structured claims, composed 2–3 hop (strict may pass), inferred NLI-only (strict will not), retrieval-poison abstain, temporal/negation/scope, source reliability
- **Bench:** `hallucination-gate eval-adversarial` / `eval-benchmark`

Author: **Shreyas G**.

## Install

```bash
pip install -U hallucination-gate
pip install "hallucination-gate[neural]"   # MiniLM + DeBERTa (quality / quality_plus)
pip install "hallucination-gate[api]"      # FastAPI sidecar
pip install "hallucination-gate[bn]"       # optional BN diagnostics
pip install "hallucination-gate[ocr]"      # optional OCR
```

Core install is heuristic (`ci`) only — no Hugging Face download. Neural models download on first `quality` / `quality_plus` call.

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

# labeled false-release on *your* DB traces:
hallucination-gate eval-corpus your_labels.jsonl

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

gate = HallucinationGate(
    quality_mode="quality",  # "ci" smoke; "quality_plus" for stronger NLI
    policy="balanced",       # or "strict" for max false-release lock
    warm=True,               # preload models — cuts cold-start tails
)
result = gate.check(query, answer, context=retrieved_docs)
# result.release_authority == "claim_status"
# result.scores_are_calibrated is False
# result.evidence_gap in {"none","retrieval","generation","mixed","contradiction"}
return result.text
```

Shadow mode logs the gate without changing user-visible text:

```python
gate = HallucinationGate(use_heuristic=True, policy="strict", shadow=True)
result = gate.check(query, answer, context=retrieved_docs)
# users still see the model answer
print(result.text)
# counterfactual enforce decision
print(result.gated_text, result.released, result.action)
```

Context chunks are **aligned/filtered** to the query+answer by default (generic overlap/embedding score — no domain lexicon). Metrics expose both `context_precision_labeled` and `context_precision_aligned`.

```python
report = gate.evaluate(samples)  # same backends as the gate
```

## OCR

```python
from hallucination_gate import Evidence, ocr_available

ev = Evidence.from_image(path="warranty_card.jpg")
ev = Evidence.from_ocr(path="scanned_policy.pdf")
```

## Drawbacks

- **BN is diagnostic.** `safe_answer` is decided by claim status, not by Bayesian posteriors. Those scores are discrete fusion (`P(high)+0.5·P(medium)`), not calibrated P(hallucination).
- **Latency & cost** — neural path adds inference time / GPU·CPU load per sample. `quality_plus` is heavier on purpose.
- **Over-refusal** — conservative gate can abstain on good extractive answers. Published rates: [docs/EVAL.md](docs/EVAL.md).
- **Only as good as evidence** — checks support, not world truth. Faithful answers to weakly aligned chunks abstain (`evidence_gap=retrieval`). The gate cannot invent missing chunks.
- **Shadow** — `HallucinationGate(shadow=True)` keeps user-visible `text` as the model answer; inspect `gated_text` / `released` as the counterfactual.
- **Hard cases** — math/code extras and equation clashes are on the lock. **Composed** multi-hop (AND of extractive facts across 2–3 chunks) may release; speculative NLI-only **inferred** joins do not in `strict`.
- **Heuristic ≠ quality gate** — `use_heuristic=True` is for CI smoke, not calibrated faithfulness.
- **Ops surface** — sidecar Prometheus `/metrics`, per-key tenant *labels*, shared process. Not a multi-tenant platform.
- **Not magic** — still needs your domain labels (`relevant_contexts` / `ground_truth`) and human review for hard cases.

## Eval (gate safety)

```bash
pip install -e ".[dev]"
set RAG_EVAL_HEURISTIC=1
pytest -q -m "not neural"
hallucination-gate eval-heldout
hallucination-gate eval-adversarial
hallucination-gate eval-benchmark
hallucination-gate eval-corpus
hallucination-gate eval-heldout --neural
hallucination-gate eval-corpus your_labels.jsonl
```

Published false-release and over-refusal: [docs/EVAL.md](docs/EVAL.md).

## License

MIT © Shreyas G
