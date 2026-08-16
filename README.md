# hallucination-gate

[![PyPI](https://img.shields.io/pypi/v/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![Python](https://img.shields.io/pypi/pyversions/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml)

**RAG evaluation + conservative release gate** for RAG and fine-tuned LLMs.

- **Eval:** RAGAS-class metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`) on **claim↔chunk** grounding — not whole-answer NLI only.
- **Gate:** pass / rewrite / abstain so production only ships supported text.

Author: **Shreyas G**.

## Install

```bash
pip install -U hallucination-gate
pip install "hallucination-gate[ocr]"   # optional OCR
```

## RAG eval (RAGAS replacement path)

```python
from hallucination_gate import RAGEval

evaler = RAGEval()  # neural; use_heuristic=True for CI
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
            # optional labeled retrieval:
            # "relevant_contexts": ["The Titan watch has a 2-year warranty covering defects."],
        }
    ]
)
print(report.aggregate)
# {'faithfulness': ..., 'answer_relevancy': ..., 'context_precision': ..., ...}
report.to_json("report.json")
```

```bash
hallucination-gate eval-dataset samples.jsonl --out report.json
```

| Metric | How this package scores it |
|---|---|
| **faithfulness** | Fraction of answer claims supported by individual chunks (contradictions penalize) |
| **answer_relevancy** | Query↔answer embedding relevance |
| **context_precision** | Labeled `relevant_contexts` if provided; else claim-aligned chunk proxy |
| **context_recall** | Requires `ground_truth` — fraction of reference facts covered by contexts |
| **groundedness / hallucination_risk / release_safety** | BN posteriors from the same evidence stack |

**Why this beats typical RAGAS setups for grounding:** claim-level soft-OR against neighbors, false-release oriented gate, multimodal/OCR evidence, and a production `safe_answer` path — not only a mean score.

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
- **Not magic** — still not a full substitute for domain-labeled regression + human review; it is a stronger *grounding-first* eval+gate stack than score-only RAGAS defaults.

## Eval (gate safety)

```bash
pip install -e ".[dev]"
set RAG_EVAL_HEURISTIC=1
pytest -q -m "not neural"
hallucination-gate eval-heldout
```

## License

MIT © Shreyas G
