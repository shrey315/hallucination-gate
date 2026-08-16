# hallucination-gate

[![PyPI](https://img.shields.io/pypi/v/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![Python](https://img.shields.io/pypi/pyversions/hallucination-gate.svg)](https://pypi.org/project/hallucination-gate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/shrey315/hallucination-gate/actions/workflows/ci.yml)

**Conservative grounding gate** for RAG and fine-tuned LLMs.  
Verify answers against *your* evidence → **pass**, **rewrite**, or **abstain**.

False release is the failure mode that matters. Neighbor chunks no longer veto a claim another chunk fully supports.

## Install

```bash
pip install -U hallucination-gate
pip install "hallucination-gate[ocr]"   # Tesseract / EasyOCR / scanned PDFs
```

## Quick start

```python
from hallucination_gate import HallucinationGate, Evidence

gate = HallucinationGate()  # neural default (production)
# gate = HallucinationGate(use_heuristic=True)  # CI / offline smoke only

result = gate.check(
    query=user_query,
    answer=llm_answer,
    context=retrieved_docs,  # str | list[str] | LangChain Document | dict
)
return result.text  # show this to users
```

```python
gate = HallucinationGate(mode="fine_tuned")
result = gate.check(query, answer, kb=your_knowledge_base)

# Images / PDFs / OCR
result = gate.check(query, answer, evidence=Evidence.from_image(path="warranty_card.jpg"))
result = gate.check(query, answer, evidence=Evidence.from_pdf("policy.pdf"))
result = gate.check(query, answer, evidence=Evidence.from_ocr(path="scanned.pdf"))
```

```python
@gate.protect
def my_rag(query: str):
    docs = retriever(query)
    answer = llm(query, docs)
    return answer, docs
```

Inspect `result.claims` / `result.diagnostics` for claim↔chunk status, citations, and reasons.

## OCR

```bash
pip install "hallucination-gate[ocr]"
# system: Tesseract binary (+ poppler for scanned PDFs)
```

```python
from hallucination_gate import Evidence, ocr_available

print(ocr_available())  # pillow / tesseract / easyocr

ev = Evidence.from_image(path="card.jpg")          # auto-OCR + preprocess
ev = Evidence.from_ocr(path="scanned_policy.pdf")  # page OCR fallback
```

Upscale → contrast → denoise, then **Tesseract** and/or **EasyOCR**. Image-only PDFs OCR when text extract is empty.

## Evidence patterns

| Pattern | When |
|---|---|
| Full retriever top-k | Default. Per-chunk soft-OR handles mixed neighbors. |
| Reranked / answer-aligned top-k | Best production default. |
| Citation-level chunks | If the generator emits citations, pass only those. |
| Top-1 only | Demos; too brittle for real retrieval. |

Pass a `list[str]` (or documents)—do **not** concatenate top-k into one bag.

## Heuristic vs neural

| Mode | How | Use for |
|---|---|---|
| **Neural** (default) | MiniLM + DeBERTa NLI | Production |
| **Heuristic** | Token / negation / number heuristics | CI smoke (`use_heuristic=True` or `RAG_EVAL_HEURISTIC=1`) |

Heuristic is a wiring check, not a calibrated quality gate.

Optional judge on uncertain claims only: `HALLUCINATION_GATE_JUDGE=1` + `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

## What this is (and is not)

| It does | It does not |
|---|---|
| Ground claims in *your* chunks / KB / OCR / PDF | Know if the KB itself is wrong |
| Abstain on contradiction, invented entities, number clashes | Read fine-tune weights |
| Drop ungrounded sentences; abstain if the rest misses the query | Replace LLM-as-judge on subtle code/math proofs |

Release is decided by **claim grounding**. BN scores are diagnostics only.

## Drawbacks

- **Latency & cost** — the neural path adds real inference time (and GPU/CPU load) on every gated answer.
- **Over-refusal** — conservative by design; correct extractive answers can still be abstained (e.g. ~0.9 release on some held-out suites).
- **Only as good as evidence** — if retrieval is wrong or incomplete, the gate cannot save you. It checks *support*, not world truth.
- **Weak on hard cases** — subtle reasoning, math, code, or paraphrases that NLI misses can slip through or get blocked wrongly.
- **Heuristic ≠ quality gate** — token heuristics are a wiring check, not a calibrated faithfulness metric.
- **Ops surface** — model downloads, Hugging Face Hub, Windows symlink quirks, and heavy deps (`sentence-transformers`, `torch`, etc.).
- **Diagnostics vs product UX** — `claims` / `diagnostics` help you debug; end users still just see abstain/rewrite text.

**Bottom line:** Strong as a conservative release firewall *after* retrieval. Weak as a fast, high-recall answer scorer or a full substitute for eval frameworks (RAGAS-style metrics, regression suites, domain labels).

## Eval

```bash
pip install -e ".[dev]"
set RAG_EVAL_HEURISTIC=1
pytest -q -m "not neural"
hallucination-gate eval-heldout
```

## HTTP API (optional)

Library users do **not** need a server.

```bash
uvicorn bayesian_rag_evaluator.api.main:app --reload --port 8000
```

`POST /v1/answer` → `{safe_answer, released, request_id, latency_ms}` only.

## License

MIT
