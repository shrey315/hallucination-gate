# hallucination-gate

A **conservative grounding gate** for RAG and fine-tuned generators. It does not decide whether an answer is true in the world. It decides whether the answer is **supported by the evidence you pass in**, then **passes**, **rewrites**, or **abstains**.

False release is treated as the failure mode that matters. If a rewrite would no longer answer the question, the gate abstains.

```python
from hallucination_gate import HallucinationGate, Evidence

gate = HallucinationGate()  # neural backends (production default)
# gate = HallucinationGate(use_heuristic=True)  # CI / offline smoke only

result = gate.check(
    query=user_query,
    answer=llm_answer,
    context=retrieved_docs,  # str | list[str] | LangChain Document | LlamaIndex node | dict
)
return result.text  # show this to users
```

Each claim is scored **against individual chunks**, then soft-OR aggregated: a supporting chunk wins even if a neighbor has unrelated numbers. Contradiction only counts from *aligned* chunks (enough lexical/semantic overlap with the claim).

```python
gate = HallucinationGate(mode="fine_tuned")
result = gate.check(query, answer, kb=your_knowledge_base)

result = gate.check(query, answer, evidence=Evidence.from_image(path="photo.jpg", ocr="..."))
result = gate.check(query, answer, evidence=Evidence.from_pdf("policy.pdf"))
```

### OCR (high-tech path)

```bash
pip install "hallucination-gate[ocr]"
# system dependency: Tesseract OCR binary (and poppler for scanned PDFs)
```

```python
from hallucination_gate import Evidence, ocr_available

print(ocr_available())  # {"pillow": True, "tesseract": True, "easyocr": True}

# Auto-OCR an image path into grounding evidence
ev = Evidence.from_image(path="warranty_card.jpg")  # preprocess + Tesseract/EasyOCR
# Or OCR a scanned PDF page set into documents
ev = Evidence.from_ocr(path="scanned_policy.pdf")
```

Engines: **Tesseract** (default) and optional **EasyOCR**, with upscale / contrast / denoise preprocess. Image-only PDFs fall back to page OCR when `pypdf` extracts no text.

```python
@gate.protect
def my_rag(query: str):
    docs = retriever(query)
    answer = llm(query, docs)
    return answer, docs
```

## What evidence to pass

Pass the chunks that *should* ground the answer. Typical RAG top-k is fine now that scoring is per-chunk, but garbage neighbors still waste work and can create borderline UNCERTAIN hits.

| Pattern | When |
|---|---|
| Full retriever top-k | Default. Claim↔chunk alignment handles mixed neighbors. |
| Answer-aligned / reranked top-k | Best production default — keep chunks that cite the same entities/numbers as the draft answer. |
| Citation-level evidence | If the generator emits citations, pass only those cited chunks. |
| Top-1 only | Debugging / demos; too brittle for real retrieval. |

Do **not** concatenate all chunks into one string before calling `check` — pass a `list[str]` (or documents) so boundaries are preserved.

Inspect grounding with `result.claims`: each claim has `status`, `source_id`, `citation`, `reason`, and `chunk_hits` (per-chunk scores).

## Heuristic vs neural (contract)

| Mode | How | Use for |
|---|---|---|
| **Neural** (default) | MiniLM embeddings + DeBERTa NLI | Production gate. Override with `embed_model=` / `nli_model=` or `RAG_EVAL_EMBED_MODEL` / `RAG_EVAL_NLI_MODEL`. |
| **Heuristic** | Token overlap / negation / number heuristics | CI smoke, offline unit tests. Set `use_heuristic=True` or `RAG_EVAL_HEURISTIC=1`. |

Heuristic is **not** calibrated to neural false-release / over-refusal rates. Treat heuristic PASS/ABSTAIN as a wiring check, not a quality gate. Ship production with neural backends (or an explicit judge escalation).

Optional: `HALLUCINATION_GATE_JUDGE=1` plus `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` escalates **uncertain** claims only, using the top aligned chunks (not the full bag).

## What this is (and is not)

| It does | It does not |
|---|---|
| Check claims against *your* retrieved chunks / KB / OCR / PDF text | Know if the KB itself is wrong |
| Abstain on contradiction, invented entities, and number clashes | Read fine-tune weights |
| Drop ungrounded sentences, then abstain if the remainder misses the query | Replace an LLM-as-judge on subtle reasoning, code, or math proofs |
| Work with any stack that can give you `query`, `answer`, `evidence` | Guarantee multilingual performance equal to English without swapping models |

Release is decided by **claim grounding**, not by the Bayesian network. BN scores are diagnostics only.

## Eval

Held-out domains (HR, API, vaccines, Redis, K8s) report **false release** and **over-refusal**:

```bash
pip install -e ".[dev]"
set RAG_EVAL_HEURISTIC=1
pytest -q -m "not neural"
hallucination-gate eval-heldout
```

## Install

```bash
pip install hallucination-gate
```

From GitHub:

```bash
pip install git+https://github.com/shrey315/hallucination-gate.git
```

From this folder:

```bash
pip install -e ".[dev]"
pytest -q
```

## HTTP API (optional)

Only needed if you want a separate service. Library users do **not** need a running server.

```bash
uvicorn bayesian_rag_evaluator.api.main:app --reload --port 8000
```

`POST /v1/answer` returns only `{safe_answer, released, request_id, latency_ms}`.

## License

MIT
