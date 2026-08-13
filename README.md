# hallucination-gate

Generic Python library that sits in front of **any** RAG or fine-tuned generator. It does not train models, does not ship a dataset, and does not care which vector store or LLM you use.

```python
from hallucination_gate import HallucinationGate, Evidence

gate = HallucinationGate()  # or mode="fine_tuned"

result = gate.check(
    query=user_query,
    answer=llm_answer,
    context=retrieved_docs,  # str | list[str] | LangChain Document | LlamaIndex node | dict
)
return result.text  # show this to users
```

```python
# Fine-tuned
gate = HallucinationGate(mode="fine_tuned")
result = gate.check(query, answer, kb=your_knowledge_base)

# Image / OCR / PDF / table / audio
result = gate.check(query, answer, evidence=Evidence.from_image(path="photo.jpg", ocr="..."))
result = gate.check(query, answer, evidence=Evidence.from_pdf("policy.pdf"))
```

```python
@gate.protect
def my_rag(query: str):
    docs = retriever(query)
    answer = llm(query, docs)
    return answer, docs
```

## Install

After the package is on PyPI:

```bash
pip install hallucination-gate
```

From GitHub (works as soon as the repo is public):

```bash
pip install git+https://github.com/YOUR_GITHUB_USERNAME/hallucination-gate.git
```

From this folder:

```bash
pip install -e .
```

Set `RAG_EVAL_HEURISTIC=1` for a lightweight scorer (no model download).

## Publish to GitHub then PyPI

PyPI Trusted Publishing expects a **public GitHub repo**. Push this project, then connect it on PyPI.

```bash
cd "E:\Baysian Optimization"
git init
git add .
git commit -m "Initial hallucination-gate library"
gh repo create hallucination-gate --public --source=. --remote=origin --push
```

On [pypi.org](https://pypi.org): account → Publishing → GitHub, with:

- Owner: your GitHub username
- Repository: `hallucination-gate`
- Workflow: `publish.yml`
- Environment: leave empty unless you created one

Then create a GitHub Release tagged `v0.4.0`. The workflow in `.github/workflows/publish.yml` builds and uploads to PyPI. After that, `pip install hallucination-gate` works for everyone.

## HTTP API (optional)

```bash
uvicorn bayesian_rag_evaluator.api.main:app --reload --port 8000
```

`POST /v1/answer` returns only `{safe_answer, released, request_id, latency_ms}`.

## License

MIT
