# Architecture — hallucination-gate

**Audience:** engineering leadership, integrators, reviewers.
**Product:** v0.9.2 — conservative grounding firewall for RAG and fine-tuned LLMs.
**North star:** do not show the user an answer the supplied evidence cannot support. Prefer abstention over a false release.

This document is the system design, not a tutorial. For install and API snippets, see [README.md](README.md).

---

## 1. Product thesis

The framework is a **post-generation gate**, not a retriever and not a generator.

The calling application already has a query, a model answer, and whatever evidence it retrieved (chunks, KB text, PDFs, OCR, tables, transcripts). This package decides whether that answer may be shown, rewritten to only the grounded fragments, or withheld.

Three consequences follow and are non-negotiable:

1. **We check support, not world truth.** Bad retrieval still produces bad (or empty) answers. The gate cannot invent missing facts.
2. **False-release is the primary SLO.** Over-refusal is an accepted cost. Similarity, topical overlap, or a high “quality” score is never enough to release a claim.
3. **The lock is claim-level verification.** The Bayesian network is a **fusion / diagnostic layer** used for eval scores, gap reports, and optional veto. Production `safe_answer` is decided by claim status, not by `P(groundedness)`.

If a design choice conflicts with (2), the design is wrong for this product.

---

## 2. What this system is not

| Not this | Why |
|---|---|
| A RAGAS clone that averages a judge score | We score **atomic claims against individual chunks** (soft-OR), then gate. |
| A vector database or retrieval stack | Callers retrieve; we consume ranked text. |
| An LLM wrapper / agent runtime | We do not generate. `wrap` / `protect` / `run` decorate *your* generate function. |
| A calibrated probability of hallucination | BN posteriors are discrete fusion scores (`P(high) + 0.5·P(medium)`), not calibrated P(truth). |
| A world-knowledge fact checker | Evidence is caller-supplied. No web search, no parametric “is this true?” |
| A heuristic CI score as a production gate | `quality_mode="ci"` / `RAG_EVAL_HEURISTIC=1` is smoke only. |

---

## 3. Design principles

1. **Firewall first.** `pass` / `rewrite` / `abstain`. User-facing surfaces return `safe_answer` only.
2. **Claim atomicity.** Answers are split into claims; each claim is grounded independently; the gate recomposes or withholds.
3. **Soft-OR against neighbors.** A chunk with unrelated numbers must not veto a claim another chunk fully supports. Contradiction only wins from *aligned* chunks, and only when no chunk supports the claim.
4. **Coverage over cosine.** `fused_support = 0.55·entailment + 0.45·(0.65·coverage + 0.35·similarity)`. Invented entities (`extra_distinctive ≥ 2`) cannot ride a copied prefix. Numeric / quoted-literal mismatch forces contradiction.
5. **Dataset-agnostic adapters.** LangChain `Document`, LlamaIndex nodes, dicts, tuples, raw strings — all flatten to `list[str]` via `normalize_context`. No domain lexicon.
6. **Two quality modes, two policies.** Mode selects backends (heuristic vs neural). Policy selects thresholds. They are orthogonal.
7. **Fail closed on contradiction.** Any contradicted claim → abstain. Partial support → rewrite iff the remaining text still answers the query.
8. **Inference is not extractive support.** Multi-hop may *tag* `grounding_kind=inferred`; strict policy will not treat it as a release. Low-reliability sources cannot be the sole SUPPORTED citation.

---

## 4. System context

```
┌─────────────────────────────────────────────────────────────────┐
│  Your app: retriever + generator (LangChain, LlamaIndex,        │
│  custom, fine-tuned endpoint — irrelevant to this package)      │
└────────────┬──────────────────────────────┬─────────────────────┘
             │ query, answer, evidence      │ labeled samples
             ▼                              ▼
┌────────────────────────┐     ┌──────────────────────────────────┐
│  Production gate       │     │  RAG quality eval (CI / nightly) │
│  HallucinationGate     │     │  RAGEval + latency + regression  │
│  /v1/answer            │     │  hallucination-gate eval-dataset │
└────────────┬───────────┘     └────────────────┬─────────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
              DiagnosticEvaluator  (engine)
                            │
                            ▼
              GatedAnswer | EvalReport | SafeAnswerResponse
```

**Surfaces (same engine):**

| Surface | Entry | Typical consumer |
|---|---|---|
| Python SDK | `from hallucination_gate import HallucinationGate, RAGEval` | App code, notebooks |
| CLI | `hallucination-gate` (`evaluate`, `eval-dataset`, `eval-heldout`, `eval-adversarial`, `eval-benchmark`, `calibrate`) | Local debug, CI |
| HTTP | FastAPI `api/main.py` — `/v1/answer`, `/evaluate`, `/evaluate/batch` | Sidecar / service |

Public package name is `hallucination-gate`. Internal engine package is `bayesian_rag_evaluator`. The thin public façade is `hallucinate_gate` (re-exported as `hallucination_gate`). Engine modules **must not** import the façade — that cycle is intentional and load-bearing.

---

## 5. Package map

```
src/
  hallucination_gate/          preferred public import alias
  hallucinate_gate/            public SDK
    gate.py                    HallucinationGate, GatedAnswer
    evidence.py                Evidence / ImageEvidence / TableEvidence
    eval.py                    RAGEval re-export
  bayesian_rag_evaluator/      engine (not the public brand)
    evaluator.py               DiagnosticEvaluator — the pipeline
    quality.py                 ci|quality|quality_plus  ×  strict|balanced
    adapters.py                normalize_context
    cli.py                     Typer app
    api/main.py                FastAPI sidecar
    models/schemas.py          Pydantic contract (ClaimVerdict + GroundingKind)
    claims/                    THE LOCK
      extractor.py             structured decompose (reason / conjunction)
      policy.py                decide_status
      verifier.py              top-k + soft-OR + reliability + multi-hop
      logic.py                 temporal / negation / scope
      multihop.py              two-chunk inferred support
      reliability.py           source_reliability stamp
      fusion.py                calibrated fused_support
    evidence/                  backends, ingest, OCR, align, scorers, multimodal
    bn/                        Discrete BN: structure, discretize, VE inference, CPT learn
    gate/engine.py             pass / rewrite / abstain
    diagnostics/engine.py      gaps, suggestions, eval verdict
    metrics/                   RAGAS-class + retrieval + latency + regression + benchmark
    data_gen/adversarial.py    false-release suite
    judge/                     optional LLM escalation for UNCERTAIN only
    config/bn_structure.yaml   graph + CPTs + suggestion copy
    config/thresholds.yaml     bins + gate + verdict floors
    config/fusion.yaml         fusion weights + calibration curve
```

Config YAMLs ship in the wheel (`package-data`). Paths resolve via `config_paths.config_file`, not CWD.

---

## 6. End-to-end request path

This is the only production path. Eval reuses it sample-by-sample.

```
HallucinationGate.check(query, answer, context|kb|Evidence)
        │
        ▼
  adapters.normalize_context  →  list[str]
  Evidence → EvaluateRequest (Pydantic)
        │
        ▼
  DiagnosticEvaluator.evaluate
        │
        ├─ 1. EvidenceExtractor.store_from_request
        │      enrich_image (OCR) · load_pdfs (pypdf, OCR fallback)
        │      align_contexts (query/answer overlap; policy.max_aligned_chunks)
        │      build_evidence_store → EvidenceUnit[]  (text, image, table, doc, audio)
        │      apply_source_reliability (optional caller map)
        │
        ├─ 2. extract_claims(answer)
        │      sentence / bullet / CJK / semicolon / because|therefore
        │      conjunction split · hedge drop · StructuredClaim facets
        │
        ├─ 3. verify_claims  ★ THE LOCK
        │      embed similarity_matrix(claims, units) · top-k=8
        │      NLI + numbers/literals + extra_distinctive + logic_mismatches
        │      calibrated fused_support · decide_status per chunk
        │      soft-OR aggregate → ClaimResult
        │      reliability floor: low-trust chunk cannot be sole SUPPORTED
        │      multi-hop if still unsupported/uncertain → grounding_kind=inferred
        │      optional refine_uncertain_claims (LLM judge, off by default)
        │
        ├─ 4. extract() → EvidenceScores
        │      query_relevance, context_faithfulness, entailment,
        │      retrieval_quality, completeness, contradiction,
        │      unsupported_claims, visual_grounding, numeric_consistency
        │
        ├─ 5. discretize_evidence → {low, medium, high} per node
        │
        ├─ 6. BayesianInferenceEngine.infer  (Variable Elimination)
        │      latents: groundedness, hallucination_risk, answer_quality,
        │               retrieval_adequacy, release_safety
        │
        ├─ 7. apply_gate  → GateResult {pass | rewrite | abstain}
        │      BN posterior veto is OFF by default (use_bn_veto=False)
        │
        └─ 8. identify_gaps · generate_suggestions · compute_verdict
               EvaluateResponse (diagnostics + safe_answer + request_id + latency_ms)
        │
        ▼
  GatedAnswer.text  =  response.safe_answer   // what users may see
```

Latency is wall-clock around steps 1–8 (`Timer`). Cold-start of sentence-transformers / NLI dominates unless `warm=True`.

---

## 7. The lock — claim verification

### 7.1 Extraction

`claims/extractor.py` splits on sentence boundaries (including CJK), bullets, semicolons, reason clauses (`because` / `therefore` / …), then long-clause conjunctions (`and` / `but` / `while` / …). Short hedges (`I think…`) are dropped. Empty split falls back to the full answer so short replies are never skipped. `extract_structured_claims` also records numbers, negation, temporal, and scope facets.

### 7.2 Per-chunk decision (`claims/policy.py`)

For each (claim, top-k chunk):

| Signal | Role |
|---|---|
| Token coverage (+ synonyms) | Must the evidence actually contain the claim’s content words? |
| Embedding similarity | Topical alignment only — **never sufficient alone** |
| NLI entailment / contradiction | Neural or heuristic |
| `numbers_agree` | Claim numbers must appear in the chunk; clash → contradiction ≥ 0.78 |
| `literals_agree` | Quoted / code-like literals must appear verbatim |
| `extra_distinctive_tokens` | ≥2 content words absent from evidence → cannot be SUPPORTED |

Status order inside `decide_status`:

1. Numeric / literal mismatch → treat as contradicted.
2. If `contradiction ≥ threshold` and `contradiction ≥ support` → `CONTRADICTED`.
3. Strong NLI (`entail ≥ min_support_entail` and `coverage ≥ 0.50`) **or** strong lexical (coverage + sim + entail, zero extra distinctive) → `SUPPORTED`.
4. Extra distinctive ≥ 2 → `UNSUPPORTED`.
5. Else if support/coverage in the uncertain band → `UNCERTAIN`.
6. Else `UNSUPPORTED`.

### 7.3 Soft-OR aggregation (`claims/verifier.py`)

Across the claim’s chunk hits:

1. Any `SUPPORTED` hit → claim is **supported** (best support, then coverage, then similarity for citation).
2. Else any `CONTRADICTED` hit that is **aligned** (`coverage ≥ 0.35` or `similarity ≥ 0.40`) → **contradicted**.
3. Else any `UNCERTAIN` → **uncertain**.
4. Else **unsupported**.

This is the false-release lock. Neighbor-chunk number clashes cannot sink a claim that another chunk copies faithfully.

After aggregation:

- If the winning `SUPPORTED` citation has `reliability < min_support_reliability` → downgrade to `UNCERTAIN` (`grounding_kind=extractive`).
- If still `UNSUPPORTED` or `UNCERTAIN`, `try_multihop` may jointly score two chunks that each contribute unique claim tokens. Success sets `grounding_kind=inferred`. **strict** keeps status `UNCERTAIN` (not a release). **balanced** may mark `SUPPORTED` when `allow_inferred_release=True`.
- `grounding_kind` is diagnostic of *how* the claim relates to evidence; gate action still follows `ClaimVerdict`.

### 7.4 Optional LLM judge

`judge/` is **off**. Enable with `HALLUCINATION_GATE_JUDGE=1` plus Anthropic / OpenAI / custom URL. It may re-label **UNCERTAIN** claims only, using the top-3 aligned citations — never the full evidence bag. Failure → leave UNCERTAIN. This path must not become the lock.

---

## 8. The gate — pass / rewrite / abstain

`gate/engine.py` `apply_gate`:

```
contradicted claims?          → ABSTAIN  (always)
no supported claims?
    balanced + uncertain-only
    + high overlap + answers query
    + not BN-unsafe            → REWRITE from usable UNCERTAIN claims
    else                       → ABSTAIN
supported + leftover junk?    → REWRITE (compose supported only)
    rewritten text must still answer the query
    (completeness ≥ floor OR Jaccard ≥ floor)
    else                       → ABSTAIN
all claims supported          → PASS  (original answer)
```

`strict=False` relaxes posterior floors (if BN veto is on) and rewrite completeness. It does **not** forgive contradiction.

**BN veto is disabled by default.** Claim status is sufficient. Posteriors still flow into eval metrics (`groundedness`, `hallucination_risk`, `release_safety`) and the diagnostic `verdict` (`pass` / `needs_improvement` / `fail`). Abstain forces eval verdict `fail`; rewrite forces `needs_improvement`.

Abstain copy is a fixed safe string — never the raw model text.

Rewrite composition joins remaining claims as prose. Citations are opt-in (`cite_sources`).

### Policies (`quality.py`)

| | `strict` | `balanced` (SDK default) |
|---|---|---|
| Support coverage / entail | 0.72 / 0.58 | 0.64 / 0.52 |
| Uncertain rewrite | no | yes |
| Max aligned chunks | all | 5 |
| Contradiction bar | 0.55 | 0.55 (unchanged — false-release lock) |

Env: `HALLUCINATION_GATE_POLICY`, `HALLUCINATION_GATE_MODE`, `RAG_EVAL_HEURISTIC`.

---

## 9. Bayesian network — fusion, not the lock

Structure: `config/bn_structure.yaml`. Implementation: pgmpy `DiscreteBayesianNetwork` + `VariableElimination`.

**Evidence nodes** (observed after discretization):
`query_relevance`, `context_faithfulness`, `entailment_score`, `retrieval_quality`, `completeness`, `contradiction`, `unsupported_claims`, `visual_grounding`, `numeric_consistency`, `model_type`.

**Latent nodes:**
`groundedness` ← faithfulness, entailment, visual;
`hallucination_risk` ← unsupported, contradiction, numeric (inverted);
`answer_quality` ← relevance, completeness, groundedness, hallucination_risk (inverted);
`retrieval_adequacy` ← retrieval_quality, model_type;
`release_safety` ← groundedness, hallucination_risk.

CPTs for high-arity latents use **composite groups**: parent states are summed (optionally inverted) into a 0…2N index, then a 3-way distribution is looked up. This keeps YAML tractable versus enumerating 3^k tables.

Posterior reported to the API is **not** `P(state=high)`. It is:

```
score = P(high) + 0.5 · P(medium)
```

Treat it as a fused diagnostic, not a calibrated probability. Thresholds in `thresholds.yaml` (`verdict.*`, `gate.*`) are operating points on that score.

**Calibration path (offline):** `hallucination-gate calibrate` learns CPTs via `BayesianEstimator` from labeled JSON/JSONL (or synthetic gold) and can dump `learned_bn.pkl`. `DiagnosticEvaluator(learned_model_path=…)` loads it. Default install uses the YAML priors.

---

## 10. Evidence plane

All modalities collapse to `EvidenceUnit{content, modality, source_id}`:

| Input | Source id | Notes |
|---|---|---|
| `context_chunks` | `context:{i}` | Optionally aligned/filtered to query+answer |
| `kb_chunks` | `kb:{i}` | Fine-tuned path when retrieval is empty |
| documents / PDFs | `document:{i}` | pypdf; empty text → page OCR |
| images | caption + OCR + alt | Optional CLIP cosine if path present |
| tables | flattened `header: value` | Numeric consistency uses extracted numbers |
| audio | transcripts only | No ASR in-process |

**Backends** (`evidence/backends.py`):

| Mode | Embed | NLI |
|---|---|---|
| `ci` / heuristic | Jaccard / token coverage | Overlap + negation + number clash |
| `quality` / neural | `paraphrase-multilingual-MiniLM-L12-v2` | `cross-encoder/nli-deberta-v3-small` |

Caches: in-process `EMBED_CACHE` / `NLI_CACHE`. `HallucinationGate(warm=True)` preloads both to cut cold-start tails.

Alignment (`evidence/align.py`) is generic lexical+embedding rerank. Balanced policy keeps top 5 above `min_score=0.12`, but never empties the bag.

---

## 11. Eval plane (RAGAS replacement path)

`RAGEval` runs the **same** `DiagnosticEvaluator` per sample, then maps `EvaluateResponse` into:

| Metric | Definition here |
|---|---|
| faithfulness | supported/n_claims; any contradiction penalizes |
| answer_relevancy | query↔answer embedding (`query_relevance`) |
| context_precision_labeled | vs `relevant_contexts` / `relevant_indices` |
| context_precision_aligned | claim-aligned chunk proxy when unlabeled |
| context_recall | requires `ground_truth` |
| groundedness / hallucination_risk / release_safety | BN fusion scores |
| hit@k, recall@k, MRR, nDCG@k | ranked `contexts` vs labels |
| latency p50/p95/p99/max | vs `LatencyBudget` |
| regression | delta vs saved baseline JSON; fail CI on drops |

Default regression floors: quality metrics −0.03, retrieval −0.02, `hallucination_risk` +0.03, latency +max(50ms, 20% of baseline).

Held-out safety eval: `hallucination-gate eval-heldout` — false-release vs over-refusal on a fixed domain set (`data_gen/heldout.py`). This is the metric that matters for a firewall.

Gold generation (`data_gen/gold.py`) exists to stress CPT learning and gate actions, not as a substitute for your labeled production traffic.

---

## 12. HTTP sidecar

`bayesian_rag_evaluator.api.main:app`

| Route | Auth | Returns |
|---|---|---|
| `GET /health` | none | status, backend, `release_authority=claim_status`, `scores_are_calibrated=false` |
| `GET /metrics` | none | Prometheus text (process-local; tenant labels if keys are set) |
| `GET /metrics.json` | none | JSON snapshot of the same registry |
| `POST /v1/answer` | `X-API-Key` / `Authorization: Bearer` if keys set | `SafeAnswerResponse` only — **never raw model text**; includes `evidence_gap` |
| `POST /evaluate` | same | full `EvaluateResponse` (internal) |
| `POST /evaluate/batch` | same | list of full responses |

Auth: `RAG_EVAL_API_KEY` (single key, tenant `RAG_EVAL_TENANT` or `default`) and/or `RAG_EVAL_API_KEYS=tenant:secret,tenant2:secret2`. `X-Tenant-Id` must match the key if sent. **Same process, shared models.** Tenant IDs are metric/log labels, not data isolation.

Hardening already in tree:

- Thread pool (`RAG_EVAL_WORKERS`, default 8) + timeout (`RAG_EVAL_TIMEOUT_SEC`, default 12) → 504
- Access log + `x-request-id` / `x-latency-ms`
- Lazy singleton evaluator (first request pays model load unless you warm at boot)

This is a **sidecar**, not a multi-tenant platform. Scrape Prometheus `/metrics`; do not expect per-tenant model isolation or durable audit storage.

---

## 13. Integration patterns

```python
gate = HallucinationGate(quality_mode="quality", policy="balanced", warm=True)

# 1. You already generated
result = gate.check(query, answer, context=docs)   # result.text

# 2. You still generate; we retrieve-optional then gate
result = gate.run(query, generate=my_llm, retrieve=my_retriever)

# 3. Drop-in wrapper
safe_generate = gate.wrap(my_llm, retrieve=my_retriever, text_only=True)

# 4. Decorator — fn may return str | (answer, ctx) | dict
@gate.protect
def rag(query: str): ...
```

`GatedAnswer` is the contract: `text`, `released`, `action`, `reason`, `claims`, `evidence_gap`, `release_authority="claim_status"`, `scores_are_calibrated=False`, `diagnostics` (claim↔chunk view). BN fusion scores live on `raw` only when `debug=True` — they are not a release probability.

---

## 14. Operating model

### Modes

| | CI smoke | Production | Harder NLI (opt-in) |
|---|---|---|---|
| Flag | `quality_mode="ci"` or `RAG_EVAL_HEURISTIC=1` | `quality_mode="quality"` | `quality_mode="quality_plus"` |
| Backends | heuristic overlap | MiniLM + DeBERTa-v3-small | mpnet + DeBERTa-v3-base |
| Use | pytest, `eval-dataset --heuristic` | live `check` / `/v1/answer` | English math/code-ish RAG |
| Honest limit | not a faithfulness gate | GPU/CPU + download cost | still not a hard-reasoning judge |

CI (`.github/workflows/ci.yml`): Python 3.11/3.12, `pytest -m "not neural"` required; neural extra is `continue-on-error`.

### Recommended rollout

1. **Offline:** `eval-dataset` on labeled gold (`relevant_contexts`, `ground_truth`). Save baseline.
2. **CI:** heuristic smoke + regression vs baseline + latency budget. Separate nightly neural job.
3. **Shadow:** `check(..., debug=True)` in prod, log `action` / `released` / `reason`, still serve the ungated answer internally.
4. **Enforce:** switch user-visible path to `result.text` / `/v1/answer`. Start `balanced`, tighten to `strict` if false-releases appear.
5. **Warm** workers at boot. Set `RAG_EVAL_TIMEOUT_SEC` from observed p99, not from hope.

### What to monitor

- Gate mix: `pass` / `rewrite` / `abstain` (registry `gate_actions`)
- `evidence_gap`: `retrieval` vs `generation` vs `contradiction` (fix retriever vs generator)
- False-release rate on a labeled canary (must stay ~0) — published in [docs/EVAL.md](docs/EVAL.md)
- Over-refusal rate (product cost — tune policy, not the contradiction bar)
- p95/p99 of `latency_ms`; timeout count
- Auth failures if the sidecar is exposed
- Per-tenant request counters when `RAG_EVAL_API_KEYS` is set

---

## 15. Threat model (honest)

| Risk | Mitigation in tree | Residual |
|---|---|---|
| Fluent but ungrounded answer | Claim coverage + extra-entity penalty + contradiction abstain | Subtle reasoning can still fool NLI or over-block |
| Math / code literals | All claim numbers must appear; equation clash; identifier extras | No CAS / AST; `quality_plus` is stronger NLI, not a prover |
| Neighbor-chunk contamination | Soft-OR; aligned-only contradiction | Alignment thresholds are heuristic |
| Cosine “this is on topic” | Coverage required for SUPPORT | Heuristic mode is weaker |
| Caller dumps entire corpus as context | `align_contexts` + `max_aligned_chunks` | Alignment can drop a needed chunk (bag never fully emptied) |
| OCR / PDF garbage | OCR confidence in caption; empty PDF → OCR pages | Garbage in, abstain or false uncertain |
| `/evaluate` leaking raw answer | `/v1/answer` strips it; API key on internals | Misconfigured public `/evaluate` |
| Treating BN score as P(hallucination) | API/SDK fields `scores_are_calibrated=false`, `release_authority=claim_status`; veto off | Dashboards can still be misread if you plot `scores.*` as probabilities |
| Bad retrieval looking like hallucination | `evidence_gap=retrieval` + retrieval abstain copy | Gate cannot invent missing chunks |
| LLM judge weakening the lock | Off; UNCERTAIN-only; fail-closed | A custom judge URL can still promote junk if you turn it on |
| Model download / Windows symlink | Documented ops surface | First-request latency, HF cache |

We do **not** currently: isolate tenant data or models, stream, batch-NLI on GPU as a service, or persist audit logs. Per-key tenant labels on Prometheus are sidecar ops, not a platform.

---

## 16. Decision log

| Decision | Choice | Why |
|---|---|---|
| Gate vs generate | Gate only | Stay framework-agnostic; one lock for every vendor |
| Lock vs BN | Claims lock, BN diagnose | Discrete BN is the wrong granularity for “this sentence is invented” |
| Soft-OR vs max-contradiction-across-k | Soft-OR | Unrelated neighbors were causing false abstains |
| Similarity never sufficient | Coverage + entailment | False-release lock |
| Heuristic ≠ quality | Explicit modes | CI must not pretend to be calibrated faithfulness |
| Public HTTP returns `safe_answer` only | `/v1/answer` | User-facing clients must not see the raw hallucination |
| Balanced default in SDK | fewer over-refusals | Strict remains one flag away; contradiction bar unchanged |
| Optional judge | UNCERTAIN band only | Do not let a chat model become the firewall |
| Inferred ≠ extractive | Tag multi-hop; strict does not release it | Composition is weaker than a copied chunk |
| Reliability floor | Downgrade sole low-trust citations | Caller can mark scraped/untrusted sources |
| BN never ships `safe_answer` | `release_authority=claim_status` on every response | Buyers must not hear “Bayesian = calibrated risk” |
| Label retrieval failure | `evidence_gap` | Operators can fix the retriever instead of blaming the generator |
| Prometheus + per-key tenants | `/metrics` text; `RAG_EVAL_API_KEYS` | Sidecar ops, not SaaS isolation |

---

## 17. Reasoning upgrades (claim lock)

These sit **on the lock**, not in the BN.

| Capability | Module | Release rule |
|---|---|---|
| Structured decomposition | `claims/extractor.py` | Reason/conjunction splits so packed hallucinations can be dropped |
| Inference vs unsupported | `ClaimResult.grounding_kind` | `inferred` ≠ `unsupported`; strict still will not release inferred |
| Multi-hop | `claims/multihop.py` | Two hops must each contribute unique claim tokens; union still needs coverage |
| Temporal / negation / scope | `claims/logic.py` | Clear mismatches raise contradiction; weak topical chunks ignored |
| Source reliability | `source_reliability` on the request | Below `min_support_reliability` → UNCERTAIN, not SUPPORTED |
| Calibrated fusion | `claims/fusion.py` + `config/fusion.yaml` | Same weights as before; optional piecewise calibration (never raises scores above evidence) |
| Adversarial + competitor bench | `eval-adversarial` / `eval-benchmark` | Gate vs overlap-only and cosine-only; false-release must stay 0 |
| Math / code lock | `claims/special.py` | Equations and identifiers can only tighten; never raise support |
| Retrieval vs generation | `GateResult.evidence_gap` | Weak retrieval is labeled, not silently called a hallucination |

---

## 18. Ownership boundaries

| Layer | Owns | Does not own |
|---|---|---|
| Caller | Retrieval quality, generation, UX copy on abstain (optional override), PII | Grounding policy |
| SDK / sidecar | Claim lock, gate action, eval report, timeouts | Vector index, LLM weights (except embed/NLI) |
| Embed / NLI models | Similarity and entailment signals | Release decision |
| BN | Fused diagnostics, suggestions | `safe_answer` unless `use_bn_veto` is explicitly enabled |

---

## 19. Published safety numbers

Heuristic CI backends on the in-repo held-out + adversarial suites. Re-run:

```bash
set RAG_EVAL_HEURISTIC=1
hallucination-gate eval-heldout
hallucination-gate eval-adversarial
hallucination-gate eval-benchmark
```

See [docs/EVAL.md](docs/EVAL.md) for the snapshot. False-release is the SLO (~0). Over-refusal is published, not hidden. Neural `quality` / `quality_plus` numbers belong in *your* customer set — this package does not claim them from MiniLM smoke tests.

---

*Shreyas G — hallucination-gate v0.9.2. Sole contributor. If a change makes it easier to release an unsupported claim, it is a regression regardless of RAGAS-like averages.*
