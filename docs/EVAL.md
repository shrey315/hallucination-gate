# Published gate numbers (v0.9.4)

**SLO:** false-release ≈ 0 on the heuristic lock. Over-refusal is an accepted cost.

Two backends are published:

| Backend | How to run |
|---|---|
| **heuristic (`ci`)** | `RAG_EVAL_HEURISTIC=1` (CI smoke) |
| **neural (`quality`)** | MiniLM + DeBERTa-small: `hallucination-gate eval-heldout --neural` |

The **in-repo corpus** (`hallucination-gate eval-corpus` with no path) is packaged FAQ-style labels (held-out + adversarial + extra HR/IT traces). It is **not a customer production mix**. Point `eval-corpus` at your own JSONL `{query, answer, contexts, expected_release}` for buy-off.

```bash
set RAG_EVAL_HEURISTIC=1
hallucination-gate eval-heldout
hallucination-gate eval-adversarial
hallucination-gate eval-corpus
hallucination-gate eval-benchmark

hallucination-gate eval-heldout --neural
hallucination-gate eval-adversarial --neural
hallucination-gate eval-corpus path/to/your_db.jsonl
```

## Held-out domains (`eval-heldout`)

Policy: `strict`. Alignment off. n = 37 (pass / rewrite / abstain, including fluent tails, composed 2–3 hop, off-topic retrieval).

| Metric | Heuristic | Neural (`quality`) |
|---|---|---|
| False-release rate | **0.000** | 0.050 (1 of 20 should-block) |
| Over-refusal rate | **0.059** (1 of 17) | 0.176 (3 of 17) |
| Precision (release) | 1.000 | 0.933 |
| Recall (release) | 0.941 | 0.824 |
| Action accuracy | 0.973 | 0.892 |

Heuristic confusion (gold → pred): pass 14/15 (1 abstain); rewrite 2/2; abstain 20/20.

## Adversarial (`eval-adversarial`)

n = 17 (negation, numbers, entities, scope, time, reliability, math, code, retrieval poison, fluent continuation, composed 2–3 hop, copy-paste control).

| Metric | Heuristic | Neural (`quality`) |
|---|---|---|
| False-release rate | **0.000** | **0.000** |
| Over-refusal rate | **0.000** | **0.000** |
| Accuracy | 1.000 | 1.000 |

## In-repo corpus (`eval-corpus`)

n = 52 packaged labels. Heuristic:

| Metric | Value |
|---|---|
| False-release rate | **0.000** |
| Over-refusal rate | **0.048** (1 of 21) |
| Precision (release) | 1.000 |
| Recall (release) | 0.952 |
| Action accuracy | 0.981 |

## Vs naive baselines (`eval-benchmark`)

Same 17 adversarial cases, heuristic.

| Competitor | False-release | Over-refusal | Accuracy |
|---|---|---|---|
| **hallucination-gate** | **0.000** | 0.000 | 1.000 |
| Overlap-only (coverage ≥ 0.50) | 0.529 | 0.000 | 0.471 |
| Cosine-only (sim ≥ 0.50) | 0.353 | 0.118 | 0.529 |

## What this does *not* claim

- Not a calibrated P(hallucination). BN fusion scores are diagnostic; `release_authority=claim_status`.
- Not `quality_plus` (mpnet + DeBERTa-base). Run `--mode quality_plus` on your set if you need that stack.
- Not your production mix. Keep false-release at 0 on `eval-corpus your_db.jsonl`.
