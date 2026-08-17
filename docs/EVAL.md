# Published gate numbers (v0.9.2)

**SLO:** false-release ≈ 0. Over-refusal is an accepted cost and is published here.

These numbers are from the **in-repo heuristic (`ci`) backends** on held-out + adversarial suites the thresholds were not fitted on. They are not MiniLM/DeBERTa production numbers, and they are not a customer corpus.

Re-run:

```bash
set RAG_EVAL_HEURISTIC=1
hallucination-gate eval-heldout
hallucination-gate eval-adversarial
hallucination-gate eval-benchmark
```

## Held-out domains (`eval-heldout`)

Policy: `strict`. Alignment off. n = 27 (pass / rewrite / abstain mix, including math, code, and off-topic retrieval).

| Metric | Value |
|---|---|
| False-release rate | **0.000** |
| Over-refusal rate (`false_abstain_rate`) | **0.083** (1 of 12 should-release) |
| Precision (release) | 1.000 |
| Recall (release) | 0.917 |
| Action accuracy | 0.963 |

Confusion (gold → pred): pass 10/11 (1 abstain); rewrite 1/1; abstain 15/15.

## Adversarial (`eval-adversarial`)

n = 11 (negation, numbers, entities, scope, time, reliability, math, code, off-topic retrieval, copy-paste control).

| Metric | Value |
|---|---|
| False-release rate | **0.000** |
| Over-refusal rate | **0.000** |
| Accuracy | 1.000 |

## Vs naive baselines (`eval-benchmark`)

Same 11 adversarial cases.

| Competitor | False-release | Over-refusal | Accuracy |
|---|---|---|---|
| **hallucination-gate** | **0.000** | 0.000 | 1.000 |
| Overlap-only (coverage ≥ 0.50) | 0.455 | 0.000 | 0.545 |
| Cosine-only (sim ≥ 0.50) | 0.182 | 0.000 | 0.818 |

Overlap and cosine “release” fluent junk that shares tokens with evidence. The gate does not.

## What this does *not* claim

- Not a calibrated P(hallucination). BN fusion scores are diagnostic; `release_authority=claim_status`.
- Not neural `quality` / `quality_plus` latency or accuracy. Run those on *your* labeled set.
- Not a customer production mix. If you need a buy-off number, score `eval-heldout`-style labels on your traffic and keep false-release at 0.
