"""Calibrated claim-support fusion. Similarity cannot override missing coverage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from bayesian_rag_evaluator.config_paths import config_file

DEFAULT_ENTAIL = 0.55
DEFAULT_LEXICAL = 0.45
DEFAULT_COVERAGE = 0.65
DEFAULT_SIMILARITY = 0.35


@dataclass(frozen=True)
class FusionConfig:
    entailment: float = DEFAULT_ENTAIL
    lexical: float = DEFAULT_LEXICAL
    coverage: float = DEFAULT_COVERAGE
    similarity: float = DEFAULT_SIMILARITY
    calibration: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))


_CACHED: FusionConfig | None = None


def load_fusion_config(path: Path | None = None) -> FusionConfig:
    global _CACHED
    if path is None and _CACHED is not None:
        return _CACHED
    cfg_path = path or config_file("fusion.yaml")
    entail, lexical, coverage, similarity = (
        DEFAULT_ENTAIL,
        DEFAULT_LEXICAL,
        DEFAULT_COVERAGE,
        DEFAULT_SIMILARITY,
    )
    curve: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
    if cfg_path.exists():
        import yaml

        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        weights = raw.get("weights") or {}
        entail = float(weights.get("entailment", entail))
        lexical = float(weights.get("lexical", lexical))
        coverage = float(weights.get("coverage", coverage))
        similarity = float(weights.get("similarity", similarity))
        points = raw.get("calibration") or curve
        curve = [(float(a), float(b)) for a, b in points]
    env = os.getenv("HALLUCINATION_GATE_FUSION")
    if env:
        extra = json.loads(env)
        entail = float(extra.get("entailment", entail))
        lexical = float(extra.get("lexical", lexical))
        coverage = float(extra.get("coverage", coverage))
        similarity = float(extra.get("similarity", similarity))
        if extra.get("calibration"):
            curve = [(float(a), float(b)) for a, b in extra["calibration"]]
    cfg = FusionConfig(
        entailment=entail,
        lexical=lexical,
        coverage=coverage,
        similarity=similarity,
        calibration=tuple(curve) if curve else ((0.0, 0.0), (1.0, 1.0)),
    )
    if path is None:
        _CACHED = cfg
    return cfg


def fused_support(
    entailment: float,
    similarity: float,
    coverage: float,
    config: FusionConfig | None = None,
) -> float:
    """Weighted support, then piecewise calibration. Coverage still dominates lexical."""
    cfg = config or load_fusion_config()
    lexical = cfg.coverage * coverage + cfg.similarity * similarity
    raw = max(0.0, min(1.0, cfg.entailment * entailment + cfg.lexical * lexical))
    return calibrate(raw, cfg.calibration)


def calibrate(value: float, curve: Sequence[tuple[float, float]]) -> float:
    """Piecewise-linear map. Identity if the curve is empty or endpoints-only 0→0, 1→1."""
    if not curve:
        return max(0.0, min(1.0, value))
    pts = sorted((float(x), float(y)) for x, y in curve)
    x = max(0.0, min(1.0, float(value)))
    if x <= pts[0][0]:
        return max(0.0, min(1.0, pts[0][1]))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            if abs(x1 - x0) < 1e-12:
                return max(0.0, min(1.0, y1))
            t = (x - x0) / (x1 - x0)
            return max(0.0, min(1.0, y0 + t * (y1 - y0)))
    return max(0.0, min(1.0, pts[-1][1]))


def learn_fusion_calibration(
    pairs: Sequence[tuple[float, bool]],
    bins: int = 5,
) -> list[tuple[float, float]]:
    """Isotonic-style bin means from (raw_score, was_truly_supported) pairs.

    Conservative: a bin's calibrated value cannot exceed the empirical support
    rate, so over-confident scores get pulled down.
    """
    if len(pairs) < 8:
        return [(0.0, 0.0), (1.0, 1.0)]
    edges = [i / bins for i in range(bins + 1)]
    buckets: list[list[float]] = [[] for _ in range(bins)]
    for score, label in pairs:
        idx = min(bins - 1, int(float(score) * bins))
        buckets[idx].append(1.0 if label else 0.0)
    curve: list[tuple[float, float]] = [(0.0, 0.0)]
    running_max = 0.0
    for i, bucket in enumerate(buckets):
        x = (edges[i] + edges[i + 1]) / 2.0
        if bucket:
            rate = sum(bucket) / len(bucket)
            running_max = max(running_max, rate)
            curve.append((x, running_max))
        else:
            curve.append((x, running_max))
    curve.append((1.0, max(running_max, curve[-1][1])))
    return curve


def save_fusion_config(config: FusionConfig, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "weights": {
            "entailment": config.entailment,
            "lexical": config.lexical,
            "coverage": config.coverage,
            "similarity": config.similarity,
        },
        "calibration": [list(p) for p in config.calibration],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    global _CACHED
    _CACHED = None
