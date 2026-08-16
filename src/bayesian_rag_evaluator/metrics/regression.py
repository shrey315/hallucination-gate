"""Regression baselines: save an eval run, diff the next, fail CI on drops."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Metric → maximum allowed *drop* (current - baseline). Negative delta beyond this fails.
# For risk-like metrics, use max allowed *increase* via RISK_METRICS.
DEFAULT_MIN_DELTAS: dict[str, float] = {
    "faithfulness": -0.03,
    "answer_relevancy": -0.03,
    "context_precision": -0.03,
    "context_recall": -0.03,
    "groundedness": -0.03,
    "release_safety": -0.03,
    "hit_at_5": -0.02,
    "hit_at_10": -0.02,
    "mrr": -0.02,
    "ndcg_at_10": -0.02,
    "release_rate": -0.05,
}

RISK_METRICS = {
    "hallucination_risk": 0.03,  # fail if increases by more than this
}


@dataclass
class RegressionResult:
    passed: bool
    baseline_path: str | None
    deltas: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    compared_keys: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def baseline_payload_from_report(report: Any) -> dict[str, Any]:
    """Compact baseline: aggregates + latency + retrieval only."""
    data = report.as_dict() if hasattr(report, "as_dict") else dict(report)
    return {
        "n": data.get("n"),
        "aggregate": dict(data.get("aggregate") or {}),
        "retrieval": dict(data.get("retrieval") or {}),
        "latency": {
            k: (data.get("latency") or {}).get(k)
            for k in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms", "n")
            if (data.get("latency") or {}).get(k) is not None
        },
        "framework": data.get("framework"),
        "metrics": data.get("metrics"),
    }


def save_baseline(report: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = baseline_payload_from_report(report)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flat_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for section in ("aggregate", "retrieval"):
        block = payload.get(section) or {}
        for key, value in block.items():
            if isinstance(value, (int, float)):
                out[key] = float(value)
    latency = payload.get("latency") or {}
    for key in ("p50_ms", "p95_ms", "p99_ms", "max_ms", "mean_ms"):
        if key in latency and isinstance(latency[key], (int, float)):
            out[f"latency_{key}"] = float(latency[key])
    return out


def compare_to_baseline(
    report: Any,
    baseline: Mapping[str, Any] | str | Path,
    *,
    min_deltas: Mapping[str, float] | None = None,
    risk_limits: Mapping[str, float] | None = None,
) -> RegressionResult:
    """Compare current report to a saved baseline.

    Quality metrics fail when ``current - baseline`` is below ``min_deltas[metric]``
    (default -0.03). Risk metrics fail when increase exceeds ``risk_limits``.
    """
    if isinstance(baseline, (str, Path)):
        baseline_path = str(baseline)
        base = load_baseline(baseline)
    else:
        baseline_path = None
        base = dict(baseline)

    current = baseline_payload_from_report(report)
    cur_m = _flat_metrics(current)
    base_m = _flat_metrics(base)
    floors = dict(DEFAULT_MIN_DELTAS)
    if min_deltas:
        floors.update(min_deltas)
    risks = dict(RISK_METRICS)
    if risk_limits:
        risks.update(risk_limits)

    deltas: dict[str, float] = {}
    failures: list[str] = []
    compared: list[str] = []

    keys = sorted(set(cur_m) & set(base_m))
    for key in keys:
        delta = round(cur_m[key] - base_m[key], 4)
        deltas[key] = delta
        compared.append(key)
        if key in risks:
            if delta > risks[key]:
                failures.append(
                    f"{key} rose {delta:+.4f} (limit +{risks[key]:.4f})"
                )
            continue
        if key.startswith("latency_"):
            # Latency regressions: any increase beyond 20% of baseline or +50ms.
            base_v = base_m[key]
            limit = max(50.0, 0.20 * base_v)
            if delta > limit:
                failures.append(
                    f"{key} rose {delta:+.2f}ms (limit +{limit:.2f}ms)"
                )
            continue
        floor = floors.get(key)
        if floor is not None and delta < floor:
            failures.append(
                f"{key} dropped {delta:+.4f} (floor {floor:+.4f})"
            )

    return RegressionResult(
        passed=not failures,
        baseline_path=baseline_path,
        deltas=deltas,
        failures=failures,
        compared_keys=compared,
    )
