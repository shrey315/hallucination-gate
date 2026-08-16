"""Latency SLO / budget checks for eval reports and CI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass
class LatencyBudget:
    """Fail eval when observed latency exceeds these ceilings (milliseconds)."""

    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LatencyReport:
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    budget: dict[str, float] = field(default_factory=dict)
    ok: bool = True
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1))))
    return float(sorted_vals[idx])


def summarize_latency(latencies_ms: Sequence[float]) -> dict[str, float]:
    vals = sorted(float(x) for x in latencies_ms if x is not None)
    if not vals:
        return {"n": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    return {
        "n": float(len(vals)),
        "p50_ms": round(_percentile(vals, 50), 2),
        "p95_ms": round(_percentile(vals, 95), 2),
        "p99_ms": round(_percentile(vals, 99), 2),
        "max_ms": round(vals[-1], 2),
        "mean_ms": round(sum(vals) / len(vals), 2),
    }


def check_latency_budget(
    latencies_ms: Sequence[float],
    budget: LatencyBudget | None,
) -> LatencyReport:
    stats = summarize_latency(latencies_ms)
    failures: list[str] = []
    budget_dict = budget.as_dict() if budget else {}
    if budget:
        checks = (
            ("p50_ms", budget.p50_ms, stats["p50_ms"]),
            ("p95_ms", budget.p95_ms, stats["p95_ms"]),
            ("p99_ms", budget.p99_ms, stats["p99_ms"]),
            ("max_ms", budget.max_ms, stats["max_ms"]),
        )
        for name, limit, observed in checks:
            if limit is not None and observed > limit:
                failures.append(f"{name} {observed:.2f}ms exceeds budget {limit:.2f}ms")
    return LatencyReport(
        n=int(stats["n"]),
        p50_ms=stats["p50_ms"],
        p95_ms=stats["p95_ms"],
        p99_ms=stats["p99_ms"],
        max_ms=stats["max_ms"],
        mean_ms=stats["mean_ms"],
        budget=budget_dict,
        ok=not failures,
        failures=failures,
    )
