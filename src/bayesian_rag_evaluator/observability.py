from __future__ import annotations

import re
import threading
import time
from collections import defaultdict


_LABEL = re.compile(r"[^a-zA-Z0-9_:]")


def _sanitize_label(value: str) -> str:
    cleaned = _LABEL.sub("_", (value or "default").strip()) or "default"
    return cleaned[:64]


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_error = 0
        self.timeouts = 0
        self.auth_failures = 0
        self.latency_ms: list[float] = []
        self.gate_actions: dict[str, int] = defaultdict(int)
        self.evidence_gaps: dict[str, int] = defaultdict(int)
        self.by_tenant: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.max_samples = 5000

    def record_request(
        self,
        latency_ms: float,
        ok: bool,
        gate_action: str | None = None,
        tenant: str = "default",
        evidence_gap: str | None = None,
    ) -> None:
        tenant = _sanitize_label(tenant)
        with self._lock:
            self.requests_total += 1
            if not ok:
                self.requests_error += 1
            self.latency_ms.append(latency_ms)
            if len(self.latency_ms) > self.max_samples:
                self.latency_ms = self.latency_ms[-self.max_samples :]
            if gate_action:
                self.gate_actions[gate_action] += 1
                self.by_tenant[tenant][f"gate_{gate_action}"] += 1
            if evidence_gap:
                self.evidence_gaps[evidence_gap] += 1
                self.by_tenant[tenant][f"gap_{evidence_gap}"] += 1
            self.by_tenant[tenant]["requests_total"] += 1
            if not ok:
                self.by_tenant[tenant]["requests_error"] += 1

    def record_timeout(self, tenant: str = "default") -> None:
        tenant = _sanitize_label(tenant)
        with self._lock:
            self.timeouts += 1
            self.requests_error += 1
            self.requests_total += 1
            self.by_tenant[tenant]["timeouts"] += 1
            self.by_tenant[tenant]["requests_error"] += 1
            self.by_tenant[tenant]["requests_total"] += 1

    def record_auth_failure(self) -> None:
        with self._lock:
            self.auth_failures += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latency_ms)

            def pct(p: float) -> float:
                if not lat:
                    return 0.0
                idx = min(len(lat) - 1, int(round((p / 100.0) * (len(lat) - 1))))
                return round(lat[idx], 2)

            return {
                "requests_total": self.requests_total,
                "requests_error": self.requests_error,
                "timeouts": self.timeouts,
                "auth_failures": self.auth_failures,
                "latency_ms": {
                    "p50": pct(50),
                    "p95": pct(95),
                    "p99": pct(99),
                    "n": len(lat),
                },
                "gate_actions": dict(self.gate_actions),
                "evidence_gaps": dict(self.evidence_gaps),
                "by_tenant": {k: dict(v) for k, v in self.by_tenant.items()},
                "isolation": "process_local",
            }

    def prometheus(self) -> str:
        snap = self.snapshot()
        lines = [
            "# HELP hallucination_gate_requests_total Requests seen by this process.",
            "# TYPE hallucination_gate_requests_total counter",
            f"hallucination_gate_requests_total {snap['requests_total']}",
            "# HELP hallucination_gate_requests_error_total Failed requests.",
            "# TYPE hallucination_gate_requests_error_total counter",
            f"hallucination_gate_requests_error_total {snap['requests_error']}",
            "# HELP hallucination_gate_timeouts_total Evaluation timeouts.",
            "# TYPE hallucination_gate_timeouts_total counter",
            f"hallucination_gate_timeouts_total {snap['timeouts']}",
            "# HELP hallucination_gate_auth_failures_total Invalid or missing API keys.",
            "# TYPE hallucination_gate_auth_failures_total counter",
            f"hallucination_gate_auth_failures_total {snap['auth_failures']}",
            "# HELP hallucination_gate_latency_ms Request latency percentiles.",
            "# TYPE hallucination_gate_latency_ms gauge",
        ]
        for pct_name, value in snap["latency_ms"].items():
            if pct_name == "n":
                lines.append(
                    f'hallucination_gate_latency_samples {value}'
                )
            else:
                lines.append(
                    f'hallucination_gate_latency_ms{{quantile="{pct_name}"}} {value}'
                )
        lines += [
            "# HELP hallucination_gate_actions_total Gate decisions by action.",
            "# TYPE hallucination_gate_actions_total counter",
        ]
        for action, count in snap["gate_actions"].items():
            lines.append(
                f'hallucination_gate_actions_total{{action="{_sanitize_label(action)}"}} {count}'
            )
        lines += [
            "# HELP hallucination_gate_evidence_gap_total Abstain/rewrite class.",
            "# TYPE hallucination_gate_evidence_gap_total counter",
        ]
        for gap, count in snap["evidence_gaps"].items():
            lines.append(
                f'hallucination_gate_evidence_gap_total{{gap="{_sanitize_label(gap)}"}} {count}'
            )
        lines += [
            "# HELP hallucination_gate_tenant_requests_total Per-API-key tenant counters.",
            "# TYPE hallucination_gate_tenant_requests_total counter",
        ]
        for tenant, counters in snap["by_tenant"].items():
            total = counters.get("requests_total", 0)
            lines.append(
                f'hallucination_gate_tenant_requests_total{{tenant="{tenant}"}} {total}'
            )
        lines.append("")
        return "\n".join(lines)


REGISTRY = MetricsRegistry()


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
