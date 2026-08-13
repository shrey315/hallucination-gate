from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total = 0
        self.requests_error = 0
        self.timeouts = 0
        self.auth_failures = 0
        self.latency_ms: list[float] = []
        self.gate_actions: dict[str, int] = defaultdict(int)
        self.max_samples = 5000

    def record_request(self, latency_ms: float, ok: bool, gate_action: str | None = None) -> None:
        with self._lock:
            self.requests_total += 1
            if not ok:
                self.requests_error += 1
            self.latency_ms.append(latency_ms)
            if len(self.latency_ms) > self.max_samples:
                self.latency_ms = self.latency_ms[-self.max_samples :]
            if gate_action:
                self.gate_actions[gate_action] += 1

    def record_timeout(self) -> None:
        with self._lock:
            self.timeouts += 1
            self.requests_error += 1
            self.requests_total += 1

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
            }


REGISTRY = MetricsRegistry()


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
