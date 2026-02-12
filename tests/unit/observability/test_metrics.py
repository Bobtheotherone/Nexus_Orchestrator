"""
nexus-orchestrator — unit tests for observability metrics

File: tests/unit/observability/test_metrics.py
Last updated: 2026-02-12

Purpose
- Verify thread-safe metric updates and deterministic snapshot/export behavior.

What this test file should cover
- Thread-safe counter increments.
- Deterministic snapshot key stability.
- JSON export to filesystem.

Functional requirements
- Offline operation.

Non-functional requirements
- Deterministic outputs.
"""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from nexus_orchestrator.observability.metrics import MetricsRegistry

if TYPE_CHECKING:
    from pathlib import Path


def test_thread_safe_counter_increments() -> None:
    registry = MetricsRegistry()

    def worker() -> None:
        for _ in range(2000):
            registry.inc("work_items_total", 1)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert registry.get_counter("work_items_total") == 12_000.0


def test_snapshot_is_deterministic_and_json_serializable() -> None:
    registry = MetricsRegistry()
    registry.inc("merge_failures_total", 2, labels={"z": "9", "a": "1"})
    registry.set_gauge("queue_depth", 3)
    registry.observe("latency_ms", 10)
    registry.observe("latency_ms", 20)

    first = registry.snapshot()
    second = registry.snapshot()

    assert first["counters"] == second["counters"]
    assert first["gauges"] == second["gauges"]
    assert first["distributions"] == second["distributions"]

    payload = json.dumps(first, sort_keys=True)
    assert "merge_failures_total{a=1,z=9}" in payload


def test_export_json_writes_file(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    registry.inc("work_items_total", 3)
    registry.set_gauge("active_agents", 2)

    out = registry.export_json(tmp_path / "metrics.json")

    assert out.exists()
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["counters"]["work_items_total"] == 3.0
    assert parsed["gauges"]["active_agents"] == 2.0
