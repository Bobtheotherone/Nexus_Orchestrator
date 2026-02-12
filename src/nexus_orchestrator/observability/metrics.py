"""Thread-safe run metrics registry with deterministic JSON export."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

if TYPE_CHECKING:
    from collections.abc import Mapping

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_MetricLabels = tuple[tuple[str, str], ...]

_METRIC_NAME_MAX_LEN: Final[int] = 128
_LABEL_KEY_MAX_LEN: Final[int] = 128
_LABEL_VALUE_MAX_LEN: Final[int] = 256


@dataclass(frozen=True, order=True, slots=True)
class _MetricKey:
    name: str
    labels: _MetricLabels


@dataclass(slots=True)
class _DistributionState:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    last: float | None = None

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.last = value
        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    def as_dict(self) -> dict[str, JSONValue]:
        avg = self.total / self.count if self.count else 0.0
        return {
            "count": self.count,
            "sum": self.total,
            "min": self.minimum,
            "max": self.maximum,
            "avg": avg,
            "last": self.last,
        }


class MetricsRegistry:
    """In-memory metrics store for scheduler/dashboard observability."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._created_at = datetime.now(tz=UTC)
        self._counters: dict[_MetricKey, float] = {}
        self._gauges: dict[_MetricKey, float] = {}
        self._distributions: dict[_MetricKey, _DistributionState] = {}

    def inc(
        self,
        name: str,
        amount: float = 1.0,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Increment a counter by ``amount`` (>= 0)."""

        delta = _as_finite_float(amount, path="amount")
        if delta < 0:
            raise ValueError("counter increment amount must be >= 0")

        key = _metric_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + delta

    def increment_counter(
        self,
        name: str,
        *,
        amount: float = 1.0,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Backwards-compatible alias for ``inc``."""

        self.inc(name, amount, labels=labels)

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Set gauge value."""

        key = _metric_key(name, labels)
        gauge_value = _as_finite_float(value, path="value")
        with self._lock:
            self._gauges[key] = gauge_value

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        """Record sample for rolling distribution statistics."""

        key = _metric_key(name, labels)
        sample = _as_finite_float(value, path="value")
        with self._lock:
            state = self._distributions.get(key)
            if state is None:
                state = _DistributionState()
                self._distributions[key] = state
            state.observe(sample)

    def get_counter(self, name: str, *, labels: Mapping[str, str] | None = None) -> float:
        key = _metric_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, *, labels: Mapping[str, str] | None = None) -> float | None:
        key = _metric_key(name, labels)
        with self._lock:
            return self._gauges.get(key)

    def get_distribution(
        self,
        name: str,
        *,
        labels: Mapping[str, str] | None = None,
    ) -> dict[str, JSONValue] | None:
        key = _metric_key(name, labels)
        with self._lock:
            state = self._distributions.get(key)
            if state is None:
                return None
            return dict(state.as_dict())

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._distributions.clear()
            self._created_at = datetime.now(tz=UTC)

    def snapshot(self) -> dict[str, JSONValue]:
        """Return deterministic snapshot with stable key ordering."""

        with self._lock:
            created_at = self._created_at
            counters = tuple(sorted(self._counters.items()))
            gauges = tuple(sorted(self._gauges.items()))
            distributions = tuple(sorted(self._distributions.items()))

        now = datetime.now(tz=UTC)
        uptime_seconds = max(0.0, (now - created_at).total_seconds())

        counters_out: dict[str, JSONValue] = {}
        counter_rates: dict[str, JSONValue] = {}
        for key, value in counters:
            ident = _metric_identifier(key)
            counters_out[ident] = value
            counter_rates[ident] = (value / uptime_seconds) if uptime_seconds > 0 else 0.0

        gauges_out: dict[str, JSONValue] = {}
        for key, value in gauges:
            gauges_out[_metric_identifier(key)] = value

        distributions_out: dict[str, JSONValue] = {}
        for key, state in distributions:
            distributions_out[_metric_identifier(key)] = state.as_dict()

        return {
            "metadata": {
                "created_at": created_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "snapshot_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "uptime_seconds": uptime_seconds,
            },
            "counters": counters_out,
            "counter_rates_per_second": counter_rates,
            "gauges": gauges_out,
            "distributions": distributions_out,
        }

    def export_dict(self) -> dict[str, JSONValue]:
        """Backwards-compatible alias for ``snapshot``."""

        return self.snapshot()

    def to_json(self, *, indent: int | None = None) -> str:
        payload = self.snapshot()
        if indent is None:
            return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return json.dumps(payload, sort_keys=True, indent=indent, ensure_ascii=False)

    def export_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write snapshot JSON to ``path`` and return normalized path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent), encoding="utf-8")
        return output_path


def _metric_key(name: str, labels: Mapping[str, str] | None) -> _MetricKey:
    return _MetricKey(name=_validate_metric_name(name), labels=_normalize_labels(labels))


def _metric_identifier(key: _MetricKey) -> str:
    if not key.labels:
        return key.name
    labels = ",".join(f"{k}={v}" for k, v in key.labels)
    return f"{key.name}{{{labels}}}"


def _validate_metric_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"metric name must be a string, got {type(name).__name__}")
    normalized = name.strip()
    if not normalized:
        raise ValueError("metric name must not be empty")
    if len(normalized) > _METRIC_NAME_MAX_LEN:
        raise ValueError(f"metric name must be <= {_METRIC_NAME_MAX_LEN} characters")
    return normalized


def _normalize_labels(labels: Mapping[str, str] | None) -> _MetricLabels:
    if labels is None:
        return ()

    out: list[tuple[str, str]] = []
    for key, value in labels.items():
        if not isinstance(key, str):
            raise ValueError(f"label key must be a string, got {type(key).__name__}")
        if not isinstance(value, str):
            raise ValueError(f"label value for {key!r} must be a string")

        key_name = key.strip()
        val_name = value.strip()

        if not key_name:
            raise ValueError("label key must not be empty")
        if not val_name:
            raise ValueError(f"label value for {key!r} must not be empty")
        if len(key_name) > _LABEL_KEY_MAX_LEN:
            raise ValueError(f"label key {key!r} exceeds {_LABEL_KEY_MAX_LEN} characters")
        if len(val_name) > _LABEL_VALUE_MAX_LEN:
            raise ValueError(f"label value for {key!r} exceeds {_LABEL_VALUE_MAX_LEN} characters")

        out.append((key_name, val_name))

    out.sort(key=lambda item: item[0])
    return tuple(out)


def _as_finite_float(value: float, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric, got {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be finite")
    return parsed


__all__ = ["JSONScalar", "JSONValue", "MetricsRegistry"]
