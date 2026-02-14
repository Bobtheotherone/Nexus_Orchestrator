"""Resource governance primitives for local backpressure decisions."""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python <3.11 compatibility.
    UTC = timezone.utc  # noqa: UP017

try:  # pragma: no cover - optional dependency.
    import psutil as _psutil  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - exercised by fallback tests.
    _psutil = None

_BYTES_PER_GIB = 1024 * 1024 * 1024
_MEMINFO_PATH = Path("/proc/meminfo")

ActionHook = Callable[[str], None]


class BackpressureLevel(str, Enum):
    """Resource pressure severity used by scheduler and dispatch loops."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ResourceMetricsSnapshot:
    """Point-in-time host resource metrics used for backpressure decisions."""

    captured_at: datetime
    cpu_percent: float | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    disk_total_bytes: int | None
    disk_free_bytes: int | None
    disk_read_bytes: int | None = None
    disk_write_bytes: int | None = None
    swap_used_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "captured_at", _coerce_datetime_utc(self.captured_at))
        _validate_optional_float(self.cpu_percent, "cpu_percent")
        _validate_optional_non_negative(self.memory_total_bytes, "memory_total_bytes")
        _validate_optional_non_negative(self.memory_available_bytes, "memory_available_bytes")
        _validate_optional_non_negative(self.disk_total_bytes, "disk_total_bytes")
        _validate_optional_non_negative(self.disk_free_bytes, "disk_free_bytes")
        _validate_optional_non_negative(self.disk_read_bytes, "disk_read_bytes")
        _validate_optional_non_negative(self.disk_write_bytes, "disk_write_bytes")
        _validate_optional_non_negative(self.swap_used_bytes, "swap_used_bytes")
        if (
            self.memory_total_bytes is not None
            and self.memory_available_bytes is not None
            and self.memory_available_bytes > self.memory_total_bytes
        ):
            raise ValueError("memory_available_bytes cannot exceed memory_total_bytes")
        if (
            self.disk_total_bytes is not None
            and self.disk_free_bytes is not None
            and self.disk_free_bytes > self.disk_total_bytes
        ):
            raise ValueError("disk_free_bytes cannot exceed disk_total_bytes")

    @property
    def memory_used_bytes(self) -> int | None:
        """Approximate used memory when total+available are known."""

        if self.memory_total_bytes is None or self.memory_available_bytes is None:
            return None
        return self.memory_total_bytes - self.memory_available_bytes

    @property
    def memory_utilization_ratio(self) -> float | None:
        """Used/total memory ratio in range [0.0, 1.0] when available."""

        if self.memory_total_bytes is None or self.memory_available_bytes is None:
            return None
        if self.memory_total_bytes <= 0:
            return None
        used = self.memory_total_bytes - self.memory_available_bytes
        return max(0.0, min(1.0, used / self.memory_total_bytes))

    @property
    def disk_free_ratio(self) -> float | None:
        """Free/total disk ratio in range [0.0, 1.0] when available."""

        if self.disk_total_bytes is None or self.disk_free_bytes is None:
            return None
        if self.disk_total_bytes <= 0:
            return None
        return max(0.0, min(1.0, self.disk_free_bytes / self.disk_total_bytes))


@dataclass(frozen=True, slots=True)
class ResourceGovernorConfig:
    """Thresholds and concurrency knobs for backpressure behavior."""

    default_verification_concurrency: int = 8
    default_dispatch_concurrency: int = 6
    minimum_verification_concurrency: int = 1
    minimum_dispatch_concurrency: int = 1
    elevated_verification_factor: float = 0.60
    critical_verification_factor: float = 0.35
    critical_dispatch_factor: float = 0.50
    elevated_cpu_percent: float = 85.0
    critical_cpu_percent: float = 95.0
    elevated_memory_utilization_ratio: float = 0.85
    critical_memory_utilization_ratio: float = 0.92
    elevated_memory_available_bytes: int = 8 * _BYTES_PER_GIB
    critical_memory_available_bytes: int = 6 * _BYTES_PER_GIB
    elevated_disk_free_bytes: int = 120 * _BYTES_PER_GIB
    critical_disk_free_bytes: int = 100 * _BYTES_PER_GIB

    def __post_init__(self) -> None:
        _validate_positive_int(
            self.default_verification_concurrency, "default_verification_concurrency"
        )
        _validate_positive_int(self.default_dispatch_concurrency, "default_dispatch_concurrency")
        _validate_positive_int(
            self.minimum_verification_concurrency, "minimum_verification_concurrency"
        )
        _validate_positive_int(self.minimum_dispatch_concurrency, "minimum_dispatch_concurrency")
        if self.minimum_verification_concurrency > self.default_verification_concurrency:
            raise ValueError("minimum_verification_concurrency cannot exceed default")
        if self.minimum_dispatch_concurrency > self.default_dispatch_concurrency:
            raise ValueError("minimum_dispatch_concurrency cannot exceed default")
        _validate_probability(self.elevated_verification_factor, "elevated_verification_factor")
        _validate_probability(self.critical_verification_factor, "critical_verification_factor")
        _validate_probability(self.critical_dispatch_factor, "critical_dispatch_factor")
        if self.critical_verification_factor > self.elevated_verification_factor:
            raise ValueError(
                "critical_verification_factor cannot exceed elevated_verification_factor"
            )
        _validate_threshold_pair(
            self.elevated_cpu_percent,
            self.critical_cpu_percent,
            field_name="cpu_percent",
            minimum=0.0,
            maximum=100.0,
        )
        _validate_threshold_pair(
            self.elevated_memory_utilization_ratio,
            self.critical_memory_utilization_ratio,
            field_name="memory_utilization_ratio",
            minimum=0.0,
            maximum=1.0,
        )
        _validate_threshold_pair(
            float(self.critical_memory_available_bytes),
            float(self.elevated_memory_available_bytes),
            field_name="memory_available_bytes_reverse",
            minimum=0.0,
            maximum=float("inf"),
        )
        _validate_threshold_pair(
            float(self.critical_disk_free_bytes),
            float(self.elevated_disk_free_bytes),
            field_name="disk_free_bytes_reverse",
            minimum=0.0,
            maximum=float("inf"),
        )


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Current scheduler/runtime limits emitted by the governor."""

    speculative_execution_enabled: bool
    verification_concurrency: int
    dispatch_concurrency: int

    def __post_init__(self) -> None:
        _validate_positive_int(self.verification_concurrency, "verification_concurrency")
        _validate_positive_int(self.dispatch_concurrency, "dispatch_concurrency")


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    """One governance decision produced from a metrics snapshot."""

    snapshot: ResourceMetricsSnapshot
    level: BackpressureLevel
    limits: ResourceLimits
    actions: tuple[str, ...] = ()

    @property
    def is_throttled(self) -> bool:
        return self.level is not BackpressureLevel.NORMAL


class MetricsProvider(Protocol):
    """Source for resource snapshots (injectable for tests)."""

    def snapshot(self) -> ResourceMetricsSnapshot: ...


class SystemMetricsProvider:
    """Collect host metrics with ``psutil`` when available, stdlib fallback otherwise."""

    def __init__(self, *, disk_path: Path | str | None = None) -> None:
        raw_disk_path = Path.cwd() if disk_path is None else Path(disk_path)
        self._disk_path = raw_disk_path.resolve(strict=False)

    def snapshot(self) -> ResourceMetricsSnapshot:
        if _psutil is not None:
            snapshot = self._snapshot_from_psutil()
            if snapshot is not None:
                return snapshot
        return self._snapshot_without_psutil()

    def _snapshot_from_psutil(self) -> ResourceMetricsSnapshot | None:
        assert _psutil is not None
        try:
            cpu_percent = float(_psutil.cpu_percent(interval=None))
            memory = _psutil.virtual_memory()
            disk = _psutil.disk_usage(str(self._disk_path))
            disk_io = _psutil.disk_io_counters()
            swap = _psutil.swap_memory()
        except Exception:
            return None

        read_bytes: int | None = None
        write_bytes: int | None = None
        if disk_io is not None:
            read_bytes = int(getattr(disk_io, "read_bytes", 0))
            write_bytes = int(getattr(disk_io, "write_bytes", 0))

        return ResourceMetricsSnapshot(
            captured_at=datetime.now(tz=UTC),
            cpu_percent=max(0.0, min(100.0, cpu_percent)),
            memory_total_bytes=int(memory.total),
            memory_available_bytes=int(memory.available),
            disk_total_bytes=int(disk.total),
            disk_free_bytes=int(disk.free),
            disk_read_bytes=read_bytes,
            disk_write_bytes=write_bytes,
            swap_used_bytes=int(swap.used),
        )

    def _snapshot_without_psutil(self) -> ResourceMetricsSnapshot:
        cpu_percent = _fallback_cpu_percent()
        memory_total, memory_available = _fallback_memory()
        try:
            disk = shutil.disk_usage(self._disk_path)
            disk_total = int(disk.total)
            disk_free = int(disk.free)
        except OSError:
            disk_total = None
            disk_free = None

        return ResourceMetricsSnapshot(
            captured_at=datetime.now(tz=UTC),
            cpu_percent=cpu_percent,
            memory_total_bytes=memory_total,
            memory_available_bytes=memory_available,
            disk_total_bytes=disk_total,
            disk_free_bytes=disk_free,
            disk_read_bytes=None,
            disk_write_bytes=None,
            swap_used_bytes=None,
        )


class ResourceGovernor:
    """Evaluates host metrics and emits ordered throttling decisions."""

    def __init__(
        self,
        *,
        metrics_provider: MetricsProvider | None = None,
        config: ResourceGovernorConfig | None = None,
        action_hook: ActionHook | None = None,
    ) -> None:
        self._config = config or ResourceGovernorConfig()
        self._metrics_provider = metrics_provider or SystemMetricsProvider()
        self._limits = ResourceLimits(
            speculative_execution_enabled=True,
            verification_concurrency=self._config.default_verification_concurrency,
            dispatch_concurrency=self._config.default_dispatch_concurrency,
        )
        self._degradation_depth = 0
        self._action_hook = action_hook

    @property
    def config(self) -> ResourceGovernorConfig:
        return self._config

    @property
    def current_limits(self) -> ResourceLimits:
        return self._limits

    @property
    def degradation_depth(self) -> int:
        return self._degradation_depth

    def snapshot(self) -> ResourceMetricsSnapshot:
        return self._metrics_provider.snapshot()

    def evaluate(self, snapshot: ResourceMetricsSnapshot | None = None) -> ResourceDecision:
        metrics = snapshot or self.snapshot()
        level = self.classify(metrics)
        target_depth = _target_depth(level)
        actions: list[str] = []

        while self._degradation_depth < target_depth:
            action = self._apply_degradation_step(level)
            actions.append(action)
            self._emit_action(action)

        while self._degradation_depth > target_depth:
            action = self._recover_step()
            actions.append(action)
            self._emit_action(action)

        adjustment = self._sync_level_specific_limits(level)
        if adjustment is not None:
            actions.append(adjustment)
            self._emit_action(adjustment)

        return ResourceDecision(
            snapshot=metrics,
            level=level,
            limits=self._limits,
            actions=tuple(actions),
        )

    def classify(self, snapshot: ResourceMetricsSnapshot) -> BackpressureLevel:
        if self._is_critical(snapshot):
            return BackpressureLevel.CRITICAL
        if self._is_elevated(snapshot):
            return BackpressureLevel.ELEVATED
        return BackpressureLevel.NORMAL

    def _is_critical(self, snapshot: ResourceMetricsSnapshot) -> bool:
        if (
            snapshot.cpu_percent is not None
            and snapshot.cpu_percent >= self._config.critical_cpu_percent
        ):
            return True

        memory_ratio = snapshot.memory_utilization_ratio
        if (
            memory_ratio is not None
            and memory_ratio >= self._config.critical_memory_utilization_ratio
        ):
            return True
        if (
            snapshot.memory_available_bytes is not None
            and snapshot.memory_available_bytes <= self._config.critical_memory_available_bytes
        ):
            return True

        return (
            snapshot.disk_free_bytes is not None
            and snapshot.disk_free_bytes <= self._config.critical_disk_free_bytes
        )

    def _is_elevated(self, snapshot: ResourceMetricsSnapshot) -> bool:
        if (
            snapshot.cpu_percent is not None
            and snapshot.cpu_percent >= self._config.elevated_cpu_percent
        ):
            return True

        memory_ratio = snapshot.memory_utilization_ratio
        if (
            memory_ratio is not None
            and memory_ratio >= self._config.elevated_memory_utilization_ratio
        ):
            return True
        if (
            snapshot.memory_available_bytes is not None
            and snapshot.memory_available_bytes <= self._config.elevated_memory_available_bytes
        ):
            return True

        return (
            snapshot.disk_free_bytes is not None
            and snapshot.disk_free_bytes <= self._config.elevated_disk_free_bytes
        )

    def _apply_degradation_step(self, level: BackpressureLevel) -> str:
        next_step = self._degradation_depth + 1

        if next_step == 1:
            self._limits = replace(self._limits, speculative_execution_enabled=False)
            self._degradation_depth = 1
            return "disable_speculative_execution"

        if next_step == 2:
            target_verification = self._target_verification_concurrency(level)
            self._limits = replace(self._limits, verification_concurrency=target_verification)
            self._degradation_depth = 2
            return "reduce_verification_concurrency"

        if next_step == 3:
            target_dispatch = self._target_dispatch_concurrency(level)
            self._limits = replace(self._limits, dispatch_concurrency=target_dispatch)
            self._degradation_depth = 3
            return "throttle_dispatch_concurrency"

        raise RuntimeError(f"unsupported degradation step: {next_step}")

    def _recover_step(self) -> str:
        if self._degradation_depth == 3:
            self._limits = replace(
                self._limits,
                dispatch_concurrency=self._config.default_dispatch_concurrency,
            )
            self._degradation_depth = 2
            return "restore_dispatch_concurrency"

        if self._degradation_depth == 2:
            self._limits = replace(
                self._limits,
                verification_concurrency=self._config.default_verification_concurrency,
            )
            self._degradation_depth = 1
            return "restore_verification_concurrency"

        if self._degradation_depth == 1:
            self._limits = replace(self._limits, speculative_execution_enabled=True)
            self._degradation_depth = 0
            return "enable_speculative_execution"

        raise RuntimeError("cannot recover when degradation depth is zero")

    def _sync_level_specific_limits(self, level: BackpressureLevel) -> str | None:
        if self._degradation_depth < 2:
            return None

        target_verification = self._target_verification_concurrency(level)
        current_verification = self._limits.verification_concurrency
        if target_verification == current_verification:
            return None

        self._limits = replace(self._limits, verification_concurrency=target_verification)
        if target_verification < current_verification:
            return "tighten_verification_concurrency"
        return "relax_verification_concurrency"

    def _target_verification_concurrency(self, level: BackpressureLevel) -> int:
        if level is BackpressureLevel.CRITICAL:
            factor = self._config.critical_verification_factor
        elif level is BackpressureLevel.ELEVATED:
            factor = self._config.elevated_verification_factor
        else:
            return self._config.default_verification_concurrency
        return _scaled_limit(
            self._config.default_verification_concurrency,
            factor=factor,
            minimum=self._config.minimum_verification_concurrency,
        )

    def _target_dispatch_concurrency(self, level: BackpressureLevel) -> int:
        if level is not BackpressureLevel.CRITICAL:
            return self._config.default_dispatch_concurrency
        return _scaled_limit(
            self._config.default_dispatch_concurrency,
            factor=self._config.critical_dispatch_factor,
            minimum=self._config.minimum_dispatch_concurrency,
        )

    def _emit_action(self, action: str) -> None:
        if self._action_hook is None:
            return
        self._action_hook(action)


def _target_depth(level: BackpressureLevel) -> int:
    if level is BackpressureLevel.NORMAL:
        return 0
    if level is BackpressureLevel.ELEVATED:
        return 2
    return 3


def _scaled_limit(base: int, *, factor: float, minimum: int) -> int:
    scaled = int(math.floor(base * factor))
    if scaled >= base and base > minimum:
        scaled = base - 1
    return max(minimum, scaled)


def _fallback_cpu_percent() -> float | None:
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return None
    try:
        one_minute, _, _ = os.getloadavg()
    except OSError:
        return None

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 0:
        return None
    value = (one_minute / cpu_count) * 100.0
    return max(0.0, min(100.0, value))


def _fallback_memory() -> tuple[int | None, int | None]:
    if not _MEMINFO_PATH.exists():
        return None, None
    try:
        payload = _MEMINFO_PATH.read_text(encoding="utf-8")
    except OSError:
        return None, None

    fields: dict[str, int] = {}
    for line in payload.splitlines():
        key, _, value = line.partition(":")
        if not key or not value:
            continue
        tokens = value.strip().split()
        if not tokens:
            continue
        try:
            kib = int(tokens[0])
        except ValueError:
            continue
        fields[key.strip()] = kib * 1024

    total = fields.get("MemTotal")
    available = fields.get("MemAvailable", fields.get("MemFree"))
    return total, available


def _coerce_datetime_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("captured_at must be datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_positive_int(value: int, field_name: str) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")


def _validate_optional_non_negative(value: int | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be >= 0")


def _validate_optional_float(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite")


def _validate_probability(value: float, field_name: str) -> None:
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{field_name} must be in (0.0, 1.0]")


def _validate_threshold_pair(
    elevated: float,
    critical: float,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> None:
    if elevated < minimum or elevated > maximum:
        raise ValueError(f"{field_name}.elevated threshold out of range")
    if critical < minimum or critical > maximum:
        raise ValueError(f"{field_name}.critical threshold out of range")
    if critical < elevated:
        raise ValueError(f"{field_name}.critical threshold cannot be lower than elevated threshold")


__all__ = [
    "BackpressureLevel",
    "MetricsProvider",
    "ResourceDecision",
    "ResourceGovernor",
    "ResourceGovernorConfig",
    "ResourceLimits",
    "ResourceMetricsSnapshot",
    "SystemMetricsProvider",
]
