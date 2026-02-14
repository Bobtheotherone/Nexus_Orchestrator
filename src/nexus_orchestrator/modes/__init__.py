"""Deterministic operating mode registry and immutable mode settings."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

MODE_GREENFIELD = "greenfield"
MODE_BROWNFIELD = "brownfield"
MODE_HARDENING = "hardening"
MODE_EXPLORATION = "exploration"

ROLE_ARCHITECT = "architect"
ROLE_IMPLEMENTER = "implementer"
ROLE_TEST_ENGINEER = "test_engineer"
ROLE_REVIEWER = "reviewer"
ROLE_SECURITY = "security"
ROLE_PERFORMANCE = "performance"
ROLE_TOOLSMITH = "toolsmith"
ROLE_INTEGRATOR = "integrator"
ROLE_CONSTRAINT_MINER = "constraint_miner"
ROLE_DOCUMENTATION = "documentation"

ALL_ROLE_NAMES: tuple[str, ...] = (
    ROLE_ARCHITECT,
    ROLE_IMPLEMENTER,
    ROLE_TEST_ENGINEER,
    ROLE_REVIEWER,
    ROLE_SECURITY,
    ROLE_PERFORMANCE,
    ROLE_TOOLSMITH,
    ROLE_INTEGRATOR,
    ROLE_CONSTRAINT_MINER,
    ROLE_DOCUMENTATION,
)
RISK_TIERS: tuple[str, ...] = ("critical", "high", "medium", "low")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{field_name} cannot be empty")
    if "\x00" in parsed:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return parsed


def _normalize_identifier(value: str, *, field_name: str) -> str:
    parsed = _validate_non_empty_str(value, field_name=field_name).lower()
    normalized = "_".join(parsed.replace("-", " ").replace("_", " ").split())
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must match ^[a-z][a-z0-9_]*$")
    return normalized


def _normalize_role_name(value: str, *, field_name: str) -> str:
    role_name = _normalize_identifier(value, field_name=field_name)
    if role_name not in ALL_ROLE_NAMES:
        allowed = ", ".join(ALL_ROLE_NAMES)
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return role_name


def _normalize_mode_name(value: str, *, field_name: str) -> str:
    mode_name = _normalize_identifier(value, field_name=field_name)
    allowed_modes = (
        MODE_GREENFIELD,
        MODE_BROWNFIELD,
        MODE_HARDENING,
        MODE_EXPLORATION,
    )
    if mode_name not in allowed_modes:
        allowed = ", ".join(allowed_modes)
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return mode_name


def _freeze_json(value: object, *, path: str) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        frozen_mapping: dict[str, JSONValue] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError(f"{path} contains non-string key")
            frozen_mapping[key] = _freeze_json(value[key], path=f"{path}.{key}")
        return MappingProxyType(frozen_mapping)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise ValueError(f"{path} contains unsupported value type: {type(value).__name__}")


def _thaw_json(value: JSONValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModeOverlay:
    """Deterministic config overlay emitted by a mode."""

    config: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        if not isinstance(self.config, Mapping):
            raise ValueError("ModeOverlay.config must be a mapping")
        object.__setattr__(
            self,
            "config",
            _freeze_json(self.config, path="ModeOverlay.config"),
        )

    def to_dict(self) -> dict[str, object]:
        return {key: _thaw_json(value) for key, value in self.config.items()}


@dataclass(frozen=True, slots=True)
class ModeRoleSettings:
    """Role enablement and deterministic role execution priority."""

    enabled_roles: tuple[str, ...]
    priority: tuple[str, ...]

    def __post_init__(self) -> None:
        enabled = tuple(
            _normalize_role_name(item, field_name="ModeRoleSettings.enabled_roles")
            for item in self.enabled_roles
        )
        if not enabled:
            raise ValueError("ModeRoleSettings.enabled_roles cannot be empty")
        if len(set(enabled)) != len(enabled):
            raise ValueError("ModeRoleSettings.enabled_roles must not contain duplicates")

        priority = tuple(
            _normalize_role_name(item, field_name="ModeRoleSettings.priority")
            for item in self.priority
        )
        if len(set(priority)) != len(priority):
            raise ValueError("ModeRoleSettings.priority must not contain duplicates")
        if set(priority) != set(enabled):
            raise ValueError(
                "ModeRoleSettings.priority must contain exactly the enabled_roles values"
            )

        object.__setattr__(self, "enabled_roles", enabled)
        object.__setattr__(self, "priority", priority)

    def disabled_roles(self, *, universe: Sequence[str] = ALL_ROLE_NAMES) -> tuple[str, ...]:
        enabled = set(self.enabled_roles)
        return tuple(role_name for role_name in universe if role_name not in enabled)


@dataclass(frozen=True, slots=True)
class ModeSchedulingSettings:
    """Scheduling policy knobs consumed by control-plane callers."""

    priorities: tuple[str, ...]
    allow_speculative_execution: bool
    max_dispatch_per_tick: int
    max_in_flight: int
    risk_tier_priority: Mapping[str, int]
    per_risk_tier_in_flight: Mapping[str, int]

    def __post_init__(self) -> None:
        normalized_priorities = tuple(
            _normalize_identifier(item, field_name="ModeSchedulingSettings.priorities")
            for item in self.priorities
        )
        if not normalized_priorities:
            raise ValueError("ModeSchedulingSettings.priorities cannot be empty")
        if len(set(normalized_priorities)) != len(normalized_priorities):
            raise ValueError("ModeSchedulingSettings.priorities must not contain duplicates")
        object.__setattr__(self, "priorities", normalized_priorities)

        if self.max_dispatch_per_tick <= 0:
            raise ValueError("ModeSchedulingSettings.max_dispatch_per_tick must be > 0")
        if self.max_in_flight <= 0:
            raise ValueError("ModeSchedulingSettings.max_in_flight must be > 0")
        if self.max_dispatch_per_tick > self.max_in_flight:
            raise ValueError("max_dispatch_per_tick cannot exceed max_in_flight")

        normalized_risk_priority = _normalize_risk_int_map(
            self.risk_tier_priority,
            field_name="ModeSchedulingSettings.risk_tier_priority",
            allow_zero=True,
            max_value=None,
        )
        normalized_tier_limits = _normalize_risk_int_map(
            self.per_risk_tier_in_flight,
            field_name="ModeSchedulingSettings.per_risk_tier_in_flight",
            allow_zero=False,
            max_value=self.max_in_flight,
        )

        object.__setattr__(self, "risk_tier_priority", normalized_risk_priority)
        object.__setattr__(self, "per_risk_tier_in_flight", normalized_tier_limits)


def _normalize_risk_int_map(
    payload: Mapping[str, int],
    *,
    field_name: str,
    allow_zero: bool,
    max_value: int | None,
) -> Mapping[str, int]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field_name} must be a mapping")

    raw_keys = set(payload.keys())
    if raw_keys != set(RISK_TIERS):
        missing = sorted(set(RISK_TIERS) - raw_keys)
        unknown = sorted(raw_keys - set(RISK_TIERS))
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(f"{field_name} must define all risk tiers ({', '.join(details)})")

    normalized: dict[str, int] = {}
    for tier in RISK_TIERS:
        raw = payload[tier]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{field_name}[{tier!r}] must be an integer")
        if raw < 0 or (raw == 0 and not allow_zero):
            comparator = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{field_name}[{tier!r}] must be {comparator}")
        if max_value is not None and raw > max_value:
            raise ValueError(f"{field_name}[{tier!r}] must be <= {max_value}")
        normalized[tier] = raw

    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ModeControllerSettings:
    """Controller-facing settings overlay for a mode."""

    roles: ModeRoleSettings
    scheduling: ModeSchedulingSettings

    def __post_init__(self) -> None:
        if not isinstance(self.roles, ModeRoleSettings):
            raise ValueError("ModeControllerSettings.roles must be ModeRoleSettings")
        if not isinstance(self.scheduling, ModeSchedulingSettings):
            raise ValueError("ModeControllerSettings.scheduling must be ModeSchedulingSettings")


@dataclass(frozen=True, slots=True)
class OperatingMode:
    """Fully-resolved immutable operating mode configuration."""

    name: str
    description: str
    overlay: ModeOverlay
    settings: ModeControllerSettings

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_mode_name(self.name, field_name="OperatingMode.name"),
        )
        object.__setattr__(
            self,
            "description",
            _validate_non_empty_str(self.description, field_name="OperatingMode.description"),
        )
        if not isinstance(self.overlay, ModeOverlay):
            raise ValueError("OperatingMode.overlay must be ModeOverlay")
        if not isinstance(self.settings, ModeControllerSettings):
            raise ValueError("OperatingMode.settings must be ModeControllerSettings")


class ModeRegistry:
    """Deterministic registry for mode lookup and resolution."""

    __slots__ = ("_default_mode_name", "_mode_names", "_modes_by_name")

    def __init__(
        self,
        modes: Sequence[OperatingMode],
        *,
        default_mode: str = MODE_GREENFIELD,
    ) -> None:
        if not modes:
            raise ValueError("ModeRegistry.modes cannot be empty")

        normalized_modes: dict[str, OperatingMode] = {}
        ordered_names: list[str] = []
        for mode in modes:
            if not isinstance(mode, OperatingMode):
                raise ValueError("ModeRegistry.modes entries must be OperatingMode")
            if mode.name in normalized_modes:
                raise ValueError(f"duplicate mode name: {mode.name}")
            normalized_modes[mode.name] = mode
            ordered_names.append(mode.name)

        normalized_default = _normalize_mode_name(default_mode, field_name="default_mode")
        if normalized_default not in normalized_modes:
            raise ValueError(f"default mode {normalized_default!r} is not present in registry")

        self._modes_by_name = MappingProxyType(normalized_modes)
        self._mode_names = tuple(ordered_names)
        self._default_mode_name = normalized_default

    @classmethod
    def default(cls) -> ModeRegistry:
        return DEFAULT_MODE_REGISTRY

    @property
    def mode_names(self) -> tuple[str, ...]:
        return self._mode_names

    @property
    def default_mode_name(self) -> str:
        return self._default_mode_name

    @property
    def default_mode(self) -> OperatingMode:
        return self._modes_by_name[self._default_mode_name]

    def lookup(self, mode_name: str | None) -> OperatingMode | None:
        if mode_name is None:
            return None
        stripped = mode_name.strip()
        if not stripped:
            return None
        normalized = _normalize_identifier(stripped, field_name="mode_name")
        if normalized == "default":
            return self.default_mode
        return self._modes_by_name.get(normalized)

    def resolve(self, mode_name: str | None = None) -> OperatingMode:
        if mode_name is None or not mode_name.strip():
            return self.default_mode
        found = self.lookup(mode_name)
        if found is not None:
            return found
        options = ", ".join(self._mode_names)
        raise ValueError(f"unknown mode {mode_name!r}; expected one of: {options}")


def resolve_mode(
    mode_name: str | None = None, *, registry: ModeRegistry | None = None
) -> OperatingMode:
    """Resolve mode name to immutable mode settings."""

    target_registry = DEFAULT_MODE_REGISTRY if registry is None else registry
    return target_registry.resolve(mode_name)


def _build_default_mode_registry() -> ModeRegistry:
    from .brownfield import BROWNFIELD_MODE
    from .exploration import EXPLORATION_MODE
    from .greenfield import GREENFIELD_MODE
    from .hardening import HARDENING_MODE

    return ModeRegistry(
        modes=(
            GREENFIELD_MODE,
            BROWNFIELD_MODE,
            HARDENING_MODE,
            EXPLORATION_MODE,
        ),
        default_mode=MODE_GREENFIELD,
    )


DEFAULT_MODE_REGISTRY = _build_default_mode_registry()
GREENFIELD_MODE = DEFAULT_MODE_REGISTRY.resolve(MODE_GREENFIELD)
BROWNFIELD_MODE = DEFAULT_MODE_REGISTRY.resolve(MODE_BROWNFIELD)
HARDENING_MODE = DEFAULT_MODE_REGISTRY.resolve(MODE_HARDENING)
EXPLORATION_MODE = DEFAULT_MODE_REGISTRY.resolve(MODE_EXPLORATION)

__all__ = [
    "ALL_ROLE_NAMES",
    "BROWNFIELD_MODE",
    "DEFAULT_MODE_REGISTRY",
    "EXPLORATION_MODE",
    "GREENFIELD_MODE",
    "HARDENING_MODE",
    "MODE_BROWNFIELD",
    "MODE_EXPLORATION",
    "MODE_GREENFIELD",
    "MODE_HARDENING",
    "ModeControllerSettings",
    "ModeOverlay",
    "ModeRegistry",
    "ModeRoleSettings",
    "ModeSchedulingSettings",
    "OperatingMode",
    "resolve_mode",
]
