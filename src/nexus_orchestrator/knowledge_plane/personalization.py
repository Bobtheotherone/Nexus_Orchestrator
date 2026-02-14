"""Deterministic operator-profile loading and personalization overlays."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from nexus_orchestrator.domain.models import Constraint, ConstraintSeverity

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

DEFAULT_OPERATOR_PROFILE_PATH = Path("profiles") / "operator_profile.toml"

_BUDGET_MAX_ITERATIONS = "max_iterations"
_BUDGET_MAX_TOKENS = "max_tokens_per_attempt"
_BUDGET_MAX_COST = "max_cost_per_work_item_usd"
_PLANNING_BUDGET_KEYS = frozenset({_BUDGET_MAX_ITERATIONS, _BUDGET_MAX_TOKENS, _BUDGET_MAX_COST})
_PROVIDER_MODEL_KEYS = ("model_code", "model_architect")


class OperatorProfileLoadError(ValueError):
    """Raised when an operator profile cannot be read or parsed."""


@dataclass(frozen=True, slots=True)
class OperatorProfile:
    """Validated immutable operator-profile preferences."""

    name: str
    owner: str
    version: int
    planning_budget_overrides: Mapping[str, int | float]
    planning_enabled_roles: tuple[str, ...]
    planning_disabled_roles: tuple[str, ...]
    planning_should_toggles: Mapping[str, bool]
    routing_provider_models: Mapping[str, Mapping[str, str]]
    routing_role_provider_preferences: Mapping[str, str]
    routing_role_capability_profiles: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_non_empty_str(self.name, "name"))
        object.__setattr__(self, "owner", _as_non_empty_str(self.owner, "owner"))
        object.__setattr__(
            self,
            "version",
            _as_positive_int(self.version, "version"),
        )
        object.__setattr__(
            self,
            "planning_budget_overrides",
            MappingProxyType(
                {
                    key: value
                    for key, value in sorted(
                        _validate_budget_overrides(self.planning_budget_overrides).items()
                    )
                }
            ),
        )
        object.__setattr__(
            self,
            "planning_enabled_roles",
            tuple(
                sorted(
                    {
                        _normalize_role_name(role, "planning_enabled_roles[]")
                        for role in self.planning_enabled_roles
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "planning_disabled_roles",
            tuple(
                sorted(
                    {
                        _normalize_role_name(role, "planning_disabled_roles[]")
                        for role in self.planning_disabled_roles
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "planning_should_toggles",
            MappingProxyType(
                {
                    key: value
                    for key, value in sorted(
                        _validate_should_toggles(self.planning_should_toggles).items()
                    )
                }
            ),
        )
        object.__setattr__(
            self,
            "routing_provider_models",
            MappingProxyType(
                {
                    provider: MappingProxyType(model_cfg)
                    for provider, model_cfg in sorted(
                        _validate_provider_model_overrides(self.routing_provider_models).items()
                    )
                }
            ),
        )
        object.__setattr__(
            self,
            "routing_role_provider_preferences",
            MappingProxyType(
                {
                    role: provider
                    for role, provider in sorted(
                        _validate_role_provider_preferences(
                            self.routing_role_provider_preferences
                        ).items()
                    )
                }
            ),
        )
        object.__setattr__(
            self,
            "routing_role_capability_profiles",
            MappingProxyType(
                {
                    role: capability
                    for role, capability in sorted(
                        _validate_role_capability_profiles(
                            self.routing_role_capability_profiles
                        ).items()
                    )
                }
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> OperatorProfile:
        root = _as_object(payload, "operator_profile")
        profile = _as_object(root.get("profile", {}), "profile")
        planning = _as_object(root.get("planning", {}), "planning")
        routing = _as_object(root.get("routing", {}), "routing")

        planning_budgets = _as_object(planning.get("budgets", {}), "planning.budgets")
        planning_roles = _as_object(planning.get("roles", {}), "planning.roles")
        planning_constraints = _as_object(
            planning.get("constraints", {}),
            "planning.constraints",
        )

        should_toggles = _as_object(
            planning_constraints.get("should_toggles", {}),
            "planning.constraints.should_toggles",
        )
        relax_should_constraints = _as_str_tuple(
            planning_constraints.get("relax_should_constraints", ()),
            "planning.constraints.relax_should_constraints",
        )
        for constraint_id in relax_should_constraints:
            should_toggles.setdefault(constraint_id, False)

        routing_providers = _as_object(routing.get("providers", {}), "routing.providers")
        routing_role_preferences = _as_object(
            routing.get("role_provider_preferences", {}),
            "routing.role_provider_preferences",
        )
        routing_capability_profiles = _as_object(
            routing.get("capability_profiles", {}),
            "routing.capability_profiles",
        )

        # Backward-compatible parser for [routing_preferences]
        # architect_preference = "anthropic"
        # implementer_preference = "openai"
        legacy_preferences = _as_object(root.get("routing_preferences", {}), "routing_preferences")
        for key, value in legacy_preferences.items():
            if key.endswith("_preference"):
                role = key.removesuffix("_preference")
                routing_role_preferences.setdefault(role, value)

        return cls(
            name=_as_non_empty_str(profile.get("name", "default"), "profile.name"),
            owner=_as_non_empty_str(profile.get("owner", "private"), "profile.owner"),
            version=_as_positive_int(profile.get("version", 1), "profile.version"),
            planning_budget_overrides=_validate_budget_overrides(planning_budgets),
            planning_enabled_roles=_as_role_tuple(
                planning_roles.get("enabled", ()),
                "planning.roles.enabled",
            ),
            planning_disabled_roles=_as_role_tuple(
                planning_roles.get("disabled", ()),
                "planning.roles.disabled",
            ),
            planning_should_toggles=_validate_should_toggles(should_toggles),
            routing_provider_models=_validate_provider_model_overrides(routing_providers),
            routing_role_provider_preferences=_validate_role_provider_preferences(
                routing_role_preferences
            ),
            routing_role_capability_profiles=_validate_role_capability_profiles(
                routing_capability_profiles
            ),
        )

    @classmethod
    def from_toml(
        cls,
        profile_path: str | Path | None = None,
        *,
        config: Mapping[str, object] | None = None,
        base_dir: str | Path | None = None,
    ) -> OperatorProfile:
        return load_operator_profile(profile_path, config=config, base_dir=base_dir)

    def apply_to_planning(
        self,
        *,
        config: Mapping[str, object] | None = None,
        constraints: Sequence[Constraint] = (),
        constraint_severity_by_id: Mapping[str, ConstraintSeverity | str] | None = None,
    ) -> dict[str, JSONValue]:
        return planning_overlay_from_profile(
            self,
            config=config,
            constraints=constraints,
            constraint_severity_by_id=constraint_severity_by_id,
        )

    def apply_to_routing(
        self,
        *,
        config: Mapping[str, object] | None = None,
    ) -> dict[str, JSONValue]:
        return routing_overlay_from_profile(self, config=config)


def resolve_operator_profile_path(
    profile_path: str | Path | None = None,
    *,
    config: Mapping[str, object] | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve profile path from explicit arg, config, or default."""

    configured: object = profile_path
    if configured is None:
        configured = _nested_get(config, ("personalization", "profile_path"))
    if configured is None:
        candidate = DEFAULT_OPERATOR_PROFILE_PATH
    elif isinstance(configured, Path):
        candidate = configured
    elif isinstance(configured, str):
        stripped = configured.strip()
        if not stripped:
            raise OperatorProfileLoadError("profile path cannot be empty")
        candidate = Path(stripped)
    else:
        raise OperatorProfileLoadError(
            f"profile path must be a string/path, got {type(configured).__name__}"
        )

    expanded = candidate.expanduser()
    if expanded.is_absolute():
        return expanded.resolve(strict=False)

    if base_dir is None:
        return (Path.cwd() / expanded).resolve(strict=False)
    return (Path(base_dir).expanduser().resolve(strict=False) / expanded).resolve(strict=False)


def load_operator_profile(
    profile_path: str | Path | None = None,
    *,
    config: Mapping[str, object] | None = None,
    base_dir: str | Path | None = None,
) -> OperatorProfile:
    """Load and parse operator profile TOML from a deterministic path."""

    resolved_path = resolve_operator_profile_path(profile_path, config=config, base_dir=base_dir)
    try:
        with resolved_path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise OperatorProfileLoadError(f"operator profile not found: {resolved_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise OperatorProfileLoadError(f"invalid TOML in operator profile: {exc}") from exc
    except OSError as exc:
        raise OperatorProfileLoadError(f"unable to read operator profile: {exc}") from exc

    if not isinstance(parsed, Mapping):
        raise OperatorProfileLoadError("operator profile root must be a mapping")
    try:
        return OperatorProfile.from_mapping(parsed)
    except ValueError as exc:
        raise OperatorProfileLoadError(f"invalid operator profile {resolved_path}: {exc}") from exc


def planning_overlay_from_profile(
    profile: OperatorProfile,
    *,
    config: Mapping[str, object] | None = None,
    constraints: Sequence[Constraint] = (),
    constraint_severity_by_id: Mapping[str, ConstraintSeverity | str] | None = None,
) -> dict[str, JSONValue]:
    """Build a deterministic planning overlay from profile preferences."""

    if not _personalization_enabled(config):
        return {}

    overlay: dict[str, JSONValue] = {}
    budgets = dict(profile.planning_budget_overrides)
    if budgets:
        overlay["budgets"] = {
            key: budgets[key]
            for key in (_BUDGET_MAX_ITERATIONS, _BUDGET_MAX_TOKENS, _BUDGET_MAX_COST)
            if key in budgets
        }

    role_overlay: dict[str, JSONValue] = {}
    if profile.planning_enabled_roles:
        role_overlay["enabled"] = list(profile.planning_enabled_roles)
    if profile.planning_disabled_roles:
        role_overlay["disabled"] = list(profile.planning_disabled_roles)
    if role_overlay:
        overlay["roles"] = role_overlay

    constraint_overlay = _apply_should_toggles(
        requested_toggles=profile.planning_should_toggles,
        may_relax_should_constraints=_may_relax_should_constraints(config),
        constraints=constraints,
        constraint_severity_by_id=constraint_severity_by_id,
    )
    if constraint_overlay:
        overlay["constraints"] = constraint_overlay

    return overlay


def routing_overlay_from_profile(
    profile: OperatorProfile,
    *,
    config: Mapping[str, object] | None = None,
) -> dict[str, JSONValue]:
    """
    Build deterministic routing overlay for RoleRegistry.from_config input.

    The overlay can include:
    - providers.*.model_code/model_architect overrides
    - role-level routing hints for provider/capability-profile preferences
    """

    if not _personalization_enabled(config):
        return {}

    overlay: dict[str, JSONValue] = {}
    provider_overlay: dict[str, JSONValue] = {}
    for provider_name, model_cfg in sorted(profile.routing_provider_models.items()):
        provider_overlay[provider_name] = {
            key: model_cfg[key] for key in _PROVIDER_MODEL_KEYS if key in model_cfg
        }
    if provider_overlay:
        overlay["providers"] = provider_overlay

    hints: dict[str, JSONValue] = {}
    if profile.routing_role_provider_preferences:
        hints["role_provider_preferences"] = dict(profile.routing_role_provider_preferences)
    if profile.routing_role_capability_profiles:
        hints["role_capability_profiles"] = dict(profile.routing_role_capability_profiles)
    if hints:
        overlay["routing_hints"] = hints

    return overlay


def _apply_should_toggles(
    *,
    requested_toggles: Mapping[str, bool],
    may_relax_should_constraints: bool,
    constraints: Sequence[Constraint],
    constraint_severity_by_id: Mapping[str, ConstraintSeverity | str] | None,
) -> dict[str, JSONValue]:
    if not requested_toggles:
        return {}

    severity_by_id = _build_constraint_severity_map(
        constraints=constraints,
        explicit=constraint_severity_by_id,
    )
    allowed_toggles: dict[str, bool] = {}
    blocked_relaxations: list[JSONValue] = []
    for constraint_id in sorted(requested_toggles):
        toggle = requested_toggles[constraint_id]
        if toggle:
            allowed_toggles[constraint_id] = True
            continue

        parsed_severity = severity_by_id.get(constraint_id)
        if parsed_severity is ConstraintSeverity.MUST:
            blocked_relaxations.append(
                {"constraint_id": constraint_id, "reason": "must_constraints_cannot_be_relaxed"}
            )
            continue
        if not may_relax_should_constraints:
            blocked_relaxations.append(
                {"constraint_id": constraint_id, "reason": "config_disallows_should_relaxation"}
            )
            continue
        if parsed_severity is None:
            blocked_relaxations.append(
                {"constraint_id": constraint_id, "reason": "unknown_constraint_severity"}
            )
            continue
        if parsed_severity is not ConstraintSeverity.SHOULD:
            blocked_relaxations.append(
                {
                    "constraint_id": constraint_id,
                    "reason": "only_should_constraints_may_be_relaxed",
                }
            )
            continue
        allowed_toggles[constraint_id] = False

    out: dict[str, JSONValue] = {}
    if allowed_toggles:
        out["should_toggles"] = dict(sorted(allowed_toggles.items()))
    if blocked_relaxations:
        out["blocked_relaxations"] = blocked_relaxations
    return out


def _build_constraint_severity_map(
    *,
    constraints: Sequence[Constraint],
    explicit: Mapping[str, ConstraintSeverity | str] | None,
) -> dict[str, ConstraintSeverity]:
    severity_by_id: dict[str, ConstraintSeverity] = {}
    for constraint in constraints:
        severity_by_id[constraint.id] = constraint.severity

    if explicit is None:
        return severity_by_id

    for key, value in explicit.items():
        constraint_id = _as_non_empty_str(key, "constraint_severity_by_id key").upper()
        severity_by_id[constraint_id] = _as_constraint_severity(
            value,
            f"constraint_severity_by_id[{constraint_id}]",
        )
    return severity_by_id


def _as_constraint_severity(value: ConstraintSeverity | str, field_name: str) -> ConstraintSeverity:
    if isinstance(value, ConstraintSeverity):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a ConstraintSeverity or string")
    normalized = value.strip().lower()
    try:
        return ConstraintSeverity(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in ConstraintSeverity)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _validate_budget_overrides(payload: Mapping[str, object]) -> dict[str, int | float]:
    parsed = _as_object(payload, "planning_budget_overrides")
    unknown = sorted(key for key in parsed if key not in _PLANNING_BUDGET_KEYS)
    if unknown:
        raise ValueError(f"unsupported planning budget override key(s): {unknown}")

    out: dict[str, int | float] = {}
    if _BUDGET_MAX_ITERATIONS in parsed:
        out[_BUDGET_MAX_ITERATIONS] = _as_positive_int(
            parsed[_BUDGET_MAX_ITERATIONS],
            f"planning.budgets.{_BUDGET_MAX_ITERATIONS}",
        )
    if _BUDGET_MAX_TOKENS in parsed:
        out[_BUDGET_MAX_TOKENS] = _as_positive_int(
            parsed[_BUDGET_MAX_TOKENS],
            f"planning.budgets.{_BUDGET_MAX_TOKENS}",
        )
    if _BUDGET_MAX_COST in parsed:
        out[_BUDGET_MAX_COST] = _as_non_negative_float(
            parsed[_BUDGET_MAX_COST],
            f"planning.budgets.{_BUDGET_MAX_COST}",
        )
    return out


def _validate_should_toggles(payload: Mapping[str, object]) -> dict[str, bool]:
    parsed = _as_object(payload, "planning_should_toggles")
    out: dict[str, bool] = {}
    for constraint_id, toggle in parsed.items():
        normalized_id = _as_non_empty_str(
            constraint_id, "planning.constraints.should_toggles key"
        ).upper()
        if not isinstance(toggle, bool):
            raise ValueError(
                f"planning.constraints.should_toggles[{normalized_id}] must be boolean"
            )
        out[normalized_id] = toggle
    return out


def _validate_provider_model_overrides(
    payload: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    parsed = _as_object(payload, "routing.providers")
    out: dict[str, dict[str, str]] = {}
    for provider_name in sorted(parsed):
        provider_cfg_raw = parsed[provider_name]
        provider_key = _as_non_empty_str(provider_name, "routing.providers provider")
        provider_cfg = _as_object(provider_cfg_raw, f"routing.providers.{provider_key}")
        model_cfg: dict[str, str] = {}
        unknown = sorted(key for key in provider_cfg if key not in _PROVIDER_MODEL_KEYS)
        if unknown:
            raise ValueError(
                "routing.providers."
                f"{provider_key} supports only {_PROVIDER_MODEL_KEYS}; got {unknown}"
            )
        for model_key in _PROVIDER_MODEL_KEYS:
            model_value = provider_cfg.get(model_key)
            if model_value is None:
                continue
            model_cfg[model_key] = _as_non_empty_str(
                model_value, f"routing.providers.{provider_key}.{model_key}"
            )
        if model_cfg:
            out[provider_key] = model_cfg
    return out


def _validate_role_provider_preferences(payload: Mapping[str, object]) -> dict[str, str]:
    parsed = _as_object(payload, "routing.role_provider_preferences")
    out: dict[str, str] = {}
    for role_name, provider in sorted(parsed.items()):
        role_key = _normalize_role_name(role_name, "routing.role_provider_preferences role")
        out[role_key] = _as_non_empty_str(
            provider,
            f"routing.role_provider_preferences.{role_name}",
        )
    return out


def _validate_role_capability_profiles(payload: Mapping[str, object]) -> dict[str, str]:
    parsed = _as_object(payload, "routing.capability_profiles")
    out: dict[str, str] = {}
    for role_name, capability in sorted(parsed.items()):
        role_key = _normalize_role_name(role_name, "routing.capability_profiles role")
        out[role_key] = _as_non_empty_str(
            capability,
            f"routing.capability_profiles.{role_name}",
        )
    return out


def _as_object(value: object, field_name: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        out[key] = item
    return out


def _as_str_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence of strings")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence of strings")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(_as_non_empty_str(item, f"{field_name}[{index}]"))
    return tuple(out)


def _as_role_tuple(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {_normalize_role_name(role, field_name) for role in _as_str_tuple(value, field_name)}
        )
    )


def _normalize_role_name(value: object, field_name: str) -> str:
    parsed = _as_non_empty_str(value, field_name).lower()
    return "_".join(part for part in parsed.replace("-", " ").replace("_", " ").split() if part)


def _as_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{field_name} cannot be empty")
    if "\x00" in parsed:
        raise ValueError(f"{field_name} cannot contain NUL bytes")
    return parsed


def _as_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _as_non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _nested_get(payload: Mapping[str, object] | None, path: Sequence[str]) -> object | None:
    if payload is None:
        return None
    current: object = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        if part not in current:
            return None
        current = current[part]
    return current


def _personalization_enabled(config: Mapping[str, object] | None) -> bool:
    value = _nested_get(config, ("personalization", "enabled"))
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError("personalization.enabled must be boolean")
    return value


def _may_relax_should_constraints(config: Mapping[str, object] | None) -> bool:
    value = _nested_get(config, ("personalization", "may_relax_should_constraints"))
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("personalization.may_relax_should_constraints must be boolean")
    return value


__all__ = [
    "DEFAULT_OPERATOR_PROFILE_PATH",
    "OperatorProfile",
    "OperatorProfileLoadError",
    "load_operator_profile",
    "planning_overlay_from_profile",
    "resolve_operator_profile_path",
    "routing_overlay_from_profile",
]
