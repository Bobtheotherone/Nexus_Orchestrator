"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/roles.py
Last updated: 2026-02-11

Purpose
- Defines agent roles, capabilities, budgets, and default prompts.

What should be included in this file
- Role enumeration and metadata (allowed tools, sandbox permissions, model preferences).
- Risk-tier mapping to required roles (e.g., critical requires Reviewer + Security).

Functional requirements
- Must be configurable via orchestrator.toml (enable/disable roles, budgets).

Non-functional requirements
- Roles must be explicit and auditable; avoid implicit permissions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TypeAlias

from nexus_orchestrator.domain.models import RiskTier
from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog, load_model_catalog

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

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

_REQUIRED_CRITICAL_ROLES = frozenset({ROLE_REVIEWER, ROLE_SECURITY})


def _validate_non_empty_str(value: str, field_name: str, *, strip: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    parsed = value.strip() if strip else value
    if not parsed:
        raise ValueError(f"{field_name} cannot be empty")
    if "\x00" in parsed:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return parsed


def _validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def _validate_non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_identifier(value: str, field_name: str) -> str:
    parsed = _validate_non_empty_str(value, field_name).lower()
    parts = parsed.replace("-", " ").replace("_", " ").split()
    if not parts:
        raise ValueError(f"{field_name} cannot be empty")
    return "_".join(parts)


def _as_risk_tier(value: RiskTier | str) -> RiskTier:
    if isinstance(value, RiskTier):
        return value
    if not isinstance(value, str):
        raise ValueError("risk tier must be a RiskTier or string")
    try:
        return RiskTier(value.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in RiskTier))
        raise ValueError(f"invalid risk tier {value!r}; expected one of: {allowed}") from exc


@dataclass(frozen=True, slots=True)
class RoleBudget:
    """Optional per-role budget envelope."""

    max_attempts: int | None = None
    max_tokens_per_attempt: int | None = None
    max_cost_per_work_item_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts is not None:
            _validate_positive_int(self.max_attempts, "RoleBudget.max_attempts")
        if self.max_tokens_per_attempt is not None:
            _validate_positive_int(
                self.max_tokens_per_attempt,
                "RoleBudget.max_tokens_per_attempt",
            )
        if self.max_cost_per_work_item_usd is not None:
            _validate_non_negative_float(
                self.max_cost_per_work_item_usd,
                "RoleBudget.max_cost_per_work_item_usd",
            )

    def merge(self, override: RoleBudget | None) -> RoleBudget:
        if override is None:
            return self
        return RoleBudget(
            max_attempts=(
                override.max_attempts if override.max_attempts is not None else self.max_attempts
            ),
            max_tokens_per_attempt=(
                override.max_tokens_per_attempt
                if override.max_tokens_per_attempt is not None
                else self.max_tokens_per_attempt
            ),
            max_cost_per_work_item_usd=(
                override.max_cost_per_work_item_usd
                if override.max_cost_per_work_item_usd is not None
                else self.max_cost_per_work_item_usd
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> RoleBudget:
        allowed_fields = {
            "max_attempts",
            "max_iterations",
            "max_tokens_per_attempt",
            "max_cost_per_work_item_usd",
        }
        unknown = sorted(key for key in payload if key not in allowed_fields)
        if unknown:
            raise ValueError(f"unsupported role budget fields: {unknown}")

        max_attempts_raw = payload.get("max_attempts", payload.get("max_iterations"))
        if (
            "max_attempts" in payload
            and "max_iterations" in payload
            and payload["max_attempts"] != payload["max_iterations"]
        ):
            raise ValueError("max_attempts and max_iterations cannot disagree")

        max_attempts: int | None = None
        if max_attempts_raw is not None:
            if isinstance(max_attempts_raw, bool) or not isinstance(max_attempts_raw, int):
                raise ValueError("max_attempts must be an integer")
            max_attempts = max_attempts_raw

        max_tokens: int | None = None
        max_tokens_raw = payload.get("max_tokens_per_attempt")
        if max_tokens_raw is not None:
            if isinstance(max_tokens_raw, bool) or not isinstance(max_tokens_raw, int):
                raise ValueError("max_tokens_per_attempt must be an integer")
            max_tokens = max_tokens_raw

        max_cost: float | None = None
        max_cost_raw = payload.get("max_cost_per_work_item_usd")
        if max_cost_raw is not None:
            max_cost = _validate_non_negative_float(
                max_cost_raw,
                "RoleBudget.max_cost_per_work_item_usd",
            )

        return cls(
            max_attempts=max_attempts,
            max_tokens_per_attempt=max_tokens,
            max_cost_per_work_item_usd=max_cost,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {}
        if self.max_attempts is not None:
            payload["max_attempts"] = self.max_attempts
        if self.max_tokens_per_attempt is not None:
            payload["max_tokens_per_attempt"] = self.max_tokens_per_attempt
        if self.max_cost_per_work_item_usd is not None:
            payload["max_cost_per_work_item_usd"] = self.max_cost_per_work_item_usd
        return payload


@dataclass(frozen=True, slots=True)
class EscalationStep:
    """A bounded chunk of attempts for a provider/model pair."""

    provider: str
    model: str
    attempts: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _validate_non_empty_str(self.provider, "EscalationStep.provider"),
        )
        object.__setattr__(
            self,
            "model",
            _validate_non_empty_str(self.model, "EscalationStep.model"),
        )
        object.__setattr__(
            self,
            "attempts",
            _validate_positive_int(self.attempts, "EscalationStep.attempts"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "provider": self.provider,
            "model": self.model,
            "attempts": self.attempts,
        }


@dataclass(frozen=True, slots=True)
class EscalationDecision:
    """Resolved deterministic provider/model binding for one attempt number."""

    attempt_number: int
    provider: str
    model: str
    stage_index: int
    stage_attempt: int
    stage_limit: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attempt_number",
            _validate_positive_int(self.attempt_number, "EscalationDecision.attempt_number"),
        )
        object.__setattr__(
            self,
            "provider",
            _validate_non_empty_str(self.provider, "EscalationDecision.provider"),
        )
        object.__setattr__(
            self,
            "model",
            _validate_non_empty_str(self.model, "EscalationDecision.model"),
        )
        if self.stage_index < 0:
            raise ValueError("EscalationDecision.stage_index must be >= 0")
        object.__setattr__(
            self,
            "stage_attempt",
            _validate_positive_int(self.stage_attempt, "EscalationDecision.stage_attempt"),
        )
        object.__setattr__(
            self,
            "stage_limit",
            _validate_positive_int(self.stage_limit, "EscalationDecision.stage_limit"),
        )
        if self.stage_attempt > self.stage_limit:
            raise ValueError("EscalationDecision.stage_attempt cannot exceed stage_limit")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "attempt_number": self.attempt_number,
            "provider": self.provider,
            "model": self.model,
            "stage_index": self.stage_index,
            "stage_attempt": self.stage_attempt,
            "stage_limit": self.stage_limit,
        }


@dataclass(frozen=True, slots=True)
class EscalationPolicy:
    """Bounded deterministic escalation ladder."""

    steps: tuple[EscalationStep, ...]

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("EscalationPolicy.steps cannot be empty")
        object.__setattr__(self, "steps", steps)

    @property
    def max_attempts(self) -> int:
        return sum(step.attempts for step in self.steps)

    def resolve_attempt(
        self,
        attempt_number: int,
        *,
        max_attempts_override: int | None = None,
    ) -> EscalationDecision | None:
        _validate_positive_int(attempt_number, "attempt_number")
        effective_max = self.max_attempts
        if max_attempts_override is not None:
            _validate_positive_int(max_attempts_override, "max_attempts_override")
            effective_max = min(effective_max, max_attempts_override)
        if attempt_number > effective_max:
            return None

        remaining = attempt_number
        for stage_index, step in enumerate(self.steps):
            if remaining <= step.attempts:
                return EscalationDecision(
                    attempt_number=attempt_number,
                    provider=step.provider,
                    model=step.model,
                    stage_index=stage_index,
                    stage_attempt=remaining,
                    stage_limit=step.attempts,
                )
            remaining -= step.attempts
        return None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "max_attempts": self.max_attempts,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class AgentRole:
    """Role metadata, prompt entrypoint, and deterministic routing policy."""

    name: str
    display_name: str
    purpose: str
    required_for: str
    prompt_template_path: str
    escalation_policy: EscalationPolicy
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    sandbox_policy: str = "read_only"
    budget: RoleBudget = field(default_factory=RoleBudget)
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_identifier(self.name, "AgentRole.name"))
        object.__setattr__(
            self,
            "display_name",
            _validate_non_empty_str(self.display_name, "AgentRole.display_name"),
        )
        object.__setattr__(
            self,
            "purpose",
            _validate_non_empty_str(self.purpose, "AgentRole.purpose"),
        )
        object.__setattr__(
            self,
            "required_for",
            _validate_non_empty_str(self.required_for, "AgentRole.required_for"),
        )
        object.__setattr__(
            self,
            "prompt_template_path",
            _validate_non_empty_str(
                self.prompt_template_path,
                "AgentRole.prompt_template_path",
            ),
        )
        object.__setattr__(
            self,
            "sandbox_policy",
            _validate_non_empty_str(self.sandbox_policy, "AgentRole.sandbox_policy"),
        )
        if not isinstance(self.escalation_policy, EscalationPolicy):
            raise ValueError("AgentRole.escalation_policy must be EscalationPolicy")
        if not isinstance(self.budget, RoleBudget):
            raise ValueError("AgentRole.budget must be RoleBudget")

        normalized_tools = tuple(
            _validate_non_empty_str(item, "AgentRole.allowed_tools") for item in self.allowed_tools
        )
        if len({item.lower() for item in normalized_tools}) != len(normalized_tools):
            raise ValueError("AgentRole.allowed_tools contains duplicates")
        object.__setattr__(self, "allowed_tools", normalized_tools)

        normalized_caps = tuple(
            _validate_non_empty_str(item, "AgentRole.capabilities") for item in self.capabilities
        )
        if len({item.lower() for item in normalized_caps}) != len(normalized_caps):
            raise ValueError("AgentRole.capabilities contains duplicates")
        object.__setattr__(self, "capabilities", normalized_caps)
        object.__setattr__(self, "enabled", bool(self.enabled))

    def route_for_attempt(
        self,
        attempt_number: int,
        *,
        budget_override: RoleBudget | None = None,
    ) -> EscalationDecision | None:
        effective_budget = self.budget.merge(budget_override)
        return self.escalation_policy.resolve_attempt(
            attempt_number=attempt_number,
            max_attempts_override=effective_budget.max_attempts,
        )

    def with_overrides(
        self,
        *,
        enabled: bool | None = None,
        budget: RoleBudget | None = None,
    ) -> AgentRole:
        next_enabled = self.enabled if enabled is None else bool(enabled)
        next_budget = self.budget if budget is None else budget
        return replace(self, enabled=next_enabled, budget=next_budget)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "purpose": self.purpose,
            "required_for": self.required_for,
            "prompt_template_path": self.prompt_template_path,
            "allowed_tools": list(self.allowed_tools),
            "capabilities": list(self.capabilities),
            "sandbox_policy": self.sandbox_policy,
            "enabled": self.enabled,
            "budget": self.budget.to_dict(),
            "escalation_policy": self.escalation_policy.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def role_id(self) -> str:
        """Compatibility alias aligned with synthesis-plane contract language."""

        return self.name

    @property
    def description(self) -> str:
        """Compatibility alias aligned with synthesis-plane contract language."""

        return self.purpose

    @property
    def preferred_provider(self) -> str:
        """Return the first provider in the deterministic escalation ladder."""

        return self.escalation_policy.steps[0].provider

    @property
    def preferred_model(self) -> str:
        """Return the first model in the deterministic escalation ladder."""

        return self.escalation_policy.steps[0].model

    @property
    def fallback_chain(self) -> tuple[EscalationStep, ...]:
        """Return fallback chain as explicit ordered escalation steps."""

        return self.escalation_policy.steps

    @property
    def budget_policy(self) -> RoleBudget:
        """Compatibility alias for role-level budget envelope."""

        return self.budget


@dataclass(frozen=True, slots=True)
class RoleRegistry:
    """Immutable role registry with deterministic lookup and risk requirements."""

    roles: tuple[AgentRole, ...]
    risk_tier_requirements: Mapping[RiskTier, tuple[str, ...]] = field(
        default_factory=lambda: _default_risk_tier_requirements()
    )
    _roles_by_key: Mapping[str, AgentRole] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        role_list = tuple(self.roles)
        if not role_list:
            raise ValueError("RoleRegistry.roles cannot be empty")

        role_lookup: dict[str, AgentRole] = {}
        display_name_lookup: set[str] = set()
        for role in role_list:
            if not isinstance(role, AgentRole):
                raise ValueError("RoleRegistry.roles entries must be AgentRole")
            role_key = _normalize_identifier(role.name, "RoleRegistry.role_name")
            if role_key in role_lookup:
                raise ValueError(f"duplicate role name: {role.name}")
            display_key = _normalize_identifier(role.display_name, "RoleRegistry.display_name")
            if display_key in display_name_lookup:
                raise ValueError(f"duplicate role display name: {role.display_name}")
            display_name_lookup.add(display_key)
            role_lookup[role_key] = role

        normalized_requirements: dict[RiskTier, tuple[str, ...]] = {}
        for tier_raw, role_names_raw in self.risk_tier_requirements.items():
            tier = _as_risk_tier(tier_raw)
            if not isinstance(role_names_raw, Sequence) or isinstance(role_names_raw, str):
                raise ValueError(
                    f"risk tier requirement for {tier.value} must be a sequence of names"
                )
            resolved_role_names: list[str] = []
            seen: set[str] = set()
            for item in role_names_raw:
                if not isinstance(item, str):
                    raise ValueError(
                        f"risk tier requirement for {tier.value} must contain only strings"
                    )
                role_key = _normalize_identifier(item, f"{tier.value} role requirement")
                if role_key in seen:
                    raise ValueError(f"risk tier {tier.value} contains duplicate role {item!r}")
                if role_key not in role_lookup:
                    raise ValueError(f"risk tier {tier.value} references unknown role {item!r}")
                seen.add(role_key)
                resolved_role_names.append(role_lookup[role_key].name)
            normalized_requirements[tier] = tuple(resolved_role_names)

        missing_tiers = [tier.value for tier in RiskTier if tier not in normalized_requirements]
        if missing_tiers:
            raise ValueError(f"risk tier requirements missing tiers: {missing_tiers}")

        critical_required = {
            _normalize_identifier(item, "risk_tier_requirements.critical")
            for item in normalized_requirements[RiskTier.CRITICAL]
        }
        missing_critical = sorted(_REQUIRED_CRITICAL_ROLES - critical_required)
        if missing_critical:
            raise ValueError(
                "critical risk tier must include reviewer and security; missing: "
                f"{missing_critical}"
            )

        object.__setattr__(self, "roles", role_list)
        object.__setattr__(self, "_roles_by_key", MappingProxyType(role_lookup))
        object.__setattr__(
            self,
            "risk_tier_requirements",
            MappingProxyType(normalized_requirements),
        )

    @classmethod
    def default(
        cls,
        *,
        model_catalog: ModelCatalog | None = None,
        default_provider: str = "anthropic",
    ) -> RoleRegistry:
        catalog = model_catalog if model_catalog is not None else load_model_catalog()
        profile_models = _resolve_provider_profile_models(config=None, model_catalog=catalog)
        return cls(
            roles=_default_roles(profile_models=profile_models, default_provider=default_provider),
            risk_tier_requirements=_default_risk_tier_requirements(),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> RoleRegistry:
        model_catalog = load_model_catalog()
        # Extract default provider from config for ladder selection
        default_provider = "anthropic"
        if isinstance(config, Mapping):
            providers_raw = config.get("providers")
            if isinstance(providers_raw, Mapping):
                dp = providers_raw.get("default")
                if isinstance(dp, str) and dp:
                    default_provider = dp
        registry = cls.default(model_catalog=model_catalog, default_provider=default_provider)
        if config is None:
            return registry
        if not isinstance(config, Mapping):
            raise ValueError("config must be a mapping")

        profile_models = _resolve_provider_profile_models(
            config=config, model_catalog=model_catalog
        )
        registry = cls(
            roles=_default_roles(profile_models=profile_models, default_provider=default_provider),
            risk_tier_requirements=_default_risk_tier_requirements(),
        )

        role_section_raw = config.get("roles", {})
        if role_section_raw is None:
            role_section: Mapping[str, object] = {}
        elif isinstance(role_section_raw, Mapping):
            role_section = role_section_raw
        else:
            raise ValueError("roles config must be a mapping")

        global_budget_override = _extract_global_budget_override(config.get("budgets"))
        role_budget_overrides = _extract_role_budget_overrides(role_section.get("budgets"))
        if not role_budget_overrides:
            role_budget_overrides = _extract_role_budget_overrides(
                role_section.get("budget_overrides")
            )

        enabled_raw = role_section.get("enabled")
        disabled_raw = role_section.get("disabled")
        enabled_set = _extract_role_name_set(enabled_raw, field_name="roles.enabled")
        disabled_set = _extract_role_name_set(disabled_raw, field_name="roles.disabled")

        known_roles = {role.name for role in registry.roles}
        _assert_known_role_names(enabled_set, known_roles, "roles.enabled")
        _assert_known_role_names(disabled_set, known_roles, "roles.disabled")
        _assert_known_role_names(set(role_budget_overrides), known_roles, "roles.budgets")

        risk_requirements: Mapping[RiskTier, tuple[str, ...]] = registry.risk_tier_requirements
        risk_raw = role_section.get("risk_tier_requirements")
        if risk_raw is not None:
            if not isinstance(risk_raw, Mapping):
                raise ValueError("roles.risk_tier_requirements must be a mapping")
            merged_risk: dict[RiskTier, tuple[str, ...]] = {}
            for tier in RiskTier:
                if tier.value in risk_raw:
                    value = risk_raw[tier.value]
                elif tier in risk_raw:
                    value = risk_raw[tier]
                else:
                    value = risk_requirements[tier]
                if isinstance(value, str):
                    raise ValueError(
                        f"roles.risk_tier_requirements.{tier.value} must be a sequence"
                    )
                if not isinstance(value, Sequence):
                    raise ValueError(
                        f"roles.risk_tier_requirements.{tier.value} must be a sequence"
                    )
                merged_risk[tier] = tuple(str(item) for item in value)
            risk_requirements = merged_risk

        configured_roles: list[AgentRole] = []
        for role in registry.roles:
            role_enabled = role.enabled
            if enabled_set is not None:
                role_enabled = role.name in enabled_set
            if disabled_set and role.name in disabled_set:
                role_enabled = False

            effective_budget = role.budget
            effective_budget = effective_budget.merge(global_budget_override)
            effective_budget = effective_budget.merge(role_budget_overrides.get(role.name))
            configured_roles.append(
                role.with_overrides(enabled=role_enabled, budget=effective_budget)
            )

        return cls(
            roles=tuple(configured_roles),
            risk_tier_requirements=risk_requirements,
        )

    def get(self, role_name: str) -> AgentRole | None:
        return self._roles_by_key.get(_normalize_identifier(role_name, "role_name"))

    def require(self, role_name: str) -> AgentRole:
        role = self.get(role_name)
        if role is None:
            raise KeyError(f"unknown role: {role_name}")
        return role

    def role_names(self) -> tuple[str, ...]:
        return tuple(role.name for role in self.roles)

    def enabled_roles(self) -> tuple[AgentRole, ...]:
        return tuple(role for role in self.roles if role.enabled)

    def required_role_names_for_risk(self, risk_tier: RiskTier | str) -> tuple[str, ...]:
        return tuple(self.risk_tier_requirements[_as_risk_tier(risk_tier)])

    def required_roles_for_risk(self, risk_tier: RiskTier | str) -> tuple[AgentRole, ...]:
        return tuple(self.require(name) for name in self.required_role_names_for_risk(risk_tier))

    def route_attempt(
        self,
        *,
        role_name: str,
        attempt_number: int,
        budget_override: RoleBudget | None = None,
    ) -> EscalationDecision | None:
        role = self.require(role_name)
        return role.route_for_attempt(
            attempt_number=attempt_number, budget_override=budget_override
        )

    def to_dict(self) -> dict[str, JSONValue]:
        risk_payload: dict[str, JSONValue] = {
            tier.value: list(role_names)
            for tier, role_names in sorted(
                self.risk_tier_requirements.items(),
                key=lambda item: item[0].value,
            )
        }
        return {
            "roles": [role.to_dict() for role in self.roles],
            "risk_tier_requirements": risk_payload,
        }

    def required_by_risk_tier(self) -> dict[str, tuple[str, ...]]:
        """
        Return deterministic inverse mapping of role_id -> risk tiers requiring that role.

        This makes role participation constraints explicit for later policy layers.
        """

        tiers_by_role: dict[str, list[str]] = {role.name: [] for role in self.roles}
        for tier in RiskTier:
            for role_name in self.required_role_names_for_risk(tier):
                tiers_by_role[role_name].append(tier.value)
        return {name: tuple(sorted(tiers)) for name, tiers in sorted(tiers_by_role.items())}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def _extract_global_budget_override(raw: object) -> RoleBudget | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("budgets config must be a mapping")
    payload: dict[str, object] = {}
    if "max_iterations" in raw:
        payload["max_attempts"] = raw["max_iterations"]
    if "max_tokens_per_attempt" in raw:
        payload["max_tokens_per_attempt"] = raw["max_tokens_per_attempt"]
    if "max_cost_per_work_item_usd" in raw:
        payload["max_cost_per_work_item_usd"] = raw["max_cost_per_work_item_usd"]
    if not payload:
        return None
    return RoleBudget.from_mapping(payload)


def _extract_role_budget_overrides(raw: object) -> dict[str, RoleBudget]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("roles.budgets must be a mapping")
    overrides: dict[str, RoleBudget] = {}
    for role_name_raw, budget_payload_raw in raw.items():
        if not isinstance(role_name_raw, str):
            raise ValueError("role budget override keys must be role names")
        if not isinstance(budget_payload_raw, Mapping):
            raise ValueError(f"budget override for {role_name_raw!r} must be a mapping")
        role_key = _normalize_identifier(role_name_raw, "roles.budgets role")
        overrides[role_key] = RoleBudget.from_mapping(budget_payload_raw)
    return overrides


def _extract_role_name_set(raw: object, *, field_name: str) -> set[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        raise ValueError(f"{field_name} must be an array of role names")
    if not isinstance(raw, Sequence):
        raise ValueError(f"{field_name} must be an array of role names")
    names: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
        names.add(_normalize_identifier(value, f"{field_name}[{index}]"))
    return names


def _assert_known_role_names(names: set[str] | None, known: set[str], field_name: str) -> None:
    if not names:
        return
    unknown = sorted(name for name in names if name not in known)
    if unknown:
        raise ValueError(f"{field_name} contains unknown roles: {unknown}")


_PROFILE_CODE = "code"
_PROFILE_ARCHITECT = "architect"


def _resolve_provider_profile_models(
    *,
    config: Mapping[str, object] | None,
    model_catalog: ModelCatalog,
) -> Mapping[str, Mapping[str, str]]:
    providers_raw = config.get("providers") if isinstance(config, Mapping) else None
    providers = providers_raw if isinstance(providers_raw, Mapping) else {}

    provider_models: dict[str, Mapping[str, str]] = {}
    for provider_name in ("openai", "anthropic", "local", "tool"):
        provider_cfg_raw = providers.get(provider_name)
        provider_cfg = provider_cfg_raw if isinstance(provider_cfg_raw, Mapping) else {}

        configured_code = provider_cfg.get("model_code")
        configured_architect = provider_cfg.get("model_architect")
        configured_code_model = configured_code if isinstance(configured_code, str) else None
        configured_architect_model = (
            configured_architect if isinstance(configured_architect, str) else None
        )
        try:
            resolved_code = model_catalog.resolve_model_for_profile(
                provider=provider_name,
                capability_profile=_PROFILE_CODE,
                configured_model=configured_code_model,
            )
            resolved_architect = model_catalog.resolve_model_for_profile(
                provider=provider_name,
                capability_profile=_PROFILE_ARCHITECT,
                configured_model=configured_architect_model,
            )
        except KeyError as exc:
            raise ValueError(
                f"unable to resolve provider model profile for {provider_name}: {exc}"
            ) from exc

        provider_models[provider_name] = MappingProxyType(
            {
                _PROFILE_CODE: resolved_code,
                _PROFILE_ARCHITECT: resolved_architect,
            }
        )

    return MappingProxyType(provider_models)


def _profile_model(
    profile_models: Mapping[str, Mapping[str, str]],
    *,
    provider: str,
    capability_profile: str,
) -> str:
    provider_map = profile_models.get(provider)
    if provider_map is None:
        raise ValueError(f"missing profile model mapping for provider {provider!r}")
    model = provider_map.get(capability_profile)
    if model is None:
        raise ValueError(
            f"missing capability profile {capability_profile!r} for provider {provider!r}"
        )
    return model


def _default_codex_ladder(
    *,
    profile_models: Mapping[str, Mapping[str, str]],
) -> EscalationPolicy:
    return EscalationPolicy(
        steps=(
            EscalationStep(
                provider="openai",
                model=_profile_model(
                    profile_models,
                    provider="openai",
                    capability_profile=_PROFILE_CODE,
                ),
                attempts=2,
            ),
            EscalationStep(
                provider="anthropic",
                model=_profile_model(
                    profile_models,
                    provider="anthropic",
                    capability_profile=_PROFILE_CODE,
                ),
                attempts=2,
            ),
            EscalationStep(
                provider="anthropic",
                model=_profile_model(
                    profile_models,
                    provider="anthropic",
                    capability_profile=_PROFILE_ARCHITECT,
                ),
                attempts=1,
            ),
        )
    )


def _default_hybrid_ladder(
    *,
    profile_models: Mapping[str, Mapping[str, str]],
) -> EscalationPolicy:
    return EscalationPolicy(
        steps=(
            EscalationStep(
                provider="openai",
                model=_profile_model(
                    profile_models,
                    provider="openai",
                    capability_profile=_PROFILE_CODE,
                ),
                attempts=1,
            ),
            EscalationStep(
                provider="anthropic",
                model=_profile_model(
                    profile_models,
                    provider="anthropic",
                    capability_profile=_PROFILE_CODE,
                ),
                attempts=2,
            ),
            EscalationStep(
                provider="anthropic",
                model=_profile_model(
                    profile_models,
                    provider="anthropic",
                    capability_profile=_PROFILE_ARCHITECT,
                ),
                attempts=1,
            ),
        )
    )


def _default_claude_ladder(
    *,
    profile_models: Mapping[str, Mapping[str, str]],
) -> EscalationPolicy:
    return EscalationPolicy(
        steps=(
            EscalationStep(
                provider="anthropic",
                model=_profile_model(
                    profile_models,
                    provider="anthropic",
                    capability_profile=_PROFILE_CODE,
                ),
                attempts=2,
            ),
            EscalationStep(
                provider="anthropic",
                model=_profile_model(
                    profile_models,
                    provider="anthropic",
                    capability_profile=_PROFILE_ARCHITECT,
                ),
                attempts=1,
            ),
        )
    )


def _default_tool_ladder(
    *,
    profile_models: Mapping[str, Mapping[str, str]],
) -> EscalationPolicy:
    """Escalation ladder that uses local CLI tool backends (codex, claude)."""
    return EscalationPolicy(
        steps=(
            EscalationStep(
                provider="tool",
                model=_profile_model(
                    profile_models,
                    provider="tool",
                    capability_profile=_PROFILE_CODE,
                ),
                attempts=3,
            ),
            EscalationStep(
                provider="tool",
                model=_profile_model(
                    profile_models,
                    provider="tool",
                    capability_profile=_PROFILE_ARCHITECT,
                ),
                attempts=2,
            ),
        )
    )


def _default_roles(
    *,
    profile_models: Mapping[str, Mapping[str, str]],
    default_provider: str = "anthropic",
) -> tuple[AgentRole, ...]:
    if default_provider == "tool" and "tool" in profile_models:
        tool_ladder = _default_tool_ladder(profile_models=profile_models)
        codex_ladder = tool_ladder
        hybrid_ladder = tool_ladder
        claude_ladder = tool_ladder
    else:
        codex_ladder = _default_codex_ladder(profile_models=profile_models)
        hybrid_ladder = _default_hybrid_ladder(profile_models=profile_models)
        claude_ladder = _default_claude_ladder(profile_models=profile_models)
    return (
        AgentRole(
            name=ROLE_ARCHITECT,
            display_name="Architect",
            purpose="Decompose work, define contracts, and produce ADR-ready plans.",
            required_for="Phase 1 planning",
            prompt_template_path="docs/prompts/templates/ARCHITECT.md",
            escalation_policy=claude_ladder,
            allowed_tools=("rg", "sed", "cat"),
            capabilities=("decomposition", "contract_design", "adr_authoring"),
            sandbox_policy="read_only",
            budget=RoleBudget(max_attempts=claude_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_IMPLEMENTER,
            display_name="Implementer",
            purpose="Write scoped production code with deterministic outputs.",
            required_for="All code work items",
            prompt_template_path="docs/prompts/templates/IMPLEMENTER.md",
            escalation_policy=codex_ladder,
            allowed_tools=("rg", "ruff", "mypy", "pytest"),
            capabilities=("code_generation", "refactoring", "bug_fixing"),
            sandbox_policy="workspace_write",
            budget=RoleBudget(max_attempts=codex_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_TEST_ENGINEER,
            display_name="Test Engineer",
            purpose="Author deterministic unit, integration, and property tests.",
            required_for="Test work items",
            prompt_template_path="docs/prompts/templates/ADVERSARIAL_TESTER.md",
            escalation_policy=hybrid_ladder,
            allowed_tools=("pytest", "hypothesis", "ruff"),
            capabilities=("unit_testing", "integration_testing", "property_testing"),
            sandbox_policy="workspace_write",
            budget=RoleBudget(max_attempts=hybrid_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_REVIEWER,
            display_name="Reviewer",
            purpose="Run adversarial review for correctness and edge-case coverage.",
            required_for="High and critical risk items",
            prompt_template_path="docs/prompts/templates/REVIEWER.md",
            escalation_policy=claude_ladder,
            allowed_tools=("rg", "pytest", "ruff"),
            capabilities=("adversarial_review", "risk_assessment", "regression_hunting"),
            sandbox_policy="read_only",
            budget=RoleBudget(max_attempts=claude_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_SECURITY,
            display_name="Security",
            purpose="Assess threat surface and enforce secure defaults.",
            required_for="Security-tagged items",
            prompt_template_path="docs/prompts/templates/SECURITY.md",
            escalation_policy=claude_ladder,
            allowed_tools=("pip-audit", "gitleaks", "pytest"),
            capabilities=("threat_modeling", "secrets_review", "dependency_auditing"),
            sandbox_policy="restricted_network",
            budget=RoleBudget(max_attempts=claude_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_PERFORMANCE,
            display_name="Performance",
            purpose="Analyze profiling evidence and suggest perf-safe optimizations.",
            required_for="Performance-constrained items",
            prompt_template_path="docs/prompts/templates/PERFORMANCE.md",
            escalation_policy=hybrid_ladder,
            allowed_tools=("pytest", "rg"),
            capabilities=("benchmarking", "profiling", "latency_optimization"),
            sandbox_policy="workspace_write",
            budget=RoleBudget(max_attempts=hybrid_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_TOOLSMITH,
            display_name="Toolsmith",
            purpose="Integrate tools, CI changes, and deterministic automation glue.",
            required_for="Infrastructure items",
            prompt_template_path="docs/prompts/templates/TOOLSMITH.md",
            escalation_policy=codex_ladder,
            allowed_tools=("ruff", "mypy", "pytest", "pip-audit"),
            capabilities=("tooling_integration", "ci_automation", "devx_improvements"),
            sandbox_policy="workspace_write",
            budget=RoleBudget(max_attempts=codex_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_INTEGRATOR,
            display_name="Integrator",
            purpose="Resolve conflicts and ensure cross-change coherence.",
            required_for="Non-trivial conflict resolution",
            prompt_template_path="docs/prompts/templates/INTEGRATOR.md",
            escalation_policy=claude_ladder,
            allowed_tools=("rg", "pytest"),
            capabilities=("merge_resolution", "coherence_validation"),
            sandbox_policy="workspace_write",
            budget=RoleBudget(max_attempts=claude_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_CONSTRAINT_MINER,
            display_name="Constraint Miner",
            purpose="Extract stable constraints from failures for future prevention.",
            required_for="Post-failure analysis",
            prompt_template_path="docs/prompts/templates/CONSTRAINT_MINER.md",
            escalation_policy=claude_ladder,
            allowed_tools=("rg", "pytest"),
            capabilities=("failure_analysis", "constraint_extraction"),
            sandbox_policy="read_only",
            budget=RoleBudget(max_attempts=claude_ladder.max_attempts),
        ),
        AgentRole(
            name=ROLE_DOCUMENTATION,
            display_name="Documentation",
            purpose="Produce clear docs, examples, and API guidance.",
            required_for="Doc-tagged items",
            prompt_template_path="docs/prompts/templates/DOCUMENTATION.md",
            escalation_policy=hybrid_ladder,
            allowed_tools=("rg", "pytest"),
            capabilities=("api_docs", "guides", "examples"),
            sandbox_policy="workspace_write",
            budget=RoleBudget(max_attempts=hybrid_ladder.max_attempts),
        ),
    )


def _default_risk_tier_requirements() -> Mapping[RiskTier, tuple[str, ...]]:
    return MappingProxyType(
        {
            RiskTier.LOW: (ROLE_IMPLEMENTER,),
            RiskTier.MEDIUM: (ROLE_IMPLEMENTER, ROLE_TEST_ENGINEER),
            RiskTier.HIGH: (ROLE_IMPLEMENTER, ROLE_TEST_ENGINEER, ROLE_REVIEWER),
            RiskTier.CRITICAL: (
                ROLE_IMPLEMENTER,
                ROLE_TEST_ENGINEER,
                ROLE_REVIEWER,
                ROLE_SECURITY,
            ),
        }
    )


DEFAULT_ROLE_REGISTRY = RoleRegistry.default()

__all__ = [
    "AgentRole",
    "DEFAULT_ROLE_REGISTRY",
    "EscalationDecision",
    "EscalationPolicy",
    "EscalationStep",
    "ROLE_ARCHITECT",
    "ROLE_CONSTRAINT_MINER",
    "ROLE_DOCUMENTATION",
    "ROLE_IMPLEMENTER",
    "ROLE_INTEGRATOR",
    "ROLE_PERFORMANCE",
    "ROLE_REVIEWER",
    "ROLE_SECURITY",
    "ROLE_TEST_ENGINEER",
    "ROLE_TOOLSMITH",
    "RoleBudget",
    "RoleRegistry",
]
