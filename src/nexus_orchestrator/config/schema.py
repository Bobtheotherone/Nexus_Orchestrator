"""
nexus-orchestrator — configuration schema and validation.

File: src/nexus_orchestrator/config/schema.py
Last updated: 2026-02-12

Purpose
- Define authoritative configuration defaults and strict validation rules.

What should be included in this file
- Schema versioning and migration guidance.
- Validation rules for required fields, types, enums, and numeric constraints.
- Profile overlay validation and deterministic deep-merge helpers.
- Redaction rules for sensitive fields.

Functional requirements
- Validate config payloads and return structured errors (field path + message).
- Support profile overlays including strict/permissive/hardening/exploration.

Non-functional requirements
- Keep rules deterministic and easy to audit.
- Preserve backwards compatibility through explicit migration messages.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, NotRequired, TypedDict

from nexus_orchestrator.constants import CONFIG_SCHEMA_VERSION

ConfigSchemaVersion: Final[int] = CONFIG_SCHEMA_VERSION
BUILTIN_PROFILE_NAMES: Final[tuple[str, ...]] = (
    "strict",
    "permissive",
    "hardening",
    "exploration",
)

_ENV_NAME_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PROFILE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_CAMEL_CASE_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

_SENSITIVE_KEY_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "token",
        "password",
        "passwd",
        "api",
        "key",
        "apikey",
        "private",
        "credential",
        "credentials",
        "auth",
    }
)
_SENSITIVE_KEY_PHRASES: Final[tuple[str, ...]] = (
    "api_key",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
    "private_key",
    "password",
    "secret",
)

# Config paths that should be normalized relative to config file location.
PATH_FIELDS: Final[tuple[tuple[str, ...], ...]] = (
    ("paths", "workspace_root"),
    ("paths", "evidence_root"),
    ("paths", "state_db"),
    ("paths", "constraint_registry"),
    ("paths", "constraint_libraries"),
    ("paths", "cache_dir"),
    ("paths", "tool_registry"),
    ("observability", "log_dir"),
    ("personalization", "profile_path"),
)


class MetaConfig(TypedDict):
    schema_version: int


class ProviderSettings(TypedDict, total=False):
    api_key_env: str
    model_code: str
    model_architect: str
    max_concurrent: int
    requests_per_minute: int


class ProvidersConfig(TypedDict):
    default: Literal["openai", "anthropic", "local"]
    openai: ProviderSettings
    anthropic: ProviderSettings
    local: ProviderSettings


class ResourcesConfig(TypedDict):
    orchestrator_cores: int
    max_heavy_verification: int
    max_light_verification: int
    ram_headroom_mb: int
    disk_min_free_gb: int
    gpu_reserved_for_project: bool


class BudgetsConfig(TypedDict):
    max_iterations: int
    max_tokens_per_attempt: int
    max_cost_per_work_item_usd: float
    max_total_cost_usd: NotRequired[float | None]


class GitConfig(TypedDict):
    main_branch: str
    integration_branch: str
    contract_branch_prefix: str
    work_branch_prefix: str
    verify_branch_prefix: str
    auto_resolve_trivial: bool


class SandboxConfig(TypedDict):
    backend: Literal["docker", "podman", "none"]
    network_policy: Literal["deny", "allowlist", "logged_permissive"]
    require_tool_vuln_scan: bool


class PathsConfig(TypedDict):
    workspace_root: str
    evidence_root: str
    state_db: str
    constraint_registry: str
    constraint_libraries: str
    cache_dir: str
    tool_registry: str


class ObservabilityConfig(TypedDict):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    log_format: Literal["json", "text"]
    log_dir: str
    redact_secrets: bool
    evidence_retention_full_runs: int
    evidence_retention_failures_days: int


class PersonalizationConfig(TypedDict):
    enabled: bool
    profile_path: str
    learning_enabled: bool
    may_relax_should_constraints: bool


class ProfileOverlay(TypedDict, total=False):
    providers: dict[str, object]
    resources: dict[str, object]
    budgets: dict[str, object]
    git: dict[str, object]
    sandbox: dict[str, object]
    paths: dict[str, object]
    observability: dict[str, object]
    personalization: dict[str, object]


class OrchestratorConfig(TypedDict):
    meta: MetaConfig
    providers: ProvidersConfig
    resources: ResourcesConfig
    budgets: BudgetsConfig
    git: GitConfig
    sandbox: SandboxConfig
    paths: PathsConfig
    observability: ObservabilityConfig
    personalization: PersonalizationConfig
    profiles: dict[str, ProfileOverlay]


DEFAULT_CONFIG: Final[OrchestratorConfig] = {
    "meta": {
        "schema_version": ConfigSchemaVersion,
    },
    "providers": {
        "default": "anthropic",
        "openai": {
            "api_key_env": "NEXUS_OPENAI_API_KEY",
        },
        "anthropic": {
            "api_key_env": "NEXUS_ANTHROPIC_API_KEY",
        },
        "local": {},
    },
    "resources": {
        "orchestrator_cores": 2,
        "max_heavy_verification": 3,
        "max_light_verification": 8,
        "ram_headroom_mb": 6144,
        "disk_min_free_gb": 100,
        "gpu_reserved_for_project": True,
    },
    "budgets": {
        "max_iterations": 5,
        "max_tokens_per_attempt": 32000,
        "max_cost_per_work_item_usd": 2.00,
        "max_total_cost_usd": None,
    },
    "git": {
        "main_branch": "main",
        "integration_branch": "integration",
        "contract_branch_prefix": "contract/",
        "work_branch_prefix": "work/",
        "verify_branch_prefix": "verify/",
        "auto_resolve_trivial": True,
    },
    "sandbox": {
        "backend": "docker",
        "network_policy": "deny",
        "require_tool_vuln_scan": True,
    },
    "paths": {
        "workspace_root": "workspaces/",
        "evidence_root": "evidence/",
        "state_db": "state/nexus.sqlite",
        "constraint_registry": "constraints/registry/",
        "constraint_libraries": "constraints/libraries/",
        "cache_dir": ".cache/",
        "tool_registry": "tools/registry.toml",
    },
    "observability": {
        "log_level": "INFO",
        "log_format": "json",
        "log_dir": "logs/",
        "redact_secrets": True,
        "evidence_retention_full_runs": 5,
        "evidence_retention_failures_days": 90,
    },
    "personalization": {
        "enabled": True,
        "profile_path": "profiles/operator_profile.toml",
        "learning_enabled": True,
        "may_relax_should_constraints": False,
    },
    "profiles": {
        "strict": {
            "sandbox": {"network_policy": "deny"},
            "budgets": {"max_iterations": 3},
        },
        "permissive": {},
        "hardening": {
            "budgets": {"max_iterations": 3},
        },
        "exploration": {
            "budgets": {"max_iterations": 8},
        },
    },
}


@dataclass(frozen=True, slots=True)
class ConfigValidationIssue:
    """Single structured validation failure."""

    path: str
    message: str


@dataclass(frozen=True, slots=True)
class ConfigValidationResult:
    """Validation result with normalized config when no issues were found."""

    config: dict[str, Any] | None
    issues: tuple[ConfigValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return self.config is not None and not self.issues


class ConfigValidationError(ValueError):
    """Raised when strict config validation fails."""

    def __init__(self, issues: Sequence[ConfigValidationIssue]) -> None:
        self.issues = tuple(issues)
        if not self.issues:
            rendered = "unknown validation failure"
        else:
            rendered = "\n".join(f"- {item.path}: {item.message}" for item in self.issues)
        super().__init__(f"invalid config:\n{rendered}")


class _IssueCollector:
    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[ConfigValidationIssue] = []

    def add(self, path: str, message: str) -> None:
        self._items.append(ConfigValidationIssue(path=path, message=message))

    def items(self) -> tuple[ConfigValidationIssue, ...]:
        return tuple(self._items)

    @property
    def has_issues(self) -> bool:
        return bool(self._items)


def default_config() -> OrchestratorConfig:
    """Return a deep copy of deterministic built-in defaults."""

    return copy.deepcopy(DEFAULT_CONFIG)


def migration_guidance(found_version: int) -> str:
    """Return deterministic migration guidance for schema version mismatch."""

    if found_version < ConfigSchemaVersion:
        return (
            f"schema version {found_version} is older than supported {ConfigSchemaVersion}; "
            "upgrade orchestrator.toml to the current schema"
        )
    if found_version > ConfigSchemaVersion:
        return (
            f"schema version {found_version} is newer than supported {ConfigSchemaVersion}; "
            "upgrade the nexus-orchestrator runtime"
        )
    return "schema version is current"


def merge_config(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, Any]:
    """Deterministically deep-merge ``overlay`` onto ``base``."""

    merged = _deep_copy_mapping(base)
    _merge_into(merged, overlay)
    return merged


def apply_profile_overlay(config: Mapping[str, object], profile: str | None) -> dict[str, Any]:
    """Apply a named profile overlay and re-validate the resulting config."""

    materialized = _deep_copy_mapping(config)
    if profile is None:
        return materialized

    selected = profile.strip()
    if not selected:
        return materialized

    profiles_raw = materialized.get("profiles")
    if not isinstance(profiles_raw, Mapping):
        raise ConfigValidationError(
            (ConfigValidationIssue("profiles", "profiles section is required"),)
        )

    overlay_raw = profiles_raw.get(selected)
    if overlay_raw is None:
        raise ConfigValidationError(
            (ConfigValidationIssue("profiles", f"profile {selected!r} is not defined"),)
        )
    if not isinstance(overlay_raw, Mapping):
        raise ConfigValidationError(
            (
                ConfigValidationIssue(
                    f"profiles.{selected}",
                    "profile overlay must be an object",
                ),
            )
        )

    merged = merge_config(materialized, overlay_raw)
    validated = assert_valid_config(merged, active_profile=selected)
    return validated


def validate_config(
    config: Mapping[str, object] | object,
    *,
    active_profile: str | None = None,
) -> ConfigValidationResult:
    """Validate config and return structured issues with deterministic paths."""

    issues = _IssueCollector()
    root = _as_object(config, "<root>", issues)
    if root is None:
        return ConfigValidationResult(config=None, issues=issues.items())

    normalized = _validate_root(root, "", issues, partial=False)
    if normalized is None:
        return ConfigValidationResult(config=None, issues=issues.items())

    selected_profile = active_profile.strip() if isinstance(active_profile, str) else None
    if selected_profile:
        profiles = normalized.get("profiles")
        if not isinstance(profiles, Mapping):
            issues.add("profiles", "profiles section is required")
        elif selected_profile not in profiles:
            issues.add("profiles", f"profile {selected_profile!r} is not defined")
        else:
            overlay = profiles[selected_profile]
            if isinstance(overlay, Mapping):
                effective = merge_config(normalized, overlay)
                _validate_root(effective, "", issues, partial=False)
            else:
                issues.add(f"profiles.{selected_profile}", "profile overlay must be an object")

    if issues.has_issues:
        return ConfigValidationResult(config=None, issues=issues.items())

    return ConfigValidationResult(config=normalized, issues=issues.items())


def assert_valid_config(
    config: Mapping[str, object] | object,
    *,
    active_profile: str | None = None,
) -> dict[str, Any]:
    """Validate config and raise ``ConfigValidationError`` on failure."""

    result = validate_config(config, active_profile=active_profile)
    if result.config is None:
        raise ConfigValidationError(result.issues)
    return result.config


def redact_config(config: Mapping[str, object] | object) -> dict[str, Any]:
    """Return deterministic redacted representation for logs/evidence."""

    if not isinstance(config, Mapping):
        return {}
    redacted = _redact_value(config, parent_key=None)
    if isinstance(redacted, dict):
        return redacted
    return {}


def dump_redacted(config: Mapping[str, object] | object) -> dict[str, Any]:
    """Alias for schema-level redacted dumps."""

    return redact_config(config)


def _validate_root(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any] | None:
    allowed = {
        "meta",
        "providers",
        "resources",
        "budgets",
        "git",
        "sandbox",
        "paths",
        "observability",
        "personalization",
        "profiles",
    }
    required = {
        "meta",
        "providers",
        "resources",
        "budgets",
        "git",
        "sandbox",
        "paths",
        "observability",
        "personalization",
    }

    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, required, path, issues)

    out: dict[str, Any] = {}

    _section(
        payload,
        key="meta",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_meta(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="providers",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_providers(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="resources",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_resources(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="budgets",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_budgets(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="git",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_git(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="sandbox",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_sandbox(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="paths",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_paths(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="observability",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_observability(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )
    _section(
        payload,
        key="personalization",
        path=path,
        issues=issues,
        validator=lambda section, section_path: _validate_personalization(
            section, section_path, issues, partial=partial
        ),
        out=out,
    )

    profiles_raw = payload.get("profiles")
    if profiles_raw is not None:
        profiles_path = _join(path, "profiles")
        profiles_obj = _as_object(profiles_raw, profiles_path, issues)
        if profiles_obj is not None:
            out["profiles"] = _validate_profiles(profiles_obj, profiles_path, issues)

    _validate_provider_cross_fields(
        out.get("providers"), _join(path, "providers"), issues, partial=partial
    )
    return out


def _section(
    payload: Mapping[str, object],
    *,
    key: str,
    path: str,
    issues: _IssueCollector,
    validator: Callable[[dict[str, object], str], dict[str, Any]],
    out: dict[str, Any],
) -> None:
    raw = payload.get(key)
    if raw is None:
        return
    section_path = _join(path, key)
    section_obj = _as_object(raw, section_path, issues)
    if section_obj is None:
        return
    out[key] = validator(section_obj, section_path)


def _validate_meta(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {"schema_version"}
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, {"schema_version"}, path, issues)

    out: dict[str, Any] = {}
    if "schema_version" in payload:
        parsed = _as_int(
            payload["schema_version"], _join(path, "schema_version"), issues, minimum=1
        )
        if parsed is not None:
            out["schema_version"] = parsed
            if parsed != ConfigSchemaVersion:
                issues.add(_join(path, "schema_version"), migration_guidance(parsed))
    return out


def _validate_providers(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {"default", "openai", "anthropic", "local"}
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, {"default", "openai", "anthropic"}, path, issues)

    out: dict[str, Any] = {}
    if "default" in payload:
        parsed_default = _as_enum(
            payload["default"],
            _join(path, "default"),
            issues,
            allowed_values=("openai", "anthropic", "local"),
        )
        if parsed_default is not None:
            out["default"] = parsed_default

    for provider_name in ("openai", "anthropic", "local"):
        raw = payload.get(provider_name)
        if raw is None:
            continue
        section_path = _join(path, provider_name)
        section = _as_object(raw, section_path, issues)
        if section is None:
            continue
        out[provider_name] = _validate_provider_settings(
            section,
            section_path,
            issues,
            require_api_key=(not partial and provider_name in {"openai", "anthropic"}),
        )
    return out


def _validate_provider_settings(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    require_api_key: bool,
) -> dict[str, Any]:
    allowed = {
        "api_key_env",
        "model_code",
        "model_architect",
        "max_concurrent",
        "requests_per_minute",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if require_api_key:
        _require_keys(payload, {"api_key_env"}, path, issues)

    out: dict[str, Any] = {}

    if "api_key_env" in payload:
        parsed_env = _as_env_name(payload["api_key_env"], _join(path, "api_key_env"), issues)
        if parsed_env is not None:
            out["api_key_env"] = parsed_env

    if "model_code" in payload:
        parsed_model_code = _as_str(payload["model_code"], _join(path, "model_code"), issues)
        if parsed_model_code is not None:
            out["model_code"] = parsed_model_code

    if "model_architect" in payload:
        parsed_model_architect = _as_str(
            payload["model_architect"], _join(path, "model_architect"), issues
        )
        if parsed_model_architect is not None:
            out["model_architect"] = parsed_model_architect

    if "max_concurrent" in payload:
        parsed_max_concurrent = _as_int(
            payload["max_concurrent"], _join(path, "max_concurrent"), issues, minimum=1
        )
        if parsed_max_concurrent is not None:
            out["max_concurrent"] = parsed_max_concurrent

    if "requests_per_minute" in payload:
        parsed_requests_per_minute = _as_int(
            payload["requests_per_minute"],
            _join(path, "requests_per_minute"),
            issues,
            minimum=1,
        )
        if parsed_requests_per_minute is not None:
            out["requests_per_minute"] = parsed_requests_per_minute

    return out


def _validate_resources(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {
        "orchestrator_cores",
        "max_heavy_verification",
        "max_light_verification",
        "ram_headroom_mb",
        "disk_min_free_gb",
        "gpu_reserved_for_project",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}
    minimums: dict[str, int] = {
        "orchestrator_cores": 1,
        "max_heavy_verification": 1,
        "max_light_verification": 1,
        "ram_headroom_mb": 1,
        "disk_min_free_gb": 0,
    }
    for key in sorted(minimums):
        if key in payload:
            parsed = _as_int(payload[key], _join(path, key), issues, minimum=minimums[key])
            if parsed is not None:
                out[key] = parsed

    if "gpu_reserved_for_project" in payload:
        parsed = _as_bool(
            payload["gpu_reserved_for_project"], _join(path, "gpu_reserved_for_project"), issues
        )
        if parsed is not None:
            out["gpu_reserved_for_project"] = parsed

    return out


def _validate_budgets(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {
        "max_iterations",
        "max_tokens_per_attempt",
        "max_cost_per_work_item_usd",
        "max_total_cost_usd",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(
            payload,
            {"max_iterations", "max_tokens_per_attempt", "max_cost_per_work_item_usd"},
            path,
            issues,
        )

    out: dict[str, Any] = {}

    if "max_iterations" in payload:
        parsed_max_iterations = _as_int(
            payload["max_iterations"], _join(path, "max_iterations"), issues, minimum=1
        )
        if parsed_max_iterations is not None:
            out["max_iterations"] = parsed_max_iterations

    if "max_tokens_per_attempt" in payload:
        parsed_max_tokens = _as_int(
            payload["max_tokens_per_attempt"],
            _join(path, "max_tokens_per_attempt"),
            issues,
            minimum=1,
        )
        if parsed_max_tokens is not None:
            out["max_tokens_per_attempt"] = parsed_max_tokens

    if "max_cost_per_work_item_usd" in payload:
        parsed_cost_per_work_item = _as_float(
            payload["max_cost_per_work_item_usd"],
            _join(path, "max_cost_per_work_item_usd"),
            issues,
            minimum=0.0,
        )
        if parsed_cost_per_work_item is not None:
            out["max_cost_per_work_item_usd"] = parsed_cost_per_work_item

    if "max_total_cost_usd" in payload:
        raw = payload["max_total_cost_usd"]
        if raw is None:
            out["max_total_cost_usd"] = None
        else:
            parsed_total_cost = _as_float(
                raw, _join(path, "max_total_cost_usd"), issues, minimum=0.0
            )
            if parsed_total_cost is not None:
                out["max_total_cost_usd"] = parsed_total_cost

    return out


def _validate_git(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {
        "main_branch",
        "integration_branch",
        "contract_branch_prefix",
        "work_branch_prefix",
        "verify_branch_prefix",
        "auto_resolve_trivial",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}
    for key in sorted(
        {
            "main_branch",
            "integration_branch",
            "contract_branch_prefix",
            "work_branch_prefix",
            "verify_branch_prefix",
        }
    ):
        if key in payload:
            parsed = _as_str(payload[key], _join(path, key), issues)
            if parsed is not None:
                out[key] = parsed

    if "auto_resolve_trivial" in payload:
        parsed_auto_resolve = _as_bool(
            payload["auto_resolve_trivial"], _join(path, "auto_resolve_trivial"), issues
        )
        if parsed_auto_resolve is not None:
            out["auto_resolve_trivial"] = parsed_auto_resolve
    return out


def _validate_sandbox(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {"backend", "network_policy", "require_tool_vuln_scan"}
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}

    if "backend" in payload:
        parsed_backend = _as_enum(
            payload["backend"],
            _join(path, "backend"),
            issues,
            allowed_values=("docker", "podman", "none"),
        )
        if parsed_backend is not None:
            out["backend"] = parsed_backend

    if "network_policy" in payload:
        parsed_network_policy = _as_enum(
            payload["network_policy"],
            _join(path, "network_policy"),
            issues,
            allowed_values=("deny", "allowlist", "logged_permissive"),
        )
        if parsed_network_policy is not None:
            out["network_policy"] = parsed_network_policy

    if "require_tool_vuln_scan" in payload:
        parsed_require_scan = _as_bool(
            payload["require_tool_vuln_scan"], _join(path, "require_tool_vuln_scan"), issues
        )
        if parsed_require_scan is not None:
            out["require_tool_vuln_scan"] = parsed_require_scan

    return out


def _validate_paths(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {
        "workspace_root",
        "evidence_root",
        "state_db",
        "constraint_registry",
        "constraint_libraries",
        "cache_dir",
        "tool_registry",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}
    for key in sorted(allowed):
        if key in payload:
            parsed = _as_path_text(payload[key], _join(path, key), issues)
            if parsed is not None:
                out[key] = parsed
    return out


def _validate_observability(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {
        "log_level",
        "log_format",
        "log_dir",
        "redact_secrets",
        "evidence_retention_full_runs",
        "evidence_retention_failures_days",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}

    if "log_level" in payload:
        parsed_log_level = _as_enum(
            payload["log_level"],
            _join(path, "log_level"),
            issues,
            allowed_values=("DEBUG", "INFO", "WARNING", "ERROR"),
        )
        if parsed_log_level is not None:
            out["log_level"] = parsed_log_level

    if "log_format" in payload:
        parsed_log_format = _as_enum(
            payload["log_format"],
            _join(path, "log_format"),
            issues,
            allowed_values=("json", "text"),
        )
        if parsed_log_format is not None:
            out["log_format"] = parsed_log_format

    if "log_dir" in payload:
        parsed_log_dir = _as_path_text(payload["log_dir"], _join(path, "log_dir"), issues)
        if parsed_log_dir is not None:
            out["log_dir"] = parsed_log_dir

    if "redact_secrets" in payload:
        parsed_redact = _as_bool(payload["redact_secrets"], _join(path, "redact_secrets"), issues)
        if parsed_redact is not None:
            out["redact_secrets"] = parsed_redact

    if "evidence_retention_full_runs" in payload:
        parsed_retention_runs = _as_int(
            payload["evidence_retention_full_runs"],
            _join(path, "evidence_retention_full_runs"),
            issues,
            minimum=0,
        )
        if parsed_retention_runs is not None:
            out["evidence_retention_full_runs"] = parsed_retention_runs

    if "evidence_retention_failures_days" in payload:
        parsed_retention_days = _as_int(
            payload["evidence_retention_failures_days"],
            _join(path, "evidence_retention_failures_days"),
            issues,
            minimum=0,
        )
        if parsed_retention_days is not None:
            out["evidence_retention_failures_days"] = parsed_retention_days

    return out


def _validate_personalization(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> dict[str, Any]:
    allowed = {
        "enabled",
        "profile_path",
        "learning_enabled",
        "may_relax_should_constraints",
    }
    _reject_unknown_keys(payload, allowed, path, issues)
    if not partial:
        _require_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}

    if "enabled" in payload:
        parsed_enabled = _as_bool(payload["enabled"], _join(path, "enabled"), issues)
        if parsed_enabled is not None:
            out["enabled"] = parsed_enabled

    if "profile_path" in payload:
        parsed_profile_path = _as_path_text(
            payload["profile_path"], _join(path, "profile_path"), issues
        )
        if parsed_profile_path is not None:
            out["profile_path"] = parsed_profile_path

    if "learning_enabled" in payload:
        parsed_learning = _as_bool(
            payload["learning_enabled"], _join(path, "learning_enabled"), issues
        )
        if parsed_learning is not None:
            out["learning_enabled"] = parsed_learning

    if "may_relax_should_constraints" in payload:
        parsed_relax = _as_bool(
            payload["may_relax_should_constraints"],
            _join(path, "may_relax_should_constraints"),
            issues,
        )
        if parsed_relax is not None:
            out["may_relax_should_constraints"] = parsed_relax

    return out


def _validate_profiles(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for profile_name in sorted(payload):
        profile_path = _join(path, profile_name)
        raw = payload[profile_name]
        if not _PROFILE_NAME_PATTERN.fullmatch(profile_name):
            issues.add(profile_path, "profile name must match ^[a-z][a-z0-9_-]*$")
            continue
        profile_obj = _as_object(raw, profile_path, issues)
        if profile_obj is None:
            continue
        out[profile_name] = _validate_profile_overlay(profile_obj, profile_path, issues)
    return out


def _validate_profile_overlay(
    payload: Mapping[str, object],
    path: str,
    issues: _IssueCollector,
) -> dict[str, Any]:
    allowed = {
        "providers",
        "resources",
        "budgets",
        "git",
        "sandbox",
        "paths",
        "observability",
        "personalization",
    }
    _reject_unknown_keys(payload, allowed, path, issues)

    out: dict[str, Any] = {}
    for section in sorted(allowed):
        raw = payload.get(section)
        if raw is None:
            continue
        section_path = _join(path, section)
        section_obj = _as_object(raw, section_path, issues)
        if section_obj is None:
            continue
        if section == "providers":
            out[section] = _validate_providers(section_obj, section_path, issues, partial=True)
        elif section == "resources":
            out[section] = _validate_resources(section_obj, section_path, issues, partial=True)
        elif section == "budgets":
            out[section] = _validate_budgets(section_obj, section_path, issues, partial=True)
        elif section == "git":
            out[section] = _validate_git(section_obj, section_path, issues, partial=True)
        elif section == "sandbox":
            out[section] = _validate_sandbox(section_obj, section_path, issues, partial=True)
        elif section == "paths":
            out[section] = _validate_paths(section_obj, section_path, issues, partial=True)
        elif section == "observability":
            out[section] = _validate_observability(section_obj, section_path, issues, partial=True)
        elif section == "personalization":
            out[section] = _validate_personalization(
                section_obj, section_path, issues, partial=True
            )

    return out


def _validate_provider_cross_fields(
    providers: object,
    path: str,
    issues: _IssueCollector,
    *,
    partial: bool,
) -> None:
    if not isinstance(providers, Mapping):
        return
    if partial:
        return

    default_provider = providers.get("default")
    if not isinstance(default_provider, str):
        return
    selected = providers.get(default_provider)
    if not isinstance(selected, Mapping):
        issues.add(
            _join(path, "default"), f"default provider {default_provider!r} has no config section"
        )
        return

    if default_provider in {"openai", "anthropic"}:
        env_name = selected.get("api_key_env")
        if not isinstance(env_name, str):
            issues.add(_join(path, default_provider), "provider requires api_key_env")


def _as_object(value: object, path: str, issues: _IssueCollector) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        issues.add(path, f"expected object, got {type(value).__name__}")
        return None
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            issues.add(path, f"object key must be string, got {type(key).__name__}")
            continue
        out[key] = item
    return out


def _as_str(value: object, path: str, issues: _IssueCollector) -> str | None:
    if not isinstance(value, str):
        issues.add(path, f"expected string, got {type(value).__name__}")
        return None
    parsed = value.strip()
    if not parsed:
        issues.add(path, "must not be empty")
        return None
    return parsed


def _as_path_text(value: object, path: str, issues: _IssueCollector) -> str | None:
    parsed = _as_str(value, path, issues)
    if parsed is None:
        return None
    if "\x00" in parsed:
        issues.add(path, "must not contain NUL bytes")
        return None
    return parsed


def _as_env_name(value: object, path: str, issues: _IssueCollector) -> str | None:
    parsed = _as_str(value, path, issues)
    if parsed is None:
        return None
    if not _ENV_NAME_PATTERN.fullmatch(parsed):
        issues.add(path, "must be an env var name (example: NEXUS_OPENAI_API_KEY)")
        return None
    return parsed


def _as_bool(value: object, path: str, issues: _IssueCollector) -> bool | None:
    if isinstance(value, bool):
        return value
    issues.add(path, f"expected boolean, got {type(value).__name__}")
    return None


def _as_int(
    value: object,
    path: str,
    issues: _IssueCollector,
    *,
    minimum: int | None = None,
) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        issues.add(path, f"expected integer, got {type(value).__name__}")
        return None
    if minimum is not None and value < minimum:
        issues.add(path, f"must be >= {minimum}")
        return None
    return value


def _as_float(
    value: object,
    path: str,
    issues: _IssueCollector,
    *,
    minimum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.add(path, f"expected number, got {type(value).__name__}")
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        issues.add(path, "must be finite")
        return None
    if minimum is not None and parsed < minimum:
        issues.add(path, f"must be >= {minimum}")
        return None
    return parsed


def _as_enum(
    value: object,
    path: str,
    issues: _IssueCollector,
    *,
    allowed_values: tuple[str, ...],
) -> str | None:
    parsed = _as_str(value, path, issues)
    if parsed is None:
        return None
    if parsed not in allowed_values:
        expected = ", ".join(sorted(allowed_values))
        issues.add(path, f"invalid value {parsed!r}; expected one of: {expected}")
        return None
    return parsed


def _reject_unknown_keys(
    payload: Mapping[str, object],
    allowed: set[str],
    path: str,
    issues: _IssueCollector,
) -> None:
    for key in sorted(payload):
        if key in allowed:
            continue
        key_path = _join(path, key)
        if _looks_sensitive_key(key):
            issues.add(
                key_path,
                "embedded secret values are forbidden; use an *_env key with an env var name",
            )
        else:
            issues.add(key_path, "unknown field")


def _require_keys(
    payload: Mapping[str, object],
    required: set[str],
    path: str,
    issues: _IssueCollector,
) -> None:
    for key in sorted(required):
        if key not in payload:
            issues.add(_join(path, key), "missing required field")


def _looks_sensitive_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized.endswith("_env"):
        return False
    if any(phrase in normalized for phrase in _SENSITIVE_KEY_PHRASES):
        return True
    tokens = tuple(token for token in normalized.split("_") if token)
    return any(token in _SENSITIVE_KEY_TOKENS for token in tokens)


def _normalize_key(key: str) -> str:
    with_boundaries = _CAMEL_CASE_BOUNDARY.sub(r"\1_\2", key.strip())
    return _NON_ALNUM.sub("_", with_boundaries.lower()).strip("_")


def _join(path: str, key: str) -> str:
    if not path:
        return key
    return f"{path}.{key}"


def _merge_into(target: dict[str, Any], overlay: Mapping[str, object]) -> None:
    for key in sorted(overlay):
        value = overlay[key]
        if isinstance(value, Mapping):
            existing = target.get(key)
            if isinstance(existing, dict):
                _merge_into(existing, value)
            elif isinstance(existing, Mapping):
                nested = _deep_copy_mapping(existing)
                _merge_into(nested, value)
                target[key] = nested
            else:
                nested_new: dict[str, Any] = {}
                _merge_into(nested_new, value)
                target[key] = nested_new
        else:
            target[key] = _deep_copy_value(value)


def _deep_copy_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(value):
        out[key] = _deep_copy_value(value[key])
    return out


def _deep_copy_value(value: object) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                out[key] = _deep_copy_value(item)
        return out
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_deep_copy_value(item) for item in value)
    return copy.deepcopy(value)


def _redact_value(value: object, parent_key: str | None) -> object:
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key in sorted(value):
            item = value[key]
            if _key_is_sensitive_for_redaction(key):
                out[key] = "<redacted>"
            else:
                out[key] = _redact_value(item, key)
        return out
    if isinstance(value, list):
        return [_redact_value(item, parent_key) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, parent_key) for item in value]
    if parent_key is not None and _key_is_sensitive_for_redaction(parent_key):
        return "<redacted>"
    return value


def _key_is_sensitive_for_redaction(key: str) -> bool:
    normalized = _normalize_key(key)
    if normalized.endswith("_env"):
        return True
    return _looks_sensitive_key(normalized)


__all__ = [
    "BUILTIN_PROFILE_NAMES",
    "ConfigSchemaVersion",
    "ConfigValidationError",
    "ConfigValidationIssue",
    "ConfigValidationResult",
    "DEFAULT_CONFIG",
    "PATH_FIELDS",
    "ProfileOverlay",
    "OrchestratorConfig",
    "apply_profile_overlay",
    "assert_valid_config",
    "default_config",
    "dump_redacted",
    "merge_config",
    "migration_guidance",
    "redact_config",
    "validate_config",
]
