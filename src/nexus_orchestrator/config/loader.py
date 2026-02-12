"""
nexus-orchestrator — runtime config loader.

File: src/nexus_orchestrator/config/loader.py
Last updated: 2026-02-12

Purpose
- Load effective runtime config from defaults, TOML file, env vars, and CLI overrides.

What should be included in this file
- Precedence logic: CLI > env (NEXUS_) > file > defaults.
- TOML loading via ``tomllib``.
- Deterministic environment variable mapping and coercion.
- Path normalization relative to config file location.
- Redacted deterministic dump of effective config.

Functional requirements
- Reject invalid/embedded-secret config via schema validation.
- Support profile overlays selected by CLI/env.

Non-functional requirements
- Keep loading deterministic and reproducible.
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from nexus_orchestrator.config.schema import (
    PATH_FIELDS,
    apply_profile_overlay,
    assert_valid_config,
    default_config,
    dump_redacted,
    merge_config,
)

DEFAULT_CONFIG_FILE: Final[str] = "orchestrator.toml"
ENV_PREFIX: Final[str] = "NEXUS_"

_BOOLEAN_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
_BOOLEAN_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n", "off"})


@dataclass(frozen=True, slots=True)
class _Binding:
    path: tuple[str, ...]
    value_type: Literal["str", "int", "float", "bool"]


class ConfigLoadError(ValueError):
    """Raised when config cannot be loaded or overrides cannot be coerced."""


def load_config(
    config_path: str | Path | None = None,
    *,
    profile: str | None = None,
    cli_overrides: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
    require_secret_env_values: bool = False,
) -> dict[str, Any]:
    """Load effective config with deterministic precedence: CLI > env > file > defaults."""

    resolved_path = _resolve_config_path(config_path)
    explicit_path = config_path is not None
    env_map = dict(os.environ if environ is None else environ)
    cli_map = dict(cli_overrides or {})

    file_payload = _load_toml_file(resolved_path, required=explicit_path)
    selected_profile = _resolve_profile(profile=profile, cli_overrides=cli_map, environ=env_map)

    merged = merge_config(default_config(), file_payload)
    merged = assert_valid_config(merged)

    if selected_profile is not None:
        merged = apply_profile_overlay(merged, selected_profile)

    env_overrides = _collect_env_overrides(merged, env_map)
    cli_payload = _materialize_cli_overrides(cli_map)

    merged = merge_config(merged, env_overrides)
    merged = merge_config(merged, cli_payload)
    merged = assert_valid_config(merged, active_profile=selected_profile)

    normalized = normalize_paths(merged, base_dir=resolved_path.parent)
    normalized = assert_valid_config(normalized, active_profile=selected_profile)

    if require_secret_env_values:
        _assert_required_secret_envs_present(normalized, env_map)

    return normalized


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load config from a specific TOML file path."""

    return load_config(path)


def normalize_paths(config: Mapping[str, object], *, base_dir: Path) -> dict[str, Any]:
    """Normalize configured path fields relative to ``base_dir``."""

    materialized = merge_config({}, config)

    for field_path in PATH_FIELDS:
        _normalize_path_field(materialized, field_path, base_dir)

    profiles = materialized.get("profiles")
    if isinstance(profiles, Mapping):
        for profile_name in sorted(profiles):
            overlay = profiles[profile_name]
            if not isinstance(overlay, Mapping):
                continue
            for suffix in PATH_FIELDS:
                if suffix[0] not in overlay:
                    continue
                _normalize_path_field(materialized, ("profiles", profile_name, *suffix), base_dir)

    return materialized


def effective_config(config: Mapping[str, object]) -> dict[str, Any]:
    """Return a redacted effective config representation suitable for logging."""

    return dump_redacted(config)


def dump_effective_config(config: Mapping[str, object]) -> str:
    """Return deterministic JSON dump of redacted effective config."""

    return json.dumps(
        effective_config(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        return (Path.cwd() / DEFAULT_CONFIG_FILE).resolve()
    return Path(config_path).expanduser().resolve()


def _load_toml_file(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise ConfigLoadError(f"config file not found: {path}")
        return {}

    try:
        with path.open("rb") as handle:
            parsed = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigLoadError(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"unable to read config file {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigLoadError(f"config root must be an object: {path}")

    return parsed


def _resolve_profile(
    *,
    profile: str | None,
    cli_overrides: Mapping[str, object],
    environ: Mapping[str, str],
) -> str | None:
    if profile is not None:
        selected = profile.strip()
        return selected or None

    cli_profile = cli_overrides.get("profile")
    if cli_profile is not None:
        if not isinstance(cli_profile, str):
            raise ConfigLoadError("cli override 'profile' must be a string")
        selected = cli_profile.strip()
        return selected or None

    env_profile = environ.get(f"{ENV_PREFIX}PROFILE")
    if env_profile is None:
        return None
    selected = env_profile.strip()
    return selected or None


def _collect_env_overrides(
    config: Mapping[str, object], environ: Mapping[str, str]
) -> dict[str, Any]:
    bindings = _build_bindings(config)
    overrides: dict[str, Any] = {}
    for env_name in sorted(bindings):
        raw = environ.get(env_name)
        if raw is None:
            continue
        binding = bindings[env_name]
        value = _coerce_env(raw, binding.value_type, env_name, binding.path)
        _set_nested(overrides, binding.path, value)
    return overrides


def _build_bindings(config: Mapping[str, object]) -> dict[str, _Binding]:
    bindings: dict[str, _Binding] = {}

    for path, value in _iter_scalar_paths(config):
        if path and path[0] == "profiles":
            continue
        kind = _kind_for_value(value)
        if kind is None:
            continue
        bindings[_env_name_for_path(path)] = _Binding(path=path, value_type=kind)

    optional: tuple[_Binding, ...] = (
        _Binding(("budgets", "max_total_cost_usd"), "float"),
        _Binding(("providers", "openai", "model_code"), "str"),
        _Binding(("providers", "openai", "model_architect"), "str"),
        _Binding(("providers", "openai", "max_concurrent"), "int"),
        _Binding(("providers", "openai", "requests_per_minute"), "int"),
        _Binding(("providers", "anthropic", "model_code"), "str"),
        _Binding(("providers", "anthropic", "model_architect"), "str"),
        _Binding(("providers", "anthropic", "max_concurrent"), "int"),
        _Binding(("providers", "anthropic", "requests_per_minute"), "int"),
        _Binding(("providers", "local", "model_code"), "str"),
        _Binding(("providers", "local", "model_architect"), "str"),
        _Binding(("providers", "local", "max_concurrent"), "int"),
        _Binding(("providers", "local", "requests_per_minute"), "int"),
    )
    for binding in optional:
        bindings.setdefault(_env_name_for_path(binding.path), binding)

    return bindings


def _iter_scalar_paths(
    payload: Mapping[str, object],
    prefix: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], object]]:
    pairs: list[tuple[tuple[str, ...], object]] = []
    for key in sorted(payload):
        value = payload[key]
        path = (*prefix, key)
        if isinstance(value, Mapping):
            pairs.extend(_iter_scalar_paths(value, path))
        else:
            pairs.append((path, value))
    return pairs


def _kind_for_value(value: object) -> Literal["str", "int", "float", "bool"] | None:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def _coerce_env(
    raw: str,
    value_type: Literal["str", "int", "float", "bool"],
    env_name: str,
    path: tuple[str, ...],
) -> object:
    value = raw.strip()
    if value_type == "str":
        return value
    if value_type == "int":
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigLoadError(f"{env_name} -> {'.'.join(path)} must be an integer") from exc
    if value_type == "float":
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigLoadError(f"{env_name} -> {'.'.join(path)} must be a number") from exc

    lowered = value.lower()
    if lowered in _BOOLEAN_TRUE:
        return True
    if lowered in _BOOLEAN_FALSE:
        return False
    raise ConfigLoadError(
        f"{env_name} -> {'.'.join(path)} must be a boolean (true/false/1/0/yes/no/on/off)"
    )


def _materialize_cli_overrides(cli_overrides: Mapping[str, object]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in sorted(cli_overrides):
        if key == "profile":
            continue
        value = cli_overrides[key]
        if "." in key:
            path = tuple(part for part in key.split(".") if part)
            if not path:
                raise ConfigLoadError(f"invalid CLI override key {key!r}")
            _set_nested(payload, path, value)
            continue
        if isinstance(value, Mapping):
            nested: dict[str, Any] = {}
            _merge_mapping(nested, value)
            payload[key] = nested
            continue
        payload[key] = value
    return payload


def _merge_mapping(target: dict[str, Any], source: Mapping[str, object]) -> None:
    for key in sorted(source):
        value = source[key]
        if isinstance(value, Mapping):
            child = target.get(key)
            if not isinstance(child, dict):
                child = {}
                target[key] = child
            _merge_mapping(child, value)
        else:
            target[key] = value


def _set_nested(target: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    cursor = target
    for part in path[:-1]:
        next_node = cursor.get(part)
        if not isinstance(next_node, dict):
            next_node = {}
            cursor[part] = next_node
        cursor = next_node
    cursor[path[-1]] = value


def _get_nested(payload: Mapping[str, object], path: tuple[str, ...]) -> object | None:
    cursor: object = payload
    for part in path:
        if not isinstance(cursor, Mapping):
            return None
        if part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _normalize_path_field(config: dict[str, Any], path: tuple[str, ...], base_dir: Path) -> None:
    value = _get_nested(config, path)
    if not isinstance(value, str):
        return
    _set_nested(config, path, _normalize_one_path(value, base_dir))


def _normalize_one_path(raw: str, base_dir: Path) -> str:
    expanded = os.path.expandvars(raw)
    candidate = Path(expanded).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    normalized = Path(os.path.normpath(str(candidate)))
    return normalized.as_posix()


def _assert_required_secret_envs_present(
    config: Mapping[str, object], environ: Mapping[str, str]
) -> None:
    providers = config.get("providers")
    if not isinstance(providers, Mapping):
        return

    missing: list[str] = []
    for provider_name in ("openai", "anthropic", "local"):
        provider = providers.get(provider_name)
        if not isinstance(provider, Mapping):
            continue
        env_name = provider.get("api_key_env")
        if not isinstance(env_name, str):
            continue
        if env_name not in environ or not environ[env_name].strip():
            missing.append(f"providers.{provider_name}.api_key_env -> {env_name}")

    if missing:
        details = ", ".join(sorted(missing))
        raise ConfigLoadError(f"missing required secret environment variable values: {details}")


def _env_name_for_path(path: tuple[str, ...]) -> str:
    return ENV_PREFIX + "_".join(part.upper() for part in path)


__all__ = [
    "ConfigLoadError",
    "DEFAULT_CONFIG_FILE",
    "ENV_PREFIX",
    "dump_effective_config",
    "effective_config",
    "load_config",
    "load_config_file",
    "normalize_paths",
]
