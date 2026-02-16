"""
nexus-orchestrator — unit tests for config loader

File: tests/unit/config/test_loader.py
Last updated: 2026-02-12

Purpose
- Validate deterministic config loading from defaults, TOML, env overrides, and CLI overrides.

What this test file should cover
- Precedence: CLI > env > file > defaults.
- Deterministic env var path mapping and type coercion.
- Path normalization relative to the config file.
- Redacted effective config dumping and offline behavior.

Functional requirements
- Works without provider keys or network.

Non-functional requirements
- Deterministic output across repeated loads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nexus_orchestrator.config.loader import (
    ConfigLoadError,
    dump_effective_config,
    load_config,
)
from nexus_orchestrator.synthesis_plane.roles import ROLE_IMPLEMENTER, RoleRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_config(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256_json(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_loader_precedence_default_file_env_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    default_path = tmp_path / "default.toml"
    _write_config(default_path, "")
    _write_config(
        config_path,
        """
[budgets]
max_iterations = 4
""".strip(),
    )

    default_loaded = load_config(default_path)
    file_loaded = load_config(config_path)
    env_loaded = load_config(config_path, environ={"NEXUS_BUDGETS_MAX_ITERATIONS": "6"})
    cli_loaded = load_config(
        config_path,
        environ={"NEXUS_BUDGETS_MAX_ITERATIONS": "6"},
        cli_overrides={"budgets.max_iterations": 7},
    )

    assert default_loaded["budgets"]["max_iterations"] == 5
    assert file_loaded["budgets"]["max_iterations"] == 4
    assert env_loaded["budgets"]["max_iterations"] == 6
    assert cli_loaded["budgets"]["max_iterations"] == 7


def test_env_mapping_supports_budgets_and_nested_provider_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    loaded = load_config(
        config_path,
        environ={
            "NEXUS_BUDGETS_MAX_ITERATIONS": "3",
            "NEXUS_PROVIDERS_OPENAI_MAX_CONCURRENT": "20",
        },
    )

    assert loaded["budgets"]["max_iterations"] == 3
    assert loaded["providers"]["openai"]["max_concurrent"] == 20


def test_invalid_env_coercion_raises_actionable_error(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    with pytest.raises(ConfigLoadError, match="NEXUS_BUDGETS_MAX_ITERATIONS"):
        load_config(config_path, environ={"NEXUS_BUDGETS_MAX_ITERATIONS": "not-an-int"})


def test_loader_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    env = {
        "NEXUS_BUDGETS_MAX_ITERATIONS": "6",
        "NEXUS_SANDBOX_REQUIRE_TOOL_VULN_SCAN": "false",
    }
    cli = {"budgets.max_cost_per_work_item_usd": 3.5}

    first = load_config(config_path, environ=env, cli_overrides=cli)
    second = load_config(config_path, environ=env, cli_overrides=cli)

    assert _sha256_json(first) == _sha256_json(second)


def test_path_normalization_is_relative_to_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "nested" / "orchestrator.toml"
    _write_config(
        config_path,
        """
[paths]
workspace_root = "workspaces"
""".strip(),
    )

    loaded = load_config(config_path)

    assert loaded["paths"]["workspace_root"] == (config_path.parent / "workspaces").as_posix()
    assert loaded["paths"]["state_db"] == (config_path.parent / "state/nexus.sqlite").as_posix()


def test_offline_provider_env_reference_does_not_crash_by_default(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    loaded = load_config(config_path, environ={})

    assert loaded["providers"]["openai"]["api_key_env"] == "NEXUS_OPENAI_API_KEY"


def test_strict_secret_env_mode_requires_secret_values(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    with pytest.raises(
        ConfigLoadError, match="missing required secret environment variable values"
    ):
        load_config(config_path, environ={}, require_secret_env_values=True)


def test_dump_effective_config_is_redacted_and_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    loaded = load_config(config_path)
    first = dump_effective_config(loaded)
    second = dump_effective_config(loaded)

    assert first == second
    parsed = json.loads(first)
    assert parsed["providers"]["openai"]["api_key_env"] == "<redacted>"


def test_can_load_repo_orchestrator_toml_with_profile_and_env_override() -> None:
    loaded = load_config(
        REPO_ROOT / "orchestrator.toml",
        profile="strict",
        environ={"NEXUS_BUDGETS_MAX_ITERATIONS": "9"},
    )

    assert loaded["budgets"]["max_iterations"] == 9


def test_config_model_overrides_affect_role_routing_without_code_edits(tmp_path: Path) -> None:
    config_path = tmp_path / "orchestrator.toml"
    _write_config(
        config_path,
        """
[providers]
default = "openai"

[providers.openai]
model_code = "gpt-4.1-mini"
model_architect = "gpt-5"

[providers.anthropic]
model_code = "claude-3-7-sonnet"
model_architect = "claude-opus-4-6"
""".strip(),
    )

    loaded = load_config(config_path)
    registry = RoleRegistry.from_config(loaded)
    first = registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=1)
    third = registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=3)

    assert first is not None
    assert first.model == "gpt-4.1-mini"
    assert third is not None
    assert third.model == "claude-3-7-sonnet"


def test_config_package_exports_loader_and_errors(tmp_path: Path) -> None:
    import nexus_orchestrator.config as config_pkg

    config_path = tmp_path / "orchestrator.toml"
    _write_config(config_path, "")

    loaded = config_pkg.load_config(config_path)
    assert loaded["meta"]["schema_version"] == 1

    with pytest.raises(config_pkg.ConfigLoadError):
        config_pkg.load_config(tmp_path / "missing.toml")
