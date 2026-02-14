"""
nexus-orchestrator — unit tests for config schema validation

File: tests/unit/config/test_schema.py
Last updated: 2026-02-12

Purpose
- Validate strict config schema behavior, structured errors, profile overlays, and redaction.

What this test file should cover
- Validates the repository's live orchestrator.toml successfully.
- Rejects unknown keys and invalid types with actionable paths.
- Rejects embedded secrets while accepting env-var references.
- Ensures redaction is recursive and non-destructive.

Functional requirements
- No network usage.

Non-functional requirements
- Deterministic and fast.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path

from nexus_orchestrator.config.schema import (
    ConfigSchemaVersion,
    apply_profile_overlay,
    dump_redacted,
    merge_config,
    validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert isinstance(data, dict)
    return data


def _as_object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    normalized: dict[str, object] = {}
    for key, item in value.items():
        assert isinstance(key, str)
        normalized[key] = item
    return normalized


def test_orchestrator_toml_validates_successfully() -> None:
    config = _load_toml(REPO_ROOT / "orchestrator.toml")

    result = validate_config(config)

    assert result.is_valid
    assert result.config is not None
    assert result.config["meta"]["schema_version"] == ConfigSchemaVersion


def test_unknown_key_rejection_is_explicit() -> None:
    config = _load_toml(REPO_ROOT / "orchestrator.toml")
    providers = _as_object_dict(config["providers"])
    providers["unknown_provider"] = {"api_key_env": "NEXUS_UNKNOWN_KEY"}
    config["providers"] = providers

    result = validate_config(config)

    assert not result.is_valid
    assert any(issue.path == "providers.unknown_provider" for issue in result.issues)


def test_type_validation_reports_structured_paths() -> None:
    config = _load_toml(REPO_ROOT / "orchestrator.toml")
    budgets = _as_object_dict(config["budgets"])
    budgets["max_iterations"] = "three"
    config["budgets"] = budgets

    result = validate_config(config)

    assert not result.is_valid
    messages = {issue.path: issue.message for issue in result.issues}
    assert "budgets.max_iterations" in messages
    assert "expected integer" in messages["budgets.max_iterations"]


def test_provider_model_fields_are_validated_against_model_catalog() -> None:
    config = _load_toml(REPO_ROOT / "orchestrator.toml")
    providers = _as_object_dict(config["providers"])
    openai = _as_object_dict(providers["openai"])
    openai["model_code"] = "nonexistent-openai-model"
    providers["openai"] = openai
    config["providers"] = providers

    result = validate_config(config)

    assert not result.is_valid
    issues = {issue.path: issue.message for issue in result.issues}
    assert "providers.openai.model_code" in issues
    assert "not present in model catalog" in issues["providers.openai.model_code"]


def test_range_violation_reports_exact_path() -> None:
    config = _load_toml(REPO_ROOT / "orchestrator.toml")
    budgets = _as_object_dict(config["budgets"])
    budgets["max_iterations"] = 0
    config["budgets"] = budgets

    result = validate_config(config)

    assert not result.is_valid
    issues = {issue.path: issue.message for issue in result.issues}
    assert issues["budgets.max_iterations"] == "must be >= 1"


def test_embedded_secret_is_rejected_but_api_key_env_is_allowed(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.toml"
    bad_path.write_text(
        """
[meta]
schema_version = 1

[providers]
default = "openai"

[providers.openai]
api_key = "sk-THISISFAKE123456789012"

[providers.anthropic]
api_key_env = "NEXUS_ANTHROPIC_API_KEY"

[resources]
orchestrator_cores = 2
max_heavy_verification = 1
max_light_verification = 2
ram_headroom_mb = 1024
disk_min_free_gb = 0
gpu_reserved_for_project = true

[budgets]
max_iterations = 2
max_tokens_per_attempt = 1000
max_cost_per_work_item_usd = 1.0

[git]
main_branch = "main"
integration_branch = "integration"
contract_branch_prefix = "contract/"
work_branch_prefix = "work/"
verify_branch_prefix = "verify/"
auto_resolve_trivial = true

[sandbox]
backend = "docker"
network_policy = "deny"
require_tool_vuln_scan = true

[paths]
workspace_root = "workspaces/"
evidence_root = "evidence/"
state_db = "state/nexus.sqlite"
constraint_registry = "constraints/registry/"
constraint_libraries = "constraints/libraries/"
cache_dir = ".cache/"
tool_registry = "tools/registry.toml"

[observability]
log_level = "INFO"
log_format = "json"
log_dir = "logs/"
redact_secrets = true
evidence_retention_full_runs = 5
evidence_retention_failures_days = 90

[personalization]
enabled = true
profile_path = "profiles/operator_profile.toml"
learning_enabled = true
may_relax_should_constraints = false
""".strip(),
        encoding="utf-8",
    )
    bad_config = _load_toml(bad_path)

    bad_result = validate_config(bad_config)

    assert not bad_result.is_valid
    assert any(issue.path == "providers.openai.api_key" for issue in bad_result.issues)

    good_config = _load_toml(REPO_ROOT / "orchestrator.toml")
    good_result = validate_config(good_config)

    assert good_result.is_valid


def test_profile_overlay_deep_merges_known_sections_and_revalidates() -> None:
    config = _load_toml(REPO_ROOT / "orchestrator.toml")
    profiles = _as_object_dict(config["profiles"])
    strict_overlay = _as_object_dict(profiles["strict"])
    strict_overlay["budgets"] = {"max_iterations": 3}
    profiles["strict"] = strict_overlay
    config["profiles"] = profiles

    merged = apply_profile_overlay(config, "strict")
    result = validate_config(merged, active_profile="strict")

    assert result.is_valid
    assert result.config is not None
    assert result.config["budgets"]["max_iterations"] == 3


def test_dump_redacted_is_recursive_and_preserves_shape() -> None:
    base = _load_toml(REPO_ROOT / "orchestrator.toml")
    merged = merge_config(
        base,
        {
            "providers": {
                "openai": {
                    "nested": {
                        "token": "tok-secret",
                        "safe": "value",
                    },
                },
            },
        },
    )

    redacted = dump_redacted(merged)

    assert redacted["providers"]["openai"]["api_key_env"] == "<redacted>"
    assert redacted["providers"]["openai"]["nested"]["token"] == "<redacted>"
    assert redacted["providers"]["openai"]["nested"]["safe"] == "value"
