"""Integration tests for mode overlays applied to planning/routing config."""

from __future__ import annotations

from nexus_orchestrator.config.schema import default_config, merge_config
from nexus_orchestrator.modes import MODE_EXPLORATION, MODE_HARDENING, resolve_mode
from nexus_orchestrator.synthesis_plane.roles import (
    ROLE_ARCHITECT,
    ROLE_PERFORMANCE,
    ROLE_SECURITY,
    ROLE_TEST_ENGINEER,
    RoleRegistry,
)


def _effective_config_for_mode(mode_name: str) -> dict[str, object]:
    base = default_config()
    mode = resolve_mode(mode_name)
    with_mode_overlay = merge_config(base, mode.overlay.to_dict())
    role_overlay = {
        "roles": {
            "enabled": list(mode.settings.roles.enabled_roles),
            "disabled": list(mode.settings.roles.disabled_roles()),
        }
    }
    return merge_config(with_mode_overlay, role_overlay)


def test_mode_overlays_produce_different_budget_profiles() -> None:
    hardening_config = _effective_config_for_mode(MODE_HARDENING)
    exploration_config = _effective_config_for_mode(MODE_EXPLORATION)

    hardening_iterations = hardening_config["budgets"]["max_iterations"]  # type: ignore[index]
    exploration_iterations = exploration_config["budgets"]["max_iterations"]  # type: ignore[index]
    assert isinstance(hardening_iterations, int)
    assert isinstance(exploration_iterations, int)
    assert hardening_iterations < exploration_iterations


def test_mode_role_settings_change_role_registry_enablement() -> None:
    hardening_registry = RoleRegistry.from_config(_effective_config_for_mode(MODE_HARDENING))
    exploration_registry = RoleRegistry.from_config(_effective_config_for_mode(MODE_EXPLORATION))

    assert hardening_registry.require(ROLE_ARCHITECT).enabled is False
    assert hardening_registry.require(ROLE_TEST_ENGINEER).enabled is True
    assert hardening_registry.require(ROLE_SECURITY).enabled is True
    assert hardening_registry.require(ROLE_PERFORMANCE).enabled is True

    assert exploration_registry.require(ROLE_ARCHITECT).enabled is True
    assert exploration_registry.require(ROLE_TEST_ENGINEER).enabled is True
