"""Unit tests for deterministic operating mode resolution and settings."""

from __future__ import annotations

import pytest

from nexus_orchestrator.modes import (
    ALL_ROLE_NAMES,
    DEFAULT_MODE_REGISTRY,
    MODE_BROWNFIELD,
    MODE_EXPLORATION,
    MODE_GREENFIELD,
    MODE_HARDENING,
    ROLE_ARCHITECT,
    ROLE_PERFORMANCE,
    ROLE_SECURITY,
    ROLE_TEST_ENGINEER,
    resolve_mode,
)


def test_registry_default_mode_is_greenfield() -> None:
    registry = DEFAULT_MODE_REGISTRY

    assert registry.default_mode_name == MODE_GREENFIELD
    assert registry.mode_names == (
        MODE_GREENFIELD,
        MODE_BROWNFIELD,
        MODE_HARDENING,
        MODE_EXPLORATION,
    )


def test_resolve_uses_default_for_empty_and_alias_inputs() -> None:
    baseline = resolve_mode(MODE_GREENFIELD)

    assert resolve_mode(None) is baseline
    assert resolve_mode("") is baseline
    assert resolve_mode("   ") is baseline
    assert resolve_mode("default") is baseline
    assert resolve_mode(" GREENFIELD ") is baseline


def test_unknown_mode_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        resolve_mode("nonexistent")


def test_mode_overlays_are_deterministic_and_different() -> None:
    greenfield = resolve_mode(MODE_GREENFIELD)
    brownfield = resolve_mode(MODE_BROWNFIELD)
    hardening = resolve_mode(MODE_HARDENING)
    exploration = resolve_mode(MODE_EXPLORATION)

    assert greenfield.overlay.to_dict() == resolve_mode(MODE_GREENFIELD).overlay.to_dict()
    assert hardening.overlay.to_dict() == resolve_mode(MODE_HARDENING).overlay.to_dict()

    assert greenfield.overlay.to_dict() != brownfield.overlay.to_dict()
    assert hardening.overlay.to_dict() != exploration.overlay.to_dict()


def test_hardening_and_exploration_budget_differences_match_requirements() -> None:
    greenfield_budgets = resolve_mode(MODE_GREENFIELD).overlay.to_dict()["budgets"]
    hardening_budgets = resolve_mode(MODE_HARDENING).overlay.to_dict()["budgets"]
    exploration_budgets = resolve_mode(MODE_EXPLORATION).overlay.to_dict()["budgets"]

    assert isinstance(greenfield_budgets, dict)
    assert isinstance(hardening_budgets, dict)
    assert isinstance(exploration_budgets, dict)

    assert hardening_budgets["max_iterations"] < greenfield_budgets["max_iterations"]
    assert exploration_budgets["max_iterations"] > greenfield_budgets["max_iterations"]


def test_hardening_prioritizes_test_security_performance_roles() -> None:
    hardening = resolve_mode(MODE_HARDENING)
    enabled = hardening.settings.roles.enabled_roles
    priority = hardening.settings.roles.priority

    assert priority[:3] == (
        ROLE_TEST_ENGINEER,
        ROLE_SECURITY,
        ROLE_PERFORMANCE,
    )
    assert ROLE_ARCHITECT not in enabled


def test_brownfield_and_hardening_have_role_enablement_differences() -> None:
    brownfield = resolve_mode(MODE_BROWNFIELD)
    hardening = resolve_mode(MODE_HARDENING)

    brownfield_enabled = set(brownfield.settings.roles.enabled_roles)
    hardening_enabled = set(hardening.settings.roles.enabled_roles)

    assert brownfield_enabled != hardening_enabled
    assert hardening_enabled < set(ALL_ROLE_NAMES)


def test_mode_scheduling_priorities_and_speculation_differ() -> None:
    brownfield = resolve_mode(MODE_BROWNFIELD).settings.scheduling
    hardening = resolve_mode(MODE_HARDENING).settings.scheduling
    exploration = resolve_mode(MODE_EXPLORATION).settings.scheduling

    assert "regression_first" in brownfield.priorities
    assert "security_first" in hardening.priorities
    assert "candidate_diversity" in exploration.priorities

    assert exploration.allow_speculative_execution is True
    assert hardening.allow_speculative_execution is False
    assert exploration.max_dispatch_per_tick > hardening.max_dispatch_per_tick
