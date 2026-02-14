"""
nexus-orchestrator — test skeleton

File: tests/unit/synthesis_plane/test_roles.py
Last updated: 2026-02-11

Purpose
- Validate role system definitions and routing metadata.

What this test file should cover
- Role registry completeness.
- Per-role budget overrides.
- Risk-tier role requirements.

Functional requirements
- No provider calls.

Non-functional requirements
- Deterministic.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from nexus_orchestrator.domain.models import RiskTier
from nexus_orchestrator.synthesis_plane.roles import (
    ROLE_ARCHITECT,
    ROLE_CONSTRAINT_MINER,
    ROLE_DOCUMENTATION,
    ROLE_IMPLEMENTER,
    ROLE_INTEGRATOR,
    ROLE_PERFORMANCE,
    ROLE_REVIEWER,
    ROLE_SECURITY,
    ROLE_TEST_ENGINEER,
    ROLE_TOOLSMITH,
    RoleBudget,
    RoleRegistry,
)


def test_default_registry_contains_ten_documented_roles() -> None:
    registry = RoleRegistry.default()

    assert registry.role_names() == (
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
    assert len(registry.enabled_roles()) == 10


def test_role_registry_rejects_duplicate_role_names() -> None:
    registry = RoleRegistry.default()
    implementer = registry.require(ROLE_IMPLEMENTER)

    with pytest.raises(ValueError, match="duplicate role name"):
        RoleRegistry(
            roles=(implementer, implementer),
            risk_tier_requirements=registry.risk_tier_requirements,
        )


def test_implementer_escalation_ladder_is_deterministic_and_bounded() -> None:
    registry = RoleRegistry.default()
    decisions = [
        registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=index)
        for index in range(1, 7)
    ]
    repeated = [
        registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=index)
        for index in range(1, 7)
    ]

    assert decisions[0] is not None
    assert decisions[0].provider == "openai"
    assert decisions[0].model == "gpt-5-codex"
    assert decisions[1] is not None
    assert decisions[1].provider == "openai"
    assert decisions[1].model == "gpt-5-codex"
    assert decisions[2] is not None
    assert decisions[2].provider == "anthropic"
    assert decisions[2].model == "claude-sonnet-4-5"
    assert decisions[3] is not None
    assert decisions[3].provider == "anthropic"
    assert decisions[3].model == "claude-sonnet-4-5"
    assert decisions[4] is not None
    assert decisions[4].provider == "anthropic"
    assert decisions[4].model == "claude-opus-4-6"
    assert decisions[5] is None
    assert [item.to_dict() if item else None for item in decisions] == [
        item.to_dict() if item else None for item in repeated
    ]


def test_provider_model_profiles_from_config_drive_routing_without_code_changes() -> None:
    registry = RoleRegistry.from_config(
        {
            "providers": {
                "openai": {
                    "model_code": "gpt-4.1-mini",
                    "model_architect": "gpt-5",
                },
                "anthropic": {
                    "model_code": "claude-3-7-sonnet",
                    "model_architect": "claude-opus-4-6",
                },
            }
        }
    )

    first = registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=1)
    third = registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=3)
    fifth = registry.route_attempt(role_name=ROLE_IMPLEMENTER, attempt_number=5)

    assert first is not None
    assert first.model == "gpt-4.1-mini"
    assert third is not None
    assert third.model == "claude-3-7-sonnet"
    assert fifth is not None
    assert fifth.model == "claude-opus-4-6"


def test_from_config_rejects_models_not_present_in_catalog() -> None:
    with pytest.raises(ValueError, match="unable to resolve provider model profile for openai"):
        RoleRegistry.from_config(
            {
                "providers": {
                    "openai": {"model_code": "not-a-real-openai-model"},
                }
            }
        )


def test_budget_override_caps_escalation_attempts() -> None:
    registry = RoleRegistry.default()
    override = RoleBudget(max_attempts=3)

    decision_three = registry.route_attempt(
        role_name=ROLE_IMPLEMENTER,
        attempt_number=3,
        budget_override=override,
    )
    decision_four = registry.route_attempt(
        role_name=ROLE_IMPLEMENTER,
        attempt_number=4,
        budget_override=override,
    )

    assert decision_three is not None
    assert decision_three.provider == "anthropic"
    assert decision_three.model == "claude-sonnet-4-5"
    assert decision_four is None


def test_risk_tier_mapping_requires_security_and_reviewer_for_critical() -> None:
    registry = RoleRegistry.default()
    required = set(registry.required_role_names_for_risk(RiskTier.CRITICAL))

    assert ROLE_REVIEWER in required
    assert ROLE_SECURITY in required

    invalid_requirements = dict(registry.risk_tier_requirements)
    invalid_requirements[RiskTier.CRITICAL] = (ROLE_IMPLEMENTER, ROLE_REVIEWER)
    with pytest.raises(ValueError, match="critical risk tier must include reviewer and security"):
        RoleRegistry(roles=registry.roles, risk_tier_requirements=invalid_requirements)


def test_from_config_applies_enabled_flags_and_budget_overrides() -> None:
    registry = RoleRegistry.from_config(
        {
            "budgets": {
                "max_iterations": 4,
                "max_tokens_per_attempt": 16000,
            },
            "roles": {
                "enabled": [ROLE_IMPLEMENTER, ROLE_REVIEWER, ROLE_SECURITY],
                "disabled": [ROLE_SECURITY],
                "budgets": {
                    ROLE_IMPLEMENTER: {"max_attempts": 2},
                },
            },
        }
    )

    assert registry.require(ROLE_IMPLEMENTER).enabled is True
    assert registry.require(ROLE_REVIEWER).enabled is True
    assert registry.require(ROLE_SECURITY).enabled is False
    assert registry.require(ROLE_ARCHITECT).enabled is False
    assert registry.require(ROLE_IMPLEMENTER).budget.max_attempts == 2
    assert registry.require(ROLE_REVIEWER).budget.max_attempts == 4
    assert registry.require(ROLE_REVIEWER).budget.max_tokens_per_attempt == 16000


def test_from_config_rejects_unknown_role_names() -> None:
    with pytest.raises(ValueError, match="roles.enabled contains unknown roles"):
        RoleRegistry.from_config({"roles": {"enabled": ["nonexistent"]}})


def test_roles_and_budgets_are_immutable() -> None:
    registry = RoleRegistry.default()
    role = registry.require(ROLE_IMPLEMENTER)

    with pytest.raises(FrozenInstanceError):
        role.enabled = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        role.budget.max_attempts = 99  # type: ignore[misc]
