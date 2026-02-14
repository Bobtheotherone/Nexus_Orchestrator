"""Greenfield mode defaults for new repository builds."""

from __future__ import annotations

from nexus_orchestrator.modes import (
    MODE_GREENFIELD,
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
    ModeControllerSettings,
    ModeOverlay,
    ModeRoleSettings,
    ModeSchedulingSettings,
    OperatingMode,
)

GREENFIELD_MODE = OperatingMode(
    name=MODE_GREENFIELD,
    description="Default mode for bootstrapping a new codebase.",
    overlay=ModeOverlay(
        config={
            "budgets": {
                "max_iterations": 5,
            },
        }
    ),
    settings=ModeControllerSettings(
        roles=ModeRoleSettings(
            enabled_roles=(
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
            ),
            priority=(
                ROLE_ARCHITECT,
                ROLE_IMPLEMENTER,
                ROLE_TOOLSMITH,
                ROLE_INTEGRATOR,
                ROLE_TEST_ENGINEER,
                ROLE_REVIEWER,
                ROLE_DOCUMENTATION,
                ROLE_CONSTRAINT_MINER,
                ROLE_SECURITY,
                ROLE_PERFORMANCE,
            ),
        ),
        scheduling=ModeSchedulingSettings(
            priorities=(
                "critical_path",
                "dependency_readiness",
                "anti_starvation",
                "non_speculative_first",
            ),
            allow_speculative_execution=False,
            max_dispatch_per_tick=6,
            max_in_flight=8,
            risk_tier_priority={
                "critical": 0,
                "high": 1,
                "medium": 2,
                "low": 3,
            },
            per_risk_tier_in_flight={
                "critical": 2,
                "high": 3,
                "medium": 5,
                "low": 6,
            },
        ),
    ),
)

__all__ = ["GREENFIELD_MODE"]
