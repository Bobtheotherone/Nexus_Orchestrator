"""Brownfield mode defaults for evolving existing repositories safely."""

from __future__ import annotations

from nexus_orchestrator.modes import (
    MODE_BROWNFIELD,
    ROLE_ARCHITECT,
    ROLE_CONSTRAINT_MINER,
    ROLE_DOCUMENTATION,
    ROLE_IMPLEMENTER,
    ROLE_INTEGRATOR,
    ROLE_PERFORMANCE,
    ROLE_REVIEWER,
    ROLE_SECURITY,
    ROLE_TEST_ENGINEER,
    ModeControllerSettings,
    ModeOverlay,
    ModeRoleSettings,
    ModeSchedulingSettings,
    OperatingMode,
)

BROWNFIELD_MODE = OperatingMode(
    name=MODE_BROWNFIELD,
    description="Caution-first mode for iterating inside an existing codebase.",
    overlay=ModeOverlay(
        config={
            "budgets": {
                "max_iterations": 5,
                "max_cost_per_work_item_usd": 2.5,
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
                ROLE_INTEGRATOR,
                ROLE_CONSTRAINT_MINER,
                ROLE_DOCUMENTATION,
            ),
            priority=(
                ROLE_REVIEWER,
                ROLE_TEST_ENGINEER,
                ROLE_IMPLEMENTER,
                ROLE_INTEGRATOR,
                ROLE_ARCHITECT,
                ROLE_SECURITY,
                ROLE_PERFORMANCE,
                ROLE_CONSTRAINT_MINER,
                ROLE_DOCUMENTATION,
            ),
        ),
        scheduling=ModeSchedulingSettings(
            priorities=(
                "regression_first",
                "critical_path",
                "anti_starvation",
                "non_speculative_first",
            ),
            allow_speculative_execution=False,
            max_dispatch_per_tick=4,
            max_in_flight=6,
            risk_tier_priority={
                "critical": 0,
                "high": 1,
                "medium": 2,
                "low": 3,
            },
            per_risk_tier_in_flight={
                "critical": 1,
                "high": 2,
                "medium": 3,
                "low": 2,
            },
        ),
    ),
)

__all__ = ["BROWNFIELD_MODE"]
