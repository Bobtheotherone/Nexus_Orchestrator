"""Exploration mode defaults for speculative, higher-budget search."""

from __future__ import annotations

from nexus_orchestrator.modes import (
    MODE_EXPLORATION,
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

EXPLORATION_MODE = OperatingMode(
    name=MODE_EXPLORATION,
    description="Search mode for ambiguous tasks using speculative candidate execution.",
    overlay=ModeOverlay(
        config={
            "budgets": {
                "max_iterations": 8,
                "max_tokens_per_attempt": 48_000,
                "max_cost_per_work_item_usd": 4.0,
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
                ROLE_TEST_ENGINEER,
                ROLE_REVIEWER,
                ROLE_INTEGRATOR,
                ROLE_CONSTRAINT_MINER,
                ROLE_SECURITY,
                ROLE_PERFORMANCE,
                ROLE_DOCUMENTATION,
            ),
        ),
        scheduling=ModeSchedulingSettings(
            priorities=(
                "candidate_diversity",
                "critical_path",
                "dependency_readiness",
                "anti_starvation",
                "speculative_expansion",
            ),
            allow_speculative_execution=True,
            max_dispatch_per_tick=10,
            max_in_flight=12,
            risk_tier_priority={
                "critical": 0,
                "high": 1,
                "medium": 2,
                "low": 3,
            },
            per_risk_tier_in_flight={
                "critical": 3,
                "high": 4,
                "medium": 5,
                "low": 6,
            },
        ),
    ),
)

__all__ = ["EXPLORATION_MODE"]
