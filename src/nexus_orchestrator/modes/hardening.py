"""Hardening mode defaults for quality-focused stabilization cycles."""

from __future__ import annotations

from nexus_orchestrator.modes import (
    MODE_HARDENING,
    ROLE_CONSTRAINT_MINER,
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

HARDENING_MODE = OperatingMode(
    name=MODE_HARDENING,
    description="Quality-first mode prioritizing regression, security, and performance hardening.",
    overlay=ModeOverlay(
        config={
            "budgets": {
                "max_iterations": 3,
                "max_tokens_per_attempt": 24_000,
                "max_cost_per_work_item_usd": 1.5,
            },
        }
    ),
    settings=ModeControllerSettings(
        roles=ModeRoleSettings(
            enabled_roles=(
                ROLE_TEST_ENGINEER,
                ROLE_SECURITY,
                ROLE_PERFORMANCE,
                ROLE_REVIEWER,
                ROLE_CONSTRAINT_MINER,
                ROLE_IMPLEMENTER,
                ROLE_INTEGRATOR,
            ),
            priority=(
                ROLE_TEST_ENGINEER,
                ROLE_SECURITY,
                ROLE_PERFORMANCE,
                ROLE_REVIEWER,
                ROLE_CONSTRAINT_MINER,
                ROLE_IMPLEMENTER,
                ROLE_INTEGRATOR,
            ),
        ),
        scheduling=ModeSchedulingSettings(
            priorities=(
                "regression_first",
                "security_first",
                "performance_budget_guard",
                "critical_path",
                "anti_starvation",
            ),
            allow_speculative_execution=False,
            max_dispatch_per_tick=3,
            max_in_flight=4,
            risk_tier_priority={
                "critical": 0,
                "high": 1,
                "medium": 2,
                "low": 3,
            },
            per_risk_tier_in_flight={
                "critical": 2,
                "high": 1,
                "medium": 1,
                "low": 1,
            },
        ),
    ),
)

__all__ = ["HARDENING_MODE"]
