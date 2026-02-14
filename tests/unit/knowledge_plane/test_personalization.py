"""Unit tests for operator-profile personalization overlays."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from nexus_orchestrator.config.schema import merge_config
from nexus_orchestrator.domain.models import Constraint, ConstraintSeverity, ConstraintSource
from nexus_orchestrator.knowledge_plane.personalization import (
    OperatorProfile,
    load_operator_profile,
    planning_overlay_from_profile,
    routing_overlay_from_profile,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc  # noqa: UP017


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip(), encoding="utf-8")


def _constraint(constraint_id: str, severity: ConstraintSeverity) -> Constraint:
    return Constraint(
        id=constraint_id,
        severity=severity,
        category="security",
        description=f"{constraint_id} description",
        checker_binding="security_checker",
        parameters={},
        requirement_links=(),
        source=ConstraintSource.MANUAL,
        created_at=datetime(2026, 2, 14, tzinfo=UTC),
    )


def test_load_operator_profile_uses_configured_default_path(tmp_path: Path) -> None:
    profile_path = tmp_path / "profiles" / "custom_profile.toml"
    _write(
        profile_path,
        """
        [profile]
        name = "custom"
        owner = "unit-test"
        version = 3
        """,
    )

    profile = load_operator_profile(
        config={"personalization": {"profile_path": "profiles/custom_profile.toml"}},
        base_dir=tmp_path,
    )

    assert profile.name == "custom"
    assert profile.owner == "unit-test"
    assert profile.version == 3


def test_load_operator_profile_uses_builtin_default_path(tmp_path: Path) -> None:
    _write(
        tmp_path / "profiles" / "operator_profile.toml",
        """
        [profile]
        name = "builtin-default"
        owner = "unit-test"
        version = 1
        """,
    )

    profile = load_operator_profile(base_dir=tmp_path)

    assert profile.name == "builtin-default"


def test_planning_overlay_never_relaxes_must_constraints() -> None:
    profile = OperatorProfile.from_mapping(
        {
            "profile": {"name": "unit", "owner": "qa", "version": 1},
            "planning": {
                "budgets": {
                    "max_iterations": 7,
                    "max_tokens_per_attempt": 2048,
                    "max_cost_per_work_item_usd": 1.75,
                },
                "roles": {
                    "enabled": ["Implementer", "Reviewer"],
                    "disabled": ["Security"],
                },
                "constraints": {
                    "should_toggles": {
                        "CON-SEC-0001": False,
                        "CON-SEC-0002": False,
                    }
                },
            },
        }
    )
    constraints = (
        _constraint("CON-SEC-0001", ConstraintSeverity.MUST),
        _constraint("CON-SEC-0002", ConstraintSeverity.SHOULD),
    )
    overlay = planning_overlay_from_profile(
        profile,
        config={"personalization": {"enabled": True, "may_relax_should_constraints": True}},
        constraints=constraints,
    )

    assert overlay["budgets"] == {
        "max_iterations": 7,
        "max_tokens_per_attempt": 2048,
        "max_cost_per_work_item_usd": 1.75,
    }
    assert overlay["roles"] == {
        "enabled": ["implementer", "reviewer"],
        "disabled": ["security"],
    }
    assert overlay["constraints"]["should_toggles"] == {"CON-SEC-0002": False}
    assert overlay["constraints"]["blocked_relaxations"] == [
        {
            "constraint_id": "CON-SEC-0001",
            "reason": "must_constraints_cannot_be_relaxed",
        }
    ]


def test_planning_overlay_blocks_should_toggles_when_config_disallows_relaxing() -> None:
    profile = OperatorProfile.from_mapping(
        {
            "profile": {"name": "unit", "owner": "qa", "version": 1},
            "planning": {
                "constraints": {"should_toggles": {"CON-COR-0100": False}},
            },
        }
    )

    overlay = planning_overlay_from_profile(
        profile,
        config={"personalization": {"enabled": True, "may_relax_should_constraints": False}},
        constraints=(_constraint("CON-COR-0100", ConstraintSeverity.SHOULD),),
    )

    assert "should_toggles" not in overlay["constraints"]
    assert overlay["constraints"]["blocked_relaxations"] == [
        {
            "constraint_id": "CON-COR-0100",
            "reason": "config_disallows_should_relaxation",
        }
    ]


def test_routing_preferences_deterministically_change_config_overlay() -> None:
    profile = OperatorProfile.from_mapping(
        {
            "profile": {"name": "route", "owner": "qa", "version": 1},
            "routing": {
                "providers": {
                    "openai": {"model_code": "openai-code-v1", "model_architect": "openai-arch-v1"},
                    "anthropic": {"model_code": "anthropic-code-v1"},
                },
                "capability_profiles": {"implementer": "code", "architect": "architect"},
            },
            "routing_preferences": {
                "implementer_preference": "openai",
                "architect_preference": "anthropic",
            },
        }
    )

    first = routing_overlay_from_profile(profile)
    second = routing_overlay_from_profile(profile)

    assert first == second
    assert first["providers"] == {
        "anthropic": {"model_code": "anthropic-code-v1"},
        "openai": {
            "model_code": "openai-code-v1",
            "model_architect": "openai-arch-v1",
        },
    }
    assert first["routing_hints"] == {
        "role_provider_preferences": {
            "architect": "anthropic",
            "implementer": "openai",
        },
        "role_capability_profiles": {
            "architect": "architect",
            "implementer": "code",
        },
    }

    base_config = {
        "providers": {
            "openai": {
                "model_code": "default-openai-code",
                "model_architect": "default-openai-architect",
            },
            "anthropic": {"model_code": "default-anthropic-code"},
        }
    }
    merged = merge_config(base_config, first)
    assert merged["providers"]["openai"]["model_code"] == "openai-code-v1"
    assert merged["providers"]["openai"]["model_architect"] == "openai-arch-v1"
    assert merged["providers"]["anthropic"]["model_code"] == "anthropic-code-v1"
