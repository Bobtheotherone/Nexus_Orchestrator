"""Unit tests for work_item_classifier — 3-way ModelAffinity routing.

File: tests/unit/test_work_item_classifier.py

Covers:
- ModelAffinity enum values and backward-compat aliases
- classify_work_item() 3-way score thresholds → GPT53/SPARK/OPUS
- Individual scoring functions
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    RiskTier,
    WorkItem,
)
from nexus_orchestrator.synthesis_plane.work_item_classifier import (
    ModelAffinity,
    classify_work_item,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _randbytes(seed: int) -> callable:
    byte_value = (seed % 251) + 1

    def _provider(size: int) -> bytes:
        return bytes([byte_value]) * size

    return _provider


def _make_work_item(
    seed: int,
    *,
    title: str = "Test item",
    description: str = "Test description",
    scope: tuple[str, ...] = ("src/module.py",),
    risk_tier: RiskTier = RiskTier.MEDIUM,
    constraint_count: int = 1,
) -> WorkItem:
    work_item_id = ids.generate_work_item_id(
        timestamp_ms=1_700_000_000_000 + seed,
        randbytes=_randbytes(seed),
    )
    constraints = tuple(
        Constraint(
            id=f"CON-TST-{(seed * 10 + i) % 9000 + 1:04d}",
            severity=ConstraintSeverity.MUST,
            category="test",
            description="Test constraint",
            checker_binding="test",
            parameters={},
            requirement_links=("REQ-0001",),
            source=ConstraintSource.MANUAL,
            created_at=_NOW,
        )
        for i in range(constraint_count)
    )
    envelope = ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=constraints,
        inherited_constraint_ids=(),
        compiled_at=_NOW,
    )
    return WorkItem(
        id=work_item_id,
        title=title,
        description=description,
        scope=scope,
        constraint_envelope=envelope,
        dependencies=(),
        risk_tier=risk_tier,
        requirement_links=("REQ-0001",),
        constraint_ids=tuple(c.id for c in constraints),
        created_at=_NOW,
        updated_at=_NOW,
    )


class TestModelAffinityEnum:
    """Test ModelAffinity enum values and backward-compat aliases."""

    def test_three_primary_values(self) -> None:
        assert ModelAffinity.GPT53.value == "gpt53"
        assert ModelAffinity.SPARK.value == "spark"
        assert ModelAffinity.OPUS.value == "opus"

    def test_backward_compat_codex_first(self) -> None:
        """CODEX_FIRST maps to GPT53."""
        assert ModelAffinity.CODEX_FIRST == ModelAffinity.GPT53
        assert ModelAffinity.CODEX_FIRST.value == "gpt53"

    def test_backward_compat_claude_first(self) -> None:
        """CLAUDE_FIRST maps to OPUS."""
        assert ModelAffinity.CLAUDE_FIRST == ModelAffinity.OPUS
        assert ModelAffinity.CLAUDE_FIRST.value == "opus"

    def test_all_unique_names(self) -> None:
        names = [m.name for m in ModelAffinity]
        # GPT53/CODEX_FIRST share a value, OPUS/CLAUDE_FIRST share a value
        # But names should be unique
        assert len(names) == len(set(names))


class TestClassifyWorkItem:
    """Test 3-way classification based on score thresholds."""

    def test_coding_task_returns_opus(self) -> None:
        """Source code scope + implementation keywords → OPUS (score > 2)."""
        wi = _make_work_item(
            1,
            title="Implement user authentication handler",
            description="Write the login endpoint with validation and parsing",
            scope=("src/auth/handler.py", "src/auth/validate.py", "tests/test_auth.py"),
            risk_tier=RiskTier.LOW,
            constraint_count=1,
        )
        assert classify_work_item(wi) == ModelAffinity.OPUS

    def test_architecture_task_returns_gpt53(self) -> None:
        """Config scope + architecture keywords + high risk → GPT53 (score < -2)."""
        wi = _make_work_item(
            2,
            title="Design infrastructure migration strategy",
            description="Architect the deployment pipeline and review configuration schema",
            scope=("infra/deploy.yaml", "config/schema.toml", "docs/ADR-001.md"),
            risk_tier=RiskTier.CRITICAL,
            constraint_count=5,
        )
        assert classify_work_item(wi) == ModelAffinity.GPT53

    def test_simple_task_returns_spark(self) -> None:
        """Mixed signals (middle ground) → SPARK (-2 ≤ score ≤ 2).

        Balanced scope (1 code file + 1 config file = tie → 0),
        balanced keywords (1 code + 1 architecture → tie → 0),
        medium risk (→ 0), moderate constraints (→ 0).
        Total score = 0, which falls in SPARK range.
        """
        wi = _make_work_item(
            3,
            title="Review the handler setup",
            description="Audit the configuration for correctness",
            scope=("src/handler.py", "config/settings.toml"),
            risk_tier=RiskTier.MEDIUM,
            constraint_count=2,
        )
        result = classify_work_item(wi)
        assert result == ModelAffinity.SPARK

    def test_deterministic_same_input(self) -> None:
        """Same input always produces same output."""
        wi = _make_work_item(4, title="Fix parser bug", description="Fix the JSON parser")
        first = classify_work_item(wi)
        second = classify_work_item(wi)
        assert first == second

    def test_many_constraints_push_toward_gpt53(self) -> None:
        """High constraint count is a signal for GPT53/reasoning tasks."""
        wi = _make_work_item(
            5,
            title="Design system with constraints",
            description="Architect a constrained design with audit requirements",
            scope=("docs/design.md", "config/rules.yaml"),
            risk_tier=RiskTier.HIGH,
            constraint_count=6,
        )
        result = classify_work_item(wi)
        # High constraints + config scope + architecture keywords → GPT53
        assert result == ModelAffinity.GPT53

    def test_low_risk_code_scope_pushes_toward_opus(self) -> None:
        """Low risk + code scope + code keywords → OPUS."""
        wi = _make_work_item(
            6,
            title="Refactor the serializer module",
            description="Fix and implement the deserialization handler function",
            scope=("src/serializer.py", "src/handler.py"),
            risk_tier=RiskTier.LOW,
            constraint_count=1,
        )
        result = classify_work_item(wi)
        assert result == ModelAffinity.OPUS
