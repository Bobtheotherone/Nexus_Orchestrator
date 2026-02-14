"""Unit tests for deterministic constraint registry loading/querying."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from nexus_orchestrator.domain.models import ConstraintSeverity
from nexus_orchestrator.knowledge_plane.constraint_registry import (
    ConstraintExemptionRecord,
    ConstraintExemptionTracker,
    ConstraintRegistry,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc  # noqa: UP017


def _constraint_payload(
    constraint_id: str,
    *,
    severity: str = "must",
    category: str = "security",
    checker: str = "security_checker",
    requirement_links: list[str] | None = None,
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": constraint_id,
        "severity": severity,
        "category": category,
        "description": f"constraint {constraint_id}",
        "checker": checker,
        "parameters": parameters if parameters is not None else {},
        "requirement_links": requirement_links if requirement_links is not None else [],
        "source": "manual",
    }


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = cast(
        "str",
        yaml.safe_dump(
            payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        ),
    )
    path.write_text(rendered, encoding="utf-8")


def test_load_base_registry_and_query_apis() -> None:
    registry = ConstraintRegistry.load()

    assert registry.source_files
    assert registry.source_files[0].name == "000_base_constraints.yaml"

    security_constraint = registry.by_id("CON-SEC-0001")
    assert security_constraint is not None
    assert security_constraint.checker_binding == "security_checker"
    assert security_constraint.created_at == datetime(1970, 1, 1, tzinfo=UTC)

    assert security_constraint in registry.by_category("security")
    assert security_constraint in registry.by_severity("must")
    assert security_constraint in registry.by_severity(ConstraintSeverity.MUST)
    assert security_constraint in registry.by_checker("security_checker")
    assert registry.by_requirement_link("REQ-0001") == ()

    assert registry.all_active() == registry.constraints


def test_load_is_lexicographic_across_registry_files(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    _write_yaml(registry_dir / "200_second.yaml", [_constraint_payload("CON-COR-9002")])
    _write_yaml(registry_dir / "010_first.yaml", [_constraint_payload("CON-COR-9001")])

    registry = ConstraintRegistry.load(registry_dir)

    assert tuple(item.id for item in registry.constraints) == ("CON-COR-9001", "CON-COR-9002")
    assert all(item.created_at == datetime(1970, 1, 1, tzinfo=UTC) for item in registry.constraints)


def test_load_rejects_duplicate_constraint_ids_across_files(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    payload = [_constraint_payload("CON-SEC-9001")]
    _write_yaml(registry_dir / "010_first.yaml", payload)
    _write_yaml(registry_dir / "020_second.yaml", payload)

    with pytest.raises(ValueError, match="duplicate constraint id"):
        ConstraintRegistry.load(registry_dir)


def test_load_rejects_missing_required_field(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    payload = _constraint_payload("CON-SEC-9001")
    del payload["severity"]
    _write_yaml(registry_dir / "010_invalid.yaml", [payload])

    with pytest.raises(ValueError, match="missing required fields"):
        ConstraintRegistry.load(registry_dir)


def test_load_rejects_bad_severity(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    _write_yaml(
        registry_dir / "010_invalid.yaml",
        [_constraint_payload("CON-SEC-9001", severity="critical")],
    )

    with pytest.raises(ValueError, match="invalid severity"):
        ConstraintRegistry.load(registry_dir)


def test_exemption_tracking_is_validated_and_deterministic(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    _write_yaml(
        registry_dir / "010_constraints.yaml",
        [
            _constraint_payload("CON-SEC-9001"),
            _constraint_payload("CON-SEC-9002"),
        ],
    )

    registry = ConstraintRegistry.load(registry_dir)

    tracker = ConstraintExemptionTracker.from_records(
        (
            {
                "constraint_id": "CON-SEC-9002",
                "justification": "approved temporary waiver",
                "approved_by": "ops-review",
                "expiry": "2999-01-01T00:00:00Z",
                "approved": True,
            },
            {
                "constraint_id": "CON-SEC-9001",
                "justification": "approved temporary waiver",
                "approved_by": "ops-review",
                "expiry": "2999-01-01T00:00:00Z",
                "approved": True,
            },
        )
    )

    assert tuple(item.constraint_id for item in tracker.records) == (
        "CON-SEC-9001",
        "CON-SEC-9002",
    )

    filtered = registry.with_exemptions(tracker.records)
    assert filtered.all_active() == ()

    with pytest.raises(ValueError, match="constraint_id"):
        ConstraintExemptionRecord(
            constraint_id="invalid-id",
            justification="x",
            approved=True,
        )


def test_add_constraint_writes_new_deterministic_file_without_overwrite(tmp_path: Path) -> None:
    registry_dir = tmp_path / "registry"
    _write_yaml(registry_dir / "000_seed.yaml", [_constraint_payload("CON-SEC-9001")])

    registry = ConstraintRegistry.load(registry_dir)

    destination = registry.add_constraint(
        _constraint_payload(
            "CON-NEW-9001",
            severity="should",
            category="reliability",
            checker="reliability_checker",
            requirement_links=["REQ-0002", "REQ-0001"],
            parameters={"z": 1, "a": {"b": 2}},
        ),
        current_date=date(2026, 2, 14),
    )

    assert destination.name == "9xx_auto_20260214.yaml"
    assert registry.by_id("CON-NEW-9001") is not None

    parsed = cast("object", yaml.safe_load(destination.read_text(encoding="utf-8")))
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    first_entry = parsed[0]
    assert isinstance(first_entry, dict)
    assert first_entry["id"] == "CON-NEW-9001"
    assert first_entry["severity"] == "should"
    assert first_entry["category"] == "reliability"
    assert first_entry["description"] == "constraint CON-NEW-9001"
    assert first_entry["checker"] == "reliability_checker"
    assert first_entry["parameters"] == {"a": {"b": 2}, "z": 1}
    assert first_entry["requirement_links"] == ["REQ-0001", "REQ-0002"]
    assert first_entry["source"] == "manual"
    assert isinstance(first_entry["created_at"], str)
    assert first_entry["created_at"].endswith("Z")

    custom_destination = registry.add_constraint(
        _constraint_payload("CON-NEW-9002"),
        filename="950_custom.yaml",
    )
    assert custom_destination.name == "950_custom.yaml"

    second_auto = registry.add_constraint(
        _constraint_payload("CON-NEW-9003"),
        current_date=date(2026, 2, 14),
    )
    assert second_auto.name == "9xx_auto_20260214_01.yaml"

    with pytest.raises(FileExistsError):
        registry.add_constraint(
            _constraint_payload("CON-NEW-9004"),
            filename="950_custom.yaml",
        )
