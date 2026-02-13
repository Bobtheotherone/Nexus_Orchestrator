"""Unit tests for core domain models."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone

import pytest

from nexus_orchestrator.domain import ids, models

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _fixed_bytes(size: int) -> bytes:
    return b"\x01" * size


def _utc_dt() -> datetime:
    return datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC)


def _sample_objects() -> dict[str, models.CanonicalModel]:
    ts = _utc_dt()
    run_id = ids.generate_run_id(timestamp_ms=100, randbytes=_fixed_bytes)
    work_item_id = ids.generate_work_item_id(timestamp_ms=101, randbytes=_fixed_bytes)
    attempt_id = ids.generate_attempt_id(timestamp_ms=102, randbytes=_fixed_bytes)
    evidence_id = ids.generate_evidence_id(timestamp_ms=103, randbytes=_fixed_bytes)
    merge_id = ids.generate_merge_id(timestamp_ms=104, randbytes=_fixed_bytes)
    incident_id = ids.generate_incident_id(timestamp_ms=105, randbytes=_fixed_bytes)
    artifact_id = ids.generate_artifact_id(timestamp_ms=106, randbytes=_fixed_bytes)

    budget = models.Budget(
        max_tokens=10_000,
        max_cost_usd=3.5,
        max_iterations=4,
        max_wall_clock_seconds=120,
    )
    sandbox_policy = models.SandboxPolicy(
        allow_network=False,
        allow_privileged_tools=False,
        allowed_tools=("rg", "pytest"),
        read_only_paths=("docs",),
        write_paths=("src", "tests"),
        max_cpu_seconds=10,
        max_memory_mb=512,
    )
    requirement = models.Requirement(
        id="REQ-0001",
        statement="IDs and serialization must be deterministic.",
        acceptance_criteria=("Generate valid ULIDs", "Canonical JSON roundtrip"),
        nfr_tags=("reliability", "security"),
        source="design_document.md#determinism",
    )
    constraint = models.Constraint(
        id="CON-SEC-0001",
        severity=models.ConstraintSeverity.MUST,
        category="security",
        description="No secrets in output payloads.",
        checker_binding="security_checker",
        parameters={"redaction": True},
        requirement_links=("REQ-0001",),
        source=models.ConstraintSource.MANUAL,
        created_at=ts,
    )
    envelope = models.ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=(constraint,),
        inherited_constraint_ids=(),
        compiled_at=ts,
    )
    work_item = models.WorkItem(
        id=work_item_id,
        title="Implement domain IDs",
        description="Add strict ULID and prefixed ID support.",
        scope=("src/nexus_orchestrator/domain/ids.py",),
        constraint_envelope=envelope,
        dependencies=(),
        status=models.WorkItemStatus.READY,
        risk_tier=models.RiskTier.MEDIUM,
        budget=budget,
        sandbox_policy=sandbox_policy,
        requirement_links=("REQ-0001",),
        expected_artifacts=("artifacts/ids-report.json",),
        created_at=ts,
        updated_at=ts,
    )
    task_graph = models.TaskGraph(
        run_id=run_id,
        work_items=(work_item,),
        edges=(),
        critical_path=(work_item_id,),
        created_at=ts,
    )
    spec_map = models.SpecMap(
        source_document="design_document.md",
        requirements=(requirement,),
        created_at=ts,
        entities=("WorkItem", "EvidenceRecord"),
        interfaces=("domain.models", "domain.events"),
        glossary={"ULID": "Universally unique, lexicographically sortable ID"},
    )
    artifact = models.Artifact(
        id=artifact_id,
        work_item_id=work_item_id,
        run_id=run_id,
        kind="report",
        path="artifacts/ids-report.json",
        sha256="a" * 64,
        size_bytes=128,
        created_at=ts,
    )
    evidence = models.EvidenceRecord(
        id=evidence_id,
        work_item_id=work_item_id,
        run_id=run_id,
        stage="lint",
        result=models.EvidenceResult.PASS,
        checker_id="ruff",
        constraint_ids=("CON-SEC-0001",),
        artifact_paths=("artifacts/ids-report.json",),
        tool_versions={"ruff": "0.14.14"},
        environment_hash="b" * 64,
        duration_ms=45,
        created_at=ts,
        summary="lint passed",
        metadata={"attempt": 1},
    )
    attempt = models.Attempt(
        id=attempt_id,
        work_item_id=work_item_id,
        run_id=run_id,
        iteration=1,
        provider="openai",
        model="gpt-5",
        role="implementer",
        prompt_hash="c" * 64,
        tokens_used=420,
        cost_usd=0.15,
        result=models.AttemptResult.SUCCESS,
        created_at=ts,
        feedback=None,
        finished_at=ts,
    )
    merge = models.MergeRecord(
        id=merge_id,
        work_item_id=work_item_id,
        run_id=run_id,
        commit_sha="a1b2c3d",
        evidence_ids=(evidence_id,),
        merged_at=ts,
    )
    incident = models.Incident(
        id=incident_id,
        run_id=run_id,
        category="runtime",
        message="Rate limit reached once.",
        created_at=ts,
        related_work_item_id=work_item_id,
        details={"retry_count": 1},
    )
    run = models.Run(
        id=run_id,
        spec_path="samples/specs/minimal_design_doc.md",
        status="running",
        started_at=ts,
        finished_at=None,
        work_item_ids=(work_item_id,),
        budget=budget,
        risk_tier=models.RiskTier.MEDIUM,
        metadata={"operator": "local"},
    )

    return {
        "Requirement": requirement,
        "SpecMap": spec_map,
        "WorkItem": work_item,
        "TaskGraph": task_graph,
        "Constraint": constraint,
        "ConstraintEnvelope": envelope,
        "EvidenceRecord": evidence,
        "Artifact": artifact,
        "Attempt": attempt,
        "MergeRecord": merge,
        "Incident": incident,
        "Run": run,
        "Budget": budget,
        "SandboxPolicy": sandbox_policy,
    }


def test_json_roundtrip_for_every_model() -> None:
    instances = _sample_objects()

    for model_name, instance in instances.items():
        cls = type(instance)
        assert hasattr(cls, "from_json"), f"{model_name} missing from_json"
        json_payload = instance.to_json()
        restored = cls.from_json(json_payload)
        assert restored == instance, model_name


def test_missing_required_field_errors_are_path_aware() -> None:
    with pytest.raises(ValueError, match=r"Requirement: missing required fields"):
        models.Requirement.from_dict({"id": "REQ-0001"})

    with pytest.raises(ValueError, match=r"WorkItem: missing required fields"):
        models.WorkItem.from_dict({"id": ids.generate_work_item_id(randbytes=_fixed_bytes)})


def test_invalid_enum_and_invalid_ids_are_rejected() -> None:
    constraint_data = {
        "id": "CON-SEC-0001",
        "severity": "invalid",
        "category": "security",
        "description": "desc",
        "checker_binding": "checker",
        "parameters": {},
        "requirement_links": ["REQ-0001"],
        "source": "manual",
        "created_at": _utc_dt().isoformat().replace("+00:00", "Z"),
    }
    with pytest.raises(ValueError, match=r"Constraint\.severity"):
        models.Constraint.from_dict(constraint_data)

    budget = models.Budget(
        max_tokens=1, max_cost_usd=1.0, max_iterations=1, max_wall_clock_seconds=1
    )
    sandbox_policy = models.SandboxPolicy(
        allow_network=False,
        allow_privileged_tools=False,
        allowed_tools=(),
        read_only_paths=(),
        write_paths=("src",),
    )
    constraint = models.Constraint(
        id="CON-SEC-0001",
        severity=models.ConstraintSeverity.MUST,
        category="security",
        description="desc",
        checker_binding="checker",
        parameters={},
        requirement_links=("REQ-0001",),
        source=models.ConstraintSource.MANUAL,
        created_at=_utc_dt(),
    )
    envelope = models.ConstraintEnvelope(
        work_item_id=ids.generate_work_item_id(randbytes=_fixed_bytes),
        constraints=(constraint,),
        compiled_at=_utc_dt(),
    )

    with pytest.raises(ValueError, match=r"WorkItem\.id"):
        models.WorkItem(
            id="wi-invalid",
            title="title",
            description="description",
            scope=("src/file.py",),
            constraint_envelope=envelope,
            budget=budget,
            sandbox_policy=sandbox_policy,
            created_at=_utc_dt(),
            updated_at=_utc_dt(),
        )


def test_datetimes_must_be_timezone_aware_utc() -> None:
    naive = datetime(2026, 1, 1, 0, 0, 0)
    budget = models.Budget(
        max_tokens=1, max_cost_usd=1.0, max_iterations=1, max_wall_clock_seconds=1
    )

    with pytest.raises(ValueError, match=r"Run\.started_at"):
        models.Run(
            id=ids.generate_run_id(randbytes=_fixed_bytes),
            spec_path="samples/specs/minimal_design_doc.md",
            status="running",
            started_at=naive,
            budget=budget,
        )


def test_canonical_json_is_stable_and_sorted() -> None:
    evidence = _sample_objects()["EvidenceRecord"]
    payload_1 = evidence.to_json()
    payload_2 = evidence.to_json()

    assert payload_1 == payload_2

    parsed = json.loads(payload_1)
    assert payload_1 == json.dumps(parsed, sort_keys=True, separators=(",", ":"))


def test_evidence_record_rejects_large_inline_blob() -> None:
    sample = _sample_objects()
    evidence = sample["EvidenceRecord"]
    evidence_dict = evidence.to_dict()
    assert "raw_log" not in evidence_dict

    evidence_dict["metadata"] = {"raw_log": "x" * 3000}
    with pytest.raises(ValueError, match=r"EvidenceRecord\.metadata\.raw_log"):
        models.EvidenceRecord.from_dict(evidence_dict)


def test_model_from_dict_rejects_unknown_fields() -> None:
    requirement = _sample_objects()["Requirement"]
    payload = requirement.to_dict()
    payload["unknown"] = True

    with pytest.raises(ValueError, match=r"Requirement: unexpected fields"):
        models.Requirement.from_dict(payload)


def test_to_dict_contains_schema_version_field() -> None:
    for instance in _sample_objects().values():
        assert is_dataclass(instance)
        if "schema_version" in {f.name for f in fields(instance)}:
            payload = instance.to_dict()
            assert payload["schema_version"] >= 1
