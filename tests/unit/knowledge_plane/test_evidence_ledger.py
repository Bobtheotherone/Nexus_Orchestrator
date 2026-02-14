"""
nexus-orchestrator — evidence ledger unit tests.

Purpose
- Validate deterministic traceability queries, coverage, integrity verification, and
  audit bundle export behavior.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from nexus_orchestrator.domain.models import (
    Constraint,
    EvidenceRecord,
    EvidenceResult,
    Incident,
    MergeRecord,
    Requirement,
    Run,
    SpecMap,
    WorkItem,
)
from nexus_orchestrator.knowledge_plane.evidence_ledger import EvidenceLedger
from nexus_orchestrator.persistence.repositories import (
    ConstraintRepo,
    EvidenceRepo,
    IncidentRepo,
    MergeRepo,
    RunRepo,
    WorkItemRepo,
)
from nexus_orchestrator.persistence.state_db import StateDB
from nexus_orchestrator.security.redaction import REDACTED_VALUE
from nexus_orchestrator.verification_plane.evidence import EvidenceWriter

from ..persistence import (
    fixed_now,
    make_constraint,
    make_evidence,
    make_incident,
    make_merge,
    make_run,
    make_work_item,
)


@dataclass(frozen=True, slots=True)
class _SeededState:
    run: Run
    work_item: WorkItem
    constraint_primary: Constraint
    constraint_secondary: Constraint
    evidence_pass: EvidenceRecord
    evidence_fail: EvidenceRecord
    merge: MergeRecord
    incident: Incident
    evidence_root: Path


def _seed_ledger(
    tmp_path: Path, *, tamper_failed_evidence: bool = False
) -> tuple[EvidenceLedger, _SeededState]:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)
    constraint_repo = ConstraintRepo(db)
    evidence_repo = EvidenceRepo(db)
    merge_repo = MergeRepo(db)
    incident_repo = IncidentRepo(db)

    work_item = make_work_item(101)
    run = make_run(100, work_item_ids=(work_item.id,))
    constraint_primary = make_constraint(101, requirement_links=("REQ-0001",))
    constraint_secondary = make_constraint(102, requirement_links=("REQ-0002",))

    run_repo.add(run)
    work_item_repo.add(run.id, work_item)
    constraint_repo.add(constraint_primary)
    constraint_repo.add(constraint_secondary)

    evidence_pass = make_evidence(
        103,
        run_id=run.id,
        work_item_id=work_item.id,
        constraint_ids=(constraint_primary.id,),
    )
    evidence_fail_payload = make_evidence(
        104,
        run_id=run.id,
        work_item_id=work_item.id,
        constraint_ids=(constraint_primary.id,),
    ).to_dict()
    evidence_fail_payload["result"] = EvidenceResult.FAIL.value
    evidence_fail_payload["summary"] = "assertion failed"
    evidence_fail = EvidenceRecord.from_dict(evidence_fail_payload)

    evidence_repo.add(evidence_pass)
    evidence_repo.add(evidence_fail)

    merge = make_merge(
        105,
        run_id=run.id,
        work_item_id=work_item.id,
        evidence_ids=(evidence_pass.id, evidence_fail.id),
    )
    merge_repo.add(merge)

    incident_payload = make_incident(106, run_id=run.id, work_item_id=work_item.id).to_dict()
    incident_payload["constraint_ids"] = [constraint_primary.id]
    incident_payload["evidence_ids"] = [evidence_fail.id]
    incident_payload["message"] = "constraint violated in runtime"
    incident = Incident.from_dict(incident_payload)
    incident_repo.add(incident)

    evidence_root = tmp_path / "evidence"
    writer = EvidenceWriter(evidence_root)
    writer.write_evidence(
        run_id=run.id,
        work_item_id=work_item.id,
        stage=evidence_pass.stage,
        evidence_id=evidence_pass.id,
        artifacts={"payload.json": {"ok": True}},
    )
    failed_write = writer.write_evidence(
        run_id=run.id,
        work_item_id=work_item.id,
        stage=evidence_fail.stage,
        evidence_id=evidence_fail.id,
        artifacts={"payload.json": {"ok": False}},
    )
    if tamper_failed_evidence:
        tamper_target = failed_write.evidence_dir / "artifacts" / "payload.json"
        tamper_target.write_text('{"ok":true}\n', encoding="utf-8")

    ledger = EvidenceLedger(db, evidence_root=evidence_root, repo_root=tmp_path)
    state = _SeededState(
        run=run,
        work_item=work_item,
        constraint_primary=constraint_primary,
        constraint_secondary=constraint_secondary,
        evidence_pass=evidence_pass,
        evidence_fail=evidence_fail,
        merge=merge,
        incident=incident,
        evidence_root=evidence_root,
    )
    return ledger, state


def _spec_map() -> SpecMap:
    return SpecMap(
        source_document="samples/specs/minimal_design_doc.md",
        requirements=(
            Requirement(id="REQ-0001", statement="primary requirement"),
            Requirement(id="REQ-0002", statement="secondary requirement"),
            Requirement(id="REQ-0003", statement="unmapped requirement"),
        ),
        created_at=fixed_now(1),
    )


def test_trace_requirement_joins_constraint_evidence_and_commit(tmp_path: Path) -> None:
    ledger, seeded = _seed_ledger(tmp_path)

    result = ledger.trace_requirement("REQ-0001")

    assert result.requirement_id == "REQ-0001"
    assert result.constraint_ids == (seeded.constraint_primary.id,)
    assert tuple(row.evidence_id for row in result.rows) == (
        seeded.evidence_pass.id,
        seeded.evidence_fail.id,
    )
    assert {row.commit_sha for row in result.rows} == {seeded.merge.commit_sha}

    req2 = ledger.trace_requirement("REQ-0002")
    assert req2.constraint_ids == (seeded.constraint_secondary.id,)
    assert len(req2.rows) == 1
    assert req2.rows[0].evidence_id is None

    assert result == ledger.trace_requirement("REQ-0001")


def test_trace_constraint_returns_evidence_and_failures(tmp_path: Path) -> None:
    ledger, seeded = _seed_ledger(tmp_path)

    result = ledger.trace_constraint(seeded.constraint_primary.id)

    assert tuple(row.evidence_id for row in result.evidence_rows) == (
        seeded.evidence_pass.id,
        seeded.evidence_fail.id,
    )
    assert len(result.failure_rows) == 2
    assert result.failure_rows[0].source == "evidence"
    assert result.failure_rows[0].evidence_id == seeded.evidence_fail.id
    assert result.failure_rows[1].source == "incident"
    assert result.failure_rows[1].incident_id == seeded.incident.id

    assert result == ledger.trace_constraint(seeded.constraint_primary.id)


def test_coverage_report_marks_requirements_with_and_without_evidence(tmp_path: Path) -> None:
    ledger, seeded = _seed_ledger(tmp_path)

    report = ledger.coverage_report(_spec_map())
    rows = {row.requirement_id: row for row in report.rows}

    req1 = rows["REQ-0001"]
    assert req1.covered is True
    assert req1.constraint_ids == (seeded.constraint_primary.id,)
    assert req1.evidence_ids == (seeded.evidence_pass.id, seeded.evidence_fail.id)

    req2 = rows["REQ-0002"]
    assert req2.covered is False
    assert req2.constraint_ids == (seeded.constraint_secondary.id,)
    assert req2.evidence_ids == ()

    req3 = rows["REQ-0003"]
    assert req3.covered is False
    assert req3.constraint_ids == ()
    assert req3.evidence_ids == ()

    assert report.total_requirements == 3
    assert report.covered_requirements == 1
    assert report.uncovered_requirement_ids == ("REQ-0002", "REQ-0003")
    assert report.coverage_ratio == pytest.approx(1.0 / 3.0)

    assert report == ledger.coverage_report(_spec_map())


def test_verify_integrity_reports_valid_and_invalid_evidence_dirs(tmp_path: Path) -> None:
    ledger, seeded = _seed_ledger(tmp_path, tamper_failed_evidence=True)

    report = ledger.verify_integrity(run_id=seeded.run.id)
    by_id = {row.evidence_id: row for row in report.rows}

    assert report.valid_count == 1
    assert report.invalid_count == 1
    assert report.missing_count == 0
    assert report.error_count == 0

    assert by_id[seeded.evidence_pass.id].status == "valid"
    failed_row = by_id[seeded.evidence_fail.id]
    assert failed_row.status == "invalid"
    assert "artifacts/payload.json" in failed_row.hash_mismatches

    assert report == ledger.verify_integrity(run_id=seeded.run.id)


def test_export_audit_bundle_is_redacted_and_deterministic(tmp_path: Path) -> None:
    ledger, seeded = _seed_ledger(tmp_path)

    key_log = tmp_path / "logs" / "runtime.log"
    key_log.parent.mkdir(parents=True, exist_ok=True)
    key_log.write_text("token=sk-SUPERSECRET12345678901234567890", encoding="utf-8")

    first = ledger.export_audit_bundle(
        run_id=seeded.run.id,
        output_path=tmp_path / "bundle-a.zip",
        key_log_paths=(Path("logs/runtime.log"),),
    )
    second = ledger.export_audit_bundle(
        run_id=seeded.run.id,
        output_path=tmp_path / "bundle-b.zip",
        key_log_paths=(Path("logs/runtime.log"),),
    )

    assert first.member_names == second.member_names
    assert first.bundle_path.read_bytes() == second.bundle_path.read_bytes()

    with zipfile.ZipFile(first.bundle_path, "r") as archive:
        names = sorted(archive.namelist())
        assert f"evidence/{seeded.evidence_pass.id}.audit.zip" in names
        assert f"evidence/{seeded.evidence_fail.id}.audit.zip" in names
        assert "snapshots/constraint_registry.json" in names
        assert "logs/incidents.json" in names
        assert "logs/merges.json" in names
        assert "logs/key/logs/runtime.log" in names

        redacted_log = archive.read("logs/key/logs/runtime.log").decode("utf-8")
        assert "sk-SUPERSECRET" not in redacted_log
        assert REDACTED_VALUE in redacted_log
