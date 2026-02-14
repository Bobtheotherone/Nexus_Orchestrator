"""
nexus-orchestrator — deterministic evidence ledger query and audit APIs.

File: src/nexus_orchestrator/knowledge_plane/evidence_ledger.py
Last updated: 2026-02-14

Purpose
- Read/query interface to the Evidence Ledger using StateDB + evidence filesystem layout.
- Provide deterministic traceability and audit export operations.

What this module includes
- Query APIs:
  1) requirement -> constraints -> evidence -> commit
  2) constraint -> evidence + failures
  3) coverage report for a SpecMap (which requirements have evidence)
- Integrity verification for manifest/hash-backed evidence directories.
- Deterministic offline audit bundle export with redacted snapshots/logs.
"""

from __future__ import annotations

import json
import math
import os
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, Protocol, cast

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Constraint,
    EvidenceRecord,
    EvidenceResult,
    Incident,
    JSONValue,
    MergeRecord,
    SpecMap,
    WorkItem,
)
from nexus_orchestrator.persistence.repositories import (
    ConstraintRepo,
    IncidentRepo,
    MergeRepo,
    RunRepo,
    WorkItemRepo,
)
from nexus_orchestrator.security.redaction import redact_structure, redact_text
from nexus_orchestrator.verification_plane.evidence import (
    export_audit_bundle as export_single_evidence_bundle,
)
from nexus_orchestrator.verification_plane.evidence import verify_integrity

if TYPE_CHECKING:
    from nexus_orchestrator.persistence.state_db import RowValue, StateDB

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc  # noqa: UP017

_ZIP_FIXED_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)
_MAX_PAGE_SIZE: Final[int] = 1_000


PathLike = str | os.PathLike[str]


class _RequirementView(Protocol):
    id: object
    statement: object


@dataclass(frozen=True, slots=True)
class RequirementTraceRow:
    """One row in requirement -> constraint -> evidence -> commit trace."""

    requirement_id: str
    constraint_id: str
    evidence_id: str | None
    run_id: str | None
    work_item_id: str | None
    stage: str | None
    evidence_result: EvidenceResult | None
    evidence_created_at: datetime | None
    merge_id: str | None
    commit_sha: str | None


@dataclass(frozen=True, slots=True)
class RequirementTraceResult:
    """Typed requirement trace result with deterministic row ordering."""

    requirement_id: str
    constraint_ids: tuple[str, ...]
    rows: tuple[RequirementTraceRow, ...]


@dataclass(frozen=True, slots=True)
class ConstraintEvidenceRow:
    """One evidence row for a constraint trace."""

    constraint_id: str
    evidence_id: str
    run_id: str
    work_item_id: str
    stage: str
    result: EvidenceResult
    created_at: datetime
    merge_id: str | None
    commit_sha: str | None


@dataclass(frozen=True, slots=True)
class ConstraintFailureRow:
    """One failure source row for a constraint trace."""

    constraint_id: str
    source: str
    identifier: str
    run_id: str | None
    work_item_id: str | None
    evidence_id: str | None
    incident_id: str | None
    created_at: datetime | None
    message: str


@dataclass(frozen=True, slots=True)
class ConstraintTraceResult:
    """Constraint evidence + failure trace."""

    constraint_id: str
    evidence_rows: tuple[ConstraintEvidenceRow, ...]
    failure_rows: tuple[ConstraintFailureRow, ...]


@dataclass(frozen=True, slots=True)
class RequirementCoverageRow:
    """Coverage row for one requirement."""

    requirement_id: str
    statement: str
    constraint_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    covered: bool
    latest_evidence_at: datetime | None


@dataclass(frozen=True, slots=True)
class SpecCoverageReport:
    """Coverage summary for a spec map."""

    rows: tuple[RequirementCoverageRow, ...]
    total_requirements: int
    covered_requirements: int
    uncovered_requirement_ids: tuple[str, ...]
    coverage_ratio: float


@dataclass(frozen=True, slots=True)
class EvidenceIntegrityRow:
    """Integrity status for one evidence record directory."""

    evidence_id: str
    run_id: str
    work_item_id: str
    stage: str
    evidence_dir: Path | None
    status: str
    missing_paths: tuple[str, ...]
    hash_mismatches: tuple[str, ...]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceIntegrityReport:
    """Aggregated deterministic integrity report."""

    rows: tuple[EvidenceIntegrityRow, ...]
    valid_count: int
    invalid_count: int
    missing_count: int
    error_count: int


@dataclass(frozen=True, slots=True)
class AuditBundleResult:
    """Result metadata for a deterministic audit bundle export."""

    bundle_path: Path
    run_id: str
    evidence_ids: tuple[str, ...]
    member_names: tuple[str, ...]


class EvidenceLedger:
    """Deterministic query/export facade over evidence records and artifact directories."""

    __slots__ = (
        "_constraint_repo",
        "_db",
        "_evidence_root",
        "_incident_repo",
        "_merge_repo",
        "_repo_root",
        "_run_repo",
        "_work_item_repo",
    )

    def __init__(
        self,
        db: StateDB,
        *,
        evidence_root: PathLike,
        repo_root: PathLike | None = None,
    ) -> None:
        self._db = db
        self._db.migrate()
        self._evidence_root = Path(evidence_root).expanduser().resolve()
        self._repo_root = (
            Path.cwd().resolve() if repo_root is None else Path(repo_root).expanduser().resolve()
        )
        self._constraint_repo = ConstraintRepo(self._db)
        self._run_repo = RunRepo(self._db)
        self._work_item_repo = WorkItemRepo(self._db)
        self._merge_repo = MergeRepo(self._db)
        self._incident_repo = IncidentRepo(self._db)

    @property
    def evidence_root(self) -> Path:
        return self._evidence_root

    def trace_requirement(
        self,
        requirement_id: str,
        *,
        run_id: str | None = None,
    ) -> RequirementTraceResult:
        """Resolve requirement -> constraints -> evidence -> commit trace rows."""

        ids.validate_requirement_id(requirement_id)
        if run_id is not None:
            ids.validate_run_id(run_id)

        constraints = tuple(
            sorted(
                (
                    item
                    for item in self._load_all_constraints()
                    if requirement_id in item.requirement_links
                ),
                key=lambda item: item.id,
            )
        )
        evidence_records = self._load_evidence_records(run_id=run_id)
        work_items = self._load_work_item_index(run_id=run_id)
        evidence_to_merges = self._build_evidence_merge_index(run_id=run_id)

        rows: list[RequirementTraceRow] = []
        for constraint in constraints:
            related = [
                record for record in evidence_records if constraint.id in record.constraint_ids
            ]
            related.sort(key=lambda item: (item.created_at, item.id))

            if not related:
                rows.append(
                    RequirementTraceRow(
                        requirement_id=requirement_id,
                        constraint_id=constraint.id,
                        evidence_id=None,
                        run_id=None,
                        work_item_id=None,
                        stage=None,
                        evidence_result=None,
                        evidence_created_at=None,
                        merge_id=None,
                        commit_sha=None,
                    )
                )
                continue

            for evidence in related:
                merges = evidence_to_merges.get(evidence.id)
                if merges:
                    for merge in merges:
                        rows.append(
                            RequirementTraceRow(
                                requirement_id=requirement_id,
                                constraint_id=constraint.id,
                                evidence_id=evidence.id,
                                run_id=evidence.run_id,
                                work_item_id=evidence.work_item_id,
                                stage=evidence.stage,
                                evidence_result=evidence.result,
                                evidence_created_at=evidence.created_at,
                                merge_id=merge.id,
                                commit_sha=merge.commit_sha,
                            )
                        )
                else:
                    work_item = work_items.get(evidence.work_item_id)
                    rows.append(
                        RequirementTraceRow(
                            requirement_id=requirement_id,
                            constraint_id=constraint.id,
                            evidence_id=evidence.id,
                            run_id=evidence.run_id,
                            work_item_id=evidence.work_item_id,
                            stage=evidence.stage,
                            evidence_result=evidence.result,
                            evidence_created_at=evidence.created_at,
                            merge_id=None,
                            commit_sha=None if work_item is None else work_item.commit_sha,
                        )
                    )

        rows.sort(
            key=lambda item: (
                item.constraint_id,
                _datetime_sort_key(item.evidence_created_at),
                item.evidence_id or "",
                item.merge_id or "",
                item.commit_sha or "",
            )
        )
        return RequirementTraceResult(
            requirement_id=requirement_id,
            constraint_ids=tuple(constraint.id for constraint in constraints),
            rows=tuple(rows),
        )

    def trace_constraint(
        self,
        constraint_id: str,
        *,
        run_id: str | None = None,
    ) -> ConstraintTraceResult:
        """Resolve constraint -> evidence + failures."""

        ids.validate_constraint_id(constraint_id)
        if run_id is not None:
            ids.validate_run_id(run_id)

        evidence_records = [
            item
            for item in self._load_evidence_records(run_id=run_id)
            if constraint_id in item.constraint_ids
        ]
        evidence_records.sort(key=lambda item: (item.created_at, item.id))

        work_items = self._load_work_item_index(run_id=run_id)
        evidence_to_merges = self._build_evidence_merge_index(run_id=run_id)

        evidence_rows: list[ConstraintEvidenceRow] = []
        for evidence in evidence_records:
            merges = evidence_to_merges.get(evidence.id)
            if merges:
                for merge in merges:
                    evidence_rows.append(
                        ConstraintEvidenceRow(
                            constraint_id=constraint_id,
                            evidence_id=evidence.id,
                            run_id=evidence.run_id,
                            work_item_id=evidence.work_item_id,
                            stage=evidence.stage,
                            result=evidence.result,
                            created_at=evidence.created_at,
                            merge_id=merge.id,
                            commit_sha=merge.commit_sha,
                        )
                    )
            else:
                work_item = work_items.get(evidence.work_item_id)
                evidence_rows.append(
                    ConstraintEvidenceRow(
                        constraint_id=constraint_id,
                        evidence_id=evidence.id,
                        run_id=evidence.run_id,
                        work_item_id=evidence.work_item_id,
                        stage=evidence.stage,
                        result=evidence.result,
                        created_at=evidence.created_at,
                        merge_id=None,
                        commit_sha=None if work_item is None else work_item.commit_sha,
                    )
                )

        evidence_rows.sort(
            key=lambda item: (
                item.created_at,
                item.evidence_id,
                item.merge_id or "",
                item.commit_sha or "",
            )
        )

        failure_rows: list[ConstraintFailureRow] = []
        for evidence in evidence_records:
            if evidence.result is EvidenceResult.FAIL:
                failure_rows.append(
                    ConstraintFailureRow(
                        constraint_id=constraint_id,
                        source="evidence",
                        identifier=evidence.id,
                        run_id=evidence.run_id,
                        work_item_id=evidence.work_item_id,
                        evidence_id=evidence.id,
                        incident_id=None,
                        created_at=evidence.created_at,
                        message=evidence.summary or "constraint failure evidence",
                    )
                )

        for incident in self._load_incidents(run_id=run_id):
            if constraint_id not in incident.constraint_ids:
                continue
            failure_rows.append(
                ConstraintFailureRow(
                    constraint_id=constraint_id,
                    source="incident",
                    identifier=incident.id,
                    run_id=incident.run_id,
                    work_item_id=incident.related_work_item_id,
                    evidence_id=incident.evidence_ids[0] if incident.evidence_ids else None,
                    incident_id=incident.id,
                    created_at=incident.created_at,
                    message=incident.message,
                )
            )

        constraint = self._constraint_repo.get(constraint_id)
        if constraint is not None:
            for index, entry in enumerate(constraint.failure_history):
                failure_rows.append(
                    ConstraintFailureRow(
                        constraint_id=constraint_id,
                        source="constraint_history",
                        identifier=f"{constraint_id}:{index}",
                        run_id=None,
                        work_item_id=None,
                        evidence_id=None,
                        incident_id=None,
                        created_at=None,
                        message=entry,
                    )
                )

        failure_rows.sort(
            key=lambda item: (
                item.source,
                _datetime_sort_key(item.created_at),
                item.identifier,
                item.message,
            )
        )
        return ConstraintTraceResult(
            constraint_id=constraint_id,
            evidence_rows=tuple(evidence_rows),
            failure_rows=tuple(failure_rows),
        )

    def coverage_report(
        self,
        spec_map: SpecMap | object,
        *,
        run_id: str | None = None,
    ) -> SpecCoverageReport:
        """Return requirement coverage status for a spec map snapshot."""

        if run_id is not None:
            ids.validate_run_id(run_id)

        requirements = _extract_requirements(spec_map)
        all_constraints = self._load_all_constraints()
        by_requirement: dict[str, list[str]] = {}
        for constraint in all_constraints:
            for requirement_id in constraint.requirement_links:
                by_requirement.setdefault(requirement_id, []).append(constraint.id)

        evidence_by_constraint: dict[str, list[EvidenceRecord]] = {}
        for evidence in self._load_evidence_records(run_id=run_id):
            for constraint_id in evidence.constraint_ids:
                evidence_by_constraint.setdefault(constraint_id, []).append(evidence)
        for records in evidence_by_constraint.values():
            records.sort(key=lambda item: (item.created_at, item.id))

        rows: list[RequirementCoverageRow] = []
        for requirement_id, statement in requirements:
            constraint_ids = tuple(sorted(set(by_requirement.get(requirement_id, []))))
            related_evidence: list[EvidenceRecord] = []
            for constraint_id in constraint_ids:
                related_evidence.extend(evidence_by_constraint.get(constraint_id, []))

            unique_by_id: dict[str, EvidenceRecord] = {}
            for record in related_evidence:
                unique_by_id.setdefault(record.id, record)
            ordered_evidence = tuple(
                sorted(unique_by_id.values(), key=lambda item: (item.created_at, item.id))
            )
            evidence_ids = tuple(item.id for item in ordered_evidence)
            latest_evidence = ordered_evidence[-1].created_at if ordered_evidence else None

            rows.append(
                RequirementCoverageRow(
                    requirement_id=requirement_id,
                    statement=statement,
                    constraint_ids=constraint_ids,
                    evidence_ids=evidence_ids,
                    covered=bool(evidence_ids),
                    latest_evidence_at=latest_evidence,
                )
            )

        rows.sort(key=lambda item: item.requirement_id)
        covered_count = sum(1 for row in rows if row.covered)
        total = len(rows)
        uncovered = tuple(row.requirement_id for row in rows if not row.covered)
        ratio = 0.0 if total == 0 else covered_count / total
        return SpecCoverageReport(
            rows=tuple(rows),
            total_requirements=total,
            covered_requirements=covered_count,
            uncovered_requirement_ids=uncovered,
            coverage_ratio=ratio,
        )

    def verify_integrity(
        self,
        *,
        run_id: str | None = None,
        work_item_id: str | None = None,
    ) -> EvidenceIntegrityReport:
        """Verify manifest/hash integrity for evidence artifact directories."""

        if run_id is not None:
            ids.validate_run_id(run_id)
        if work_item_id is not None:
            ids.validate_work_item_id(work_item_id)

        rows: list[EvidenceIntegrityRow] = []
        for record in self._load_evidence_records(run_id=run_id, work_item_id=work_item_id):
            evidence_dir = self._resolve_evidence_dir(record)
            if evidence_dir is None:
                rows.append(
                    EvidenceIntegrityRow(
                        evidence_id=record.id,
                        run_id=record.run_id,
                        work_item_id=record.work_item_id,
                        stage=record.stage,
                        evidence_dir=None,
                        status="missing",
                        missing_paths=(),
                        hash_mismatches=(),
                        error="evidence directory not found",
                    )
                )
                continue

            try:
                integrity = verify_integrity(evidence_dir)
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    EvidenceIntegrityRow(
                        evidence_id=record.id,
                        run_id=record.run_id,
                        work_item_id=record.work_item_id,
                        stage=record.stage,
                        evidence_dir=evidence_dir,
                        status="error",
                        missing_paths=(),
                        hash_mismatches=(),
                        error=str(exc),
                    )
                )
                continue

            status = "valid" if integrity.is_valid else "invalid"
            rows.append(
                EvidenceIntegrityRow(
                    evidence_id=record.id,
                    run_id=record.run_id,
                    work_item_id=record.work_item_id,
                    stage=record.stage,
                    evidence_dir=evidence_dir,
                    status=status,
                    missing_paths=tuple(sorted(integrity.missing_paths)),
                    hash_mismatches=tuple(sorted(integrity.hash_mismatches)),
                    error=None,
                )
            )

        rows.sort(key=lambda item: (item.run_id, item.work_item_id, item.evidence_id))
        valid_count = sum(1 for row in rows if row.status == "valid")
        invalid_count = sum(1 for row in rows if row.status == "invalid")
        missing_count = sum(1 for row in rows if row.status == "missing")
        error_count = sum(1 for row in rows if row.status == "error")
        return EvidenceIntegrityReport(
            rows=tuple(rows),
            valid_count=valid_count,
            invalid_count=invalid_count,
            missing_count=missing_count,
            error_count=error_count,
        )

    def export_audit_bundle(
        self,
        *,
        run_id: str,
        output_path: PathLike | None = None,
        key_log_paths: Sequence[PathLike] = (),
    ) -> AuditBundleResult:
        """
        Export deterministic audit bundle with evidence, registry snapshot, and key logs.

        The exported archive is fully offline and deterministic for a fixed state snapshot.
        """

        ids.validate_run_id(run_id)
        integrity_report = self.verify_integrity(run_id=run_id)
        broken = tuple(row.evidence_id for row in integrity_report.rows if row.status != "valid")
        if broken:
            raise ValueError(
                "cannot export audit bundle with invalid evidence integrity: " + ", ".join(broken)
            )

        run = self._run_repo.get(run_id)
        if run is None:
            raise ValueError(f"run_id not found: {run_id}")

        constraints = self._load_all_constraints()
        work_items = self._load_all_work_items_for_run(run_id)
        merges = self._load_merges(run_id=run_id)
        incidents = self._load_incidents(run_id=run_id)

        entries: dict[str, bytes] = {}
        entries["snapshots/run.json"] = _encode_json(_redact_json_value(run.to_dict()))
        entries["snapshots/constraint_registry.json"] = _encode_json(
            _redact_json_value(
                [
                    item.to_dict()
                    for item in sorted(
                        constraints,
                        key=lambda item: (item.id, item.created_at, item.category),
                    )
                ]
            )
        )
        entries["snapshots/work_items.json"] = _encode_json(
            _redact_json_value([item.to_dict() for item in work_items])
        )
        entries["logs/merges.json"] = _encode_json(
            _redact_json_value([item.to_dict() for item in merges])
        )
        entries["logs/incidents.json"] = _encode_json(
            _redact_json_value([item.to_dict() for item in incidents])
        )
        entries["snapshots/evidence_integrity.json"] = _encode_json(
            _redact_json_value(
                [
                    {
                        "evidence_id": row.evidence_id,
                        "run_id": row.run_id,
                        "work_item_id": row.work_item_id,
                        "stage": row.stage,
                        "status": row.status,
                        "missing_paths": list(row.missing_paths),
                        "hash_mismatches": list(row.hash_mismatches),
                    }
                    for row in integrity_report.rows
                ]
            )
        )

        with TemporaryDirectory(prefix="nexus-audit-") as temp_dir:
            temp_root = Path(temp_dir)
            evidence_ids: list[str] = []
            for row in integrity_report.rows:
                if row.evidence_dir is None:
                    continue
                archive = export_single_evidence_bundle(
                    row.evidence_dir,
                    output_path=temp_root / f"{row.evidence_id}.audit.zip",
                )
                member_name = f"evidence/{row.evidence_id}.audit.zip"
                entries[member_name] = archive.read_bytes()
                evidence_ids.append(row.evidence_id)

            for source in sorted(
                (Path(item) for item in key_log_paths), key=lambda item: str(item)
            ):
                resolved = source
                if not source.is_absolute():
                    resolved = (self._repo_root / source).resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f"key log path not found: {resolved}")
                log_text = resolved.read_text(encoding="utf-8", errors="replace")
                redacted = redact_text(log_text)
                member_name = "logs/key/" + _normalize_log_member_name(source)
                entries[member_name] = (redacted + "\n").encode("utf-8")

        resolved_output = (
            self._evidence_root / f"{run_id}.audit.bundle.zip"
            if output_path is None
            else Path(output_path).expanduser().resolve()
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        member_names = _write_deterministic_zip(resolved_output, entries)

        return AuditBundleResult(
            bundle_path=resolved_output,
            run_id=run_id,
            evidence_ids=tuple(sorted(evidence_ids)),
            member_names=member_names,
        )

    def _load_all_constraints(self) -> tuple[Constraint, ...]:
        constraints: list[Constraint] = []
        offset = 0
        while True:
            batch = self._constraint_repo.list(
                active_only=False,
                limit=_MAX_PAGE_SIZE,
                offset=offset,
            )
            if not batch:
                break
            constraints.extend(batch)
            offset += len(batch)
            if len(batch) < _MAX_PAGE_SIZE:
                break
        constraints.sort(key=lambda item: (item.id, item.created_at, item.category))
        return tuple(constraints)

    def _load_all_work_items_for_run(self, run_id: str) -> tuple[WorkItem, ...]:
        items: list[WorkItem] = []
        offset = 0
        while True:
            batch = self._work_item_repo.list_for_run(
                run_id,
                limit=_MAX_PAGE_SIZE,
                offset=offset,
            )
            if not batch:
                break
            items.extend(batch)
            offset += len(batch)
            if len(batch) < _MAX_PAGE_SIZE:
                break
        items.sort(key=lambda item: (item.created_at, item.id))
        return tuple(items)

    def _load_evidence_records(
        self,
        *,
        run_id: str | None = None,
        work_item_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        sql = "SELECT payload_json FROM evidence"
        params: list[object] = []
        where_parts: list[str] = []
        if run_id is not None:
            where_parts.append("run_id = ?")
            params.append(run_id)
        if work_item_id is not None:
            where_parts.append("work_item_id = ?")
            params.append(work_item_id)
        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY run_id ASC, work_item_id ASC, created_at ASC, id ASC"
        rows = self._db.query_all(sql, cast("tuple[RowValue, ...]", tuple(params)))
        records: list[EvidenceRecord] = []
        for row in rows:
            payload = _row_text(row, "payload_json", "evidence.payload_json")
            records.append(EvidenceRecord.from_json(payload))
        return tuple(records)

    def _load_work_item_index(self, *, run_id: str | None) -> dict[str, WorkItem]:
        work_items: dict[str, WorkItem] = {}
        if run_id is not None:
            for item in self._load_all_work_items_for_run(run_id):
                work_items[item.id] = item
            return work_items

        rows = self._db.query_all(
            "SELECT payload_json FROM work_items ORDER BY run_id ASC, created_at ASC, id ASC",
            (),
        )
        for row in rows:
            payload = _row_text(row, "payload_json", "work_items.payload_json")
            item = WorkItem.from_json(payload)
            work_items[item.id] = item
        return work_items

    def _build_evidence_merge_index(
        self,
        *,
        run_id: str | None,
    ) -> dict[str, tuple[MergeRecord, ...]]:
        merges = self._load_merges(run_id=run_id)
        index: dict[str, list[MergeRecord]] = {}
        for merge in merges:
            for evidence_id in merge.evidence_ids:
                index.setdefault(evidence_id, []).append(merge)
        out: dict[str, tuple[MergeRecord, ...]] = {}
        for evidence_id, records in index.items():
            records.sort(key=lambda item: (item.merged_at, item.id, item.commit_sha))
            out[evidence_id] = tuple(records)
        return out

    def _load_merges(self, *, run_id: str | None) -> tuple[MergeRecord, ...]:
        if run_id is None:
            rows = self._db.query_all(
                "SELECT payload_json FROM merges ORDER BY run_id ASC, merged_at ASC, id ASC",
                (),
            )
            all_merge_rows = [
                MergeRecord.from_json(_row_text(row, "payload_json", "merges.payload_json"))
                for row in rows
            ]
            return tuple(all_merge_rows)

        scoped_merges: list[MergeRecord] = []
        offset = 0
        while True:
            batch = self._merge_repo.list_for_run(
                run_id,
                limit=_MAX_PAGE_SIZE,
                offset=offset,
            )
            if not batch:
                break
            scoped_merges.extend(batch)
            offset += len(batch)
            if len(batch) < _MAX_PAGE_SIZE:
                break
        scoped_merges.sort(key=lambda item: (item.merged_at, item.id, item.commit_sha))
        return tuple(scoped_merges)

    def _load_incidents(self, *, run_id: str | None) -> tuple[Incident, ...]:
        if run_id is None:
            rows = self._db.query_all(
                "SELECT payload_json FROM incidents ORDER BY run_id ASC, created_at ASC, id ASC",
                (),
            )
            all_incident_rows = [
                Incident.from_json(_row_text(row, "payload_json", "incidents.payload_json"))
                for row in rows
            ]
            return tuple(all_incident_rows)

        scoped_incidents: list[Incident] = []
        offset = 0
        while True:
            batch = self._incident_repo.list_for_run(
                run_id,
                limit=_MAX_PAGE_SIZE,
                offset=offset,
            )
            if not batch:
                break
            scoped_incidents.extend(batch)
            offset += len(batch)
            if len(batch) < _MAX_PAGE_SIZE:
                break
        scoped_incidents.sort(key=lambda item: (item.created_at, item.id))
        return tuple(scoped_incidents)

    def _resolve_evidence_dir(self, record: EvidenceRecord) -> Path | None:
        candidates = _evidence_dir_candidates(self._evidence_root, record)
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return None


def _extract_requirements(spec_map: object) -> tuple[tuple[str, str], ...]:
    if not hasattr(spec_map, "requirements"):
        raise TypeError("spec_map must expose a requirements field")
    requirements = spec_map.requirements
    if not isinstance(requirements, Sequence):
        raise TypeError("spec_map.requirements must be a sequence")

    extracted: list[tuple[str, str]] = []
    for index, item in enumerate(requirements):
        if not hasattr(item, "id") or not hasattr(item, "statement"):
            raise TypeError(f"spec_map.requirements[{index}] must provide id and statement fields")
        typed_item = cast("_RequirementView", item)
        requirement_id = typed_item.id
        statement = typed_item.statement
        if not isinstance(requirement_id, str):
            raise TypeError(f"spec_map.requirements[{index}].id must be a string")
        if not isinstance(statement, str):
            raise TypeError(f"spec_map.requirements[{index}].statement must be a string")
        normalized_id = requirement_id.strip()
        if not normalized_id:
            raise TypeError(f"spec_map.requirements[{index}].id must not be empty")
        # Accept non-canonical IDs (for example NFR-* from ingestion) and keep raw linkage.
        extracted.append((normalized_id, statement.strip()))

    extracted.sort(key=lambda item: item[0])
    return tuple(extracted)


def _evidence_dir_candidates(evidence_root: Path, record: EvidenceRecord) -> tuple[Path, ...]:
    run_segment = _safe_segment(record.run_id)
    work_item_segment = _safe_segment(record.work_item_id)
    stage_segment = _safe_segment(record.stage)
    evidence_segment = _safe_segment(record.id)
    if (
        run_segment is None
        or work_item_segment is None
        or stage_segment is None
        or evidence_segment is None
    ):
        return ()

    attempt_id = _attempt_id_from_metadata(record.metadata)
    candidates: list[Path] = []
    if attempt_id is not None:
        attempt_segment = _safe_segment(attempt_id)
        if attempt_segment is not None:
            candidates.append(
                evidence_root
                / run_segment
                / work_item_segment
                / attempt_segment
                / stage_segment
                / evidence_segment
            )
    candidates.append(
        evidence_root / run_segment / work_item_segment / stage_segment / evidence_segment
    )
    return tuple(candidates)


def _attempt_id_from_metadata(metadata: Mapping[str, JSONValue]) -> str | None:
    raw = metadata.get("attempt_id")
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    return normalized


def _safe_segment(value: str) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized in {".", ".."}:
        return None
    if "/" in normalized or "\\" in normalized:
        return None
    return normalized


def _normalize_log_member_name(path_value: Path) -> str:
    text = str(path_value).replace("\\", "/").strip()
    posix = PurePosixPath(text)
    parts = [part for part in posix.parts if part not in {"", ".", ".."}]
    if not parts:
        raise ValueError(f"invalid key log path: {path_value}")
    return "/".join(parts)


def _write_deterministic_zip(output_path: Path, entries: Mapping[str, bytes]) -> tuple[str, ...]:
    ordered_names = tuple(sorted(entries.keys()))
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for name in ordered_names:
                member = name.replace("\\", "/")
                info = zipfile.ZipInfo(member)
                info.date_time = _ZIP_FIXED_TIMESTAMP
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                info.create_system = 3
                archive.writestr(info, entries[name])
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return ordered_names


def _redact_json_value(value: object) -> JSONValue:
    redacted = redact_structure(value)
    return _to_json_value(redacted)


def _datetime_sort_key(value: datetime | None) -> float:
    if value is None:
        return float("-inf")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC).timestamp()
    return value.timestamp()


def _to_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key in sorted(value.keys(), key=str):
            out[str(key)] = _to_json_value(value[key])
        return out
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        return [_to_json_value(item) for item in value]
    return str(value)


def _encode_json(value: object) -> bytes:
    payload = json.dumps(
        _to_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"{payload}\n".encode()


def _row_text(row: Mapping[str, RowValue], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    return value


__all__ = [
    "AuditBundleResult",
    "ConstraintEvidenceRow",
    "ConstraintFailureRow",
    "ConstraintTraceResult",
    "EvidenceIntegrityReport",
    "EvidenceIntegrityRow",
    "EvidenceLedger",
    "RequirementCoverageRow",
    "RequirementTraceResult",
    "RequirementTraceRow",
    "SpecCoverageReport",
]
