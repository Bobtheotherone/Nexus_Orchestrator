"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/persistence/repositories.py
Last updated: 2026-02-11

Purpose
- Repository/DAO interfaces for reading/writing domain entities to the state DB.

What should be included in this file
- Repositories: RunRepo, WorkItemRepo, ConstraintRepo, EvidenceRepo, MergeRepo, ToolRepo, ProviderCallRepo.
- Query patterns needed by scheduler and UI (e.g., next runnable work items).
- Pagination and indexing considerations.

Functional requirements
- Must provide atomic updates for state transitions (e.g., work item status changes).
- Must support append-only evidence recording semantics.

Non-functional requirements
- Must be efficient; avoid loading entire run history into memory.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, cast

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Attempt,
    Constraint,
    EvidenceRecord,
    Incident,
    MergeRecord,
    Run,
    RunStatus,
    TaskGraph,
    WorkItem,
    WorkItemStatus,
)
from nexus_orchestrator.persistence.state_db import RowValue, SQLParams, StateDB, canonical_json
from nexus_orchestrator.security.redaction import redact_structure, scan_for_secrets
from nexus_orchestrator.utils.hashing import sha256_text

if TYPE_CHECKING:
    import sqlite3

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_MAX_PAGE_SIZE: Final[int] = 1_000

_RUN_ACTIVE_STATUSES: Final[frozenset[str]] = frozenset(
    {RunStatus.PLANNING.value, RunStatus.RUNNING.value}
)
_RUNNABLE_WORK_ITEM_STATUSES: Final[frozenset[str]] = frozenset(
    {WorkItemStatus.PENDING.value, WorkItemStatus.READY.value}
)


@dataclass(slots=True)
class ProviderCallRecord:
    """Typed persistence record for provider API usage accounting."""

    id: str
    attempt_id: str
    provider: str
    tokens: int
    cost_usd: float
    latency_ms: int
    created_at: datetime
    model: str | None = None
    request_id: str | None = None
    error: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _as_non_empty_str(self.id, "ProviderCallRecord.id")
        ids.validate_attempt_id(self.attempt_id)
        self.provider = _as_non_empty_str(self.provider, "ProviderCallRecord.provider")
        self.tokens = _as_non_negative_int(self.tokens, "ProviderCallRecord.tokens")
        self.cost_usd = _as_non_negative_float(self.cost_usd, "ProviderCallRecord.cost_usd")
        self.latency_ms = _as_non_negative_int(self.latency_ms, "ProviderCallRecord.latency_ms")
        self.created_at = _as_utc_datetime(self.created_at, "ProviderCallRecord.created_at")
        if self.model is not None:
            self.model = _as_non_empty_str(self.model, "ProviderCallRecord.model")
        if self.request_id is not None:
            self.request_id = _as_non_empty_str(self.request_id, "ProviderCallRecord.request_id")
        if self.error is not None:
            self.error = _as_non_empty_str(self.error, "ProviderCallRecord.error")
        self.metadata = _as_json_object(self.metadata, "ProviderCallRecord.metadata")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "attempt_id": self.attempt_id,
            "provider": self.provider,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "created_at": _iso8601z(self.created_at),
            "model": self.model,
            "request_id": self.request_id,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProviderCallRecord:
        data = _as_mapping(payload, "ProviderCallRecord")
        return cls(
            id=_as_non_empty_str(data["id"], "ProviderCallRecord.id"),
            attempt_id=_as_non_empty_str(data["attempt_id"], "ProviderCallRecord.attempt_id"),
            provider=_as_non_empty_str(data["provider"], "ProviderCallRecord.provider"),
            tokens=_as_non_negative_int(data["tokens"], "ProviderCallRecord.tokens"),
            cost_usd=_as_non_negative_float(data["cost_usd"], "ProviderCallRecord.cost_usd"),
            latency_ms=_as_non_negative_int(data["latency_ms"], "ProviderCallRecord.latency_ms"),
            created_at=_as_utc_datetime(data["created_at"], "ProviderCallRecord.created_at"),
            model=(
                _as_non_empty_str(data["model"], "ProviderCallRecord.model")
                if data.get("model") is not None
                else None
            ),
            request_id=(
                _as_non_empty_str(data["request_id"], "ProviderCallRecord.request_id")
                if data.get("request_id") is not None
                else None
            ),
            error=(
                _as_non_empty_str(data["error"], "ProviderCallRecord.error")
                if data.get("error") is not None
                else None
            ),
            metadata=_as_json_object(data.get("metadata", {}), "ProviderCallRecord.metadata"),
        )

    @classmethod
    def from_json(cls, payload: str) -> ProviderCallRecord:
        return cls.from_dict(_load_json_object(payload, "ProviderCallRecord"))


@dataclass(slots=True)
class ToolInstallRecord:
    """Typed persistence record for tool install audit trails."""

    id: str
    tool: str
    version: str
    checksum: str
    approved: bool
    installed_at: datetime
    installed_by: str | None = None
    notes: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _as_non_empty_str(self.id, "ToolInstallRecord.id")
        self.tool = _as_non_empty_str(self.tool, "ToolInstallRecord.tool")
        self.version = _as_non_empty_str(self.version, "ToolInstallRecord.version")
        self.checksum = _as_non_empty_str(self.checksum, "ToolInstallRecord.checksum")
        self.approved = bool(self.approved)
        self.installed_at = _as_utc_datetime(self.installed_at, "ToolInstallRecord.installed_at")
        if self.installed_by is not None:
            self.installed_by = _as_non_empty_str(
                self.installed_by, "ToolInstallRecord.installed_by"
            )
        if self.notes is not None:
            self.notes = _as_non_empty_str(self.notes, "ToolInstallRecord.notes")
        self.metadata = _as_json_object(self.metadata, "ToolInstallRecord.metadata")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "tool": self.tool,
            "version": self.version,
            "checksum": self.checksum,
            "approved": self.approved,
            "installed_at": _iso8601z(self.installed_at),
            "installed_by": self.installed_by,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ToolInstallRecord:
        data = _as_mapping(payload, "ToolInstallRecord")
        return cls(
            id=_as_non_empty_str(data["id"], "ToolInstallRecord.id"),
            tool=_as_non_empty_str(data["tool"], "ToolInstallRecord.tool"),
            version=_as_non_empty_str(data["version"], "ToolInstallRecord.version"),
            checksum=_as_non_empty_str(data["checksum"], "ToolInstallRecord.checksum"),
            approved=bool(data["approved"]),
            installed_at=_as_utc_datetime(data["installed_at"], "ToolInstallRecord.installed_at"),
            installed_by=(
                _as_non_empty_str(data["installed_by"], "ToolInstallRecord.installed_by")
                if data.get("installed_by") is not None
                else None
            ),
            notes=(
                _as_non_empty_str(data["notes"], "ToolInstallRecord.notes")
                if data.get("notes") is not None
                else None
            ),
            metadata=_as_json_object(data.get("metadata", {}), "ToolInstallRecord.metadata"),
        )

    @classmethod
    def from_json(cls, payload: str) -> ToolInstallRecord:
        return cls.from_dict(_load_json_object(payload, "ToolInstallRecord"))


class _BaseRepo:
    def __init__(self, db: StateDB) -> None:
        self._db = db
        self._db.migrate()

    @staticmethod
    def _validate_page(limit: int, offset: int) -> None:
        if limit <= 0 or limit > _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be in [1, {_MAX_PAGE_SIZE}]")
        if offset < 0:
            raise ValueError("offset must be >= 0")


class RunRepo(_BaseRepo):
    """Repository for run state and lifecycle transitions."""

    def add(self, run: Run) -> Run:
        return self._persist(run, upsert=False)

    def upsert(self, run: Run) -> Run:
        return self._persist(run, upsert=True)

    def save(self, run: Run) -> Run:
        return self.upsert(run)

    def get(self, run_id: str) -> Run | None:
        ids.validate_run_id(run_id)
        row = self._db.query_one("SELECT payload_json FROM runs WHERE id = ?", (run_id,))
        if row is None:
            return None
        return Run.from_json(_row_text(row, "payload_json", "runs.payload_json"))

    def list(
        self,
        *,
        status: RunStatus | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Run]:
        self._validate_page(limit, offset)
        sql = "SELECT payload_json FROM runs"
        params: list[object] = []
        if status is not None:
            parsed_status = _as_run_status(status, "status")
            sql += " WHERE status = ?"
            params.append(parsed_status.value)
        sql += " ORDER BY started_at DESC, id DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        rows = self._db.query_all(sql, cast("SQLParams", tuple(params)))
        return [Run.from_json(_row_text(row, "payload_json", "runs.payload_json")) for row in rows]

    def set_status(
        self,
        run_id: str,
        status: RunStatus | str,
        *,
        finished_at: datetime | None = None,
    ) -> Run:
        ids.validate_run_id(run_id)
        next_status = _as_run_status(status, "status")
        run = self.get(run_id)
        if run is None:
            raise ValueError(f"run_id not found: {run_id}")

        payload = run.to_dict()
        payload["status"] = next_status.value

        terminal = {
            RunStatus.FAILED,
            RunStatus.COMPLETED,
            RunStatus.CANCELLED,
        }
        if finished_at is not None:
            payload["finished_at"] = _iso8601z(_as_utc_datetime(finished_at, "finished_at"))
        elif next_status in terminal:
            payload["finished_at"] = _iso8601z(_utc_now())

        updated = Run.from_dict(payload)
        return self.upsert(updated)

    def attach_work_item(self, run_id: str, work_item_id: str) -> Run:
        ids.validate_run_id(run_id)
        ids.validate_work_item_id(work_item_id)
        run = self.get(run_id)
        if run is None:
            raise ValueError(f"run_id not found: {run_id}")
        if work_item_id in run.work_item_ids:
            return run

        payload = run.to_dict()
        current_ids = list(run.work_item_ids)
        current_ids.append(work_item_id)
        payload["work_item_ids"] = cast("JSONValue", current_ids)
        return self.upsert(Run.from_dict(payload))

    def _persist(self, run: Run, *, upsert: bool) -> Run:
        sanitized = _sanitize_run(run)
        payload_json = sanitized.to_json()
        budget_json = sanitized.budget.to_json()
        metadata_json = canonical_json(sanitized.metadata)
        work_item_ids_json = canonical_json(list(sanitized.work_item_ids))
        config_hash = sha256_text(metadata_json)
        started_at = _iso8601z(sanitized.started_at)
        finished_at = (
            _iso8601z(sanitized.finished_at) if sanitized.finished_at is not None else None
        )
        created_at = started_at
        updated_at = _iso8601z(_utc_now())

        if upsert:
            sql = """
            INSERT INTO runs (
                id,
                spec_path,
                status,
                started_at,
                finished_at,
                config_hash,
                risk_tier,
                budget_json,
                metadata_json,
                work_item_ids_json,
                payload_json,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                spec_path=excluded.spec_path,
                status=excluded.status,
                started_at=excluded.started_at,
                finished_at=excluded.finished_at,
                config_hash=excluded.config_hash,
                risk_tier=excluded.risk_tier,
                budget_json=excluded.budget_json,
                metadata_json=excluded.metadata_json,
                work_item_ids_json=excluded.work_item_ids_json,
                payload_json=excluded.payload_json,
                updated_at=excluded.updated_at
            """
        else:
            sql = """
            INSERT INTO runs (
                id,
                spec_path,
                status,
                started_at,
                finished_at,
                config_hash,
                risk_tier,
                budget_json,
                metadata_json,
                work_item_ids_json,
                payload_json,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        params: SQLParams = (
            sanitized.id,
            sanitized.spec_path,
            _as_run_status(sanitized.status, "Run.status").value,
            started_at,
            finished_at,
            config_hash,
            sanitized.risk_tier.value,
            budget_json,
            metadata_json,
            work_item_ids_json,
            payload_json,
            created_at,
            updated_at,
        )

        self._db.execute(sql, params)
        return sanitized


class WorkItemRepo(_BaseRepo):
    """Repository for work-item lifecycle, DAG edges, and runnable selection."""

    def add(self, run_id: str, work_item: WorkItem) -> WorkItem:
        return self._persist(run_id, work_item, upsert=False)

    def upsert(self, run_id: str, work_item: WorkItem) -> WorkItem:
        return self._persist(run_id, work_item, upsert=True)

    def save(self, run_id: str, work_item: WorkItem) -> WorkItem:
        return self.upsert(run_id, work_item)

    def get(self, work_item_id: str) -> WorkItem | None:
        ids.validate_work_item_id(work_item_id)
        row = self._db.query_one(
            "SELECT payload_json FROM work_items WHERE id = ?", (work_item_id,)
        )
        if row is None:
            return None
        return WorkItem.from_json(_row_text(row, "payload_json", "work_items.payload_json"))

    def get_for_run(self, run_id: str, work_item_id: str) -> WorkItem | None:
        ids.validate_run_id(run_id)
        ids.validate_work_item_id(work_item_id)
        row = self._db.query_one(
            "SELECT payload_json FROM work_items WHERE run_id = ? AND id = ?",
            (run_id, work_item_id),
        )
        if row is None:
            return None
        return WorkItem.from_json(_row_text(row, "payload_json", "work_items.payload_json"))

    def list_for_run(
        self,
        run_id: str,
        *,
        statuses: Sequence[WorkItemStatus | str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkItem]:
        ids.validate_run_id(run_id)
        self._validate_page(limit, offset)

        params: list[object] = [run_id]
        sql = "SELECT payload_json FROM work_items WHERE run_id = ?"

        if statuses is not None:
            parsed = [_as_work_item_status(item, "statuses[]").value for item in statuses]
            if parsed:
                placeholders = ",".join("?" for _ in parsed)
                sql += f" AND status IN ({placeholders})"
                params.extend(parsed)

        sql += " ORDER BY created_at ASC, id ASC LIMIT ? OFFSET ?"
        params.extend((limit, offset))

        rows = self._db.query_all(sql, cast("SQLParams", tuple(params)))
        return [
            WorkItem.from_json(_row_text(row, "payload_json", "work_items.payload_json"))
            for row in rows
        ]

    def set_status(
        self,
        work_item_id: str,
        status: WorkItemStatus | str,
        *,
        run_id: str | None = None,
    ) -> WorkItem:
        ids.validate_work_item_id(work_item_id)
        if run_id is not None:
            ids.validate_run_id(run_id)

        next_status = _as_work_item_status(status, "status")
        with self._db.transaction() as conn:
            row = self._db.query_one(
                "SELECT run_id, payload_json FROM work_items WHERE id = ?",
                (work_item_id,),
                conn=conn,
            )
            if row is None:
                raise ValueError(f"work_item_id not found: {work_item_id}")
            row_run_id = _row_text(row, "run_id", "work_items.run_id")
            if run_id is not None and row_run_id != run_id:
                raise ValueError(
                    f"work_item_id {work_item_id} belongs to run {row_run_id}, not {run_id}"
                )

            current = WorkItem.from_json(_row_text(row, "payload_json", "work_items.payload_json"))
            payload = current.to_dict()
            payload["status"] = next_status.value
            payload["updated_at"] = _iso8601z(_utc_now())
            updated = WorkItem.from_dict(payload)
            self._persist_row(conn, row_run_id, updated, upsert=True)
            return updated

    def get_next_runnable(self, run_id: str, *, limit: int = 32) -> list[WorkItem]:
        ids.validate_run_id(run_id)
        if limit <= 0:
            raise ValueError("limit must be > 0")

        sql = """
        SELECT wi.payload_json
        FROM work_items wi
        JOIN runs r ON r.id = wi.run_id
        WHERE wi.run_id = ?
          AND r.status IN ('planning', 'running')
          AND wi.status IN ('pending', 'ready')
          AND NOT EXISTS (
              SELECT 1
              FROM task_graph_edges dep
              JOIN work_items parent ON parent.id = dep.parent_id
              WHERE dep.run_id = wi.run_id
                AND dep.child_id = wi.id
                AND parent.status <> 'merged'
          )
        ORDER BY
            CASE wi.risk_tier
                WHEN 'critical' THEN 4
                WHEN 'high' THEN 3
                WHEN 'medium' THEN 2
                ELSE 1
            END DESC,
            wi.created_at ASC,
            wi.id ASC
        LIMIT ?
        """

        rows = self._db.query_all(sql, (run_id, limit))
        return [
            WorkItem.from_json(_row_text(row, "payload_json", "work_items.payload_json"))
            for row in rows
        ]

    def _persist(self, run_id: str, work_item: WorkItem, *, upsert: bool) -> WorkItem:
        ids.validate_run_id(run_id)
        sanitized = _sanitize_work_item(work_item)
        if sanitized.id in sanitized.dependencies:
            raise ValueError(f"work_item {sanitized.id} cannot depend on itself")

        with self._db.transaction() as conn:
            self._ensure_run_exists(run_id, conn)
            existing_row = self._db.query_one(
                "SELECT run_id FROM work_items WHERE id = ?",
                (sanitized.id,),
                conn=conn,
            )
            if existing_row is not None:
                existing_run_id = _row_text(existing_row, "run_id", "work_items.run_id")
                if not upsert:
                    raise ValueError(f"work_item already exists: {sanitized.id}")
                if existing_run_id != run_id:
                    raise ValueError(
                        f"work_item_id {sanitized.id} belongs to run {existing_run_id}, not {run_id}"
                    )

            self._persist_row(conn, run_id, sanitized, upsert=upsert)
            return sanitized

    def _persist_row(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        work_item: WorkItem,
        *,
        upsert: bool,
    ) -> None:
        _persist_work_item_row(self._db, conn, run_id=run_id, work_item=work_item, upsert=upsert)

    def _ensure_run_exists(self, run_id: str, conn: sqlite3.Connection) -> None:
        row = self._db.query_one("SELECT id FROM runs WHERE id = ?", (run_id,), conn=conn)
        if row is None:
            raise ValueError(f"run_id not found: {run_id}")


class ConstraintRepo(_BaseRepo):
    """Repository for active/inactive constraints."""

    def add(self, constraint: Constraint, *, active: bool = True) -> Constraint:
        return self._persist(constraint, active=active, upsert=False)

    def upsert(self, constraint: Constraint, *, active: bool = True) -> Constraint:
        return self._persist(constraint, active=active, upsert=True)

    def save(self, constraint: Constraint, *, active: bool = True) -> Constraint:
        return self.upsert(constraint, active=active)

    def get(self, constraint_id: str) -> Constraint | None:
        ids.validate_constraint_id(constraint_id)
        row = self._db.query_one(
            "SELECT payload_json FROM constraints WHERE id = ?", (constraint_id,)
        )
        if row is None:
            return None
        return Constraint.from_json(_row_text(row, "payload_json", "constraints.payload_json"))

    def list(
        self,
        *,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Constraint]:
        self._validate_page(limit, offset)
        if active_only:
            rows = self._db.query_all(
                """
                SELECT payload_json FROM constraints
                WHERE active = 1
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        else:
            rows = self._db.query_all(
                """
                SELECT payload_json FROM constraints
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        return [
            Constraint.from_json(_row_text(row, "payload_json", "constraints.payload_json"))
            for row in rows
        ]

    def set_active(self, constraint_id: str, *, active: bool) -> None:
        ids.validate_constraint_id(constraint_id)
        updated = self._db.execute(
            "UPDATE constraints SET active = ? WHERE id = ?",
            (1 if active else 0, constraint_id),
        )
        if updated == 0:
            raise ValueError(f"constraint_id not found: {constraint_id}")

    def _persist(self, constraint: Constraint, *, active: bool, upsert: bool) -> Constraint:
        sanitized = _sanitize_constraint(constraint)
        payload_json = sanitized.to_json()

        if upsert:
            sql = """
            INSERT INTO constraints (
                id,
                severity,
                category,
                checker_binding,
                active,
                created_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                severity=excluded.severity,
                category=excluded.category,
                checker_binding=excluded.checker_binding,
                active=excluded.active,
                created_at=excluded.created_at,
                payload_json=excluded.payload_json
            """
        else:
            sql = """
            INSERT INTO constraints (
                id,
                severity,
                category,
                checker_binding,
                active,
                created_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """

        self._db.execute(
            sql,
            (
                sanitized.id,
                sanitized.severity.value,
                sanitized.category,
                sanitized.checker_binding,
                1 if active else 0,
                _iso8601z(sanitized.created_at),
                payload_json,
            ),
        )
        return sanitized


class EvidenceRepo(_BaseRepo):
    """Append-only repository for evidence records."""

    def add(self, record: EvidenceRecord) -> EvidenceRecord:
        sanitized = _sanitize_evidence(record)

        with self._db.transaction() as conn:
            wi_row = self._db.query_one(
                "SELECT run_id, payload_json FROM work_items WHERE id = ?",
                (sanitized.work_item_id,),
                conn=conn,
            )
            if wi_row is None:
                raise ValueError(f"work_item_id not found: {sanitized.work_item_id}")
            work_item_run_id = _row_text(wi_row, "run_id", "work_items.run_id")
            if work_item_run_id != sanitized.run_id:
                raise ValueError(
                    f"evidence run_id mismatch: work_item={work_item_run_id}, record={sanitized.run_id}"
                )
            self._db.execute(
                """
                INSERT INTO evidence (
                    id,
                    work_item_id,
                    run_id,
                    stage,
                    result,
                    created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sanitized.id,
                    sanitized.work_item_id,
                    sanitized.run_id,
                    sanitized.stage,
                    sanitized.result.value,
                    _iso8601z(sanitized.created_at),
                    sanitized.to_json(),
                ),
                conn=conn,
            )
            _append_evidence_to_work_item(conn, self._db, sanitized)
        return sanitized

    def save(self, record: EvidenceRecord) -> EvidenceRecord:
        return self.add(record)

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        ids.validate_evidence_id(evidence_id)
        row = self._db.query_one("SELECT payload_json FROM evidence WHERE id = ?", (evidence_id,))
        if row is None:
            return None
        return EvidenceRecord.from_json(_row_text(row, "payload_json", "evidence.payload_json"))

    def list_for_work_item(
        self,
        work_item_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EvidenceRecord]:
        ids.validate_work_item_id(work_item_id)
        self._validate_page(limit, offset)
        rows = self._db.query_all(
            """
            SELECT payload_json
            FROM evidence
            WHERE work_item_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (work_item_id, limit, offset),
        )
        return [
            EvidenceRecord.from_json(_row_text(row, "payload_json", "evidence.payload_json"))
            for row in rows
        ]


class AttemptRepo(_BaseRepo):
    """Repository for synthesis attempts."""

    def add(self, attempt: Attempt) -> Attempt:
        return self._persist(attempt, upsert=False)

    def upsert(self, attempt: Attempt) -> Attempt:
        return self._persist(attempt, upsert=True)

    def save(self, attempt: Attempt) -> Attempt:
        return self.upsert(attempt)

    def get(self, attempt_id: str) -> Attempt | None:
        ids.validate_attempt_id(attempt_id)
        row = self._db.query_one("SELECT payload_json FROM attempts WHERE id = ?", (attempt_id,))
        if row is None:
            return None
        return Attempt.from_json(_row_text(row, "payload_json", "attempts.payload_json"))

    def list_for_work_item(
        self,
        work_item_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Attempt]:
        ids.validate_work_item_id(work_item_id)
        self._validate_page(limit, offset)
        rows = self._db.query_all(
            """
            SELECT payload_json
            FROM attempts
            WHERE work_item_id = ?
            ORDER BY iteration DESC
            LIMIT ? OFFSET ?
            """,
            (work_item_id, limit, offset),
        )
        return [
            Attempt.from_json(_row_text(row, "payload_json", "attempts.payload_json"))
            for row in rows
        ]

    def _persist(self, attempt: Attempt, *, upsert: bool) -> Attempt:
        sanitized = _sanitize_attempt(attempt)

        with self._db.transaction() as conn:
            wi_row = self._db.query_one(
                "SELECT run_id FROM work_items WHERE id = ?",
                (sanitized.work_item_id,),
                conn=conn,
            )
            if wi_row is None:
                raise ValueError(f"work_item_id not found: {sanitized.work_item_id}")
            wi_run_id = _row_text(wi_row, "run_id", "work_items.run_id")
            if wi_run_id != sanitized.run_id:
                raise ValueError(
                    f"attempt run_id mismatch: work_item={wi_run_id}, attempt={sanitized.run_id}"
                )

            if upsert:
                sql = """
                INSERT INTO attempts (
                    id,
                    work_item_id,
                    run_id,
                    iteration,
                    provider,
                    model,
                    result,
                    cost_usd,
                    created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    work_item_id=excluded.work_item_id,
                    run_id=excluded.run_id,
                    iteration=excluded.iteration,
                    provider=excluded.provider,
                    model=excluded.model,
                    result=excluded.result,
                    cost_usd=excluded.cost_usd,
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """
            else:
                sql = """
                INSERT INTO attempts (
                    id,
                    work_item_id,
                    run_id,
                    iteration,
                    provider,
                    model,
                    result,
                    cost_usd,
                    created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

            self._db.execute(
                sql,
                (
                    sanitized.id,
                    sanitized.work_item_id,
                    sanitized.run_id,
                    sanitized.iteration,
                    sanitized.provider,
                    sanitized.model,
                    sanitized.result.value,
                    sanitized.cost_usd,
                    _iso8601z(sanitized.created_at),
                    sanitized.to_json(),
                ),
                conn=conn,
            )

        return sanitized


class MergeRepo(_BaseRepo):
    """Repository for merge queue records and merge-completion transitions."""

    def add(self, merge: MergeRecord) -> MergeRecord:
        sanitized = _sanitize_merge(merge)

        with self._db.transaction() as conn:
            wi_row = self._db.query_one(
                "SELECT run_id, payload_json FROM work_items WHERE id = ?",
                (sanitized.work_item_id,),
                conn=conn,
            )
            if wi_row is None:
                raise ValueError(f"work_item_id not found: {sanitized.work_item_id}")
            wi_run_id = _row_text(wi_row, "run_id", "work_items.run_id")
            if wi_run_id != sanitized.run_id:
                raise ValueError(
                    f"merge run_id mismatch: work_item={wi_run_id}, merge={sanitized.run_id}"
                )
            for evidence_id in sanitized.evidence_ids:
                evidence_row = self._db.query_one(
                    "SELECT id FROM evidence WHERE id = ?",
                    (evidence_id,),
                    conn=conn,
                )
                if evidence_row is None:
                    raise ValueError(f"evidence_id not found: {evidence_id}")

            self._db.execute(
                """
                INSERT INTO merges (
                    id,
                    work_item_id,
                    run_id,
                    commit_sha,
                    merged_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sanitized.id,
                    sanitized.work_item_id,
                    sanitized.run_id,
                    sanitized.commit_sha,
                    _iso8601z(sanitized.merged_at),
                    sanitized.to_json(),
                ),
                conn=conn,
            )

            current_item = WorkItem.from_json(
                _row_text(wi_row, "payload_json", "work_items.payload_json")
            )
            payload = current_item.to_dict()
            payload["status"] = WorkItemStatus.MERGED.value
            payload["commit_sha"] = sanitized.commit_sha
            payload["updated_at"] = _iso8601z(_utc_now())

            evidence_ids = list(current_item.evidence_ids)
            for evidence_id in sanitized.evidence_ids:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
            payload["evidence_ids"] = cast("JSONValue", evidence_ids)
            updated_item = WorkItem.from_dict(payload)
            _persist_work_item_row(
                self._db,
                conn,
                run_id=sanitized.run_id,
                work_item=updated_item,
                upsert=True,
            )

        return sanitized

    def save(self, merge: MergeRecord) -> MergeRecord:
        return self.add(merge)

    def get(self, merge_id: str) -> MergeRecord | None:
        ids.validate_merge_id(merge_id)
        row = self._db.query_one("SELECT payload_json FROM merges WHERE id = ?", (merge_id,))
        if row is None:
            return None
        return MergeRecord.from_json(_row_text(row, "payload_json", "merges.payload_json"))

    def list_for_run(self, run_id: str, *, limit: int = 100, offset: int = 0) -> list[MergeRecord]:
        ids.validate_run_id(run_id)
        self._validate_page(limit, offset)
        rows = self._db.query_all(
            """
            SELECT payload_json
            FROM merges
            WHERE run_id = ?
            ORDER BY merged_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (run_id, limit, offset),
        )
        return [
            MergeRecord.from_json(_row_text(row, "payload_json", "merges.payload_json"))
            for row in rows
        ]


class ProviderCallRepo(_BaseRepo):
    """Repository for provider call accounting and latency/cost traces."""

    def add(self, record: ProviderCallRecord) -> ProviderCallRecord:
        return self._persist(record, upsert=False)

    def upsert(self, record: ProviderCallRecord) -> ProviderCallRecord:
        return self._persist(record, upsert=True)

    def save(self, record: ProviderCallRecord) -> ProviderCallRecord:
        return self.upsert(record)

    def get(self, record_id: str) -> ProviderCallRecord | None:
        record_id = _as_non_empty_str(record_id, "record_id")
        row = self._db.query_one(
            "SELECT payload_json FROM provider_calls WHERE id = ?", (record_id,)
        )
        if row is None:
            return None
        return ProviderCallRecord.from_json(
            _row_text(row, "payload_json", "provider_calls.payload_json")
        )

    def list_for_attempt(
        self,
        attempt_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProviderCallRecord]:
        ids.validate_attempt_id(attempt_id)
        self._validate_page(limit, offset)
        rows = self._db.query_all(
            """
            SELECT payload_json
            FROM provider_calls
            WHERE attempt_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (attempt_id, limit, offset),
        )
        return [
            ProviderCallRecord.from_json(
                _row_text(row, "payload_json", "provider_calls.payload_json")
            )
            for row in rows
        ]

    def _persist(self, record: ProviderCallRecord, *, upsert: bool) -> ProviderCallRecord:
        sanitized = _sanitize_provider_call(record)
        with self._db.transaction() as conn:
            attempt_row = self._db.query_one(
                "SELECT id FROM attempts WHERE id = ?",
                (sanitized.attempt_id,),
                conn=conn,
            )
            if attempt_row is None:
                raise ValueError(f"attempt_id not found: {sanitized.attempt_id}")

            if upsert:
                sql = """
                INSERT INTO provider_calls (
                    id,
                    attempt_id,
                    provider,
                    tokens,
                    cost_usd,
                    latency_ms,
                    created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    attempt_id=excluded.attempt_id,
                    provider=excluded.provider,
                    tokens=excluded.tokens,
                    cost_usd=excluded.cost_usd,
                    latency_ms=excluded.latency_ms,
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """
            else:
                sql = """
                INSERT INTO provider_calls (
                    id,
                    attempt_id,
                    provider,
                    tokens,
                    cost_usd,
                    latency_ms,
                    created_at,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """

            self._db.execute(
                sql,
                (
                    sanitized.id,
                    sanitized.attempt_id,
                    sanitized.provider,
                    sanitized.tokens,
                    sanitized.cost_usd,
                    sanitized.latency_ms,
                    _iso8601z(sanitized.created_at),
                    sanitized.to_json(),
                ),
                conn=conn,
            )

        return sanitized


class IncidentRepo(_BaseRepo):
    """Repository for incident recording and lookup."""

    def add(self, incident: Incident) -> Incident:
        sanitized = _sanitize_incident(incident)
        self._db.execute(
            """
            INSERT INTO incidents (
                id,
                run_id,
                related_work_item_id,
                category,
                message,
                created_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sanitized.id,
                sanitized.run_id,
                sanitized.related_work_item_id,
                sanitized.category,
                sanitized.message,
                _iso8601z(sanitized.created_at),
                sanitized.to_json(),
            ),
        )
        return sanitized

    def save(self, incident: Incident) -> Incident:
        return self.add(incident)

    def get(self, incident_id: str) -> Incident | None:
        ids.validate_incident_id(incident_id)
        row = self._db.query_one("SELECT payload_json FROM incidents WHERE id = ?", (incident_id,))
        if row is None:
            return None
        return Incident.from_json(_row_text(row, "payload_json", "incidents.payload_json"))

    def list_for_run(self, run_id: str, *, limit: int = 100, offset: int = 0) -> list[Incident]:
        ids.validate_run_id(run_id)
        self._validate_page(limit, offset)
        rows = self._db.query_all(
            """
            SELECT payload_json
            FROM incidents
            WHERE run_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (run_id, limit, offset),
        )
        return [
            Incident.from_json(_row_text(row, "payload_json", "incidents.payload_json"))
            for row in rows
        ]


class TaskGraphRepo(_BaseRepo):
    """Repository for task-graph snapshots and normalized edges."""

    def add(self, graph: TaskGraph) -> TaskGraph:
        return self._persist(graph, upsert=False)

    def upsert(self, graph: TaskGraph) -> TaskGraph:
        return self._persist(graph, upsert=True)

    def save(self, graph: TaskGraph) -> TaskGraph:
        return self.upsert(graph)

    def get(self, run_id: str) -> TaskGraph | None:
        ids.validate_run_id(run_id)
        row = self._db.query_one("SELECT payload_json FROM task_graphs WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return TaskGraph.from_json(_row_text(row, "payload_json", "task_graphs.payload_json"))

    def get_edges(self, run_id: str) -> tuple[tuple[str, str], ...]:
        ids.validate_run_id(run_id)
        rows = self._db.query_all(
            """
            SELECT parent_id, child_id
            FROM task_graph_edges
            WHERE run_id = ?
            ORDER BY parent_id ASC, child_id ASC
            """,
            (run_id,),
        )
        edges: list[tuple[str, str]] = []
        for row in rows:
            parent_id = _row_text(row, "parent_id", "task_graph_edges.parent_id")
            child_id = _row_text(row, "child_id", "task_graph_edges.child_id")
            edges.append((parent_id, child_id))
        return tuple(edges)

    def _persist(self, graph: TaskGraph, *, upsert: bool) -> TaskGraph:
        sanitized = _sanitize_task_graph(graph)
        with self._db.transaction() as conn:
            run_row = self._db.query_one(
                "SELECT id FROM runs WHERE id = ?", (sanitized.run_id,), conn=conn
            )
            if run_row is None:
                raise ValueError(f"run_id not found: {sanitized.run_id}")

            if upsert:
                sql = """
                INSERT INTO task_graphs (run_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    payload_json=excluded.payload_json
                """
            else:
                sql = "INSERT INTO task_graphs (run_id, created_at, payload_json) VALUES (?, ?, ?)"

            self._db.execute(
                sql,
                (
                    sanitized.run_id,
                    _iso8601z(sanitized.created_at),
                    sanitized.to_json(),
                ),
                conn=conn,
            )

            self._db.execute(
                "DELETE FROM task_graph_edges WHERE run_id = ?",
                (sanitized.run_id,),
                conn=conn,
            )
            for parent_id, child_id in sanitized.edges:
                self._db.execute(
                    """
                    INSERT INTO task_graph_edges (run_id, parent_id, child_id, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (sanitized.run_id, parent_id, child_id, _iso8601z(sanitized.created_at)),
                    conn=conn,
                )
        return sanitized


class ToolInstallRepo(_BaseRepo):
    """Repository for tool-install audit metadata."""

    def add(self, record: ToolInstallRecord) -> ToolInstallRecord:
        return self._persist(record, upsert=False)

    def upsert(self, record: ToolInstallRecord) -> ToolInstallRecord:
        return self._persist(record, upsert=True)

    def save(self, record: ToolInstallRecord) -> ToolInstallRecord:
        return self.upsert(record)

    def get(self, record_id: str) -> ToolInstallRecord | None:
        record_id = _as_non_empty_str(record_id, "record_id")
        row = self._db.query_one(
            "SELECT payload_json FROM tool_installs WHERE id = ?", (record_id,)
        )
        if row is None:
            return None
        return ToolInstallRecord.from_json(
            _row_text(row, "payload_json", "tool_installs.payload_json")
        )

    def list(
        self,
        *,
        tool: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ToolInstallRecord]:
        self._validate_page(limit, offset)
        if tool is None:
            rows = self._db.query_all(
                """
                SELECT payload_json
                FROM tool_installs
                ORDER BY installed_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        else:
            parsed_tool = _as_non_empty_str(tool, "tool")
            rows = self._db.query_all(
                """
                SELECT payload_json
                FROM tool_installs
                WHERE tool = ?
                ORDER BY installed_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (parsed_tool, limit, offset),
            )

        return [
            ToolInstallRecord.from_json(
                _row_text(row, "payload_json", "tool_installs.payload_json")
            )
            for row in rows
        ]

    def _persist(self, record: ToolInstallRecord, *, upsert: bool) -> ToolInstallRecord:
        sanitized = _sanitize_tool_install(record)
        if upsert:
            sql = """
            INSERT INTO tool_installs (
                id,
                tool,
                version,
                checksum,
                approved,
                installed_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                tool=excluded.tool,
                version=excluded.version,
                checksum=excluded.checksum,
                approved=excluded.approved,
                installed_at=excluded.installed_at,
                payload_json=excluded.payload_json
            """
        else:
            sql = """
            INSERT INTO tool_installs (
                id,
                tool,
                version,
                checksum,
                approved,
                installed_at,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """

        self._db.execute(
            sql,
            (
                sanitized.id,
                sanitized.tool,
                sanitized.version,
                sanitized.checksum,
                1 if sanitized.approved else 0,
                _iso8601z(sanitized.installed_at),
                sanitized.to_json(),
            ),
        )
        return sanitized


ToolRepo = ToolInstallRepo


def _persist_work_item_row(
    db: StateDB,
    conn: sqlite3.Connection,
    *,
    run_id: str,
    work_item: WorkItem,
    upsert: bool,
) -> None:
    payload_json = work_item.to_json()
    envelope_hash = sha256_text(work_item.constraint_envelope.to_json())

    sql: str
    if upsert:
        sql = """
        INSERT INTO work_items (
            id,
            run_id,
            status,
            risk_tier,
            constraint_envelope_hash,
            created_at,
            updated_at,
            payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            run_id=excluded.run_id,
            status=excluded.status,
            risk_tier=excluded.risk_tier,
            constraint_envelope_hash=excluded.constraint_envelope_hash,
            created_at=excluded.created_at,
            updated_at=excluded.updated_at,
            payload_json=excluded.payload_json
        """
    else:
        sql = """
        INSERT INTO work_items (
            id,
            run_id,
            status,
            risk_tier,
            constraint_envelope_hash,
            created_at,
            updated_at,
            payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

    params: SQLParams = (
        work_item.id,
        run_id,
        work_item.status.value,
        work_item.risk_tier.value,
        envelope_hash,
        _iso8601z(work_item.created_at),
        _iso8601z(work_item.updated_at),
        payload_json,
    )
    db.execute(sql, params, conn=conn)

    db.execute(
        "DELETE FROM task_graph_edges WHERE run_id = ? AND child_id = ?",
        (run_id, work_item.id),
        conn=conn,
    )
    for parent_id in work_item.dependencies:
        db.execute(
            """
            INSERT INTO task_graph_edges (run_id, parent_id, child_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, parent_id, work_item.id, _iso8601z(work_item.created_at)),
            conn=conn,
        )


def _append_evidence_to_work_item(
    conn: sqlite3.Connection, db: StateDB, evidence: EvidenceRecord
) -> None:
    row = db.query_one(
        "SELECT run_id, payload_json FROM work_items WHERE id = ?",
        (evidence.work_item_id,),
        conn=conn,
    )
    if row is None:
        raise ValueError(f"work_item_id not found: {evidence.work_item_id}")

    work_item = WorkItem.from_json(_row_text(row, "payload_json", "work_items.payload_json"))
    payload = work_item.to_dict()
    evidence_ids = list(work_item.evidence_ids)
    if evidence.id not in evidence_ids:
        evidence_ids.append(evidence.id)
    payload["evidence_ids"] = cast("JSONValue", evidence_ids)
    payload["updated_at"] = _iso8601z(_utc_now())
    updated = WorkItem.from_dict(payload)
    _persist_work_item_row(db, conn, run_id=evidence.run_id, work_item=updated, upsert=True)


def _sanitize_run(run: Run) -> Run:
    payload = run.to_dict()
    _assert_no_secret_values(payload, "Run")
    return Run.from_dict(payload)


def _sanitize_work_item(work_item: WorkItem) -> WorkItem:
    payload = work_item.to_dict()
    _assert_no_secret_values(payload, "WorkItem")
    return WorkItem.from_dict(payload)


def _sanitize_constraint(constraint: Constraint) -> Constraint:
    payload = constraint.to_dict()
    _assert_no_secret_values(payload, "Constraint")
    return Constraint.from_dict(payload)


def _sanitize_evidence(record: EvidenceRecord) -> EvidenceRecord:
    payload = record.to_dict()
    _assert_no_secret_values(payload, "EvidenceRecord")
    return EvidenceRecord.from_dict(payload)


def _sanitize_attempt(attempt: Attempt) -> Attempt:
    payload = attempt.to_dict()
    _assert_no_secret_values(payload, "Attempt")
    return Attempt.from_dict(payload)


def _sanitize_merge(merge: MergeRecord) -> MergeRecord:
    payload = merge.to_dict()
    _assert_no_secret_values(payload, "MergeRecord")
    return MergeRecord.from_dict(payload)


def _sanitize_incident(incident: Incident) -> Incident:
    payload = incident.to_dict()
    _assert_no_secret_values(payload, "Incident")
    return Incident.from_dict(payload)


def _sanitize_task_graph(graph: TaskGraph) -> TaskGraph:
    payload = graph.to_dict()
    _assert_no_secret_values(payload, "TaskGraph")
    return TaskGraph.from_dict(payload)


def _sanitize_provider_call(record: ProviderCallRecord) -> ProviderCallRecord:
    return ProviderCallRecord.from_dict(_redacted_mapping(record.to_dict(), "ProviderCallRecord"))


def _sanitize_tool_install(record: ToolInstallRecord) -> ToolInstallRecord:
    return ToolInstallRecord.from_dict(_redacted_mapping(record.to_dict(), "ToolInstallRecord"))


def _redacted_mapping(payload: Mapping[str, object], path: str) -> dict[str, object]:
    redacted = redact_structure(dict(payload))
    if not isinstance(redacted, Mapping):
        raise ValueError(f"{path}: redacted payload is not an object")

    out: dict[str, object] = {}
    for key, value in redacted.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: non-string key after redaction")
        out[key] = cast("object", value)
    return out


def _assert_no_secret_values(value: object, path: str) -> None:
    if value is None or isinstance(value, (bool, int, float)):
        return

    if isinstance(value, str):
        findings = scan_for_secrets(value)
        if findings:
            rules = ", ".join(sorted({finding.rule for finding in findings}))
            raise ValueError(
                f"{path}: contains secret-like value ({rules}); "
                "use env var references, not raw secrets"
            )
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_part = key if isinstance(key, str) else str(key)
            _assert_no_secret_values(item, f"{path}.{key_part}")
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_secret_values(item, f"{path}[{index}]")
        return


def _row_text(row: Mapping[str, RowValue], key: str, path: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected text value")
    return value


def _as_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected object")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: object keys must be strings")
        parsed[key] = item
    return parsed


def _load_json_object(payload: str, path: str) -> dict[str, object]:
    if not isinstance(payload, str):
        raise ValueError(f"{path}: expected JSON string")
    try:
        loaded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: JSON root must be object")
    out: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: key must be text")
        out[key] = value
    return out


def _as_non_empty_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected string")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{path}: must not be empty")
    return parsed


def _as_non_negative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected integer")
    if value < 0:
        raise ValueError(f"{path}: must be >= 0")
    return value


def _as_non_negative_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: must be finite")
    if parsed < 0:
        raise ValueError(f"{path}: must be >= 0")
    return parsed


def _as_json_object(value: object, path: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected object")
    parsed: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: key must be string")
        parsed[key] = _as_json_value(item, f"{path}.{key}")
    return parsed


def _as_json_value(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: float must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_as_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, tuple):
        return [_as_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object key must be string")
            out[key] = _as_json_value(item, f"{path}.{key}")
        return out
    raise ValueError(f"{path}: value is not JSON-serializable")


def _as_utc_datetime(value: object, path: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid ISO-8601 datetime ({exc})") from exc
    else:
        raise ValueError(f"{path}: expected datetime or ISO-8601 string")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{path}: datetime must be timezone-aware UTC")
    return parsed.astimezone(UTC)


def _iso8601z(value: datetime) -> str:
    return (
        _as_utc_datetime(value, "datetime")
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_run_status(value: RunStatus | str, path: str) -> RunStatus:
    if isinstance(value, RunStatus):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected run status string")
    try:
        return RunStatus(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in RunStatus))
        raise ValueError(f"{path}: invalid run status {value!r}; allowed: {allowed}") from exc


def _as_work_item_status(value: WorkItemStatus | str, path: str) -> WorkItemStatus:
    if isinstance(value, WorkItemStatus):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected work-item status string")
    try:
        return WorkItemStatus(value)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in WorkItemStatus))
        raise ValueError(f"{path}: invalid work-item status {value!r}; allowed: {allowed}") from exc


__all__ = [
    "AttemptRepo",
    "ConstraintRepo",
    "EvidenceRepo",
    "IncidentRepo",
    "MergeRepo",
    "ProviderCallRecord",
    "ProviderCallRepo",
    "RunRepo",
    "TaskGraphRepo",
    "ToolInstallRecord",
    "ToolInstallRepo",
    "ToolRepo",
    "WorkItemRepo",
]
