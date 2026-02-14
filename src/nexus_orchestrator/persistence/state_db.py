"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/persistence/state_db.py
Last updated: 2026-02-11

Purpose
- SQLite schema management, migrations, and connection lifecycle.

What should be included in this file
- Schema version table and migration runner design.
- Safe locking strategy and busy timeout handling.
- Backup/restore helpers (export).

Functional requirements
- Must support idempotent migration application.
- Must record run metadata and incidents even on failure.

Non-functional requirements
- Must avoid long-lived locks that block UI/status commands.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, TypeVar, overload

from nexus_orchestrator.constants import RISK_TIERS, STATE_DB_SCHEMA_VERSION
from nexus_orchestrator.domain.models import (
    AttemptResult,
    ConstraintSeverity,
    ConstraintSource,
    EvidenceResult,
    RunStatus,
    WorkItemStatus,
)

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

SQLValue = str | int | float | bytes | None
SQLParams = Sequence[SQLValue]
RowValue = str | int | float | bytes | None
T = TypeVar("T")
AsyncBlockingPolicy: TypeAlias = Literal["allow", "strict"]

DEFAULT_BUSY_TIMEOUT_MS: Final[int] = 5_000
DEFAULT_BUSY_RETRY_LIMIT: Final[int] = 4
DEFAULT_BUSY_RETRY_BACKOFF_MS: Final[int] = 25
DEFAULT_ASYNC_BLOCKING_POLICY: Final[AsyncBlockingPolicy] = "allow"
ASYNC_BLOCKING_POLICIES: Final[tuple[AsyncBlockingPolicy, ...]] = ("allow", "strict")


class _StateDBClassConnect(Protocol):
    def __call__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        busy_retry_limit: int = DEFAULT_BUSY_RETRY_LIMIT,
        busy_retry_backoff_ms: int = DEFAULT_BUSY_RETRY_BACKOFF_MS,
        async_blocking_policy: AsyncBlockingPolicy = DEFAULT_ASYNC_BLOCKING_POLICY,
    ) -> StateDB: ...


class _StateDBInstanceConnect(Protocol):
    def __call__(self) -> sqlite3.Connection: ...


class _StateDBConnectDescriptor:
    @overload
    def __get__(self, instance: None, owner: type[StateDB]) -> _StateDBClassConnect: ...

    @overload
    def __get__(self, instance: StateDB, owner: type[StateDB]) -> _StateDBInstanceConnect: ...

    def __get__(
        self,
        instance: StateDB | None,
        owner: type[StateDB],
    ) -> _StateDBClassConnect | _StateDBInstanceConnect:
        if instance is None:
            return owner._connect_constructor
        return instance._connect_connection


def _sql_enum(values: Sequence[str]) -> str:
    return ",".join(f"'{value}'" for value in values)


def _enum_values(
    values: type[RunStatus]
    | type[WorkItemStatus]
    | type[ConstraintSeverity]
    | type[ConstraintSource]
    | type[AttemptResult]
    | type[EvidenceResult],
) -> tuple[str, ...]:
    return tuple(sorted(item.value for item in values))


_RUN_STATUS_VALUES: Final[tuple[str, ...]] = _enum_values(RunStatus)
_WORK_ITEM_STATUS_VALUES: Final[tuple[str, ...]] = _enum_values(WorkItemStatus)
_CONSTRAINT_SEVERITY_VALUES: Final[tuple[str, ...]] = _enum_values(ConstraintSeverity)
_CONSTRAINT_SOURCE_VALUES: Final[tuple[str, ...]] = _enum_values(ConstraintSource)
_ATTEMPT_RESULT_VALUES: Final[tuple[str, ...]] = _enum_values(AttemptResult)
_EVIDENCE_RESULT_VALUES: Final[tuple[str, ...]] = _enum_values(EvidenceResult)

_SCHEMA_VERSIONS_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_versions (
    version INTEGER PRIMARY KEY CHECK (version > 0),
    name TEXT NOT NULL,
    checksum TEXT NOT NULL CHECK (length(checksum) = 64),
    applied_at TEXT NOT NULL
)
"""

_MIGRATION_0001_STATEMENTS: Final[tuple[str, ...]] = (
    _SCHEMA_VERSIONS_TABLE_SQL,
    f"""
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        spec_path TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({_sql_enum(_RUN_STATUS_VALUES)})),
        started_at TEXT NOT NULL,
        finished_at TEXT,
        config_hash TEXT NOT NULL CHECK (length(config_hash) = 64),
        risk_tier TEXT NOT NULL CHECK (risk_tier IN ({_sql_enum(RISK_TIERS)})),
        budget_json TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        work_item_ids_json TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS work_items (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({_sql_enum(_WORK_ITEM_STATUS_VALUES)})),
        risk_tier TEXT NOT NULL CHECK (risk_tier IN ({_sql_enum(RISK_TIERS)})),
        constraint_envelope_hash TEXT NOT NULL CHECK (length(constraint_envelope_hash) = 64),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_graph_edges (
        run_id TEXT NOT NULL,
        parent_id TEXT NOT NULL,
        child_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, parent_id, child_id),
        CHECK (parent_id <> child_id),
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
        FOREIGN KEY(parent_id) REFERENCES work_items(id) ON DELETE CASCADE,
        FOREIGN KEY(child_id) REFERENCES work_items(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_graphs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS constraints (
        id TEXT PRIMARY KEY,
        severity TEXT NOT NULL CHECK (severity IN ({_sql_enum(_CONSTRAINT_SEVERITY_VALUES)})),
        category TEXT NOT NULL,
        checker_binding TEXT NOT NULL,
        active INTEGER NOT NULL CHECK (active IN (0, 1)),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS attempts (
        id TEXT PRIMARY KEY,
        work_item_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        iteration INTEGER NOT NULL CHECK (iteration >= 1),
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        result TEXT NOT NULL CHECK (result IN ({_sql_enum(_ATTEMPT_RESULT_VALUES)})),
        cost_usd REAL NOT NULL CHECK (cost_usd >= 0),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        UNIQUE(work_item_id, iteration),
        FOREIGN KEY(work_item_id) REFERENCES work_items(id) ON DELETE CASCADE,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS evidence (
        id TEXT PRIMARY KEY,
        work_item_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        stage TEXT NOT NULL,
        result TEXT NOT NULL CHECK (result IN ({_sql_enum(_EVIDENCE_RESULT_VALUES)})),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(work_item_id) REFERENCES work_items(id) ON DELETE CASCADE,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS merges (
        id TEXT PRIMARY KEY,
        work_item_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        commit_sha TEXT NOT NULL,
        merged_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(work_item_id) REFERENCES work_items(id) ON DELETE CASCADE,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_calls (
        id TEXT PRIMARY KEY,
        attempt_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        tokens INTEGER NOT NULL CHECK (tokens >= 0),
        cost_usd REAL NOT NULL CHECK (cost_usd >= 0),
        latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(attempt_id) REFERENCES attempts(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_installs (
        id TEXT PRIMARY KEY,
        tool TEXT NOT NULL,
        version TEXT NOT NULL,
        checksum TEXT NOT NULL,
        approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
        installed_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS incidents (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        related_work_item_id TEXT,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE,
        FOREIGN KEY(related_work_item_id) REFERENCES work_items(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS evidence_append_only_update
    BEFORE UPDATE ON evidence
    BEGIN
        SELECT RAISE(ABORT, 'evidence is append-only');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS evidence_append_only_delete
    BEFORE DELETE ON evidence
    BEGIN
        SELECT RAISE(ABORT, 'evidence is append-only');
    END
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_status_started ON runs(status, started_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_updated ON runs(updated_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_work_items_run_status ON work_items(run_id, status)",
    """
    CREATE INDEX IF NOT EXISTS idx_work_items_run_risk_status
    ON work_items(run_id, risk_tier, status)
    """,
    "CREATE INDEX IF NOT EXISTS idx_task_graph_edges_run_child ON task_graph_edges(run_id, child_id)",
    "CREATE INDEX IF NOT EXISTS idx_task_graph_edges_run_parent ON task_graph_edges(run_id, parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_constraints_active_category ON constraints(active, category)",
    "CREATE INDEX IF NOT EXISTS idx_attempts_work_item_iteration ON attempts(work_item_id, iteration DESC)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_work_item_created ON evidence(work_item_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_merges_run_merged_at ON merges(run_id, merged_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_provider_calls_attempt_created ON provider_calls(attempt_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_tool_installs_tool_version ON tool_installs(tool, version)",
    "CREATE INDEX IF NOT EXISTS idx_incidents_run_created ON incidents(run_id, created_at DESC)",
)

_MIGRATION_0002_STATEMENTS: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS merge_queue_states (
        integration_branch TEXT PRIMARY KEY,
        schema_version INTEGER NOT NULL CHECK (schema_version > 0),
        next_arrival INTEGER NOT NULL CHECK (next_arrival >= 0),
        completed_work_items_json TEXT NOT NULL,
        failed_work_items_json TEXT NOT NULL,
        queue_json TEXT NOT NULL,
        imported_from_legacy INTEGER NOT NULL CHECK (imported_from_legacy IN (0, 1)) DEFAULT 0,
        legacy_state_path TEXT,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_merge_queue_states_updated
    ON merge_queue_states(updated_at DESC)
    """,
)


@dataclass(frozen=True, slots=True)
class MigrationRecord:
    version: int
    name: str
    checksum: str
    applied_at: str


@dataclass(frozen=True, slots=True)
class _Migration:
    version: int
    name: str
    statements: tuple[str, ...]
    checksum: str


def _migration_checksum(version: int, name: str, statements: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"{version}:{name}\n".encode())
    for statement in statements:
        normalized = "\n".join(line.rstrip() for line in statement.strip().splitlines())
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\n--\n")
    return digest.hexdigest()


_MIGRATIONS: Final[tuple[_Migration, ...]] = (
    _Migration(
        version=1,
        name="initial_state_schema",
        statements=_MIGRATION_0001_STATEMENTS,
        checksum=_migration_checksum(1, "initial_state_schema", _MIGRATION_0001_STATEMENTS),
    ),
    _Migration(
        version=2,
        name="merge_queue_state_store",
        statements=_MIGRATION_0002_STATEMENTS,
        checksum=_migration_checksum(2, "merge_queue_state_store", _MIGRATION_0002_STATEMENTS),
    ),
)

_SQLITE_BUSY_CODES: Final[frozenset[int]] = frozenset(
    code
    for code in (
        getattr(sqlite3, "SQLITE_BUSY", None),
        getattr(sqlite3, "SQLITE_BUSY_RECOVERY", None),
        getattr(sqlite3, "SQLITE_BUSY_SNAPSHOT", None),
        getattr(sqlite3, "SQLITE_LOCKED", None),
        getattr(sqlite3, "SQLITE_LOCKED_SHAREDCACHE", None),
    )
    if isinstance(code, int)
)

_SQLITE_CORRUPTION_CODES: Final[frozenset[int]] = frozenset(
    code
    for code in (
        getattr(sqlite3, "SQLITE_CORRUPT", None),
        getattr(sqlite3, "SQLITE_NOTADB", None),
    )
    if isinstance(code, int)
)

_BUSY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "database is locked",
    "database table is locked",
    "database schema is locked",
)

_CORRUPTION_SUBSTRINGS: Final[tuple[str, ...]] = (
    "database disk image is malformed",
    "malformed database",
    "file is not a database",
)


class StateDBError(RuntimeError):
    """Base class for persistence DB errors."""


class StateDBBusyError(StateDBError):
    """Raised when bounded busy retries are exhausted."""


class StateDBMigrationError(StateDBError):
    """Raised when migrations cannot be applied safely."""


class StateDBCorruptionError(StateDBError):
    """Raised when SQLite reports possible corruption."""


class StateDBAsyncPolicyError(StateDBError):
    """Raised when sync DB I/O is attempted from an active async event loop."""


class StateDB:
    """SQLite state DB manager with deterministic migrations and safe helpers."""

    connect: _StateDBConnectDescriptor = _StateDBConnectDescriptor()

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        busy_retry_limit: int = DEFAULT_BUSY_RETRY_LIMIT,
        busy_retry_backoff_ms: int = DEFAULT_BUSY_RETRY_BACKOFF_MS,
        async_blocking_policy: AsyncBlockingPolicy = DEFAULT_ASYNC_BLOCKING_POLICY,
    ) -> None:
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be >= 0")
        if busy_retry_limit < 0:
            raise ValueError("busy_retry_limit must be >= 0")
        if busy_retry_backoff_ms < 0:
            raise ValueError("busy_retry_backoff_ms must be >= 0")
        if async_blocking_policy not in ASYNC_BLOCKING_POLICIES:
            allowed = ", ".join(ASYNC_BLOCKING_POLICIES)
            raise ValueError(
                f"async_blocking_policy must be one of: {allowed}; got {async_blocking_policy!r}"
            )

        self._path = Path(path).expanduser()
        self._busy_timeout_ms = busy_timeout_ms
        self._busy_retry_limit = busy_retry_limit
        self._busy_retry_backoff_ms = busy_retry_backoff_ms
        self._async_blocking_policy = async_blocking_policy
        self._savepoint_counter = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def async_blocking_policy(self) -> str:
        return self._async_blocking_policy

    def _is_async_event_loop_thread(self) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _assert_sync_io_allowed(self, *, operation: str) -> None:
        if self._async_blocking_policy != "strict":
            return
        if not self._is_async_event_loop_thread():
            return
        raise StateDBAsyncPolicyError(
            f"{operation} is disallowed from an active event loop thread for {self._path}; "
            "use the async StateDB APIs (e.g. execute_async/query_*_async/migrate_async) "
            "or offload sync calls via asyncio.to_thread(...)"
        )

    @classmethod
    def _connect_constructor(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        busy_retry_limit: int = DEFAULT_BUSY_RETRY_LIMIT,
        busy_retry_backoff_ms: int = DEFAULT_BUSY_RETRY_BACKOFF_MS,
        async_blocking_policy: AsyncBlockingPolicy = DEFAULT_ASYNC_BLOCKING_POLICY,
    ) -> StateDB:
        if isinstance(path, StateDB):
            raise TypeError(
                "StateDB.connect(db) is invalid; call db.connect() to open a sqlite3.Connection."
            )
        if not isinstance(path, (str, Path)):
            raise TypeError(
                "StateDB.connect(path, ...) expects `path` to be str or pathlib.Path; "
                f"got {type(path).__name__}."
            )
        return cls(
            path,
            busy_timeout_ms=busy_timeout_ms,
            busy_retry_limit=busy_retry_limit,
            busy_retry_backoff_ms=busy_retry_backoff_ms,
            async_blocking_policy=async_blocking_policy,
        )

    @classmethod
    async def aconnect(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        busy_retry_limit: int = DEFAULT_BUSY_RETRY_LIMIT,
        busy_retry_backoff_ms: int = DEFAULT_BUSY_RETRY_BACKOFF_MS,
        async_blocking_policy: AsyncBlockingPolicy = DEFAULT_ASYNC_BLOCKING_POLICY,
    ) -> StateDB:
        return cls._connect_constructor(
            path,
            busy_timeout_ms=busy_timeout_ms,
            busy_retry_limit=busy_retry_limit,
            busy_retry_backoff_ms=busy_retry_backoff_ms,
            async_blocking_policy=async_blocking_policy,
        )

    @classmethod
    async def connect_async(
        cls,
        path: str | Path,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
        busy_retry_limit: int = DEFAULT_BUSY_RETRY_LIMIT,
        busy_retry_backoff_ms: int = DEFAULT_BUSY_RETRY_BACKOFF_MS,
        async_blocking_policy: AsyncBlockingPolicy = DEFAULT_ASYNC_BLOCKING_POLICY,
    ) -> StateDB:
        return await cls.aconnect(
            path,
            busy_timeout_ms=busy_timeout_ms,
            busy_retry_limit=busy_retry_limit,
            busy_retry_backoff_ms=busy_retry_backoff_ms,
            async_blocking_policy=async_blocking_policy,
        )

    def _connect_connection(self, *args: object, **kwargs: object) -> sqlite3.Connection:
        """Open a configured SQLite connection for the state DB."""

        if args or kwargs:
            raise TypeError(
                "StateDB.connect() on an instance accepts no arguments; "
                "use StateDB.connect(path, ...) to construct a StateDB."
            )
        self._assert_sync_io_allowed(operation="connect")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        self._configure_connection(conn)
        return conn

    def __await__(self) -> Iterator[object]:
        async def _raise_await_misuse() -> None:
            raise TypeError(
                "StateDB is not awaitable. Use StateDB.connect(path, ...) for sync construction "
                "or await StateDB.aconnect(path, ...) for async construction."
            )

        return _raise_await_misuse().__await__()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self._assert_sync_io_allowed(operation="connection")
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        immediate: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        """Run statements inside an atomic transaction with savepoint support."""

        self._assert_sync_io_allowed(operation="transaction")
        if conn is None:
            with self.connection() as owned_conn:
                with self.transaction(conn=owned_conn, immediate=immediate) as txn_conn:
                    yield txn_conn
                return

        if conn.in_transaction:
            savepoint = self._next_savepoint_name()
            self._execute_with_retry(conn, f"SAVEPOINT {savepoint}", (), operation="savepoint")
            try:
                yield conn
            except Exception:
                self._execute_with_retry(
                    conn,
                    f"ROLLBACK TO SAVEPOINT {savepoint}",
                    (),
                    operation="rollback to savepoint",
                )
                self._execute_with_retry(
                    conn,
                    f"RELEASE SAVEPOINT {savepoint}",
                    (),
                    operation="release savepoint",
                )
                raise
            else:
                self._execute_with_retry(
                    conn,
                    f"RELEASE SAVEPOINT {savepoint}",
                    (),
                    operation="release savepoint",
                )
            return

        begin_sql = "BEGIN IMMEDIATE" if immediate else "BEGIN"
        self._execute_with_retry(conn, begin_sql, (), operation="begin transaction")
        try:
            yield conn
        except Exception:
            self._execute_with_retry(conn, "ROLLBACK", (), operation="rollback transaction")
            raise
        else:
            self._execute_with_retry(conn, "COMMIT", (), operation="commit transaction")

    def migrate(self) -> int:
        """Apply migrations idempotently and return current schema version."""

        self._assert_sync_io_allowed(operation="migrate")
        self._validate_migration_chain(STATE_DB_SCHEMA_VERSION)
        with self.connection() as conn:
            self._execute_with_retry(
                conn,
                _SCHEMA_VERSIONS_TABLE_SQL,
                (),
                operation="create schema_versions table",
            )
            applied = self._load_applied_migrations(conn)
            current_version = max(applied, default=0)
            if current_version > STATE_DB_SCHEMA_VERSION:
                raise StateDBMigrationError(
                    "database schema is newer than supported by this binary "
                    f"(db={current_version}, code={STATE_DB_SCHEMA_VERSION})"
                )

            for migration in _MIGRATIONS:
                if migration.version > STATE_DB_SCHEMA_VERSION:
                    continue

                record = applied.get(migration.version)
                if record is not None:
                    if record.checksum != migration.checksum:
                        raise StateDBMigrationError(
                            "migration checksum mismatch for version "
                            f"{migration.version}: db={record.checksum} code={migration.checksum}"
                        )
                    continue

                with self.transaction(conn=conn, immediate=True) as tx:
                    for statement in migration.statements:
                        self._execute_with_retry(
                            tx,
                            statement,
                            (),
                            operation=f"apply migration {migration.version}",
                        )
                    self._execute_with_retry(
                        tx,
                        """
                        INSERT INTO schema_versions (version, name, checksum, applied_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            migration.version,
                            migration.name,
                            migration.checksum,
                            _utc_now_iso(),
                        ),
                        operation=f"record migration {migration.version}",
                    )

                applied[migration.version] = MigrationRecord(
                    version=migration.version,
                    name=migration.name,
                    checksum=migration.checksum,
                    applied_at=_utc_now_iso(),
                )

            return self.schema_version(conn=conn)

    def ensure_schema(self) -> int:
        """Compatibility alias for ``migrate``."""

        return self.migrate()

    def apply_migrations(self) -> int:
        """Compatibility alias for ``migrate``."""

        return self.migrate()

    async def migrate_async(self) -> int:
        return await asyncio.to_thread(self.migrate)

    def schema_version(self, *, conn: sqlite3.Connection | None = None) -> int:
        if conn is None:
            self._assert_sync_io_allowed(operation="schema_version")
        row = self.query_one(
            "SELECT COALESCE(MAX(version), 0) AS version FROM schema_versions",
            conn=conn,
        )
        if row is None:
            return 0
        value = row["version"]
        if not isinstance(value, int):
            raise StateDBMigrationError("schema_versions.version must be an integer")
        return value

    def schema_history(self) -> list[MigrationRecord]:
        self._assert_sync_io_allowed(operation="schema_history")
        rows = self.query_all(
            """
            SELECT version, name, checksum, applied_at
            FROM schema_versions
            ORDER BY version ASC
            """
        )
        records: list[MigrationRecord] = []
        for row in rows:
            version = row.get("version")
            name = row.get("name")
            checksum = row.get("checksum")
            applied_at = row.get("applied_at")
            if not isinstance(version, int):
                raise StateDBMigrationError("schema_versions.version must be integer")
            if not isinstance(name, str):
                raise StateDBMigrationError("schema_versions.name must be text")
            if not isinstance(checksum, str):
                raise StateDBMigrationError("schema_versions.checksum must be text")
            if not isinstance(applied_at, str):
                raise StateDBMigrationError("schema_versions.applied_at must be text")
            records.append(
                MigrationRecord(
                    version=version,
                    name=name,
                    checksum=checksum,
                    applied_at=applied_at,
                )
            )
        return records

    def execute(
        self,
        sql: str,
        params: SQLParams = (),
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Execute a parameterized statement and return affected row count."""

        self._assert_sync_io_allowed(operation="execute")
        if conn is not None:
            cursor = self._execute_with_retry(conn, sql, params, operation="execute statement")
            return cursor.rowcount

        with self.transaction(immediate=True) as tx:
            cursor = self._execute_with_retry(tx, sql, params, operation="execute statement")
            return cursor.rowcount

    def executemany(
        self,
        sql: str,
        params_iter: Iterable[SQLParams],
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Execute a parameterized statement for a sequence of parameter tuples."""

        self._assert_sync_io_allowed(operation="executemany")
        params_list = [tuple(params) for params in params_iter]
        if conn is not None:
            return self._executemany_with_retry(conn, sql, params_list, operation="execute many")

        with self.transaction(immediate=True) as tx:
            return self._executemany_with_retry(tx, sql, params_list, operation="execute many")

    def query_all(
        self,
        sql: str,
        params: SQLParams = (),
        *,
        conn: sqlite3.Connection | None = None,
    ) -> list[dict[str, RowValue]]:
        """Run a query and return rows as typed dictionaries."""

        self._assert_sync_io_allowed(operation="query_all")
        if conn is not None:
            cursor = self._execute_with_retry(conn, sql, params, operation="query all")
            return [_row_to_dict(row) for row in cursor.fetchall()]

        with self.connection() as owned_conn:
            cursor = self._execute_with_retry(owned_conn, sql, params, operation="query all")
            return [_row_to_dict(row) for row in cursor.fetchall()]

    def query_one(
        self,
        sql: str,
        params: SQLParams = (),
        *,
        conn: sqlite3.Connection | None = None,
    ) -> dict[str, RowValue] | None:
        """Run a query and return the first row as a typed dictionary."""

        self._assert_sync_io_allowed(operation="query_one")
        if conn is not None:
            cursor = self._execute_with_retry(conn, sql, params, operation="query one")
            row = cursor.fetchone()
            return None if row is None else _row_to_dict(row)

        with self.connection() as owned_conn:
            cursor = self._execute_with_retry(owned_conn, sql, params, operation="query one")
            row = cursor.fetchone()
            return None if row is None else _row_to_dict(row)

    async def execute_async(self, sql: str, params: SQLParams = ()) -> int:
        return await asyncio.to_thread(self.execute, sql, tuple(params))

    async def executemany_async(self, sql: str, params_iter: Iterable[SQLParams]) -> int:
        params_list = [tuple(params) for params in params_iter]
        return await asyncio.to_thread(self.executemany, sql, params_list)

    async def query_all_async(
        self,
        sql: str,
        params: SQLParams = (),
    ) -> list[dict[str, RowValue]]:
        return await asyncio.to_thread(self.query_all, sql, tuple(params))

    async def query_one_async(
        self,
        sql: str,
        params: SQLParams = (),
    ) -> dict[str, RowValue] | None:
        return await asyncio.to_thread(self.query_one, sql, tuple(params))

    def record_run_metadata(
        self,
        *,
        run_id: str,
        spec_path: str,
        status: str,
        started_at: str,
        config_hash: str,
        risk_tier: str = "medium",
        budget: object | None = None,
        metadata: object | None = None,
        work_item_ids: Sequence[str] = (),
        finished_at: str | None = None,
    ) -> None:
        """Record run metadata via idempotent upsert for failure-safe bookkeeping."""

        self._assert_sync_io_allowed(operation="record_run_metadata")
        if status not in _RUN_STATUS_VALUES:
            raise ValueError(f"unsupported run status: {status}")
        if risk_tier not in RISK_TIERS:
            raise ValueError(f"unsupported risk tier: {risk_tier}")
        if len(config_hash) != 64:
            raise ValueError("config_hash must be a 64-character hex digest")

        budget_json = canonical_json({} if budget is None else budget)
        metadata_json = canonical_json({} if metadata is None else metadata)
        work_item_ids_json = canonical_json(list(work_item_ids))
        payload_json = canonical_json(
            {
                "id": run_id,
                "spec_path": spec_path,
                "status": status,
                "started_at": started_at,
                "finished_at": finished_at,
                "risk_tier": risk_tier,
                "budget": {} if budget is None else budget,
                "metadata": {} if metadata is None else metadata,
                "work_item_ids": list(work_item_ids),
            }
        )

        self.execute(
            """
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
            """,
            (
                run_id,
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
                started_at,
                _utc_now_iso(),
            ),
        )

    def record_incident(
        self,
        *,
        incident_id: str,
        run_id: str,
        category: str,
        message: str,
        related_work_item_id: str | None = None,
        created_at: str | None = None,
        payload: object | None = None,
    ) -> None:
        """Record incident metadata via idempotent upsert for failure-safe bookkeeping."""

        self._assert_sync_io_allowed(operation="record_incident")
        incident_payload = (
            {"id": incident_id, "run_id": run_id, "category": category, "message": message}
            if payload is None
            else payload
        )
        created = _utc_now_iso() if created_at is None else created_at
        self.execute(
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
            ON CONFLICT(id) DO UPDATE SET
                run_id=excluded.run_id,
                related_work_item_id=excluded.related_work_item_id,
                category=excluded.category,
                message=excluded.message,
                created_at=excluded.created_at,
                payload_json=excluded.payload_json
            """,
            (
                incident_id,
                run_id,
                related_work_item_id,
                category,
                message,
                created,
                canonical_json(incident_payload),
            ),
        )

    async def record_run_metadata_async(
        self,
        *,
        run_id: str,
        spec_path: str,
        status: str,
        started_at: str,
        config_hash: str,
        risk_tier: str = "medium",
        budget: object | None = None,
        metadata: object | None = None,
        work_item_ids: Sequence[str] = (),
        finished_at: str | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.record_run_metadata,
            run_id=run_id,
            spec_path=spec_path,
            status=status,
            started_at=started_at,
            config_hash=config_hash,
            risk_tier=risk_tier,
            budget=budget,
            metadata=metadata,
            work_item_ids=work_item_ids,
            finished_at=finished_at,
        )

    async def record_incident_async(
        self,
        *,
        incident_id: str,
        run_id: str,
        category: str,
        message: str,
        related_work_item_id: str | None = None,
        created_at: str | None = None,
        payload: object | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.record_incident,
            incident_id=incident_id,
            run_id=run_id,
            category=category,
            message=message,
            related_work_item_id=related_work_item_id,
            created_at=created_at,
            payload=payload,
        )

    def backup(self, destination: str | Path) -> Path:
        """Create a consistent snapshot using SQLite backup API."""

        self._assert_sync_io_allowed(operation="backup")
        destination_path = Path(destination).expanduser()
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            self.connection() as source,
            sqlite3.connect(
                destination_path,
                timeout=self._busy_timeout_ms / 1000.0,
                isolation_level=None,
                check_same_thread=False,
            ) as target,
        ):
            source.backup(target)
            target.execute("PRAGMA foreign_keys=ON")
            target.execute("PRAGMA journal_mode=WAL")
            target.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")

        return destination_path

    def integrity_check(self, *, max_errors: int = 100) -> tuple[str, ...]:
        """Return integrity-check errors; empty tuple means OK."""

        self._assert_sync_io_allowed(operation="integrity_check")
        if max_errors <= 0:
            raise ValueError("max_errors must be > 0")
        rows = self.query_all(f"PRAGMA integrity_check({max_errors})")
        messages = tuple(str(row.get("integrity_check", "")) for row in rows)
        if messages == ("ok",):
            return ()
        return messages

    def close(self) -> None:
        """Compatibility no-op (connections are short-lived)."""

    def __enter__(self) -> StateDB:
        self.migrate()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        journal_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if journal_row is None:
            raise StateDBError("failed to configure journal_mode")
        journal_mode = str(journal_row[0]).lower()
        if journal_mode != "wal":
            raise StateDBError(f"journal_mode must be WAL, got {journal_mode!r}")

    def _load_applied_migrations(self, conn: sqlite3.Connection) -> dict[int, MigrationRecord]:
        cursor = self._execute_with_retry(
            conn,
            """
            SELECT version, name, checksum, applied_at
            FROM schema_versions
            ORDER BY version ASC
            """,
            (),
            operation="load schema_versions",
        )
        out: dict[int, MigrationRecord] = {}
        for row in cursor.fetchall():
            if not isinstance(row["version"], int):
                raise StateDBMigrationError("schema_versions.version must be integer")
            if not isinstance(row["name"], str):
                raise StateDBMigrationError("schema_versions.name must be text")
            if not isinstance(row["checksum"], str):
                raise StateDBMigrationError("schema_versions.checksum must be text")
            if not isinstance(row["applied_at"], str):
                raise StateDBMigrationError("schema_versions.applied_at must be text")
            out[row["version"]] = MigrationRecord(
                version=row["version"],
                name=row["name"],
                checksum=row["checksum"],
                applied_at=row["applied_at"],
            )
        return out

    def _validate_migration_chain(self, target_version: int) -> None:
        if target_version < 0:
            raise StateDBMigrationError("target schema version must be >= 0")
        migration_versions = {migration.version for migration in _MIGRATIONS}
        if target_version > max(migration_versions, default=0):
            raise StateDBMigrationError(
                "schema target exceeds known migrations "
                f"(target={target_version}, known={max(migration_versions, default=0)})"
            )
        for version in range(1, target_version + 1):
            if version not in migration_versions:
                raise StateDBMigrationError(f"missing migration for schema version {version}")

    def _next_savepoint_name(self) -> str:
        self._savepoint_counter += 1
        return f"sp_{self._savepoint_counter}"

    def _execute_with_retry(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: SQLParams,
        *,
        operation: str,
    ) -> sqlite3.Cursor:
        for attempt in range(self._busy_retry_limit + 1):
            try:
                return conn.execute(sql, tuple(params))
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as exc:
                if self._is_busy_error(exc) and attempt < self._busy_retry_limit:
                    time.sleep((self._busy_retry_backoff_ms / 1000.0) * float(2**attempt))
                    continue
                self._raise_actionable_error(exc, operation=operation)
        raise StateDBBusyError(f"{operation} exhausted retries unexpectedly")

    def _executemany_with_retry(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params_list: Sequence[SQLParams],
        *,
        operation: str,
    ) -> int:
        for attempt in range(self._busy_retry_limit + 1):
            try:
                cursor = conn.executemany(sql, params_list)
                return cursor.rowcount
            except sqlite3.IntegrityError:
                raise
            except sqlite3.Error as exc:
                if self._is_busy_error(exc) and attempt < self._busy_retry_limit:
                    time.sleep((self._busy_retry_backoff_ms / 1000.0) * float(2**attempt))
                    continue
                self._raise_actionable_error(exc, operation=operation)
        raise StateDBBusyError(f"{operation} exhausted retries unexpectedly")

    def _is_busy_error(self, exc: sqlite3.Error) -> bool:
        code = getattr(exc, "sqlite_errorcode", None)
        if isinstance(code, int) and code in _SQLITE_BUSY_CODES:
            return True
        message = str(exc).lower()
        return any(fragment in message for fragment in _BUSY_SUBSTRINGS)

    def _is_corruption_error(self, exc: sqlite3.Error) -> bool:
        code = getattr(exc, "sqlite_errorcode", None)
        if isinstance(code, int) and code in _SQLITE_CORRUPTION_CODES:
            return True
        message = str(exc).lower()
        return any(fragment in message for fragment in _CORRUPTION_SUBSTRINGS)

    def _raise_actionable_error(self, exc: sqlite3.Error, *, operation: str) -> None:
        if self._is_corruption_error(exc):
            raise StateDBCorruptionError(
                f"{operation} failed for {self._path}: {exc}. "
                "Run `StateDB.integrity_check()` and restore from `StateDB.backup(...)` if needed."
            ) from exc
        if self._is_busy_error(exc):
            raise StateDBBusyError(
                f"{operation} hit SQLITE_BUSY for {self._path} after "
                f"{self._busy_retry_limit + 1} attempt(s): {exc}"
            ) from exc
        raise StateDBError(f"{operation} failed for {self._path}: {exc}") from exc


def _row_to_dict(row: sqlite3.Row) -> dict[str, RowValue]:
    raw = dict(row)
    return {str(key): raw[key] for key in raw}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(value: object) -> str:
    """Deterministic JSON for persistence payloads."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


__all__ = [
    "ASYNC_BLOCKING_POLICIES",
    "DEFAULT_ASYNC_BLOCKING_POLICY",
    "DEFAULT_BUSY_RETRY_BACKOFF_MS",
    "DEFAULT_BUSY_RETRY_LIMIT",
    "DEFAULT_BUSY_TIMEOUT_MS",
    "MigrationRecord",
    "RowValue",
    "SQLParams",
    "SQLValue",
    "StateDB",
    "StateDBBusyError",
    "StateDBCorruptionError",
    "StateDBError",
    "StateDBAsyncPolicyError",
    "StateDBMigrationError",
    "canonical_json",
]
