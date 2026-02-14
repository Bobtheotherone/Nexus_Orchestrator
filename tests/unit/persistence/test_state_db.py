"""State DB migration, pragmas, backup, and concurrency tests."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.constants import STATE_DB_SCHEMA_VERSION
from nexus_orchestrator.persistence.repositories import RunRepo
from nexus_orchestrator.persistence.state_db import (
    StateDB,
    StateDBAsyncPolicyError,
)

from . import make_run

if TYPE_CHECKING:
    from pathlib import Path


def test_migration_idempotence_schema_version_and_pragmas(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3", busy_timeout_ms=4_321)

    version_first = db.migrate()
    version_second = db.migrate()

    assert version_first == STATE_DB_SCHEMA_VERSION
    assert version_second == STATE_DB_SCHEMA_VERSION

    with db.connection() as conn:
        tables = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                ORDER BY name
                """
            ).fetchall()
        }
        expected_tables = {
            "schema_versions",
            "runs",
            "work_items",
            "task_graph_edges",
            "task_graphs",
            "constraints",
            "attempts",
            "evidence",
            "merges",
            "provider_calls",
            "tool_installs",
            "incidents",
            "merge_queue_states",
        }
        assert expected_tables.issubset(tables)

        index_names = {
            str(row[0])
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index'
                ORDER BY name
                """
            ).fetchall()
        }
        assert "idx_work_items_run_status" in index_names
        assert "idx_task_graph_edges_run_child" in index_names
        assert "idx_runs_status_started" in index_names

        pragma_fk = conn.execute("PRAGMA foreign_keys").fetchone()
        pragma_journal = conn.execute("PRAGMA journal_mode").fetchone()
        pragma_busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()

        assert pragma_fk is not None
        assert int(pragma_fk[0]) == 1
        assert pragma_journal is not None
        assert str(pragma_journal[0]).lower() == "wal"
        assert pragma_busy_timeout is not None
        assert int(pragma_busy_timeout[0]) == 4_321

        migration_rows = conn.execute(
            "SELECT version, name, checksum, applied_at FROM schema_versions ORDER BY version"
        ).fetchall()
        assert len(migration_rows) == STATE_DB_SCHEMA_VERSION


def test_backup_and_integrity_check(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "nexus.sqlite3"
    db = StateDB(db_path)
    db.migrate()

    run_repo = RunRepo(db)
    run = make_run(1_000)
    run_repo.add(run)

    assert db.integrity_check() == ()

    backup_path = db.backup(tmp_path / "state" / "backup.sqlite3")
    assert backup_path.exists()
    assert backup_path.stat().st_size > 0

    with sqlite3.connect(backup_path) as conn:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        assert row is not None
        assert str(row[0]).lower() == "ok"


def test_wal_allows_reader_during_open_writer_transaction(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    run = make_run(2_000)
    run_repo.add(run)

    writer_conn = db.connect()
    reader_conn = db.connect()

    writer_started = threading.Event()
    reader_finished = threading.Event()

    errors: list[str] = []
    reader_elapsed: float | None = None
    reader_count: int | None = None

    def writer() -> None:
        nonlocal errors
        try:
            writer_conn.execute("BEGIN IMMEDIATE")
            writer_conn.execute(
                "UPDATE runs SET status = ? WHERE id = ?",
                ("running", run.id),
            )
            writer_started.set()
            if not reader_finished.wait(timeout=2.0):
                errors.append("reader did not finish while writer transaction was open")
            writer_conn.execute("ROLLBACK")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"writer failed: {exc}")

    def reader() -> None:
        nonlocal errors, reader_elapsed, reader_count
        if not writer_started.wait(timeout=2.0):
            errors.append("writer did not start")
            reader_finished.set()
            return

        try:
            start = time.monotonic()
            row = reader_conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            end = time.monotonic()
            if row is None:
                errors.append("reader returned no row")
                return
            reader_count = int(row[0])
            reader_elapsed = end - start
        except Exception as exc:  # noqa: BLE001
            errors.append(f"reader failed: {exc}")
        finally:
            reader_finished.set()

    writer_thread = threading.Thread(target=writer, name="state-db-writer", daemon=True)
    reader_thread = threading.Thread(target=reader, name="state-db-reader", daemon=True)

    writer_thread.start()
    reader_thread.start()

    writer_thread.join(timeout=5.0)
    reader_thread.join(timeout=5.0)

    writer_conn.close()
    reader_conn.close()

    assert not writer_thread.is_alive()
    assert not reader_thread.is_alive()
    assert not errors
    assert reader_count == 1
    assert reader_elapsed is not None
    assert reader_elapsed < 0.75


@pytest.mark.asyncio
async def test_strict_async_policy_blocks_sync_db_io_on_event_loop(tmp_path: Path) -> None:
    db = StateDB(
        tmp_path / "state" / "nexus.sqlite3",
        async_blocking_policy="strict",
    )

    with pytest.raises(StateDBAsyncPolicyError):
        db.migrate()

    version = await db.migrate_async()
    assert version == STATE_DB_SCHEMA_VERSION

    with pytest.raises(StateDBAsyncPolicyError):
        db.query_one("SELECT 1 AS one")


@pytest.mark.asyncio
async def test_async_db_stress_does_not_stall_event_loop(tmp_path: Path) -> None:
    db = StateDB(
        tmp_path / "state" / "nexus.sqlite3",
        async_blocking_policy="strict",
    )
    await db.migrate_async()
    await db.execute_async(
        """
        CREATE TABLE IF NOT EXISTS async_stress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            payload TEXT NOT NULL
        )
        """
    )

    worker_count = 8
    iterations_per_worker = 80
    monitor_interval_seconds = 0.005
    max_allowed_stall_seconds = 0.12

    max_observed_stall = 0.0
    monitor_stop = asyncio.Event()

    async def monitor_loop_stall() -> None:
        nonlocal max_observed_stall
        while not monitor_stop.is_set():
            start = time.monotonic()
            await asyncio.sleep(monitor_interval_seconds)
            drift = max(0.0, time.monotonic() - start - monitor_interval_seconds)
            max_observed_stall = max(max_observed_stall, drift)

    async def db_worker(worker_id: int) -> None:
        for iteration in range(iterations_per_worker):
            await db.execute_async(
                "INSERT INTO async_stress (payload) VALUES (?)",
                (f"{worker_id}:{iteration}",),
            )
            if iteration % 10 == 0:
                row = await db.query_one_async("SELECT COUNT(*) AS c FROM async_stress")
                assert row is not None
                assert "c" in row
            await asyncio.sleep(0)

    monitor_task = asyncio.create_task(monitor_loop_stall())
    await asyncio.gather(*(db_worker(index) for index in range(worker_count)))
    monitor_stop.set()
    await monitor_task

    row = await db.query_one_async("SELECT COUNT(*) AS c FROM async_stress")
    assert row is not None
    assert int(row["c"]) == worker_count * iterations_per_worker
    assert max_observed_stall < max_allowed_stall_seconds
