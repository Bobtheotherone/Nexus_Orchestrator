"""Regression tests for class-level StateDB.connect behavior."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from nexus_orchestrator.constants import STATE_DB_SCHEMA_VERSION
from nexus_orchestrator.persistence.state_db import StateDB

if TYPE_CHECKING:
    from collections.abc import Awaitable


def test_connect_class_call_returns_state_db_and_migrate_works(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "class-connect.sqlite3"

    db = StateDB.connect(db_path)

    assert isinstance(db, StateDB)
    assert db.path == db_path
    assert db.migrate() == STATE_DB_SCHEMA_VERSION
    assert db.schema_version() == STATE_DB_SCHEMA_VERSION


def test_connect_signature_introspection_exposes_class_path_support(tmp_path: Path) -> None:
    class_signature = inspect.signature(StateDB.connect)
    class_parameters = tuple(class_signature.parameters.values())
    assert class_parameters
    assert any("path" in parameter.name.lower() for parameter in class_parameters)

    # Instance-level connect usage remains a zero-required-arg call.
    instance = StateDB(tmp_path / "state" / "signature.sqlite3")
    instance_signature = inspect.signature(instance.connect)
    required_parameters = [
        parameter
        for parameter in instance_signature.parameters.values()
        if parameter.default is inspect.Signature.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    assert required_parameters == []


def test_connect_usage_snippet_matches_reproduction(tmp_path: Path) -> None:
    # Repro snippet shape from issue report: class-level connect + Path input.
    state_db_path = Path(tmp_path / "state" / "repro.sqlite3")

    db = StateDB.connect(state_db_path)
    assert isinstance(db, StateDB)

    schema_version = db.migrate()
    assert schema_version == STATE_DB_SCHEMA_VERSION

    row = db.query_one("SELECT 1 AS one")
    assert row == {"one": 1}


@pytest.mark.asyncio
async def test_awaiting_class_connect_is_actionable_or_async_supported(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "await-misuse.sqlite3"

    try:
        connect_result = StateDB.connect(db_path)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        assert "_assert_sync_io_allowed" not in message
        assert "PosixPath" not in message
        pytest.fail(f"class connect raised unexpected pre-await failure: {exc!r}")

    with pytest.raises(TypeError) as excinfo:
        await cast("Awaitable[object]", connect_result)

    message = str(excinfo.value)
    assert "await" in message.lower()
    assert "aconnect" in message
    assert "_assert_sync_io_allowed" not in message
    assert "posixpath" not in message.lower()


@pytest.mark.asyncio
async def test_aconnect_returns_state_db_instance(tmp_path: Path) -> None:
    db = await StateDB.aconnect(tmp_path / "state" / "aconnect.sqlite3")

    assert isinstance(db, StateDB)
    assert await db.migrate_async() == STATE_DB_SCHEMA_VERSION
