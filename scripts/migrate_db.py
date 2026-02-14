"""
nexus-orchestrator — migrate state DB schema.

Purpose
- Apply deterministic SQLite migrations for the state DB.
- Provide migration status output for both apply and dry-run flows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if TYPE_CHECKING:
    from nexus_orchestrator.persistence.state_db import MigrationRecord

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _ensure_src_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply or inspect StateDB migrations deterministically.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("state") / "nexus.sqlite",
        help="Path to the state SQLite database.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show migration status without mutating the target database.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args(argv)


def _migration_catalog() -> tuple[MigrationRecord, ...]:
    """Load code-defined migration metadata via a temporary StateDB."""

    _ensure_src_path()
    from nexus_orchestrator.persistence.state_db import StateDB

    with TemporaryDirectory(prefix="nexus-migrate-catalog-") as temp_dir:
        catalog_db_path = Path(temp_dir) / "catalog.sqlite"
        catalog_db = StateDB(catalog_db_path)
        catalog_db.migrate()
        history = tuple(sorted(catalog_db.schema_history(), key=lambda row: row.version))
    return history


def _read_target_history(db_path: Path) -> tuple[int, tuple[MigrationRecord, ...]]:
    """Read currently applied migration records without forcing migrations."""

    _ensure_src_path()
    from nexus_orchestrator.persistence.state_db import StateDB

    resolved_path = db_path.expanduser().resolve()
    if not resolved_path.exists():
        return 0, ()

    db = StateDB(resolved_path)
    table_exists = db.query_one(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'schema_versions'"
    )
    if table_exists is None:
        return 0, ()

    history = tuple(sorted(db.schema_history(), key=lambda row: row.version))
    current_version = history[-1].version if history else 0
    return current_version, history


def _migration_row_version(row: Mapping[str, object]) -> int:
    version = row.get("version")
    if not isinstance(version, int):
        raise TypeError("migration row 'version' must be an integer")
    return version


def _status_rows(
    *,
    catalog: tuple[MigrationRecord, ...],
    applied_history: tuple[MigrationRecord, ...],
) -> tuple[dict[str, object], ...]:
    applied_by_version = {row.version: row for row in applied_history}
    catalog_versions = {row.version for row in catalog}

    rows: list[dict[str, object]] = []
    for migration in catalog:
        applied = applied_by_version.get(migration.version)
        if applied is None:
            status = "pending"
            applied_at: str | None = None
        elif applied.checksum != migration.checksum:
            status = "checksum_mismatch"
            applied_at = applied.applied_at
        else:
            status = "applied"
            applied_at = applied.applied_at

        rows.append(
            {
                "version": migration.version,
                "name": migration.name,
                "checksum": migration.checksum,
                "status": status,
                "applied_at": applied_at,
            }
        )

    for applied in sorted(applied_history, key=lambda row: row.version):
        if applied.version in catalog_versions:
            continue
        rows.append(
            {
                "version": applied.version,
                "name": applied.name,
                "checksum": applied.checksum,
                "status": "unknown_to_binary",
                "applied_at": applied.applied_at,
            }
        )

    rows.sort(key=_migration_row_version)
    return tuple(rows)


def _to_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if is_dataclass(value) and not isinstance(value, type):
        return _to_json_value(asdict(value))
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            out[str(key)] = _to_json_value(value[key])
        return out
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    return str(value)


def _emit_json(payload: Mapping[str, object]) -> None:
    print(
        json.dumps(
            _to_json_value(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def _emit_text(payload: Mapping[str, object]) -> None:
    print(f"db_path: {payload['db_path']}")
    print(f"dry_run: {payload['dry_run']}")
    print(f"schema_version: {payload['schema_version']}")
    print(f"target_schema_version: {payload['target_schema_version']}")
    print(f"up_to_date: {payload['up_to_date']}")
    print("migrations:")

    migrations_obj = payload.get("migrations")
    migrations = migrations_obj if isinstance(migrations_obj, list) else []
    for row in migrations:
        if not isinstance(row, Mapping):
            continue
        version = row.get("version")
        status = row.get("status")
        name = row.get("name")
        checksum = row.get("checksum")
        print(f"  v{version}: {status} ({name}, checksum={checksum})")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    resolved_db_path = args.db.expanduser().resolve()
    existed_before = resolved_db_path.exists()

    _ensure_src_path()
    from nexus_orchestrator.constants import STATE_DB_SCHEMA_VERSION
    from nexus_orchestrator.persistence.state_db import StateDB

    try:
        catalog = _migration_catalog()
        if args.dry_run:
            current_version, applied_history = _read_target_history(resolved_db_path)
        else:
            db = StateDB(resolved_db_path)
            db.migrate()
            current_version, applied_history = _read_target_history(resolved_db_path)

        rows = _status_rows(catalog=catalog, applied_history=applied_history)
        pending = sum(1 for row in rows if row["status"] == "pending")
        mismatches = sum(1 for row in rows if row["status"] == "checksum_mismatch")
        unknown = sum(1 for row in rows if row["status"] == "unknown_to_binary")

        payload: dict[str, object] = {
            "db_path": resolved_db_path.as_posix(),
            "dry_run": bool(args.dry_run),
            "db_existed": existed_before,
            "schema_version": current_version,
            "target_schema_version": STATE_DB_SCHEMA_VERSION,
            "up_to_date": pending == 0 and mismatches == 0 and unknown == 0,
            "pending_migrations": pending,
            "checksum_mismatch_count": mismatches,
            "unknown_to_binary_count": unknown,
            "migrations": [dict(row) for row in rows],
        }

        if args.json:
            _emit_json(payload)
        else:
            _emit_text(payload)
        return 0 if mismatches == 0 and unknown == 0 else 1
    except Exception as exc:  # noqa: BLE001
        error_payload = {
            "db_path": resolved_db_path.as_posix(),
            "dry_run": bool(args.dry_run),
            "error": str(exc),
        }
        if args.json:
            _emit_json(cast("Mapping[str, object]", error_payload))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
