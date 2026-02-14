"""
nexus-orchestrator — verify evidence integrity.

Purpose
- Verify evidence artifact directories against manifest hashes via the evidence ledger.
- Produce deterministic, machine-readable output suitable for automation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if TYPE_CHECKING:
    from nexus_orchestrator.domain.models import EvidenceRecord
    from nexus_orchestrator.persistence.repositories import EvidenceRepo, WorkItemRepo
    from nexus_orchestrator.persistence.state_db import RowValue, StateDB

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_PAGE_SIZE = 500


def _ensure_src_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify evidence hashes/manifests using the deterministic evidence ledger.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("state") / "nexus.sqlite",
        help="State DB path.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("evidence"),
        help="Evidence root path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID scope.",
    )
    parser.add_argument(
        "--work-item-id",
        type=str,
        default=None,
        help="Optional work item ID scope.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args(argv)


def _to_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return normalized.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            out[str(key)] = _to_json_value(value[key])
        return out
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
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
    print(f"evidence_root: {payload['evidence_root']}")
    print(f"run_id: {payload['run_id']}")
    print(f"work_item_id: {payload['work_item_id']}")
    print(f"ok: {payload['ok']}")
    print(
        "counts: "
        + f"valid={payload['valid_count']} "
        + f"invalid={payload['invalid_count']} "
        + f"missing={payload['missing_count']} "
        + f"error={payload['error_count']}"
    )

    rows_obj = payload.get("rows")
    rows = rows_obj if isinstance(rows_obj, list) else []
    failing = [row for row in rows if isinstance(row, Mapping) and row.get("status") != "valid"]
    if failing:
        print("failing_rows:")
        for row in failing:
            evidence_id = row.get("evidence_id", "")
            status = row.get("status", "")
            error = row.get("error")
            print(f"  - {evidence_id}: {status} ({error})")


def _row_text(row: Mapping[str, RowValue], field: str, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    return value


def _list_work_items_for_run(work_item_repo: WorkItemRepo, run_id: str) -> tuple[str, ...]:
    work_item_ids: list[str] = []
    offset = 0
    while True:
        batch = work_item_repo.list_for_run(run_id, limit=_PAGE_SIZE, offset=offset)
        if not batch:
            break
        work_item_ids.extend(item.id for item in batch)
        offset += len(batch)
        if len(batch) < _PAGE_SIZE:
            break
    work_item_ids.sort()
    return tuple(work_item_ids)


def _list_evidence_for_work_item(
    evidence_repo: EvidenceRepo,
    work_item_id: str,
) -> tuple[EvidenceRecord, ...]:
    records: list[EvidenceRecord] = []
    offset = 0
    while True:
        batch = evidence_repo.list_for_work_item(work_item_id, limit=_PAGE_SIZE, offset=offset)
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(batch) < _PAGE_SIZE:
            break
    records.sort(key=lambda item: (item.run_id, item.work_item_id, item.created_at, item.id))
    return tuple(records)


def _load_scoped_evidence_records(
    *,
    db: StateDB,
    run_id: str | None,
    work_item_id: str | None,
) -> tuple[EvidenceRecord, ...]:
    _ensure_src_path()
    from nexus_orchestrator.domain.models import EvidenceRecord
    from nexus_orchestrator.persistence.repositories import EvidenceRepo, WorkItemRepo

    evidence_repo = EvidenceRepo(db)
    work_item_repo = WorkItemRepo(db)

    if work_item_id is not None:
        return _list_evidence_for_work_item(evidence_repo, work_item_id)

    if run_id is not None:
        records: list[EvidenceRecord] = []
        for item_id in _list_work_items_for_run(work_item_repo, run_id):
            records.extend(_list_evidence_for_work_item(evidence_repo, item_id))
        records.sort(key=lambda item: (item.run_id, item.work_item_id, item.created_at, item.id))
        return tuple(records)

    rows = db.query_all(
        "SELECT payload_json FROM evidence ORDER BY run_id ASC, work_item_id ASC, created_at ASC, id ASC"
    )
    records_all = [
        EvidenceRecord.from_json(_row_text(row, "payload_json", "evidence.payload_json"))
        for row in rows
    ]
    return tuple(records_all)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = args.db.expanduser().resolve()
    evidence_root = args.path.expanduser().resolve()

    _ensure_src_path()
    from nexus_orchestrator.knowledge_plane.evidence_ledger import EvidenceLedger
    from nexus_orchestrator.persistence.state_db import StateDB

    try:
        db = StateDB(db_path)
        ledger = EvidenceLedger(db, evidence_root=evidence_root, repo_root=REPO_ROOT)
        report = ledger.verify_integrity(run_id=args.run_id, work_item_id=args.work_item_id)

        evidence_records = _load_scoped_evidence_records(
            db=db,
            run_id=args.run_id,
            work_item_id=args.work_item_id,
        )
        evidence_index = {record.id: record for record in evidence_records}

        row_payloads: list[dict[str, object]] = []
        for row in report.rows:
            record = evidence_index.get(row.evidence_id)
            tool_versions = (
                {key: record.tool_versions[key] for key in sorted(record.tool_versions)}
                if record is not None
                else {}
            )
            row_payloads.append(
                {
                    "evidence_id": row.evidence_id,
                    "run_id": row.run_id,
                    "work_item_id": row.work_item_id,
                    "stage": row.stage,
                    "status": row.status,
                    "evidence_dir": None if row.evidence_dir is None else row.evidence_dir.as_posix(),
                    "missing_paths": list(row.missing_paths),
                    "hash_mismatches": list(row.hash_mismatches),
                    "error": row.error,
                    "checker_id": None if record is None else record.checker_id,
                    "tool_versions": tool_versions,
                }
            )

        ok = report.invalid_count == 0 and report.missing_count == 0 and report.error_count == 0
        payload: dict[str, object] = {
            "db_path": db_path.as_posix(),
            "evidence_root": evidence_root.as_posix(),
            "run_id": args.run_id,
            "work_item_id": args.work_item_id,
            "ok": ok,
            "valid_count": report.valid_count,
            "invalid_count": report.invalid_count,
            "missing_count": report.missing_count,
            "error_count": report.error_count,
            "rows": row_payloads,
        }

        if args.json:
            _emit_json(payload)
        else:
            _emit_text(payload)
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        error_payload = {
            "db_path": db_path.as_posix(),
            "evidence_root": evidence_root.as_posix(),
            "run_id": args.run_id,
            "work_item_id": args.work_item_id,
            "error": str(exc),
        }
        if args.json:
            _emit_json(error_payload)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
