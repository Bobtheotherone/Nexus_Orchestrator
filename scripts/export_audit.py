"""
nexus-orchestrator — export deterministic audit bundles.

Purpose
- Export a deterministic, redacted audit bundle for a run via `EvidenceLedger`.
- Provide stable machine-readable metadata for downstream automation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _ensure_src_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a deterministic audit bundle for one run.",
    )
    parser.add_argument("--run-id", required=True, help="Run ID to export.")
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
        "--out",
        type=Path,
        default=None,
        help="Optional output path for the bundle zip.",
    )
    parser.add_argument(
        "--key-log",
        action="append",
        default=[],
        help="Repeatable key log path (relative to repo root or absolute).",
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
    print(f"run_id: {payload['run_id']}")
    print(f"db_path: {payload['db_path']}")
    print(f"evidence_root: {payload['evidence_root']}")
    print(f"bundle_path: {payload['bundle_path']}")
    print(f"bundle_size_bytes: {payload['bundle_size_bytes']}")
    print(f"bundle_sha256: {payload['bundle_sha256']}")
    print(f"evidence_count: {payload['evidence_count']}")
    print(f"member_count: {payload['member_count']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = args.db.expanduser().resolve()
    evidence_root = args.path.expanduser().resolve()
    output_path = args.out.expanduser().resolve() if args.out is not None else None

    _ensure_src_path()
    from nexus_orchestrator.knowledge_plane.evidence_ledger import EvidenceLedger
    from nexus_orchestrator.persistence.state_db import StateDB
    from nexus_orchestrator.utils.hashing import sha256_file

    try:
        db = StateDB(db_path)
        ledger = EvidenceLedger(db, evidence_root=evidence_root, repo_root=REPO_ROOT)
        key_logs = tuple(sorted((Path(item) for item in args.key_log), key=lambda path: str(path)))
        result = ledger.export_audit_bundle(
            run_id=args.run_id,
            output_path=output_path,
            key_log_paths=key_logs,
        )

        bundle_size = result.bundle_path.stat().st_size
        bundle_sha = sha256_file(result.bundle_path)

        payload: dict[str, object] = {
            "run_id": result.run_id,
            "db_path": db_path.as_posix(),
            "evidence_root": evidence_root.as_posix(),
            "bundle_path": result.bundle_path.as_posix(),
            "bundle_size_bytes": bundle_size,
            "bundle_sha256": bundle_sha,
            "evidence_count": len(result.evidence_ids),
            "evidence_ids": list(result.evidence_ids),
            "member_count": len(result.member_names),
            "member_names": list(result.member_names),
            "key_logs": [str(path) for path in key_logs],
        }

        if args.json:
            _emit_json(payload)
        else:
            _emit_text(payload)
        return 0
    except Exception as exc:  # noqa: BLE001
        error_payload = {
            "run_id": args.run_id,
            "db_path": db_path.as_posix(),
            "evidence_root": evidence_root.as_posix(),
            "error": str(exc),
        }
        if args.json:
            _emit_json(error_payload)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
