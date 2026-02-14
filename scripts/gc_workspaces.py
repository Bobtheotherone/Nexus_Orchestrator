"""
nexus-orchestrator — workspace garbage collection.

Purpose
- Remove stale, safe-to-delete git workspaces under a configured workspace root.
- Keep behavior conservative and deterministic via `WorkspaceManager.gc(...)`.
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
        description="Garbage-collect stale workspaces without scanning outside workspace root.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing the git worktrees.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("workspaces"),
        help="Workspace root (absolute, or relative to repo root).",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=168.0,
        help="Delete workspaces older than this many hours.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stale workspaces without deleting them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args(argv)


def _resolve_workspace_root(*, repo_root: Path, workspace_root: Path) -> Path:
    candidate = workspace_root.expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root / candidate).resolve()


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
    print(f"repo_root: {payload['repo_root']}")
    print(f"workspace_root: {payload['workspace_root']}")
    print(f"max_age_hours: {payload['max_age_hours']}")
    print(f"dry_run: {payload['dry_run']}")
    print(f"active_workspace_count: {payload['active_workspace_count']}")
    print(f"removed_count: {payload['removed_count']}")

    active_paths_obj = payload.get("active_workspace_paths")
    active_paths = active_paths_obj if isinstance(active_paths_obj, list) else []
    if active_paths:
        print("active_workspace_paths:")
        for item in active_paths:
            print(f"  - {item}")

    removed_paths_obj = payload.get("removed_paths")
    removed_paths = removed_paths_obj if isinstance(removed_paths_obj, list) else []
    if removed_paths:
        print("removed_paths:")
        for item in removed_paths:
            print(f"  - {item}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    resolved_repo_root = args.repo_root.expanduser().resolve()
    resolved_workspace_root = _resolve_workspace_root(
        repo_root=resolved_repo_root,
        workspace_root=args.workspace_root,
    )

    _ensure_src_path()
    from nexus_orchestrator.integration_plane.workspace_manager import WorkspaceManager

    try:
        manager = WorkspaceManager(repo_root=resolved_repo_root, workspace_root=resolved_workspace_root)
        active = manager.list_active_workspaces()
        removed = manager.gc(max_age_hours=args.max_age_hours, dry_run=args.dry_run)

        active_paths = sorted(workspace.paths.workspace_dir.as_posix() for workspace in active)
        removed_paths = sorted(path.as_posix() for path in removed)

        payload: dict[str, object] = {
            "repo_root": resolved_repo_root.as_posix(),
            "workspace_root": resolved_workspace_root.as_posix(),
            "max_age_hours": float(args.max_age_hours),
            "dry_run": bool(args.dry_run),
            "active_workspace_count": len(active_paths),
            "active_workspace_paths": active_paths,
            "removed_count": len(removed_paths),
            "removed_paths": removed_paths,
        }

        if args.json:
            _emit_json(payload)
        else:
            _emit_text(payload)
        return 0
    except Exception as exc:  # noqa: BLE001
        error_payload = {
            "repo_root": resolved_repo_root.as_posix(),
            "workspace_root": resolved_workspace_root.as_posix(),
            "dry_run": bool(args.dry_run),
            "error": str(exc),
        }
        if args.json:
            _emit_json(error_payload)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
