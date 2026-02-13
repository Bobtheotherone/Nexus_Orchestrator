"""Git-worktree-backed workspace lifecycle management."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_orchestrator.constants import (
    DEFAULT_INTEGRATION_BRANCH,
    DEFAULT_WORK_BRANCH_PREFIX,
    WORKSPACES_DIR,
)
from nexus_orchestrator.utils.fs import safe_delete

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

_WORKSPACE_METADATA_FILE = ".nexus-workspace.json"
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Resolved workspace path bundle."""

    repo_root: Path
    workspace_root: Path
    workspace_dir: Path


@dataclass(frozen=True, slots=True)
class Workspace:
    """
    Workspace descriptor returned by the manager.

    Required fields:
    - paths
    - branch_name
    - allowed_scope
    - created_at
    - dependency_cache_mounts
    """

    paths: WorkspacePaths
    branch_name: str
    allowed_scope: tuple[str, ...]
    created_at: datetime
    dependency_cache_mounts: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _WorkspaceMetadata:
    branch_name: str
    allowed_scope: tuple[str, ...]
    created_at: datetime
    dependency_cache_mounts: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class _WorktreeRecord:
    path: Path
    branch_ref: str | None


class WorkspaceManager:
    """Manage per-attempt isolated git workspaces using `git worktree`."""

    def __init__(
        self,
        repo_root: str | Path,
        workspace_root: str | Path | None = None,
        *,
        dependency_cache_mounts: Sequence[str | Path] = (),
        now_fn: Callable[[], datetime] | None = None,
        env_overrides: Mapping[str, str] | None = None,
    ) -> None:
        resolved_repo = Path(repo_root).expanduser().resolve(strict=True)
        if not resolved_repo.is_dir():
            raise NotADirectoryError(f"{resolved_repo} is not a directory")

        default_workspace_root = resolved_repo.joinpath(*WORKSPACES_DIR.parts)
        if workspace_root is None:
            resolved_workspace_root = default_workspace_root
        else:
            candidate_root = Path(workspace_root).expanduser()
            resolved_workspace_root = (
                candidate_root.resolve(strict=False)
                if candidate_root.is_absolute()
                else (resolved_repo / candidate_root).resolve(strict=False)
            )

        self._repo_root = resolved_repo
        self._workspace_root = resolved_workspace_root
        self._dependency_cache_mounts = tuple(
            Path(path).expanduser().resolve(strict=False) for path in dependency_cache_mounts
        )
        self._now_fn: Callable[[], datetime] = now_fn if now_fn is not None else _utc_now
        self._env_overrides = dict(env_overrides or {})
        self._lock = threading.RLock()

    def create_workspace(
        self,
        work_item_id: str,
        attempt_id: str,
        scope: Sequence[str],
        base_branch: str = DEFAULT_INTEGRATION_BRANCH,
    ) -> Workspace:
        """Create and register one isolated worktree for an attempt."""

        normalized_work_item = _validate_identifier(work_item_id, "work_item_id")
        normalized_attempt = _validate_identifier(attempt_id, "attempt_id")
        normalized_scope = _normalize_scope(scope)

        branch_name = f"{DEFAULT_WORK_BRANCH_PREFIX}/{normalized_work_item}/{normalized_attempt}"
        workspace_dir = self._workspace_root / normalized_work_item / normalized_attempt

        with self._lock:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
            workspace_dir.parent.mkdir(parents=True, exist_ok=True)

            if workspace_dir.exists() or workspace_dir.is_symlink():
                raise FileExistsError(f"workspace directory already exists: {workspace_dir}")
            if self._branch_exists(branch_name):
                raise FileExistsError(f"workspace branch already exists: {branch_name}")

            self._run_git(
                [
                    "worktree",
                    "add",
                    "--quiet",
                    "-b",
                    branch_name,
                    str(workspace_dir),
                    base_branch,
                ],
                check=True,
            )

            workspace = Workspace(
                paths=WorkspacePaths(
                    repo_root=self._repo_root,
                    workspace_root=self._workspace_root.resolve(strict=False),
                    workspace_dir=workspace_dir.resolve(strict=False),
                ),
                branch_name=branch_name,
                allowed_scope=normalized_scope,
                created_at=self._ensure_aware_utc(self._now_fn()),
                dependency_cache_mounts=self._dependency_cache_mounts,
            )
            try:
                self._write_workspace_metadata(workspace)
            except Exception:
                self._cleanup_workspace_internal(workspace, delete_branch=True)
                raise

            return workspace

    def list_active_workspaces(self) -> tuple[Workspace, ...]:
        """Return all active worktree-backed workspaces under ``workspace_root``."""

        with self._lock:
            records = self._worktree_records()

        workspaces: list[Workspace] = []
        for record in records:
            candidate = record.path.resolve(strict=False)
            if not _is_relative_to(candidate, self._workspace_root.resolve(strict=False)):
                continue
            if not candidate.exists():
                continue

            metadata = self._read_workspace_metadata(candidate)
            branch_name = ""
            if record.branch_ref and record.branch_ref.startswith("refs/heads/"):
                branch_name = record.branch_ref.removeprefix("refs/heads/")
            elif metadata is not None:
                branch_name = metadata.branch_name

            if metadata is not None:
                allowed_scope = metadata.allowed_scope
                created_at = metadata.created_at
                dependency_cache_mounts = metadata.dependency_cache_mounts
            else:
                allowed_scope = ()
                created_at = _mtime_as_utc(candidate)
                dependency_cache_mounts = self._dependency_cache_mounts

            workspaces.append(
                Workspace(
                    paths=WorkspacePaths(
                        repo_root=self._repo_root,
                        workspace_root=self._workspace_root.resolve(strict=False),
                        workspace_dir=candidate,
                    ),
                    branch_name=branch_name,
                    allowed_scope=allowed_scope,
                    created_at=created_at,
                    dependency_cache_mounts=dependency_cache_mounts,
                )
            )

        workspaces.sort(key=lambda workspace: workspace.paths.workspace_dir.as_posix())
        return tuple(workspaces)

    def cleanup_workspace(self, workspace: Workspace) -> None:
        """Remove one managed worktree and delete its branch when safe."""

        with self._lock:
            self._cleanup_workspace_internal(workspace, delete_branch=True)

    def gc(self, max_age_hours: float, dry_run: bool = False) -> tuple[Path, ...]:
        """
        Garbage-collect stale workspace directories within ``workspace_root``.

        Safety rules:
        - Never delete anything outside ``workspace_root``.
        - Never delete active git worktrees.
        - Never traverse symlinked parent directories.
        """

        if max_age_hours < 0:
            raise ValueError("max_age_hours must be >= 0")

        cutoff = self._ensure_aware_utc(self._now_fn()) - timedelta(hours=max_age_hours)
        removed: list[Path] = []

        with self._lock:
            active_records = self._worktree_records()
            active_paths = {
                record.path.resolve(strict=False)
                for record in active_records
                if _is_relative_to(
                    record.path.resolve(strict=False), self._workspace_root.resolve(strict=False)
                )
            }
            active_branch_refs = {
                record.branch_ref for record in active_records if record.branch_ref
            }

            if not self._workspace_root.exists():
                return ()

            for item_dir in sorted(self._workspace_root.iterdir(), key=lambda path: path.name):
                if item_dir.is_symlink() or not item_dir.is_dir():
                    continue

                for attempt_dir in sorted(item_dir.iterdir(), key=lambda path: path.name):
                    if not (attempt_dir.is_dir() or attempt_dir.is_symlink()):
                        continue

                    managed_path = _managed_path(attempt_dir)
                    if not _is_relative_to(
                        managed_path,
                        self._workspace_root.resolve(strict=False),
                    ):
                        continue

                    if managed_path.resolve(strict=False) in active_paths:
                        continue

                    created_at = self._candidate_created_at(managed_path)
                    if created_at > cutoff:
                        continue

                    if dry_run:
                        removed.append(managed_path)
                        continue

                    metadata = self._read_workspace_metadata(managed_path)
                    branch_name = self._resolve_branch_name_for_path(
                        managed_path,
                        metadata=metadata,
                    )

                    self._run_git(
                        ["worktree", "remove", "--force", str(managed_path)],
                        check=False,
                    )
                    if managed_path.exists() or managed_path.is_symlink():
                        safe_delete(managed_path, self._workspace_root)

                    if (
                        branch_name is not None
                        and f"refs/heads/{branch_name}" not in active_branch_refs
                        and self._branch_exists(branch_name)
                    ):
                        self._run_git(["branch", "-D", branch_name], check=True)

                    self._prune_empty_workspace_parents(managed_path)
                    removed.append(managed_path)

        return tuple(removed)

    def _cleanup_workspace_internal(self, workspace: Workspace, *, delete_branch: bool) -> None:
        workspace_dir = _managed_path(workspace.paths.workspace_dir)
        if not _is_relative_to(workspace_dir, self._workspace_root.resolve(strict=False)):
            raise ValueError(f"workspace path is outside workspace root: {workspace_dir}")

        self._run_git(["worktree", "remove", "--force", str(workspace_dir)], check=False)
        if workspace_dir.exists() or workspace_dir.is_symlink():
            safe_delete(workspace_dir, self._workspace_root)

        if delete_branch and workspace.branch_name:
            active_branch_refs = {
                record.branch_ref for record in self._worktree_records() if record.branch_ref
            }
            branch_ref = f"refs/heads/{workspace.branch_name}"
            if branch_ref not in active_branch_refs and self._branch_exists(workspace.branch_name):
                self._run_git(["branch", "-D", workspace.branch_name], check=True)

        self._prune_empty_workspace_parents(workspace_dir)

    def _resolve_branch_name_for_path(
        self,
        workspace_dir: Path,
        *,
        metadata: _WorkspaceMetadata | None,
    ) -> str | None:
        if metadata is not None and _is_valid_workspace_branch_name(metadata.branch_name):
            return metadata.branch_name

        try:
            relative = workspace_dir.relative_to(self._workspace_root.resolve(strict=False))
        except ValueError:
            return None
        if len(relative.parts) != 2:
            return None
        work_item, attempt = relative.parts
        if not _SAFE_ID_PATTERN.fullmatch(work_item):
            return None
        if not _SAFE_ID_PATTERN.fullmatch(attempt):
            return None
        return f"{DEFAULT_WORK_BRANCH_PREFIX}/{work_item}/{attempt}"

    def _candidate_created_at(self, workspace_dir: Path) -> datetime:
        metadata = self._read_workspace_metadata(workspace_dir)
        if metadata is not None:
            return metadata.created_at
        return _mtime_as_utc(workspace_dir)

    def _write_workspace_metadata(self, workspace: Workspace) -> None:
        metadata_path = workspace.paths.workspace_dir / _WORKSPACE_METADATA_FILE
        payload = {
            "branch_name": workspace.branch_name,
            "allowed_scope": list(workspace.allowed_scope),
            "created_at": workspace.created_at.isoformat(),
            "dependency_cache_mounts": [
                path.as_posix() for path in workspace.dependency_cache_mounts
            ],
        }
        metadata_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _read_workspace_metadata(self, workspace_dir: Path) -> _WorkspaceMetadata | None:
        metadata_path = workspace_dir / _WORKSPACE_METADATA_FILE
        if not metadata_path.exists() or metadata_path.is_symlink():
            return None

        try:
            raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw_payload, dict):
            return None

        branch_name_obj = raw_payload.get("branch_name")
        created_at_obj = raw_payload.get("created_at")
        allowed_scope_obj = raw_payload.get("allowed_scope")
        dependency_mounts_obj = raw_payload.get("dependency_cache_mounts")

        if not isinstance(branch_name_obj, str):
            return None
        if not isinstance(created_at_obj, str):
            return None

        try:
            created_at = self._ensure_aware_utc(datetime.fromisoformat(created_at_obj))
        except ValueError:
            return None

        allowed_scope: tuple[str, ...]
        if isinstance(allowed_scope_obj, list):
            scope_values = [item for item in allowed_scope_obj if isinstance(item, str)]
            allowed_scope = _normalize_scope(scope_values)
        else:
            allowed_scope = ()

        dependency_cache_mounts: tuple[Path, ...]
        if isinstance(dependency_mounts_obj, list):
            mounts: list[Path] = []
            for mount in dependency_mounts_obj:
                if isinstance(mount, str):
                    mounts.append(Path(mount).expanduser().resolve(strict=False))
            dependency_cache_mounts = tuple(mounts)
        else:
            dependency_cache_mounts = self._dependency_cache_mounts

        return _WorkspaceMetadata(
            branch_name=branch_name_obj,
            allowed_scope=allowed_scope,
            created_at=created_at,
            dependency_cache_mounts=dependency_cache_mounts,
        )

    def _branch_exists(self, branch_name: str) -> bool:
        result = self._run_git(
            ["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            check=False,
        )
        return result.returncode == 0

    def _worktree_records(self) -> tuple[_WorktreeRecord, ...]:
        result = self._run_git(["worktree", "list", "--porcelain"], check=True)
        records: list[_WorktreeRecord] = []

        path_value: Path | None = None
        branch_ref: str | None = None
        for line in result.stdout.splitlines():
            if not line.strip():
                if path_value is not None:
                    records.append(
                        _WorktreeRecord(
                            path=path_value,
                            branch_ref=branch_ref,
                        )
                    )
                path_value = None
                branch_ref = None
                continue

            field, _, value = line.partition(" ")
            if field == "worktree":
                path_value = Path(value.strip()).expanduser().resolve(strict=False)
            elif field == "branch":
                branch_ref = value.strip()

        if path_value is not None:
            records.append(_WorktreeRecord(path=path_value, branch_ref=branch_ref))

        records.sort(key=lambda record: record.path.as_posix())
        return tuple(records)

    def _run_git(
        self,
        args: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
        env.update(self._env_overrides)
        proc = subprocess.run(
            ["git", *args],
            cwd=self._repo_root,
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        if check and proc.returncode != 0:
            cmd = "git " + " ".join(args)
            detail = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"command failed with exit code {proc.returncode}: {cmd}: {detail}")
        return proc

    def _ensure_aware_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _prune_empty_workspace_parents(self, workspace_dir: Path) -> None:
        root = self._workspace_root.resolve(strict=False)
        for candidate in (workspace_dir.parent,):
            try:
                resolved = candidate.resolve(strict=False)
            except OSError:
                continue
            if resolved == root:
                continue
            if not _is_relative_to(resolved, root):
                continue
            if candidate.exists() and candidate.is_dir() and not any(candidate.iterdir()):
                candidate.rmdir()


def _is_valid_workspace_branch_name(value: str) -> bool:
    parts = value.split("/")
    return (
        len(parts) == 3
        and parts[0] == DEFAULT_WORK_BRANCH_PREFIX
        and _SAFE_ID_PATTERN.fullmatch(parts[1]) is not None
        and _SAFE_ID_PATTERN.fullmatch(parts[2]) is not None
    )


def _validate_identifier(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if "/" in value:
        raise ValueError(f"{field_name} must not contain '/'")
    if value in {".", ".."}:
        raise ValueError(f"{field_name} must not be '.' or '..'")
    if _SAFE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} contains unsupported characters: {value!r}")
    return value


def _normalize_scope(scope: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in scope:
        cleaned = raw_value.strip().replace("\\", "/")
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        if not cleaned:
            continue
        pure_parts = Path(cleaned).parts
        if Path(cleaned).is_absolute():
            raise ValueError(f"scope path must be relative: {raw_value!r}")
        if ".." in pure_parts:
            raise ValueError(f"scope path must not traverse upwards: {raw_value!r}")
        if cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return tuple(normalized)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _managed_path(path: Path) -> Path:
    parent = path.parent.resolve(strict=False)
    return parent / path.name


def _mtime_as_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.lstat().st_mtime, tz=UTC)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


__all__ = ["Workspace", "WorkspaceManager", "WorkspacePaths"]
