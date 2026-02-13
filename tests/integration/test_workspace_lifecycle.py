"""
Integration tests for git-worktree-backed workspace lifecycle.

Coverage:
- directory layout and branch naming
- `git worktree list` visibility
- cleanup of worktree and branch
- GC dry-run candidate listing without deletion
- GC behavior for stale inactive workspaces
- GC safety for malicious/symlinked paths
- parallel workspace creation isolation
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.integration_plane import workspace_manager as wm
from nexus_orchestrator.integration_plane.workspace_manager import (
    Workspace,
    WorkspaceManager,
    WorkspacePaths,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        cmd = "git " + " ".join(args)
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git command failed: {cmd}: {detail}")
    return result


def _init_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    _git(repo_root, "init", "--initial-branch=integration", "--quiet")

    src_dir = repo_root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "seed.py").write_text("def seed() -> int:\n    return 1\n", encoding="utf-8")

    _git(repo_root, "add", ".")
    _git(
        repo_root,
        "-c",
        "user.name=Workspace Test",
        "-c",
        "user.email=workspace-test@example.com",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    return repo_root


def _write_workspace_metadata(
    workspace_dir: Path,
    *,
    branch_name: str,
    created_at: datetime,
) -> None:
    payload = {
        "branch_name": branch_name,
        "allowed_scope": [],
        "created_at": created_at.astimezone(UTC).isoformat(),
        "dependency_cache_mounts": [],
    }
    (workspace_dir / ".nexus-workspace.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def isolate_git_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


@pytest.mark.integration
def test_workspace_creation_layout_and_git_worktree_listing(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(repo_root=repo_root)

    workspace = manager.create_workspace(
        work_item_id="item-100",
        attempt_id="attempt-1",
        scope=("src/seed.py",),
    )

    expected_dir = repo_root / "workspaces" / "item-100" / "attempt-1"
    assert workspace.paths.workspace_dir == expected_dir.resolve(strict=False)
    assert workspace.paths.workspace_root == (repo_root / "workspaces").resolve(strict=False)
    assert workspace.branch_name == "work/item-100/attempt-1"
    assert workspace.allowed_scope == ("src/seed.py",)
    assert workspace.created_at.tzinfo is not None
    assert expected_dir.is_dir()

    listed_paths = {item.paths.workspace_dir for item in manager.list_active_workspaces()}
    assert expected_dir.resolve(strict=False) in listed_paths

    worktree_listing = _git(repo_root, "worktree", "list", "--porcelain").stdout
    assert f"worktree {expected_dir.resolve(strict=False)}" in worktree_listing
    assert "branch refs/heads/work/item-100/attempt-1" in worktree_listing


@pytest.mark.integration
def test_cleanup_workspace_removes_worktree_and_branch(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(repo_root=repo_root)

    workspace = manager.create_workspace(
        work_item_id="cleanup-item",
        attempt_id="attempt-1",
        scope=("src/seed.py",),
    )
    workspace_dir = workspace.paths.workspace_dir
    branch_ref = f"refs/heads/{workspace.branch_name}"

    assert workspace_dir.is_dir()
    assert _git(repo_root, "show-ref", "--verify", branch_ref, check=False).returncode == 0

    manager.cleanup_workspace(workspace)

    assert not workspace_dir.exists()
    assert _git(repo_root, "show-ref", "--verify", branch_ref, check=False).returncode != 0
    worktree_listing = _git(repo_root, "worktree", "list", "--porcelain").stdout
    assert f"worktree {workspace_dir}" not in worktree_listing


@pytest.mark.integration
def test_workspace_manager_rejects_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WorkspaceManager(repo_root=tmp_path / "missing-repo")

    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(repo_root=repo_root)

    with pytest.raises(ValueError):
        manager.create_workspace(work_item_id="../bad", attempt_id="a1", scope=("src/seed.py",))

    with pytest.raises(ValueError):
        manager.create_workspace(work_item_id="ok", attempt_id="a1", scope=("/abs/path.py",))

    with pytest.raises(ValueError):
        manager.gc(max_age_hours=-1)


@pytest.mark.integration
def test_create_workspace_duplicate_attempt_raises(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(repo_root=repo_root)
    manager.create_workspace(
        work_item_id="dup-item",
        attempt_id="attempt-1",
        scope=("src/seed.py",),
    )
    with pytest.raises(FileExistsError):
        manager.create_workspace(
            work_item_id="dup-item",
            attempt_id="attempt-1",
            scope=("src/seed.py",),
        )


@pytest.mark.integration
def test_gc_dry_run_returns_stale_candidates_without_deleting(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    fixed_now = datetime(2035, 1, 15, 12, 0, tzinfo=UTC)
    manager = WorkspaceManager(repo_root=repo_root, now_fn=lambda: fixed_now)

    stale_workspace = repo_root / "workspaces" / "dry-run-item" / "attempt-1"
    stale_workspace.mkdir(parents=True, exist_ok=True)
    stale_artifact = stale_workspace / "artifact.txt"
    stale_artifact.write_text("stale\n", encoding="utf-8")
    _write_workspace_metadata(
        stale_workspace,
        branch_name="work/dry-run-item/attempt-1",
        created_at=fixed_now - timedelta(hours=6),
    )

    removed = manager.gc(max_age_hours=1, dry_run=True)

    assert removed == (stale_workspace.resolve(strict=False),)
    assert stale_workspace.exists()
    assert stale_artifact.exists()


@pytest.mark.integration
def test_gc_removes_old_inactive_workspace_but_preserves_old_active(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    clock = {"now": datetime(2035, 1, 15, 9, 0, tzinfo=UTC)}
    manager = WorkspaceManager(repo_root=repo_root, now_fn=lambda: clock["now"])

    active_workspace = manager.create_workspace(
        work_item_id="active-item",
        attempt_id="attempt-1",
        scope=("src/seed.py",),
    )

    stale_workspace = repo_root / "workspaces" / "stale-item" / "attempt-9"
    stale_workspace.mkdir(parents=True, exist_ok=True)
    (stale_workspace / "artifact.txt").write_text("stale\n", encoding="utf-8")
    _write_workspace_metadata(
        stale_workspace,
        branch_name="work/stale-item/attempt-9",
        created_at=clock["now"] - timedelta(hours=8),
    )

    clock["now"] = clock["now"] + timedelta(hours=24)

    removed = manager.gc(max_age_hours=1, dry_run=False)

    assert stale_workspace.resolve(strict=False) in removed
    assert active_workspace.paths.workspace_dir not in removed
    assert not stale_workspace.exists()
    assert active_workspace.paths.workspace_dir.exists()


@pytest.mark.integration
def test_list_active_workspaces_falls_back_when_metadata_invalid(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(repo_root=repo_root)

    workspace = manager.create_workspace(
        work_item_id="invalid-meta",
        attempt_id="attempt-1",
        scope=("src/seed.py",),
    )
    metadata_path = workspace.paths.workspace_dir / ".nexus-workspace.json"
    metadata_path.write_text("{invalid-json", encoding="utf-8")

    listed = manager.list_active_workspaces()
    only = next(item for item in listed if item.branch_name == workspace.branch_name)
    assert only.allowed_scope == ()
    assert only.created_at.tzinfo is not None


@pytest.mark.integration
def test_cleanup_workspace_rejects_path_outside_workspace_root(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(repo_root=repo_root)
    outside_dir = tmp_path / "outside-ws"
    outside_dir.mkdir(parents=True, exist_ok=True)

    fake_workspace = Workspace(
        paths=WorkspacePaths(
            repo_root=repo_root.resolve(strict=False),
            workspace_root=(repo_root / "workspaces").resolve(strict=False),
            workspace_dir=outside_dir.resolve(strict=False),
        ),
        branch_name="work/fake/attempt",
        allowed_scope=(),
        created_at=datetime.now(tz=UTC),
        dependency_cache_mounts=(),
    )
    with pytest.raises(ValueError, match="outside workspace root"):
        manager.cleanup_workspace(fake_workspace)


@pytest.mark.integration
def test_gc_safety_does_not_delete_outside_workspace_root_for_symlink(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    manager = WorkspaceManager(
        repo_root=repo_root,
        now_fn=lambda: datetime(2100, 1, 1, tzinfo=UTC),
    )

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    sentinel = outside_dir / "sentinel.txt"
    sentinel.write_text("do-not-delete\n", encoding="utf-8")

    malicious_parent = repo_root / "workspaces" / "malicious-item"
    malicious_parent.mkdir(parents=True, exist_ok=True)
    malicious_workspace = malicious_parent / "attempt-1"
    try:
        malicious_workspace.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are not supported in this environment: {exc}")

    removed = manager.gc(max_age_hours=1, dry_run=False)

    assert malicious_workspace in removed
    assert not malicious_workspace.exists()
    assert not malicious_workspace.is_symlink()
    assert outside_dir.exists()
    assert sentinel.exists()
    assert sentinel.read_text(encoding="utf-8") == "do-not-delete\n"


@pytest.mark.integration
def test_parallel_workspace_creation_isolated_without_collisions(tmp_path: Path) -> None:
    repo_root = _init_repo(tmp_path)
    fixed_now = datetime(2035, 1, 16, 8, 0, tzinfo=UTC)
    manager = WorkspaceManager(repo_root=repo_root, now_fn=lambda: fixed_now)

    def _create(index: int) -> Workspace:
        return manager.create_workspace(
            work_item_id="parallel-item",
            attempt_id=f"attempt-{index}",
            scope=("src/seed.py",),
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        workspaces = list(executor.map(_create, range(8)))

    workspace_paths = {workspace.paths.workspace_dir for workspace in workspaces}
    branch_names = {workspace.branch_name for workspace in workspaces}
    expected_paths = {
        (repo_root / "workspaces" / "parallel-item" / f"attempt-{index}").resolve(strict=False)
        for index in range(8)
    }
    expected_branches = {f"work/parallel-item/attempt-{index}" for index in range(8)}

    assert workspace_paths == expected_paths
    assert branch_names == expected_branches
    assert all(path.is_dir() for path in workspace_paths)

    for workspace in workspaces:
        metadata_payload = json.loads(
            (workspace.paths.workspace_dir / ".nexus-workspace.json").read_text(encoding="utf-8")
        )
        assert metadata_payload["branch_name"] == workspace.branch_name
        assert metadata_payload["created_at"] == fixed_now.isoformat()

    worktree_listing = _git(repo_root, "worktree", "list", "--porcelain").stdout
    for workspace in workspaces:
        assert f"worktree {workspace.paths.workspace_dir}" in worktree_listing


@pytest.mark.integration
def test_workspace_helper_validators_cover_error_paths() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        wm._validate_identifier("", "field")
    with pytest.raises(ValueError, match="must not contain '/'"):
        wm._validate_identifier("a/b", "field")
    with pytest.raises(ValueError, match="must not be '.' or '..'"):
        wm._validate_identifier("..", "field")
    with pytest.raises(ValueError, match="unsupported characters"):
        wm._validate_identifier("bad value", "field")

    assert wm._is_valid_workspace_branch_name("work/item/attempt") is True
    assert wm._is_valid_workspace_branch_name("work/item") is False
    assert wm._is_valid_workspace_branch_name("bad/item/attempt") is False

    normalized = wm._normalize_scope(("src/a.py", "./src/a.py", ""))
    assert normalized == ("src/a.py",)
    with pytest.raises(ValueError, match="relative"):
        wm._normalize_scope(("/abs/path.py",))
    with pytest.raises(ValueError, match="traverse upwards"):
        wm._normalize_scope(("src/../bad.py",))
