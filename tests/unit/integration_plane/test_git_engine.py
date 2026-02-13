"""
nexus-orchestrator — test suite for integration plane git engine.

File: tests/unit/integration_plane/test_git_engine.py
Last updated: 2026-02-13

Purpose
- Validate deterministic/safe GitEngine primitives over local temporary repositories.
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.integration_plane.git_engine import (
    CommandResult as GitRunResult,
)
from nexus_orchestrator.integration_plane.git_engine import (
    GitCommandError,
    GitEngine,
    GitEngineError,
    ProtectedBranchError,
    SanitizationError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        msg = f"git command failed: git {' '.join(args)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        raise AssertionError(msg)
    return completed


@pytest.fixture(autouse=True)
def isolated_git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    home.mkdir()
    xdg.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


def init_engine(tmp_path: Path) -> tuple[GitEngine, Path]:
    repo = tmp_path / "repo"
    engine = GitEngine(repo)
    engine.init_or_open()
    return engine, repo


def commit_file(worktree: Path, rel_path: str, content: str, message: str) -> str:
    path = worktree / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    run_git(worktree, "add", "--all")
    run_git(worktree, "commit", "-m", message)
    return run_git(worktree, "rev-parse", "HEAD").stdout.strip()


def test_init_or_open_ensures_main_and_integration_branches(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    engine = GitEngine(repo)

    first = engine.init_or_open()
    assert first.created is True

    branches = set(run_git(repo, "branch", "--format=%(refname:short)").stdout.splitlines())
    assert {"main", "integration"}.issubset(branches)

    second = engine.init_or_open()
    assert second.created is False


@pytest.mark.parametrize("unsafe", ["bad value", "..", "-bad", "bad:value", "bad\nvalue"])
def test_create_work_branch_rejects_unsafe_tokens(tmp_path: Path, unsafe: str) -> None:
    engine, _repo = init_engine(tmp_path)

    with pytest.raises(SanitizationError):
        engine.create_work_branch(unsafe, "attempt1")

    with pytest.raises(SanitizationError):
        engine.create_work_branch("workitem", unsafe)


def test_create_work_branch_uses_work_prefix(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)

    created = engine.create_work_branch("WI_42", "attempt-2")
    assert created.name == "work/WI_42/attempt-2"
    assert created.base == "integration"

    ref = "refs/heads/work/WI_42/attempt-2"
    assert run_git(repo, "show-ref", "--verify", "--quiet", ref, check=False).returncode == 0


def test_create_work_branch_rejects_duplicate_branch(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    engine.create_work_branch("WI", "dup")
    with pytest.raises(GitEngineError, match="Branch already exists"):
        engine.create_work_branch("WI", "dup")


def test_create_and_remove_worktree(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "worktree"

    created = engine.create_worktree(branch, worktree_path)
    assert created.path == worktree_path
    assert worktree_path.exists()
    assert run_git(worktree_path, "branch", "--show-current").stdout.strip() == branch

    engine.remove_worktree(worktree_path)
    assert not worktree_path.exists()


def test_apply_unified_diff_patch_in_worktree(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "patch-worktree"
    engine.create_worktree(branch, worktree_path)

    patch = """--- /dev/null
+++ b/hello.txt
@@ -0,0 +1 @@
+hello
"""
    engine.apply_patch(worktree_path, patch)

    assert (worktree_path / "hello.txt").read_text(encoding="utf-8") == "hello\n"
    status = run_git(worktree_path, "status", "--short").stdout.splitlines()
    assert status == ["A  hello.txt"]

    engine.remove_worktree(worktree_path)


def test_apply_patch_rejects_empty_content(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "empty-patch-worktree"
    engine.create_worktree(branch, worktree_path)
    with pytest.raises(GitEngineError, match="Patch content cannot be empty"):
        engine.apply_patch(worktree_path, "   ")
    engine.remove_worktree(worktree_path)


@pytest.mark.parametrize(
    ("target_path", "expected_error"),
    [
        ("b/../escape.txt", "path traversal"),
        ("/tmp/absolute.txt", "absolute paths"),
        ("b/.git/config", ".git targets"),
    ],
)
def test_apply_patch_rejects_forbidden_paths_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_path: str,
    expected_error: str,
) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "unsafe-patch-worktree"
    engine.create_worktree(branch, worktree_path)

    apply_calls: list[tuple[str, ...]] = []
    original_run_git = engine._run_git

    def tracked_run_git(
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        input_text: str | None = None,
    ) -> GitRunResult:
        command_args = tuple(args)
        if command_args[:1] == ("apply",):
            apply_calls.append(command_args)
        return original_run_git(args, cwd=cwd, check=check, input_text=input_text)

    monkeypatch.setattr(engine, "_run_git", tracked_run_git)

    patch = f"""--- /dev/null
+++ {target_path}
@@ -0,0 +1 @@
+blocked
"""

    with pytest.raises(GitEngineError) as excinfo:
        engine.apply_patch(worktree_path, patch)

    message = str(excinfo.value)
    assert expected_error in message
    assert "Patch path" in message
    assert apply_calls == []

    engine.remove_worktree(worktree_path)


def test_commit_adds_required_nexus_trailers(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "commit-worktree"
    engine.create_worktree(branch, worktree_path)

    (worktree_path / "notes.txt").write_text("change\n", encoding="utf-8")
    result = engine.commit(
        worktree_path,
        "Implement notes",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=3,
    )

    body = run_git(worktree_path, "show", "-s", "--format=%B", result.commit).stdout
    trailer_lines = [line for line in body.splitlines() if line.startswith("NEXUS-")]
    parsed = {}
    for line in trailer_lines:
        key, sep, value = line.partition(": ")
        assert sep == ": "
        assert key and value
        parsed[key] = value

    assert parsed["NEXUS-WorkItem"] == "WI-1"
    assert parsed["NEXUS-Evidence"] == "EVID-1"
    assert parsed["NEXUS-Agent"] == "agent-7"
    assert parsed["NEXUS-Iteration"] == "3"

    engine.remove_worktree(worktree_path)


def test_commit_supports_multiple_evidence_ids(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "commit-evidence-worktree"
    engine.create_worktree(branch, worktree_path)

    (worktree_path / "notes.txt").write_text("change\n", encoding="utf-8")
    result = engine.commit(
        worktree_path,
        "Implement notes",
        work_item="WI-1",
        evidence=("EVID-1", "EVID-2"),
        agent="agent-7",
        iteration=3,
    )

    body = run_git(worktree_path, "show", "-s", "--format=%B", result.commit).stdout
    assert "NEXUS-Evidence: EVID-1,EVID-2" in body
    engine.remove_worktree(worktree_path)


def test_commit_excludes_internal_workspace_metadata(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "commit-filtered-worktree"
    engine.create_worktree(branch, worktree_path)

    (worktree_path / ".nexus-workspace.json").write_text('{"internal": true}\n', encoding="utf-8")
    (worktree_path / "public.txt").write_text("visible\n", encoding="utf-8")
    result = engine.commit(
        worktree_path,
        "Filtered commit",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    )

    names = run_git(
        worktree_path, "show", "--name-only", "--pretty=", result.commit
    ).stdout.splitlines()
    assert "public.txt" in names
    assert ".nexus-workspace.json" not in names
    engine.remove_worktree(worktree_path)


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("work_item", "bad value"),
        ("evidence", ".."),
        ("agent", "-agent"),
        ("iteration", "3:4"),
        ("evidence", "line\nbreak"),
    ],
)
def test_commit_rejects_unsafe_trailer_values(
    tmp_path: Path,
    field: str,
    unsafe: str,
) -> None:
    engine, _repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "unsafe-trailer-worktree"
    engine.create_worktree(branch, worktree_path)

    (worktree_path / "file.txt").write_text("x\n", encoding="utf-8")
    payload = {
        "work_item": "WI-1",
        "evidence": "EVID-1",
        "agent": "agent-7",
        "iteration": "1",
    }
    payload[field] = unsafe

    with pytest.raises(SanitizationError):
        engine.commit(worktree_path, "message", **payload)

    engine.remove_worktree(worktree_path)


@pytest.mark.parametrize("protected", ["main", "integration"])
def test_commit_blocks_direct_commits_to_protected_branches(
    tmp_path: Path,
    protected: str,
) -> None:
    engine, _repo = init_engine(tmp_path)
    worktree_path = tmp_path / f"protected-{protected}"
    engine.create_worktree(protected, worktree_path)
    (worktree_path / "blocked.txt").write_text("blocked\n", encoding="utf-8")

    with pytest.raises(ProtectedBranchError):
        engine.commit(
            worktree_path,
            "should fail",
            work_item="WI-1",
            evidence="EVID-1",
            agent="agent-7",
            iteration=1,
        )

    engine.remove_worktree(worktree_path)


def test_rebase_work_branch_onto_integration(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)

    integration_worktree_1 = tmp_path / "integration-1"
    engine.create_worktree("integration", integration_worktree_1)
    commit_file(integration_worktree_1, "base.txt", "base\n", "base commit")
    engine.remove_worktree(integration_worktree_1)

    work_branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "work"
    engine.create_worktree(work_branch, worktree_path)
    (worktree_path / "work.txt").write_text("work\n", encoding="utf-8")
    work_commit = engine.commit(
        worktree_path,
        "work commit",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    ).commit
    engine.remove_worktree(worktree_path)

    integration_worktree_2 = tmp_path / "integration-2"
    engine.create_worktree("integration", integration_worktree_2)
    commit_file(integration_worktree_2, "integration.txt", "integration\n", "integration commit")
    integration_head = run_git(repo, "rev-parse", "integration").stdout.strip()
    engine.remove_worktree(integration_worktree_2)

    result = engine.rebase_onto_integration(work_branch)
    assert result.old_head == work_commit
    assert result.new_head != work_commit
    assert (
        run_git(repo, "merge-base", work_branch, "integration").stdout.strip() == integration_head
    )

    verify_worktree = tmp_path / "verify-rebase"
    engine.create_worktree(work_branch, verify_worktree)
    assert (verify_worktree / "work.txt").read_text(encoding="utf-8") == "work\n"
    assert (verify_worktree / "integration.txt").read_text(encoding="utf-8") == "integration\n"
    engine.remove_worktree(verify_worktree)


def test_rebase_uses_existing_worktree_when_present(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)

    integration_worktree = tmp_path / "integration-existing"
    engine.create_worktree("integration", integration_worktree)
    commit_file(integration_worktree, "base.txt", "base\n", "base commit")
    engine.remove_worktree(integration_worktree)

    work_branch = engine.create_work_branch("WI", "existing").name
    existing_worktree = tmp_path / "existing-worktree"
    engine.create_worktree(work_branch, existing_worktree)
    (existing_worktree / "work.txt").write_text("work\n", encoding="utf-8")
    engine.commit(
        existing_worktree,
        "work commit",
        work_item="WI-existing",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    )

    integration_worktree_2 = tmp_path / "integration-existing-2"
    engine.create_worktree("integration", integration_worktree_2)
    commit_file(integration_worktree_2, "integration.txt", "integration\n", "integration commit")
    engine.remove_worktree(integration_worktree_2)

    engine.rebase_onto_integration(work_branch)
    assert existing_worktree.exists()
    assert (existing_worktree / "integration.txt").read_text(encoding="utf-8") == "integration\n"
    engine.remove_worktree(existing_worktree)


def test_merge_ff_only_into_integration(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)

    work_branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "ff-merge-work"
    engine.create_worktree(work_branch, worktree_path)
    (worktree_path / "ff.txt").write_text("ff\n", encoding="utf-8")
    engine.commit(
        worktree_path,
        "ff commit",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    )
    engine.remove_worktree(worktree_path)

    merge_result = engine.merge_ff_only(work_branch, "integration")
    assert merge_result.target == "integration"
    assert (
        run_git(repo, "rev-parse", "integration").stdout.strip()
        == run_git(
            repo,
            "rev-parse",
            work_branch,
        ).stdout.strip()
    )


def test_merge_ff_only_rejects_non_fast_forward(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)

    work_branch = engine.create_work_branch("WI", "1").name
    worktree = tmp_path / "work-diverged"
    engine.create_worktree(work_branch, worktree)
    commit_file(worktree, "work-only.txt", "w\n", "work advance")
    engine.remove_worktree(worktree)

    integration_worktree = tmp_path / "integration-advance"
    engine.create_worktree("integration", integration_worktree)
    commit_file(integration_worktree, "integration-only.txt", "i\n", "integration advance")
    engine.remove_worktree(integration_worktree)

    with pytest.raises(GitCommandError):
        engine.merge_ff_only(work_branch, "integration")


def test_merge_ff_only_rejects_non_integration_target(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)
    work_branch = engine.create_work_branch("WI", "1").name
    with pytest.raises(GitEngineError, match="only supported into the integration branch"):
        engine.merge_ff_only(work_branch, "main")


def test_revert_commit_creates_inverse_change(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)

    work_branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "revert-work"
    engine.create_worktree(work_branch, worktree_path)
    (worktree_path / "undo.txt").write_text("to be reverted\n", encoding="utf-8")
    commit_sha = engine.commit(
        worktree_path,
        "add file",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    ).commit
    engine.remove_worktree(worktree_path)

    revert_result = engine.revert_commit(work_branch, commit_sha)
    assert revert_result.reverted_commit == commit_sha

    verify_worktree = tmp_path / "verify-revert"
    engine.create_worktree(work_branch, verify_worktree)
    assert not (verify_worktree / "undo.txt").exists()
    engine.remove_worktree(verify_worktree)


def test_head_alias_methods_and_reset_hard(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)
    work_branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "aliases-work"
    engine.create_worktree(work_branch, worktree_path)
    (worktree_path / "state.txt").write_text("v1\n", encoding="utf-8")
    first = engine.commit(
        worktree_path,
        "first",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    ).commit
    (worktree_path / "state.txt").write_text("v2\n", encoding="utf-8")
    second = engine.commit(
        worktree_path,
        "second",
        work_item="WI-1",
        evidence="EVID-2",
        agent="agent-7",
        iteration=2,
    ).commit
    assert first != second
    engine.remove_worktree(worktree_path)

    assert engine.get_head(work_branch) == second
    assert engine.get_branch_head(work_branch) == second
    assert engine.rev_parse(work_branch) == second
    assert engine.get_integration_head() == run_git(repo, "rev-parse", "integration").stdout.strip()

    engine.reset_hard(work_branch, first)
    assert engine.get_head(work_branch) == first


def test_changed_files_reports_added_modified_deleted_entries(tmp_path: Path) -> None:
    engine, _repo = init_engine(tmp_path)

    work_branch = engine.create_work_branch("WI", "1").name
    worktree_path = tmp_path / "diff-work"
    engine.create_worktree(work_branch, worktree_path)

    commit_file(worktree_path, "keep.txt", "v1\n", "add keep")
    commit_file(worktree_path, "drop.txt", "drop\n", "add drop")
    base_commit = run_git(worktree_path, "rev-parse", "HEAD").stdout.strip()

    (worktree_path / "keep.txt").write_text("v2\n", encoding="utf-8")
    (worktree_path / "drop.txt").unlink()
    (worktree_path / "add.txt").write_text("new\n", encoding="utf-8")
    run_git(worktree_path, "add", "--all")
    run_git(worktree_path, "commit", "-m", "update files")
    head_commit = run_git(worktree_path, "rev-parse", "HEAD").stdout.strip()

    entries = engine.changed_files(base_commit, head_commit)
    status_by_path = {entry.path: entry.status for entry in entries}

    assert status_by_path["add.txt"] == "A"
    assert status_by_path["keep.txt"] == "M"
    assert status_by_path["drop.txt"] == "D"

    engine.remove_worktree(worktree_path)


def test_check_scope_supports_exact_prefix_and_glob(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)

    result = engine.check_scope(
        repo,
        changed_paths=["README.md", "src/pkg/mod.py", "tests/unit/test_scope.py"],
        exact_files=("README.md",),
        directory_prefixes=("src/pkg",),
        allowed_globs=("tests/**/*.py",),
    )

    assert result.allowed is True
    assert result.violations == ()


def test_check_scope_parses_allowed_scope_prefix_and_glob(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)
    result = engine.check_scope(
        repo,
        changed_paths=["src/pkg/a.py", "tests/unit/test_a.py"],
        allowed_scope=("src/pkg/", "tests/**/*.py"),
    )
    assert result.allowed is True
    assert result.violations == ()


def test_check_scope_blocks_when_no_scope_rules_are_declared(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)
    result = engine.check_scope(repo, changed_paths=["src/app.py"])
    assert result.allowed is False
    assert result.violations == ("src/app.py",)


def test_check_scope_blocks_added_modified_and_deleted_paths(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)
    branch = engine.create_work_branch("WI", "scope").name
    worktree = tmp_path / "scope-worktree"
    engine.create_worktree(branch, worktree)

    (worktree / "allowed.txt").write_text("allowed\n", encoding="utf-8")
    (worktree / "blocked.txt").write_text("blocked\n", encoding="utf-8")
    run_git(worktree, "add", "--all")
    run_git(worktree, "commit", "-m", "base")
    base = run_git(worktree, "rev-parse", "HEAD").stdout.strip()

    (worktree / "allowed.txt").write_text("changed\n", encoding="utf-8")  # modified in scope
    (worktree / "blocked.txt").unlink()  # deleted out of scope
    (worktree / "added.txt").write_text("added\n", encoding="utf-8")  # added out of scope
    run_git(worktree, "add", "--all")
    run_git(worktree, "commit", "-m", "change")
    head = run_git(worktree, "rev-parse", "HEAD").stdout.strip()

    changed_paths = [entry.path for entry in engine.changed_files(base, head)]
    scope_result = engine.check_scope(
        repo,
        changed_paths=changed_paths,
        exact_files=("allowed.txt",),
    )

    assert scope_result.allowed is False
    assert set(scope_result.violations) == {"added.txt", "blocked.txt"}
    engine.remove_worktree(worktree)


def test_check_scope_rejects_symlink_escape(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)

    outside = tmp_path / "outside"
    outside.mkdir()
    link = repo / "escape"

    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlinks are not available on this platform")

    result = engine.check_scope(
        repo,
        changed_paths=["escape/secret.txt"],
        directory_prefixes=("escape",),
    )

    assert result.allowed is False
    assert result.violations == ("escape/secret.txt",)


def test_dry_run_merge_detects_conflicts_and_leaves_repo_clean(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)

    integration_worktree_1 = tmp_path / "integration-base"
    engine.create_worktree("integration", integration_worktree_1)
    commit_file(integration_worktree_1, "conflict.txt", "base\n", "base conflict")
    engine.remove_worktree(integration_worktree_1)

    source_branch = engine.create_work_branch("WI", "1").name
    source_worktree = tmp_path / "source"
    engine.create_worktree(source_branch, source_worktree)
    (source_worktree / "conflict.txt").write_text("source\n", encoding="utf-8")
    engine.commit(
        source_worktree,
        "source change",
        work_item="WI-1",
        evidence="EVID-1",
        agent="agent-7",
        iteration=1,
    )
    engine.remove_worktree(source_worktree)

    integration_worktree_2 = tmp_path / "integration-change"
    engine.create_worktree("integration", integration_worktree_2)
    (integration_worktree_2 / "conflict.txt").write_text("integration\n", encoding="utf-8")
    run_git(integration_worktree_2, "add", "--all")
    run_git(integration_worktree_2, "commit", "-m", "integration change")
    engine.remove_worktree(integration_worktree_2)

    target_before = run_git(repo, "rev-parse", "integration").stdout.strip()
    result = engine.dry_run_merge(source_branch, "integration")
    target_after = run_git(repo, "rev-parse", "integration").stdout.strip()

    assert result.clean_merge is False
    assert "conflict.txt" in result.conflicts
    assert target_before == target_after
    assert run_git(repo, "status", "--porcelain").stdout == ""
    assert not (repo / ".git" / "MERGE_HEAD").exists()


def test_dry_run_merge_reports_clean_merge(tmp_path: Path) -> None:
    engine, repo = init_engine(tmp_path)
    source_branch = engine.create_work_branch("WI", "clean").name
    source_worktree = tmp_path / "source-clean"
    engine.create_worktree(source_branch, source_worktree)
    (source_worktree / "new.txt").write_text("ok\n", encoding="utf-8")
    engine.commit(
        source_worktree,
        "clean source",
        work_item="WI-clean",
        evidence="EVID-clean",
        agent="agent-7",
        iteration=1,
    )
    engine.remove_worktree(source_worktree)

    result = engine.dry_run_merge(source_branch, "integration")
    assert result.clean_merge is True
    assert result.conflicts == ()
    assert run_git(repo, "status", "--porcelain").stdout == ""
