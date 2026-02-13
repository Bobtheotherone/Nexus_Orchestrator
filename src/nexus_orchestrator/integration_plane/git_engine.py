"""Deterministic Git integration helpers for orchestration workflows."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

_BRANCH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TRAILER_VALUE_RE = re.compile(r"^[A-Za-z0-9._/@,-]+$")
_FORBIDDEN_SANITIZER_CHARS = (" ", "\t", "\r", "\n", ":")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


class GitEngineError(RuntimeError):
    """Base error for git engine failures."""


class SanitizationError(GitEngineError):
    """Raised when a branch token or trailer value is unsafe."""


class ProtectedBranchError(GitEngineError):
    """Raised when an operation is blocked on a protected branch."""


class GitCommandError(GitEngineError):
    """Raised when a git subprocess command exits non-zero."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        message = f"git command failed ({returncode}): {' '.join(command)}"
        if stderr.strip():
            message = f"{message}: {stderr.strip()}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Normalized subprocess result for deterministic git wrapper behavior."""

    command: tuple[str, ...]
    cwd: str
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class RepoInitResult:
    """Result for repository open/initialization."""

    repo_path: Path
    created: bool
    main_branch: str
    integration_branch: str


@dataclass(frozen=True, slots=True)
class BranchResult:
    """Result for work branch creation."""

    name: str
    base: str


@dataclass(frozen=True, slots=True)
class WorktreeResult:
    """Result for worktree creation."""

    branch: str
    path: Path


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Result for commit operation."""

    branch: str
    commit: str
    trailers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RebaseResult:
    """Result for branch rebase operation."""

    branch: str
    old_head: str
    new_head: str


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result for ff-only merge."""

    source: str
    target: str
    target_head: str


@dataclass(frozen=True, slots=True)
class RevertResult:
    """Result for revert operation."""

    branch: str
    reverted_commit: str
    new_head: str


@dataclass(frozen=True, slots=True)
class ChangedFileEntry:
    """Diff entry with normalized status."""

    status: str
    path: str


@dataclass(frozen=True, slots=True)
class ScopeCheckResult:
    """Result for path scope checks."""

    allowed: bool
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DryRunMergeResult:
    """Result for dry-run merge conflict detection."""

    clean_merge: bool
    conflicts: tuple[str, ...]


class GitEngine:
    """Deterministic wrapper around git CLI for orchestration workflows."""

    def __init__(
        self,
        repo_path: Path | str,
        *,
        main_branch: str = "main",
        integration_branch: str = "integration",
        env_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.main_branch = main_branch
        self.integration_branch = integration_branch
        self._env_overrides = dict(env_overrides or {})

    def init_or_open(self) -> RepoInitResult:
        """Open a repository or initialize it, always ensuring main+integration branches."""
        self.repo_path.mkdir(parents=True, exist_ok=True)
        created = not (self.repo_path / ".git").exists()

        if created:
            self._run_git(["init", "--initial-branch", self.main_branch], cwd=self.repo_path)
        else:
            self._run_git(["rev-parse", "--git-dir"], cwd=self.repo_path)

        self._ensure_local_identity()

        if self._run_git(["rev-parse", "--verify", "HEAD"], check=False).returncode != 0:
            self._run_git(["symbolic-ref", "HEAD", f"refs/heads/{self.main_branch}"])
            self._run_git(["commit", "--allow-empty", "-m", "Initialize repository"])

        if not self._branch_exists(self.main_branch):
            self._run_git(["branch", self.main_branch, "HEAD"])

        if not self._branch_exists(self.integration_branch):
            self._run_git(["branch", self.integration_branch, self.main_branch])

        return RepoInitResult(
            repo_path=self.repo_path,
            created=created,
            main_branch=self.main_branch,
            integration_branch=self.integration_branch,
        )

    def create_work_branch(
        self,
        work_item: str,
        attempt: str,
        *,
        base_branch: str | None = None,
    ) -> BranchResult:
        """Create a work branch as work/<work_item>/<attempt> from integration."""
        base = base_branch if base_branch is not None else self.integration_branch
        self._require_branch(base)

        safe_work_item = self._sanitize_branch_component(work_item, field="work_item")
        safe_attempt = self._sanitize_branch_component(attempt, field="attempt")
        branch_name = f"work/{safe_work_item}/{safe_attempt}"

        if self._branch_exists(branch_name):
            msg = f"Branch already exists: {branch_name}"
            raise GitEngineError(msg)

        self._run_git(["branch", branch_name, base])
        return BranchResult(name=branch_name, base=base)

    def create_worktree(self, branch: str, worktree_path: Path | str) -> WorktreeResult:
        """Create a worktree for branch at the provided path."""
        self._require_branch(branch)
        path = Path(worktree_path)
        self._run_git(["worktree", "add", "--force", str(path), branch])
        return WorktreeResult(branch=branch, path=path)

    def remove_worktree(self, worktree_path: Path | str) -> None:
        """Remove a worktree path and prune stale entries."""
        path = Path(worktree_path)
        self._run_git(["worktree", "remove", "--force", str(path)])
        self._run_git(["worktree", "prune"], check=False)

    def apply_patch(self, worktree_path: Path | str, unified_diff: str) -> None:
        """Apply a unified diff patch in a worktree."""
        if not unified_diff.strip():
            raise GitEngineError("Patch content cannot be empty.")

        patch_text = unified_diff if unified_diff.endswith("\n") else f"{unified_diff}\n"
        self._validate_patch_paths(patch_text)
        self._run_git(
            ["apply", "--index", "--whitespace=nowarn", "-"],
            cwd=Path(worktree_path),
            input_text=patch_text,
        )

    def commit(
        self,
        worktree_path: Path | str,
        message: str,
        *,
        work_item: str,
        evidence: str | Sequence[str],
        agent: str,
        iteration: str | int,
    ) -> CommitResult:
        """Commit staged/untracked work with required NEXUS trailers."""
        worktree = Path(worktree_path)
        branch = self._current_branch(worktree)
        self._assert_not_protected(branch, action="commit")

        title = message.strip()
        if not title:
            raise GitEngineError("Commit message cannot be empty.")

        safe_work_item = self._sanitize_trailer_value(str(work_item), field="work_item")
        safe_evidence = self._normalize_evidence_value(evidence)
        safe_agent = self._sanitize_trailer_value(str(agent), field="agent")
        safe_iteration = self._sanitize_trailer_value(str(iteration), field="iteration")

        trailers = (
            f"NEXUS-WorkItem: {safe_work_item}",
            f"NEXUS-Evidence: {safe_evidence}",
            f"NEXUS-Agent: {safe_agent}",
            f"NEXUS-Iteration: {safe_iteration}",
        )

        self._run_git(["add", "--all"], cwd=worktree)
        self._run_git(
            ["reset", "--quiet", "HEAD", "--", ".nexus-workspace.json"],
            cwd=worktree,
            check=False,
        )
        has_staged_changes = bool(
            self._run_git(["diff", "--cached", "--name-only"], cwd=worktree).stdout.strip()
        )
        if not has_staged_changes:
            raise GitEngineError("No staged changes to commit after filtering internal files.")
        self._run_git(
            ["commit", "--no-gpg-sign", "-m", title, "-m", "\n".join(trailers)], cwd=worktree
        )

        commit_sha = self._run_git(["rev-parse", "HEAD"], cwd=worktree).stdout.strip()
        return CommitResult(branch=branch, commit=commit_sha, trailers=trailers)

    def rebase_onto_integration(self, branch: str) -> RebaseResult:
        """Rebase a work branch onto the integration branch."""
        self._require_branch(branch)
        self._assert_not_protected(branch, action="rebase")
        self._require_branch(self.integration_branch)

        old_head = self._rev_parse(branch)

        with self._temporary_worktree(branch) as temp_worktree:
            try:
                self._run_git(["rebase", self.integration_branch], cwd=temp_worktree)
            except GitCommandError:
                self._run_git(["rebase", "--abort"], cwd=temp_worktree, check=False)
                raise

        new_head = self._rev_parse(branch)
        return RebaseResult(branch=branch, old_head=old_head, new_head=new_head)

    def merge_ff_only(self, source_branch: str, target_branch: str = "integration") -> MergeResult:
        """Fast-forward source branch into integration only."""
        if target_branch != self.integration_branch:
            msg = "ff-only merges are only supported into the integration branch."
            raise GitEngineError(msg)

        self._require_branch(source_branch)
        self._require_branch(target_branch)

        with self._temporary_worktree(target_branch) as temp_worktree:
            self._run_git(["merge", "--ff-only", source_branch], cwd=temp_worktree)

        return MergeResult(
            source=source_branch,
            target=target_branch,
            target_head=self._rev_parse(target_branch),
        )

    def fast_forward_merge(
        self, source_branch: str, target_branch: str = "integration"
    ) -> MergeResult:
        """Compatibility alias used by merge queue adapters."""
        return self.merge_ff_only(source_branch, target_branch)

    def get_head(self, branch: str) -> str:
        """Return the commit SHA for ``branch``."""
        return self._rev_parse(branch)

    def get_branch_head(self, branch: str) -> str:
        """Compatibility alias for branch head lookup."""
        return self.get_head(branch)

    def get_integration_head(self) -> str:
        """Return the integration branch head SHA."""
        return self._rev_parse(self.integration_branch)

    def rev_parse(self, ref: str) -> str:
        """Public ref parser for queue adapters."""
        return self._rev_parse(ref)

    def reset_hard(self, branch: str, commit_sha: str) -> None:
        """Hard-reset ``branch`` to ``commit_sha`` in a temporary worktree."""
        self._require_branch(branch)
        with self._temporary_worktree(branch) as temp_worktree:
            self._run_git(["reset", "--hard", commit_sha], cwd=temp_worktree)

    def revert_commit(self, branch: str, commit_sha: str) -> RevertResult:
        """Revert a commit on the provided branch by creating a revert commit."""
        self._require_branch(branch)

        with self._temporary_worktree(branch) as temp_worktree:
            try:
                self._run_git(["revert", "--no-edit", commit_sha], cwd=temp_worktree)
            except GitCommandError:
                self._run_git(["revert", "--abort"], cwd=temp_worktree, check=False)
                raise

        return RevertResult(
            branch=branch,
            reverted_commit=commit_sha,
            new_head=self._rev_parse(branch),
        )

    def changed_files(self, base_ref: str, head_ref: str) -> tuple[ChangedFileEntry, ...]:
        """Return changed file entries between refs with A/M/D statuses."""
        output = self._run_git(
            ["diff", "--name-status", "--diff-filter=AMD", base_ref, head_ref]
        ).stdout

        entries: list[ChangedFileEntry] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            status_and_path = line.split("\t", 1)
            if len(status_and_path) != 2:
                continue
            status = status_and_path[0][:1]
            if status not in {"A", "M", "D"}:
                continue
            entries.append(ChangedFileEntry(status=status, path=status_and_path[1]))

        return tuple(entries)

    def check_scope(
        self,
        repo_root: Path | str,
        changed_paths: Sequence[str],
        *,
        allowed_scope: Sequence[str] | None = None,
        exact_files: Sequence[str] = (),
        directory_prefixes: Sequence[str] = (),
        allowed_globs: Sequence[str] = (),
    ) -> ScopeCheckResult:
        """Check changed files against exact paths, directory prefixes, and globs."""
        root = Path(repo_root).resolve(strict=True)

        all_exact_files = list(exact_files)
        all_directory_prefixes = list(directory_prefixes)
        all_allowed_globs = list(allowed_globs)

        if allowed_scope is not None:
            for entry in allowed_scope:
                normalized_entry = self._normalize_scope_pattern(entry)
                if any(char in normalized_entry for char in ("*", "?", "[")):
                    all_allowed_globs.append(normalized_entry)
                elif entry.endswith("/"):
                    all_directory_prefixes.append(normalized_entry)
                else:
                    all_exact_files.append(normalized_entry)

        exact = {self._normalize_scope_pattern(path) for path in all_exact_files}
        prefixes = {
            self._normalize_scope_pattern(path).rstrip("/") for path in all_directory_prefixes
        }
        glob_patterns = tuple(self._normalize_scope_pattern(path) for path in all_allowed_globs)
        has_rules = bool(exact or prefixes or glob_patterns)

        violations: list[str] = []
        for changed_path in changed_paths:
            normalized = self._normalize_changed_path(changed_path)
            if normalized in {"", "."}:
                violations.append(changed_path)
                continue

            candidate = Path(normalized)
            resolved = (
                candidate.resolve(strict=False)
                if candidate.is_absolute()
                else (root / candidate).resolve(strict=False)
            )
            if not resolved.is_relative_to(root):
                violations.append(normalized)
                continue

            if not has_rules:
                violations.append(normalized)
                continue

            if normalized in exact:
                continue

            if any(
                normalized == prefix or normalized.startswith(f"{prefix}/") for prefix in prefixes
            ):
                continue

            if any(fnmatch(normalized, pattern) for pattern in glob_patterns):
                continue

            violations.append(normalized)

        return ScopeCheckResult(allowed=not violations, violations=tuple(violations))

    def dry_run_merge(self, source_branch: str, target_branch: str) -> DryRunMergeResult:
        """Detect merge conflicts without changing repository refs or leaving merge state behind."""
        self._require_branch(source_branch)
        self._require_branch(target_branch)

        target_before = self._rev_parse(target_branch)
        conflicts: list[str] = []
        clean_merge = True

        with self._temporary_worktree(target_branch) as temp_worktree:
            merge_result = self._run_git(
                ["merge", "--no-commit", "--no-ff", source_branch],
                cwd=temp_worktree,
                check=False,
            )

            if merge_result.returncode != 0:
                clean_merge = False
                conflict_output = self._run_git(
                    ["diff", "--name-only", "--diff-filter=U"],
                    cwd=temp_worktree,
                    check=False,
                ).stdout
                conflicts = [line.strip() for line in conflict_output.splitlines() if line.strip()]

            self._run_git(["merge", "--abort"], cwd=temp_worktree, check=False)
            self._run_git(["reset", "--hard", "HEAD"], cwd=temp_worktree, check=False)

        target_after = self._rev_parse(target_branch)
        if target_before != target_after:
            raise GitEngineError("dry_run_merge modified the target branch unexpectedly.")

        return DryRunMergeResult(clean_merge=clean_merge, conflicts=tuple(conflicts))

    def _ensure_local_identity(self) -> None:
        if self._run_git(["config", "--local", "--get", "user.name"], check=False).returncode != 0:
            self._run_git(["config", "--local", "user.name", "nexus-orchestrator"])
        if self._run_git(["config", "--local", "--get", "user.email"], check=False).returncode != 0:
            self._run_git(["config", "--local", "user.email", "nexus@example.invalid"])

    def _require_branch(self, branch: str) -> None:
        if not self._branch_exists(branch):
            raise GitEngineError(f"Branch does not exist: {branch}")

    def _branch_exists(self, branch: str) -> bool:
        ref = f"refs/heads/{branch}"
        return self._run_git(["show-ref", "--verify", "--quiet", ref], check=False).returncode == 0

    def _rev_parse(self, ref: str) -> str:
        return self._run_git(["rev-parse", ref]).stdout.strip()

    def _assert_not_protected(self, branch: str, *, action: str) -> None:
        if branch in {self.main_branch, self.integration_branch}:
            raise ProtectedBranchError(f"Cannot {action} directly on protected branch '{branch}'.")

    def _current_branch(self, worktree: Path) -> str:
        branch = self._run_git(["branch", "--show-current"], cwd=worktree).stdout.strip()
        if not branch:
            raise GitEngineError("Detached HEAD is not supported for this operation.")
        return branch

    def _sanitize_branch_component(self, value: str, *, field: str) -> str:
        return self._sanitize_value(value, field=field, pattern=_BRANCH_COMPONENT_RE)

    def _sanitize_trailer_value(self, value: str, *, field: str) -> str:
        return self._sanitize_value(value, field=field, pattern=_TRAILER_VALUE_RE)

    def _sanitize_value(self, value: str, *, field: str, pattern: re.Pattern[str]) -> str:
        if value == "":
            raise SanitizationError(f"{field} cannot be empty.")

        if any(ch in value for ch in _FORBIDDEN_SANITIZER_CHARS):
            raise SanitizationError(
                f"{field} contains forbidden characters (spaces, colon, or control chars)."
            )

        if ".." in value:
            raise SanitizationError(f"{field} cannot contain '..'.")

        if value.startswith("-"):
            raise SanitizationError(f"{field} cannot start with '-'.")

        if not pattern.fullmatch(value):
            raise SanitizationError(f"{field} contains unsupported characters.")

        return value

    def _normalize_evidence_value(self, value: str | Sequence[str]) -> str:
        if isinstance(value, str):
            return self._sanitize_trailer_value(value, field="evidence")

        if not value:
            raise SanitizationError("evidence cannot be empty.")

        normalized = [self._sanitize_trailer_value(item, field="evidence") for item in value]
        if any("," in item for item in normalized):
            raise SanitizationError("evidence list values must not contain commas.")
        return ",".join(normalized)

    def _normalize_scope_pattern(self, pattern: str) -> str:
        normalized = PurePosixPath(pattern).as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.rstrip("/")

        if normalized in {"", "."}:
            raise GitEngineError("Scope pattern cannot be empty.")

        if PurePosixPath(normalized).is_absolute():
            raise GitEngineError("Scope patterns must be repository-relative paths.")

        return normalized

    def _normalize_changed_path(self, changed_path: str) -> str:
        normalized = PurePosixPath(changed_path).as_posix()
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized

    def _validate_patch_paths(self, unified_diff: str) -> None:
        for raw_path in self._extract_patch_paths(unified_diff):
            normalized = self._normalize_patch_path(raw_path)
            self._assert_patch_path_allowed(raw_path=raw_path, normalized_path=normalized)

    def _extract_patch_paths(self, unified_diff: str) -> tuple[str, ...]:
        paths: list[str] = []
        for line in unified_diff.splitlines():
            candidate: str | None = None
            if line.startswith("--- ") or line.startswith("+++ "):
                candidate = line[4:].split("\t", 1)[0].strip()
            elif line.startswith("rename from "):
                candidate = line[len("rename from ") :].strip()
            elif line.startswith("rename to "):
                candidate = line[len("rename to ") :].strip()
            elif line.startswith("copy from "):
                candidate = line[len("copy from ") :].strip()
            elif line.startswith("copy to "):
                candidate = line[len("copy to ") :].strip()

            if candidate is None or candidate == "/dev/null":
                continue

            paths.append(candidate)

        return tuple(paths)

    def _normalize_patch_path(self, patch_path: str) -> str:
        normalized = patch_path.strip()
        if normalized.startswith('"') and normalized.endswith('"') and len(normalized) >= 2:
            normalized = normalized[1:-1]

        while normalized.startswith("./"):
            normalized = normalized[2:]

        if normalized.startswith("a/") or normalized.startswith("b/"):
            normalized = normalized[2:]

        return PurePosixPath(normalized).as_posix()

    def _assert_patch_path_allowed(self, *, raw_path: str, normalized_path: str) -> None:
        if normalized_path in {"", "."}:
            raise GitEngineError(f"Patch path '{raw_path}' is not allowed: path cannot be empty.")

        if PurePosixPath(normalized_path).is_absolute() or _WINDOWS_ABSOLUTE_PATH_RE.match(
            normalized_path
        ):
            raise GitEngineError(
                f"Patch path '{raw_path}' is not allowed: absolute paths are forbidden."
            )

        if ".." in PurePosixPath(normalized_path).parts:
            raise GitEngineError(
                f"Patch path '{raw_path}' is not allowed: path traversal ('..') is forbidden."
            )

        if ".git" in PurePosixPath(normalized_path).parts:
            raise GitEngineError(
                f"Patch path '{raw_path}' is not allowed: .git targets are forbidden."
            )

    @contextmanager
    def _temporary_worktree(self, branch: str) -> Iterator[Path]:
        existing = self._existing_worktree_for_branch(branch)
        if existing is not None:
            yield existing
            return

        temp_path = Path(tempfile.mkdtemp(prefix="nexus-git-engine-"))
        added = False
        try:
            self._run_git(["worktree", "add", "--force", str(temp_path), branch])
            added = True
            yield temp_path
        finally:
            if added:
                self._run_git(["worktree", "remove", "--force", str(temp_path)], check=False)
                self._run_git(["worktree", "prune"], check=False)
            shutil.rmtree(temp_path, ignore_errors=True)

    def _existing_worktree_for_branch(self, branch: str) -> Path | None:
        output = self._run_git(["worktree", "list", "--porcelain"], check=False).stdout
        active_path: Path | None = None
        branch_ref = f"refs/heads/{branch}"

        current_worktree: Path | None = None
        current_ref: str | None = None
        for line in output.splitlines():
            if not line:
                if current_worktree is not None and current_ref == branch_ref:
                    active_path = current_worktree
                    break
                current_worktree = None
                current_ref = None
                continue

            key, _, value = line.partition(" ")
            value = value.strip()
            if key == "worktree":
                current_worktree = Path(value).resolve(strict=False)
            elif key == "branch":
                current_ref = value

        if active_path is None and current_worktree is not None and current_ref == branch_ref:
            active_path = current_worktree
        return active_path

    def _run_git(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
        input_text: str | None = None,
    ) -> CommandResult:
        command = ("git", *args)
        run_cwd = (cwd if cwd is not None else self.repo_path).resolve()
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
        env.update(self._env_overrides)

        completed = subprocess.run(
            command,
            cwd=run_cwd,
            env=env,
            text=True,
            capture_output=True,
            input=input_text,
            check=False,
        )

        result = CommandResult(
            command=command,
            cwd=run_cwd.as_posix(),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

        if check and result.returncode != 0:
            raise GitCommandError(
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        return result
