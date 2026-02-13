"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/scope_checker.py
Last updated: 2026-02-11

Purpose
- Enforce work-item scope boundaries (file/path ownership) before a change is considered merge-eligible.

What should be included in this file
- A ScopeChecker implementing the BaseChecker interface and producing deterministic CheckResult artifacts.
- Logic to compute the *effective changed files* for a candidate patch (git diff) and compare against allowed scope.
- Support for scoped exceptions via an explicit, audited 'scope extension request' mechanism (never silent).
- Integration points with WorkspaceManager / CodeOwnershipMap if used.

Functional requirements
- Must fail if any modified/added/deleted file is outside the work item’s declared scope allowlist.
- Must support glob/path-prefix rules and explicit file lists.
- Must produce a machine-readable evidence artifact listing out-of-scope paths and the scope rule violated.
- Must support an explicit override flow that requires justification + operator approval, recorded in evidence ledger.

Non-functional requirements
- Deterministic: same inputs → same results.
- Fast: should run in < 1s for typical diffs; avoid expensive repo scans.
- Secure: treat symlinks/path traversal defensively when interpreting file lists.

Notes
- This checker is referenced by baseline constraint CON-ARC-0001 in constraints/registry/000_base_constraints.yaml.
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Protocol, cast

from nexus_orchestrator.verification_plane.checkers.base import (
    BaseChecker,
    CheckerContext,
    CheckResult,
    CheckStatus,
    CommandSpec,
    JSONValue,
    Violation,
    normalize_artifact_paths,
    register_builtin_checker,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import (
    capture_tool_version,
    checker_parameters,
    extract_constraint_ids,
    first_constraint_id,
)

ChangeKind = Literal["added", "modified", "deleted"]

_DEFAULT_REPORT_PATH: Final[str] = ".nexus_orchestrator/checker_artifacts/scope_checker_report.json"
_DRIVE_PREFIX_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:")


@dataclass(frozen=True, slots=True)
class ChangedPath:
    """One changed file path with deterministic normalized kind."""

    path: str
    kind: ChangeKind


@dataclass(frozen=True, slots=True)
class ScopeOverrideRecord:
    """Explicit override record used for audited out-of-scope exemptions."""

    approved: bool
    justification: str
    approved_by: str
    ticket: str | None = None

    @property
    def valid(self) -> bool:
        return self.approved and bool(self.justification.strip()) and bool(self.approved_by.strip())


@dataclass(frozen=True, slots=True)
class ScopeRuleSet:
    """Normalized allowlist rules for scope checker matching."""

    explicit_files: tuple[str, ...]
    glob_patterns: tuple[str, ...]
    prefix_rules: tuple[str, ...]

    def matches(self, path: str) -> bool:
        if path in self.explicit_files:
            return True
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in self.glob_patterns):
            return True
        return any(path == prefix or path.startswith(prefix + "/") for prefix in self.prefix_rules)


class ScopeChangeProvider(Protocol):
    """Abstraction for changed-file discovery."""

    async def get_changed_paths(
        self,
        *,
        context: CheckerContext,
        timeout_seconds: float,
    ) -> tuple[ChangedPath, ...]:
        """Return normalized changed paths for scope evaluation."""


class GitScopeChangeProvider:
    """Git-backed change provider using command executor abstraction."""

    async def get_changed_paths(
        self,
        *,
        context: CheckerContext,
        timeout_seconds: float,
    ) -> tuple[ChangedPath, ...]:
        diff_result = await context.require_executor().run(
            CommandSpec(
                argv=(
                    "git",
                    "-c",
                    "core.quotepath=off",
                    "diff",
                    "--name-status",
                    "--find-renames",
                    "--diff-filter=ACDMRTUXB",
                    "--relative",
                    "HEAD",
                ),
                cwd=context.workspace_path,
                timeout_seconds=timeout_seconds,
            )
        )

        if diff_result.timed_out:
            raise TimeoutError(f"git diff timed out after {timeout_seconds:.3f}s")
        if diff_result.error is not None:
            raise RuntimeError(diff_result.error)
        if diff_result.exit_code not in (0, None):
            message = diff_result.stderr.strip() or diff_result.stdout.strip() or "git diff failed"
            if _is_missing_head_error(message):
                raise RuntimeError(
                    "git HEAD is unavailable; create an initial commit or pass config.changed_files"
                )
            raise RuntimeError(message)

        untracked_result = await context.require_executor().run(
            CommandSpec(
                argv=("git", "ls-files", "--others", "--exclude-standard"),
                cwd=context.workspace_path,
                timeout_seconds=timeout_seconds,
            )
        )
        if untracked_result.timed_out:
            raise TimeoutError(f"git ls-files timed out after {timeout_seconds:.3f}s")
        if untracked_result.error is not None:
            raise RuntimeError(untracked_result.error)
        if untracked_result.exit_code not in (0, None):
            message = (
                untracked_result.stderr.strip()
                or untracked_result.stdout.strip()
                or "git ls-files failed"
            )
            raise RuntimeError(message)

        entries: list[ChangedPath] = []
        for raw_line in diff_result.stdout.splitlines():
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = [segment.strip() for segment in line.split("\t") if segment.strip()]
            if len(parts) < 2:
                continue

            status = parts[0]
            kind = _kind_from_name_status(status)
            if kind is None:
                continue
            if status.startswith("R") and len(parts) >= 3:
                old_path = parts[1]
                new_path = parts[2]
                if old_path:
                    entries.append(ChangedPath(path=old_path, kind="deleted"))
                if new_path:
                    entries.append(ChangedPath(path=new_path, kind="added"))
                continue

            path = parts[-1]
            entries.append(ChangedPath(path=path, kind=kind))

        for raw_line in untracked_result.stdout.splitlines():
            path = raw_line.strip()
            if path:
                entries.append(ChangedPath(path=path, kind="added"))

        deduped = {
            (item.path.replace("\\", "/"), item.kind): ChangedPath(
                path=item.path.replace("\\", "/"),
                kind=item.kind,
            )
            for item in entries
        }
        return tuple(sorted(deduped.values(), key=lambda item: (item.path, item.kind)))


@register_builtin_checker("scope_checker")
class ScopeChecker(BaseChecker):
    """Enforce declared scope boundaries with explicit override auditing."""

    checker_id = "scope_checker"
    stage = "scope"
    covered_constraint_ids = ("CON-ARC-0001",)

    def __init__(self, *, change_provider: ScopeChangeProvider | None = None) -> None:
        self._change_provider = (
            change_provider if change_provider is not None else GitScopeChangeProvider()
        )

    async def check(self, context: CheckerContext) -> CheckResult:
        started = time.monotonic()
        params = checker_parameters(context, self.checker_id)
        covered_constraint_ids = extract_constraint_ids(
            context,
            params,
            self.covered_constraint_ids,
        )
        primary_constraint = first_constraint_id(covered_constraint_ids)
        timeout_seconds = _to_positive_float(params.get("timeout_seconds"), default=1.0)

        try:
            changed_paths = await _load_changed_paths(
                provider=self._change_provider,
                context=context,
                params=params,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)
            return CheckResult(
                status=CheckStatus.TIMEOUT,
                violations=(
                    Violation(
                        constraint_id=primary_constraint,
                        code="scope.timeout",
                        message=f"scope checker timed out after {timeout_seconds:.3f}s",
                    ),
                ),
                covered_constraint_ids=covered_constraint_ids,
                tool_versions=await _tool_versions(context, timeout_seconds),
                artifact_paths=(),
                logs_path=None,
                command_lines=(
                    "git -c core.quotepath=off diff --name-status --find-renames "
                    "--diff-filter=ACDMRTUXB --relative HEAD",
                    "git ls-files --others --exclude-standard",
                ),
                duration_ms=duration_ms,
                metadata={"timeout_seconds": timeout_seconds},
                checker_id=self.checker_id,
                stage=self.stage,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)
            return CheckResult(
                status=CheckStatus.ERROR,
                violations=(
                    Violation(
                        constraint_id=primary_constraint,
                        code="scope.change_provider_error",
                        message=str(exc),
                    ),
                ),
                covered_constraint_ids=covered_constraint_ids,
                tool_versions=await _tool_versions(context, timeout_seconds),
                artifact_paths=(),
                logs_path=None,
                command_lines=(
                    "git -c core.quotepath=off diff --name-status --find-renames "
                    "--diff-filter=ACDMRTUXB --relative HEAD",
                    "git ls-files --others --exclude-standard",
                ),
                duration_ms=duration_ms,
                metadata={"timeout_seconds": timeout_seconds},
                checker_id=self.checker_id,
                stage=self.stage,
            )

        rule_set, rule_violations = _build_scope_rule_set(params, primary_constraint)
        override_record = _parse_override(params)

        workspace_root = Path(context.workspace_path).resolve()
        unsafe_violations: list[Violation] = []
        out_of_scope: list[dict[str, str]] = []
        normalized_changes: list[dict[str, str]] = []

        for item in changed_paths:
            normalized_path, unsafe_reason = _normalize_repo_relative_path(
                raw_path=item.path,
                workspace_root=workspace_root,
            )
            if normalized_path is None:
                unsafe_violations.append(
                    Violation(
                        constraint_id=primary_constraint,
                        code="scope.path_unsafe",
                        message=f"unsafe changed path rejected: {unsafe_reason or item.path!r}",
                    )
                )
                continue

            normalized_changes.append({"path": normalized_path, "kind": item.kind})
            if not rule_set.matches(normalized_path):
                out_of_scope.append(
                    {
                        "path": normalized_path,
                        "kind": item.kind,
                        "reason": "outside_scope_allowlist",
                    }
                )

        violations: list[Violation] = []
        violations.extend(rule_violations)
        violations.extend(unsafe_violations)

        override_applied = (
            bool(out_of_scope) and override_record is not None and override_record.valid
        )

        for entry in out_of_scope:
            violations.append(
                Violation(
                    constraint_id=primary_constraint,
                    code="scope.out_of_bounds",
                    message=f"{entry['kind']} path violates scope allowlist",
                    path=entry["path"],
                    severity="warning" if override_applied else "error",
                )
            )

        if bool(out_of_scope) and not override_applied:
            violations.append(
                Violation(
                    constraint_id=primary_constraint,
                    code="scope.override_required",
                    message=(
                        "out-of-scope changes require explicit override with "
                        "approved=true, justification, and approved_by"
                    ),
                )
            )

        if override_applied:
            violations.append(
                Violation(
                    constraint_id=primary_constraint,
                    code="scope.override_applied",
                    message="approved scope override used",
                    severity="warning",
                )
            )

        report_payload: dict[str, object] = {
            "checker_id": self.checker_id,
            "changed_files": sorted(
                normalized_changes, key=lambda item: (item["path"], item["kind"])
            ),
            "out_of_scope": sorted(out_of_scope, key=lambda item: (item["path"], item["kind"])),
            "rules": {
                "explicit_files": list(rule_set.explicit_files),
                "glob_patterns": list(rule_set.glob_patterns),
                "prefix_rules": list(rule_set.prefix_rules),
            },
            "override": _override_json(override_record),
        }
        report_path, report_error = _write_report(
            workspace_root=workspace_root,
            report_path_raw=params.get("report_path"),
            payload=report_payload,
        )
        if report_error is not None:
            violations.append(
                Violation(
                    constraint_id=primary_constraint,
                    code="scope.report_write_failed",
                    message=report_error,
                )
            )

        has_blocking = any(item.severity.lower() == "error" for item in violations)
        if has_blocking:
            status = CheckStatus.FAIL
        elif override_applied:
            status = CheckStatus.WARN
        else:
            status = CheckStatus.PASS
        duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)
        uses_git_commands = params.get("changed_files") is None
        command_lines = (
            (
                "git -c core.quotepath=off diff --name-status --find-renames "
                "--diff-filter=ACDMRTUXB --relative HEAD",
                "git ls-files --others --exclude-standard",
            )
            if uses_git_commands
            else ()
        )

        return CheckResult(
            status=status,
            violations=tuple(violations),
            covered_constraint_ids=covered_constraint_ids,
            tool_versions=await _tool_versions(context, timeout_seconds),
            artifact_paths=normalize_artifact_paths((report_path,)) if report_path else (),
            logs_path=None,
            command_lines=command_lines,
            duration_ms=duration_ms,
            metadata=cast(
                "dict[str, JSONValue]",
                {
                    "changed_files_count": len(normalized_changes),
                    "out_of_scope_count": len(out_of_scope),
                    "override_used": override_applied,
                    "override": _override_json(override_record),
                },
            ),
            checker_id=self.checker_id,
            stage=self.stage,
        )


async def _load_changed_paths(
    *,
    provider: ScopeChangeProvider,
    context: CheckerContext,
    params: Mapping[str, object],
    timeout_seconds: float,
) -> tuple[ChangedPath, ...]:
    supplied = params.get("changed_files")
    if supplied is None:
        return await provider.get_changed_paths(context=context, timeout_seconds=timeout_seconds)

    parsed: list[ChangedPath] = []
    if isinstance(supplied, str) and supplied.strip():
        parsed.append(ChangedPath(path=supplied.strip(), kind="modified"))
    elif isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes, bytearray)):
        for item in supplied:
            if isinstance(item, str) and item.strip():
                parsed.append(ChangedPath(path=item.strip(), kind="modified"))
                continue
            if isinstance(item, Mapping):
                raw_path = item.get("path")
                raw_kind = item.get("kind", "modified")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                kind = _coerce_change_kind(raw_kind)
                if kind is None:
                    continue
                parsed.append(ChangedPath(path=raw_path.strip(), kind=kind))

    deduped = {(item.path.replace("\\", "/"), item.kind): item for item in parsed}
    return tuple(
        sorted(deduped.values(), key=lambda item: (item.path.replace("\\", "/"), item.kind))
    )


def _build_scope_rule_set(
    params: Mapping[str, object],
    constraint_id: str,
) -> tuple[ScopeRuleSet, tuple[Violation, ...]]:
    explicit_raw = _collect_string_values(
        params,
        keys=("explicit_files", "allowed_files", "files", "scope_files", "file_allowlist"),
    )
    glob_raw = _collect_string_values(
        params,
        keys=("glob_patterns", "allowed_globs", "globs", "scope_globs"),
    )
    prefix_raw = _collect_string_values(
        params,
        keys=("prefix_rules", "allowed_prefixes", "prefixes", "scope_prefixes"),
    )

    violations: list[Violation] = []

    explicit: list[str] = []
    for item in explicit_raw:
        normalized, reason = _normalize_rule_path(item)
        if normalized is None:
            violations.append(
                Violation(
                    constraint_id=constraint_id,
                    code="scope.rule_invalid",
                    message=f"invalid explicit file rule {item!r}: {reason}",
                )
            )
            continue
        explicit.append(normalized)

    globs: list[str] = []
    for item in glob_raw:
        normalized = item.replace("\\", "/").strip()
        if not normalized:
            continue
        pure = PurePosixPath(normalized)
        if pure.is_absolute() or ".." in pure.parts:
            violations.append(
                Violation(
                    constraint_id=constraint_id,
                    code="scope.rule_invalid",
                    message=f"invalid glob rule {item!r}: must be relative and traversal-safe",
                )
            )
            continue
        globs.append(normalized)

    prefixes: list[str] = []
    for item in prefix_raw:
        normalized, reason = _normalize_rule_path(item)
        if normalized is None:
            violations.append(
                Violation(
                    constraint_id=constraint_id,
                    code="scope.rule_invalid",
                    message=f"invalid prefix rule {item!r}: {reason}",
                )
            )
            continue
        prefixes.append(normalized)

    if not explicit and not globs and not prefixes:
        violations.append(
            Violation(
                constraint_id=constraint_id,
                code="scope.rules_missing",
                message="scope rules are required (explicit_files/glob_patterns/prefix_rules)",
            )
        )

    rules = ScopeRuleSet(
        explicit_files=tuple(sorted(set(explicit))),
        glob_patterns=tuple(sorted(set(globs))),
        prefix_rules=tuple(sorted(set(prefixes))),
    )
    return rules, tuple(violations)


def _normalize_repo_relative_path(
    *,
    raw_path: str,
    workspace_root: Path,
) -> tuple[str | None, str | None]:
    candidate = raw_path.replace("\\", "/").strip()
    if not candidate:
        return None, "empty path"
    if "\x00" in candidate:
        return None, "contains NUL"
    if candidate.startswith("/"):
        return None, "absolute path"
    if _DRIVE_PREFIX_RE.match(candidate) is not None:
        return None, "drive-prefixed path"

    pure = PurePosixPath(candidate)
    cleaned_parts: list[str] = []
    for part in pure.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            return None, "path traversal segment"
        cleaned_parts.append(part)

    if not cleaned_parts:
        return None, "empty normalized path"

    normalized = "/".join(cleaned_parts)
    root = workspace_root.resolve()
    resolved = (root / normalized).resolve(strict=False)

    try:
        resolved.relative_to(root)
    except ValueError:
        return None, "path resolves outside workspace"

    cursor = root
    for part in cleaned_parts:
        cursor = cursor / part
        try:
            if cursor.exists() and cursor.is_symlink():
                return None, f"symlink segment: {cursor.relative_to(root)}"
        except OSError:
            return None, "path stat failure"

    return normalized, None


def _parse_override(params: Mapping[str, object]) -> ScopeOverrideRecord | None:
    raw = params.get("override")
    if raw is None:
        raw = params.get("scope_override")
    if not isinstance(raw, Mapping):
        return None

    approved = bool(raw.get("approved")) if isinstance(raw.get("approved"), bool) else False
    justification = raw.get("justification")
    approved_by = raw.get("approved_by")
    ticket = raw.get("ticket")

    return ScopeOverrideRecord(
        approved=approved,
        justification=justification.strip() if isinstance(justification, str) else "",
        approved_by=approved_by.strip() if isinstance(approved_by, str) else "",
        ticket=ticket.strip() if isinstance(ticket, str) and ticket.strip() else None,
    )


def _override_json(record: ScopeOverrideRecord | None) -> dict[str, str | bool] | None:
    if record is None:
        return None
    payload: dict[str, str | bool] = {
        "approved": record.approved,
        "justification": record.justification,
        "approved_by": record.approved_by,
    }
    if record.ticket is not None:
        payload["ticket"] = record.ticket
    return payload


def _kind_from_name_status(status: str) -> ChangeKind | None:
    normalized = status.strip().upper()
    if not normalized:
        return None
    if normalized.startswith("D"):
        return "deleted"
    if normalized.startswith("A"):
        return "added"
    if normalized.startswith(("M", "R", "C", "T", "U")):
        return "modified"
    return None


def _is_missing_head_error(message: str) -> bool:
    normalized = message.strip().lower()
    if "head" not in normalized:
        return False
    indicators = (
        "unknown revision or path not in the working tree",
        "ambiguous argument",
        "bad revision",
        "bad object",
        "unknown revision",
    )
    return any(indicator in normalized for indicator in indicators)


def _coerce_change_kind(raw: object) -> ChangeKind | None:
    if not isinstance(raw, str):
        return None
    normalized = raw.strip().lower()
    if normalized in {"added", "modified", "deleted"}:
        return cast("ChangeKind", normalized)
    return None


def _collect_string_values(params: Mapping[str, object], *, keys: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = params.get(key)
        if isinstance(raw, str) and raw.strip():
            values.append(raw.strip())
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            values.extend(item.strip() for item in raw if isinstance(item, str) and item.strip())
    return tuple(values)


def _normalize_rule_path(raw: str) -> tuple[str | None, str | None]:
    candidate = raw.replace("\\", "/").strip()
    if not candidate:
        return None, "empty value"
    pure = PurePosixPath(candidate)
    if pure.is_absolute():
        return None, "absolute path"
    if _DRIVE_PREFIX_RE.match(candidate) is not None:
        return None, "drive-prefixed path"
    if ".." in pure.parts:
        return None, "path traversal segment"

    cleaned = [part for part in pure.parts if part not in {"", "."}]
    if not cleaned:
        return None, "empty normalized path"
    return "/".join(cleaned), None


def _to_positive_float(raw: object, *, default: float) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        parsed = float(raw)
        if parsed > 0:
            return parsed
    return default


async def _tool_versions(context: CheckerContext, timeout_seconds: float) -> dict[str, str]:
    version = await capture_tool_version(
        context=context,
        command=("git", "status"),
        timeout_seconds=timeout_seconds,
        version_command=("git", "--version"),
    )
    return {"git": version, "scope_checker": "builtin-1"}


def _write_report(
    *,
    workspace_root: Path,
    report_path_raw: object,
    payload: Mapping[str, object],
) -> tuple[str | None, str | None]:
    report_rel = _coerce_report_path(report_path_raw)
    destination = workspace_root / report_rel
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        return None, f"failed to write scope report {report_rel}: {exc}"

    return report_rel, None


def _coerce_report_path(raw: object) -> str:
    if isinstance(raw, str) and raw.strip():
        candidate = raw.replace("\\", "/").strip()
        normalized, reason = _normalize_rule_path(candidate)
        if normalized is not None and reason is None:
            return normalized
    return _DEFAULT_REPORT_PATH


__all__ = [
    "ChangedPath",
    "GitScopeChangeProvider",
    "ScopeChangeProvider",
    "ScopeChecker",
    "ScopeOverrideRecord",
    "ScopeRuleSet",
]
