"""
Build Checker — verification stage.

Functional requirements:
- Implements BaseChecker interface.
- Runs appropriate tool in sandbox and captures output.
- Produces CheckResult with evidence artifacts.
- Supports configuration via constraint parameters.

Non-functional requirements:
- Must be deterministic. Non-deterministic results flagged as flaky.
- Must record exact tool version used.
- Must respect timeout limits.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Final

from nexus_orchestrator.verification_plane.checkers.base import (
    BaseChecker,
    CheckerContext,
    CheckResult,
    CheckStatus,
    CommandResult,
    CommandSpec,
    JSONValue,
    Violation,
    normalize_artifact_paths,
    register_builtin_checker,
)

PathLikeSeq = Sequence[str]

_PATH_LINE_COL_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<path>[^:\s][^:]*?):(?P<line>\d+):(?P<column>\d+):\s*(?P<message>.+)$"
)
_PATH_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<path>[^:\s][^:]*?):(?P<line>\d+):\s*(?P<message>.+)$"
)
_COMPILE_ERROR_RE: Final[re.Pattern[str]] = re.compile(
    r"^\*\*\* Error compiling '(?P<path>[^']+)':\s*(?P<message>.+)$"
)


def checker_parameters(context: CheckerContext, checker_id: str) -> dict[str, object]:
    """Resolve checker parameters deterministically from context config."""

    params: dict[str, object] = {}

    for key, value in sorted(context.config.items()):
        if key not in {"checkers", "parameters"}:
            params[key] = value

    checkers_obj = context.config.get("checkers")
    if isinstance(checkers_obj, Mapping):
        checker_obj = checkers_obj.get(checker_id)
        if isinstance(checker_obj, Mapping):
            for key, value in sorted(checker_obj.items()):
                if isinstance(key, str):
                    params[key] = value

    generic_params_obj = context.config.get("parameters")
    if isinstance(generic_params_obj, Mapping):
        for key, value in sorted(generic_params_obj.items()):
            if isinstance(key, str):
                params[key] = value

    return params


def extract_constraint_ids(
    context: CheckerContext,
    params: Mapping[str, object],
    defaults: Sequence[str],
) -> tuple[str, ...]:
    """Collect covered constraints from context/defaults/parameters."""

    collected: set[str] = {item.strip() for item in defaults if item.strip()}
    collected.update(context.constraint_ids)

    constraint_id = params.get("constraint_id")
    if isinstance(constraint_id, str) and constraint_id.strip():
        collected.add(constraint_id.strip())

    constraint_ids = params.get("constraint_ids")
    if isinstance(constraint_ids, str) and constraint_ids.strip():
        collected.add(constraint_ids.strip())
    elif isinstance(constraint_ids, Sequence) and not isinstance(
        constraint_ids, (str, bytes, bytearray)
    ):
        for item in constraint_ids:
            if isinstance(item, str) and item.strip():
                collected.add(item.strip())

    return tuple(sorted(collected))


def first_constraint_id(constraint_ids: Sequence[str]) -> str:
    """Return deterministic fallback constraint id for violations."""

    return constraint_ids[0] if constraint_ids else "UNMAPPED"


def _to_command(value: object, *, default: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = tuple(part for part in shlex.split(value) if part.strip())
        if parts:
            return parts
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
        if parts:
            return parts

    fallback = tuple(item.strip() for item in default if item.strip())
    if not fallback:
        raise ValueError("command is required")
    return fallback


def _to_optional_command(value: object, *, default: Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        if default is None:
            return None
        return _to_command(default, default=())
    resolved = _to_command(value, default=())
    return resolved if resolved else None


def _to_positive_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        parsed = float(value)
        if parsed > 0:
            return parsed
    return default


def _to_positive_int(value: object, *, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _to_exit_codes(value: object, *, default: Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed = tuple(sorted({int(item) for item in value if isinstance(item, int)}))
        if parsed:
            return parsed
    fallback = tuple(sorted({int(item) for item in default}))
    return fallback if fallback else (0,)


def _normalize_relative_path(value: str) -> str | None:
    candidate = value.replace("\\", "/").strip()
    if not candidate:
        return None

    pure = PurePosixPath(candidate)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        return None

    cleaned = [part for part in pure.parts if part not in {"", "."}]
    if not cleaned:
        return None
    return "/".join(cleaned)


def _artifact_paths_from_params(params: Mapping[str, object]) -> tuple[str, ...]:
    raw = params.get("artifact_paths")
    candidates: list[str] = []
    if isinstance(raw, str) and raw.strip():
        candidates.append(raw)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        candidates.extend(item for item in raw if isinstance(item, str) and item.strip())

    normalized = [
        item for item in (_normalize_relative_path(value) for value in candidates) if item
    ]
    return normalize_artifact_paths(normalized)


def _logs_path_from_params(params: Mapping[str, object]) -> str | None:
    raw = params.get("logs_path")
    if not isinstance(raw, str):
        return None
    return _normalize_relative_path(raw)


def _parse_common_violations(
    *,
    stdout: str,
    stderr: str,
    constraint_id: str,
    fallback_code: str,
) -> tuple[Violation, ...]:
    lines = [line.strip() for line in f"{stdout}\n{stderr}".splitlines() if line.strip()]
    findings: dict[tuple[str, str, int | None, int | None, str], Violation] = {}

    for line in lines:
        path_line_col_match = _PATH_LINE_COL_RE.match(line)
        if path_line_col_match is not None:
            path = _normalize_relative_path(path_line_col_match.group("path"))
            violation = Violation(
                constraint_id=constraint_id,
                code=fallback_code,
                message=path_line_col_match.group("message").strip(),
                path=path,
                line=int(path_line_col_match.group("line")),
                column=int(path_line_col_match.group("column")),
            )
            findings[
                (
                    violation.code,
                    violation.path or "",
                    violation.line,
                    violation.column,
                    violation.message,
                )
            ] = violation
            continue

        path_line_match = _PATH_LINE_RE.match(line)
        if path_line_match is not None:
            path = _normalize_relative_path(path_line_match.group("path"))
            violation = Violation(
                constraint_id=constraint_id,
                code=fallback_code,
                message=path_line_match.group("message").strip(),
                path=path,
                line=int(path_line_match.group("line")),
            )
            findings[
                (
                    violation.code,
                    violation.path or "",
                    violation.line,
                    violation.column,
                    violation.message,
                )
            ] = violation
            continue

        if line.startswith(("FAILED", "FAIL", "ERROR", "E   ")):
            violation = Violation(
                constraint_id=constraint_id,
                code=fallback_code,
                message=line,
            )
            findings[
                (
                    violation.code,
                    violation.path or "",
                    violation.line,
                    violation.column,
                    violation.message,
                )
            ] = violation

    if findings:
        return tuple(sorted(findings.values(), key=lambda item: item.sort_key()))

    first_line = lines[0] if lines else "command failed with non-zero exit code"
    return (
        Violation(
            constraint_id=constraint_id,
            code=fallback_code,
            message=first_line,
        ),
    )


async def capture_tool_version(
    *,
    context: CheckerContext,
    command: Sequence[str],
    timeout_seconds: float,
    version_command: Sequence[str] | None,
) -> str:
    """Capture deterministic tool version string from executor."""

    resolved_version_command = (
        tuple(version_command) if version_command is not None else (command[0], "--version")
    )

    result = await context.require_executor().run(
        CommandSpec(
            argv=resolved_version_command,
            cwd=context.workspace_path,
            timeout_seconds=min(timeout_seconds, 10.0),
        )
    )

    if result.timed_out or result.error is not None:
        return "unavailable"

    for line in f"{result.stdout}\n{result.stderr}".splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "unavailable"


class CommandChecker(BaseChecker):
    """Shared async command-based checker implementation."""

    checker_id: str = "command_checker"
    stage: str = "build"
    covered_constraint_ids: tuple[str, ...] = ()

    tool_name: str = "tool"
    default_command: tuple[str, ...] = ()
    default_version_command: tuple[str, ...] | None = None
    default_timeout_seconds: float = 60.0
    default_probe_runs: int = 1
    default_allowed_exit_codes: tuple[int, ...] = (0,)

    def build_command(self, params: Mapping[str, object]) -> tuple[str, ...]:
        return _to_command(params.get("command"), default=self.default_command)

    def allowed_exit_codes(self, params: Mapping[str, object]) -> tuple[int, ...]:
        return _to_exit_codes(
            params.get("allowed_exit_codes"),
            default=self.default_allowed_exit_codes,
        )

    def parse_failures(
        self,
        *,
        result: CommandResult,
        constraint_id: str,
    ) -> tuple[Violation, ...]:
        return _parse_common_violations(
            stdout=result.stdout,
            stderr=result.stderr,
            constraint_id=constraint_id,
            fallback_code=f"{self.checker_id}.failure",
        )

    async def check(self, context: CheckerContext) -> CheckResult:
        params = checker_parameters(context, self.checker_id)
        covered_constraint_ids = extract_constraint_ids(
            context,
            params,
            self.covered_constraint_ids,
        )
        primary_constraint_id = first_constraint_id(covered_constraint_ids)

        command = self.build_command(params)
        timeout_seconds = _to_positive_float(
            params.get("timeout_seconds"),
            default=self.default_timeout_seconds,
        )
        probe_runs = _to_positive_int(
            params.get("probe_runs"),
            default=self.default_probe_runs,
        )
        allowed_exit_codes = self.allowed_exit_codes(params)

        command_spec = CommandSpec(
            argv=command,
            cwd=context.workspace_path,
            timeout_seconds=timeout_seconds,
            allowed_exit_codes=allowed_exit_codes,
        )

        probe_results: list[CommandResult] = []
        for _ in range(probe_runs):
            probe_results.append(await context.require_executor().run(command_spec))

        result = probe_results[0]

        probe_fingerprints = {
            (probe.exit_code, probe.timed_out, probe.error or "") for probe in probe_results
        }
        tool_version = await capture_tool_version(
            context=context,
            command=command,
            timeout_seconds=timeout_seconds,
            version_command=_to_optional_command(
                params.get("version_command"),
                default=self.default_version_command,
            ),
        )
        tool_versions = {self.tool_name: tool_version}

        status: CheckStatus
        violations: tuple[Violation, ...]

        if len(probe_fingerprints) > 1:
            status = CheckStatus.FAIL
            violations = (
                Violation(
                    constraint_id=primary_constraint_id,
                    code=f"{self.checker_id}.flaky",
                    message="checker command produced non-deterministic outcomes across probe runs",
                ),
            )
        elif result.timed_out:
            status = CheckStatus.TIMEOUT
            violations = (
                Violation(
                    constraint_id=primary_constraint_id,
                    code=f"{self.checker_id}.timeout",
                    message=(
                        f"checker timed out after {timeout_seconds:.3f}s: {' '.join(command)}"
                    ),
                ),
            )
        elif result.error is not None:
            status = CheckStatus.ERROR
            violations = (
                Violation(
                    constraint_id=primary_constraint_id,
                    code=f"{self.checker_id}.error",
                    message=result.error,
                ),
            )
        elif result.is_success(command_spec):
            status = CheckStatus.PASS
            violations = ()
        else:
            status = CheckStatus.FAIL
            violations = self.parse_failures(result=result, constraint_id=primary_constraint_id)

        metadata: dict[str, JSONValue] = {
            "command": list(command),
            "allowed_exit_codes": list(allowed_exit_codes),
            "probe_runs": probe_runs,
            "probe_results": [
                {
                    "exit_code": probe.exit_code,
                    "timed_out": probe.timed_out,
                    "error": probe.error,
                }
                for probe in probe_results
            ],
            "stdout_excerpt": result.stdout[:500],
            "stderr_excerpt": result.stderr[:500],
        }

        return CheckResult(
            status=status,
            violations=violations,
            covered_constraint_ids=covered_constraint_ids,
            tool_versions=tool_versions,
            artifact_paths=_artifact_paths_from_params(params),
            logs_path=_logs_path_from_params(params),
            duration_ms=result.duration_ms,
            metadata=metadata,
            checker_id=self.checker_id,
            stage=self.stage,
        )


@register_builtin_checker("build_checker")
class BuildChecker(CommandChecker):
    """Build checker using compilation/parsing validation commands."""

    checker_id = "build_checker"
    stage = "build"
    tool_name = "python"
    covered_constraint_ids = ("CON-COR-0001",)

    default_command = ("python", "-m", "compileall", "-q", "src")
    default_version_command = ("python", "--version")
    default_timeout_seconds = 120.0

    def parse_failures(
        self,
        *,
        result: CommandResult,
        constraint_id: str,
    ) -> tuple[Violation, ...]:
        findings: dict[tuple[str, str, int | None, int | None, str], Violation] = {}

        for line in f"{result.stdout}\n{result.stderr}".splitlines():
            match = _COMPILE_ERROR_RE.match(line.strip())
            if match is None:
                continue
            path = _normalize_relative_path(match.group("path"))
            violation = Violation(
                constraint_id=constraint_id,
                code="build.compile_error",
                message=match.group("message").strip(),
                path=path,
            )
            findings[
                (
                    violation.code,
                    violation.path or "",
                    violation.line,
                    violation.column,
                    violation.message,
                )
            ] = violation

        for item in _parse_common_violations(
            stdout=result.stdout,
            stderr=result.stderr,
            constraint_id=constraint_id,
            fallback_code="build.failure",
        ):
            findings[(item.code, item.path or "", item.line, item.column, item.message)] = item

        return tuple(sorted(findings.values(), key=lambda item: item.sort_key()))


__all__ = [
    "BuildChecker",
    "CommandChecker",
    "capture_tool_version",
    "checker_parameters",
    "extract_constraint_ids",
    "first_constraint_id",
]
