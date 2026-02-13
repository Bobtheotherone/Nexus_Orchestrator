"""
Security Checker — verification stage.

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

import fnmatch
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from nexus_orchestrator.verification_plane.checkers.build_checker import (
    capture_tool_version,
    checker_parameters,
    extract_constraint_ids,
    first_constraint_id,
)

_SECRET_CONSTRAINT: Final[str] = "CON-SEC-0001"
_DEP_AUDIT_CONSTRAINT: Final[str] = "CON-SEC-0002"
_DEFAULT_REPORT_PATH: Final[str] = (
    ".nexus_orchestrator/checker_artifacts/security_checker_report.json"
)
_DEFAULT_EXCLUDES: Final[tuple[str, ...]] = (
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    ".env.example",
)


@dataclass(frozen=True, slots=True)
class SecretPattern:
    code: str
    regex: re.Pattern[str]
    description: str


@register_builtin_checker("security_checker")
class SecurityChecker(BaseChecker):
    """Secret scanner + optional dependency audit checker."""

    checker_id = "security_checker"
    stage = "security"
    covered_constraint_ids = (_SECRET_CONSTRAINT, _DEP_AUDIT_CONSTRAINT)

    async def check(self, context: CheckerContext) -> CheckResult:
        started = time.monotonic()
        params = checker_parameters(context, self.checker_id)
        covered_constraint_ids = extract_constraint_ids(
            context,
            params,
            self.covered_constraint_ids,
        )

        timeout_seconds = _to_positive_float(params.get("timeout_seconds"), default=120.0)
        scan_type = _scan_type(params.get("scan_type"))

        violations: list[Violation] = []
        report: dict[str, object] = {
            "checker_id": self.checker_id,
            "scan_type": scan_type,
            "secret_scan": {},
            "dependency_audit": {},
        }
        command_lines: list[str] = []

        tool_versions: dict[str, str] = {
            "security_checker": "builtin-1",
            "secret_scanner": "builtin-regex-v1",
        }
        secret_timed_out = False

        if scan_type in {"secret_patterns", "all"}:
            secret_violations, secret_report, secret_timed_out = await _run_secret_scan(
                context=context,
                params=params,
                started=started,
                timeout_seconds=timeout_seconds,
            )
            violations.extend(secret_violations)
            report["secret_scan"] = secret_report

        dep_timed_out = False
        if scan_type in {"dependency_audit", "all"}:
            dep_required = (
                bool(params.get("dependency_audit_required"))
                if isinstance(params.get("dependency_audit_required"), bool)
                else (_DEP_AUDIT_CONSTRAINT in covered_constraint_ids)
            )
            dep_violations, dep_report, dep_versions, dep_timed_out = await _run_dependency_audit(
                context=context,
                params=params,
                timeout_seconds=timeout_seconds,
                required=dep_required,
            )
            violations.extend(dep_violations)
            report["dependency_audit"] = dep_report
            tool_versions.update(dep_versions)
            dep_command = dep_report.get("command")
            if isinstance(dep_command, list) and all(isinstance(item, str) for item in dep_command):
                command_lines.append(" ".join(dep_command))

        report_path, report_write_error = _write_report(
            workspace_root=Path(context.workspace_path),
            report_path_raw=params.get("report_path"),
            payload=report,
        )
        if report_write_error is not None:
            violations.append(
                Violation(
                    constraint_id=first_constraint_id(covered_constraint_ids),
                    code="security.report_write_failed",
                    message=report_write_error,
                    severity="warning",
                )
            )

        status = _derive_status(
            violations=violations,
            dependency_timed_out=dep_timed_out,
            secret_timed_out=secret_timed_out,
        )
        if status is CheckStatus.TIMEOUT and not any(
            v.code.endswith(".timeout") for v in violations
        ):
            violations.append(
                Violation(
                    constraint_id=first_constraint_id(covered_constraint_ids),
                    code="security.timeout",
                    message=f"security checks timed out after {timeout_seconds:.3f}s",
                )
            )

        artifact_paths = normalize_artifact_paths((report_path,)) if report_path else ()
        metadata: dict[str, JSONValue] = {
            "scan_type": scan_type,
            "timeout_seconds": timeout_seconds,
            "violation_count": len(violations),
        }
        duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)

        return CheckResult(
            status=status,
            violations=tuple(violations),
            covered_constraint_ids=covered_constraint_ids,
            tool_versions={key: tool_versions[key] for key in sorted(tool_versions)},
            artifact_paths=artifact_paths,
            logs_path=None,
            command_lines=tuple(command_lines),
            duration_ms=duration_ms,
            metadata=metadata,
            checker_id=self.checker_id,
            stage=self.stage,
        )


async def _run_secret_scan(
    *,
    context: CheckerContext,
    params: Mapping[str, object],
    started: float,
    timeout_seconds: float,
) -> tuple[list[Violation], dict[str, object], bool]:
    patterns = _resolve_secret_patterns(params.get("patterns"))
    excludes = _resolve_excludes(params.get("exclude_paths"))
    max_file_size = _to_positive_int(params.get("max_file_size_bytes"), default=1_048_576)

    workspace_root = Path(context.workspace_path).resolve()
    files = _collect_scan_files(workspace_root=workspace_root, raw_paths=params.get("scan_paths"))

    violations: list[Violation] = []
    scanned_files: list[str] = []

    for file_path in files:
        if _timed_out(started, timeout_seconds):
            return (
                violations,
                {
                    "patterns": [item.code for item in patterns],
                    "scanned_files": sorted(scanned_files),
                    "timeout_seconds": timeout_seconds,
                    "timed_out": True,
                },
                True,
            )
        rel = _relative_safe(file_path, workspace_root)
        if rel is None:
            continue
        if any(fnmatch.fnmatchcase(rel, pattern) for pattern in excludes):
            continue

        try:
            stat = file_path.stat()
        except OSError:
            continue
        if stat.st_size > max_file_size:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        scanned_files.append(rel)
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern.regex.search(line) is None:
                    continue
                violations.append(
                    Violation(
                        constraint_id=_SECRET_CONSTRAINT,
                        code=pattern.code,
                        message=pattern.description,
                        path=rel,
                        line=line_no,
                    )
                )

    deduped = {
        (
            item.constraint_id,
            item.code,
            item.path,
            item.line,
            item.column,
            item.message,
        ): item
        for item in violations
    }

    report = {
        "patterns": [item.code for item in patterns],
        "scanned_files": sorted(scanned_files),
        "timeout_seconds": timeout_seconds,
        "timed_out": False,
    }
    return list(sorted(deduped.values(), key=lambda item: item.sort_key())), report, False


async def _run_dependency_audit(
    *,
    context: CheckerContext,
    params: Mapping[str, object],
    timeout_seconds: float,
    required: bool,
) -> tuple[list[Violation], dict[str, object], dict[str, str], bool]:
    raw_command = params.get("dependency_audit_command", ("pip-audit", "-f", "json"))
    command = _to_command(raw_command, default=("pip-audit", "-f", "json"))

    spec = CommandSpec(
        argv=command,
        cwd=context.workspace_path,
        timeout_seconds=min(timeout_seconds, 60.0),
    )
    result = await context.require_executor().run(spec)

    version = await capture_tool_version(
        context=context,
        command=command,
        timeout_seconds=timeout_seconds,
        version_command=None,
    )
    tool_versions = {command[0]: version}

    report: dict[str, object] = {
        "command": list(command),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "required": required,
    }

    if result.timed_out:
        return (
            [
                Violation(
                    constraint_id=_DEP_AUDIT_CONSTRAINT,
                    code="security.dependency_audit.timeout",
                    message=f"dependency audit timed out after {min(timeout_seconds, 60.0):.3f}s",
                )
            ],
            report,
            tool_versions,
            True,
        )

    if result.error is not None or _looks_unavailable(result):
        severity = "error" if required else "warning"
        message = (
            "dependency audit tool unavailable; install pip-audit or configure "
            "dependency_audit_command"
        )
        return (
            [
                Violation(
                    constraint_id=_DEP_AUDIT_CONSTRAINT,
                    code="security.dependency_audit.unavailable",
                    message=message,
                    severity=severity,
                )
            ],
            report,
            tool_versions,
            False,
        )

    if result.is_success(spec):
        return ([], report, tool_versions, False)

    parsed = _parse_dependency_audit_json(result.stdout)
    if not parsed:
        parsed = [
            Violation(
                constraint_id=_DEP_AUDIT_CONSTRAINT,
                code="security.dependency_audit.failed",
                message=(
                    result.stderr.strip() or result.stdout.strip() or "dependency audit failed"
                ),
            )
        ]

    return (parsed, report, tool_versions, False)


def _parse_dependency_audit_json(payload: str) -> list[Violation]:
    text = payload.strip()
    if not text:
        return []

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(decoded, Mapping):
        return []

    deps_raw = decoded.get("dependencies")
    if not isinstance(deps_raw, Sequence) or isinstance(deps_raw, (str, bytes, bytearray)):
        return []

    findings: dict[tuple[str, str], Violation] = {}

    for dep in deps_raw:
        if not isinstance(dep, Mapping):
            continue
        dep_name = dep.get("name")
        dep_version = dep.get("version")
        vulns = dep.get("vulns")
        if not isinstance(dep_name, str) or not isinstance(dep_version, str):
            continue
        if not isinstance(vulns, Sequence) or isinstance(vulns, (str, bytes, bytearray)):
            continue

        for vuln in vulns:
            if not isinstance(vuln, Mapping):
                continue
            vuln_id = vuln.get("id")
            description = vuln.get("description")
            issue_id = (
                vuln_id.strip() if isinstance(vuln_id, str) and vuln_id.strip() else "unknown"
            )
            summary = (
                description.strip()
                if isinstance(description, str) and description.strip()
                else "vulnerability detected"
            )
            message = f"{dep_name}=={dep_version} vulnerable ({issue_id}): {summary}"
            violation = Violation(
                constraint_id=_DEP_AUDIT_CONSTRAINT,
                code="security.dependency_audit.vulnerability",
                message=message,
            )
            findings[(violation.code, violation.message)] = violation

    return [findings[key] for key in sorted(findings)]


def _resolve_secret_patterns(raw: object) -> tuple[SecretPattern, ...]:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        patterns: list[SecretPattern] = []
        for index, item in enumerate(raw):
            if isinstance(item, str) and item.strip():
                token = re.escape(item.strip())
                patterns.append(
                    SecretPattern(
                        code=f"security.secret.pattern_{index:03d}",
                        regex=re.compile(token),
                        description=f"matched configured secret token {item.strip()!r}",
                    )
                )
            elif isinstance(item, Mapping):
                code_raw = item.get("code")
                regex_raw = item.get("regex")
                description_raw = item.get("description")
                if not isinstance(regex_raw, str) or not regex_raw.strip():
                    continue
                try:
                    compiled = re.compile(regex_raw)
                except re.error:
                    continue
                code = (
                    code_raw.strip()
                    if isinstance(code_raw, str) and code_raw.strip()
                    else f"security.secret.pattern_{index:03d}"
                )
                description = (
                    description_raw.strip()
                    if isinstance(description_raw, str) and description_raw.strip()
                    else "matched configured secret regex"
                )
                patterns.append(SecretPattern(code=code, regex=compiled, description=description))

        if patterns:
            return tuple(patterns)

    return (
        SecretPattern(
            code="security.secret.api_key",
            regex=re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            description="possible API key detected",
        ),
        SecretPattern(
            code="security.secret.aws_access_key",
            regex=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            description="possible AWS access key detected",
        ),
        SecretPattern(
            code="security.secret.hardcoded_assignment",
            regex=re.compile(
                r"(?i)\b(api[_-]?key|secret|password|token|private[_-]?key)\b"
                r"\s*[:=]\s*['\"][^'\"]{8,}['\"]"
            ),
            description="possible hard-coded secret assignment detected",
        ),
    )


def _resolve_excludes(raw: object) -> tuple[str, ...]:
    values = set(_DEFAULT_EXCLUDES)
    if isinstance(raw, str) and raw.strip():
        values.add(raw.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.add(item.strip())
    return tuple(sorted(values))


def _collect_scan_files(*, workspace_root: Path, raw_paths: object) -> list[Path]:
    candidates: list[Path] = []

    if raw_paths is None:
        candidates.extend(path for path in workspace_root.rglob("*") if path.is_file())
    elif isinstance(raw_paths, str):
        path = workspace_root / raw_paths
        if path.is_file():
            candidates.append(path)
        elif path.is_dir():
            candidates.extend(item for item in path.rglob("*") if item.is_file())
    elif isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes, bytearray)):
        for raw in raw_paths:
            if not isinstance(raw, str) or not raw.strip():
                continue
            path = workspace_root / raw
            if path.is_file():
                candidates.append(path)
            elif path.is_dir():
                candidates.extend(item for item in path.rglob("*") if item.is_file())

    deduped = sorted({path.resolve(strict=False) for path in candidates})
    return [path for path in deduped if _relative_safe(path, workspace_root) is not None]


def _relative_safe(path: Path, workspace_root: Path) -> str | None:
    try:
        relative = path.resolve(strict=False).relative_to(workspace_root)
    except ValueError:
        return None
    return relative.as_posix()


def _to_command(raw: object, *, default: Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = tuple(part for part in raw.split(" ") if part.strip())
        if parts:
            return parts
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        parts = tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())
        if parts:
            return parts
    fallback = tuple(item.strip() for item in default if item.strip())
    return fallback if fallback else tuple(default)


def _scan_type(raw: object) -> str:
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"secret_patterns", "dependency_audit", "all"}:
            return lowered
    return "secret_patterns"


def _to_positive_float(raw: object, *, default: float) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        parsed = float(raw)
        if parsed > 0:
            return parsed
    return default


def _timed_out(started: float, timeout_seconds: float) -> bool:
    return (time.monotonic() - started) > timeout_seconds


def _to_positive_int(raw: object, *, default: int) -> int:
    if isinstance(raw, int) and raw > 0:
        return raw
    return default


def _as_bool(raw: object, *, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    return default


def _looks_unavailable(result: CommandResult) -> bool:
    text = f"{result.stdout}\n{result.stderr}".lower()
    markers = (
        "command not found",
        "no such file or directory",
        "not recognized as an internal or external command",
    )
    return any(marker in text for marker in markers)


def _derive_status(
    *,
    violations: Sequence[Violation],
    dependency_timed_out: bool,
    secret_timed_out: bool,
) -> CheckStatus:
    if dependency_timed_out or secret_timed_out:
        return CheckStatus.TIMEOUT
    if any(item.severity.lower() == "error" for item in violations):
        return CheckStatus.FAIL
    return CheckStatus.PASS


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
        return None, f"failed to write security report {report_rel}: {exc}"

    return report_rel, None


def _coerce_report_path(raw: object) -> str:
    if isinstance(raw, str) and raw.strip():
        candidate = raw.replace("\\", "/").strip()
        pure = PurePosixPath(candidate)
        if not pure.is_absolute() and ".." not in pure.parts:
            return candidate
    return _DEFAULT_REPORT_PATH


__all__ = ["SecurityChecker", "SecretPattern"]
