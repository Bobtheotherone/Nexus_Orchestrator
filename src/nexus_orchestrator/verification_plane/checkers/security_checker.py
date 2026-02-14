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
import shlex
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
_DEFAULT_GITLEAKS_REPORT_PATH: Final[str] = (
    ".nexus_orchestrator/checker_artifacts/security_checker_gitleaks.json"
)
_DEFAULT_EXCLUDES: Final[tuple[str, ...]] = (
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".mypy_cache/**",
    ".env.example",
)
_GITLEAKS_EVIDENCE_TAG: Final[str] = "security.secret.gitleaks"
_REGEX_FALLBACK_EVIDENCE_TAG: Final[str] = "security.secret.regex_fallback"


@dataclass(frozen=True, slots=True)
class SecretPattern:
    code: str
    regex: re.Pattern[str]
    description: str


@dataclass(frozen=True, slots=True)
class GitleaksFinding:
    rule_id: str
    description: str
    path: str | None
    line: int | None
    secret: str | None
    fingerprint: str | None

    def sort_key(self) -> tuple[str, int, str, str, str]:
        return (
            self.path or "",
            self.line if self.line is not None else -1,
            self.rule_id,
            self.fingerprint or "",
            self.description,
        )


@dataclass(frozen=True, slots=True)
class SecretAllowlistRule:
    fingerprint: str | None = None
    rule_id: str | None = None
    path_glob: str | None = None
    secret: str | None = None

    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.fingerprint or "",
            self.rule_id or "",
            self.path_glob or "",
            self.secret or "",
        )

    def matches(self, finding: GitleaksFinding) -> bool:
        if self.fingerprint is not None and finding.fingerprint != self.fingerprint:
            return False
        if self.rule_id is not None and finding.rule_id != self.rule_id:
            return False
        if self.secret is not None and finding.secret != self.secret:
            return False
        if self.path_glob is not None:
            if finding.path is None:
                return False
            if not fnmatch.fnmatchcase(finding.path, self.path_glob):
                return False
        return True


@dataclass(frozen=True, slots=True)
class GitleaksScanOutcome:
    status: str
    violations: tuple[Violation, ...]
    report: dict[str, object]
    tool_versions: dict[str, str]
    command_lines: tuple[str, ...]
    timed_out: bool


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
        evidence_tags: list[str] = []

        tool_versions: dict[str, str] = {"security_checker": "builtin-1"}
        secret_timed_out = False
        secret_scan_mode = "not_run"

        if scan_type in {"secret_patterns", "all"}:
            (
                secret_violations,
                secret_report,
                secret_versions,
                secret_command_lines,
                secret_timed_out,
                secret_evidence_tags,
            ) = await _run_secret_scan(
                context=context,
                params=params,
                started=started,
                timeout_seconds=timeout_seconds,
            )
            violations.extend(secret_violations)
            report["secret_scan"] = secret_report
            tool_versions.update(secret_versions)
            command_lines.extend(secret_command_lines)
            evidence_tags.extend(secret_evidence_tags)
            scanner_raw = secret_report.get("scanner")
            if isinstance(scanner_raw, str) and scanner_raw.strip():
                secret_scan_mode = scanner_raw.strip()

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
            default_path=_DEFAULT_REPORT_PATH,
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
        evidence_tags_json: list[JSONValue] = [tag for tag in sorted(set(evidence_tags))]
        metadata: dict[str, JSONValue] = {
            "scan_type": scan_type,
            "timeout_seconds": timeout_seconds,
            "violation_count": len(violations),
            "secret_scan_mode": secret_scan_mode,
            "evidence_tags": evidence_tags_json,
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
) -> tuple[
    list[Violation], dict[str, object], dict[str, str], tuple[str, ...], bool, tuple[str, ...]
]:
    prefer_gitleaks = _as_bool(params.get("prefer_gitleaks"), default=True)
    allowlist = _resolve_allowlist(
        params.get("gitleaks_allowlist", params.get("secret_allowlist", params.get("allowlist")))
    )

    if prefer_gitleaks:
        gitleaks_outcome = await _run_gitleaks_scan(
            context=context,
            params=params,
            timeout_seconds=timeout_seconds,
            allowlist=allowlist,
        )
        if gitleaks_outcome.status == "available":
            return (
                list(gitleaks_outcome.violations),
                gitleaks_outcome.report,
                gitleaks_outcome.tool_versions,
                gitleaks_outcome.command_lines,
                gitleaks_outcome.timed_out,
                (_GITLEAKS_EVIDENCE_TAG,),
            )
        if gitleaks_outcome.status == "timeout":
            return (
                list(gitleaks_outcome.violations),
                gitleaks_outcome.report,
                gitleaks_outcome.tool_versions,
                gitleaks_outcome.command_lines,
                True,
                (_GITLEAKS_EVIDENCE_TAG,),
            )

        regex_violations, regex_report, regex_timed_out = _run_regex_secret_scan(
            context=context,
            params=params,
            started=started,
            timeout_seconds=timeout_seconds,
        )
        warning_violation = Violation(
            constraint_id=_SECRET_CONSTRAINT,
            code="security.secret.gitleaks_unavailable",
            message=("gitleaks unavailable in PATH; used deterministic regex fallback scanner"),
            severity="warning",
            details={"evidence_tag": _REGEX_FALLBACK_EVIDENCE_TAG},
        )

        fallback_report: dict[str, object] = {
            "scanner": "regex_fallback",
            "fallback_reason": gitleaks_outcome.status,
            "evidence_tags": [_REGEX_FALLBACK_EVIDENCE_TAG],
            "gitleaks": gitleaks_outcome.report,
            "regex_scan": regex_report,
        }
        return (
            [warning_violation, *regex_violations],
            fallback_report,
            {"secret_scanner": "builtin-regex-v2"},
            gitleaks_outcome.command_lines,
            regex_timed_out,
            (_REGEX_FALLBACK_EVIDENCE_TAG,),
        )

    regex_violations, regex_report, regex_timed_out = _run_regex_secret_scan(
        context=context,
        params=params,
        started=started,
        timeout_seconds=timeout_seconds,
    )
    report: dict[str, object] = {
        "scanner": "regex",
        "regex_scan": regex_report,
        "evidence_tags": [],
    }
    return (
        regex_violations,
        report,
        {"secret_scanner": "builtin-regex-v2"},
        (),
        regex_timed_out,
        (),
    )


async def _run_gitleaks_scan(
    *,
    context: CheckerContext,
    params: Mapping[str, object],
    timeout_seconds: float,
    allowlist: Sequence[SecretAllowlistRule],
) -> GitleaksScanOutcome:
    workspace_root = Path(context.workspace_path).resolve()
    executor = context.require_executor()
    base_command = _to_command(params.get("gitleaks_command"), default=("gitleaks",))
    probe_command = base_command + ("--version",)
    probe_result = await executor.run(
        CommandSpec(
            argv=probe_command,
            cwd=context.workspace_path,
            timeout_seconds=min(timeout_seconds, 15.0),
        )
    )

    if probe_result.timed_out:
        violation = Violation(
            constraint_id=_SECRET_CONSTRAINT,
            code="security.secret.gitleaks.timeout",
            message="gitleaks version probe timed out",
        )
        report = {
            "scanner": "gitleaks",
            "probe_command": list(probe_command),
            "timed_out": True,
        }
        return GitleaksScanOutcome(
            status="timeout",
            violations=(violation,),
            report=report,
            tool_versions={"secret_scanner": "gitleaks"},
            command_lines=(),
            timed_out=True,
        )

    if not probe_result.is_success() and not _looks_unavailable(probe_result):
        fallback_probe_command = base_command + ("version",)
        fallback_probe_result = await executor.run(
            CommandSpec(
                argv=fallback_probe_command,
                cwd=context.workspace_path,
                timeout_seconds=min(timeout_seconds, 15.0),
            )
        )
        if fallback_probe_result.is_success():
            probe_command = fallback_probe_command
            probe_result = fallback_probe_result

    if _looks_unavailable(probe_result) or probe_result.error is not None:
        report = {
            "scanner": "gitleaks",
            "probe_command": list(probe_command),
            "timed_out": False,
            "available": False,
            "error": probe_result.error
            or probe_result.stderr.strip()
            or probe_result.stdout.strip(),
        }
        return GitleaksScanOutcome(
            status="unavailable",
            violations=(),
            report=report,
            tool_versions={},
            command_lines=(),
            timed_out=False,
        )

    version = _extract_version_text(probe_result)
    report_rel = _coerce_report_path(
        params.get("gitleaks_report_path"),
        default=_DEFAULT_GITLEAKS_REPORT_PATH,
    )
    report_abs = workspace_root / report_rel
    report_abs.parent.mkdir(parents=True, exist_ok=True)

    extra_args = _to_command(params.get("gitleaks_args"), default=())
    detect_command = (
        base_command
        + (
            "detect",
            "--source",
            str(workspace_root),
            "--no-git",
            "--report-format",
            "json",
            "--report-path",
            str(report_abs),
            "--redact",
        )
        + extra_args
    )
    detect_result = await executor.run(
        CommandSpec(
            argv=detect_command,
            cwd=context.workspace_path,
            timeout_seconds=min(timeout_seconds, 90.0),
            allowed_exit_codes=(0, 1),
        )
    )

    if detect_result.timed_out:
        violation = Violation(
            constraint_id=_SECRET_CONSTRAINT,
            code="security.secret.gitleaks.timeout",
            message=f"gitleaks scan timed out after {min(timeout_seconds, 90.0):.3f}s",
        )
        report = {
            "scanner": "gitleaks",
            "available": True,
            "version": version,
            "probe_command": list(probe_command),
            "command": list(detect_command),
            "timed_out": True,
        }
        return GitleaksScanOutcome(
            status="timeout",
            violations=(violation,),
            report=report,
            tool_versions={"secret_scanner": "gitleaks", "gitleaks": version},
            command_lines=(" ".join(detect_command),),
            timed_out=True,
        )

    if _looks_unavailable(detect_result):
        report = {
            "scanner": "gitleaks",
            "available": False,
            "version": version,
            "probe_command": list(probe_command),
            "command": list(detect_command),
            "timed_out": False,
            "error": detect_result.error
            or detect_result.stderr.strip()
            or detect_result.stdout.strip(),
        }
        return GitleaksScanOutcome(
            status="unavailable",
            violations=(),
            report=report,
            tool_versions={},
            command_lines=(" ".join(detect_command),),
            timed_out=False,
        )

    findings_payload, parse_error = _load_gitleaks_payload(report_abs, detect_result.stdout)
    findings = _parse_gitleaks_findings(findings_payload, workspace_root=workspace_root)
    allowed, effective = _apply_allowlist(findings, allowlist=allowlist)

    violations: list[Violation] = []
    if parse_error is not None:
        violations.append(
            Violation(
                constraint_id=_SECRET_CONSTRAINT,
                code="security.secret.gitleaks.report_parse_failed",
                message=parse_error,
            )
        )

    violations.extend(_violations_from_findings(effective))
    if (
        detect_result.exit_code is not None
        and detect_result.exit_code not in (0, 1)
        and not violations
    ):
        violations.append(
            Violation(
                constraint_id=_SECRET_CONSTRAINT,
                code="security.secret.gitleaks.failed",
                message=(
                    detect_result.stderr.strip()
                    or detect_result.stdout.strip()
                    or "gitleaks scan failed"
                ),
            )
        )

    available_report: dict[str, object] = {
        "scanner": "gitleaks",
        "available": True,
        "version": version,
        "probe_command": list(probe_command),
        "command": list(detect_command),
        "report_path": report_rel,
        "timed_out": False,
        "finding_count_total": len(findings),
        "finding_count_allowlisted": len(allowed),
        "finding_count_effective": len(effective),
        "allowlist_rule_count": len(allowlist),
        "effective_findings": [finding_to_dict(item) for item in effective],
        "allowlisted_findings": [finding_to_dict(item) for item in allowed],
    }

    return GitleaksScanOutcome(
        status="available",
        violations=tuple(violations),
        report=available_report,
        tool_versions={"secret_scanner": "gitleaks", "gitleaks": version},
        command_lines=(" ".join(detect_command),),
        timed_out=False,
    )


def finding_to_dict(finding: GitleaksFinding) -> dict[str, object]:
    return {
        "rule_id": finding.rule_id,
        "description": finding.description,
        "path": finding.path,
        "line": finding.line,
        "secret": finding.secret,
        "fingerprint": finding.fingerprint,
    }


def _load_gitleaks_payload(report_path: Path, stdout: str) -> tuple[object | None, str | None]:
    payload_text: str | None = None

    if report_path.exists():
        try:
            payload_text = report_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return None, f"failed to read gitleaks report {report_path}: {exc}"
    elif stdout.strip():
        payload_text = stdout

    if payload_text is None or not payload_text.strip():
        return None, None

    try:
        return json.loads(payload_text), None
    except json.JSONDecodeError as exc:
        return None, f"invalid gitleaks JSON report: {exc}"


def _parse_gitleaks_findings(
    payload: object | None, *, workspace_root: Path
) -> list[GitleaksFinding]:
    if payload is None:
        return []

    candidates: Sequence[object]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        candidates = payload
    elif isinstance(payload, Mapping):
        findings_raw = payload.get("findings")
        if isinstance(findings_raw, Sequence) and not isinstance(
            findings_raw, (str, bytes, bytearray)
        ):
            candidates = findings_raw
        else:
            return []
    else:
        return []

    findings: dict[tuple[str, int, str, str, str], GitleaksFinding] = {}
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        parsed = _parse_gitleaks_finding(item, workspace_root=workspace_root)
        if parsed is None:
            continue
        findings[parsed.sort_key()] = parsed

    return [findings[key] for key in sorted(findings)]


def _parse_gitleaks_finding(
    payload: Mapping[object, object],
    *,
    workspace_root: Path,
) -> GitleaksFinding | None:
    payload_map = {str(key): value for key, value in payload.items() if isinstance(key, str)}
    rule_id = _clean_optional_text(
        payload_map.get("RuleID")
        or payload_map.get("ruleID")
        or payload_map.get("rule_id")
        or payload_map.get("rule")
    )
    description = _clean_optional_text(
        payload_map.get("Description") or payload_map.get("description")
    )
    path_raw = payload_map.get("File") or payload_map.get("file")
    line_raw = (
        payload_map.get("StartLine") or payload_map.get("line") or payload_map.get("startLine")
    )
    secret = _clean_optional_text(payload_map.get("Secret") or payload_map.get("secret"))
    fingerprint = _clean_optional_text(
        payload_map.get("Fingerprint") or payload_map.get("fingerprint")
    )

    path = _normalize_path(path_raw, workspace_root=workspace_root)
    line = _to_positive_int_or_none(line_raw)
    resolved_rule_id = rule_id or "unknown_rule"
    resolved_description = description or "secret detected by gitleaks"

    if path is None and fingerprint is None and secret is None:
        return None
    return GitleaksFinding(
        rule_id=resolved_rule_id,
        description=resolved_description,
        path=path,
        line=line,
        secret=secret,
        fingerprint=fingerprint,
    )


def _apply_allowlist(
    findings: Sequence[GitleaksFinding],
    *,
    allowlist: Sequence[SecretAllowlistRule],
) -> tuple[list[GitleaksFinding], list[GitleaksFinding]]:
    if not allowlist:
        return [], list(findings)

    allowed: list[GitleaksFinding] = []
    effective: list[GitleaksFinding] = []
    for finding in findings:
        if any(rule.matches(finding) for rule in allowlist):
            allowed.append(finding)
        else:
            effective.append(finding)
    return allowed, effective


def _resolve_allowlist(raw: object) -> tuple[SecretAllowlistRule, ...]:
    if raw is None:
        return ()

    if isinstance(raw, Mapping):
        raw_values: Sequence[object] = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        raw_values = raw
    else:
        raw_values = (raw,)

    resolved: dict[tuple[str, str, str, str], SecretAllowlistRule] = {}
    for item in raw_values:
        if isinstance(item, str):
            fingerprint = item.strip()
            if not fingerprint:
                continue
            rule = SecretAllowlistRule(fingerprint=fingerprint)
            resolved[rule.sort_key()] = rule
            continue

        if not isinstance(item, Mapping):
            continue
        payload = {str(key): value for key, value in item.items() if isinstance(key, str)}
        rule = SecretAllowlistRule(
            fingerprint=_clean_optional_text(payload.get("fingerprint")),
            rule_id=_clean_optional_text(payload.get("rule_id") or payload.get("rule")),
            path_glob=_clean_optional_text(
                payload.get("path_glob") or payload.get("path") or payload.get("file")
            ),
            secret=_clean_optional_text(payload.get("secret")),
        )
        if (
            rule.fingerprint is None
            and rule.rule_id is None
            and rule.path_glob is None
            and rule.secret is None
        ):
            continue
        resolved[rule.sort_key()] = rule

    return tuple(resolved[key] for key in sorted(resolved))


def _violations_from_findings(findings: Sequence[GitleaksFinding]) -> list[Violation]:
    violations: list[Violation] = []
    for finding in findings:
        message = f"gitleaks detected secret via {finding.rule_id}: {finding.description}"
        violations.append(
            Violation(
                constraint_id=_SECRET_CONSTRAINT,
                code="security.secret.gitleaks",
                message=message,
                path=finding.path,
                line=finding.line,
                details={
                    "fingerprint": finding.fingerprint or "",
                    "rule_id": finding.rule_id,
                },
            )
        )
    return violations


def _run_regex_secret_scan(
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
        parts = tuple(part for part in shlex.split(raw) if part.strip())
        if parts:
            return parts
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        parts = tuple(item.strip() for item in raw if isinstance(item, str) and item.strip())
        if parts:
            return parts
    fallback = tuple(item.strip() for item in default if item.strip())
    return fallback


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


def _to_positive_int_or_none(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw > 0 else None
    if isinstance(raw, float):
        parsed = int(raw)
        return parsed if parsed > 0 else None
    return None


def _as_bool(raw: object, *, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    return default


def _looks_unavailable(result: CommandResult) -> bool:
    text = f"{result.stdout}\n{result.stderr}\n{result.error or ''}".lower()
    markers = (
        "command not found",
        "no such file or directory",
        "not recognized as an internal or external command",
        "executable file not found",
    )
    return any(marker in text for marker in markers)


def _extract_version_text(result: CommandResult) -> str:
    for line in f"{result.stdout}\n{result.stderr}".splitlines():
        normalized = line.strip()
        if normalized:
            return normalized
    return "unknown"


def _clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_path(value: object, *, workspace_root: Path) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    candidate_path = Path(normalized)
    if candidate_path.is_absolute():
        try:
            normalized = candidate_path.resolve(strict=False).relative_to(workspace_root).as_posix()
        except ValueError:
            normalized = candidate_path.as_posix()
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    return pure.as_posix()


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
    if any(item.severity.lower() == "warning" for item in violations):
        return CheckStatus.WARN
    return CheckStatus.PASS


def _write_report(
    *,
    workspace_root: Path,
    report_path_raw: object,
    payload: Mapping[str, object],
    default_path: str,
) -> tuple[str | None, str | None]:
    report_rel = _coerce_report_path(report_path_raw, default=default_path)
    destination = workspace_root / report_rel
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        return None, f"failed to write security report {report_rel}: {exc}"

    return report_rel, None


def _coerce_report_path(raw: object, *, default: str) -> str:
    if isinstance(raw, str) and raw.strip():
        candidate = raw.replace("\\", "/").strip()
        pure = PurePosixPath(candidate)
        if not pure.is_absolute() and ".." not in pure.parts:
            return candidate
    return default


__all__ = ["SecurityChecker", "SecretPattern"]
