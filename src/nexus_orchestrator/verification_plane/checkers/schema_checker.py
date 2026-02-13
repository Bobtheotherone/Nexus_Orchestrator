"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/schema_checker.py
Last updated: 2026-02-11

Purpose
- Validate that structured registries (constraints, config, evidence ledger) conform to their schemas and invariants.

What should be included in this file
- A SchemaChecker implementing BaseChecker.
- Schema definitions (JSON Schema / YAML schema) and validation wiring.
- Registry-specific invariants: unique IDs, required fields, no dangling references, valid severity/category enums.
- Helpful error reporting for agents (precise paths, suggested fixes).

Functional requirements
- Must validate constraint registry YAML against schema and internal invariants.
- Must validate orchestrator config against schema (after normalization) and fail fast on unknown keys.
- Must emit machine-readable validation reports (JSON) for ingestion by FeedbackSynthesizer.

Non-functional requirements
- Deterministic and fast; schema validation should be cheap.
- No network requirements.
"""

from __future__ import annotations

import json
import re
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

import yaml

from nexus_orchestrator.config.schema import validate_config
from nexus_orchestrator.verification_plane.checkers.base import (
    BaseChecker,
    CheckerContext,
    CheckResult,
    CheckStatus,
    Violation,
    normalize_artifact_paths,
    register_builtin_checker,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import (
    checker_parameters,
    extract_constraint_ids,
)

_DEFAULT_REPORT_PATH: Final[str] = (
    ".nexus_orchestrator/checker_artifacts/schema_checker_report.json"
)
_DEFAULT_REGISTRY_PATH: Final[str] = "constraints/registry"
_DEFAULT_CONFIG_PATH: Final[str] = "orchestrator.toml"
_DEFAULT_LEDGER_PATH: Final[str] = "state"

_CONSTRAINT_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "severity",
        "category",
        "description",
        "checker",
        "parameters",
        "requirement_links",
        "source",
    }
)
_CONSTRAINT_ALLOWED_FIELDS: Final[frozenset[str]] = _CONSTRAINT_REQUIRED_FIELDS | frozenset(
    {"checker_binding"}
)
_CONSTRAINT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^CON-[A-Z0-9-]+-\d{4}$")
_ALLOWED_SEVERITY: Final[frozenset[str]] = frozenset({"must", "should", "may"})
_ALLOWED_SOURCE: Final[frozenset[str]] = frozenset({"manual", "spec_derived", "failure_derived"})
_ALLOWED_CATEGORY: Final[frozenset[str]] = frozenset(
    {
        "structural",
        "behavioral",
        "performance",
        "security",
        "style",
        "correctness",
        "documentation",
        "reliability",
        "audit",
    }
)
_ALLOWED_CHECKERS: Final[frozenset[str]] = frozenset(
    {
        "scope_checker",
        "schema_checker",
        "build_checker",
        "lint_checker",
        "typecheck_checker",
        "test_checker",
        "security_checker",
        "documentation_checker",
        "reliability_checker",
        "performance_checker",
    }
)

_TARGET_TO_CONSTRAINT: Final[dict[str, str]] = {
    "constraint_registry": "CON-REG-0001",
    "orchestrator_config": "CON-CONF-0001",
    "evidence_ledger": "CON-EVI-0001",
}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One deterministic schema/invariant validation issue."""

    target: str
    code: str
    path: str
    message: str


@register_builtin_checker("schema_checker")
class SchemaChecker(BaseChecker):
    """Constraint registry/config/evidence schema checker."""

    checker_id = "schema_checker"
    stage = "schema"
    covered_constraint_ids = ("CON-REG-0001", "CON-CONF-0001", "CON-EVI-0001")

    async def check(self, context: CheckerContext) -> CheckResult:
        started = time.monotonic()
        params = checker_parameters(context, self.checker_id)
        timeout_seconds = _to_positive_float(params.get("timeout_seconds"), default=30.0)
        targets = _resolve_targets(params.get("target_kind"))

        covered_constraint_ids = tuple(
            sorted(
                {
                    *extract_constraint_ids(context, params, self.covered_constraint_ids),
                    *(_TARGET_TO_CONSTRAINT[target] for target in targets),
                }
            )
        )
        if params.get("force_timeout") is True:
            return _timeout_result(self, covered_constraint_ids, timeout_seconds)

        issues: list[ValidationIssue] = []
        report_sections: dict[str, object] = {}
        workspace_root = Path(context.workspace_path)

        if _timed_out(started, timeout_seconds):
            return _timeout_result(self, covered_constraint_ids, timeout_seconds)

        if "constraint_registry" in targets:
            if _timed_out(started, timeout_seconds):
                return _timeout_result(self, covered_constraint_ids, timeout_seconds)
            section_report, section_issues = _validate_constraint_registry(
                workspace_root=workspace_root,
                registry_rel=_coerce_rel_path(
                    params.get("registry_path"), default=_DEFAULT_REGISTRY_PATH
                ),
            )
            report_sections["constraint_registry"] = section_report
            issues.extend(section_issues)

        if "orchestrator_config" in targets:
            if _timed_out(started, timeout_seconds):
                return _timeout_result(self, covered_constraint_ids, timeout_seconds)
            section_report, section_issues = _validate_orchestrator_config(
                workspace_root=workspace_root,
                config_rel=_coerce_rel_path(
                    params.get("config_path"), default=_DEFAULT_CONFIG_PATH
                ),
            )
            report_sections["orchestrator_config"] = section_report
            issues.extend(section_issues)

        if "evidence_ledger" in targets:
            if _timed_out(started, timeout_seconds):
                return _timeout_result(self, covered_constraint_ids, timeout_seconds)
            section_report, section_issues = _validate_evidence_ledger(
                workspace_root=workspace_root,
                ledger_rel=_coerce_rel_path(
                    params.get("ledger_path"), default=_DEFAULT_LEDGER_PATH
                ),
            )
            report_sections["evidence_ledger"] = section_report
            issues.extend(section_issues)

        ordered_issues = sorted(
            issues,
            key=lambda item: (item.target, item.path, item.code, item.message),
        )

        report_payload: dict[str, object] = {
            "checker_id": self.checker_id,
            "targets": list(targets),
            "issues": [
                {
                    "target": issue.target,
                    "code": issue.code,
                    "path": issue.path,
                    "message": issue.message,
                }
                for issue in ordered_issues
            ],
            "reports": report_sections,
        }

        report_path, report_error = _write_report(
            workspace_root=workspace_root,
            report_path_raw=params.get("report_path"),
            payload=report_payload,
        )

        violations = [
            Violation(
                constraint_id=_TARGET_TO_CONSTRAINT.get(issue.target, "UNMAPPED"),
                code=issue.code,
                message=issue.message,
                path=issue.path,
            )
            for issue in ordered_issues
        ]
        if report_error is not None:
            violations.append(
                Violation(
                    constraint_id="UNMAPPED",
                    code="schema.report_write_failed",
                    message=report_error,
                    severity="warning",
                )
            )

        status = (
            CheckStatus.PASS
            if not any(v.severity.lower() == "error" for v in violations)
            else CheckStatus.FAIL
        )
        duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)

        return CheckResult(
            status=status,
            violations=tuple(violations),
            covered_constraint_ids=covered_constraint_ids,
            tool_versions={
                "schema_checker": "builtin-1",
                "pyyaml": str(getattr(yaml, "__version__", "unknown")),
                "config_validator": "nexus_orchestrator.config.schema.validate_config",
            },
            artifact_paths=normalize_artifact_paths((report_path,)) if report_path else (),
            logs_path=None,
            command_lines=(),
            duration_ms=duration_ms,
            metadata={
                "targets": list(targets),
                "issue_count": len(ordered_issues),
                "timeout_seconds": timeout_seconds,
            },
            checker_id=self.checker_id,
            stage=self.stage,
        )


def _resolve_targets(raw: object) -> tuple[str, ...]:
    valid = {"constraint_registry", "orchestrator_config", "evidence_ledger"}
    if isinstance(raw, str):
        candidate = raw.strip().lower()
        return (candidate,) if candidate in valid else tuple(sorted(valid))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        collected = sorted(
            {
                item.strip().lower()
                for item in raw
                if isinstance(item, str) and item.strip().lower() in valid
            }
        )
        if collected:
            return tuple(collected)
    return ("constraint_registry", "orchestrator_config", "evidence_ledger")


def _validate_constraint_registry(
    *,
    workspace_root: Path,
    registry_rel: str,
) -> tuple[dict[str, object], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    report: dict[str, object] = {"registry_path": registry_rel, "files": [], "constraint_count": 0}

    registry_dir = workspace_root / registry_rel
    if not registry_dir.exists():
        issues.append(
            ValidationIssue(
                target="constraint_registry",
                code="schema.registry.missing",
                path=registry_rel,
                message="constraint registry path does not exist",
            )
        )
        return report, issues
    if not registry_dir.is_dir():
        issues.append(
            ValidationIssue(
                target="constraint_registry",
                code="schema.registry.invalid",
                path=registry_rel,
                message="constraint registry path must be a directory",
            )
        )
        return report, issues

    yaml_files = sorted(
        path
        for path in registry_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not yaml_files:
        issues.append(
            ValidationIssue(
                target="constraint_registry",
                code="schema.registry.empty",
                path=registry_rel,
                message="constraint registry has no YAML files",
            )
        )
        return report, issues

    seen_ids: dict[str, str] = {}
    total = 0

    for file_path in yaml_files:
        rel_file = _relative_to_workspace(file_path, workspace_root)
        files = report.get("files")
        if isinstance(files, list):
            files.append(rel_file)

        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except OSError as exc:
            issues.append(
                ValidationIssue(
                    target="constraint_registry",
                    code="schema.registry.read_error",
                    path=rel_file,
                    message=f"unable to read YAML: {exc}",
                )
            )
            continue
        except yaml.YAMLError as exc:
            issues.append(
                ValidationIssue(
                    target="constraint_registry",
                    code="schema.registry.parse_error",
                    path=rel_file,
                    message=f"invalid YAML: {exc}",
                )
            )
            continue

        if isinstance(payload, list):
            records: Sequence[object] = payload
        elif isinstance(payload, Mapping):
            constraints_obj = payload.get("constraints")
            if isinstance(constraints_obj, Sequence) and not isinstance(
                constraints_obj, (str, bytes, bytearray)
            ):
                records = constraints_obj
            else:
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.registry.shape",
                        path=rel_file,
                        message="root must be a list or object with 'constraints' list",
                    )
                )
                continue
        else:
            issues.append(
                ValidationIssue(
                    target="constraint_registry",
                    code="schema.registry.shape",
                    path=rel_file,
                    message="root must be a list or object with 'constraints' list",
                )
            )
            continue

        for index, record in enumerate(records):
            entry_path = f"{rel_file}[{index}]"
            if not isinstance(record, Mapping):
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.type",
                        path=entry_path,
                        message="constraint entry must be an object",
                    )
                )
                continue

            total += 1
            entry = {str(key): value for key, value in record.items()}

            unknown_fields = sorted(key for key in entry if key not in _CONSTRAINT_ALLOWED_FIELDS)
            for field_name in unknown_fields:
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.unknown_field",
                        path=f"{entry_path}.{field_name}",
                        message="unknown field",
                    )
                )

            missing_fields = sorted(
                field for field in _CONSTRAINT_REQUIRED_FIELDS if field not in entry
            )
            for field_name in missing_fields:
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.missing_field",
                        path=f"{entry_path}.{field_name}",
                        message="required field missing",
                    )
                )

            constraint_id = entry.get("id")
            if (
                not isinstance(constraint_id, str)
                or not constraint_id.strip()
                or _CONSTRAINT_ID_RE.match(constraint_id.strip()) is None
            ):
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.id_invalid",
                        path=f"{entry_path}.id",
                        message="id must match pattern CON-<GROUP>-<NNNN>",
                    )
                )
            else:
                normalized_id = constraint_id.strip()
                if normalized_id in seen_ids:
                    issues.append(
                        ValidationIssue(
                            target="constraint_registry",
                            code="schema.constraint.id_duplicate",
                            path=f"{entry_path}.id",
                            message=f"duplicate id also seen at {seen_ids[normalized_id]}",
                        )
                    )
                else:
                    seen_ids[normalized_id] = f"{entry_path}.id"

            severity = entry.get("severity")
            if not isinstance(severity, str) or severity.strip() not in _ALLOWED_SEVERITY:
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.severity_invalid",
                        path=f"{entry_path}.severity",
                        message=f"severity must be one of {sorted(_ALLOWED_SEVERITY)}",
                    )
                )

            category = entry.get("category")
            if not isinstance(category, str) or category.strip() not in _ALLOWED_CATEGORY:
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.category_invalid",
                        path=f"{entry_path}.category",
                        message=f"category must be one of {sorted(_ALLOWED_CATEGORY)}",
                    )
                )

            source = entry.get("source")
            if not isinstance(source, str) or source.strip() not in _ALLOWED_SOURCE:
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.source_invalid",
                        path=f"{entry_path}.source",
                        message=f"source must be one of {sorted(_ALLOWED_SOURCE)}",
                    )
                )

            checker_binding = entry.get("checker")
            if not isinstance(checker_binding, str):
                checker_binding = (
                    entry.get("checker_binding")
                    if isinstance(entry.get("checker_binding"), str)
                    else None
                )
            if (
                not isinstance(checker_binding, str)
                or checker_binding.strip() not in _ALLOWED_CHECKERS
            ):
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.checker_invalid",
                        path=f"{entry_path}.checker",
                        message=f"checker must be one of {sorted(_ALLOWED_CHECKERS)}",
                    )
                )

            description = entry.get("description")
            if not isinstance(description, str) or not description.strip():
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.description_invalid",
                        path=f"{entry_path}.description",
                        message="description must be non-empty",
                    )
                )

            parameters = entry.get("parameters")
            if not isinstance(parameters, Mapping):
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.parameters_invalid",
                        path=f"{entry_path}.parameters",
                        message="parameters must be an object",
                    )
                )

            req_links = entry.get("requirement_links")
            if not isinstance(req_links, Sequence) or isinstance(
                req_links, (str, bytes, bytearray)
            ):
                issues.append(
                    ValidationIssue(
                        target="constraint_registry",
                        code="schema.constraint.requirement_links_invalid",
                        path=f"{entry_path}.requirement_links",
                        message="requirement_links must be a list of strings",
                    )
                )
            else:
                for req_idx, req in enumerate(req_links):
                    if not isinstance(req, str) or not req.strip():
                        issues.append(
                            ValidationIssue(
                                target="constraint_registry",
                                code="schema.constraint.requirement_link_invalid",
                                path=f"{entry_path}.requirement_links[{req_idx}]",
                                message="requirement link must be non-empty string",
                            )
                        )

    report["constraint_count"] = total
    return report, issues


def _validate_orchestrator_config(
    *,
    workspace_root: Path,
    config_rel: str,
) -> tuple[dict[str, object], list[ValidationIssue]]:
    report: dict[str, object] = {"config_path": config_rel, "is_valid": False, "issues": []}
    issues: list[ValidationIssue] = []

    config_path = workspace_root / config_rel
    if not config_path.exists():
        issues.append(
            ValidationIssue(
                target="orchestrator_config",
                code="schema.config.missing",
                path=config_rel,
                message="orchestrator config file does not exist",
            )
        )
        return report, issues

    try:
        with config_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except OSError as exc:
        issues.append(
            ValidationIssue(
                target="orchestrator_config",
                code="schema.config.read_error",
                path=config_rel,
                message=f"unable to read config: {exc}",
            )
        )
        return report, issues
    except tomllib.TOMLDecodeError as exc:
        issues.append(
            ValidationIssue(
                target="orchestrator_config",
                code="schema.config.parse_error",
                path=config_rel,
                message=f"invalid TOML: {exc}",
            )
        )
        return report, issues

    validation = validate_config(payload)
    if validation.is_valid:
        report["is_valid"] = True
        return report, issues

    structured_issues: list[dict[str, str]] = []
    for issue in validation.issues:
        issue_path = issue.path.strip() if issue.path.strip() else "<root>"
        full_path = f"{config_rel}.{issue_path}" if issue_path != "<root>" else config_rel
        issues.append(
            ValidationIssue(
                target="orchestrator_config",
                code="schema.config.invalid",
                path=full_path,
                message=issue.message,
            )
        )
        structured_issues.append({"path": full_path, "message": issue.message})

    report["issues"] = sorted(structured_issues, key=lambda item: (item["path"], item["message"]))
    return report, issues


def _validate_evidence_ledger(
    *,
    workspace_root: Path,
    ledger_rel: str,
) -> tuple[dict[str, object], list[ValidationIssue]]:
    report: dict[str, object] = {
        "ledger_path": ledger_rel,
        "present": False,
        "entries": [],
    }
    issues: list[ValidationIssue] = []

    ledger_root = workspace_root / ledger_rel
    if not ledger_root.exists():
        return report, issues

    if ledger_root.is_file():
        report["present"] = True
        _validate_ledger_entry(
            file_path=ledger_root,
            workspace_root=workspace_root,
            report=report,
            issues=issues,
        )
        return report, issues

    entries = sorted(path for path in ledger_root.rglob("*") if path.is_file())
    materialized = [path for path in entries if path.name.lower() != "readme.md"]
    report["present"] = bool(materialized)

    for file_path in materialized:
        _validate_ledger_entry(
            file_path=file_path,
            workspace_root=workspace_root,
            report=report,
            issues=issues,
        )

    if materialized and not _contains_ledger_anchor(materialized):
        issues.append(
            ValidationIssue(
                target="evidence_ledger",
                code="schema.ledger.layout_invalid",
                path=ledger_rel,
                message=(
                    "ledger present without anchor; expected index.json, ledger.json, "
                    "evidence_ledger.json, or *.sqlite/*.db"
                ),
            )
        )

    return report, issues


def _validate_ledger_entry(
    *,
    file_path: Path,
    workspace_root: Path,
    report: dict[str, object],
    issues: list[ValidationIssue],
) -> None:
    rel = _relative_to_workspace(file_path, workspace_root)

    entries = report.get("entries")
    if isinstance(entries, list):
        entries.append(rel)

    try:
        if file_path.is_symlink():
            issues.append(
                ValidationIssue(
                    target="evidence_ledger",
                    code="schema.ledger.symlink_disallowed",
                    path=rel,
                    message="ledger entries must not be symlinks",
                )
            )
    except OSError as exc:
        issues.append(
            ValidationIssue(
                target="evidence_ledger",
                code="schema.ledger.stat_error",
                path=rel,
                message=f"unable to stat entry: {exc}",
            )
        )
        return

    allowed_suffixes = {
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".log",
        ".md",
        ".sqlite",
        ".sqlite3",
        ".db",
    }
    suffix = file_path.suffix.lower()
    if suffix not in allowed_suffixes:
        issues.append(
            ValidationIssue(
                target="evidence_ledger",
                code="schema.ledger.suffix_invalid",
                path=rel,
                message=f"unsupported ledger suffix {suffix!r}",
            )
        )


def _contains_ledger_anchor(entries: Sequence[Path]) -> bool:
    for path in entries:
        name = path.name.lower()
        if name in {"index.json", "ledger.json", "evidence_ledger.json"}:
            return True
        if path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            return True
    return False


def _coerce_rel_path(raw: object, *, default: str) -> str:
    if isinstance(raw, str) and raw.strip():
        candidate = raw.replace("\\", "/").strip()
        pure = PurePosixPath(candidate)
        if not pure.is_absolute() and ".." not in pure.parts:
            return candidate
    return default


def _write_report(
    *,
    workspace_root: Path,
    report_path_raw: object,
    payload: Mapping[str, object],
) -> tuple[str | None, str | None]:
    report_rel = _coerce_rel_path(report_path_raw, default=_DEFAULT_REPORT_PATH)
    destination = workspace_root / report_rel
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        return None, f"failed to write schema report {report_rel}: {exc}"

    return report_rel, None


def _relative_to_workspace(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def _timeout_result(
    checker: SchemaChecker,
    covered_constraint_ids: tuple[str, ...],
    timeout_seconds: float,
) -> CheckResult:
    return CheckResult(
        status=CheckStatus.TIMEOUT,
        violations=(
            Violation(
                constraint_id=covered_constraint_ids[0] if covered_constraint_ids else "UNMAPPED",
                code="schema.timeout",
                message=f"schema checker timed out after {timeout_seconds:.3f}s",
            ),
        ),
        covered_constraint_ids=covered_constraint_ids,
        tool_versions={
            "schema_checker": "builtin-1",
            "pyyaml": str(getattr(yaml, "__version__", "unknown")),
        },
        artifact_paths=(),
        logs_path=None,
        duration_ms=0,
        metadata={"timeout_seconds": timeout_seconds},
        checker_id=checker.checker_id,
        stage=checker.stage,
    )


__all__ = ["SchemaChecker", "ValidationIssue"]
