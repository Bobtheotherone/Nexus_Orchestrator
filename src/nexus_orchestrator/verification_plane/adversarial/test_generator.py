"""Deterministic offline adversarial test generation checker."""

from __future__ import annotations

import ast
import json
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
    Violation,
    normalize_artifact_paths,
    register_builtin_checker,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import (
    checker_parameters,
    extract_constraint_ids,
    first_constraint_id,
)

_ADVERSARIAL_CONSTRAINT: Final[str] = "CON-ADV-0001"
_DEFAULT_REPORT_PATH: Final[str] = (
    ".nexus_orchestrator/checker_artifacts/adversarial_test_plan.json"
)
_DEFAULT_SCAN_PATHS: Final[tuple[str, ...]] = ("src",)
_DEFAULT_EXCLUDES: Final[tuple[str, ...]] = (
    ".git/**",
    ".venv/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    "__pycache__/**",
)
_NUMERIC_ARG_HINTS: Final[frozenset[str]] = frozenset(
    {"count", "size", "limit", "index", "offset", "timeout", "retries", "port"}
)
_PATH_ARG_HINTS: Final[frozenset[str]] = frozenset(
    {"path", "file", "filepath", "filename", "uri", "url", "directory"}
)
_SEQUENCE_ARG_HINTS: Final[frozenset[str]] = frozenset({"items", "values", "entries", "list"})
_STRING_ARG_HINTS: Final[frozenset[str]] = frozenset(
    {"name", "text", "message", "token", "secret", "password"}
)


@dataclass(frozen=True, slots=True)
class GeneratedAdversarialTarget:
    path: str
    function: str
    line: int
    case_ids: tuple[str, ...]

    @property
    def generated_case_count(self) -> int:
        return len(self.case_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "function": self.function,
            "line": self.line,
            "case_ids": list(self.case_ids),
            "generated_case_count": self.generated_case_count,
        }


@register_builtin_checker("adversarial/test_generator")
class AdversarialTestGenerator(BaseChecker):
    """Offline adversarial planner using deterministic AST heuristics."""

    checker_id = "adversarial/test_generator"
    stage = "adversarial_tests"
    covered_constraint_ids = (_ADVERSARIAL_CONSTRAINT,)

    async def check(self, context: CheckerContext) -> CheckResult:
        started = time.monotonic()
        params = checker_parameters(context, self.checker_id)
        covered_constraint_ids = extract_constraint_ids(
            context,
            params,
            self.covered_constraint_ids,
        )
        timeout_seconds = _to_positive_float(params.get("timeout_seconds"), default=30.0)
        max_file_size_bytes = _to_positive_int(params.get("max_file_size_bytes"), default=524_288)
        max_files = _to_positive_int(params.get("max_files"), default=200)
        minimum_targets = _to_non_negative_int(params.get("minimum_targets"), default=1)
        minimum_case_count = _to_non_negative_int(params.get("minimum_case_count"), default=3)

        workspace_root = Path(context.workspace_path).resolve()
        raw_scan_paths = params.get("scan_paths")
        raw_excludes = params.get("exclude_paths")
        scan_paths = _resolve_scan_paths(raw_scan_paths)
        excludes = _resolve_excludes(raw_excludes)
        source_files = _collect_source_files(
            workspace_root=workspace_root,
            scan_paths=scan_paths,
            excludes=excludes,
            max_files=max_files,
        )

        violations: list[Violation] = []
        targets: list[GeneratedAdversarialTarget] = []
        timed_out = False
        scanned_count = 0

        for file_path in source_files:
            if (time.monotonic() - started) > timeout_seconds:
                timed_out = True
                violations.append(
                    Violation(
                        constraint_id=first_constraint_id(covered_constraint_ids),
                        code="adversarial.timeout",
                        message=f"adversarial planning timed out after {timeout_seconds:.3f}s",
                    )
                )
                break

            rel_path = _relative_safe(file_path, workspace_root)
            if rel_path is None:
                continue

            try:
                file_stat = file_path.stat()
            except OSError:
                continue
            if file_stat.st_size > max_file_size_bytes:
                continue

            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            scanned_count += 1
            targets.extend(_extract_targets(source=source, relative_path=rel_path))

        targets = sorted(targets, key=lambda item: (item.path, item.line, item.function))
        generated_case_count = sum(item.generated_case_count for item in targets)

        if not timed_out and len(targets) < minimum_targets:
            violations.append(
                Violation(
                    constraint_id=first_constraint_id(covered_constraint_ids),
                    code="adversarial.targets_insufficient",
                    message=(
                        "insufficient adversarial targets generated "
                        f"({len(targets)} < {minimum_targets})"
                    ),
                )
            )

        if not timed_out and generated_case_count < minimum_case_count:
            violations.append(
                Violation(
                    constraint_id=first_constraint_id(covered_constraint_ids),
                    code="adversarial.cases_insufficient",
                    message=(
                        "insufficient adversarial cases generated "
                        f"({generated_case_count} < {minimum_case_count})"
                    ),
                )
            )

        report_payload: dict[str, object] = {
            "checker_id": self.checker_id,
            "mode": "offline_deterministic",
            "scan_paths": list(scan_paths),
            "exclude_paths": list(excludes),
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
            "scanned_file_count": scanned_count,
            "target_count": len(targets),
            "generated_case_count": generated_case_count,
            "minimum_targets": minimum_targets,
            "minimum_case_count": minimum_case_count,
            "evidence_tags": [
                "adversarial.offline",
                "adversarial.deterministic",
            ],
            "targets": [target.to_dict() for target in targets],
        }

        report_path, report_error = _write_report(
            workspace_root=workspace_root,
            report_path_raw=params.get("report_path"),
            payload=report_payload,
        )
        if report_error is not None:
            violations.append(
                Violation(
                    constraint_id=first_constraint_id(covered_constraint_ids),
                    code="adversarial.report_write_failed",
                    message=report_error,
                    severity="warning",
                )
            )

        status = CheckStatus.TIMEOUT if timed_out else _derive_status(violations)
        duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)

        return CheckResult(
            status=status,
            violations=tuple(violations),
            covered_constraint_ids=covered_constraint_ids,
            tool_versions={
                "adversarial/test_generator": "offline-ast-v1",
                "adversarial_planner": "builtin-1",
            },
            artifact_paths=normalize_artifact_paths((report_path,)) if report_path else (),
            logs_path=None,
            command_lines=(),
            duration_ms=duration_ms,
            metadata={
                "mode": "offline_deterministic",
                "timed_out": timed_out,
                "scanned_file_count": scanned_count,
                "target_count": len(targets),
                "generated_case_count": generated_case_count,
                "minimum_targets": minimum_targets,
                "minimum_case_count": minimum_case_count,
                "evidence_tags": ["adversarial.offline", "adversarial.deterministic"],
            },
            checker_id=self.checker_id,
            stage=self.stage,
        )


def _derive_status(violations: Sequence[Violation]) -> CheckStatus:
    if any(item.severity.lower() == "error" for item in violations):
        return CheckStatus.FAIL
    if any(item.severity.lower() == "warning" for item in violations):
        return CheckStatus.WARN
    return CheckStatus.PASS


def _resolve_scan_paths(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str) and raw.strip():
        return (raw.strip(),)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = tuple(
            sorted({item.strip() for item in raw if isinstance(item, str) and item.strip()})
        )
        if values:
            return values
    return _DEFAULT_SCAN_PATHS


def _resolve_excludes(raw: object) -> tuple[str, ...]:
    values = set(_DEFAULT_EXCLUDES)
    if isinstance(raw, str) and raw.strip():
        values.add(raw.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.add(item.strip())
    return tuple(sorted(values))


def _collect_source_files(
    *,
    workspace_root: Path,
    scan_paths: Sequence[str],
    excludes: Sequence[str],
    max_files: int,
) -> tuple[Path, ...]:
    files: set[Path] = set()
    for raw_path in scan_paths:
        candidate = workspace_root / raw_path
        if candidate.is_file() and candidate.suffix == ".py":
            files.add(candidate.resolve(strict=False))
            continue
        if not candidate.is_dir():
            continue
        for entry in candidate.rglob("*.py"):
            rel = _relative_safe(entry, workspace_root)
            if rel is None:
                continue
            if _matches_any(rel, excludes):
                continue
            files.add(entry.resolve(strict=False))
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break
    return tuple(sorted(files))


def _extract_targets(*, source: str, relative_path: str) -> list[GeneratedAdversarialTarget]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    targets: list[GeneratedAdversarialTarget] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        function_name = node.name.strip()
        if not function_name or function_name.startswith("_"):
            continue

        case_ids = _case_ids_for_function(node)
        if not case_ids:
            continue

        targets.append(
            GeneratedAdversarialTarget(
                path=relative_path,
                function=function_name,
                line=max(getattr(node, "lineno", 1), 1),
                case_ids=case_ids,
            )
        )

    return targets


def _case_ids_for_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    case_ids: list[str] = ["nominal_smoke", "exception_path"]
    args = [arg.arg.strip() for arg in node.args.args if arg.arg.strip() and arg.arg != "self"]
    if args:
        case_ids.extend(["null_inputs", "empty_inputs"])

    arg_hints = {arg.lower() for arg in args}
    if arg_hints & _NUMERIC_ARG_HINTS:
        case_ids.extend(["boundary_zero", "boundary_negative", "boundary_large"])
    if arg_hints & _PATH_ARG_HINTS:
        case_ids.extend(["path_traversal", "malformed_path"])
    if arg_hints & _SEQUENCE_ARG_HINTS:
        case_ids.extend(["duplicate_items", "oversized_sequence"])
    if arg_hints & _STRING_ARG_HINTS:
        case_ids.extend(["control_chars", "unicode_edge"])

    deduped = sorted(set(case_ids))
    return tuple(deduped)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _relative_safe(path: Path, workspace_root: Path) -> str | None:
    try:
        relative = path.resolve(strict=False).relative_to(workspace_root)
    except ValueError:
        return None
    return relative.as_posix()


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
        return None, f"failed to write adversarial report {report_rel}: {exc}"

    return report_rel, None


def _coerce_report_path(raw: object) -> str:
    if isinstance(raw, str) and raw.strip():
        candidate = raw.replace("\\", "/").strip()
        pure = PurePosixPath(candidate)
        if not pure.is_absolute() and ".." not in pure.parts:
            return candidate
    return _DEFAULT_REPORT_PATH


def _to_positive_float(raw: object, *, default: float) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        parsed = float(raw)
        if parsed > 0:
            return parsed
    return default


def _to_positive_int(raw: object, *, default: int) -> int:
    if isinstance(raw, int) and raw > 0:
        return raw
    return default


def _to_non_negative_int(raw: object, *, default: int) -> int:
    if isinstance(raw, int) and raw >= 0:
        return raw
    return default


__all__ = ["AdversarialTestGenerator", "GeneratedAdversarialTarget"]
