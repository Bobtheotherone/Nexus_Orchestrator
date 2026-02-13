"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/documentation_checker.py
Last updated: 2026-02-11

Purpose
- Enforce documentation constraints (e.g., public API changes require docs updates) and validate documentation structure.

What should be included in this file
- A DocumentationChecker implementing BaseChecker.
- Heuristics for detecting public API surface changes (language/stack dependent; pluggable).
- Rules for required docs updates (README/module docs/ADRs) when contracts or public APIs change.
- Optional doc build/validation hooks (mkdocs, mdbook, Sphinx) — pluggable, stack-specific.

Functional requirements
- Must support rules expressed as constraints (parameters define which files/APIs trigger which docs).
- Must provide evidence artifacts: list of detected API changes and the doc files checked.
- Must support ‘doc-only’ work items where code changes are prohibited but docs changes required.

Non-functional requirements
- Keep heuristics conservative; false positives should be resolved via parameters/allowlists, not disabling the checker.
- Deterministic and offline-capable (no network needed).

Notes
- Baseline constraint CON-DOC-0001 references documentation_checker.
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from nexus_orchestrator.verification_plane.checkers.base import (
    BaseChecker,
    CheckerContext,
    CheckResult,
    CheckStatus,
    JSONValue,
    Violation,
    register_builtin_checker,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import (
    checker_parameters,
    extract_constraint_ids,
    first_constraint_id,
)


@register_builtin_checker("documentation_checker")
class DocumentationChecker(BaseChecker):
    """Conservative docs-change policy checker."""

    checker_id = "documentation_checker"
    stage = "documentation"
    covered_constraint_ids = ("CON-DOC-0001",)

    async def check(self, context: CheckerContext) -> CheckResult:
        started = time.monotonic()
        params = checker_parameters(context, self.checker_id)
        timeout_seconds = _to_positive_float(params.get("timeout_seconds"), default=30.0)
        covered_constraint_ids = extract_constraint_ids(
            context,
            params,
            self.covered_constraint_ids,
        )
        if params.get("force_timeout") is True:
            return _timeout_result(
                checker_id=self.checker_id,
                stage=self.stage,
                constraint_ids=covered_constraint_ids,
                timeout_seconds=timeout_seconds,
            )
        primary_constraint = first_constraint_id(covered_constraint_ids)

        if _timed_out(started, timeout_seconds):
            return _timeout_result(
                checker_id=self.checker_id,
                stage=self.stage,
                constraint_ids=covered_constraint_ids,
                timeout_seconds=timeout_seconds,
            )

        changed_files = _resolve_changed_files(params.get("changed_files"))
        api_globs = _resolve_globs(params.get("api_globs"), default=("src/**/*.py",))
        doc_globs = _resolve_globs(params.get("doc_globs"), default=("docs/**/*.md", "README.md"))
        doc_only = (
            bool(params.get("doc_only")) if isinstance(params.get("doc_only"), bool) else False
        )

        api_changes = tuple(sorted(path for path in changed_files if _matches_any(path, api_globs)))
        doc_changes = tuple(sorted(path for path in changed_files if _matches_any(path, doc_globs)))

        violations: list[Violation] = []

        if doc_only and api_changes:
            for path in api_changes:
                if _timed_out(started, timeout_seconds):
                    return _timeout_result(
                        checker_id=self.checker_id,
                        stage=self.stage,
                        constraint_ids=covered_constraint_ids,
                        timeout_seconds=timeout_seconds,
                    )
                violations.append(
                    Violation(
                        constraint_id=primary_constraint,
                        code="documentation.doc_only_violation",
                        message="doc-only work item cannot change code/API files",
                        path=path,
                    )
                )

        if api_changes and not doc_changes:
            violations.append(
                Violation(
                    constraint_id=primary_constraint,
                    code="documentation.missing_updates",
                    message="public/API changes detected without corresponding documentation updates",
                )
            )

        status = CheckStatus.PASS if not violations else CheckStatus.FAIL
        metadata: dict[str, JSONValue] = {
            "changed_files": list(changed_files),
            "api_changes": list(api_changes),
            "doc_changes": list(doc_changes),
            "api_globs": list(api_globs),
            "doc_globs": list(doc_globs),
            "doc_only": doc_only,
            "timeout_seconds": timeout_seconds,
        }

        return CheckResult(
            status=status,
            violations=tuple(violations),
            covered_constraint_ids=covered_constraint_ids,
            tool_versions={"documentation_checker": "builtin-1"},
            artifact_paths=(),
            logs_path=None,
            duration_ms=0,
            metadata=metadata,
            checker_id=self.checker_id,
            stage=self.stage,
        )


def _resolve_changed_files(raw: object) -> tuple[str, ...]:
    values: set[str] = set()
    if isinstance(raw, str) and raw.strip():
        normalized = _normalize_rel_path(raw)
        if normalized is not None:
            values.add(normalized)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                normalized = _normalize_rel_path(item)
                if normalized is not None:
                    values.add(normalized)
            elif isinstance(item, Mapping):
                path = item.get("path")
                if isinstance(path, str) and path.strip():
                    normalized = _normalize_rel_path(path)
                    if normalized is not None:
                        values.add(normalized)
    return tuple(sorted(values))


def _resolve_globs(raw: object, *, default: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    if isinstance(raw, str) and raw.strip():
        values.add(raw.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, str) and item.strip():
                values.add(item.strip())
    if not values:
        values.update(default)
    return tuple(sorted(values))


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _normalize_rel_path(path: str) -> str | None:
    candidate = path.replace("\\", "/").strip()
    if not candidate:
        return None
    pure = PurePosixPath(candidate)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    cleaned = [part for part in pure.parts if part not in {"", "."}]
    if not cleaned:
        return None
    return "/".join(cleaned)


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
    *,
    checker_id: str,
    stage: str,
    constraint_ids: tuple[str, ...],
    timeout_seconds: float,
) -> CheckResult:
    return CheckResult(
        status=CheckStatus.TIMEOUT,
        violations=(
            Violation(
                constraint_id=constraint_ids[0] if constraint_ids else "UNMAPPED",
                code="documentation.timeout",
                message=f"documentation checker timed out after {timeout_seconds:.3f}s",
            ),
        ),
        covered_constraint_ids=constraint_ids,
        tool_versions={"documentation_checker": "builtin-1"},
        artifact_paths=(),
        logs_path=None,
        duration_ms=0,
        metadata={"timeout_seconds": timeout_seconds},
        checker_id=checker_id,
        stage=stage,
    )


__all__ = ["DocumentationChecker"]
