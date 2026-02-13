"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/reliability_checker.py
Last updated: 2026-02-11

Purpose
- Enforce reliability constraints for orchestrated projects and for the orchestrator itself (timeouts, retries, circuit breakers, idempotency rules).

What should be included in this file
- A ReliabilityChecker implementing BaseChecker.
- Static checks for presence of timeout/retry policies around external calls (stack-specific; pluggable).
- Optional runtime checks via targeted tests or lints (e.g., ensuring async calls have timeouts).
- Config-driven rules for what counts as an 'external call' boundary.

Functional requirements
- Must validate that required reliability policies exist for designated call sites/modules.
- Must emit actionable evidence (file+line references, missing policy type).
- Must support waivers that are explicit and time-bounded (recorded in evidence ledger).

Non-functional requirements
- Prefer static/deterministic analysis where possible.
- Keep runtime overhead low; reliability checks should not require long-running integration environments by default.

Notes
- Baseline constraint CON-REL-0001 references reliability_checker.
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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


@register_builtin_checker("reliability_checker")
class ReliabilityChecker(BaseChecker):
    """Static timeout/retry policy checker with explicit time-bounded waivers."""

    checker_id = "reliability_checker"
    stage = "reliability"
    covered_constraint_ids = ("CON-REL-0001",)

    async def check(self, context: CheckerContext) -> CheckResult:
        started = time.monotonic()
        params = checker_parameters(context, self.checker_id)
        timeout_seconds = _resolve_timeout(params.get("timeout_seconds"), default=30.0)
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

        workspace_root = Path(context.workspace_path).resolve()
        include_globs = _resolve_globs(params.get("include_globs"), default=("src/**/*.py",))
        exclude_globs = _resolve_globs(params.get("exclude_paths"), default=())
        external_markers = _resolve_markers(
            params.get("external_call_markers"),
            default=("requests.", "httpx.", "subprocess.", "urllib."),
        )
        timeout_markers = _resolve_markers(
            params.get("timeout_markers"),
            default=("timeout=", "run_with_timeout(", "asyncio.wait_for("),
        )
        retry_markers = _resolve_markers(
            params.get("retry_markers"),
            default=("retry", "retries=", "backoff"),
        )
        now_epoch = _resolve_now_epoch(params.get("now_epoch"))
        waivers = _resolve_waivers(params.get("waivers"), now_epoch=now_epoch)

        violations: list[Violation] = []
        waiver_hits: list[dict[str, JSONValue]] = []

        for path in _collect_files(workspace_root, include_globs, exclude_globs):
            if _timed_out(started, timeout_seconds):
                return _timeout_result(
                    checker_id=self.checker_id,
                    stage=self.stage,
                    constraint_ids=covered_constraint_ids,
                    timeout_seconds=timeout_seconds,
                )
            rel = _relative_to_workspace(path, workspace_root)
            if rel is None:
                continue

            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for index, line in enumerate(lines, start=1):
                if not any(marker in line for marker in external_markers):
                    continue

                neighborhood = "\n".join(lines[index - 1 : min(index + 2, len(lines))]).lower()
                missing_timeout = not any(
                    marker.lower() in neighborhood for marker in timeout_markers
                )
                missing_retry = not any(marker.lower() in neighborhood for marker in retry_markers)

                if not missing_timeout and not missing_retry:
                    continue

                waiver = _find_waiver(waivers, path=rel, line=index)
                if waiver is not None:
                    waiver_hits.append(
                        {
                            "path": rel,
                            "line": index,
                            "approved_by": waiver.approved_by,
                            "reason": waiver.reason,
                            "expires_at": waiver.expires_at,
                        }
                    )
                    violations.append(
                        Violation(
                            constraint_id=primary_constraint,
                            code="reliability.waived",
                            message="reliability policy gap waived by approved record",
                            path=rel,
                            line=index,
                            severity="warning",
                        )
                    )
                    continue

                if missing_timeout:
                    violations.append(
                        Violation(
                            constraint_id=primary_constraint,
                            code="reliability.missing_timeout",
                            message="external call appears to lack timeout policy",
                            path=rel,
                            line=index,
                        )
                    )
                if missing_retry:
                    violations.append(
                        Violation(
                            constraint_id=primary_constraint,
                            code="reliability.missing_retry",
                            message="external call appears to lack retry/backoff policy",
                            path=rel,
                            line=index,
                        )
                    )

        blocking = any(item.severity.lower() == "error" for item in violations)
        status = CheckStatus.FAIL if blocking else CheckStatus.PASS

        waiver_hits_json: list[JSONValue] = [entry for entry in waiver_hits]
        metadata: dict[str, JSONValue] = {
            "include_globs": list(include_globs),
            "exclude_globs": list(exclude_globs),
            "waiver_hits": waiver_hits_json,
            "scanned_files": [
                rel
                for rel in (
                    _relative_to_workspace(path, workspace_root)
                    for path in _collect_files(workspace_root, include_globs, exclude_globs)
                )
                if rel is not None
            ],
            "timeout_seconds": timeout_seconds,
        }
        duration_ms = max(int(round((time.monotonic() - started) * 1000)), 0)

        return CheckResult(
            status=status,
            violations=tuple(violations),
            covered_constraint_ids=covered_constraint_ids,
            tool_versions={"reliability_checker": "builtin-1"},
            artifact_paths=(),
            logs_path=None,
            command_lines=(),
            duration_ms=duration_ms,
            metadata=metadata,
            checker_id=self.checker_id,
            stage=self.stage,
        )


@dataclass(frozen=True, slots=True)
class WaiverRecord:
    path: str
    line: int
    reason: str
    approved_by: str
    expires_at: int


def _collect_files(
    workspace_root: Path, include_globs: Sequence[str], exclude_globs: Sequence[str]
) -> list[Path]:
    candidates = sorted(path for path in workspace_root.rglob("*") if path.is_file())
    filtered: list[Path] = []
    for path in candidates:
        rel = _relative_to_workspace(path, workspace_root)
        if rel is None:
            continue
        if not any(fnmatch.fnmatchcase(rel, pattern) for pattern in include_globs):
            continue
        if any(fnmatch.fnmatchcase(rel, pattern) for pattern in exclude_globs):
            continue
        filtered.append(path)
    return filtered


def _resolve_globs(raw: object, *, default: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    if isinstance(raw, str) and raw.strip():
        values.add(raw.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values.update(item.strip() for item in raw if isinstance(item, str) and item.strip())
    if not values:
        values.update(default)
    return tuple(sorted(values))


def _resolve_markers(raw: object, *, default: Sequence[str]) -> tuple[str, ...]:
    values: set[str] = set()
    if isinstance(raw, str) and raw.strip():
        values.add(raw.strip())
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values.update(item.strip() for item in raw if isinstance(item, str) and item.strip())
    if not values:
        values.update(default)
    return tuple(sorted(values))


def _resolve_now_epoch(raw: object) -> int:
    if isinstance(raw, int) and raw >= 0:
        return raw
    return 0


def _resolve_timeout(raw: object, *, default: float) -> float:
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        parsed = float(raw)
        if parsed > 0:
            return parsed
    return default


def _resolve_waivers(raw: object, *, now_epoch: int) -> tuple[WaiverRecord, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()

    parsed: list[WaiverRecord] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        path_raw = item.get("path")
        line_raw = item.get("line")
        reason_raw = item.get("reason")
        approved_by_raw = item.get("approved_by")
        expires_raw = item.get("expires_at")

        if not isinstance(path_raw, str) or not path_raw.strip():
            continue
        normalized_path = _normalize_rel_path(path_raw)
        if normalized_path is None:
            continue
        if not isinstance(line_raw, int) or line_raw <= 0:
            continue
        if not isinstance(reason_raw, str) or not reason_raw.strip():
            continue
        if not isinstance(approved_by_raw, str) or not approved_by_raw.strip():
            continue
        if not isinstance(expires_raw, int) or expires_raw <= now_epoch:
            continue

        parsed.append(
            WaiverRecord(
                path=normalized_path,
                line=line_raw,
                reason=reason_raw.strip(),
                approved_by=approved_by_raw.strip(),
                expires_at=expires_raw,
            )
        )

    return tuple(sorted(parsed, key=lambda item: (item.path, item.line)))


def _find_waiver(waivers: Sequence[WaiverRecord], *, path: str, line: int) -> WaiverRecord | None:
    for waiver in waivers:
        if waiver.path == path and waiver.line == line:
            return waiver
    return None


def _relative_to_workspace(path: Path, workspace_root: Path) -> str | None:
    try:
        rel = path.resolve(strict=False).relative_to(workspace_root)
    except ValueError:
        return None
    normalized = _normalize_rel_path(rel.as_posix())
    return normalized


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
                code="reliability.timeout",
                message=f"reliability checker timed out after {timeout_seconds:.3f}s",
            ),
        ),
        covered_constraint_ids=constraint_ids,
        tool_versions={"reliability_checker": "builtin-1"},
        artifact_paths=(),
        logs_path=None,
        duration_ms=0,
        metadata={"timeout_seconds": timeout_seconds},
        checker_id=checker_id,
        stage=stage,
    )


__all__ = ["ReliabilityChecker"]
