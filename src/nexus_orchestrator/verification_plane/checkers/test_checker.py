"""
Test Checker — verification stage.

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

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from nexus_orchestrator.verification_plane.checkers.base import register_builtin_checker
from nexus_orchestrator.verification_plane.checkers.build_checker import CommandChecker


@register_builtin_checker("test_checker")
class TestChecker(CommandChecker):
    """Parameterized test checker for unit/integration/coverage runs."""

    checker_id = "test_checker"
    stage = "test"
    tool_name = "pytest"
    covered_constraint_ids = ("CON-COR-0002", "CON-COR-0003")

    default_command = ("pytest", "tests/unit", "-q")
    default_version_command = ("pytest", "--version")
    default_timeout_seconds = 300.0

    def build_command(self, params: Mapping[str, object]) -> tuple[str, ...]:
        command = params.get("command")
        if command is not None:
            return super().build_command(params)

        test_type_raw = params.get("test_type")
        test_type = test_type_raw.strip().lower() if isinstance(test_type_raw, str) else "unit"
        mode_raw = params.get("mode")
        mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else "full"

        if test_type == "integration":
            return ("pytest", "tests/integration", "-q")
        if test_type == "smoke":
            return ("pytest", "tests/smoke", "-q")
        if test_type == "coverage":
            target_raw = params.get("coverage_target")
            target = (
                target_raw.strip()
                if isinstance(target_raw, str) and target_raw.strip()
                else "src/nexus_orchestrator"
            )
            min_coverage = _coverage_threshold(params.get("min_coverage_percent"))
            return (
                "pytest",
                "tests/unit",
                "--cov",
                target,
                "--cov-report",
                "term-missing",
                "--cov-fail-under",
                f"{min_coverage:.2f}",
                "-q",
            )

        if mode == "incremental":
            selected_tests = _map_changed_files_to_tests(
                params.get("changed_files"),
                workspace_root=_workspace_root_from_params(params),
            )
            if selected_tests:
                return ("pytest", *selected_tests, "-q")
        return ("pytest", "tests/unit", "-q")


def _coverage_threshold(raw: object) -> float:
    if isinstance(raw, bool):
        return 80.0
    if isinstance(raw, (int, float)):
        threshold = float(raw)
        if threshold <= 0:
            return 80.0
        if threshold > 100:
            return 100.0
        return threshold
    return 80.0


def _workspace_root_from_params(params: Mapping[str, object]) -> Path | None:
    raw_workspace_path = params.get("__workspace_path")
    if isinstance(raw_workspace_path, str) and raw_workspace_path.strip():
        return Path(raw_workspace_path).resolve(strict=False)
    return None


def _path_exists(path: str, *, workspace_root: Path | None) -> bool:
    if workspace_root is None:
        return True
    return (workspace_root / path).resolve(strict=False).is_file()


def _map_changed_files_to_tests(
    raw: object,
    *,
    workspace_root: Path | None = None,
) -> tuple[str, ...]:
    changed_paths = _coerce_changed_paths(raw)
    selected: set[str] = set()
    for path in changed_paths:
        if (
            path.startswith("tests/")
            and path.endswith(".py")
            and _path_exists(path, workspace_root=workspace_root)
        ):
            selected.add(path)
            continue

        if path.startswith("src/") and path.endswith(".py"):
            module_stem = PurePosixPath(path).stem
            for candidate in (f"tests/unit/test_{module_stem}.py", f"tests/test_{module_stem}.py"):
                if _path_exists(candidate, workspace_root=workspace_root):
                    selected.add(candidate)

    return tuple(sorted(selected))


def _coerce_changed_paths(raw: object) -> tuple[str, ...]:
    if isinstance(raw, str):
        values: Sequence[object] = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values = raw
    else:
        values = ()

    normalized: set[str] = set()
    for item in values:
        candidate: str | None = None
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, Mapping):
            raw_kind = item.get("kind")
            if isinstance(raw_kind, str) and raw_kind.strip().lower() == "deleted":
                continue
            maybe = item.get("path")
            if isinstance(maybe, str):
                candidate = maybe
        if candidate is None:
            continue

        clean = candidate.replace("\\", "/").strip()
        if not clean or clean.startswith("/") or ".." in PurePosixPath(clean).parts:
            continue
        normalized.add(clean)

    return tuple(sorted(normalized))


__all__ = ["TestChecker"]
