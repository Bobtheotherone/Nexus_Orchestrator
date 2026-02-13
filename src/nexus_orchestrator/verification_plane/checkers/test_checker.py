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

from typing import TYPE_CHECKING

from nexus_orchestrator.verification_plane.checkers.base import register_builtin_checker
from nexus_orchestrator.verification_plane.checkers.build_checker import CommandChecker

if TYPE_CHECKING:
    from collections.abc import Mapping


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
            return (
                "pytest",
                "tests/unit",
                "--cov",
                target,
                "--cov-report",
                "term-missing",
                "-q",
            )
        return ("pytest", "tests/unit", "-q")


__all__ = ["TestChecker"]
