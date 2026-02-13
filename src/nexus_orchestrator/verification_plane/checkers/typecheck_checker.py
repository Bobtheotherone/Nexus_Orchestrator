"""
Typecheck Checker — verification stage.

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


@register_builtin_checker("typecheck_checker")
class TypecheckChecker(CommandChecker):
    """Static type checker with strict-mode support."""

    checker_id = "typecheck_checker"
    stage = "typecheck"
    tool_name = "mypy"
    covered_constraint_ids = ("CON-STY-0003",)

    default_command = ("mypy", "src/nexus_orchestrator")
    default_version_command = ("mypy", "--version")
    default_timeout_seconds = 120.0

    def build_command(self, params: Mapping[str, object]) -> tuple[str, ...]:
        command = params.get("command")
        if command is not None:
            return super().build_command(params)

        target_raw = params.get("target")
        target = (
            target_raw.strip()
            if isinstance(target_raw, str) and target_raw.strip()
            else "src/nexus_orchestrator"
        )
        mode_raw = params.get("mode")
        mode = mode_raw.strip().lower() if isinstance(mode_raw, str) else "default"

        if mode == "strict":
            return ("mypy", "--strict", target)
        return ("mypy", target)


__all__ = ["TypecheckChecker"]
