"""
Lint Checker — verification stage.

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
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

from nexus_orchestrator.verification_plane.checkers.base import (
    CommandResult,
    Violation,
    register_builtin_checker,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import CommandChecker

_FORMAT_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:Would reformat|would reformat)\s+(?P<path>.+)$"
)


@register_builtin_checker("lint_checker")
class LintChecker(CommandChecker):
    """Lint/format checker with parameterized command selection."""

    checker_id = "lint_checker"
    stage = "lint"
    tool_name = "ruff"
    covered_constraint_ids = ("CON-STY-0001", "CON-STY-0002")

    default_command = ("ruff", "check", "src")
    default_version_command = ("ruff", "--version")
    default_timeout_seconds = 90.0

    def build_command(self, params: Mapping[str, object]) -> tuple[str, ...]:
        command = params.get("command")
        if command is not None:
            return super().build_command(params)

        check_type_raw = params.get("check_type")
        check_type = check_type_raw.strip().lower() if isinstance(check_type_raw, str) else "lint"
        target_raw = params.get("target")
        target = target_raw.strip() if isinstance(target_raw, str) and target_raw.strip() else "src"

        if check_type == "format":
            return ("ruff", "format", "--check", target)
        return ("ruff", "check", target)

    def parse_failures(
        self,
        *,
        result: CommandResult,
        constraint_id: str,
    ) -> tuple[Violation, ...]:
        parsed = list(super().parse_failures(result=result, constraint_id=constraint_id))

        for line in f"{result.stdout}\n{result.stderr}".splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = _FORMAT_RE.match(stripped)
            if match is None:
                continue
            parsed.append(
                Violation(
                    constraint_id=constraint_id,
                    code="lint.format_mismatch",
                    message="file does not match formatter output",
                    path=match.group("path").replace("\\", "/").strip(),
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
            for item in parsed
        }
        return tuple(sorted(deduped.values(), key=lambda item: item.sort_key()))


__all__ = ["LintChecker"]
