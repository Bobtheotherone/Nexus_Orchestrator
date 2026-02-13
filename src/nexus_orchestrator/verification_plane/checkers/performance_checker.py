"""
Performance Checker — verification stage.

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

from nexus_orchestrator.verification_plane.checkers.base import (
    CheckerContext,
    CheckResult,
    CheckStatus,
    Violation,
    register_builtin_checker,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import (
    CommandChecker,
    checker_parameters,
)


@register_builtin_checker("performance_checker")
class PerformanceChecker(CommandChecker):
    """Performance checker with deterministic duration threshold enforcement."""

    checker_id = "performance_checker"
    stage = "performance"
    tool_name = "pytest"
    covered_constraint_ids: tuple[str, ...] = ()

    default_command = ("pytest", "tests/smoke", "-q")
    default_version_command = ("pytest", "--version")
    default_timeout_seconds = 300.0

    async def check(self, context: CheckerContext) -> CheckResult:
        result = await super().check(context)
        params = checker_parameters(context, self.checker_id)
        max_duration = params.get("max_duration_seconds")

        if result.status is not CheckStatus.PASS:
            return result
        if isinstance(max_duration, bool) or not isinstance(max_duration, (int, float)):
            return result
        if float(max_duration) <= 0:
            return result

        duration_seconds = result.duration_ms / 1000.0
        threshold = float(max_duration)
        if duration_seconds <= threshold:
            return result

        violation = Violation(
            constraint_id=result.covered_constraint_ids[0]
            if result.covered_constraint_ids
            else "UNMAPPED",
            code="performance.budget_exceeded",
            message=(
                "performance budget exceeded: "
                f"duration={duration_seconds:.3f}s max={threshold:.3f}s"
            ),
        )

        metadata = dict(result.metadata)
        metadata["duration_seconds"] = round(duration_seconds, 6)
        metadata["max_duration_seconds"] = threshold

        return CheckResult(
            status=CheckStatus.FAIL,
            violations=(*result.violations, violation),
            covered_constraint_ids=result.covered_constraint_ids,
            tool_versions=result.tool_versions,
            artifact_paths=result.artifact_paths,
            logs_path=result.logs_path,
            duration_ms=result.duration_ms,
            metadata=metadata,
            checker_id=result.checker_id,
            stage=result.stage,
        )


__all__ = ["PerformanceChecker"]
