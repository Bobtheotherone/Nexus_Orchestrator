from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.verification_plane.checkers import (
    BuildChecker,
    CheckerContext,
    CheckStatus,
    LocalSubprocessExecutor,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.asyncio
async def test_build_checker_runs_real_compileall_offline(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    source_dir = workspace / "src"
    source_dir.mkdir(parents=True)
    (source_dir / "ok.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")

    checker = BuildChecker()
    context = CheckerContext(
        workspace_path=str(workspace),
        config={
            "command": ["python", "-m", "compileall", "-q", "src"],
            "constraint_ids": ["CON-COR-0001"],
        },
        command_executor=LocalSubprocessExecutor(default_timeout_seconds=10.0),
    )

    result = await checker.check(context)

    assert result.status is CheckStatus.PASS
    assert result.covered_constraint_ids == ("CON-COR-0001",)
    assert "python" in result.tool_versions
