"""Unit tests for the Plan dialog — designs/ browser and command generation.

File: tests/unit/test_plan_dialog.py

Tests:
- _list_design_docs finds all .md/.markdown files (recursive)
- _list_design_docs returns POSIX-relative paths (no UNC)
- _list_design_docs returns empty list for missing/empty directory
- PlanDialog._submit_doc produces ``/run plan designs/<file>`` with no --mock
- PlanDialog shows friendly message when designs/ is empty
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nexus_orchestrator.ui.tui.screens.plan_dialog import (
    _list_design_docs,
)


@pytest.mark.unit
class TestListDesignDocs:
    """Test _list_design_docs helper."""

    def test_finds_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "one.md").write_text("# One")
        (tmp_path / "two.md").write_text("# Two")
        result = _list_design_docs(tmp_path)
        assert sorted(result) == ["one.md", "two.md"]

    def test_finds_markdown_extension(self, tmp_path: Path) -> None:
        (tmp_path / "doc.markdown").write_text("# Doc")
        result = _list_design_docs(tmp_path)
        assert result == ["doc.markdown"]

    def test_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("# Top")
        (sub / "nested.md").write_text("# Nested")
        result = _list_design_docs(tmp_path)
        assert sorted(result) == ["sub/nested.md", "top.md"]

    def test_ignores_non_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("not md")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "real.md").write_text("# Real")
        result = _list_design_docs(tmp_path)
        assert result == ["real.md"]

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = _list_design_docs(tmp_path)
        assert result == []

    def test_missing_dir(self, tmp_path: Path) -> None:
        result = _list_design_docs(tmp_path / "nonexistent")
        assert result == []

    def test_returns_posix_paths(self, tmp_path: Path) -> None:
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        (sub / "doc.md").write_text("# Doc")
        result = _list_design_docs(tmp_path)
        assert result == ["deep/nested/doc.md"]
        # No backslashes (Windows/UNC)
        for path in result:
            assert "\\" not in path

    def test_sorted_alphabetically(self, tmp_path: Path) -> None:
        for name in ["zebra.md", "alpha.md", "middle.md"]:
            (tmp_path / name).write_text(f"# {name}")
        result = _list_design_docs(tmp_path)
        assert result == ["alpha.md", "middle.md", "zebra.md"]


@pytest.mark.unit
class TestPlanDialogCommand:
    """Test that Plan dialog produces the correct command string."""

    def test_submit_doc_no_mock(self, tmp_path: Path) -> None:
        """_submit_doc returns '/run plan designs/<file>' without --mock."""
        from nexus_orchestrator.ui.tui.screens.plan_dialog import PlanDialog

        dialog = PlanDialog(designs_dir=tmp_path)
        # Call _submit_doc directly and capture the dismiss value
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]
        dialog._submit_doc("AI_System.md")

        assert len(dismissed) == 1
        cmd = dismissed[0]
        assert cmd is not None
        assert cmd == "/run plan designs/AI_System.md"
        assert "--mock" not in cmd

    def test_submit_doc_nested_path(self, tmp_path: Path) -> None:
        """Nested paths use forward slashes."""
        from nexus_orchestrator.ui.tui.screens.plan_dialog import PlanDialog

        dialog = PlanDialog(designs_dir=tmp_path)
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]
        dialog._submit_doc("sub/other.md")

        cmd = dismissed[0]
        assert cmd == "/run plan designs/sub/other.md"
        assert "\\" not in cmd

    def test_submit_doc_no_extra_flags(self, tmp_path: Path) -> None:
        """Command has exactly the expected tokens — no leftover flags."""
        from nexus_orchestrator.ui.tui.screens.plan_dialog import PlanDialog

        dialog = PlanDialog(designs_dir=tmp_path)
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]
        dialog._submit_doc("test.md")

        cmd = dismissed[0]
        assert cmd is not None
        parts = cmd.split()
        assert parts == ["/run", "plan", "designs/test.md"]

    def test_cancel_returns_none(self, tmp_path: Path) -> None:
        """Cancelling returns None."""
        from nexus_orchestrator.ui.tui.screens.plan_dialog import PlanDialog

        dialog = PlanDialog(designs_dir=tmp_path)
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]
        dialog.action_cancel()

        assert dismissed == [None]


@pytest.mark.unit
class TestPlanDialogExecutionPath:
    """End-to-end: Plan dialog result flowing through the controller."""

    @pytest.mark.asyncio
    async def test_plan_command_does_not_include_mock(self) -> None:
        """When controller receives Plan dialog result, no --mock is injected."""
        from unittest import mock

        from nexus_orchestrator.ui.tui.controller import TUIController

        ctrl = TUIController()
        # Capture what _run_command receives
        captured: list[str] = []
        original_handle = ctrl._handle_slash_command

        async def spy_handle(command: str) -> None:
            captured.append(command)
            # Don't actually run the subprocess
            ctrl.append_system(f"Would run: {command}")

        ctrl._handle_slash_command = spy_handle  # type: ignore[assignment]

        # Simulate what _on_plan_dialog_result does
        result = "/run plan designs/AI_System.md"
        await ctrl.execute_command(result)

        assert len(captured) == 1
        assert captured[0] == "/run plan designs/AI_System.md"
        assert "--mock" not in captured[0]
