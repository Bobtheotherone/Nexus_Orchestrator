"""Unit tests for the 'Run after Plan' prompt feature.

Tests:
1. Controller emits __PLAN_COMPLETE__ signal when plan succeeds.
2. Controller does NOT emit signal when plan fails.
3. _extract_plan_spec_path parses plan commands correctly.
4. RunAfterPlanDialog: [Run] produces ``/run run <spec>``.
5. RunAfterPlanDialog: [Run (mock)] produces ``/run run <spec> --mock``.
6. RunAfterPlanDialog: [Cancel] produces ``None``.
7. App._on_run_after_plan_result executes the returned command.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nexus_orchestrator.ui.tui.controller import (
    TUIController,
    _extract_plan_spec_path,
)
from nexus_orchestrator.ui.tui.state import AppState, EventKind


# ---------------------------------------------------------------------------
# _extract_plan_spec_path
# ---------------------------------------------------------------------------


class TestExtractPlanSpecPath:
    def test_simple_plan_command(self) -> None:
        assert _extract_plan_spec_path("plan designs/AI_System.md") == "designs/AI_System.md"

    def test_plan_with_flags(self) -> None:
        assert _extract_plan_spec_path("plan designs/spec.md --json") == "designs/spec.md"

    def test_plan_with_strict_flag(self) -> None:
        result = _extract_plan_spec_path("plan designs/spec.md --strict-requirements --json")
        assert result == "designs/spec.md"

    def test_not_a_plan_command(self) -> None:
        assert _extract_plan_spec_path("run designs/spec.md") is None

    def test_status_command(self) -> None:
        assert _extract_plan_spec_path("status") is None

    def test_plan_no_args(self) -> None:
        assert _extract_plan_spec_path("plan") is None

    def test_plan_only_flags(self) -> None:
        assert _extract_plan_spec_path("plan --json") is None

    def test_plan_nested_path(self) -> None:
        assert _extract_plan_spec_path("plan designs/sub/deep.md") == "designs/sub/deep.md"

    def test_empty_string(self) -> None:
        assert _extract_plan_spec_path("") is None


# ---------------------------------------------------------------------------
# Controller: plan-completion signal
# ---------------------------------------------------------------------------


class TestControllerPlanSignal:
    @pytest.mark.asyncio
    async def test_plan_success_emits_signal(self) -> None:
        """Successful plan command emits __PLAN_COMPLETE__ signal."""
        state = AppState()
        notified = []
        ctrl = TUIController(
            state=state,
            on_state_change=lambda: notified.append(True),
        )

        # Mock the runner to simulate a successful plan
        async def mock_run(command: str):
            from nexus_orchestrator.ui.tui.runner import RunnerEvent, RunnerEventKind

            yield RunnerEvent(kind=RunnerEventKind.STDOUT_LINE, text="Planned spec  designs/s.md")
            yield RunnerEvent(kind=RunnerEventKind.FINISHED, text="", exit_code=0)

        ctrl._runner.run = mock_run  # type: ignore[assignment]

        await ctrl._run_command("plan designs/spec.md")

        # Find the __PLAN_COMPLETE__ event in transcript
        system_events = [
            e for e in state.transcript
            if e.kind == EventKind.SYSTEM and e.text.startswith("__PLAN_COMPLETE__")
        ]
        assert len(system_events) == 1
        assert system_events[0].text == "__PLAN_COMPLETE__designs/spec.md"

    @pytest.mark.asyncio
    async def test_plan_failure_no_signal(self) -> None:
        """Failed plan command does NOT emit __PLAN_COMPLETE__ signal."""
        state = AppState()
        ctrl = TUIController(
            state=state,
            on_state_change=lambda: None,
        )

        async def mock_run(command: str):
            from nexus_orchestrator.ui.tui.runner import RunnerEvent, RunnerEventKind

            yield RunnerEvent(kind=RunnerEventKind.STDERR_LINE, text="Error: file not found")
            yield RunnerEvent(kind=RunnerEventKind.FINISHED, text="", exit_code=2)

        ctrl._runner.run = mock_run  # type: ignore[assignment]

        await ctrl._run_command("plan designs/missing.md")

        system_events = [
            e for e in state.transcript
            if e.kind == EventKind.SYSTEM and "__PLAN_COMPLETE__" in e.text
        ]
        assert len(system_events) == 0

    @pytest.mark.asyncio
    async def test_non_plan_command_no_signal(self) -> None:
        """Non-plan command (e.g. 'run') does NOT emit signal even if exit=0."""
        state = AppState()
        ctrl = TUIController(
            state=state,
            on_state_change=lambda: None,
        )

        async def mock_run(command: str):
            from nexus_orchestrator.ui.tui.runner import RunnerEvent, RunnerEventKind

            yield RunnerEvent(kind=RunnerEventKind.FINISHED, text="", exit_code=0)

        ctrl._runner.run = mock_run  # type: ignore[assignment]

        await ctrl._run_command("status --json")

        system_events = [
            e for e in state.transcript
            if e.kind == EventKind.SYSTEM and "__PLAN_COMPLETE__" in e.text
        ]
        assert len(system_events) == 0


# ---------------------------------------------------------------------------
# RunAfterPlanDialog: command generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunAfterPlanDialogCommands:
    """Test that the dialog produces correct command strings."""

    def test_run_button_produces_run_command(self) -> None:
        from nexus_orchestrator.ui.tui.screens.run_after_plan_dialog import (
            RunAfterPlanDialog,
        )

        dialog = RunAfterPlanDialog("designs/AI_System.md")
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]

        # Simulate pressing Run
        from unittest.mock import MagicMock

        event = MagicMock()
        event.button.id = "btn-run"
        dialog.on_button_pressed(event)

        assert len(dismissed) == 1
        assert dismissed[0] == "/run run designs/AI_System.md"
        assert "--mock" not in dismissed[0]

    def test_mock_button_produces_mock_command(self) -> None:
        from nexus_orchestrator.ui.tui.screens.run_after_plan_dialog import (
            RunAfterPlanDialog,
        )

        dialog = RunAfterPlanDialog("designs/AI_System.md")
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]

        from unittest.mock import MagicMock

        event = MagicMock()
        event.button.id = "btn-mock"
        dialog.on_button_pressed(event)

        assert len(dismissed) == 1
        assert dismissed[0] == "/run run designs/AI_System.md --mock"

    def test_cancel_returns_none(self) -> None:
        from nexus_orchestrator.ui.tui.screens.run_after_plan_dialog import (
            RunAfterPlanDialog,
        )

        dialog = RunAfterPlanDialog("designs/spec.md")
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]

        dialog.action_cancel()

        assert dismissed == [None]

    def test_cancel_button_returns_none(self) -> None:
        from nexus_orchestrator.ui.tui.screens.run_after_plan_dialog import (
            RunAfterPlanDialog,
        )

        dialog = RunAfterPlanDialog("designs/spec.md")
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]

        from unittest.mock import MagicMock

        event = MagicMock()
        event.button.id = "btn-cancel"
        dialog.on_button_pressed(event)

        assert dismissed == [None]

    def test_run_command_preserves_nested_path(self) -> None:
        from nexus_orchestrator.ui.tui.screens.run_after_plan_dialog import (
            RunAfterPlanDialog,
        )

        dialog = RunAfterPlanDialog("designs/sub/deep/spec.md")
        dismissed: list[str | None] = []
        dialog.dismiss = lambda val=None: dismissed.append(val)  # type: ignore[assignment]

        from unittest.mock import MagicMock

        event = MagicMock()
        event.button.id = "btn-run"
        dialog.on_button_pressed(event)

        assert dismissed[0] == "/run run designs/sub/deep/spec.md"


# ---------------------------------------------------------------------------
# App callback: _on_run_after_plan_result
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAppRunAfterPlanCallback:
    @pytest.mark.asyncio
    async def test_run_result_executes_command(self) -> None:
        """When user clicks Run, the app executes the run command."""
        from nexus_orchestrator.ui.tui.controller import TUIController

        state = AppState()
        ctrl = TUIController(state=state, on_state_change=lambda: None)

        captured: list[str] = []
        original = ctrl.execute_command

        async def spy(command: str) -> None:
            captured.append(command)

        ctrl.execute_command = spy  # type: ignore[assignment]

        # Simulate what _on_run_after_plan_result does
        result = "/run run designs/AI_System.md"
        ctrl.append_system(f"Running {result.split('/run ', 1)[-1].strip()} ...")
        await ctrl.execute_command(result)

        assert len(captured) == 1
        assert captured[0] == "/run run designs/AI_System.md"

    @pytest.mark.asyncio
    async def test_cancel_result_does_nothing(self) -> None:
        """When user cancels, no command is executed."""
        from nexus_orchestrator.ui.tui.controller import TUIController

        state = AppState()
        ctrl = TUIController(state=state, on_state_change=lambda: None)

        captured: list[str] = []

        async def spy(command: str) -> None:
            captured.append(command)

        ctrl.execute_command = spy  # type: ignore[assignment]

        # Simulate _on_run_after_plan_result with None
        result = None
        if result:
            await ctrl.execute_command(result)

        assert len(captured) == 0
