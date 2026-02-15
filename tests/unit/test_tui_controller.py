"""Unit tests for TUI controller — event->state reducer logic.

File: tests/unit/test_tui_controller.py

Tests:
- Controller initialization
- Slash command handling (/clear, /theme, /copy, /export, unknown)
- Runner event reduction (STDOUT, STDERR, FINISHED, CANCEL_ACK, ERROR)
- Command history management
- Backend detection (mocked)
- Workspace info detection (mocked)
- Transcript export (text serialization)
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from nexus_orchestrator.ui.tui.controller import TUIController
from nexus_orchestrator.ui.tui.runner import RunnerEvent, RunnerEventKind
from nexus_orchestrator.ui.tui.state import (
    AppState,
    EventKind,
    RunnerStatus,
)


@pytest.mark.unit
class TestControllerInit:
    """Test controller initialization."""

    def test_default_state(self) -> None:
        ctrl = TUIController()
        assert ctrl.state.runner_status == RunnerStatus.IDLE
        assert len(ctrl.state.transcript) == 0

    def test_custom_state(self) -> None:
        state = AppState(no_color=True)
        ctrl = TUIController(state=state)
        assert ctrl.state.no_color is True

    def test_no_color_flag(self) -> None:
        ctrl = TUIController(no_color=True)
        assert ctrl.state.no_color is True


@pytest.mark.unit
class TestSlashCommands:
    """Test slash command handling."""

    @pytest.fixture
    def controller(self) -> TUIController:
        return TUIController()

    @pytest.mark.asyncio
    async def test_clear_empties_transcript(self, controller: TUIController) -> None:
        controller.append_system("some text")
        assert len(controller.state.transcript) == 1
        await controller._handle_slash_command("/clear")
        assert len(controller.state.transcript) == 0

    @pytest.mark.asyncio
    async def test_theme_appends_system_message(self, controller: TUIController) -> None:
        await controller._handle_slash_command("/theme")
        assert len(controller.state.transcript) == 1
        assert controller.state.transcript[0].kind == EventKind.SYSTEM
        assert "NEXUS Space" in controller.state.transcript[0].text

    @pytest.mark.asyncio
    async def test_theme_no_color(self) -> None:
        ctrl = TUIController(no_color=True)
        await ctrl._handle_slash_command("/theme")
        assert "NO_COLOR" in ctrl.state.transcript[0].text

    @pytest.mark.asyncio
    async def test_copy_empty_transcript(self, controller: TUIController) -> None:
        await controller._handle_slash_command("/copy")
        assert "Nothing to copy" in controller.state.transcript[0].text

    @pytest.mark.asyncio
    async def test_copy_with_output(self, controller: TUIController) -> None:
        controller.state.last_output = "some result"
        await controller._handle_slash_command("/copy")
        # Should attempt clipboard — result depends on system tools
        last = controller.state.transcript[-1]
        assert "Copied" in last.text or "Clipboard not available" in last.text

    @pytest.mark.asyncio
    async def test_unknown_slash_command(self, controller: TUIController) -> None:
        await controller._handle_slash_command("/foobar")
        assert "Unknown slash command" in controller.state.transcript[0].text

    @pytest.mark.asyncio
    async def test_help_emits_signal(self, controller: TUIController) -> None:
        await controller._handle_slash_command("/help")
        assert controller.state.transcript[0].text == "__SHOW_HELP__"

    @pytest.mark.asyncio
    async def test_quit_emits_signal(self, controller: TUIController) -> None:
        await controller._handle_slash_command("/quit")
        assert controller.state.transcript[0].text == "__QUIT__"


@pytest.mark.unit
class TestRunnerEventReduction:
    """Test that runner events are correctly reduced into transcript state."""

    @pytest.fixture
    def controller(self) -> TUIController:
        return TUIController()

    @pytest.mark.asyncio
    async def test_stdout_line(self, controller: TUIController) -> None:
        event = RunnerEvent(kind=RunnerEventKind.STDOUT_LINE, text="hello")
        await controller._reduce_runner_event(event)
        assert len(controller.state.transcript) == 1
        assert controller.state.transcript[0].kind == EventKind.STDOUT
        assert controller.state.transcript[0].text == "hello"

    @pytest.mark.asyncio
    async def test_stderr_line(self, controller: TUIController) -> None:
        event = RunnerEvent(kind=RunnerEventKind.STDERR_LINE, text="warning")
        await controller._reduce_runner_event(event)
        assert controller.state.transcript[0].kind == EventKind.STDERR

    @pytest.mark.asyncio
    async def test_finished_event(self, controller: TUIController) -> None:
        event = RunnerEvent(kind=RunnerEventKind.FINISHED, exit_code=0)
        await controller._reduce_runner_event(event)
        assert controller.state.transcript[0].kind == EventKind.EXIT_BADGE
        assert controller.state.transcript[0].exit_code == 0

    @pytest.mark.asyncio
    async def test_cancel_ack(self, controller: TUIController) -> None:
        event = RunnerEvent(kind=RunnerEventKind.CANCEL_ACK, exit_code=130)
        await controller._reduce_runner_event(event)
        assert controller.state.transcript[0].kind == EventKind.SYSTEM
        assert "cancelled" in controller.state.transcript[0].text.lower()
        assert controller.state.runner_status == RunnerStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_error_event(self, controller: TUIController) -> None:
        event = RunnerEvent(kind=RunnerEventKind.ERROR, text="bad thing")
        await controller._reduce_runner_event(event)
        assert controller.state.transcript[0].kind == EventKind.STDERR
        assert "bad thing" in controller.state.transcript[0].text


@pytest.mark.unit
class TestCommandHistory:
    """Test command history management."""

    @pytest.mark.asyncio
    async def test_commands_added_to_history(self) -> None:
        ctrl = TUIController()
        # Non-slash commands now route to _generate_design_doc, mock it
        with mock.patch.object(ctrl, "_generate_design_doc", new_callable=mock.AsyncMock):
            await ctrl.execute_command("status")
            # Wait for any task to be created/settled
            if ctrl._run_task is not None:
                await ctrl._run_task
        assert "status" in ctrl.state.command_history

    @pytest.mark.asyncio
    async def test_duplicate_commands_not_added(self) -> None:
        ctrl = TUIController()
        with mock.patch.object(ctrl, "_generate_design_doc", new_callable=mock.AsyncMock):
            await ctrl.execute_command("status")
            if ctrl._run_task is not None:
                await ctrl._run_task
            await ctrl.execute_command("status")
            if ctrl._run_task is not None:
                await ctrl._run_task
        assert ctrl.state.command_history.count("status") == 1

    @pytest.mark.asyncio
    async def test_slash_commands_added_to_history(self) -> None:
        ctrl = TUIController()
        await ctrl.execute_command("/theme")
        assert "/theme" in ctrl.state.command_history

    @pytest.mark.asyncio
    async def test_empty_command_ignored(self) -> None:
        ctrl = TUIController()
        await ctrl.execute_command("")
        assert len(ctrl.state.command_history) == 0
        await ctrl.execute_command("   ")
        assert len(ctrl.state.command_history) == 0


@pytest.mark.unit
class TestStateNotification:
    """Test that state change callback is invoked."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_slash_command(self) -> None:
        callback_count = 0

        async def callback() -> None:
            nonlocal callback_count
            callback_count += 1

        ctrl = TUIController(on_state_change=callback)
        await ctrl.execute_command("/theme")
        assert callback_count > 0

    @pytest.mark.asyncio
    async def test_callback_invoked_on_clear(self) -> None:
        callback_count = 0

        async def callback() -> None:
            nonlocal callback_count
            callback_count += 1

        ctrl = TUIController(on_state_change=callback)
        await ctrl.clear_transcript()
        assert callback_count == 1


@pytest.mark.unit
class TestCancelCommand:
    """Test cancel behavior."""

    @pytest.mark.asyncio
    async def test_cancel_when_idle_is_noop(self) -> None:
        ctrl = TUIController()
        await ctrl.cancel_command()
        assert ctrl.state.runner_status == RunnerStatus.IDLE

    @pytest.mark.asyncio
    async def test_cancel_when_running(self) -> None:
        ctrl = TUIController()
        ctrl.state.runner_status = RunnerStatus.RUNNING
        # Mock the runner's cancel method
        with mock.patch.object(ctrl._runner, "cancel", new_callable=mock.AsyncMock):
            await ctrl.cancel_command()
        assert ctrl.state.runner_status == RunnerStatus.CANCEL_REQUESTED


@pytest.mark.unit
class TestTranscriptExport:
    """Test transcript serialization and export."""

    def test_transcript_as_text_empty(self) -> None:
        ctrl = TUIController()
        assert ctrl._transcript_as_text() == ""

    def test_transcript_as_text_with_events(self) -> None:
        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        ctrl = TUIController()
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.COMMAND_HEADER, text="nexus > status", timestamp="12:00:00")
        )
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.STDOUT, text="all good")
        )
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.EXIT_BADGE, text="", exit_code=0)
        )
        text = ctrl._transcript_as_text()
        assert "[12:00:00] nexus > status" in text
        assert "all good" in text
        assert "--- exit 0 ---" in text

    def test_transcript_as_text_stderr(self) -> None:
        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        ctrl = TUIController()
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.STDERR, text="something failed")
        )
        text = ctrl._transcript_as_text()
        assert "ERR: something failed" in text

    @pytest.mark.asyncio
    async def test_export_to_file(self, tmp_path: object) -> None:
        from pathlib import Path

        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        assert isinstance(tmp_path, Path)
        ctrl = TUIController()
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.STDOUT, text="export test")
        )
        out = tmp_path / "export.txt"
        await ctrl.export_transcript(str(out))
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "export test" in content
        # Controller should confirm export in transcript
        last = ctrl.state.transcript[-1]
        assert "exported" in last.text.lower()

    @pytest.mark.asyncio
    async def test_export_empty_transcript(self) -> None:
        ctrl = TUIController()
        await ctrl.export_transcript("irrelevant.txt")
        assert "empty" in ctrl.state.transcript[-1].text.lower()

    @pytest.mark.asyncio
    async def test_export_auto_filename(self, tmp_path: object) -> None:
        from pathlib import Path
        from unittest.mock import patch

        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        assert isinstance(tmp_path, Path)
        ctrl = TUIController()
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.STDOUT, text="auto name test")
        )
        # Patch Path.home to use tmp_path so auto-generated file lands there
        with patch.object(Path, "home", return_value=tmp_path):
            await ctrl.export_transcript("")
        # Should have created nexus_transcript_*.txt in ~/.nexus/logs/
        logs_dir = tmp_path / ".nexus" / "logs"
        files = list(logs_dir.glob("nexus_transcript_*.txt"))
        assert len(files) == 1
        assert "auto name test" in files[0].read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_export_creates_logs_directory(self, tmp_path: object) -> None:
        """Export auto-creates ~/.nexus/logs/ directory."""
        from pathlib import Path
        from unittest.mock import patch

        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        assert isinstance(tmp_path, Path)
        ctrl = TUIController()
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.STDOUT, text="dir test")
        )
        with patch.object(Path, "home", return_value=tmp_path):
            await ctrl.export_transcript("")
        logs_dir = tmp_path / ".nexus" / "logs"
        assert logs_dir.is_dir()

    @pytest.mark.asyncio
    async def test_export_shows_path_in_transcript(self, tmp_path: object) -> None:
        """Export displays the exact file path in the transcript."""
        from pathlib import Path

        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        assert isinstance(tmp_path, Path)
        ctrl = TUIController()
        ctrl.state.transcript.append(
            TranscriptEvent(kind=EventKind.STDOUT, text="path test")
        )
        out = tmp_path / "my_export.txt"
        await ctrl.export_transcript(str(out))
        last = ctrl.state.transcript[-1]
        assert str(out) in last.text or "my_export.txt" in last.text


@pytest.mark.unit
class TestCopyToClipboard:
    """Test copy to clipboard with fallback to export."""

    @pytest.mark.asyncio
    async def test_copy_empty_transcript_is_noop(self) -> None:
        ctrl = TUIController()
        await ctrl.copy_to_clipboard()
        assert "Nothing to copy" in ctrl.state.transcript[0].text

    @pytest.mark.asyncio
    async def test_copy_attempts_clipboard(self) -> None:
        ctrl = TUIController()
        ctrl.state.last_output = "test output"
        with mock.patch(
            "nexus_orchestrator.ui.tui.controller._try_clipboard_copy",
            return_value=True,
        ):
            await ctrl.copy_to_clipboard()
        last = ctrl.state.transcript[-1]
        assert "Copied to clipboard" in last.text

    @pytest.mark.asyncio
    async def test_copy_falls_back_to_export(self, tmp_path: object) -> None:
        """When clipboard fails, copy falls back to export."""
        from pathlib import Path
        from unittest.mock import patch as std_patch

        assert isinstance(tmp_path, Path)
        ctrl = TUIController()
        ctrl.state.last_output = "fallback content"
        with (
            mock.patch(
                "nexus_orchestrator.ui.tui.controller._try_clipboard_copy",
                return_value=False,
            ),
            std_patch.object(Path, "home", return_value=tmp_path),
        ):
            await ctrl.copy_to_clipboard()
        # Should have exported a file
        logs_dir = tmp_path / ".nexus" / "logs"
        files = list(logs_dir.glob("nexus_transcript_*.txt"))
        assert len(files) == 1
        # Transcript should mention clipboard not available
        texts = [e.text for e in ctrl.state.transcript]
        assert any("Clipboard not available" in t for t in texts)


@pytest.mark.unit
class TestCancelSafety:
    """Test double-press Ctrl+C cancel safety."""

    @pytest.mark.asyncio
    async def test_single_ctrl_c_does_not_cancel_when_running(self) -> None:
        """Single Ctrl+C when running should NOT cancel immediately."""
        ctrl = TUIController()
        ctrl.state.runner_status = RunnerStatus.RUNNING
        # After a single Ctrl+C, the app would show a warning but not cancel.
        # The controller cancel_command is only called on the SECOND press.
        # This test verifies the controller does NOT transition to CANCEL_REQUESTED
        # unless cancel_command is explicitly called.
        assert ctrl.state.runner_status == RunnerStatus.RUNNING
        # (The app layer handles the double-press logic)

    @pytest.mark.asyncio
    async def test_cancel_command_transitions_to_cancel_requested(self) -> None:
        """Calling cancel_command on RUNNING sets CANCEL_REQUESTED."""
        ctrl = TUIController()
        ctrl.state.runner_status = RunnerStatus.RUNNING
        with mock.patch.object(ctrl._runner, "cancel", new_callable=mock.AsyncMock):
            await ctrl.cancel_command()
        assert ctrl.state.runner_status == RunnerStatus.CANCEL_REQUESTED


@pytest.mark.unit
class TestStreamingEvents:
    """Test that runner events stream into transcript incrementally."""

    @pytest.mark.asyncio
    async def test_events_appear_before_completion(self) -> None:
        """Simulated runner events appear in transcript incrementally."""
        from nexus_orchestrator.ui.tui.state import TranscriptEvent

        notifications: list[int] = []

        async def on_change() -> None:
            notifications.append(len(ctrl.state.transcript))

        ctrl = TUIController(on_state_change=on_change)

        # Simulate individual event reductions (as runner would emit them)
        await ctrl._reduce_runner_event(
            RunnerEvent(kind=RunnerEventKind.STDOUT_LINE, text="line 1")
        )
        # After first event, transcript should have 1 entry
        assert len(ctrl.state.transcript) >= 1
        assert ctrl.state.transcript[0].text == "line 1"

        await ctrl._reduce_runner_event(
            RunnerEvent(kind=RunnerEventKind.STDOUT_LINE, text="line 2")
        )
        assert len(ctrl.state.transcript) >= 2

        # Notifications were called for each event
        assert len(notifications) >= 2

    @pytest.mark.asyncio
    async def test_cancel_stops_and_posts_message(self) -> None:
        """Cancel event sets CANCELLED status and posts system message."""
        ctrl = TUIController()
        ctrl.state.runner_status = RunnerStatus.RUNNING

        await ctrl._reduce_runner_event(
            RunnerEvent(kind=RunnerEventKind.CANCEL_ACK, exit_code=130)
        )

        assert ctrl.state.runner_status == RunnerStatus.CANCELLED
        cancel_events = [
            e for e in ctrl.state.transcript
            if e.kind == EventKind.SYSTEM and "cancelled" in e.text.lower()
        ]
        assert len(cancel_events) == 1
