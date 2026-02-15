"""Unit tests for TUI state module — pure data, no Textual dependency.

File: tests/unit/test_tui_state.py

Tests:
- TranscriptEvent creation and fields
- AppState initialization and defaults
- Ring buffer behavior (deque maxlen)
- RunnerStatus enum values
- EventKind enum values
"""

from __future__ import annotations

import pytest

from nexus_orchestrator.ui.tui.state import (
    MAX_TRANSCRIPT_LINES,
    AppState,
    BackendInfo,
    EventKind,
    RecentRun,
    RunnerStatus,
    TranscriptEvent,
)


@pytest.mark.unit
class TestTranscriptEvent:
    """Test TranscriptEvent creation."""

    def test_stdout_event(self) -> None:
        event = TranscriptEvent(kind=EventKind.STDOUT, text="hello world")
        assert event.kind == EventKind.STDOUT
        assert event.text == "hello world"
        assert event.timestamp  # non-empty
        assert event.exit_code is None

    def test_exit_badge_event(self) -> None:
        event = TranscriptEvent(kind=EventKind.EXIT_BADGE, text="", exit_code=0)
        assert event.exit_code == 0

    def test_events_are_frozen(self) -> None:
        event = TranscriptEvent(kind=EventKind.SYSTEM, text="test")
        with pytest.raises(AttributeError):
            event.text = "modified"  # type: ignore[misc]


@pytest.mark.unit
class TestEventKind:
    """Test EventKind enum values."""

    def test_all_kinds_exist(self) -> None:
        kinds = {e.value for e in EventKind}
        assert "stdout" in kinds
        assert "stderr" in kinds
        assert "system" in kinds
        assert "command_header" in kinds
        assert "exit_badge" in kinds


@pytest.mark.unit
class TestRunnerStatus:
    """Test RunnerStatus enum."""

    def test_all_statuses_exist(self) -> None:
        statuses = {s.value for s in RunnerStatus}
        assert "idle" in statuses
        assert "running" in statuses
        assert "cancel_requested" in statuses
        assert "cancelled" in statuses


@pytest.mark.unit
class TestAppState:
    """Test AppState initialization and defaults."""

    def test_default_state(self) -> None:
        state = AppState()
        assert state.runner_status == RunnerStatus.IDLE
        assert state.current_command == ""
        assert len(state.transcript) == 0
        assert state.backends == []
        assert state.recent_runs == []
        assert state.command_history == []
        assert state.workspace_path == ""
        assert state.git_branch == ""
        assert state.git_dirty is False
        assert state.last_output == ""
        assert state.no_color is False

    def test_transcript_ring_buffer_limit(self) -> None:
        """Transcript deque has maxlen = MAX_TRANSCRIPT_LINES."""
        state = AppState()
        assert state.transcript.maxlen == MAX_TRANSCRIPT_LINES

    def test_transcript_ring_buffer_eviction(self) -> None:
        """When transcript exceeds maxlen, oldest events are evicted."""
        state = AppState()
        # Fill past capacity
        for i in range(MAX_TRANSCRIPT_LINES + 100):
            state.transcript.append(
                TranscriptEvent(kind=EventKind.STDOUT, text=f"line {i}")
            )
        assert len(state.transcript) == MAX_TRANSCRIPT_LINES
        # First event should be line 100 (earliest 100 evicted)
        assert state.transcript[0].text == "line 100"

    def test_recent_runs(self) -> None:
        state = AppState()
        state.recent_runs.append(RecentRun(command="status", exit_code=0))
        state.recent_runs.append(RecentRun(command="doctor", exit_code=1))
        assert len(state.recent_runs) == 2
        assert state.recent_runs[0].command == "status"
        assert state.recent_runs[1].exit_code == 1

    def test_backend_info(self) -> None:
        bi = BackendInfo(name="codex", version="1.2.3")
        assert bi.name == "codex"
        assert bi.version == "1.2.3"

    def test_backend_info_no_version(self) -> None:
        bi = BackendInfo(name="claude", version=None)
        assert bi.version is None
