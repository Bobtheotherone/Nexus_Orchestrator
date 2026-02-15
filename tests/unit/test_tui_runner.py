"""Unit tests for TUI runner — streaming output and cancellation.

File: tests/unit/test_tui_runner.py

Tests:
- RunnerEvent creation
- InProcessRunner basic execution
- SubprocessRunner argv construction
- create_runner factory
- Cancel behavior
"""

from __future__ import annotations

import asyncio
import sys
from unittest import mock

import pytest

from nexus_orchestrator.ui.tui.runner import (
    InProcessRunner,
    RunnerEvent,
    RunnerEventKind,
    SubprocessRunner,
    create_runner,
)


@pytest.mark.unit
class TestRunnerEvent:
    """Test RunnerEvent data model."""

    def test_stdout_event(self) -> None:
        event = RunnerEvent(kind=RunnerEventKind.STDOUT_LINE, text="hello")
        assert event.kind == RunnerEventKind.STDOUT_LINE
        assert event.text == "hello"
        assert event.exit_code is None

    def test_finished_event(self) -> None:
        event = RunnerEvent(kind=RunnerEventKind.FINISHED, exit_code=0)
        assert event.exit_code == 0

    def test_events_are_frozen(self) -> None:
        event = RunnerEvent(kind=RunnerEventKind.STARTED, text="test")
        with pytest.raises(AttributeError):
            event.text = "modified"  # type: ignore[misc]


@pytest.mark.unit
class TestRunnerEventKind:
    """Test all event kinds exist."""

    def test_all_kinds(self) -> None:
        kinds = {e.value for e in RunnerEventKind}
        expected = {"stdout_line", "stderr_line", "started", "finished", "cancel_ack", "error"}
        assert kinds == expected


@pytest.mark.unit
class TestSubprocessRunnerArgv:
    """Test SubprocessRunner._build_argv()."""

    def test_build_argv_simple_command(self) -> None:
        argv = SubprocessRunner._build_argv("status")
        assert len(argv) >= 2
        assert argv[-1] == "status"

    def test_build_argv_with_flags(self) -> None:
        argv = SubprocessRunner._build_argv("plan spec.md --mock")
        assert "plan" in argv
        assert "spec.md" in argv
        assert "--mock" in argv

    def test_build_argv_quoted_path(self) -> None:
        argv = SubprocessRunner._build_argv('plan "path with spaces/spec.md"')
        assert "path with spaces/spec.md" in argv


@pytest.mark.unit
class TestInProcessRunner:
    """Test InProcessRunner basic behavior."""

    @pytest.mark.asyncio
    async def test_run_emits_started_and_finished(self) -> None:
        runner = InProcessRunner()
        events = []
        async for event in runner.run("doctor"):
            events.append(event)

        kinds = [e.kind for e in events]
        assert RunnerEventKind.STARTED in kinds
        assert RunnerEventKind.FINISHED in kinds

    @pytest.mark.asyncio
    async def test_run_started_has_command(self) -> None:
        runner = InProcessRunner()
        events = []
        async for event in runner.run("config"):
            events.append(event)

        started = [e for e in events if e.kind == RunnerEventKind.STARTED]
        assert len(started) == 1
        assert started[0].text == "config"

    @pytest.mark.asyncio
    async def test_run_captures_output(self) -> None:
        runner = InProcessRunner()
        events = []
        async for event in runner.run("doctor"):
            events.append(event)

        # Doctor should produce some stdout
        stdout_events = [e for e in events if e.kind == RunnerEventKind.STDOUT_LINE]
        assert len(stdout_events) > 0

    @pytest.mark.asyncio
    async def test_cancel_sets_flag(self) -> None:
        runner = InProcessRunner()
        assert not runner._cancel_requested
        await runner.cancel()
        assert runner._cancel_requested


@pytest.mark.unit
class TestCreateRunner:
    """Test runner factory."""

    def test_creates_in_process_when_no_nexus_binary(self) -> None:
        with mock.patch("shutil.which", return_value=None):
            runner = create_runner(prefer_subprocess=True)
        assert isinstance(runner, InProcessRunner)

    def test_creates_subprocess_when_nexus_on_path(self) -> None:
        with mock.patch("shutil.which", return_value="/usr/bin/nexus"):
            runner = create_runner(prefer_subprocess=True)
        assert isinstance(runner, SubprocessRunner)

    def test_prefers_in_process_when_disabled(self) -> None:
        runner = create_runner(prefer_subprocess=False)
        assert isinstance(runner, InProcessRunner)
