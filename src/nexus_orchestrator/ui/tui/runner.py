"""Streaming command runner with real cancellation — NO Textual imports.

File: src/nexus_orchestrator/ui/tui/runner.py

Provides two execution strategies:
A) SubprocessRunner — spawns ``nexus`` CLI as a child process, streams
   stdout/stderr line-by-line, and cancels via SIGINT + SIGTERM.
B) InProcessRunner — calls run_cli() directly with captured IO (legacy
   compat, non-streaming).

Both implement the Runner protocol and emit RunnerEvent objects that the
controller can reduce into state updates.
"""

from __future__ import annotations

import asyncio
import enum
import os
import shlex
import signal
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

# ---------------------------------------------------------------------------
# Runner event model
# ---------------------------------------------------------------------------


class RunnerEventKind(enum.Enum):
    STDOUT_LINE = "stdout_line"
    STDERR_LINE = "stderr_line"
    STARTED = "started"
    FINISHED = "finished"
    CANCEL_ACK = "cancel_ack"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class RunnerEvent:
    kind: RunnerEventKind
    text: str = ""
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Runner protocol
# ---------------------------------------------------------------------------


class Runner(Protocol):
    async def run(self, command: str) -> AsyncIterator[RunnerEvent]: ...
    async def cancel(self) -> None: ...


# ---------------------------------------------------------------------------
# Subprocess runner (streaming + real cancel via SIGINT)
# ---------------------------------------------------------------------------

# Grace period before SIGTERM after SIGINT
_SIGTERM_GRACE_SECS = 2.0
_KILL_GRACE_SECS = 3.0


class SubprocessRunner:
    """Spawns ``nexus <command>`` as a subprocess, streams output, cancels via signal."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_requested: bool = False

    async def run(self, command: str) -> AsyncIterator[RunnerEvent]:
        """Execute *command* (e.g. ``"plan spec.md --mock"``), yielding events."""
        self._cancel_requested = False

        argv = self._build_argv(command)

        yield RunnerEvent(kind=RunnerEventKind.STARTED, text=command)

        # Ensure unbuffered output from Python subprocesses
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            self._process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Start new process group so we can signal it cleanly
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )
        except FileNotFoundError:
            yield RunnerEvent(
                kind=RunnerEventKind.ERROR,
                text=f"Command not found: {argv[0]}",
                exit_code=127,
            )
            yield RunnerEvent(kind=RunnerEventKind.FINISHED, exit_code=127)
            return
        except OSError as exc:
            yield RunnerEvent(
                kind=RunnerEventKind.ERROR,
                text=f"Failed to start process: {exc}",
                exit_code=1,
            )
            yield RunnerEvent(kind=RunnerEventKind.FINISHED, exit_code=1)
            return

        assert self._process.stdout is not None
        assert self._process.stderr is not None

        async def _read_stream(
            stream: asyncio.StreamReader,
            kind: RunnerEventKind,
        ) -> list[RunnerEvent]:
            events: list[RunnerEvent] = []
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                events.append(RunnerEvent(kind=kind, text=text))
            return events

        # Read both streams concurrently
        stdout_task = asyncio.create_task(
            _read_stream(self._process.stdout, RunnerEventKind.STDOUT_LINE)
        )
        stderr_task = asyncio.create_task(
            _read_stream(self._process.stderr, RunnerEventKind.STDERR_LINE)
        )

        # Interleave events as they arrive via a queue
        queue: asyncio.Queue[RunnerEvent | None] = asyncio.Queue()

        async def _enqueue_stream(
            stream: asyncio.StreamReader,
            kind: RunnerEventKind,
        ) -> None:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                await queue.put(RunnerEvent(kind=kind, text=text))

        # Cancel the non-interleaved tasks — we'll use the queue approach instead
        stdout_task.cancel()
        stderr_task.cancel()

        enqueue_stdout = asyncio.create_task(
            _enqueue_stream(self._process.stdout, RunnerEventKind.STDOUT_LINE)
        )
        enqueue_stderr = asyncio.create_task(
            _enqueue_stream(self._process.stderr, RunnerEventKind.STDERR_LINE)
        )

        async def _wait_done() -> None:
            await asyncio.gather(enqueue_stdout, enqueue_stderr)
            await queue.put(None)  # sentinel

        done_task = asyncio.create_task(_wait_done())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        await self._process.wait()
        exit_code = self._process.returncode or 0

        if self._cancel_requested:
            yield RunnerEvent(kind=RunnerEventKind.CANCEL_ACK, exit_code=exit_code)

        yield RunnerEvent(kind=RunnerEventKind.FINISHED, exit_code=exit_code)
        self._process = None

        # Suppress any lingering task warnings
        import contextlib

        done_task.cancel()
        for t in (enqueue_stdout, enqueue_stderr, done_task):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t

    async def cancel(self) -> None:
        """Send SIGINT, then SIGTERM after grace period."""
        if self._process is None or self._process.returncode is not None:
            return

        self._cancel_requested = True

        if sys.platform == "win32":
            self._process.terminate()
            return

        # SIGINT to the process group
        try:
            pgid = os.getpgid(self._process.pid)
            os.killpg(pgid, signal.SIGINT)
        except (ProcessLookupError, OSError):
            return

        # Wait for grace period then escalate
        try:
            await asyncio.wait_for(
                self._process.wait(), timeout=_SIGTERM_GRACE_SECS
            )
        except TimeoutError:
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                return
            try:
                await asyncio.wait_for(
                    self._process.wait(), timeout=_KILL_GRACE_SECS
                )
            except TimeoutError:
                self._process.kill()

    @staticmethod
    def _build_argv(command: str) -> list[str]:
        """Build subprocess argv from a command string.

        Prepends the ``nexus`` entrypoint if needed.
        """
        parts = shlex.split(command)
        # Find the nexus entrypoint
        nexus_exe = _find_nexus_executable()
        return [nexus_exe, *parts]


# ---------------------------------------------------------------------------
# In-process runner (legacy compat — blocking, non-streaming)
# ---------------------------------------------------------------------------


class InProcessRunner:
    """Runs commands in-process by calling run_cli(). Non-streaming."""

    def __init__(self) -> None:
        self._cancel_requested: bool = False

    async def run(self, command: str) -> AsyncIterator[RunnerEvent]:
        """Execute via run_cli(), yielding events when complete."""
        import io

        self._cancel_requested = False
        yield RunnerEvent(kind=RunnerEventKind.STARTED, text=command)

        from nexus_orchestrator.ui.cli import run_cli

        argv = shlex.split(command)
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 1

        try:
            loop = asyncio.get_event_loop()
            exit_code = await loop.run_in_executor(
                None,
                lambda: _run_cli_captured(run_cli, argv, stdout_buf, stderr_buf),
            )
        except Exception as exc:
            yield RunnerEvent(
                kind=RunnerEventKind.ERROR,
                text=f"error: {exc}",
                exit_code=4,
            )
            exit_code = 4

        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()

        if stdout_text.strip():
            for line in stdout_text.strip().splitlines():
                yield RunnerEvent(kind=RunnerEventKind.STDOUT_LINE, text=line)

        if stderr_text.strip():
            for line in stderr_text.strip().splitlines():
                yield RunnerEvent(kind=RunnerEventKind.STDERR_LINE, text=line)

        yield RunnerEvent(kind=RunnerEventKind.FINISHED, exit_code=exit_code)

    async def cancel(self) -> None:
        self._cancel_requested = True


def _run_cli_captured(
    run_cli_fn: object,
    argv: list[str],
    stdout_buf: object,
    stderr_buf: object,
) -> int:
    """Run CLI in a thread with captured IO."""
    import contextlib
    import io

    assert callable(run_cli_fn)
    assert isinstance(stdout_buf, io.StringIO)
    assert isinstance(stderr_buf, io.StringIO)

    exit_code = 1
    try:
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            exit_code = run_cli_fn(argv)
    except SystemExit as exc:
        code = exc.code
        exit_code = int(code) if isinstance(code, int) else 1
    except Exception as exc:
        stderr_buf.write(f"error: {exc}\n")
        exit_code = 4
    return exit_code


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _find_nexus_executable() -> str:
    """Find the nexus CLI entrypoint for subprocess execution."""
    import shutil

    # 1. Check if nexus is on PATH
    nexus_path = shutil.which("nexus")
    if nexus_path:
        return nexus_path

    # 2. Fall back to python -m nexus_orchestrator
    return sys.executable


def create_runner(*, prefer_subprocess: bool = True) -> SubprocessRunner | InProcessRunner:
    """Factory: create the best available runner."""
    if prefer_subprocess:
        import shutil

        if shutil.which("nexus"):
            return SubprocessRunner()
    return InProcessRunner()


__all__ = [
    "InProcessRunner",
    "Runner",
    "RunnerEvent",
    "RunnerEventKind",
    "SubprocessRunner",
    "create_runner",
]
