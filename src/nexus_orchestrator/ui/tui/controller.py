"""Controller layer — owns AppState, translates intents to service calls.

File: src/nexus_orchestrator/ui/tui/controller.py

NO widget/Textual imports. The controller:
1. Receives intents (user actions) from the view layer.
2. Dispatches to runners/services.
3. Reduces runner events into AppState mutations.
4. Notifies the view layer via a callback when state changes.

This keeps all business logic testable without Textual.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_orchestrator.ui.tui.runner import (
    InProcessRunner,
    RunnerEvent,
    RunnerEventKind,
    SubprocessRunner,
    create_runner,
)
from nexus_orchestrator.ui.tui.state import (
    AppState,
    BackendInfo,
    EventKind,
    RecentRun,
    RunnerStatus,
    TranscriptEvent,
)

if TYPE_CHECKING:
    from nexus_orchestrator.ui.tui.services.design_doc import DesignDocGenerator

# Type alias for the state-change notification callback
StateCallback = Callable[[], Awaitable[None] | None]


class TUIController:
    """Application controller — manages state and dispatches commands."""

    def __init__(
        self,
        *,
        state: AppState | None = None,
        on_state_change: StateCallback | None = None,
        no_color: bool = False,
    ) -> None:
        self.state = state or AppState()
        # Only override no_color if explicitly provided (not via custom state)
        if state is None:
            self.state.no_color = no_color
        self._on_state_change = on_state_change
        self._runner: SubprocessRunner | InProcessRunner = create_runner()
        self._run_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # State notification
    # ------------------------------------------------------------------

    async def _notify(self) -> None:
        if self._on_state_change is not None:
            result = self._on_state_change()
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------
    # Transcript helpers
    # ------------------------------------------------------------------

    def _append_event(self, event: TranscriptEvent) -> None:
        self.state.transcript.append(event)

    def append_system(self, text: str) -> None:
        """Append a system message to the transcript."""
        self._append_event(TranscriptEvent(kind=EventKind.SYSTEM, text=text))

    # ------------------------------------------------------------------
    # Intents (actions from the view)
    # ------------------------------------------------------------------

    async def execute_command(self, command: str) -> None:
        """Intent: user submitted a command string."""
        command = command.strip()
        if not command:
            return

        # Push to history
        if not self.state.command_history or self.state.command_history[-1] != command:
            self.state.command_history.append(command)

        # Handle slash commands
        if command.startswith("/"):
            await self._handle_slash_command(command)
            return

        # Default: generate engineering design doc
        self.state.runner_status = RunnerStatus.GENERATING
        self.state.current_command = command
        await self._notify()

        self._run_task = asyncio.create_task(self._generate_design_doc(command))

    async def cancel_command(self) -> None:
        """Intent: user pressed cancel."""
        if self.state.runner_status == RunnerStatus.RUNNING:
            self.state.runner_status = RunnerStatus.CANCEL_REQUESTED
            await self._notify()
            await self._runner.cancel()
        elif self.state.runner_status == RunnerStatus.GENERATING:
            self.state.runner_status = RunnerStatus.CANCEL_REQUESTED
            await self._notify()
            if self._run_task is not None:
                self._run_task.cancel()

    async def clear_transcript(self) -> None:
        """Intent: user requested transcript clear."""
        self.state.transcript.clear()
        await self._notify()

    async def detect_backends(self) -> None:
        """Intent: detect tool backends (run off UI thread)."""
        loop = asyncio.get_event_loop()
        backends = await loop.run_in_executor(None, _detect_backends_sync)
        self.state.backends = backends
        await self._notify()

    async def detect_workspace_info(self) -> None:
        """Intent: detect workspace path and git info."""
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _detect_workspace_sync)
        self.state.workspace_path = str(info.get("path", ""))
        self.state.git_branch = str(info.get("branch", ""))
        self.state.git_dirty = bool(info.get("dirty", False))
        await self._notify()

    # ------------------------------------------------------------------
    # Slash command handling
    # ------------------------------------------------------------------

    async def _handle_slash_command(self, command: str) -> None:
        cmd = command.split()[0].lower()

        if cmd == "/help":
            # View layer handles this — signal via system event
            self.append_system("__SHOW_HELP__")
            await self._notify()

        elif cmd == "/clear":
            self.state.transcript.clear()
            await self._notify()

        elif cmd == "/quit":
            self.append_system("__QUIT__")
            await self._notify()

        elif cmd == "/theme":
            if self.state.no_color:
                self.append_system("Theme: NO_COLOR mode (monochrome)")
            else:
                self.append_system(
                    "Theme: NEXUS Space\n"
                    "  --bg:       #05070c\n"
                    "  --panel:    #0b1020\n"
                    "  --primary:  #3fa9f5\n"
                    "  --primary2: #72c7ff\n"
                    "  --muted:    #7f8aa3"
                )
            await self._notify()

        elif cmd == "/copy":
            await self._handle_copy()

        elif cmd == "/export":
            parts = command.split(maxsplit=1)
            path = parts[1].strip() if len(parts) > 1 else ""
            await self._handle_export(path)

        elif cmd in ("/run", "/exec"):
            parts = command.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                self.append_system(f"Usage: {cmd} <command>")
                await self._notify()
                return
            cli_command = parts[1].strip()
            self.state.runner_status = RunnerStatus.RUNNING
            self.state.current_command = cli_command
            await self._notify()
            self._run_task = asyncio.create_task(self._run_command(cli_command))

        else:
            self.append_system(f"Unknown slash command: {cmd}")
            await self._notify()

    # ------------------------------------------------------------------
    # Copy / Export
    # ------------------------------------------------------------------

    def _transcript_as_text(self) -> str:
        """Serialize the transcript to plain text."""
        lines: list[str] = []
        for ev in self.state.transcript:
            if ev.kind == EventKind.COMMAND_HEADER:
                lines.append(f"[{ev.timestamp}] {ev.text}")
            elif ev.kind == EventKind.STDOUT:
                lines.append(ev.text)
            elif ev.kind == EventKind.STDERR:
                lines.append(f"ERR: {ev.text}")
            elif ev.kind == EventKind.EXIT_BADGE:
                code = ev.exit_code if ev.exit_code is not None else "?"
                lines.append(f"--- exit {code} ---")
            elif ev.kind == EventKind.SYSTEM:
                lines.append(f"[system] {ev.text}")
            elif ev.kind == EventKind.AGENT_HEADER:
                lines.append(f"[{ev.timestamp}] {ev.text}")
            elif ev.kind == EventKind.AGENT_RESPONSE:
                lines.append(ev.text)
        return "\n".join(lines)

    async def _handle_copy(self) -> None:
        """Copy last output (or full transcript) to clipboard via best-effort.

        Falls back to export if clipboard is unavailable.
        """
        text = self.state.last_output or self._transcript_as_text()
        if not text.strip():
            self.append_system("Nothing to copy.")
            await self._notify()
            return

        copied = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _try_clipboard_copy(text)
        )
        if copied:
            self.append_system("Copied to clipboard.")
        else:
            # Fallback: auto-export to file with the text we wanted to copy
            await self._handle_export("", override_text=text)
            self.append_system(
                "Clipboard not available (install xclip, xsel, or wl-copy). "
                "Transcript exported to file instead."
            )
        await self._notify()

    async def _handle_export(
        self, path_str: str, *, override_text: str = ""
    ) -> None:
        """Export transcript to file.

        Auto-generates a timestamped filename in ``~/.nexus/logs/`` if empty.
        If *override_text* is provided, it is written instead of the transcript.
        """
        if not path_str:
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            logs_dir = Path.home() / ".nexus" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            path_str = str(logs_dir / f"nexus_transcript_{ts}.txt")

        text = override_text or self._transcript_as_text()
        if not text.strip():
            self.append_system("Transcript is empty — nothing to export.")
            await self._notify()
            return

        try:
            out_path = Path(path_str).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
            self.append_system(f"Transcript exported to {out_path}")
        except OSError as exc:
            self.append_system(f"Export failed: {exc}")
        await self._notify()

    async def export_transcript(self, path_str: str = "") -> None:
        """Public API: export transcript (called from keybinding handler)."""
        await self._handle_export(path_str)

    async def copy_to_clipboard(self) -> None:
        """Public API: copy transcript to clipboard (called from Ctrl+Y)."""
        await self._handle_copy()

    # ------------------------------------------------------------------
    # Design doc generation
    # ------------------------------------------------------------------

    def _ensure_design_doc_generator(self) -> DesignDocGenerator:
        """Lazy-init the design doc generator."""
        if not hasattr(self, "_design_doc_generator"):
            from nexus_orchestrator.ui.tui.services.design_doc import DesignDocGenerator

            self._design_doc_generator = DesignDocGenerator()
        return self._design_doc_generator

    async def _generate_design_doc(self, prompt: str) -> None:
        """Generate a design document via the architect agent."""
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        self._append_event(
            TranscriptEvent(
                kind=EventKind.AGENT_HEADER,
                text=f"Design Doc > {prompt}",
                timestamp=ts,
            )
        )
        await self._notify()

        try:
            generator = self._ensure_design_doc_generator()
            result = await generator.generate(prompt)

            for line in result.content.splitlines():
                self._append_event(
                    TranscriptEvent(kind=EventKind.AGENT_RESPONSE, text=line)
                )
            await self._notify()

            saved_msg = f"Saved to {result.file_path}" if result.file_path else "Not saved to disk"
            self.append_system(
                f"Design doc generated ({result.model}, "
                f"{result.tokens_used} tokens, ${result.cost_usd:.4f}). "
                f"{saved_msg}"
            )
            self.state.last_output = result.content

        except asyncio.CancelledError:
            self.append_system("Design doc generation cancelled.")
        except Exception as exc:
            # Import here to avoid circular imports
            from nexus_orchestrator.ui.tui.services.design_doc import (
                NoProviderAvailableError,
            )

            if isinstance(exc, NoProviderAvailableError):
                # Show the full remediation message (it's user-friendly)
                self.append_system(str(exc))
            else:
                self.append_system(f"Design doc generation failed: {exc}")

        self.state.runner_status = RunnerStatus.IDLE
        self.state.current_command = ""
        self._run_task = None
        await self._notify()

    # ------------------------------------------------------------------
    # Runner integration
    # ------------------------------------------------------------------

    async def _run_command(self, command: str) -> None:
        """Run a command via the runner and reduce events into state."""
        # Add command header to transcript
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        self._append_event(
            TranscriptEvent(
                kind=EventKind.COMMAND_HEADER,
                text=f"nexus > {command}",
                timestamp=ts,
            )
        )
        await self._notify()

        exit_code = 1
        last_stdout = ""

        try:
            async for event in self._runner.run(command):
                await self._reduce_runner_event(event)
                if event.kind == RunnerEventKind.STDOUT_LINE:
                    last_stdout = event.text
                if event.kind == RunnerEventKind.FINISHED:
                    exit_code = event.exit_code or 0
        except Exception as exc:
            self.append_system(f"Runner error: {exc}")
            exit_code = 4

        # Update state
        if last_stdout:
            self.state.last_output = last_stdout

        self.state.recent_runs.append(RecentRun(command=command, exit_code=exit_code))
        if len(self.state.recent_runs) > self.state.max_recent:
            self.state.recent_runs = self.state.recent_runs[-self.state.max_recent :]

        self.state.runner_status = RunnerStatus.IDLE
        self.state.current_command = ""
        self._run_task = None
        await self._notify()

    async def _reduce_runner_event(self, event: RunnerEvent) -> None:
        """Reduce a single runner event into transcript state."""
        if event.kind == RunnerEventKind.STDOUT_LINE:
            self._append_event(
                TranscriptEvent(kind=EventKind.STDOUT, text=event.text)
            )
        elif event.kind == RunnerEventKind.STDERR_LINE:
            self._append_event(
                TranscriptEvent(kind=EventKind.STDERR, text=event.text)
            )
        elif event.kind == RunnerEventKind.FINISHED:
            self._append_event(
                TranscriptEvent(
                    kind=EventKind.EXIT_BADGE,
                    text="",
                    exit_code=event.exit_code,
                )
            )
        elif event.kind == RunnerEventKind.CANCEL_ACK:
            self._append_event(
                TranscriptEvent(kind=EventKind.SYSTEM, text="Command cancelled.")
            )
            self.state.runner_status = RunnerStatus.CANCELLED
        elif event.kind == RunnerEventKind.ERROR:
            self._append_event(
                TranscriptEvent(kind=EventKind.STDERR, text=event.text)
            )

        await self._notify()
        # Yield to the event loop so Textual can render between events
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Sync helpers (run in executor)
# ---------------------------------------------------------------------------


def _detect_backends_sync() -> list[BackendInfo]:
    """Detect all backends with auth status (called from executor)."""
    try:
        from nexus_orchestrator.auth.strategy import detect_all_auth

        statuses = detect_all_auth()
        return [
            BackendInfo(
                name=s.name,
                version=s.version,
                auth_mode=s.auth_mode.value,
                available=s.available,
                logged_in=s.logged_in,
                has_api_key=s.has_api_key,
                remediation=s.remediation,
            )
            for s in statuses
        ]
    except Exception:
        return []


def _detect_workspace_sync() -> dict[str, object]:
    """Detect workspace path, git branch, dirty state."""
    result: dict[str, object] = {"path": os.getcwd(), "branch": "", "dirty": False}

    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if branch.returncode == 0:
            result["branch"] = branch.stdout.strip()

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if status.returncode == 0:
            result["dirty"] = bool(status.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return result


def _try_clipboard_copy(text: str) -> bool:
    """Best-effort copy *text* to the system clipboard. Returns True on success."""
    # Try in order: pbcopy (macOS), wl-copy (Wayland), xclip, xsel
    candidates: list[list[str]] = [
        ["pbcopy"],
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ]
    for cmd in candidates:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                timeout=5,
                check=False,
                capture_output=True,
            )
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            continue
    return False


__all__ = [
    "TUIController",
]
