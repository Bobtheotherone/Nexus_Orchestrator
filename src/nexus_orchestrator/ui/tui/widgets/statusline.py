"""Status line widget — always-visible bottom bar with workspace info.

File: src/nexus_orchestrator/ui/tui/widgets/statusline.py

Shows: workspace path, git branch/dirty state, runner status.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

from nexus_orchestrator.ui.tui.state import AppState, RunnerStatus
from nexus_orchestrator.ui.tui.widgets.sidebar import icon


class StatusLine(Widget):
    """Always-visible status bar showing workspace context and runner state."""

    DEFAULT_CSS = """
    StatusLine {
        dock: bottom;
        height: 1;
        background: #0b1020;
        color: #7f8aa3;
        padding: 0 1;
    }
    #status-inner {
        width: 100%;
        height: 1;
    }
    #status-workspace {
        width: 1fr;
    }
    #status-git {
        width: auto;
        min-width: 16;
    }
    #status-runner {
        width: auto;
        min-width: 12;
        text-align: right;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="status-inner"):
            yield Static("", id="status-workspace")
            yield Static("", id="status-git")
            yield Static("", id="status-runner")

    def update_from_state(self, state: AppState) -> None:
        """Refresh all status segments from app state."""
        # Workspace
        ws = state.workspace_path
        if len(ws) > 40:
            ws = "..." + ws[-37:]
        self.query_one("#status-workspace", Static).update(f" {ws}")

        # Git
        git_text = ""
        if state.git_branch:
            dirty = "*" if state.git_dirty else ""
            ic = icon("branch")
            git_text = f" {ic} {state.git_branch}{dirty} " if ic else f" {state.git_branch}{dirty} "
        self.query_one("#status-git", Static).update(git_text)

        # Runner status
        runner_text = _runner_status_text(state.runner_status, state.current_command)
        self.query_one("#status-runner", Static).update(runner_text)


def _prefixed(icon_name: str, text: str) -> str:
    """Format ' icon text ' or just ' text ' if icon is empty."""
    ic = icon(icon_name)
    if ic:
        return f" {ic} {text} "
    return f" {text} "


def _runner_status_text(status: RunnerStatus, command: str) -> str:
    if status == RunnerStatus.RUNNING:
        cmd_short = command.split()[0] if command.split() else command
        return _prefixed("run", cmd_short)
    if status == RunnerStatus.GENERATING:
        return _prefixed("run", "generating...")
    if status == RunnerStatus.CANCEL_REQUESTED:
        return _prefixed("stop", "cancelling...")
    if status == RunnerStatus.CANCELLED:
        return _prefixed("cancel", "cancelled")
    return _prefixed("ready", "ready")


__all__ = ["StatusLine"]
