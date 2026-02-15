"""Animated spinner widget — visible only during active runs.

File: src/nexus_orchestrator/ui/tui/widgets/spinner.py

Shows a smooth animation next to activity text to indicate the TUI is working.
Driven by a Textual timer; hidden when idle.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from nexus_orchestrator.ui.tui.state import AppState, RunnerStatus

# Braille-dot spinner frames — smooth 8-frame cycle
_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧")
_SPINNER_INTERVAL = 0.1  # 100ms per frame


class RunnerSpinner(Widget):
    """Animated spinner visible only while a run is active."""

    DEFAULT_CSS = """
    RunnerSpinner {
        height: 1;
        width: 100%;
        padding: 0 1;
        background: #0b1020;
        color: #3fa9f5;
        display: none;
    }
    RunnerSpinner.visible {
        display: block;
    }
    """

    active: reactive[bool] = reactive(False)
    activity_text: reactive[str] = reactive("")

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._frame_index = 0
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Static("", id="spinner-content")

    def on_mount(self) -> None:
        self._timer = self.set_interval(
            _SPINNER_INTERVAL, self._advance_frame, pause=True
        )

    def watch_active(self, value: bool) -> None:
        """Show/hide spinner and start/stop timer."""
        if value:
            self.add_class("visible")
            self._frame_index = 0
            if self._timer is not None:
                self._timer.resume()
        else:
            self.remove_class("visible")
            if self._timer is not None:
                self._timer.pause()

    def _advance_frame(self) -> None:
        """Advance the spinner animation by one frame."""
        self._frame_index = (self._frame_index + 1) % len(_SPINNER_FRAMES)
        frame = _SPINNER_FRAMES[self._frame_index]
        label = self.activity_text or "Working..."
        self.query_one("#spinner-content", Static).update(f" {frame} {label}")

    def update_from_state(self, state: AppState) -> None:
        """Update spinner visibility and text from app state."""
        if state.runner_status in (
            RunnerStatus.RUNNING,
            RunnerStatus.GENERATING,
            RunnerStatus.CANCEL_REQUESTED,
        ):
            cmd = state.current_command
            if state.runner_status == RunnerStatus.CANCEL_REQUESTED:
                label = f"Cancelling: {cmd}..." if cmd else "Cancelling..."
            elif state.runner_status == RunnerStatus.GENERATING:
                label = f"Generating: {cmd}..." if cmd else "Generating..."
            else:
                cmd_short = cmd.split()[0] if cmd and cmd.split() else cmd
                label = f"Running: {cmd_short}..." if cmd_short else "Running..."
            self.activity_text = label
            self.active = True
        else:
            self.active = False


__all__ = ["RunnerSpinner"]
