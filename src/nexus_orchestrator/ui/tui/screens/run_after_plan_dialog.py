"""Run-after-plan dialog — prompts the user to run after a successful plan.

File: src/nexus_orchestrator/ui/tui/screens/run_after_plan_dialog.py

Shown automatically when ``plan <spec>`` finishes with exit code 0.
Returns the slash-command to execute, or ``None`` if the user cancels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class RunAfterPlanDialog(ModalScreen[str | None]):
    """Modal: 'Plan complete — run now?' with [Run] [Cancel]."""

    DEFAULT_CSS = """
    RunAfterPlanDialog {
        align: center middle;
    }
    #rap-box {
        width: 60;
        max-width: 80%;
        height: auto;
        background: #0b1020;
        border: round #3fa9f5;
        padding: 1 2;
    }
    #rap-title {
        text-style: bold;
        color: #3fa9f5;
        padding: 0 0 1 0;
    }
    #rap-spec {
        color: #7f8aa3;
        padding: 0 0 1 0;
    }
    #rap-buttons {
        padding: 1 0 0 0;
        height: auto;
        align: center middle;
    }
    #rap-buttons Button {
        margin: 0 1;
    }
    #btn-run {
        background: #1a6b35;
    }
    #btn-run:hover {
        background: #2a8b45;
    }
    #btn-cancel {
        background: #333;
    }
    #btn-cancel:hover {
        background: #555;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, spec_path: str) -> None:
        super().__init__()
        self._spec_path = spec_path

    def compose(self) -> ComposeResult:
        with Vertical(id="rap-box"):
            yield Static("Plan complete. Run now?", id="rap-title")
            yield Static(f"Spec: {self._spec_path}", id="rap-spec")
            with Horizontal(id="rap-buttons"):
                yield Button("Run", id="btn-run", variant="success")
                yield Button("Cancel", id="btn-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#btn-run", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if btn_id == "btn-run":
            self.dismiss(f"/run run {self._spec_path}")
        elif btn_id == "btn-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["RunAfterPlanDialog"]
