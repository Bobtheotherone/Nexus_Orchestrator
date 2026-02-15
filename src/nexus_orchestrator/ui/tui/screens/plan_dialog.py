"""Plan dialog — modal for selecting a spec file before running ``plan``.

File: src/nexus_orchestrator/ui/tui/screens/plan_dialog.py

Replaces the hardcoded "plan samples/specs/minimal_design_doc.md --mock" with
a modal that shows recent files and an input field for a custom path.
"""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

# User config for persisted last-used spec paths
_CONFIG_DIR = Path.home() / ".config" / "nexus_orchestrator"
_RECENT_SPECS_FILE = _CONFIG_DIR / "recent_specs.json"
_MAX_RECENT_SPECS = 5


def _load_recent_specs() -> list[str]:
    """Load recently-used spec paths."""
    try:
        if _RECENT_SPECS_FILE.exists():
            data = json.loads(_RECENT_SPECS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(s) for s in data[:_MAX_RECENT_SPECS]]
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_recent_spec(path: str) -> None:
    """Persist a spec path to recent list."""
    recent = _load_recent_specs()
    # Move to front if already present
    if path in recent:
        recent.remove(path)
    recent.insert(0, path)
    recent = recent[:_MAX_RECENT_SPECS]

    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        _RECENT_SPECS_FILE.write_text(
            json.dumps(recent, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


class PlanDialog(ModalScreen[str | None]):
    """Modal dialog for selecting a spec file to plan."""

    DEFAULT_CSS = """
    PlanDialog {
        align: center middle;
    }
    #plan-dialog-box {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: #0b1020;
        border: solid #3fa9f5;
        padding: 2;
    }
    #plan-dialog-title {
        text-style: bold;
        color: #3fa9f5;
        padding: 0 0 1 0;
    }
    #plan-dialog-input {
        width: 100%;
        margin: 1 0;
    }
    .recent-spec-btn {
        width: 100%;
        height: 1;
        margin: 0;
        background: transparent;
        border: none;
        content-align: left middle;
    }
    .recent-spec-btn:hover {
        background: #111830;
    }
    .recent-spec-btn:focus {
        background: #111830;
        text-style: bold;
    }
    #plan-dialog-flags {
        color: #7f8aa3;
        padding: 1 0;
    }
    #plan-dialog-actions {
        padding: 1 0 0 0;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="plan-dialog-box"):
            yield Static("Plan \u2014 Select Spec File", id="plan-dialog-title")

            # Recent specs
            recent = _load_recent_specs()
            if recent:
                yield Static("Recent:", classes="sidebar-heading")
                for i, spec_path in enumerate(recent):
                    short = spec_path
                    if len(short) > 55:
                        short = "..." + short[-52:]
                    yield Button(
                        short,
                        id=f"recent-spec-{i}",
                        classes="recent-spec-btn",
                    )

            # Manual path input
            yield Static("Or enter path:", classes="sidebar-heading")
            yield Input(
                placeholder="path/to/spec.md",
                id="plan-dialog-input",
            )
            yield Static("Flags: --mock (always appended in v1)", id="plan-dialog-flags")

    def on_mount(self) -> None:
        self.query_one("#plan-dialog-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle recent spec button click."""
        btn_id = event.button.id or ""
        if btn_id.startswith("recent-spec-"):
            idx = int(btn_id.split("-")[-1])
            recent = _load_recent_specs()
            if idx < len(recent):
                self._submit_spec(recent[idx])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle manual path submission."""
        path = event.value.strip()
        if path:
            self._submit_spec(path)

    def _submit_spec(self, spec_path: str) -> None:
        """Save recent and dismiss with the command."""
        _save_recent_spec(spec_path)
        self.dismiss(f"/run plan {spec_path} --mock")

    def action_cancel(self) -> None:
        self.dismiss(None)


__all__ = ["PlanDialog"]
