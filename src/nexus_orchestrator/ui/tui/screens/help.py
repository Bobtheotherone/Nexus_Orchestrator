"""Help screen overlay — F1 keybinding reference and command guide.

File: src/nexus_orchestrator/ui/tui/screens/help.py
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer
from textual.screen import ModalScreen
from textual.widgets import Static

# Command descriptions
COMMAND_DESCRIPTIONS: dict[str, str] = {
    "plan <spec.md>": "Plan work items from a spec document",
    "run [--mock]": "Run orchestration (mock or real)",
    "status": "Show current run status",
    "inspect": "Inspect the last run in detail",
    "export": "Export evidence bundle",
    "clean": "Clean ephemeral state (dry-run default)",
    "doctor": "Run diagnostic health checks",
    "config": "Show effective configuration",
}

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help overlay",
    "/clear": "Clear the transcript",
    "/quit": "Exit the TUI",
    "/theme": "Show current theme variables",
    "/copy": "Copy last output to clipboard (xclip/pbcopy/wl-copy)",
    "/export [path]": "Export transcript to file (auto-named if no path)",
    "/run <cmd>": "Run a CLI command (plan, status, inspect, etc.)",
    "/exec <cmd>": "Alias for /run",
}


class HelpScreen(ModalScreen[None]):
    """Full-screen help overlay showing keybindings and command reference."""

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-scroll {
        width: 80;
        max-width: 90%;
        height: 80%;
        background: #0b1020;
        border: solid #3fa9f5;
        padding: 2;
        overflow-y: auto;
    }
    #help-content {
        color: #72c7ff;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_help", "Close"),
        Binding("f1", "dismiss_help", "Close"),
        Binding("q", "dismiss_help", "Close"),
    ]

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="help-scroll"):
            yield Static(self._build_help_text(), id="help-content")

    def _build_help_text(self) -> str:
        lines = [
            "NEXUS Orchestrator -- Help",
            "",
            "Keybindings:",
            "  Tab / Shift+Tab  Cycle focus: sidebar <-> transcript <-> input",
            "  Ctrl+P           Open command palette (search all commands)",
            "  F1               Open this help overlay",
            "  Up / Down        Browse command history (when input is focused)",
            "  Enter            Execute command / activate quick action",
            "  Ctrl+C           Cancel task (double-press) / exit (double-press when idle)",
            "  Ctrl+E           Export transcript to file (~/.nexus/logs/)",
            "  Ctrl+Y           Copy transcript to clipboard (falls back to export)",
            "  Ctrl+Q           Quit immediately",
            "",
            "Copy & Export:",
            "  Ctrl+Y copies the last output (or full transcript) to the system",
            "  clipboard. If clipboard tools are unavailable (xclip, xsel, pbcopy,",
            "  wl-copy), the transcript is auto-exported to ~/.nexus/logs/ instead.",
            "  Ctrl+E always exports to a timestamped file in ~/.nexus/logs/.",
            "  Tip: Shift+drag selects text in most terminals for native copy.",
            "",
            "Cancel Safety:",
            "  While a task is running, Ctrl+C requires a double-press within 2s",
            "  to cancel. The first press shows a warning. This prevents accidental",
            "  cancellation when you intended to copy.",
            "",
            "Default Behavior:",
            "  Typing any text generates an engineering design document (.md)",
            "  using the strongest architect agent (claude-opus-4-6).",
            "  The document is displayed in the transcript and saved to designs/.",
            "",
            "Quick Actions:",
            "  Sidebar buttons execute commands immediately on click/Enter.",
            '  "Plan..." opens a dialog to select a spec file.',
            "",
            "CLI Commands (use /run <cmd>):",
        ]
        for cmd, desc in COMMAND_DESCRIPTIONS.items():
            lines.append(f"  {cmd:20s}  {desc}")
        lines.extend(["", "Slash Commands:"])
        for sc, desc in SLASH_COMMANDS.items():
            lines.append(f"  {sc:16s}  {desc}")
        lines.extend(
            [
                "",
                "Icon Theme:",
                "  Set NEXUS_TUI_ICONS to change icon style:",
                "    (unset/none)  Plain text labels, no decorations (default)",
                "    minimal       Simple ASCII markers for status indicators",
                "    emoji         Unicode emoji icons on labels",
                "    nerd          Nerd Font glyphs on labels",
                "",
                "Approval Mode:",
                "  Commands run immediately in mock mode. For real execution,",
                "  the orchestrator uses constraint gates (must/should/may).",
                "",
                "Tool Backends:",
                "  Codex CLI and Claude Code CLI as execution backends.",
                "  Run 'doctor' to check backend status.",
                "",
                "Press Escape, F1, or Q to close.",
                "",
            ]
        )
        return "\n".join(lines)

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


__all__ = ["HelpScreen"]
