"""Sidebar widget — quick actions with real activation, backend status, recent runs.

File: src/nexus_orchestrator/ui/tui/widgets/sidebar.py

Quick actions are plain-text labels by default (no icon prefixes).
Focus/selection is communicated via CSS styling (background highlight + bold).
Icon prefixes are opt-in via NEXUS_TUI_ICONS=emoji or =nerd.
"""

from __future__ import annotations

import os

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

from nexus_orchestrator.ui.tui.state import AppState, BackendInfo, RecentRun

# ---------------------------------------------------------------------------
# Icon theme — none by default, opt-in via NEXUS_TUI_ICONS
#
# Modes:
#   none (default) — no icons, no decorations, plain text everywhere
#   minimal        — simple ASCII markers for status indicators only
#   emoji          — Unicode emoji/symbol icons on labels + decorations
#   nerd           — Nerd Font glyphs on labels + decorations
# ---------------------------------------------------------------------------

# (none, minimal, emoji, nerd) keyed by semantic name
_ICON_TABLE: dict[str, tuple[str, str, str, str]] = {
    "play":    ("", ">", "\u25b6",     "\uf04b"),
    "info":    ("", "i", "\u2139",     "\uf129"),
    "search":  ("", "?", "\U0001f50d", "\uf002"),
    "package": ("", "=", "\U0001f4e6", "\uf466"),
    "broom":   ("", "~", "\U0001f9f9", "\uf51a"),
    "medkit":  ("", "+", "\U0001f3e5", "\uf0fa"),
    "gear":    ("", "*", "\u2699",     "\uf013"),
    "check":   ("", "+", "\u2713",     "\uf00c"),
    "cross":   ("", "-", "\u2717",     "\uf00d"),
    "branch":  ("", "/", "\u2387",     "\ue725"),
    "run":     ("", ">", "\u25b6",     "\uf04b"),
    "stop":    ("", "#", "\u23f9",     "\uf04d"),
    "cancel":  ("", "x", "\u2716",     "\uf00d"),
    "ready":   ("", "+", "\u2714",     "\uf00c"),
    "star1":   ("", "-", "\u2726",     "*"),
    "star2":   ("", "-", "\u2727",     "."),
    "dot":     ("", "-", "\u00b7",     "."),
    "diamond": ("", "-", "\u22c6",     "*"),
}


def _icon_mode() -> int:
    """Return 0=none, 1=minimal, 2=emoji, 3=nerd based on NEXUS_TUI_ICONS."""
    val = os.environ.get("NEXUS_TUI_ICONS", "").strip().lower()
    if val == "minimal":
        return 1
    if val in ("emoji", "unicode"):
        return 2
    if val in ("nerd", "nerdfonts", "nerd-font", "nerd_font"):
        return 3
    return 0


def icon(name: str) -> str:
    """Get icon glyph by semantic name. Returns '' in none mode."""
    entry = _ICON_TABLE.get(name)
    if entry is None:
        return ""
    return entry[_icon_mode()]


def star_sep() -> str:
    """Decorative separator. Returns '' in none/minimal modes."""
    mode = _icon_mode()
    if mode <= 1:
        return ""
    return f"  {icon('star1')}  {icon('star2')}  {icon('dot')}  {icon('diamond')}  "


# ---------------------------------------------------------------------------
# Quick action definitions
# ---------------------------------------------------------------------------

# (id, icon_key, label_text, command, needs_args)
_QUICK_ACTION_DEFS: list[tuple[str, str, str, str, bool]] = [
    ("qa-plan",    "play",    "Plan\u2026",   "/run plan",      True),
    ("qa-run",     "play",    "Run --mock",   "/run run --mock", False),
    ("qa-status",  "info",    "Status",       "/run status",    False),
    ("qa-inspect", "search",  "Inspect",      "/run inspect",   False),
    ("qa-export",  "package", "Export",        "/run export",    False),
    ("qa-clean",   "broom",   "Clean",         "/run clean",     False),
    ("qa-doctor",  "medkit",  "Doctor",        "/run doctor",    False),
    ("qa-config",  "gear",    "Config",        "/run config",    False),
]

# The plain-text labels that should appear by default (no prefixes)
QUICK_ACTION_LABELS: list[str] = [text for _, _, text, _, _ in _QUICK_ACTION_DEFS]


def _build_quick_actions() -> list[tuple[str, str, str, bool]]:
    """Build (id, label, command, needs_args) using the active icon theme.

    In none/minimal mode: labels are plain text (no icon prefix).
    In emoji/nerd mode: labels are prefixed with the icon glyph.
    """
    mode = _icon_mode()
    result: list[tuple[str, str, str, bool]] = []
    for qid, ic, text, cmd, needs_args in _QUICK_ACTION_DEFS:
        if mode >= 2:
            glyph = icon(ic)
            label = f"{glyph} {text}" if glyph else text
        else:
            label = text
        result.append((qid, label, cmd, needs_args))
    return result


# ---------------------------------------------------------------------------
# Sidebar widget
# ---------------------------------------------------------------------------


class SidebarWidget(Widget):
    """Left sidebar with quick action buttons, backend status, and recent runs."""

    DEFAULT_CSS = """
    SidebarWidget {
        dock: left;
        width: 26;
        padding: 1;
        overflow-y: auto;
    }
    .sidebar-heading {
        text-style: bold;
        padding: 1 0 0 0;
        margin: 0 0 1 0;
    }
    .sidebar-rule {
        color: #2a3050;
        height: 1;
        margin: 0;
    }
    .qa-button {
        width: 100%;
        height: 1;
        margin: 0;
        padding: 0 1;
        text-style: none;
        background: transparent;
        border: none;
        content-align: left middle;
    }
    .qa-button:hover {
        background: #111830;
        color: #72c7ff;
    }
    .qa-button:focus {
        background: #111830;
        color: #3fa9f5;
        text-style: bold;
    }
    .qa-button.-active {
        background: #1a2550;
    }
    #sidebar-backends {
        height: auto;
        color: #7f8aa3;
        padding: 0 1;
    }
    #sidebar-recent {
        height: auto;
        color: #7f8aa3;
        padding: 0 1;
    }
    """

    class QuickActionActivated(Message):
        """Emitted when a quick action button is pressed."""

        def __init__(self, command: str, *, needs_args: bool = False) -> None:
            self.command = command
            self.needs_args = needs_args
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("Quick Actions", classes="sidebar-heading")

        self._qa_list = _build_quick_actions()
        for action_id, label, _command, _needs_args in self._qa_list:
            yield Button(label, id=action_id, classes="qa-button")

        yield Static("Backends", classes="sidebar-heading")
        yield Static("(detecting\u2026)", id="sidebar-backends")

        yield Static("Recent", classes="sidebar-heading")
        yield Static("(none yet)", id="sidebar-recent")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle quick action button press — emit activation event."""
        for action_id, _label, command, needs_args in self._qa_list:
            if event.button.id == action_id:
                self.post_message(
                    self.QuickActionActivated(command, needs_args=needs_args)
                )
                return

    def update_from_state(self, state: AppState) -> None:
        """Update sidebar contents from app state."""
        self._update_backends(state.backends)
        self._update_recent(state.recent_runs)

    def _update_backends(self, backends: list[BackendInfo]) -> None:
        try:
            widget = self.query_one("#sidebar-backends", Static)
        except Exception:
            return

        if backends:
            lines = []
            for b in backends:
                lines.append(_format_backend_line(b))
            widget.update("\n".join(lines))
        else:
            widget.update("(detecting...)")

    def _update_recent(self, recent_runs: list[RecentRun]) -> None:
        try:
            widget = self.query_one("#sidebar-recent", Static)
        except Exception:
            return

        if recent_runs:
            lines = []
            for run in recent_runs[-5:]:
                cmd_short = run.command.split()[0] if run.command.split() else run.command
                badge = icon("check") if run.exit_code == 0 else icon("cross")
                if badge:
                    lines.append(f"{badge} {cmd_short} ({run.exit_code})")
                else:
                    lines.append(f"{cmd_short} ({run.exit_code})")
            widget.update("\n".join(lines))
        else:
            widget.update("(none yet)")


_BACKEND_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude CLI",
    "codex": "Codex CLI",
    "anthropic": "Anthropic API",
    "openai": "OpenAI API",
}


def _format_backend_line(b: BackendInfo) -> str:
    """Format a single backend for sidebar display with auth status."""
    display = _BACKEND_DISPLAY_NAMES.get(b.name, b.name)

    if not b.available:
        if b.auth_mode == "local_cli":
            return f"{display}: not installed"
        if b.auth_mode == "api_key":
            if b.has_api_key is False and (getattr(b, "sdk_installed", None) is False
                                           or not b.has_api_key):
                return f"{display}: not configured"
            return f"{display}: not configured"
        return f"{display}: unavailable"

    if b.auth_mode == "local_cli":
        ver = b.version or ""
        if b.logged_in:
            status = icon("check") + " " if icon("check") else ""
            return f"{display}: {status}ready" + (f" ({ver})" if ver else "")
        return f"{display}: login required"

    if b.auth_mode == "api_key":
        status = icon("check") + " " if icon("check") else ""
        return f"{display}: {status}ready"

    ver = b.version or "?"
    return f"{display}: {ver}"


__all__ = ["QUICK_ACTION_LABELS", "SidebarWidget", "icon", "star_sep"]
