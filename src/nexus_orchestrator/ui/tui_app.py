"""Full-screen interactive TUI for nexus-orchestrator using Textual.

File: src/nexus_orchestrator/ui/tui_app.py

Purpose
- Provide a Claude Code / Codex-style interactive terminal dashboard.
- Header with compact Auto mascot, sidebar with quick actions, scrollable transcript, command input.
- First-run onboarding wizard with credential capture and animated splash.
- Command palette (Ctrl+P), help overlay (F1), crash recovery banner.
- Double Ctrl+C behavior for safe exit.

Non-functional requirements
- Offline-first: no network required for launch.
- Deterministic: stable rendering, deterministic animation frames (no randomness).
- Respects NO_COLOR and --no-color flag.
- Never hard-codes model IDs; routing is config + catalog driven.
"""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import time
from datetime import UTC, datetime
from pathlib import Path

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits
from textual.command import Provider as CommandProvider
from textual.containers import Horizontal, ScrollableContainer
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from nexus_orchestrator.synthesis_plane.providers.tool_detection import (
    detect_all_backends,
    detect_tool_backend,
)
from nexus_orchestrator.ui.onboarding import (
    API_KEY_NOTICE,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_KEY_URLS,
    TOOL_BACKEND_DISPLAY,
    TOOL_BACKEND_NOTICE,
    TOOL_INSTALL_HINTS,
    detect_env_credentials,
    is_onboarding_complete,
    mark_onboarding_complete,
    open_provider_key_page,
    redact_key,
    store_credential,
    validate_key_format,
)

# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# Maximum header mascot dimensions
HEADER_MASCOT_MAX_WIDTH = 14
HEADER_MASCOT_MAX_HEIGHT = 3


def _load_asset(filename: str) -> str:
    """Load an ASCII art asset file as plain text."""
    path = _ASSETS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").rstrip("\n")
    return f"(asset not found: {filename})"


def _load_auto_header() -> str:
    """Load Auto mascot from the purpose-built header asset.

    Returns a block <= HEADER_MASCOT_MAX_WIDTH chars wide
    and <= HEADER_MASCOT_MAX_HEIGHT lines tall.
    """
    raw = _load_asset("auto_header_ascii.txt")
    if raw.startswith("(asset not found"):
        return "(auto)"
    lines = raw.splitlines()[:HEADER_MASCOT_MAX_HEIGHT]
    return "\n".join(line[:HEADER_MASCOT_MAX_WIDTH] for line in lines)


# ---------------------------------------------------------------------------
# Deterministic splash animation helpers
# ---------------------------------------------------------------------------

# Fixed seed for deterministic "star twinkle" positions
_TWINKLE_SEED: list[int] = [
    7,
    23,
    41,
    59,
    73,
    2,
    37,
    61,
    11,
    47,
    83,
    19,
    53,
    67,
    3,
    29,
    71,
    13,
    43,
    89,
    5,
    31,
    79,
    17,
    97,
    37,
    63,
    8,
    51,
    77,
    22,
    44,
    66,
    88,
    14,
    36,
    58,
    80,
    9,
    27,
]

# Blue gradient palette for sweep animation (dark to bright)
_GRADIENT_COLORS: list[str] = [
    "#0a1628",
    "#0f2040",
    "#153060",
    "#1a4080",
    "#2060a0",
    "#2878b8",
    "#3090d0",
    "#3fa9f5",
    "#5cbcff",
    "#72c7ff",
    "#90d4ff",
    "#b0e0ff",
]

_TWINKLE_CHARS = ["\u00b7", "\u22c6", "\u2727", "\u2726"]  # · ⋆ ✧ ✦

# Compact fallback when terminal is too small for full splash
_COMPACT_SPLASH = "\u2726 NEXUS \u2726\nAgentic AI Orchestrator\n\nResize terminal for full splash"

# Minimum terminal size for full splash display
_SPLASH_MIN_WIDTH = 80
_SPLASH_MIN_HEIGHT = 30


def splash_frame_count() -> int:
    """Return total animation frames for the splash (deterministic)."""
    return 30  # ~1.5s at 50ms/frame


def render_splash_frame(splash_text: str, frame: int, total_frames: int) -> str:
    """Render a single animation frame of the splash.

    Uses a blue gradient sweep (line-by-line reveal) with deterministic
    twinkle overlay. Frame 0 = all dark, frame total-1 = fully revealed.
    """
    lines = splash_text.splitlines()
    n_lines = len(lines)
    if n_lines == 0:
        return ""

    # Progress 0.0 to 1.0
    progress = min(1.0, frame / max(1, total_frames - 1))

    # How many lines are fully revealed
    revealed = int(progress * n_lines)

    result_lines: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            result_lines.append(line)
            continue

        if i < revealed:
            # Fully revealed — bright color with optional twinkle
            twinkle = ""
            seed_idx = (frame + i) % len(_TWINKLE_SEED)
            if _TWINKLE_SEED[seed_idx] % 7 == 0 and progress > 0.4:
                char_idx = (frame + i) % len(_TWINKLE_CHARS)
                twinkle = f" {_TWINKLE_CHARS[char_idx]}"
            result_lines.append(f"[#72c7ff]{line}[/]{twinkle}")
        elif i == revealed:
            # Sweep line — gradient color
            grad_idx = min(len(_GRADIENT_COLORS) - 1, int(progress * len(_GRADIENT_COLORS)))
            color = _GRADIENT_COLORS[grad_idx]
            result_lines.append(f"[{color}]{line}[/]")
        else:
            # Not yet revealed — dark
            result_lines.append(f"[#0a1628]{line}[/]")

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Star separators
# ---------------------------------------------------------------------------

STAR_SEP = "  \u2726  \u2727  \u00b7  \u22c6  "  # ✦  ✧  ·  ⋆


# ---------------------------------------------------------------------------
# Command runner — reuses CLI handlers without exiting the process
# ---------------------------------------------------------------------------


def run_command(command_str: str) -> tuple[int, str, str]:
    """Run a nexus CLI command string, capturing stdout/stderr.

    Returns (exit_code, stdout, stderr).
    """
    from nexus_orchestrator.ui.cli import run_cli

    argv = shlex.split(command_str)

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    exit_code = 1

    try:
        with (
            contextlib.redirect_stdout(stdout_buf),
            contextlib.redirect_stderr(stderr_buf),
        ):
            exit_code = run_cli(argv)
    except SystemExit as exc:
        code = exc.code
        exit_code = int(code) if isinstance(code, int) else 1
    except Exception as exc:  # noqa: BLE001
        stderr_buf.write(f"error: {exc}\n")
        exit_code = 4

    return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------

SLASH_COMMANDS = {
    "/help": "Show help overlay (keybindings + examples).",
    "/clear": "Clear the transcript.",
    "/quit": "Exit the TUI.",
    "/theme": "Show current theme variables.",
    "/copy": "Copy last output to clipboard (if supported).",
}

NEXUS_COMMANDS = ["plan", "run", "status", "inspect", "export", "clean", "doctor", "config"]

# Command descriptions for palette
_COMMAND_DESCRIPTIONS: dict[str, str] = {
    "plan": "Plan work items from a spec document",
    "run --mock": "Run orchestration in mock mode",
    "status": "Show current run status",
    "inspect": "Inspect the last run in detail",
    "export": "Export evidence bundle",
    "clean": "Clean ephemeral state (dry-run default)",
    "doctor": "Run diagnostic health checks",
    "config": "Show effective configuration",
}


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class MascotWidget(Static):
    """Displays a compact Auto mascot in the header area."""

    def on_mount(self) -> None:
        mascot = _load_auto_header()
        self.update(mascot)


class HeaderBar(Static):
    """Top header bar with compact mascot + branding title."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-inner"):
            yield MascotWidget(id="header-mascot")
            yield Static(
                f"{STAR_SEP}NEXUS \u00b7 Agentic AI Orchestrator{STAR_SEP}",
                id="header-title",
            )


class SidebarWidget(Widget):
    """Left sidebar with quick actions, backend status, and recent runs."""

    def compose(self) -> ComposeResult:
        yield Static(f"{STAR_SEP}", classes="star-line")
        yield Static("Quick Actions", classes="sidebar-heading")
        yield ListView(
            ListItem(Label("plan"), id="sb-plan"),
            ListItem(Label("run --mock"), id="sb-run"),
            ListItem(Label("status"), id="sb-status"),
            ListItem(Label("inspect"), id="sb-inspect"),
            ListItem(Label("export"), id="sb-export"),
            ListItem(Label("clean"), id="sb-clean"),
            ListItem(Label("doctor"), id="sb-doctor"),
            ListItem(Label("config"), id="sb-config"),
            id="sidebar-list",
        )
        yield Static(f"\n{STAR_SEP}", classes="star-line")
        yield Static("Backends", classes="sidebar-heading")
        yield Static("  (detecting...)", id="sidebar-backends")
        yield Static(f"\n{STAR_SEP}", classes="star-line")
        yield Static("Recent", classes="sidebar-heading")
        yield Static("  (none yet)", id="sidebar-recent")
        yield Static(f"\n{STAR_SEP}", classes="star-line")
        yield Static("", id="sidebar-status")

    def on_mount(self) -> None:
        self._populate_backends()

    def _populate_backends(self) -> None:
        """Detect tool backends and update the sidebar panel."""
        try:
            backends = detect_all_backends()
            if backends:
                lines = []
                for b in backends:
                    display = TOOL_BACKEND_DISPLAY.get(b.name, b.name)
                    ver = b.version or "?"
                    lines.append(f"  {display}: {ver}")
                text = "\n".join(lines)
            else:
                text = "  (none detected)"
            self.query_one("#sidebar-backends", Static).update(text)
        except NoMatches:
            pass


class TranscriptWidget(Static):
    """Scrollable transcript area showing command output."""

    _lines: list[str] = []

    def on_mount(self) -> None:
        self._lines = []

    def append_output(self, text: str, *, style: str = "") -> None:
        """Append text to the transcript."""
        if style:
            self._lines.append(f"[{style}]{text}[/{style}]")
        else:
            self._lines.append(text)
        self.update("\n".join(self._lines))

    def append_command_header(self, command: str) -> None:
        """Add a command header with timestamp."""
        ts = datetime.now(UTC).strftime("%H:%M:%S")
        self._lines.append(f"\n[bold]nexus > {command}[/bold]  [{ts}]")
        self._lines.append("\u2500" * 60)
        self.update("\n".join(self._lines))

    def append_exit_badge(self, exit_code: int, *, no_color: bool = False) -> None:
        """Add exit code badge."""
        self._lines.append("\u2500" * 60)
        if no_color:
            badge = f"Exit {exit_code}"
        elif exit_code == 0:
            badge = "[#4ec990]OK (0)[/]"
        else:
            badge = f"[#e05555]FAIL ({exit_code})[/]"
        self._lines.append(badge)
        self._lines.append("")
        self.update("\n".join(self._lines))

    def append_stderr(self, text: str, *, no_color: bool = False) -> None:
        """Add stderr in muted blue style (not harsh red)."""
        if no_color:
            self._lines.append(text)
        else:
            self._lines.append(f"[#7f8aa3]{text}[/]")
        self.update("\n".join(self._lines))

    def clear_transcript(self) -> None:
        """Clear all transcript content."""
        self._lines = []
        self.update("")


class CommandInput(Input):
    """Bottom command input with nexus prompt, history, and autocomplete."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._history: list[str] = []
        self._history_index: int = -1
        self._stashed_value: str = ""

    def push_history(self, command: str) -> None:
        """Add a command to history."""
        if command and (not self._history or self._history[-1] != command):
            self._history.append(command)
        self._history_index = -1

    def on_key(self, event: events.Key) -> None:
        """Handle Up/Down for command history navigation."""
        if event.key == "up":
            event.prevent_default()
            if not self._history:
                return
            if self._history_index == -1:
                self._stashed_value = self.value
                self._history_index = len(self._history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            self.value = self._history[self._history_index]
            self.cursor_position = len(self.value)
        elif event.key == "down":
            event.prevent_default()
            if self._history_index == -1:
                return
            if self._history_index < len(self._history) - 1:
                self._history_index += 1
                self.value = self._history[self._history_index]
            else:
                self._history_index = -1
                self.value = self._stashed_value
            self.cursor_position = len(self.value)


# ---------------------------------------------------------------------------
# Crash Recovery Banner
# ---------------------------------------------------------------------------


class CrashRecoveryBanner(Widget):
    """Banner shown when a previous TUI crash was detected."""

    class ViewReport(Message):
        """User wants to view the crash report."""

    class DismissBanner(Message):
        """User dismissed the banner."""

    def __init__(self, crash_data: dict[str, object], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._crash_data = crash_data

    def compose(self) -> ComposeResult:
        ts = str(self._crash_data.get("timestamp", "unknown"))
        exc_type = str(self._crash_data.get("exception_type", "unknown"))
        yield Static(
            f"[bold]Recovered from crash[/bold] ({exc_type} at {ts})  [V] View report  [D] Dismiss",
            id="crash-banner-text",
        )

    def on_key(self, event: events.Key) -> None:
        if event.key == "v":
            self.post_message(self.ViewReport())
        elif event.key == "d":
            self.post_message(self.DismissBanner())

    def get_crash_data(self) -> dict[str, object]:
        return self._crash_data


# ---------------------------------------------------------------------------
# Splash Screen (animated, first-run only)
# ---------------------------------------------------------------------------


class SplashScreen(Widget):
    """Animated NEXUS splash screen shown on first successful onboarding.

    Features:
    - Blue gradient sweep across ASCII art (line-by-line)
    - Deterministic twinkle star overlay
    - Centered with scroll fallback for small terminals
    - Duration ~1.5s (30 frames at 50ms)
    """

    _frame: reactive[int] = reactive(0)
    _total_frames: int = 0
    _splash_text: str = ""
    _timer_handle: object = None
    _compact_mode: bool = False

    class Dismissed(Message):
        """Sent when splash animation completes."""

    def on_mount(self) -> None:
        self._splash_text = _load_asset("nexus_ascii.txt")
        self._total_frames = splash_frame_count()
        self._frame = 0

        # Check terminal size for compact fallback
        try:
            w = self.app.size.width
            h = self.app.size.height
            if w < _SPLASH_MIN_WIDTH or h < _SPLASH_MIN_HEIGHT:
                self._compact_mode = True
        except Exception:  # noqa: BLE001
            pass

        if self._compact_mode:
            self._show_compact()
        else:
            self._render_current_frame()
            self._timer_handle = self.set_interval(0.05, self._advance_frame)

    def _show_compact(self) -> None:
        """Show compact splash for small terminals."""
        content = self.query_one("#splash-content", Static)
        content.update(f"[#3fa9f5]{_COMPACT_SPLASH}[/]")
        self.set_timer(1.5, self._dismiss)

    def _advance_frame(self) -> None:
        self._frame += 1
        if self._frame >= self._total_frames:
            if self._timer_handle is not None:
                with contextlib.suppress(Exception):
                    self._timer_handle.stop()  # type: ignore[union-attr]
            self.set_timer(0.5, self._dismiss)
            return
        self._render_current_frame()

    def _render_current_frame(self) -> None:
        content = self.query_one("#splash-content", Static)
        rendered = render_splash_frame(self._splash_text, self._frame, self._total_frames)
        content.update(rendered)

    def _dismiss(self) -> None:
        self.post_message(self.Dismissed())

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="splash-scroll"):
            yield Static("", id="splash-content")


# ---------------------------------------------------------------------------
# Help Screen (F1 overlay)
# ---------------------------------------------------------------------------


class HelpScreen(ModalScreen[None]):
    """Full-screen help overlay showing keybindings and command reference."""

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
            f"{STAR_SEP}",
            "NEXUS Orchestrator \u2014 Help",
            f"{STAR_SEP}",
            "",
            "Keybindings:",
            "  Tab / Shift+Tab  Cycle focus: sidebar \u2194 transcript \u2194 command input",
            "  Ctrl+P           Open command palette (search all commands)",
            "  F1               Open this help overlay",
            "  Up / Down        Browse command history (when input is focused)",
            "  Ctrl+C           Cancel task (1st press) / exit (2nd within 2s)",
            "  Ctrl+Q           Quit immediately",
            "  Enter            Execute command or select sidebar item",
            "",
            "CLI Commands:",
        ]
        for cmd, desc in _COMMAND_DESCRIPTIONS.items():
            lines.append(f"  {cmd:16s}  {desc}")
        lines.extend(
            [
                "",
                "Slash Commands:",
            ]
        )
        for sc, desc in SLASH_COMMANDS.items():
            lines.append(f"  {sc:10s}  {desc}")
        lines.extend(
            [
                "",
                "Tool Backends:",
                "  NEXUS can use Codex CLI and Claude Code CLI as execution backends.",
                "  These use your existing subscriptions (no API keys needed).",
                "  Run 'doctor' to check backend status.",
                "",
                "Press Escape, F1, or Q to close.",
                "",
            ]
        )
        return "\n".join(lines)

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Command Palette Provider
# ---------------------------------------------------------------------------


class NexusCommandProvider(CommandProvider):
    """Provides all NEXUS commands and slash commands for the command palette."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        # Nexus CLI commands
        for cmd, desc in _COMMAND_DESCRIPTIONS.items():
            score = matcher.match(f"{cmd} {desc}")
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(cmd),
                    self._make_callback(cmd),
                    help=desc,
                )

        # Slash commands
        for slash_cmd, desc in SLASH_COMMANDS.items():
            score = matcher.match(f"{slash_cmd} {desc}")
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(slash_cmd),
                    self._make_callback(slash_cmd),
                    help=desc,
                )

    def _make_callback(self, cmd: str):  # type: ignore[no-untyped-def]
        async def callback() -> None:
            try:
                inp = self.app.query_one("#command-input", CommandInput)
                inp.value = cmd
                inp.focus()
                inp.cursor_position = len(cmd)
            except NoMatches:
                pass

        return callback


# ---------------------------------------------------------------------------
# Onboarding Widget (premium step-by-step wizard)
# ---------------------------------------------------------------------------


class OnboardingComplete(Message):
    """Message sent when onboarding is finished."""

    def __init__(self, providers: list[str], *, show_splash: bool = False) -> None:
        self.providers = providers
        self.show_splash = show_splash
        super().__init__()


class OnboardingWidget(Widget):
    """First-run onboarding wizard — subscription-first with tool backends.

    Step-by-step flow:
    1) Mock mode only (no keys or tools needed)
    2) Use Codex CLI (OpenAI subscription)
    3) Use Claude Code CLI (Anthropic subscription)
    4) Use Both Codex + Claude Code (recommended)
    5) Advanced: API keys (pay-as-you-go)
    """

    stage: reactive[str] = reactive("welcome")
    selected_provider: reactive[str] = reactive("")
    selected_backend: reactive[str] = reactive("")
    configured_providers: reactive[list[str]] = reactive(list, init=False)
    configured_backends: reactive[list[str]] = reactive(list, init=False)

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.configured_providers = []
        self.configured_backends = []
        self._pending_key: str = ""

    def compose(self) -> ComposeResult:
        yield Static("", id="onboard-content")
        yield Input(placeholder="Enter your choice...", id="onboard-input")

    def on_mount(self) -> None:
        # Check for env var credentials first
        env_providers = detect_env_credentials()
        if env_providers:
            self.configured_providers = list(env_providers)
            mark_onboarding_complete(env_providers)
            content = self.query_one("#onboard-content", Static)
            names = ", ".join(PROVIDER_DISPLAY_NAMES.get(p, p) for p in env_providers)
            content.update(
                f"Detected API keys from environment for: {names}\n\n"
                "Onboarding complete! Starting NEXUS..."
            )
            self.stage = "done"
            self.set_timer(1.5, self._finish_onboarding)
            return
        self._show_welcome()

    def _show_welcome(self) -> None:
        """Show the welcome / backend selection menu."""
        content = self.query_one("#onboard-content", Static)
        content.update(
            f"{STAR_SEP}\n"
            "Welcome to NEXUS Orchestrator\n"
            f"{STAR_SEP}\n\n"
            f"{TOOL_BACKEND_NOTICE}\n\n"
            "Choose how to get started:\n\n"
            "  [1] Use Mock Mode only (no external tools required)\n"
            "  [2] Use OpenAI Codex (requires codex installed + logged in)\n"
            "  [3] Use Claude Code (requires claude installed + logged in)\n"
            "  [4] Use Both Codex + Claude Code (recommended)\n"
            "  [5] Advanced: Use API Keys (pay-as-you-go)\n\n"
            "Enter choice (1/2/3/4/5):"
        )
        inp = self.query_one("#onboard-input", Input)
        inp.password = False
        inp.placeholder = "Enter your choice..."
        self.stage = "welcome"

    # ----- Tool backend detection stages -----

    def _show_tool_detect(self, backend: str) -> None:
        """Detect a CLI tool backend and show result."""
        self.selected_backend = backend
        display = TOOL_BACKEND_DISPLAY.get(backend, backend)
        content = self.query_one("#onboard-content", Static)

        info = detect_tool_backend(backend)
        if info is not None:
            version_str = info.version or "unknown"
            content.update(
                f"Found {display}\n\n"
                f"  Binary: {info.binary_path}\n"
                f"  Version: {version_str}\n\n"
                "  [1] Use this backend\n"
                "  [2] Back to menu\n\n"
                "Enter choice (1/2):"
            )
            self.stage = "tool_found"
        else:
            hint = TOOL_INSTALL_HINTS.get(backend, "")
            content.update(
                f"{display} not found on PATH.\n\n"
                f"Install: {hint}\n\n"
                "After installing, make sure the tool is on your PATH\n"
                "and you are logged in.\n\n"
                "  [1] Re-check\n"
                "  [2] Back to menu\n\n"
                "Enter choice (1/2):"
            )
            self.stage = "tool_not_found"

    def _show_tool_detect_both(self) -> None:
        """Detect both Codex and Claude CLI backends and show combined results."""
        content = self.query_one("#onboard-content", Static)
        codex_info = detect_tool_backend("codex")
        claude_info = detect_tool_backend("claude")

        lines = ["Detecting tool backends...\n"]
        found_any = False

        if codex_info is not None:
            ver = codex_info.version or "unknown"
            lines.append(f"  Codex CLI: found ({ver})")
            found_any = True
        else:
            hint = TOOL_INSTALL_HINTS.get("codex", "")
            lines.append(f"  Codex CLI: not found (install: {hint})")

        if claude_info is not None:
            ver = claude_info.version or "unknown"
            lines.append(f"  Claude Code CLI: found ({ver})")
            found_any = True
        else:
            hint = TOOL_INSTALL_HINTS.get("claude", "")
            lines.append(f"  Claude Code CLI: not found (install: {hint})")

        if found_any:
            lines.append("\n  [1] Use all detected backends")
            lines.append("  [2] Back to menu\n")
            lines.append("Enter choice (1/2):")
            self.stage = "tool_both_found"
        else:
            lines.append("\nNo tool backends found. Install at least one and try again.")
            lines.append("\n  [1] Re-check")
            lines.append("  [2] Back to menu\n")
            lines.append("Enter choice (1/2):")
            self.stage = "tool_both_not_found"

        # Store detection results for use after selection
        self._both_codex = codex_info
        self._both_claude = claude_info

        content.update("\n".join(lines))

    def _configure_backend(self, backend: str) -> None:
        """Mark a tool backend as configured and show next options."""
        if backend not in self.configured_backends:
            self.configured_backends = [*self.configured_backends, backend]

        display = TOOL_BACKEND_DISPLAY.get(backend, backend)
        content = self.query_one("#onboard-content", Static)
        content.update(
            f"{display} configured!\n\n"
            "Configure another backend?\n\n"
            "  [1] Add Codex CLI\n"
            "  [2] Add Claude Code CLI\n"
            "  [3] Done - start NEXUS\n\n"
            "Enter choice (1/2/3):"
        )
        self.stage = "backend_menu_again"

    def _configure_both_backends(self) -> None:
        """Configure all detected backends from the 'both' detection."""
        if self._both_codex is not None and "codex" not in self.configured_backends:
            self.configured_backends = [*self.configured_backends, "codex"]
        if self._both_claude is not None and "claude" not in self.configured_backends:
            self.configured_backends = [*self.configured_backends, "claude"]
        self._finish_onboarding()

    # ----- Advanced: API key stages -----

    def _show_api_key_menu(self) -> None:
        """Show the advanced API key configuration menu."""
        content = self.query_one("#onboard-content", Static)
        content.update(
            "Advanced: API Key Configuration\n\n"
            f"{API_KEY_NOTICE}\n\n"
            "  [1] Configure OpenAI API key\n"
            "  [2] Configure Anthropic API key\n"
            "  [3] I'll set env vars myself (OPENAI_API_KEY / ANTHROPIC_API_KEY)\n"
            "  [4] Back to main menu\n\n"
            "Enter choice (1/2/3/4):"
        )
        self.stage = "api_key_menu"

    def _show_key_input(self, provider: str) -> None:
        """Show API key configuration for a provider."""
        self.selected_provider = provider
        content = self.query_one("#onboard-content", Static)
        display = PROVIDER_DISPLAY_NAMES.get(provider, provider)
        url = PROVIDER_KEY_URLS.get(provider, "")

        content.update(
            f"Configure {display} API Key\n\n"
            f"{API_KEY_NOTICE}\n\n"
            f"  [a] Paste API key\n"
            f"  [b] Open browser to get a key ({url})\n"
            f"  [c] Back to menu\n\n"
            "Enter choice (a/b/c):"
        )
        self.stage = "key_method"

    def _show_paste_prompt(self) -> None:
        """Show prompt to paste API key."""
        content = self.query_one("#onboard-content", Static)
        display = PROVIDER_DISPLAY_NAMES.get(self.selected_provider, self.selected_provider)
        content.update(
            f"Paste your {display} API key below.\n"
            "(The key will be hidden as you type)\n\n"
            "After pasting, press Enter to confirm."
        )
        inp = self.query_one("#onboard-input", Input)
        inp.password = True
        inp.placeholder = "Paste API key here..."
        self.stage = "paste_key"

    def _process_key(self, key: str) -> None:
        """Validate, store the API key, and continue."""
        content = self.query_one("#onboard-content", Static)
        provider = self.selected_provider
        display = PROVIDER_DISPLAY_NAMES.get(provider, provider)

        if not key.strip():
            content.update("No key entered. Returning to menu...")
            self.set_timer(1.0, self._show_welcome)
            return

        key = key.strip()

        # Local format validation (no network call)
        valid, msg = validate_key_format(provider, key)
        if not valid:
            content.update(
                f"Key format check: {msg}\n\n"
                "  [a] Try again\n"
                "  [b] Store anyway (skip format check)\n"
                "  [c] Back to menu\n\n"
                "Enter choice (a/b/c):"
            )
            self.stage = "key_invalid"
            self._pending_key = key
            return

        self._store_and_continue(provider, display, key)

    def _store_and_continue(self, provider: str, display: str, key: str) -> None:
        """Store key and show success + next options."""
        storage_method = store_credential(provider, key)
        redacted = redact_key(key)
        self.configured_providers = [*self.configured_providers, provider]

        content = self.query_one("#onboard-content", Static)
        content.update(
            f"{display} key stored via {storage_method}: {redacted}\n\n"
            "Configure another provider?\n\n"
            "  [1] Configure OpenAI\n"
            "  [2] Configure Anthropic\n"
            "  [3] Done - start NEXUS\n\n"
            "Enter choice (1/2/3):"
        )
        inp = self.query_one("#onboard-input", Input)
        inp.password = False
        inp.placeholder = "Enter your choice..."
        self.stage = "provider_menu_again"

    # ----- Finish -----

    def _finish_onboarding(self) -> None:
        """Mark onboarding complete and signal the app."""
        first_time = not is_onboarding_complete()
        backends = self.configured_backends if self.configured_backends else None
        if self.configured_providers or self.configured_backends:
            mark_onboarding_complete(self.configured_providers, backends=backends)
        else:
            # Mock mode — mark complete with empty providers
            mark_onboarding_complete([])
        self.post_message(
            OnboardingComplete(
                self.configured_providers,
                show_splash=first_time,
            )
        )

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input during onboarding stages."""
        value = event.value.strip()
        event.input.value = ""

        if self.stage == "welcome":
            if value == "1":
                # Mock mode — skip everything
                self._finish_onboarding()
            elif value == "2":
                self._show_tool_detect("codex")
            elif value == "3":
                self._show_tool_detect("claude")
            elif value == "4":
                self._show_tool_detect_both()
            elif value == "5":
                self._show_api_key_menu()
            else:
                self._show_welcome()

        elif self.stage == "tool_found":
            if value == "1":
                self._configure_backend(self.selected_backend)
            else:
                self._show_welcome()

        elif self.stage == "tool_not_found":
            if value == "1":
                # Re-check
                self._show_tool_detect(self.selected_backend)
            else:
                self._show_welcome()

        elif self.stage == "tool_both_found":
            if value == "1":
                self._configure_both_backends()
            else:
                self._show_welcome()

        elif self.stage == "tool_both_not_found":
            if value == "1":
                self._show_tool_detect_both()
            else:
                self._show_welcome()

        elif self.stage == "backend_menu_again":
            if value == "1":
                self._show_tool_detect("codex")
            elif value == "2":
                self._show_tool_detect("claude")
            else:
                self._finish_onboarding()

        elif self.stage == "api_key_menu":
            if value == "1":
                self._show_key_input("openai")
            elif value == "2":
                self._show_key_input("anthropic")
            elif value == "3":
                # Env vars mode — skip
                env_providers = detect_env_credentials()
                if env_providers:
                    self.configured_providers = list(env_providers)
                self._finish_onboarding()
            else:
                self._show_welcome()

        elif self.stage == "key_method":
            if value.lower() == "a":
                self._show_paste_prompt()
            elif value.lower() == "b":
                open_provider_key_page(self.selected_provider)
                # Show instructions while browser is open
                content = self.query_one("#onboard-content", Static)
                display = PROVIDER_DISPLAY_NAMES.get(self.selected_provider, self.selected_provider)
                url = PROVIDER_KEY_URLS.get(self.selected_provider, "")
                content.update(
                    f"Opening browser to {display} API keys page...\n"
                    f"URL: {url}\n\n"
                    "Once you have your API key, choose:\n\n"
                    "  [a] Paste API key\n"
                    "  [c] Back to menu\n\n"
                    "Enter choice (a/c):"
                )
                self.stage = "after_browser"
            elif value.lower() == "c":
                self._show_api_key_menu()
            else:
                self._show_key_input(self.selected_provider)

        elif self.stage == "after_browser":
            if value.lower() == "a":
                self._show_paste_prompt()
            else:
                self._show_api_key_menu()

        elif self.stage == "paste_key":
            self._process_key(value)

        elif self.stage == "key_invalid":
            if value.lower() == "a":
                self._show_paste_prompt()
            elif value.lower() == "b":
                # Store anyway
                provider = self.selected_provider
                display = PROVIDER_DISPLAY_NAMES.get(provider, provider)
                self._store_and_continue(provider, display, self._pending_key)
            else:
                self._show_api_key_menu()

        elif self.stage == "provider_menu_again":
            if value == "1":
                self._show_key_input("openai")
            elif value == "2":
                self._show_key_input("anthropic")
            else:
                self._finish_onboarding()


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------


_CSS = """\
/* NEXUS TUI Theme — space & stars, deep black + ice blue */

$bg: #05070c;
$panel: #0b1020;
$primary: #3fa9f5;
$primary2: #72c7ff;
$muted: #7f8aa3;
$surface: #111830;
$error: #e05555;
$success: #4ec990;

Screen {
    background: $bg;
    color: $primary2;
}

#header-bar {
    dock: top;
    height: 3;
    background: $panel;
    color: $primary;
    text-style: bold;
    padding: 0 2;
    border-bottom: solid $primary 40%;
}

#header-inner {
    width: 100%;
    height: 100%;
    align: center middle;
}

#header-mascot {
    width: 16;
    height: 3;
    color: $primary;
    overflow: hidden;
}

#header-title {
    width: 1fr;
    text-align: center;
    color: $primary;
}

#main-area {
    height: 1fr;
}

#sidebar {
    dock: left;
    width: 26;
    background: $panel;
    border-right: solid $primary 30%;
    padding: 1;
    overflow-y: auto;
}

.sidebar-heading {
    text-style: bold;
    color: $primary;
    padding: 0 0 1 0;
}

.star-line {
    color: $muted;
}

#sidebar-list {
    height: auto;
    max-height: 12;
}

#sidebar-list > ListItem {
    color: $primary2;
    padding: 0 1;
}

#sidebar-list > ListItem:hover {
    background: $surface;
}

#sidebar-backends {
    color: $muted;
    height: auto;
}

#sidebar-recent {
    color: $muted;
    height: auto;
}

#sidebar-status {
    color: $muted;
    padding: 1 0;
}

#transcript-area {
    width: 1fr;
    height: 1fr;
    background: $bg;
    padding: 1 2;
    overflow-y: auto;
}

#transcript-area:focus {
    border: tall $primary 50%;
}

#transcript {
    width: 100%;
    color: $primary2;
}

#command-bar {
    dock: bottom;
    height: 3;
    background: $panel;
    border-top: solid $primary 40%;
    padding: 0 1;
}

#prompt-label {
    width: 10;
    color: $primary;
    text-style: bold;
    padding: 1 0;
}

#command-input {
    width: 1fr;
    background: $surface;
    color: $primary2;
    border: none;
}

#command-input:focus {
    border: tall $primary;
}

/* Crash recovery banner */
#crash-banner {
    dock: top;
    height: 3;
    background: #1a1000;
    color: #e0a040;
    padding: 1 2;
    border-bottom: solid #e0a040 40%;
}

#crash-banner-text {
    width: 100%;
}

/* Onboarding styles */
#onboarding {
    width: 100%;
    height: 100%;
    background: $bg;
    padding: 2 4;
    align: center middle;
}

#onboard-content {
    width: 100%;
    height: 1fr;
    color: $primary2;
    padding: 1;
}

#onboard-input {
    width: 60;
    background: $surface;
    color: $primary2;
    border: tall $primary;
}

/* Splash screen styles */
#splash {
    width: 100%;
    height: 100%;
    background: $bg;
    align: center middle;
}

#splash-scroll {
    width: 100%;
    height: 100%;
    align: center middle;
    overflow-y: auto;
}

#splash-content {
    width: auto;
    height: auto;
    color: $primary2;
    text-align: center;
    padding: 2;
}

/* Help screen overlay */
HelpScreen {
    align: center middle;
}

#help-scroll {
    width: 80;
    max-width: 90%;
    height: 80%;
    background: $panel;
    border: solid $primary;
    padding: 2;
    overflow-y: auto;
}

#help-content {
    color: $primary2;
    width: 100%;
}

/* Footer */
Footer {
    background: $panel;
    color: $muted;
}
"""

_CSS_NO_COLOR = """\
/* Minimal CSS for NO_COLOR mode */

Screen {
    background: black;
    color: white;
}

#header-bar {
    dock: top;
    height: 3;
    padding: 0 2;
    border-bottom: solid white;
}

#header-inner {
    width: 100%;
    height: 100%;
    align: center middle;
}

#header-mascot {
    width: 16;
    height: 3;
    overflow: hidden;
}

#header-title {
    width: 1fr;
    text-align: center;
}

#main-area {
    height: 1fr;
}

#sidebar {
    dock: left;
    width: 26;
    border-right: solid white;
    padding: 1;
    overflow-y: auto;
}

.sidebar-heading {
    text-style: bold;
    padding: 0 0 1 0;
}

#sidebar-list {
    height: auto;
    max-height: 12;
}

#sidebar-list > ListItem {
    padding: 0 1;
}

#sidebar-backends {
    height: auto;
}

#sidebar-recent {
    height: auto;
}

#transcript-area {
    width: 1fr;
    height: 1fr;
    padding: 1 2;
    overflow-y: auto;
}

#transcript-area:focus {
    border: tall white;
}

#transcript {
    width: 100%;
}

#command-bar {
    dock: bottom;
    height: 3;
    border-top: solid white;
    padding: 0 1;
}

#prompt-label {
    width: 10;
    text-style: bold;
    padding: 1 0;
}

#command-input {
    width: 1fr;
    border: none;
}

/* Crash recovery banner */
#crash-banner {
    dock: top;
    height: 3;
    padding: 1 2;
    border-bottom: solid white;
}

#crash-banner-text {
    width: 100%;
}

#onboarding {
    width: 100%;
    height: 100%;
    padding: 2 4;
    align: center middle;
}

#onboard-content {
    width: 100%;
    height: 1fr;
    padding: 1;
}

#onboard-input {
    width: 60;
    border: tall white;
}

#splash {
    width: 100%;
    height: 100%;
    align: center middle;
}

#splash-scroll {
    width: 100%;
    height: 100%;
    align: center middle;
    overflow-y: auto;
}

#splash-content {
    width: auto;
    height: auto;
    text-align: center;
    padding: 2;
}

HelpScreen {
    align: center middle;
}

#help-scroll {
    width: 80;
    max-width: 90%;
    height: 80%;
    border: solid white;
    padding: 2;
    overflow-y: auto;
}

#help-content {
    width: 100%;
}
"""


class NexusTUI(App[int]):
    """Full-screen NEXUS orchestrator TUI."""

    TITLE = "NEXUS Orchestrator"
    # CSS is set dynamically by run_tui_app() before instantiation.
    # Textual reads the CSS class variable — it does NOT accept css= in __init__.
    CSS = _CSS
    COMMANDS = {NexusCommandProvider}
    BINDINGS = [
        Binding("ctrl+c", "handle_ctrl_c", "Cancel/Exit", show=True),
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("f1", "show_help", "Help", show=True),
    ]

    # Ctrl+C state
    _ctrl_c_armed: bool = False
    _ctrl_c_timer: float = 0.0
    _task_running: bool = False
    _last_output: str = ""
    _recent_runs: list[str] = []

    def __init__(self, *, no_color: bool = False) -> None:
        self._no_color = no_color or bool(os.environ.get("NO_COLOR", ""))
        super().__init__()
        self._show_onboarding = not is_onboarding_complete()
        self._recent_runs = []

    def compose(self) -> ComposeResult:
        if self._show_onboarding:
            yield OnboardingWidget(id="onboarding")
        else:
            yield from self._compose_main()

    def _compose_main(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")
        # Crash recovery banner (conditional)
        from nexus_orchestrator.ui.crash_report import load_crash_report

        crash_data = load_crash_report()
        if crash_data:
            yield CrashRecoveryBanner(crash_data, id="crash-banner")
        with Horizontal(id="main-area"):
            yield SidebarWidget(id="sidebar")
            with ScrollableContainer(id="transcript-area"):
                yield TranscriptWidget(id="transcript")
        with Horizontal(id="command-bar"):
            yield Static("nexus > ", id="prompt-label")
            all_commands = NEXUS_COMMANDS + list(SLASH_COMMANDS.keys())
            yield CommandInput(
                placeholder="Type a command...",
                id="command-input",
                suggester=SuggestFromList(all_commands, case_sensitive=False),
            )
        yield Footer()

    def on_mount(self) -> None:
        if not self._show_onboarding:
            self._show_welcome()
            self.set_timer(0.05, self._focus_input)

    def _show_welcome(self) -> None:
        """Show welcome message in transcript."""
        try:
            transcript = self.query_one("#transcript", TranscriptWidget)
            transcript.append_output(
                f"{STAR_SEP}\n"
                "Welcome to NEXUS Orchestrator TUI\n"
                "Type a command or press Ctrl+P for the command palette.\n"
                "Press F1 for help.\n"
                f"{STAR_SEP}"
            )
        except NoMatches:
            pass

    # ----- Help overlay -----

    def action_show_help(self) -> None:
        """Show the help overlay."""
        self.push_screen(HelpScreen())

    # ----- Onboarding completion -----

    def on_onboarding_complete(self, message: OnboardingComplete) -> None:
        """Handle onboarding completion — optionally show splash, then main TUI."""
        self._show_onboarding = False
        # Remove onboarding widget
        try:
            onboarding = self.query_one("#onboarding", OnboardingWidget)
            onboarding.remove()
        except NoMatches:
            pass

        if message.show_splash:
            # First-time: show animated splash before main TUI
            self.mount(SplashScreen(id="splash"))
        else:
            self._mount_main_ui()

    def on_splash_screen_dismissed(self, message: SplashScreen.Dismissed) -> None:
        """Handle splash completion — switch to main TUI."""
        try:
            splash = self.query_one("#splash", SplashScreen)
            splash.remove()
        except NoMatches:
            pass
        self._mount_main_ui()

    def _mount_main_ui(self) -> None:
        """Mount the main TUI interface."""
        self.mount_all(list(self._compose_main()))
        self._show_welcome()
        self.set_timer(0.1, self._focus_input)

    def _focus_input(self) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#command-input", CommandInput).focus()

    # ----- Crash recovery banner -----

    def on_crash_recovery_banner_view_report(self, message: CrashRecoveryBanner.ViewReport) -> None:
        """Show crash report in transcript and clear the file."""
        from nexus_orchestrator.ui.crash_report import clear_crash_report, load_crash_report

        data = load_crash_report()
        if data:
            try:
                transcript = self.query_one("#transcript", TranscriptWidget)
                transcript.append_output(
                    "Crash Report:\n"
                    f"  Time: {data.get('timestamp')}\n"
                    f"  Type: {data.get('exception_type')}\n"
                    f"  Message: {data.get('message')}\n"
                    f"  Traceback:\n{data.get('traceback', '(none)')}\n"
                )
            except NoMatches:
                pass
        clear_crash_report()
        self._remove_crash_banner()

    def on_crash_recovery_banner_dismiss_banner(
        self, message: CrashRecoveryBanner.DismissBanner
    ) -> None:
        """Dismiss crash banner and clear the file."""
        from nexus_orchestrator.ui.crash_report import clear_crash_report

        clear_crash_report()
        self._remove_crash_banner()

    def _remove_crash_banner(self) -> None:
        try:
            banner = self.query_one("#crash-banner", CrashRecoveryBanner)
            banner.remove()
        except NoMatches:
            pass

    # ----- Sidebar selection (prefills command bar) -----

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle sidebar quick action selection — prefill command bar."""
        item_id = event.item.id or ""
        command_map = {
            "sb-plan": "plan samples/specs/minimal_design_doc.md --mock",
            "sb-run": "run --mock",
            "sb-status": "status",
            "sb-inspect": "inspect",
            "sb-export": "export",
            "sb-clean": "clean",
            "sb-doctor": "doctor",
            "sb-config": "config",
        }
        command = command_map.get(item_id)
        if command:
            try:
                inp = self.query_one("#command-input", CommandInput)
                inp.value = command
                inp.focus()
                inp.cursor_position = len(command)
            except NoMatches:
                pass

    # ----- Command input -----

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle command input submission."""
        if event.input.id != "command-input":
            return

        command = event.value.strip()
        event.input.value = ""

        if not command:
            return

        # Push to input history
        try:
            inp = self.query_one("#command-input", CommandInput)
            inp.push_history(command)
        except NoMatches:
            pass

        # Handle slash commands
        if command.startswith("/"):
            self._handle_slash_command(command)
            return

        self._execute_command(command)

    def _handle_slash_command(self, command: str) -> None:
        """Process TUI slash commands."""
        try:
            transcript = self.query_one("#transcript", TranscriptWidget)
        except NoMatches:
            return

        cmd = command.split()[0].lower()

        if cmd == "/help":
            self.push_screen(HelpScreen())

        elif cmd == "/clear":
            transcript.clear_transcript()

        elif cmd == "/quit":
            self.exit(0)

        elif cmd == "/theme":
            if self._no_color:
                transcript.append_output("Theme: NO_COLOR mode (monochrome)")
            else:
                transcript.append_output(
                    "Theme: NEXUS Space\n"
                    "  --bg:       #05070c\n"
                    "  --panel:    #0b1020\n"
                    "  --primary:  #3fa9f5\n"
                    "  --primary2: #72c7ff\n"
                    "  --muted:    #7f8aa3"
                )

        elif cmd == "/copy":
            if self._last_output:
                transcript.append_output(
                    "(Last output ready for copy \u2014 clipboard support depends on terminal)"
                )
            else:
                transcript.append_output("No output to copy.")

        else:
            transcript.append_output(f"Unknown slash command: {cmd}")

    @work(thread=True)
    def _execute_command(self, command: str) -> None:
        """Execute a nexus CLI command in a background thread."""
        self._task_running = True

        try:
            transcript = self.query_one("#transcript", TranscriptWidget)
        except NoMatches:
            self._task_running = False
            return

        self.call_from_thread(transcript.append_command_header, command)

        exit_code, stdout, stderr = run_command(command)

        if stdout.strip():
            self._last_output = stdout.strip()
            self.call_from_thread(transcript.append_output, stdout.strip())

        if stderr.strip():
            self.call_from_thread(
                transcript.append_stderr,
                stderr.strip(),
                no_color=self._no_color,
            )

        self.call_from_thread(
            transcript.append_exit_badge,
            exit_code,
            no_color=self._no_color,
        )

        self._task_running = False
        self._ctrl_c_armed = False

        # Update sidebar status + recent runs
        self._update_sidebar_status(command, exit_code)

    def _update_sidebar_status(self, command: str, exit_code: int) -> None:
        """Update sidebar with latest command result and recent runs."""
        # Update status line
        try:
            status = self.query_one("#sidebar-status", Static)
            cmd_short = command.split()[0] if command.split() else command
            self.call_from_thread(
                status.update,
                f"Last: {cmd_short}\nExit: {exit_code}",
            )
        except NoMatches:
            pass

        # Update recent runs list (max 5)
        cmd_short = command.split()[0] if command.split() else command
        badge = "\u2713" if exit_code == 0 else "\u2717"
        entry = f"  {badge} {cmd_short} ({exit_code})"
        self._recent_runs.append(entry)
        if len(self._recent_runs) > 5:
            self._recent_runs = self._recent_runs[-5:]
        try:
            recent = self.query_one("#sidebar-recent", Static)
            self.call_from_thread(recent.update, "\n".join(self._recent_runs))
        except NoMatches:
            pass

    # ----- Ctrl+C handling -----

    def action_handle_ctrl_c(self) -> None:
        """Handle Ctrl+C: cancel task or arm exit."""
        now = time.monotonic()

        if self._task_running:
            # First Ctrl+C cancels the running task
            self._task_running = False
            try:
                transcript = self.query_one("#transcript", TranscriptWidget)
                transcript.append_output("\nCancelled. Press Ctrl+C again to exit.")
            except NoMatches:
                pass
            self._ctrl_c_armed = True
            self._ctrl_c_timer = now
            return

        if self._ctrl_c_armed and (now - self._ctrl_c_timer) < 2.0:
            # Second Ctrl+C within window — exit
            self.exit(0)
            return

        # First Ctrl+C with no task — arm exit
        self._ctrl_c_armed = True
        self._ctrl_c_timer = now
        try:
            transcript = self.query_one("#transcript", TranscriptWidget)
            transcript.append_output("Press Ctrl+C again to exit.")
        except NoMatches:
            pass

    def action_quit_app(self) -> None:
        """Quit the TUI cleanly."""
        self.exit(0)


def run_tui_app(*, no_color: bool = False) -> int:
    """Create and run the TUI app, returning exit code."""
    from nexus_orchestrator.ui.crash_report import save_crash_report

    effective_no_color = no_color or bool(os.environ.get("NO_COLOR", ""))
    # Set CSS class variable before instantiation — Textual reads it from
    # the class, not from an __init__ keyword argument.
    NexusTUI.CSS = _CSS_NO_COLOR if effective_no_color else _CSS
    app = NexusTUI(no_color=no_color)
    try:
        result = app.run()
        return result if isinstance(result, int) else 0
    except Exception as exc:
        save_crash_report(exc)
        raise


__all__ = ["NexusTUI", "run_command", "run_tui_app"]
