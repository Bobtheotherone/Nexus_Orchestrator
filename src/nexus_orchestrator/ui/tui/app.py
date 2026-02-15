"""Main Textual App — view layer that renders state and emits intents.

File: src/nexus_orchestrator/ui/tui/app.py

This is the top-level Textual App. It:
- Composes the layout (header, sidebar, transcript, composer, status line)
- Wires widget events to controller intents
- Subscribes to controller state changes and updates widgets
- Handles onboarding, splash, crash recovery (delegated from old tui_app.py)

All business logic lives in the controller; this file only does rendering.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits
from textual.command import Provider as CommandProvider
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Static

from nexus_orchestrator.ui.tui.controller import TUIController
from nexus_orchestrator.ui.tui.screens.help import (
    COMMAND_DESCRIPTIONS,
    SLASH_COMMANDS,
    HelpScreen,
)
from nexus_orchestrator.ui.tui.screens.plan_dialog import PlanDialog
from nexus_orchestrator.ui.tui.state import AppState, EventKind
from nexus_orchestrator.ui.tui.widgets.composer import Composer
from nexus_orchestrator.ui.tui.widgets.sidebar import SidebarWidget
from nexus_orchestrator.ui.tui.widgets.spinner import RunnerSpinner
from nexus_orchestrator.ui.tui.widgets.statusline import StatusLine
from nexus_orchestrator.ui.tui.widgets.transcript import TranscriptWidget

# ---------------------------------------------------------------------------
# Asset loading (kept here for the header mascot)
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

HEADER_MASCOT_MAX_WIDTH = 14
HEADER_MASCOT_MAX_HEIGHT = 4


NEXUS_COMMANDS = ["plan", "run", "status", "inspect", "export", "clean", "doctor", "config"]


def _load_asset(filename: str) -> str:
    path = _ASSETS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").rstrip("\n")
    return f"(asset not found: {filename})"


def _load_auto_header() -> str:
    raw = _load_asset("auto_header_ascii.txt")
    if raw.startswith("(asset not found"):
        return "(auto)"
    lines = raw.splitlines()[:HEADER_MASCOT_MAX_HEIGHT]
    return "\n".join(line[:HEADER_MASCOT_MAX_WIDTH] for line in lines)


_STAR_GLYPHS: tuple[str, ...] = ("·", "✦", "★")
_STAR_GLYPH_WEIGHTS: tuple[float, ...] = (0.70, 0.22, 0.08)
_STAR_BASE_SEED = 1337


def _star_line(width: int, *, row: int, seed: int) -> str:
    """Generate a natural, deterministic starfield line of exactly `width`."""
    if width <= 0:
        return ""

    rng = random.Random(seed)
    chars = [" "] * width

    # --- Walk in ~8-col cells, skip some for organic gaps ---
    cell = 8
    phase = (row * 7) % cell

    occupied: list[int] = []
    min_sep = 4

    x = phase + rng.randint(0, cell // 2)
    while x < width:
        # Skip ~25% of cells for natural empty patches
        if rng.random() < 0.25:
            x += cell
            continue

        # Jitter within the cell
        p = x + rng.randint(-2, 2)
        p = max(1, min(width - 2, p))

        if not any(abs(p - o) < min_sep for o in occupied):
            # Glyph selection: good mix of all three sizes
            r = rng.random()
            if r < 0.10:
                chars[p] = "★"
            elif r < 0.35:
                chars[p] = "✦"
            else:
                chars[p] = "·"
            occupied.append(p)

            # Companion dot near medium/large stars
            if chars[p] in ("★", "✦") and rng.random() < 0.45:
                offset = rng.choice([-5, -4, 4, 5])
                cp = p + offset
                if 0 <= cp < width and chars[cp] == " ":
                    if not any(abs(cp - o) < 3 for o in occupied if o != p):
                        chars[cp] = "·"
                        occupied.append(cp)

        x += cell

    return "".join(chars)

def _overlay_title_with_void(
    stars: str,
    title: str,
    *,
    pad: int = 8,
    render: bool = True,
) -> str:
    """Clear a star-free void around the (centered) title span, and optionally render it.

    Use render=False for the row ABOVE the title to create vertical padding
    while keeping the void aligned to the real title position.
    """
    width = len(stars)
    if width <= 0:
        return ""

    # Fit title to width (with ellipsis) so void + text always align.
    if len(title) > width:
        if width <= 1:
            title_fit = title[:width]
        else:
            title_fit = title[: width - 1] + "…"
    else:
        title_fit = title

    # Compute the centered span from the (fitted) title.
    start = max(0, (width - len(title_fit)) // 2)
    void_start = max(0, start - pad)
    void_end = min(width, start + len(title_fit) + pad)

    chars = list(stars)

    # Clear the void completely (this nukes that annoying star/dot above the title).
    for i in range(void_start, void_end):
        chars[i] = " "

    # Render the title back into the void if requested.
    if render:
        for i, ch in enumerate(title_fit):
            j = start + i
            if 0 <= j < width:
                chars[j] = ch

    return "".join(chars)





def _build_header_block(width: int = 80) -> str:
    """Build the full header: robot + full-width star field + centered title with void."""
    width = max(1, width)

    mascot_lines = _load_auto_header().splitlines()
    # Pad mascot to exactly 4 lines
    while len(mascot_lines) < HEADER_MASCOT_MAX_HEIGHT:
        mascot_lines.append("")
    mascot_lines = mascot_lines[:HEADER_MASCOT_MAX_HEIGHT]

    mascot_w = max(len(ln) for ln in mascot_lines) + 2  # +2 for spacing after mascot
    star_w = max(0, width - mascot_w)

    title = "NEXUS  -  Agentic AI Orchestrator"
    title_row = 2  # place title higher (matches header feel + screenshot)

    out: list[str] = []
    for row in range(HEADER_MASCOT_MAX_HEIGHT):
        robot = mascot_lines[row].ljust(mascot_w)

        # Deterministic seed per width+row so it doesn't "cut off" and stays stable.
        seed = _STAR_BASE_SEED + (width * 31) + (row * 9973)
        stars = _star_line(star_w, row=row, seed=seed)

        if row > title_row:
            stars = _overlay_title_with_void(stars, title, pad=10, render=False)  # clears above
        elif row == title_row:
            stars = _overlay_title_with_void(stars, title, pad=10, render=True)   # draws title
        elif row < title_row:
            stars = _overlay_title_with_void(stars, title, pad=10, render=False)  # clear below too

        line = robot + stars
        # Ensure we never exceed width (Textual will clip, but keep exact here too)
        out.append(line[:width].ljust(width))

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Header widgets
# ---------------------------------------------------------------------------


class HeaderBar(Static):
    def _refresh_header(self) -> None:
        # Prefer content width (excludes padding/borders), fall back to widget width.
        width = int(getattr(self, "content_size", self.size).width) or int(self.size.width) or 80
        self.update(_build_header_block(width=width))

    def on_mount(self) -> None:
        self._refresh_header()

    def on_resize(self, event: object) -> None:
        self._refresh_header()


# ---------------------------------------------------------------------------
# Crash Recovery Banner
# ---------------------------------------------------------------------------


class CrashRecoveryBanner(Static):
    def __init__(self, crash_data: dict[str, object], **kwargs: object) -> None:
        ts = str(crash_data.get("timestamp", "unknown"))
        exc_type = str(crash_data.get("exception_type", "unknown"))
        super().__init__(
            f"[bold]Recovered from crash[/bold] ({exc_type} at {ts})"
            "  Press [V] to view, [D] to dismiss",
            id="crash-banner",
            **kwargs,
        )
        self._crash_data = crash_data


# ---------------------------------------------------------------------------
# Command Palette Provider
# ---------------------------------------------------------------------------


class NexusCommandProvider(CommandProvider):
    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)

        for cmd, desc in COMMAND_DESCRIPTIONS.items():
            prefixed = f"/run {cmd}"
            score = matcher.match(f"{prefixed} {desc}")
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(prefixed),
                    self._make_callback(prefixed),
                    help=desc,
                )

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
                app = self.app
                if isinstance(app, NexusTUI):
                    app.composer.set_value(cmd)
                    app.composer.focus_input()
            except Exception:
                pass

        return callback


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------

# CSS loaded from theme.tcss
_THEME_PATH = Path(__file__).resolve().parent / "theme.tcss"
_CSS_NO_COLOR = """
Screen { background: black; color: white; }
#header-bar { dock: top; height: 6; border-bottom: solid white; padding: 0 2; }
#main-area { height: 1fr; }
SidebarWidget { dock: left; width: 26; border-right: solid white; padding: 1; overflow-y: auto; }
.sidebar-heading { text-style: bold; padding: 0 0 1 0; }
#transcript-container { width: 1fr; height: 1fr; padding: 1 2; }
StatusLine { dock: bottom; height: 1; }
Composer { dock: bottom; height: auto; max-height: 5; padding: 0 1; }
"""


def _load_theme_css() -> str:
    try:
        return _THEME_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


class NexusTUI(App[int]):
    """Production-quality NEXUS orchestrator TUI."""

    TITLE = "NEXUS Orchestrator"
    CSS = _load_theme_css()
    COMMANDS = {NexusCommandProvider}
    BINDINGS = [
        Binding("ctrl+c", "handle_ctrl_c", "Cancel/Exit", show=True),
        Binding("ctrl+q", "quit_app", "Quit", show=True),
        Binding("ctrl+e", "export_transcript", "Export", show=True),
        Binding("ctrl+y", "copy_to_clipboard", "Copy", show=True),
        Binding("f1", "show_help", "Help", show=True),
    ]

    # Ctrl+C state
    _ctrl_c_armed: bool = False
    _ctrl_c_timer: float = 0.0

    def __init__(self, *, no_color: bool = False) -> None:
        self._no_color = no_color or bool(os.environ.get("NO_COLOR", ""))
        super().__init__()

        self._state = AppState(no_color=self._no_color)
        self._controller = TUIController(
            state=self._state,
            on_state_change=self._on_state_change,
            no_color=self._no_color,
        )

        # Check onboarding
        from nexus_orchestrator.ui.onboarding import is_onboarding_complete

        self._show_onboarding = not is_onboarding_complete()

    # ------------------------------------------------------------------
    # State change callback — update all widgets
    # ------------------------------------------------------------------

    async def _on_state_change(self) -> None:
        """Called by controller when state changes — update widgets."""
        # Check for special system commands
        if self._state.transcript:
            last = self._state.transcript[-1]
            if last.kind == EventKind.SYSTEM:
                if last.text == "__SHOW_HELP__":
                    self._state.transcript.pop()
                    self.push_screen(HelpScreen())
                    return
                if last.text == "__QUIT__":
                    self._state.transcript.pop()
                    self.exit(0)
                    return

        # Update transcript
        self._sync_transcript()

        # Update sidebar
        try:
            sidebar = self.query_one(SidebarWidget)
            sidebar.update_from_state(self._state)
        except NoMatches:
            pass

        # Update status line
        try:
            status = self.query_one(StatusLine)
            status.update_from_state(self._state)
        except NoMatches:
            pass

        # Update spinner visibility
        try:
            spinner = self.query_one(RunnerSpinner)
            spinner.update_from_state(self._state)
        except NoMatches:
            pass

    # Track how many events we've rendered
    _rendered_event_count: int = 0

    def _sync_transcript(self) -> None:
        """Sync transcript widget with state — only append new events."""
        try:
            transcript = self.query_one(TranscriptWidget)
        except NoMatches:
            return

        events = list(self._state.transcript)
        new_count = len(events)

        if new_count == 0 and self._rendered_event_count > 0:
            # Cleared
            transcript.clear()
            self._rendered_event_count = 0
            return

        # Append only new events
        if new_count > self._rendered_event_count:
            new_events = events[self._rendered_event_count:]
            transcript.append_events(new_events)
            self._rendered_event_count = new_count

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        if self._show_onboarding:
            from nexus_orchestrator.ui.tui_app import OnboardingWidget

            yield OnboardingWidget(id="onboarding")
        else:
            yield from self._compose_main()

    def _compose_main(self) -> ComposeResult:
        yield HeaderBar(id="header-bar")

        # Crash recovery
        from nexus_orchestrator.ui.crash_report import load_crash_report

        crash_data = load_crash_report()
        if crash_data:
            yield CrashRecoveryBanner(crash_data)

        with Horizontal(id="main-area"):
            yield SidebarWidget()
            yield TranscriptWidget(
                no_color=self._no_color,
                id="transcript-container",
            )

        all_commands = NEXUS_COMMANDS + list(SLASH_COMMANDS.keys())
        yield RunnerSpinner(id="runner-spinner")
        yield StatusLine()
        yield Composer(suggestions=all_commands, id="composer")

    @property
    def composer(self) -> Composer:
        return self.query_one("#composer", Composer)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        if not self._show_onboarding:
            self._setup_main_ui()

    @work(thread=False)
    async def _setup_main_ui(self) -> None:
        """Initialize main UI: welcome message, backend detection, workspace info."""
        self._controller.append_system(
            "Welcome to NEXUS Orchestrator TUI\n"
            "Type a prompt to generate an engineering design doc.\n"
            "Use /run <cmd> for CLI commands. Press F1 for help.\n"
            "No API keys required — uses local CLI tools when available."
        )
        await self._on_state_change()

        # Focus composer
        self.set_timer(0.05, self._focus_composer)

        # Background: detect backends and workspace (off UI thread)
        await self._controller.detect_workspace_info()
        await self._controller.detect_backends()

    def _focus_composer(self) -> None:
        import contextlib

        with contextlib.suppress(NoMatches):
            self.composer.focus_input()

    # ------------------------------------------------------------------
    # Event handlers — translate widget events to controller intents
    # ------------------------------------------------------------------

    async def on_composer_command_submitted(self, event: Composer.CommandSubmitted) -> None:
        """User submitted a command from the composer."""
        await self._controller.execute_command(event.command)

    async def on_sidebar_widget_quick_action_activated(
        self, event: SidebarWidget.QuickActionActivated
    ) -> None:
        """Quick action button pressed."""
        if event.needs_args:
            # "Plan..." — open dialog with callback (NOT push_screen_wait,
            # which requires a Textual worker context)
            self.push_screen(PlanDialog(), callback=self._on_plan_dialog_result)
        else:
            await self._controller.execute_command(event.command)

    async def _on_plan_dialog_result(self, result: str | None) -> None:
        """Callback from PlanDialog — execute the plan command if not cancelled."""
        if result:
            await self._controller.execute_command(result)

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    # ------------------------------------------------------------------
    # Export transcript (Ctrl+E) and Copy to clipboard (Ctrl+Y)
    # ------------------------------------------------------------------

    @work(thread=False)
    async def action_export_transcript(self) -> None:
        await self._controller.export_transcript()

    @work(thread=False)
    async def action_copy_to_clipboard(self) -> None:
        await self._controller.copy_to_clipboard()

    # ------------------------------------------------------------------
    # Ctrl+C handling
    # ------------------------------------------------------------------

    async def action_handle_ctrl_c(self) -> None:
        now = time.monotonic()

        from nexus_orchestrator.ui.tui.state import RunnerStatus

        is_busy = self._state.runner_status in (
            RunnerStatus.RUNNING,
            RunnerStatus.GENERATING,
        )

        # Already cancel-requested: second press exits
        if self._state.runner_status == RunnerStatus.CANCEL_REQUESTED:
            self.exit(0)
            return

        if is_busy:
            # Double-press to cancel: first press shows warning, second cancels
            if self._ctrl_c_armed and (now - self._ctrl_c_timer) < 2.0:
                await self._controller.cancel_command()
                return
            self._ctrl_c_armed = True
            self._ctrl_c_timer = now
            self._controller.append_system(
                "Press Ctrl+C again within 2s to cancel the running task (or Esc to dismiss)."
            )
            await self._on_state_change()
            return

        # Not busy: double-press to exit
        if self._ctrl_c_armed and (now - self._ctrl_c_timer) < 2.0:
            self.exit(0)
            return

        self._ctrl_c_armed = True
        self._ctrl_c_timer = now
        self._controller.append_system("Press Ctrl+C again to exit.")
        await self._on_state_change()

    def action_quit_app(self) -> None:
        self.exit(0)

    # ------------------------------------------------------------------
    # Onboarding integration
    # ------------------------------------------------------------------

    def on_onboarding_complete(self, message: object) -> None:
        """Handle onboarding completion."""
        from nexus_orchestrator.ui.tui_app import OnboardingComplete, SplashScreen

        if not isinstance(message, OnboardingComplete):
            return

        self._show_onboarding = False
        try:
            onboarding = self.query_one("#onboarding")
            onboarding.remove()
        except NoMatches:
            pass

        if message.show_splash:
            from nexus_orchestrator.ui.tui_app import SplashScreen

            self.mount(SplashScreen(id="splash"))
        else:
            self._mount_main_ui()

    def on_splash_screen_dismissed(self, message: object) -> None:
        try:
            splash = self.query_one("#splash")
            splash.remove()
        except NoMatches:
            pass
        self._mount_main_ui()

    def _mount_main_ui(self) -> None:
        self.mount_all(list(self._compose_main()))
        self._setup_main_ui()

    # ------------------------------------------------------------------
    # Crash recovery key handler
    # ------------------------------------------------------------------

    def on_key(self, event: object) -> None:
        """Handle crash banner keys."""
        from textual import events as _events

        if not isinstance(event, _events.Key):
            return

        try:
            self.query_one("#crash-banner", CrashRecoveryBanner)
        except NoMatches:
            return

        if event.key == "v":
            self._view_crash_report()
        elif event.key == "d":
            self._dismiss_crash_banner()

    def _view_crash_report(self) -> None:
        from nexus_orchestrator.ui.crash_report import clear_crash_report, load_crash_report

        data = load_crash_report()
        if data:
            self._controller.append_system(
                "Crash Report:\n"
                f"  Time: {data.get('timestamp')}\n"
                f"  Type: {data.get('exception_type')}\n"
                f"  Message: {data.get('message')}\n"
                f"  Traceback:\n{data.get('traceback', '(none)')}\n"
            )
        clear_crash_report()
        self._dismiss_crash_banner()

    def _dismiss_crash_banner(self) -> None:
        from nexus_orchestrator.ui.crash_report import clear_crash_report

        clear_crash_report()
        try:
            banner = self.query_one("#crash-banner", CrashRecoveryBanner)
            banner.remove()
        except NoMatches:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_tui_app(*, no_color: bool = False) -> int:
    """Create and run the TUI app, returning exit code."""
    from nexus_orchestrator.ui.crash_report import save_crash_report

    effective_no_color = no_color or bool(os.environ.get("NO_COLOR", ""))
    if effective_no_color:
        NexusTUI.CSS = _CSS_NO_COLOR
    else:
        NexusTUI.CSS = _load_theme_css()

    app = NexusTUI(no_color=no_color)
    try:
        result = app.run()
        return result if isinstance(result, int) else 0
    except Exception as exc:
        save_crash_report(exc)
        raise


__all__ = [
    "HEADER_MASCOT_MAX_HEIGHT",
    "HEADER_MASCOT_MAX_WIDTH",
    "NexusTUI",
    "_load_auto_header",
    "run_tui_app",
]
