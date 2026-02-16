"""Transcript widget — RichLog with model-colored syntax display.

File: src/nexus_orchestrator/ui/tui/widgets/transcript.py

Uses Textual's RichLog widget for styled output. Agent code blocks
(Write, Edit, Bash) are displayed with model-specific color palettes:
  - Green: GPT models ([gpt53], [spark])
  - Orange: Anthropic models ([opus])
  - Blue: System/nexus messages and legacy tags ([codex], [claude])
The built-in max_lines parameter handles overflow (ring buffer).
"""

from __future__ import annotations

import re

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog

from nexus_orchestrator.ui.tui.state import EventKind, TranscriptEvent

# Maximum number of lines kept in the RichLog ring buffer.
# Must stay <= the state deque (5,000) to avoid orphaned lines.
# Lower values keep rendering snappy (RichLog is O(n) on scroll recompute).
_MAX_BUFFER_LINES = 2_000

# --- Shared styles (not model-specific) ---
_S_DEFAULT = Style(color="#c8cdd8")
_S_SEPARATOR = Style(color="#1a2550")
_S_STDERR = Style(color="#e05555")
_S_SYSTEM = Style(color="#3fa9f5")
_S_RESULT_OK = Style(color="#4ec990", bold=True)
_S_RESULT_FAIL = Style(color="#e05555", bold=True)
_S_HEADER = Style(color="#3fa9f5", bold=True)
_S_DIM = Style(color="#7f8aa3")

# --- Blue palette (system/nexus, legacy [codex]/[claude] tags) ---
_S_BLUE_TAG = Style(color="#7f8aa3")
_S_BLUE_TOOL_HEADER = Style(color="#3fa9f5", bold=True)
_S_BLUE_FILE_PATH = Style(color="#72c7ff")
_S_BLUE_CODE = Style(color="#c8cdd8")
_S_BLUE_CODE_ADD = Style(color="#72c7ff")
_S_BLUE_CODE_DEL = Style(color="#5b7fa3")
_S_BLUE_TEXT = Style(color="#c8cdd8")
_S_BLUE_RESULT = Style(color="#4ec990", bold=True)

# --- Green palette (GPT models: [gpt53], [spark]) ---
_S_GREEN_TAG = Style(color="#7da38a")
_S_GREEN_TOOL_HEADER = Style(color="#4ec990", bold=True)
_S_GREEN_FILE_PATH = Style(color="#7ee8b5")
_S_GREEN_CODE = Style(color="#c8e6d0")
_S_GREEN_CODE_ADD = Style(color="#7ee8b5")
_S_GREEN_CODE_DEL = Style(color="#5b8a6a")
_S_GREEN_TEXT = Style(color="#c8e6d0")
_S_GREEN_RESULT = Style(color="#4ec990", bold=True)

# --- Orange palette (Anthropic models: [opus]) ---
_S_ORANGE_TAG = Style(color="#a3897f")
_S_ORANGE_TOOL_HEADER = Style(color="#f5a623", bold=True)
_S_ORANGE_FILE_PATH = Style(color="#ffc870")
_S_ORANGE_CODE = Style(color="#e6d8c8")
_S_ORANGE_CODE_ADD = Style(color="#ffc870")
_S_ORANGE_CODE_DEL = Style(color="#a38965")
_S_ORANGE_TEXT = Style(color="#e6d8c8")
_S_ORANGE_RESULT = Style(color="#f5a623", bold=True)


def _styles_for_tag(tag_text: str) -> tuple[Style, Style, Style, Style, Style, Style, Style, Style]:
    """Return (tag, tool_header, file_path, code, code_add, code_del, text, result) styles for a tag.

    Maps model-specific tags to color palettes:
      [gpt53], [spark] -> green
      [opus]           -> orange
      [codex], [claude] and others -> blue (default)
    """
    # Normalize: strip whitespace and brackets
    tag_clean = tag_text.strip().strip("[]").lower()
    if tag_clean in ("gpt53", "spark"):
        return (
            _S_GREEN_TAG, _S_GREEN_TOOL_HEADER, _S_GREEN_FILE_PATH,
            _S_GREEN_CODE, _S_GREEN_CODE_ADD, _S_GREEN_CODE_DEL,
            _S_GREEN_TEXT, _S_GREEN_RESULT,
        )
    if tag_clean in ("opus",):
        return (
            _S_ORANGE_TAG, _S_ORANGE_TOOL_HEADER, _S_ORANGE_FILE_PATH,
            _S_ORANGE_CODE, _S_ORANGE_CODE_ADD, _S_ORANGE_CODE_DEL,
            _S_ORANGE_TEXT, _S_ORANGE_RESULT,
        )
    return (
        _S_BLUE_TAG, _S_BLUE_TOOL_HEADER, _S_BLUE_FILE_PATH,
        _S_BLUE_CODE, _S_BLUE_CODE_ADD, _S_BLUE_CODE_DEL,
        _S_BLUE_TEXT, _S_BLUE_RESULT,
    )

# Pattern to detect agent-tagged output lines: "  [claude] ..."
_TAG_PATTERN = re.compile(r"^(\s*\[[\w.-]+\])\s*(.*)")
# Pattern to detect code block markers
_TOOL_MARKER = re.compile(r"^\s*---\s*(Write|Edit|Bash)(?::\s*(.+?))?\s*---\s*$")
_END_MARKER = re.compile(r"^\s*---\s*end\s*---\s*$")


class TranscriptWidget(Widget):
    """Styled transcript log backed by a RichLog widget."""

    DEFAULT_CSS = """
    TranscriptWidget {
        height: 1fr;
        width: 1fr;
        overflow: hidden;
    }
    #transcript-area {
        height: 1fr;
        width: 1fr;
        min-height: 5;
        background: #0b1020;
        scrollbar-color: #3fa9f5 40%;
        scrollbar-background: #05070c;
        scrollbar-size-vertical: 1;
    }
    """

    def __init__(self, *, no_color: bool = False, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._no_color = no_color
        self._event_count = 0

    def compose(self) -> ComposeResult:
        rl = RichLog(
            max_lines=_MAX_BUFFER_LINES,
            wrap=True,
            markup=False,
            auto_scroll=True,
            id="transcript-area",
        )
        # Prevent RichLog from ever stealing focus from the composer input.
        # RichLog inherits can_focus=True from ScrollView; override per-instance.
        rl.can_focus = False
        yield rl

    @property
    def rich_log(self) -> RichLog:
        return self.query_one("#transcript-area", RichLog)

    def append_event(self, event: TranscriptEvent) -> None:
        """Append a single event to the transcript."""
        log = self.rich_log
        renderable = _format_event_rich(event, no_color=self._no_color)
        log.write(renderable)
        self._event_count += 1

    def append_events(self, events: list[TranscriptEvent]) -> None:
        """Append multiple events efficiently.

        Relies on RichLog's built-in auto_scroll=True to keep the view
        at the bottom.  Does NOT call scroll_end() explicitly because
        that steals focus from the Composer input widget.
        """
        if not events:
            return
        log = self.rich_log
        for event in events:
            renderable = _format_event_rich(event, no_color=self._no_color)
            log.write(renderable)
        self._event_count += len(events)

    @property
    def text(self) -> str:
        """Return the current transcript content as plain text.

        Primarily used for testing and export. Iterates over the RichLog's
        internal line cache to reconstruct a plain text representation.
        """
        log = self.rich_log
        lines: list[str] = []
        for line_obj in log.lines:
            if hasattr(line_obj, "plain"):
                lines.append(line_obj.plain)
            else:
                lines.append(str(line_obj))
        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all transcript content."""
        self.rich_log.clear()
        self._event_count = 0


def _format_event_rich(event: TranscriptEvent, *, no_color: bool = False) -> Text:
    """Format a transcript event as a styled Rich Text object."""
    if no_color:
        return _format_event_plain(event)

    if event.kind == EventKind.COMMAND_HEADER:
        sep = "\u2500" * 60
        text = Text()
        text.append(f"{event.text}  [{event.timestamp}]\n", style=_S_HEADER)
        text.append(sep, style=_S_SEPARATOR)
        return text

    if event.kind == EventKind.EXIT_BADGE:
        sep = "\u2500" * 60
        code = event.exit_code or 0
        text = Text()
        text.append(f"{sep}\n", style=_S_SEPARATOR)
        if code == 0:
            text.append(f"OK ({code})", style=_S_RESULT_OK)
        else:
            text.append(f"FAIL ({code})", style=_S_RESULT_FAIL)
        return text

    if event.kind == EventKind.STDERR:
        text = Text()
        text.append(f"ERR: {event.text}", style=_S_STDERR)
        return text

    if event.kind == EventKind.SYSTEM:
        text = Text()
        text.append(f"[system] {event.text}", style=_S_SYSTEM)
        return text

    if event.kind == EventKind.AGENT_HEADER:
        sep = "\u2500" * 60
        text = Text()
        text.append(f"{event.text}  [{event.timestamp}]\n", style=_S_HEADER)
        text.append(sep, style=_S_SEPARATOR)
        return text

    if event.kind == EventKind.AGENT_RESPONSE:
        return _style_stdout_line(event.text)

    # STDOUT — apply blue-themed styling to agent output
    return _style_stdout_line(event.text)


def _style_stdout_line(line: str) -> Text:
    """Style a single stdout line with model-specific colors.

    Detects agent tags ([gpt53], [spark], [opus], [codex], [claude]) and
    code block markers (--- Write: path ---, | code, $ command, etc.)
    for targeted coloring based on model family.
    """
    text = Text()

    # Check for agent-tagged line: "  [gpt53] content"
    tag_match = _TAG_PATTERN.match(line)
    if tag_match:
        tag_part = tag_match.group(1)
        content = tag_match.group(2)

        # Look up model-specific palette
        s_tag, s_tool_hdr, s_fpath, s_code, s_code_add, s_code_del, s_text, s_result = (
            _styles_for_tag(tag_part)
        )

        text.append(tag_part, style=s_tag)
        text.append(" ")

        # Detect tool markers: --- Write: path ---
        tool_match = _TOOL_MARKER.match(content)
        if tool_match:
            tool_name = tool_match.group(1)
            file_path = tool_match.group(2) or ""
            text.append(f"--- {tool_name}", style=s_tool_hdr)
            if file_path:
                text.append(": ", style=s_tool_hdr)
                text.append(file_path, style=s_fpath)
            text.append(" ---", style=s_tool_hdr)
            return text

        # Detect end marker
        if _END_MARKER.match(content):
            text.append(content, style=_S_SEPARATOR)
            return text

        # Detect code lines: "  | code", "  + code", "  - code", "  $ cmd"
        stripped = content.lstrip()
        if stripped.startswith("| "):
            text.append(content, style=s_code)
            return text
        if stripped.startswith("+ "):
            text.append(content, style=s_code_add)
            return text
        if stripped.startswith("- "):
            text.append(content, style=s_code_del)
            return text
        if stripped.startswith("$ "):
            text.append(content, style=s_code)
            return text
        if stripped == ">>>":
            text.append(content, style=_S_SEPARATOR)
            return text

        # Check for result lines
        if "Agent complete" in content:
            text.append(content, style=s_result)
            return text

        # Default agent output
        text.append(content, style=s_text)
        return text

    # Non-tagged line — default style
    text.append(line, style=_S_DEFAULT)
    return text


def _format_event_plain(event: TranscriptEvent) -> Text:
    """Format a transcript event as unstyled Text (for no_color mode)."""
    if event.kind == EventKind.COMMAND_HEADER:
        sep = "\u2500" * 60
        return Text(f"{event.text}  [{event.timestamp}]\n{sep}")

    if event.kind == EventKind.EXIT_BADGE:
        sep = "\u2500" * 60
        code = event.exit_code or 0
        badge = f"OK ({code})" if code == 0 else f"FAIL ({code})"
        return Text(f"{sep}\n{badge}")

    if event.kind == EventKind.STDERR:
        return Text(f"ERR: {event.text}")

    if event.kind == EventKind.SYSTEM:
        return Text(f"[system] {event.text}")

    if event.kind == EventKind.AGENT_HEADER:
        sep = "\u2500" * 60
        return Text(f"{event.text}  [{event.timestamp}]\n{sep}")

    return Text(event.text)


__all__ = ["TranscriptWidget"]
