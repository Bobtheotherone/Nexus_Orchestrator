"""Transcript widget — RichLog with blue-themed syntax coloring.

File: src/nexus_orchestrator/ui/tui/widgets/transcript.py

Uses Textual's RichLog widget for styled output. Agent code blocks
(Write, Edit, Bash) are displayed with a blue color palette. The
built-in max_lines parameter handles overflow (ring buffer).
"""

from __future__ import annotations

import re

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog

from nexus_orchestrator.ui.tui.state import EventKind, TranscriptEvent

# Maximum number of lines kept in the RichLog buffer
_MAX_BUFFER_LINES = 10_000

# --- Blue-themed styles matching the NEXUS palette ---
_S_DEFAULT = Style(color="#c8cdd8")
_S_TOOL_HEADER = Style(color="#3fa9f5", bold=True)
_S_FILE_PATH = Style(color="#72c7ff")
_S_CODE = Style(color="#c8cdd8")
_S_CODE_ADD = Style(color="#72c7ff")
_S_CODE_DEL = Style(color="#5b7fa3")
_S_TAG = Style(color="#7f8aa3")
_S_SEPARATOR = Style(color="#1a2550")
_S_STDERR = Style(color="#e05555")
_S_SYSTEM = Style(color="#3fa9f5")
_S_RESULT_OK = Style(color="#4ec990", bold=True)
_S_RESULT_FAIL = Style(color="#e05555", bold=True)
_S_HEADER = Style(color="#3fa9f5", bold=True)
_S_DIM = Style(color="#7f8aa3")

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
    }
    #transcript-area {
        height: 1fr;
        width: 1fr;
        background: #0b1020;
        scrollbar-color: #3fa9f5 40%;
        scrollbar-background: #05070c;
    }
    """

    def __init__(self, *, no_color: bool = False, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._no_color = no_color
        self._event_count = 0

    def compose(self) -> ComposeResult:
        yield RichLog(
            max_lines=_MAX_BUFFER_LINES,
            wrap=True,
            markup=False,
            auto_scroll=True,
            id="transcript-area",
        )

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
        """Append multiple events efficiently."""
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
    """Style a single stdout line with blue theme colors.

    Detects agent tags ([claude], [codex]) and code block markers
    (--- Write: path ---, | code, $ command, etc.) for targeted coloring.
    """
    text = Text()

    # Check for agent-tagged line: "  [claude] content"
    tag_match = _TAG_PATTERN.match(line)
    if tag_match:
        tag_part = tag_match.group(1)
        content = tag_match.group(2)

        text.append(tag_part, style=_S_TAG)
        text.append(" ")

        # Detect tool markers: --- Write: path ---
        tool_match = _TOOL_MARKER.match(content)
        if tool_match:
            tool_name = tool_match.group(1)
            file_path = tool_match.group(2) or ""
            text.append(f"--- {tool_name}", style=_S_TOOL_HEADER)
            if file_path:
                text.append(": ", style=_S_TOOL_HEADER)
                text.append(file_path, style=_S_FILE_PATH)
            text.append(" ---", style=_S_TOOL_HEADER)
            return text

        # Detect end marker
        if _END_MARKER.match(content):
            text.append(content, style=_S_SEPARATOR)
            return text

        # Detect code lines: "  | code", "  + code", "  - code", "  $ cmd"
        stripped = content.lstrip()
        if stripped.startswith("| "):
            text.append(content, style=_S_CODE)
            return text
        if stripped.startswith("+ "):
            text.append(content, style=_S_CODE_ADD)
            return text
        if stripped.startswith("- "):
            text.append(content, style=_S_CODE_DEL)
            return text
        if stripped.startswith("$ "):
            text.append(content, style=_S_CODE)
            return text
        if stripped == ">>>":
            text.append(content, style=_S_SEPARATOR)
            return text

        # Check for result lines
        if "Agent complete" in content:
            text.append(content, style=_S_RESULT_OK)
            return text

        # Default agent output
        text.append(content, style=_S_DEFAULT)
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
