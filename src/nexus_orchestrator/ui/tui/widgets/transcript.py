"""Selectable transcript widget — read-only TextArea with click-and-drag selection.

File: src/nexus_orchestrator/ui/tui/widgets/transcript.py

Uses Textual's TextArea in read-only mode so all transcript content is
selectable via click-and-drag. Events are formatted as plain text with
clear visual markers for each event kind.
"""

from __future__ import annotations

from rich.style import Style
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import TextArea
from textual.widgets.text_area import TextAreaTheme

from nexus_orchestrator.ui.tui.state import EventKind, TranscriptEvent

# Maximum number of lines kept in the TextArea buffer
_MAX_BUFFER_LINES = 10_000

# Custom theme matching the NEXUS color palette
_NEXUS_THEME = TextAreaTheme(
    name="nexus",
    base_style=Style(color="#c8cdd8", bgcolor="#0b1020"),
    cursor_style=Style(color="#c8cdd8", bgcolor="#1a2550"),
    cursor_line_style=Style(bgcolor="#0e1428"),
    selection_style=Style(color="#c8cdd8", bgcolor="#1a3a6a"),
    cursor_line_gutter_style=Style(color="#7f8aa3", bgcolor="#0b1020"),
    gutter_style=Style(color="#7f8aa3", bgcolor="#0b1020"),
)


class TranscriptWidget(Widget):
    """Selectable transcript log backed by a read-only TextArea."""

    DEFAULT_CSS = """
    TranscriptWidget {
        height: 1fr;
        width: 1fr;
    }
    #transcript-area {
        height: 1fr;
        width: 1fr;
    }
    """

    def __init__(self, *, no_color: bool = False, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self._no_color = no_color
        self._event_count = 0
        self._follow_tail = True

    def compose(self) -> ComposeResult:
        area = TextArea(
            "",
            read_only=True,
            show_line_numbers=False,
            soft_wrap=True,
            id="transcript-area",
        )
        yield area

    def on_mount(self) -> None:
        area = self.query_one("#transcript-area", TextArea)
        area.register_theme(_NEXUS_THEME)
        area.theme = "nexus"

    @property
    def text_area(self) -> TextArea:
        return self.query_one("#transcript-area", TextArea)

    def append_event(self, event: TranscriptEvent) -> None:
        """Append a single event to the transcript."""
        area = self.text_area
        formatted = _format_event(event)

        # Insert at end
        current = area.text
        if current:
            text_to_insert = "\n" + formatted
        else:
            text_to_insert = formatted

        # TextArea.insert at end — get end location from text length
        lines = current.split("\n") if current else [""]
        end_row = len(lines) - 1
        end_col = len(lines[-1])
        area.insert(text_to_insert, location=(end_row, end_col))

        self._event_count += 1

        # Enforce ring buffer
        new_text = area.text
        line_list = new_text.split("\n")
        if len(line_list) > _MAX_BUFFER_LINES:
            excess = len(line_list) - _MAX_BUFFER_LINES
            area.delete((0, 0), (excess, 0))

        # Follow tail — move cursor to end and scroll
        if self._follow_tail:
            end_text = area.text
            end_lines = end_text.split("\n") if end_text else [""]
            last_row = len(end_lines) - 1
            last_col = len(end_lines[-1])
            area.move_cursor((last_row, last_col))
            area.scroll_cursor_visible()

    def append_events(self, events: list[TranscriptEvent]) -> None:
        """Append multiple events efficiently."""
        if not events:
            return

        area = self.text_area
        current = area.text

        chunks: list[str] = []
        for event in events:
            chunks.append(_format_event(event))
        combined = "\n".join(chunks)

        if current:
            text_to_insert = "\n" + combined
        else:
            text_to_insert = combined

        lines = current.split("\n") if current else [""]
        end_row = len(lines) - 1
        end_col = len(lines[-1])
        area.insert(text_to_insert, location=(end_row, end_col))

        self._event_count += len(events)

        # Enforce ring buffer
        new_text = area.text
        line_list = new_text.split("\n")
        if len(line_list) > _MAX_BUFFER_LINES:
            excess = len(line_list) - _MAX_BUFFER_LINES
            area.delete((0, 0), (excess, 0))

        # Follow tail
        if self._follow_tail:
            end_text = area.text
            end_lines = end_text.split("\n") if end_text else [""]
            last_row = len(end_lines) - 1
            last_col = len(end_lines[-1])
            area.move_cursor((last_row, last_col))
            area.scroll_cursor_visible()

    def clear(self) -> None:
        """Remove all transcript content."""
        area = self.text_area
        area.clear()
        self._event_count = 0

    def on_text_area_scroll_up(self) -> None:
        """User scrolled up — disable follow tail."""
        self._follow_tail = False

    def on_text_area_scroll_down(self) -> None:
        """User scrolled to bottom — re-enable follow tail."""
        area = self.text_area
        if area.scroll_y >= area.max_scroll_y - 1:
            self._follow_tail = True

    def on_key(self, event: object) -> None:
        """Track scroll position for follow-tail."""
        area = self.text_area
        if area.scroll_y >= area.max_scroll_y - 1:
            self._follow_tail = True
        else:
            self._follow_tail = False


def _format_event(event: TranscriptEvent) -> str:
    """Format a transcript event as plain text for the TextArea."""
    if event.kind == EventKind.COMMAND_HEADER:
        sep = "\u2500" * 60
        return f"{event.text}  [{event.timestamp}]\n{sep}"

    if event.kind == EventKind.EXIT_BADGE:
        sep = "\u2500" * 60
        code = event.exit_code or 0
        badge = f"OK ({code})" if code == 0 else f"FAIL ({code})"
        return f"{sep}\n{badge}"

    if event.kind == EventKind.STDERR:
        return f"ERR: {event.text}"

    if event.kind == EventKind.SYSTEM:
        return f"[system] {event.text}"

    if event.kind == EventKind.AGENT_HEADER:
        sep = "\u2500" * 60
        return f"{event.text}  [{event.timestamp}]\n{sep}"

    if event.kind == EventKind.AGENT_RESPONSE:
        return event.text

    # STDOUT — plain text
    return event.text


__all__ = ["TranscriptWidget"]
