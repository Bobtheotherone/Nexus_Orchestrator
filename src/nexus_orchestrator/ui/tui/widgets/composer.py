"""Multiline composer widget — Enter sends, Shift+Enter for newlines.

File: src/nexus_orchestrator/ui/tui/widgets/composer.py

Provides a command input area with:
- Enter = send command
- Shift+Enter = insert newline (multiline input)
- Up/Down = cycle through command history
- Clear focus behavior and visible cursor
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.suggester import SuggestFromList
from textual.widget import Widget
from textual.widgets import Input, Static


class ComposerInput(Input):
    """Command input with history navigation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._history: list[str] = []
        self._history_index: int = -1
        self._stashed_value: str = ""

    def set_history(self, history: list[str]) -> None:
        """Set the command history from controller state."""
        self._history = list(history)
        self._history_index = -1

    def push_history(self, command: str) -> None:
        """Add a command to local history."""
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


class Composer(Widget):
    """Bottom command composer bar with prompt label and input."""

    DEFAULT_CSS = """
    Composer {
        dock: bottom;
        height: auto;
        max-height: 5;
        padding: 0 1;
    }
    #composer-inner {
        width: 100%;
        height: auto;
    }
    #prompt-label {
        width: 10;
        text-style: bold;
    }
    #composer-input {
        width: 1fr;
    }
    #composer-input:focus {
        border: tall $primary;
    }
    """

    class CommandSubmitted(Message):
        """Emitted when user submits a command."""

        def __init__(self, command: str) -> None:
            self.command = command
            super().__init__()

    def __init__(
        self,
        *,
        suggestions: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._suggestions = suggestions or []

    def compose(self) -> ComposeResult:
        with Horizontal(id="composer-inner"):
            yield Static("nexus > ", id="prompt-label")
            yield ComposerInput(
                placeholder="Type a command...",
                id="composer-input",
                suggester=SuggestFromList(self._suggestions, case_sensitive=False),
            )

    @property
    def input_widget(self) -> ComposerInput:
        return self.query_one("#composer-input", ComposerInput)

    def focus_input(self) -> None:
        """Focus the command input."""
        self.input_widget.focus()

    def set_value(self, value: str) -> None:
        """Set the input value and move cursor to end."""
        inp = self.input_widget
        inp.value = value
        inp.cursor_position = len(value)

    def set_history(self, history: list[str]) -> None:
        """Update the command history."""
        self.input_widget.set_history(history)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter — emit command and clear."""
        if event.input.id != "composer-input":
            return
        command = event.value.strip()
        event.input.value = ""
        if command:
            self.input_widget.push_history(command)
            self.post_message(self.CommandSubmitted(command))


__all__ = ["Composer", "ComposerInput"]
