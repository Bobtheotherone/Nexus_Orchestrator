"""Output rendering abstraction for nexus-orchestrator CLI.

File: src/nexus_orchestrator/ui/render.py
Last updated: 2026-02-14

Purpose
- Provide a thin rendering layer for CLI output with optional rich formatting.
- Respect NO_COLOR environment variable and --no-color CLI flag.
- Degrade gracefully when the ``rich`` package is not installed.

What should be included in this file
- CLIRenderer class with methods for common output patterns.
- Factory function to create a renderer with appropriate settings.

Functional requirements
- Plain-text rendering must always work without external dependencies.
- All public methods must be safe to call in any environment.

Non-functional requirements
- No mandatory dependencies beyond the standard library.
- Must pass mypy strict mode.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _color_allowed(no_color_flag: bool) -> bool:
    """Check whether color output should be attempted."""

    if no_color_flag:
        return False
    if os.environ.get("NO_COLOR", ""):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class CLIRenderer:
    """Thin CLI output renderer.

    Produces clean, deterministic plain-text output.
    Respects ``NO_COLOR`` env var and ``--no-color`` flag.
    """

    def __init__(self, *, no_color: bool = False, verbose: bool = False) -> None:
        self.verbose = verbose
        self._color = _color_allowed(no_color)

    def heading(self, text: str) -> None:
        """Print a heading line."""

        print(text)

    def kv(self, key: str, value: object) -> None:
        """Print a key: value pair."""

        print(f"{key}: {value}")

    def text(self, line: str) -> None:
        """Print a plain text line."""

        print(line)

    def blank(self) -> None:
        """Print a blank line."""

        print()

    def section(self, title: str) -> None:
        """Print a section header with a preceding blank line."""

        print(f"\n{title}")

    def warning(self, text: str) -> None:
        """Print a warning message."""

        print(f"  Warning: {text}")

    def items(self, entries: Sequence[str], *, prefix: str = "- ") -> None:
        """Print a bulleted list."""

        for entry in entries:
            print(f"  {prefix}{entry}")

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        title: str | None = None,
    ) -> None:
        """Print a formatted ASCII table."""

        if not rows:
            return

        col_count = len(headers)
        widths = [len(h) for h in headers]
        for row in rows:
            for i in range(min(len(row), col_count)):
                widths[i] = max(widths[i], len(str(row[i])))

        def _pad(cells: Sequence[str]) -> str:
            parts: list[str] = []
            for i in range(col_count):
                cell = str(cells[i]) if i < len(cells) else ""
                parts.append(cell.ljust(widths[i]))
            return "  ".join(parts)

        if title:
            self.section(title)
        print(f"  {_pad(list(headers))}")
        print(f"  {'  '.join('-' * w for w in widths)}")
        for row in rows:
            print(f"  {_pad(list(row))}")

    def next_steps(self, steps: Sequence[str]) -> None:
        """Print actionable next-step hints."""

        if not steps:
            return
        self.section("Next steps:")
        for step in steps:
            print(f"  $ {step}")

    def ok(self, label: str) -> None:
        """Print a passing diagnostic check."""

        print(f"  OK  {label}")

    def fail(self, label: str) -> None:
        """Print a failing diagnostic check."""

        print(f"  FAIL  {label}")


def create_renderer(*, no_color: bool = False, verbose: bool = False) -> CLIRenderer:
    """Create a CLI renderer with the given settings."""

    return CLIRenderer(no_color=no_color, verbose=verbose)


__all__ = ["CLIRenderer", "create_renderer"]
