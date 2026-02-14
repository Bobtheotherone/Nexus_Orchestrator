"""Optional TUI entrypoint with graceful fallback when dependencies are absent."""

from __future__ import annotations

import sys
from importlib.util import find_spec


def tui_available() -> bool:
    """Return whether optional TUI dependencies are available in this environment."""

    return find_spec("textual") is not None


def tui_entrypoint() -> int:
    """Run optional TUI entrypoint or emit a deterministic fallback message."""

    if not tui_available():
        print(
            "TUI unavailable: install optional dependencies (e.g. pip install .[tui]).",
            file=sys.stderr,
        )
        return 2

    print(
        "TUI dependencies detected, but dashboard is not implemented in this build.",
        file=sys.stderr,
    )
    return 2


__all__ = ["tui_available", "tui_entrypoint"]
