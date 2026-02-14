"""Optional TUI entrypoint with graceful fallback when dependencies are absent.

File: src/nexus_orchestrator/ui/tui.py

Purpose
- Expose ``tui_available()`` for dependency checks and ``run_tui()`` for launching.
- Print a clear install hint and exit code 2 if Textual is not installed.
- Delegate to ``tui_app.run_tui_app()`` when dependencies are satisfied.

Non-functional requirements
- No mandatory dependency on Textual — this module must import cleanly without it.
- Must pass mypy strict mode.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def tui_available() -> bool:
    """Return whether optional TUI dependencies are available in this environment."""

    return find_spec("textual") is not None


def run_tui(argv: Sequence[str] | None = None) -> int:
    """Run the interactive TUI, or exit with code 2 and install hint if unavailable."""

    if not tui_available():
        print(
            "TUI requires optional dependency. Install: pip install -e '.[tui]'",
            file=sys.stderr,
        )
        return 2

    no_color = False
    if argv is not None:
        args = list(argv)
        if "--no-color" in args:
            no_color = True

    from nexus_orchestrator.ui.tui_app import run_tui_app

    return run_tui_app(no_color=no_color)


def tui_entrypoint() -> int:
    """Console-script compatible entrypoint for the TUI."""

    return run_tui()


__all__ = ["run_tui", "tui_available", "tui_entrypoint"]
