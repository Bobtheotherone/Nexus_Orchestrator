"""UI package exports for CLI, rendering, and optional TUI surfaces."""

from nexus_orchestrator.ui.cli import build_parser, cli_entrypoint, main, run_cli
from nexus_orchestrator.ui.crash_report import (
    clear_crash_report,
    load_crash_report,
    save_crash_report,
)
from nexus_orchestrator.ui.render import CLIRenderer, create_renderer
from nexus_orchestrator.ui.tui import run_tui, tui_available, tui_entrypoint

__all__ = [
    "CLIRenderer",
    "build_parser",
    "clear_crash_report",
    "cli_entrypoint",
    "create_renderer",
    "load_crash_report",
    "main",
    "run_cli",
    "run_tui",
    "save_crash_report",
    "tui_available",
    "tui_entrypoint",
]
