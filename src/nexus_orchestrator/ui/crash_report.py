"""TUI crash report capture and recovery.

File: src/nexus_orchestrator/ui/crash_report.py

Purpose
- Capture TUI crashes to a JSON file for post-mortem diagnosis.
- Provide load/clear helpers for the recovery banner on next launch.

Security
- Never includes API keys, environment variables, or secrets.
- Traceback and message are truncated to prevent leaking file contents.
"""

from __future__ import annotations

import json
import os
import platform
import traceback
from datetime import UTC, datetime
from pathlib import Path

_CRASH_FILENAME = "last_tui_crash.json"
_MAX_TB_LINES = 50
_MAX_MESSAGE_LEN = 500


def _config_dir() -> Path:
    """Return the platform-appropriate user config directory for nexus_orchestrator."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nexus_orchestrator"


def _crash_path() -> Path:
    """Return the path to the crash report file."""
    return _config_dir() / _CRASH_FILENAME


def save_crash_report(exc: BaseException) -> Path:
    """Write a crash report JSON file. Returns the file path.

    The report contains NO secrets — only exception metadata and a
    truncated traceback.
    """
    path = _crash_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    # Keep only the last _MAX_TB_LINES lines
    tb_text = "".join(tb_lines[-_MAX_TB_LINES:])

    data: dict[str, object] = {
        "exception_type": type(exc).__qualname__,
        "message": str(exc)[:_MAX_MESSAGE_LEN],
        "timestamp": datetime.now(UTC).isoformat(),
        "traceback": tb_text[:5000],
    }
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_crash_report() -> dict[str, object] | None:
    """Load crash report if it exists, else None."""
    path = _crash_path()
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return None


def clear_crash_report() -> None:
    """Delete the crash report file if it exists."""
    path = _crash_path()
    path.unlink(missing_ok=True)


__all__ = [
    "clear_crash_report",
    "load_crash_report",
    "save_crash_report",
]
