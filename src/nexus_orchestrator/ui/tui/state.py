"""Application state — pure data, NO Textual imports.

File: src/nexus_orchestrator/ui/tui/state.py

Owns the canonical state for the TUI controller layer.
All state mutations go through the controller; widgets read snapshots.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final

# ---------------------------------------------------------------------------
# Transcript event model
# ---------------------------------------------------------------------------

MAX_TRANSCRIPT_LINES: Final[int] = 5_000


class EventKind(enum.Enum):
    """Classifies a transcript event for rendering."""

    STDOUT = "stdout"
    STDERR = "stderr"
    SYSTEM = "system"
    COMMAND_HEADER = "command_header"
    EXIT_BADGE = "exit_badge"
    AGENT_HEADER = "agent_header"
    AGENT_RESPONSE = "agent_response"


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """A single event in the transcript log."""

    kind: EventKind
    text: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).strftime("%H:%M:%S"))
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Runner state
# ---------------------------------------------------------------------------


class RunnerStatus(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    GENERATING = "generating"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


@dataclass
class BackendInfo:
    """Detected tool backend info (no Textual dependency)."""

    name: str
    version: str | None
    auth_mode: str | None = None  # "local_cli" or "api_key"
    available: bool = True
    logged_in: bool | None = None
    has_api_key: bool | None = None
    remediation: str | None = None


@dataclass
class RecentRun:
    """A recently-executed command result."""

    command: str
    exit_code: int


@dataclass
class AppState:
    """Root state object for the TUI — mutated only by the controller."""

    # Transcript (ring buffer)
    transcript: deque[TranscriptEvent] = field(
        default_factory=lambda: deque(maxlen=MAX_TRANSCRIPT_LINES)
    )

    # Runner
    runner_status: RunnerStatus = RunnerStatus.IDLE
    current_command: str = ""

    # Sidebar
    backends: list[BackendInfo] = field(default_factory=list)
    recent_runs: list[RecentRun] = field(default_factory=list)
    max_recent: int = 5

    # Command history
    command_history: list[str] = field(default_factory=list)

    # Workspace info (status line)
    workspace_path: str = ""
    git_branch: str = ""
    git_dirty: bool = False

    # Last output (for /copy)
    last_output: str = ""

    # NO_COLOR mode
    no_color: bool = False


__all__ = [
    "AppState",
    "BackendInfo",
    "EventKind",
    "MAX_TRANSCRIPT_LINES",
    "RecentRun",
    "RunnerStatus",
    "TranscriptEvent",
]
