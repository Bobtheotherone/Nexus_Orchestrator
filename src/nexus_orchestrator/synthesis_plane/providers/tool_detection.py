"""Tool backend detection for local CLI tools (codex, claude).

File: src/nexus_orchestrator/synthesis_plane/providers/tool_detection.py

Purpose
- Detect whether codex and/or claude CLI binaries are installed and usable.
- Return structured metadata (path, version) for onboarding and status display.

Security
- Never logs secrets or attempts to extract auth tokens.
- Detection is offline (no network calls); only runs --version locally.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import Final

# Canonical binary names for each tool backend
TOOL_BINARY_NAMES: Final[dict[str, str]] = {
    "codex": "codex",
    "claude": "claude",
}

# Deterministic detection order
_DETECTION_ORDER: Final[tuple[str, ...]] = ("codex", "claude")


@dataclass(frozen=True, slots=True)
class ToolBackendInfo:
    """Metadata about a detected CLI tool backend."""

    name: str
    binary_path: str
    version: str | None


def detect_tool_backend(name: str) -> ToolBackendInfo | None:
    """Detect a single CLI tool backend by name.

    Returns ToolBackendInfo if the binary is found on PATH, or None.
    The version field may be None if --version fails or times out.
    """
    binary = TOOL_BINARY_NAMES.get(name, name)
    path = shutil.which(binary)
    if path is None:
        return None

    version = _get_version(path)
    return ToolBackendInfo(name=name, binary_path=path, version=version)


def detect_codex_cli() -> ToolBackendInfo | None:
    """Detect the OpenAI Codex CLI."""
    return detect_tool_backend("codex")


def detect_claude_code_cli() -> ToolBackendInfo | None:
    """Detect the Anthropic Claude Code CLI."""
    return detect_tool_backend("claude")


def detect_all_backends() -> list[ToolBackendInfo]:
    """Detect all available CLI tool backends in deterministic order."""
    found: list[ToolBackendInfo] = []
    for name in _DETECTION_ORDER:
        info = detect_tool_backend(name)
        if info is not None:
            found.append(info)
    return found


def _get_version(binary_path: str) -> str | None:
    """Run binary --version and extract the version string.

    Returns None on any failure (timeout, non-zero exit, parse error).
    """
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


__all__ = [
    "TOOL_BINARY_NAMES",
    "ToolBackendInfo",
    "detect_all_backends",
    "detect_claude_code_cli",
    "detect_codex_cli",
    "detect_tool_backend",
]
