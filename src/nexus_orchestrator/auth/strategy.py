"""Authentication strategy — capability detection for LOCAL_CLI vs API_KEY modes.

File: src/nexus_orchestrator/auth/strategy.py

NO Textual imports. Pure logic for detecting which auth modes are available
per backend (codex, claude, anthropic-api, openai-api).

Key principle: LOCAL_CLI is the preferred default. API_KEY mode is opt-in.
The TUI must launch and be fully usable with ZERO API keys set.
"""

from __future__ import annotations

import asyncio
import enum
import importlib
import importlib.util
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Auth mode enum
# ---------------------------------------------------------------------------


class AuthMode(enum.Enum):
    """How a backend authenticates."""

    LOCAL_CLI = "local_cli"
    API_KEY = "api_key"


# ---------------------------------------------------------------------------
# Backend auth status
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendAuthStatus:
    """Auth capability snapshot for a single backend."""

    name: str
    auth_mode: AuthMode
    available: bool
    logged_in: bool | None  # None = not applicable (e.g. API_KEY mode)
    has_api_key: bool | None  # None = not applicable (e.g. LOCAL_CLI mode)
    version: str | None = None
    binary_path: str | None = None
    sdk_installed: bool | None = None
    remediation: str | None = None


# ---------------------------------------------------------------------------
# CLI binary names and env vars
# ---------------------------------------------------------------------------

_CLI_BACKENDS: Final[dict[str, str]] = {
    "claude": "claude",
    "codex": "codex",
}

_API_BACKENDS: Final[dict[str, dict[str, str]]] = {
    "anthropic": {
        "sdk": "anthropic",
        "env_vars": "ANTHROPIC_API_KEY,NEXUS_ANTHROPIC_API_KEY",
        "display": "Anthropic API",
        "install_hint": "pip install nexus-orchestrator[anthropic]",
    },
    "openai": {
        "sdk": "openai",
        "env_vars": "OPENAI_API_KEY,NEXUS_OPENAI_API_KEY",
        "display": "OpenAI API",
        "install_hint": "pip install nexus-orchestrator[openai]",
    },
}

_CLI_LOGIN_CHECK: Final[dict[str, list[str]]] = {
    "claude": ["claude", "--version"],
    "codex": ["codex", "--version"],
}

_CLI_INSTALL_HINTS: Final[dict[str, str]] = {
    "claude": "See https://claude.ai/download",
    "codex": "npm install -g @openai/codex",
}


# ---------------------------------------------------------------------------
# CLI detection
# ---------------------------------------------------------------------------


def detect_cli(name: str) -> BackendAuthStatus:
    """Detect a LOCAL_CLI backend (claude or codex).

    Checks: binary exists on PATH, --version succeeds, login state.
    """
    binary = _CLI_BACKENDS.get(name, name)
    path = shutil.which(binary)

    if path is None:
        hint = _CLI_INSTALL_HINTS.get(name, f"Install {name} CLI")
        return BackendAuthStatus(
            name=name,
            auth_mode=AuthMode.LOCAL_CLI,
            available=False,
            logged_in=None,
            has_api_key=None,
            remediation=f"{name} CLI not found. {hint}",
        )

    version = _get_version(path)
    logged_in = _check_login_state(name, path)

    if not logged_in:
        return BackendAuthStatus(
            name=name,
            auth_mode=AuthMode.LOCAL_CLI,
            available=True,
            logged_in=False,
            has_api_key=None,
            version=version,
            binary_path=path,
            remediation=f"Run `{binary}` to log in, then retry.",
        )

    return BackendAuthStatus(
        name=name,
        auth_mode=AuthMode.LOCAL_CLI,
        available=True,
        logged_in=True,
        has_api_key=None,
        version=version,
        binary_path=path,
    )


def detect_api_mode(name: str) -> BackendAuthStatus:
    """Detect an API_KEY backend (anthropic or openai).

    Checks: SDK importable, API key present in env.
    """
    info = _API_BACKENDS.get(name)
    if info is None:
        return BackendAuthStatus(
            name=name,
            auth_mode=AuthMode.API_KEY,
            available=False,
            logged_in=None,
            has_api_key=None,
            remediation=f"Unknown API backend: {name}",
        )

    sdk_name = info["sdk"]
    sdk_installed = importlib.util.find_spec(sdk_name) is not None

    env_vars = info["env_vars"].split(",")
    has_key = any(
        os.environ.get(var, "").strip()
        for var in env_vars
    )

    if not sdk_installed and not has_key:
        return BackendAuthStatus(
            name=name,
            auth_mode=AuthMode.API_KEY,
            available=False,
            logged_in=None,
            has_api_key=False,
            sdk_installed=False,
            remediation=(
                f"{info['display']} SDK not installed and no API key set. "
                f"Install: {info['install_hint']}  "
                f"Then set {env_vars[0]}."
            ),
        )

    if not sdk_installed:
        return BackendAuthStatus(
            name=name,
            auth_mode=AuthMode.API_KEY,
            available=False,
            logged_in=None,
            has_api_key=True,
            sdk_installed=False,
            remediation=(
                f"{info['display']} SDK not installed. "
                f"Install: {info['install_hint']}"
            ),
        )

    if not has_key:
        return BackendAuthStatus(
            name=name,
            auth_mode=AuthMode.API_KEY,
            available=False,
            logged_in=None,
            has_api_key=False,
            sdk_installed=True,
            remediation=f"Set {env_vars[0]} to enable {info['display']} mode.",
        )

    return BackendAuthStatus(
        name=name,
        auth_mode=AuthMode.API_KEY,
        available=True,
        logged_in=None,
        has_api_key=True,
        sdk_installed=True,
    )


# ---------------------------------------------------------------------------
# Full auth detection
# ---------------------------------------------------------------------------


def detect_all_auth() -> list[BackendAuthStatus]:
    """Detect all backends in deterministic order: CLI backends first, then API."""
    results: list[BackendAuthStatus] = []
    for name in _CLI_BACKENDS:
        results.append(detect_cli(name))
    for name in _API_BACKENDS:
        results.append(detect_api_mode(name))
    return results


def resolve_auth(
    *,
    prefer: AuthMode = AuthMode.LOCAL_CLI,
    provider_hint: str | None = None,
) -> BackendAuthStatus | None:
    """Resolve the best available backend for the preferred auth mode.

    When prefer=LOCAL_CLI (default):
      1. Try claude CLI (logged in)
      2. Try codex CLI (logged in)
      3. Fall back to anthropic API if available
      4. Fall back to openai API if available

    When prefer=API_KEY:
      1. Try anthropic API
      2. Try openai API
      3. Fall back to claude CLI
      4. Fall back to codex CLI

    If provider_hint is set, prefer that specific backend name.
    """
    all_statuses = detect_all_auth()

    # If a specific provider is hinted, check it first
    if provider_hint:
        for status in all_statuses:
            if status.name == provider_hint and status.available:
                if status.auth_mode == AuthMode.LOCAL_CLI and status.logged_in:
                    return status
                if status.auth_mode == AuthMode.API_KEY and status.has_api_key:
                    return status

    if prefer == AuthMode.LOCAL_CLI:
        # CLI backends first
        for status in all_statuses:
            if status.auth_mode == AuthMode.LOCAL_CLI and status.available and status.logged_in:
                return status
        # API fallback
        for status in all_statuses:
            if status.auth_mode == AuthMode.API_KEY and status.available:
                return status
    else:
        # API backends first
        for status in all_statuses:
            if status.auth_mode == AuthMode.API_KEY and status.available:
                return status
        # CLI fallback
        for status in all_statuses:
            if status.auth_mode == AuthMode.LOCAL_CLI and status.available and status.logged_in:
                return status

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_version(binary_path: str) -> str | None:
    """Run binary --version and extract version string."""
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


def _check_login_state(name: str, binary_path: str) -> bool:
    """Check if a CLI tool is logged in.

    For both claude and codex, running --version succeeding is a reasonable
    indicator the tool is functional. A more thorough check would attempt
    a minimal prompt, but that's too slow for detection.

    We trust that if the binary exists and --version succeeds, the user
    has completed their login flow (which is managed by those tools).
    The ToolProvider._handle_error will catch auth errors at runtime.
    """
    # If --version succeeds, the tool is installed and functional.
    # Auth errors will be caught at runtime by ToolProvider._handle_error.
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


__all__ = [
    "AuthMode",
    "BackendAuthStatus",
    "detect_all_auth",
    "detect_api_mode",
    "detect_cli",
    "resolve_auth",
]
