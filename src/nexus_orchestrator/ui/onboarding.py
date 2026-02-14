"""Credential onboarding wizard and persistence helpers.

File: src/nexus_orchestrator/ui/onboarding.py

Purpose
- Manage first-run credential onboarding for provider API keys.
- Persist onboarding completion state in a sentinel JSON file.
- Store credentials securely via OS keychain (keyring) or local file with strict permissions.
- Support env-var-only mode as a fallback (no storage, detected at runtime).

Security
- Never log or display full API keys; only show last 4 characters.
- Credential file uses mode 0600 on POSIX systems.
- onboarding.json contains NO secrets — only provider names and completion timestamp.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

# Provider key page URLs (opened via stdlib webbrowser, not OAuth)
PROVIDER_KEY_URLS: Final[dict[str, str]] = {
    "openai": "https://platform.openai.com/api-keys",
    "anthropic": "https://console.anthropic.com/settings/keys",
}

# Friendly display names
PROVIDER_DISPLAY_NAMES: Final[dict[str, str]] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
}

# Environment variable names per provider
PROVIDER_ENV_VARS: Final[dict[str, str]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Key format prefixes for local validation (no network call)
_KEY_PREFIX_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "openai": ("sk-",),
    "anthropic": ("sk-ant-",),
}

# Keyring service name
_KEYRING_SERVICE: Final[str] = "nexus-orchestrator"

# UX copy: API key vs subscription explanation
API_KEY_NOTICE: Final[str] = (
    "Note: ChatGPT / Claude monthly subscriptions are separate from API access.\n"
    "To use APIs you generally need an API key from the provider console.\n"
    "API usage may be billed separately depending on your account settings."
)

# Tool backend display names
TOOL_BACKEND_DISPLAY: Final[dict[str, str]] = {
    "codex": "Codex CLI (OpenAI)",
    "claude": "Claude Code CLI (Anthropic)",
}

# Install hints for missing tools
TOOL_INSTALL_HINTS: Final[dict[str, str]] = {
    "codex": "npm install -g @openai/codex",
    "claude": "See https://claude.ai/download",
}

# UX copy: subscription-first tool backend explanation
TOOL_BACKEND_NOTICE: Final[str] = (
    "Your ChatGPT/Claude web subscriptions aren't the same as API access.\n"
    "However, you can use Codex and Claude Code (local tools you're already\n"
    "logged into) as execution backends for Nexus.\n"
    "No API keys required for this path."
)


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


def onboarding_path() -> Path:
    """Return the path to onboarding.json sentinel file."""
    return _config_dir() / "onboarding.json"


def credentials_path() -> Path:
    """Return the path to credentials.toml file."""
    return _config_dir() / "credentials.toml"


def is_onboarding_complete() -> bool:
    """Check whether onboarding has been completed (at least one provider configured)."""
    sentinel = onboarding_path()
    if not sentinel.exists():
        return False
    try:
        data = json.loads(sentinel.read_text(encoding="utf-8"))
        return bool(data.get("completed", False))
    except (json.JSONDecodeError, OSError):
        return False


def get_onboarding_state() -> dict[str, object]:
    """Read the onboarding sentinel, returning empty dict if missing."""
    sentinel = onboarding_path()
    if not sentinel.exists():
        return {}
    try:
        return dict(json.loads(sentinel.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


def mark_onboarding_complete(
    providers: list[str],
    *,
    backends: list[str] | None = None,
) -> None:
    """Write the onboarding sentinel file marking completion.

    The sentinel contains NO secrets — only provider/backend names and timestamp.
    """
    sentinel = onboarding_path()
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "backends": sorted(backends) if backends else [],
        "completed": True,
        "completed_at": datetime.now(UTC).isoformat(),
        "providers": sorted(providers),
    }
    sentinel.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def redact_key(key: str) -> str:
    """Redact an API key, showing only the last 4 characters."""
    if len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


def validate_key_format(provider: str, key: str) -> tuple[bool, str]:
    """Validate API key format locally (no network call).

    Returns (is_valid, message).
    """
    key = key.strip()
    if not key:
        return False, "Key is empty."
    if len(key) < 10:
        return False, "Key is too short (expected 20+ characters)."
    prefixes = _KEY_PREFIX_PATTERNS.get(provider)
    if prefixes and not any(key.startswith(p) for p in prefixes):
        expected = " or ".join(f'"{p}..."' for p in prefixes)
        display = PROVIDER_DISPLAY_NAMES.get(provider, provider)
        return False, f"{display} keys typically start with {expected}."
    return True, "Key format looks valid."


def _keyring_available() -> bool:
    """Check if the keyring package is importable and functional."""
    try:
        import importlib

        _kr = importlib.import_module("keyring")
        # Verify backend is usable (not the fail backend)
        backend = _kr.get_keyring()
        backend_name = type(backend).__name__.lower()
        return "fail" not in backend_name and "null" not in backend_name
    except Exception:  # noqa: BLE001
        return False


def store_credential(provider: str, api_key: str) -> str:
    """Store an API key securely. Returns storage method used: 'keyring' or 'file'."""
    if _keyring_available():
        import importlib

        _kr = importlib.import_module("keyring")
        _kr.set_password(_KEYRING_SERVICE, provider, api_key)
        return "keyring"

    # Fall back to credentials.toml with strict permissions
    cred_path = credentials_path()
    cred_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing credentials if any
    existing: dict[str, str] = {}
    if cred_path.exists():
        existing = _load_credentials_file(cred_path)

    existing[provider] = api_key
    _write_credentials_file(cred_path, existing)
    return "file"


def load_credential(provider: str) -> str | None:
    """Load a credential for a provider. Checks: env var -> keyring -> file."""
    # 1. Check environment variable
    env_var = PROVIDER_ENV_VARS.get(provider, "")
    if env_var:
        env_value = os.environ.get(env_var, "").strip()
        if env_value:
            return env_value

    # 2. Check keyring
    if _keyring_available():
        try:
            import importlib as _il

            _kr = _il.import_module("keyring")
            value = _kr.get_password(_KEYRING_SERVICE, provider)
            if isinstance(value, str) and value:
                return value
        except Exception:  # noqa: BLE001
            pass

    # 3. Check credentials file
    cred_path = credentials_path()
    if cred_path.exists():
        creds = _load_credentials_file(cred_path)
        value = creds.get(provider, "").strip()
        if value:
            return value

    return None


def has_any_credential() -> bool:
    """Check if at least one provider has a credential available."""
    return any(load_credential(provider) is not None for provider in PROVIDER_ENV_VARS)


def detect_env_credentials() -> list[str]:
    """Return list of providers whose keys are available via environment variables."""
    found: list[str] = []
    for provider, env_var in sorted(PROVIDER_ENV_VARS.items()):
        if os.environ.get(env_var, "").strip():
            found.append(provider)
    return found


def open_provider_key_page(provider: str) -> bool:
    """Open the provider's API key page in the default browser. Returns success."""
    import webbrowser

    url = PROVIDER_KEY_URLS.get(provider)
    if not url:
        return False
    try:
        webbrowser.open(url)
        return True
    except Exception:  # noqa: BLE001
        print(f"Could not open browser. Visit: {url}", file=sys.stderr)
        return False


def _load_credentials_file(path: Path) -> dict[str, str]:
    """Parse a simple TOML-like credentials file (key = "value" lines)."""
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    result[key] = value
    except OSError:
        pass
    return result


def _write_credentials_file(path: Path, credentials: dict[str, str]) -> None:
    """Write credentials to a TOML-like file with strict POSIX permissions."""
    lines = ["# nexus-orchestrator credentials (auto-generated)\n"]
    for key in sorted(credentials):
        lines.append(f'{key} = "{credentials[key]}"\n')

    path.write_text("".join(lines), encoding="utf-8")

    # Set strict permissions on POSIX
    if platform.system() != "Windows":
        with contextlib.suppress(OSError):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600


def get_onboarding_backends() -> list[str]:
    """Return tool backends from onboarding state, or empty list."""
    state = get_onboarding_state()
    backends = state.get("backends", [])
    return list(backends) if isinstance(backends, list) else []


__all__ = [
    "API_KEY_NOTICE",
    "PROVIDER_DISPLAY_NAMES",
    "PROVIDER_ENV_VARS",
    "PROVIDER_KEY_URLS",
    "TOOL_BACKEND_DISPLAY",
    "TOOL_BACKEND_NOTICE",
    "TOOL_INSTALL_HINTS",
    "credentials_path",
    "detect_env_credentials",
    "get_onboarding_backends",
    "get_onboarding_state",
    "has_any_credential",
    "is_onboarding_complete",
    "load_credential",
    "mark_onboarding_complete",
    "onboarding_path",
    "open_provider_key_page",
    "redact_key",
    "store_credential",
    "validate_key_format",
]
