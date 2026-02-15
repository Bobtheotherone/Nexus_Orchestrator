"""Tests for auth strategy — capability detection, mode resolution, zero-key operation.

Tests cover:
- AuthMode enum values
- BackendAuthStatus construction
- detect_cli() with mocked binary lookups
- detect_api_mode() with mocked SDK/env detection
- detect_all_auth() returns all backends
- resolve_auth() preference ordering
- Zero-key startup (no API keys, no CLIs → returns None gracefully)
- LOCAL_CLI mode works when CLI is available
- API_KEY mode works when SDK + key are available
- Not-logged-in remediation messages
- Missing API key remediation messages
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from nexus_orchestrator.auth.strategy import (
    AuthMode,
    BackendAuthStatus,
    detect_all_auth,
    detect_api_mode,
    detect_cli,
    resolve_auth,
)


# ---------------------------------------------------------------------------
# AuthMode / BackendAuthStatus basics
# ---------------------------------------------------------------------------


class TestAuthMode:
    def test_values(self) -> None:
        assert AuthMode.LOCAL_CLI.value == "local_cli"
        assert AuthMode.API_KEY.value == "api_key"

    def test_all_members(self) -> None:
        assert set(AuthMode) == {AuthMode.LOCAL_CLI, AuthMode.API_KEY}


class TestBackendAuthStatus:
    def test_construction(self) -> None:
        status = BackendAuthStatus(
            name="claude",
            auth_mode=AuthMode.LOCAL_CLI,
            available=True,
            logged_in=True,
            has_api_key=None,
            version="1.0.0",
            binary_path="/usr/bin/claude",
        )
        assert status.name == "claude"
        assert status.auth_mode == AuthMode.LOCAL_CLI
        assert status.available is True
        assert status.logged_in is True
        assert status.has_api_key is None
        assert status.version == "1.0.0"
        assert status.binary_path == "/usr/bin/claude"
        assert status.remediation is None

    def test_unavailable_with_remediation(self) -> None:
        status = BackendAuthStatus(
            name="codex",
            auth_mode=AuthMode.LOCAL_CLI,
            available=False,
            logged_in=None,
            has_api_key=None,
            remediation="Install codex CLI",
        )
        assert status.available is False
        assert status.remediation == "Install codex CLI"


# ---------------------------------------------------------------------------
# detect_cli()
# ---------------------------------------------------------------------------


class TestDetectCli:
    def test_cli_not_found(self) -> None:
        with patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None):
            result = detect_cli("claude")
        assert result.name == "claude"
        assert result.auth_mode == AuthMode.LOCAL_CLI
        assert result.available is False
        assert result.logged_in is None
        assert result.remediation is not None
        assert "not found" in result.remediation

    def test_cli_found_and_logged_in(self) -> None:
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/claude"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="1.2.3"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=True),
        ):
            result = detect_cli("claude")
        assert result.name == "claude"
        assert result.available is True
        assert result.logged_in is True
        assert result.version == "1.2.3"
        assert result.binary_path == "/usr/bin/claude"
        assert result.remediation is None

    def test_cli_found_but_not_logged_in(self) -> None:
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/codex"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="0.5.0"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=False),
        ):
            result = detect_cli("codex")
        assert result.available is True
        assert result.logged_in is False
        assert result.remediation is not None
        assert "log in" in result.remediation.lower()

    def test_unknown_cli_backend(self) -> None:
        with patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None):
            result = detect_cli("unknown_tool")
        assert result.available is False


# ---------------------------------------------------------------------------
# detect_api_mode()
# ---------------------------------------------------------------------------


class TestDetectApiMode:
    def test_unknown_api_backend(self) -> None:
        result = detect_api_mode("unknown_provider")
        assert result.available is False
        assert result.remediation is not None

    def test_no_sdk_no_key(self) -> None:
        with (
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = detect_api_mode("anthropic")
        assert result.available is False
        assert result.sdk_installed is False
        assert result.has_api_key is False
        assert "SDK not installed" in (result.remediation or "")
        assert "API key" in (result.remediation or "")

    def test_sdk_installed_no_key(self) -> None:
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = detect_api_mode("anthropic")
        assert result.available is False
        assert result.sdk_installed is True
        assert result.has_api_key is False
        assert "ANTHROPIC_API_KEY" in (result.remediation or "")

    def test_no_sdk_but_key_set(self) -> None:
        with (
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=True),
        ):
            result = detect_api_mode("anthropic")
        assert result.available is False
        assert result.sdk_installed is False
        assert result.has_api_key is True
        assert "SDK not installed" in (result.remediation or "")

    def test_sdk_and_key_available(self) -> None:
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=True),
        ):
            result = detect_api_mode("anthropic")
        assert result.available is True
        assert result.sdk_installed is True
        assert result.has_api_key is True
        assert result.remediation is None

    def test_nexus_prefixed_key(self) -> None:
        """NEXUS_ANTHROPIC_API_KEY should also work."""
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"NEXUS_ANTHROPIC_API_KEY": "sk-ant-test123"}, clear=True),
        ):
            result = detect_api_mode("anthropic")
        assert result.available is True
        assert result.has_api_key is True

    def test_openai_api_mode(self) -> None:
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}, clear=True),
        ):
            result = detect_api_mode("openai")
        assert result.name == "openai"
        assert result.available is True
        assert result.auth_mode == AuthMode.API_KEY


# ---------------------------------------------------------------------------
# detect_all_auth()
# ---------------------------------------------------------------------------


class TestDetectAllAuth:
    def test_returns_all_backends(self) -> None:
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            results = detect_all_auth()

        names = [r.name for r in results]
        assert "claude" in names
        assert "codex" in names
        assert "anthropic" in names
        assert "openai" in names
        # CLI backends come first
        assert names.index("claude") < names.index("anthropic")
        assert names.index("codex") < names.index("anthropic")

    def test_zero_key_all_unavailable(self) -> None:
        """With no CLIs and no API keys, all backends are unavailable but no crash."""
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            results = detect_all_auth()
        assert all(not r.available for r in results)
        # Every backend has a remediation hint
        assert all(r.remediation is not None for r in results)


# ---------------------------------------------------------------------------
# resolve_auth()
# ---------------------------------------------------------------------------


class TestResolveAuth:
    def test_no_backends_returns_none(self) -> None:
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = resolve_auth()
        assert result is None

    def test_prefers_local_cli_over_api(self) -> None:
        """When both claude CLI and anthropic API are available, prefer CLI."""
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/claude"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="1.0"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=True),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
        ):
            result = resolve_auth(prefer=AuthMode.LOCAL_CLI)
        assert result is not None
        assert result.auth_mode == AuthMode.LOCAL_CLI
        assert result.name in ("claude", "codex")

    def test_api_preferred_when_requested(self) -> None:
        """When prefer=API_KEY and API is available, use API."""
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/claude"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="1.0"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=True),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
        ):
            result = resolve_auth(prefer=AuthMode.API_KEY)
        assert result is not None
        assert result.auth_mode == AuthMode.API_KEY

    def test_falls_back_to_api_when_no_cli(self) -> None:
        """When prefer=LOCAL_CLI but no CLIs installed, falls back to API."""
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
        ):
            result = resolve_auth(prefer=AuthMode.LOCAL_CLI)
        assert result is not None
        assert result.auth_mode == AuthMode.API_KEY

    def test_falls_back_to_cli_when_no_api(self) -> None:
        """When prefer=API_KEY but no API available, falls back to CLI."""
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/claude"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="1.0"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=True),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = resolve_auth(prefer=AuthMode.API_KEY)
        assert result is not None
        assert result.auth_mode == AuthMode.LOCAL_CLI

    def test_provider_hint_overrides_preference(self) -> None:
        """provider_hint selects a specific backend."""
        mock_spec = MagicMock()
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/claude"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="1.0"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=True),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=mock_spec),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
        ):
            result = resolve_auth(prefer=AuthMode.LOCAL_CLI, provider_hint="anthropic")
        assert result is not None
        assert result.name == "anthropic"
        assert result.auth_mode == AuthMode.API_KEY

    def test_cli_not_logged_in_skipped(self) -> None:
        """CLI that exists but is not logged in should be skipped."""
        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value="/usr/bin/claude"),
            patch("nexus_orchestrator.auth.strategy._get_version", return_value="1.0"),
            patch("nexus_orchestrator.auth.strategy._check_login_state", return_value=False),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            result = resolve_auth(prefer=AuthMode.LOCAL_CLI)
        # CLI not logged in, no API available → None
        assert result is None


# ---------------------------------------------------------------------------
# DesignDocGenerator with auth strategy
# ---------------------------------------------------------------------------


class TestDesignDocGeneratorAuth:
    def test_no_provider_available_error(self) -> None:
        """DesignDocGenerator raises NoProviderAvailableError when no backend."""
        from nexus_orchestrator.ui.tui.services.design_doc import (
            DesignDocGenerator,
            NoProviderAvailableError,
        )

        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            gen = DesignDocGenerator()
            with pytest.raises(NoProviderAvailableError) as exc_info:
                gen._ensure_provider()
            msg = str(exc_info.value)
            assert "No provider available" in msg
            assert "Claude Code CLI" in msg
            assert "ANTHROPIC_API_KEY" in msg

    def test_injected_provider_bypasses_auth(self) -> None:
        """When a provider is injected, auth resolution is bypassed."""
        from nexus_orchestrator.ui.tui.services.design_doc import DesignDocGenerator

        mock_provider = MagicMock()
        gen = DesignDocGenerator(provider=mock_provider)
        result = gen._ensure_provider()
        assert result is mock_provider

    def test_creates_tool_provider_for_cli(self) -> None:
        """When resolve_auth returns a CLI backend, ToolProvider is created."""
        from nexus_orchestrator.ui.tui.services.design_doc import _create_provider_from_auth

        auth = BackendAuthStatus(
            name="claude",
            auth_mode=AuthMode.LOCAL_CLI,
            available=True,
            logged_in=True,
            has_api_key=None,
            binary_path="/usr/bin/claude",
        )
        # We need to mock ToolProvider to avoid it checking the binary
        with patch(
            "nexus_orchestrator.synthesis_plane.providers.tool_adapter.ToolProvider"
        ) as mock_tp:
            mock_tp.return_value = MagicMock()
            provider = _create_provider_from_auth(auth, model="claude-opus-4-6")
            mock_tp.assert_called_once()


# ---------------------------------------------------------------------------
# TUI controller backend detection with auth
# ---------------------------------------------------------------------------


class TestControllerBackendDetection:
    def test_detect_backends_uses_auth_strategy(self) -> None:
        """Controller._detect_backends_sync uses auth strategy."""
        from nexus_orchestrator.ui.tui.controller import _detect_backends_sync

        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            backends = _detect_backends_sync()

        assert len(backends) >= 4  # claude, codex, anthropic, openai
        for b in backends:
            assert b.auth_mode is not None
            assert b.auth_mode in ("local_cli", "api_key")

    def test_backend_info_has_remediation(self) -> None:
        """BackendInfo from detection includes remediation for unavailable backends."""
        from nexus_orchestrator.ui.tui.controller import _detect_backends_sync

        with (
            patch("nexus_orchestrator.auth.strategy.shutil.which", return_value=None),
            patch("nexus_orchestrator.auth.strategy.importlib.util.find_spec", return_value=None),
            patch.dict(os.environ, {}, clear=True),
        ):
            backends = _detect_backends_sync()

        for b in backends:
            assert not b.available
            assert b.remediation is not None
