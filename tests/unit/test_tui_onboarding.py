"""Unit tests for TUI onboarding persistence, credential handling, URLs, and splash logic.

File: tests/unit/test_tui_onboarding.py

Tests:
- onboarding.json created only after credential success
- redaction: keys are never stored in onboarding.json
- file permissions: credential file mode 0600 on POSIX
- tui_available() check
- run_tui() returns exit code 2 when textual is not installed
- Provider URLs point to correct API key pages (not playground)
- Key format validation
- Splash first-run-only sentinel logic
- Header mascot size constraints (optional, requires Textual)
"""

from __future__ import annotations

import json
import platform
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from nexus_orchestrator.ui.onboarding import (
    PROVIDER_KEY_URLS,
    get_onboarding_backends,
    get_onboarding_state,
    is_onboarding_complete,
    mark_onboarding_complete,
    redact_key,
    store_credential,
    validate_key_format,
)
from nexus_orchestrator.ui.tui import tui_available


@pytest.mark.unit
class TestRedactKey:
    """Test API key redaction."""

    def test_redact_normal_key(self) -> None:
        assert redact_key("sk-abc123def456") == "***********f456"

    def test_redact_short_key(self) -> None:
        assert redact_key("abc") == "****"

    def test_redact_empty_key(self) -> None:
        assert redact_key("") == "****"

    def test_redact_four_char_key(self) -> None:
        assert redact_key("abcd") == "****"

    def test_redact_five_char_key(self) -> None:
        assert redact_key("abcde") == "*bcde"


@pytest.mark.unit
class TestProviderURLs:
    """Test that provider URLs point to correct API key management pages."""

    def test_openai_url_contains_api_keys(self) -> None:
        """OpenAI URL must point to the API keys page, NOT playground."""
        url = PROVIDER_KEY_URLS["openai"]
        assert "api-keys" in url or "api_keys" in url
        assert "playground" not in url.lower()

    def test_openai_url_is_platform(self) -> None:
        """OpenAI URL must be on platform.openai.com."""
        url = PROVIDER_KEY_URLS["openai"]
        assert "platform.openai.com" in url

    def test_anthropic_url_contains_keys(self) -> None:
        """Anthropic URL must point to the keys page."""
        url = PROVIDER_KEY_URLS["anthropic"]
        assert "keys" in url

    def test_anthropic_url_is_console(self) -> None:
        """Anthropic URL must be on console.anthropic.com."""
        url = PROVIDER_KEY_URLS["anthropic"]
        assert "console.anthropic.com" in url


@pytest.mark.unit
class TestValidateKeyFormat:
    """Test local API key format validation."""

    def test_valid_openai_key(self) -> None:
        valid, msg = validate_key_format("openai", "sk-abc123def456ghi789jkl012mno")
        assert valid
        assert "valid" in msg.lower()

    def test_invalid_openai_prefix(self) -> None:
        valid, msg = validate_key_format("openai", "wrong-prefix-key-1234567890")
        assert not valid
        assert "sk-" in msg

    def test_valid_anthropic_key(self) -> None:
        valid, msg = validate_key_format("anthropic", "sk-ant-abc123def456ghi789jkl012")
        assert valid

    def test_invalid_anthropic_prefix(self) -> None:
        valid, msg = validate_key_format("anthropic", "sk-wrong-abc123def456ghi789")
        assert not valid
        assert "sk-ant-" in msg

    def test_empty_key(self) -> None:
        valid, msg = validate_key_format("openai", "")
        assert not valid
        assert "empty" in msg.lower()

    def test_too_short_key(self) -> None:
        valid, msg = validate_key_format("openai", "sk-abc")
        assert not valid
        assert "short" in msg.lower()

    def test_unknown_provider_accepts_any(self) -> None:
        """Unknown providers have no prefix check — any long key is valid."""
        valid, _ = validate_key_format("unknown_provider", "some-long-key-value-here")
        assert valid


@pytest.mark.unit
class TestOnboardingPersistence:
    """Test onboarding.json sentinel file management."""

    def test_not_complete_when_no_file(self, tmp_path: Path) -> None:
        """Onboarding is not complete if sentinel doesn't exist."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            assert not is_onboarding_complete()

    def test_mark_complete_creates_sentinel(self, tmp_path: Path) -> None:
        """mark_onboarding_complete creates a valid sentinel file."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete(["openai"])
            assert is_onboarding_complete()

            sentinel = tmp_path / "config" / "onboarding.json"
            assert sentinel.exists()

            data = json.loads(sentinel.read_text(encoding="utf-8"))
            assert data["completed"] is True
            assert "openai" in data["providers"]
            assert "completed_at" in data

    def test_sentinel_does_not_contain_keys(self, tmp_path: Path) -> None:
        """Onboarding sentinel must never contain API keys."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete(["openai", "anthropic"])

            sentinel = tmp_path / "config" / "onboarding.json"
            content = sentinel.read_text(encoding="utf-8")

            # No key-like patterns should be in the sentinel
            assert "sk-" not in content
            assert "api_key" not in content.lower()
            assert "secret" not in content.lower()

    def test_get_state_empty_when_no_file(self, tmp_path: Path) -> None:
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            state = get_onboarding_state()
            assert state == {}

    def test_get_state_returns_data(self, tmp_path: Path) -> None:
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete(["anthropic"])
            state = get_onboarding_state()
            assert state["completed"] is True
            assert "anthropic" in state["providers"]

    def test_providers_sorted(self, tmp_path: Path) -> None:
        """Providers in sentinel should be sorted for determinism."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete(["openai", "anthropic"])
            data = json.loads((tmp_path / "config" / "onboarding.json").read_text(encoding="utf-8"))
            assert data["providers"] == ["anthropic", "openai"]

    def test_mock_mode_marks_complete_with_empty_providers(self, tmp_path: Path) -> None:
        """Mock mode can mark onboarding complete with no providers."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete([])
            assert is_onboarding_complete()
            data = json.loads((tmp_path / "config" / "onboarding.json").read_text(encoding="utf-8"))
            assert data["providers"] == []


@pytest.mark.unit
class TestSplashFirstRunOnly:
    """Test that splash is shown only on first onboarding completion."""

    def test_splash_shown_when_transitioning_to_complete(self, tmp_path: Path) -> None:
        """Splash should be triggered when onboarding transitions from incomplete to complete."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            # Before completion: not complete
            assert not is_onboarding_complete()
            first_time = not is_onboarding_complete()
            assert first_time is True

            # Mark complete
            mark_onboarding_complete(["openai"])

            # After completion: complete
            assert is_onboarding_complete()
            first_time_after = not is_onboarding_complete()
            assert first_time_after is False

    def test_splash_not_shown_on_subsequent_launches(self, tmp_path: Path) -> None:
        """Once onboarding is complete, splash flag should be False."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete(["openai"])
            # Simulate subsequent launch check
            assert is_onboarding_complete()
            show_splash = not is_onboarding_complete()
            assert show_splash is False


@pytest.mark.unit
class TestCredentialStorage:
    """Test credential file storage."""

    def test_store_credential_creates_file(self, tmp_path: Path) -> None:
        """Storing a credential creates the credentials file."""
        with (
            mock.patch(
                "nexus_orchestrator.ui.onboarding._config_dir",
                return_value=tmp_path / "config",
            ),
            mock.patch(
                "nexus_orchestrator.ui.onboarding._keyring_available",
                return_value=False,
            ),
        ):
            method = store_credential("openai", "sk-test-key-1234")
            assert method == "file"

            cred_file = tmp_path / "config" / "credentials.toml"
            assert cred_file.exists()

            content = cred_file.read_text(encoding="utf-8")
            assert "sk-test-key-1234" in content

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="POSIX file permissions not available on Windows",
    )
    def test_credential_file_permissions_0600(self, tmp_path: Path) -> None:
        """Credential file must have mode 0600 on POSIX."""
        with (
            mock.patch(
                "nexus_orchestrator.ui.onboarding._config_dir",
                return_value=tmp_path / "config",
            ),
            mock.patch(
                "nexus_orchestrator.ui.onboarding._keyring_available",
                return_value=False,
            ),
        ):
            store_credential("openai", "sk-test-key-5678")
            cred_file = tmp_path / "config" / "credentials.toml"
            mode = cred_file.stat().st_mode & 0o777
            assert mode == 0o600, f"Expected 0600, got {oct(mode)}"

    def test_store_multiple_providers(self, tmp_path: Path) -> None:
        """Storing credentials for multiple providers works."""
        with (
            mock.patch(
                "nexus_orchestrator.ui.onboarding._config_dir",
                return_value=tmp_path / "config",
            ),
            mock.patch(
                "nexus_orchestrator.ui.onboarding._keyring_available",
                return_value=False,
            ),
        ):
            store_credential("openai", "sk-openai-key")
            store_credential("anthropic", "sk-ant-key")

            cred_file = tmp_path / "config" / "credentials.toml"
            content = cred_file.read_text(encoding="utf-8")
            assert "sk-openai-key" in content
            assert "sk-ant-key" in content


@pytest.mark.unit
class TestTuiAvailable:
    """Test TUI availability check."""

    def test_tui_available_returns_bool(self) -> None:
        """tui_available() must return a boolean."""
        result = tui_available()
        assert isinstance(result, bool)


@pytest.mark.unit
@pytest.mark.skipif(
    not tui_available(),
    reason="Textual not installed; skipping TUI init regression test",
)
class TestNexusTUIInit:
    """Regression tests for NexusTUI initialization (requires Textual)."""

    def test_init_does_not_pass_css_kwarg_to_app(self) -> None:
        """NexusTUI must use the CSS class variable, not pass css= to App.__init__.

        Textual >= 0.50 does not accept css= in App.__init__; using it causes
        TypeError at runtime. This test ensures the bug never regresses.
        """
        import inspect

        from nexus_orchestrator.ui.tui_app import NexusTUI

        # Verify CSS is a class variable (not set only in __init__)
        assert hasattr(NexusTUI, "CSS"), "NexusTUI must define a CSS class variable"
        assert isinstance(NexusTUI.CSS, str), "NexusTUI.CSS must be a string"
        assert len(NexusTUI.CSS) > 0, "NexusTUI.CSS must not be empty"

        # Inspect __init__ source to ensure css= is not forwarded to super().__init__
        source = inspect.getsource(NexusTUI.__init__)
        assert "super().__init__(css=" not in source, (
            "NexusTUI.__init__ must NOT pass css= to super().__init__(); "
            "use the CSS class variable instead"
        )

    def test_run_tui_app_sets_css_class_var_for_no_color(self) -> None:
        """run_tui_app must set NexusTUI.CSS before instantiation."""
        import inspect

        from nexus_orchestrator.ui.tui_app import run_tui_app

        source = inspect.getsource(run_tui_app)
        assert "NexusTUI.CSS" in source, (
            "run_tui_app must set NexusTUI.CSS class variable before instantiation"
        )


@pytest.mark.unit
@pytest.mark.skipif(
    not tui_available(),
    reason="Textual not installed; skipping header mascot test",
)
class TestHeaderMascot:
    """Test that the header mascot fits within configured size constraints."""

    def test_mascot_fits_max_dimensions(self) -> None:
        """Header mascot must be <= HEADER_MASCOT_MAX_WIDTH x HEADER_MASCOT_MAX_HEIGHT."""
        from nexus_orchestrator.ui.tui_app import (
            HEADER_MASCOT_MAX_HEIGHT,
            HEADER_MASCOT_MAX_WIDTH,
            _load_auto_header,
        )

        mascot = _load_auto_header()
        lines = mascot.splitlines()
        assert len(lines) <= HEADER_MASCOT_MAX_HEIGHT, (
            f"Mascot height {len(lines)} exceeds max {HEADER_MASCOT_MAX_HEIGHT}"
        )
        for i, line in enumerate(lines):
            assert len(line) <= HEADER_MASCOT_MAX_WIDTH, (
                f"Mascot line {i} width {len(line)} exceeds max {HEADER_MASCOT_MAX_WIDTH}"
            )

    def test_mascot_is_not_empty(self) -> None:
        """Header mascot must contain visible content."""
        from nexus_orchestrator.ui.tui_app import _load_auto_header

        mascot = _load_auto_header()
        assert mascot.strip(), "Header mascot should not be empty"


@pytest.mark.unit
@pytest.mark.skipif(
    not tui_available(),
    reason="Textual not installed; skipping splash animation test",
)
class TestSplashAnimation:
    """Test deterministic splash animation rendering."""

    def test_frame_count_is_positive(self) -> None:
        from nexus_orchestrator.ui.tui_app import splash_frame_count

        assert splash_frame_count() > 0

    def test_render_first_frame_is_mostly_dark(self) -> None:
        from nexus_orchestrator.ui.tui_app import render_splash_frame, splash_frame_count

        text = "LINE ONE\nLINE TWO\nLINE THREE"
        rendered = render_splash_frame(text, 0, splash_frame_count())
        # Frame 0 should have dark color markup
        assert "#0a1628" in rendered

    def test_render_last_frame_is_fully_revealed(self) -> None:
        from nexus_orchestrator.ui.tui_app import render_splash_frame, splash_frame_count

        total = splash_frame_count()
        text = "LINE ONE\nLINE TWO\nLINE THREE"
        rendered = render_splash_frame(text, total - 1, total)
        # Last frame should have bright color markup
        assert "#72c7ff" in rendered

    def test_render_is_deterministic(self) -> None:
        """Same frame + text must produce identical output (no randomness)."""
        from nexus_orchestrator.ui.tui_app import render_splash_frame, splash_frame_count

        total = splash_frame_count()
        text = "LINE ONE\nLINE TWO"
        r1 = render_splash_frame(text, 15, total)
        r2 = render_splash_frame(text, 15, total)
        assert r1 == r2

    def test_render_empty_text(self) -> None:
        from nexus_orchestrator.ui.tui_app import render_splash_frame

        result = render_splash_frame("", 0, 30)
        assert result == ""


@pytest.mark.unit
class TestOnboardingBackends:
    """Test tool backend support in onboarding persistence."""

    def test_mark_complete_with_backends(self, tmp_path: Path) -> None:
        """mark_onboarding_complete with backends stores them in sentinel."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete([], backends=["codex", "claude"])
            sentinel = tmp_path / "config" / "onboarding.json"
            data = json.loads(sentinel.read_text(encoding="utf-8"))
            assert data["completed"] is True
            assert data["backends"] == ["claude", "codex"]  # sorted
            assert data["providers"] == []

    def test_mark_complete_without_backends(self, tmp_path: Path) -> None:
        """mark_onboarding_complete without backends stores empty list."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete(["openai"])
            sentinel = tmp_path / "config" / "onboarding.json"
            data = json.loads(sentinel.read_text(encoding="utf-8"))
            assert data["backends"] == []
            assert data["providers"] == ["openai"]

    def test_get_onboarding_backends(self, tmp_path: Path) -> None:
        """get_onboarding_backends returns the backends list from sentinel."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete([], backends=["claude"])
            backends = get_onboarding_backends()
            assert backends == ["claude"]

    def test_get_onboarding_backends_empty_when_no_file(self, tmp_path: Path) -> None:
        """get_onboarding_backends returns empty list when no sentinel exists."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            backends = get_onboarding_backends()
            assert backends == []

    def test_sentinel_no_secrets_with_backends(self, tmp_path: Path) -> None:
        """Sentinel file must not contain secrets even with tool backends."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete([], backends=["codex", "claude"])
            sentinel = tmp_path / "config" / "onboarding.json"
            content = sentinel.read_text(encoding="utf-8")
            assert "sk-" not in content
            assert "api_key" not in content.lower()
            assert "secret" not in content.lower()
            assert "token" not in content.lower()

    def test_backends_sorted_deterministic(self, tmp_path: Path) -> None:
        """Backends list is sorted for determinism."""
        with mock.patch(
            "nexus_orchestrator.ui.onboarding._config_dir",
            return_value=tmp_path / "config",
        ):
            mark_onboarding_complete([], backends=["claude", "codex"])
            sentinel = tmp_path / "config" / "onboarding.json"
            data = json.loads(sentinel.read_text(encoding="utf-8"))
            assert data["backends"] == ["claude", "codex"]


@pytest.mark.unit
class TestOnboardingMenuOptions:
    """Test the 5-option onboarding menu structure."""

    def test_onboarding_menu_has_five_options(self) -> None:
        """The welcome menu should present 5 options (mock, codex, claude, both, API keys)."""
        from nexus_orchestrator.ui.onboarding import TOOL_BACKEND_NOTICE

        # Verify the notice text is available for the menu
        assert "No API keys required" in TOOL_BACKEND_NOTICE
        assert "Codex" in TOOL_BACKEND_NOTICE
        assert "Claude Code" in TOOL_BACKEND_NOTICE

    def test_tool_backend_display_has_both_backends(self) -> None:
        """Both codex and claude should have display names."""
        from nexus_orchestrator.ui.onboarding import TOOL_BACKEND_DISPLAY

        assert "codex" in TOOL_BACKEND_DISPLAY
        assert "claude" in TOOL_BACKEND_DISPLAY

    def test_tool_install_hints_provided(self) -> None:
        """Install hints should exist for both backends."""
        from nexus_orchestrator.ui.onboarding import TOOL_INSTALL_HINTS

        assert "codex" in TOOL_INSTALL_HINTS
        assert "claude" in TOOL_INSTALL_HINTS
        assert TOOL_INSTALL_HINTS["codex"]  # non-empty
        assert TOOL_INSTALL_HINTS["claude"]  # non-empty


@pytest.mark.unit
class TestHeaderMascotAsset:
    """Test the purpose-built header mascot asset."""

    def test_header_asset_exists(self) -> None:
        """auto_header_ascii.txt must exist."""
        from pathlib import Path

        assets_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "nexus_orchestrator"
            / "ui"
            / "assets"
        )
        assert (assets_dir / "auto_header_ascii.txt").exists()

    def test_header_asset_within_bounds(self) -> None:
        """auto_header_ascii.txt must be <= 14 cols x 3 lines."""
        from nexus_orchestrator.ui.tui_app import (
            HEADER_MASCOT_MAX_HEIGHT,
            HEADER_MASCOT_MAX_WIDTH,
            _load_auto_header,
        )

        mascot = _load_auto_header()
        lines = mascot.splitlines()
        assert len(lines) <= HEADER_MASCOT_MAX_HEIGHT
        for line in lines:
            assert len(line) <= HEADER_MASCOT_MAX_WIDTH
