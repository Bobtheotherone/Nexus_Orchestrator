"""Unit tests for tool backend detection.

Tests verify offline CLI tool detection (codex, claude) using mocked
shutil.which and subprocess calls. No real binaries are required.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from nexus_orchestrator.synthesis_plane.providers.tool_detection import (
    ToolBackendInfo,
    detect_all_backends,
    detect_claude_code_cli,
    detect_codex_cli,
    detect_tool_backend,
)


class TestDetectToolBackend:
    """Tests for detect_tool_backend()."""

    def test_codex_found_with_version(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/local/bin/codex"),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["codex", "--version"],
                    returncode=0,
                    stdout="codex 0.1.2\n",
                    stderr="",
                ),
            ),
        ):
            info = detect_tool_backend("codex")
            assert info is not None
            assert info.name == "codex"
            assert info.binary_path == "/usr/local/bin/codex"
            assert info.version == "codex 0.1.2"

    def test_claude_found_with_version(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["claude", "--version"],
                    returncode=0,
                    stdout="claude 1.0.5\n",
                    stderr="",
                ),
            ),
        ):
            info = detect_tool_backend("claude")
            assert info is not None
            assert info.name == "claude"
            assert info.binary_path == "/usr/bin/claude"
            assert info.version == "claude 1.0.5"

    def test_not_found_returns_none(self) -> None:
        with patch("shutil.which", return_value=None):
            info = detect_tool_backend("codex")
            assert info is None

    def test_version_timeout_returns_none_version(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/codex"),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="codex", timeout=5),
            ),
        ):
            info = detect_tool_backend("codex")
            assert info is not None
            assert info.binary_path == "/usr/bin/codex"
            assert info.version is None

    def test_version_nonzero_exit_returns_none_version(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/codex"),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["codex", "--version"],
                    returncode=1,
                    stdout="",
                    stderr="error",
                ),
            ),
        ):
            info = detect_tool_backend("codex")
            assert info is not None
            assert info.version is None

    def test_version_file_not_found(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/codex"),
            patch("subprocess.run", side_effect=FileNotFoundError),
        ):
            info = detect_tool_backend("codex")
            assert info is not None
            assert info.version is None

    def test_multiline_version_takes_first_line(self) -> None:
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=["claude", "--version"],
                    returncode=0,
                    stdout="claude 1.0.5\nsome extra info\n",
                    stderr="",
                ),
            ),
        ):
            info = detect_tool_backend("claude")
            assert info is not None
            assert info.version == "claude 1.0.5"


class TestDetectConvenienceFunctions:
    """Tests for detect_codex_cli() and detect_claude_code_cli()."""

    def test_detect_codex_cli_delegates(self) -> None:
        with patch(
            "nexus_orchestrator.synthesis_plane.providers.tool_detection.detect_tool_backend"
        ) as mock:
            mock.return_value = ToolBackendInfo(
                name="codex", binary_path="/bin/codex", version="1.0"
            )
            result = detect_codex_cli()
            mock.assert_called_once_with("codex")
            assert result is not None
            assert result.name == "codex"

    def test_detect_claude_code_cli_delegates(self) -> None:
        with patch(
            "nexus_orchestrator.synthesis_plane.providers.tool_detection.detect_tool_backend"
        ) as mock:
            mock.return_value = ToolBackendInfo(
                name="claude", binary_path="/bin/claude", version="2.0"
            )
            result = detect_claude_code_cli()
            mock.assert_called_once_with("claude")
            assert result is not None
            assert result.name == "claude"


class TestDetectAllBackends:
    """Tests for detect_all_backends()."""

    def test_both_found(self) -> None:
        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in {"codex", "claude"} else None

        with (
            patch("shutil.which", side_effect=fake_which),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="v1.0\n", stderr=""
                ),
            ),
        ):
            result = detect_all_backends()
            assert len(result) == 2
            assert result[0].name == "codex"
            assert result[1].name == "claude"

    def test_neither_found(self) -> None:
        with patch("shutil.which", return_value=None):
            result = detect_all_backends()
            assert result == []

    def test_only_claude_found(self) -> None:
        def fake_which(name: str) -> str | None:
            return "/usr/bin/claude" if name == "claude" else None

        with (
            patch("shutil.which", side_effect=fake_which),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="v1.0\n", stderr=""
                ),
            ),
        ):
            result = detect_all_backends()
            assert len(result) == 1
            assert result[0].name == "claude"

    def test_deterministic_order(self) -> None:
        """Backends are always returned in codex, claude order."""

        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name in {"codex", "claude"} else None

        with (
            patch("shutil.which", side_effect=fake_which),
            patch(
                "subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="v1.0\n", stderr=""
                ),
            ),
        ):
            result = detect_all_backends()
            names = [info.name for info in result]
            assert names == ["codex", "claude"]


class TestToolBackendInfoFrozen:
    """Verify ToolBackendInfo is immutable."""

    def test_frozen(self) -> None:
        info = ToolBackendInfo(name="codex", binary_path="/bin/codex", version="1.0")
        with pytest.raises(AttributeError):
            info.name = "changed"  # type: ignore[misc]
