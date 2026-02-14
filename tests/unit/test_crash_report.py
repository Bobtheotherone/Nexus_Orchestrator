"""Unit tests for TUI crash report capture and recovery.

Tests verify crash reports are created, loaded, and cleared correctly.
No real crashes are needed — we create synthetic exceptions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from nexus_orchestrator.ui.crash_report import (
    clear_crash_report,
    load_crash_report,
    save_crash_report,
)


def _make_crash_dir(tmp_path: Path) -> str:
    """Return a mock config dir path under tmp_path."""
    config_dir = tmp_path / "nexus_orchestrator"
    config_dir.mkdir(parents=True, exist_ok=True)
    return str(config_dir)


@pytest.mark.unit
class TestSaveCrashReport:
    """Tests for save_crash_report()."""

    def test_creates_file(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            exc = ValueError("test error")
            path = save_crash_report(exc)
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["exception_type"] == "ValueError"
            assert data["message"] == "test error"
            assert "timestamp" in data
            assert "traceback" in data

    def test_truncates_message(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            long_msg = "x" * 1000
            exc = RuntimeError(long_msg)
            path = save_crash_report(exc)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data["message"]) <= 500

    def test_no_secrets_in_report(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            exc = ValueError("sk-ant-test-secret-key")
            path = save_crash_report(exc)
            content = path.read_text(encoding="utf-8")
            # The exception message itself may contain the string,
            # but we verify no env vars or credential paths are dumped
            assert "OPENAI_API_KEY" not in content
            assert "ANTHROPIC_API_KEY" not in content
            assert "credentials.toml" not in content

    def test_traceback_included(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            try:
                raise TypeError("intentional test error")
            except TypeError as exc:
                path = save_crash_report(exc)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert "TypeError" in data["traceback"]
            assert "intentional test error" in data["traceback"]

    def test_sorted_json_keys(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            exc = ValueError("test")
            path = save_crash_report(exc)
            data = json.loads(path.read_text(encoding="utf-8"))
            keys = list(data.keys())
            assert keys == sorted(keys)


@pytest.mark.unit
class TestLoadCrashReport:
    """Tests for load_crash_report()."""

    def test_returns_data_when_exists(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            exc = ValueError("test")
            save_crash_report(exc)
            data = load_crash_report()
            assert data is not None
            assert data["exception_type"] == "ValueError"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            data = load_crash_report()
            assert data is None

    def test_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        config_dir.mkdir(parents=True, exist_ok=True)
        crash_file = config_dir / "last_tui_crash.json"
        crash_file.write_text("not valid json", encoding="utf-8")
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            data = load_crash_report()
            assert data is None


@pytest.mark.unit
class TestClearCrashReport:
    """Tests for clear_crash_report()."""

    def test_deletes_file(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            exc = ValueError("test")
            path = save_crash_report(exc)
            assert path.exists()
            clear_crash_report()
            assert not path.exists()

    def test_noop_when_missing(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "nexus_orchestrator"
        config_dir.mkdir(parents=True, exist_ok=True)
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            # Should not raise
            clear_crash_report()

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save, load, clear, load — verify full lifecycle."""
        config_dir = tmp_path / "nexus_orchestrator"
        with mock.patch(
            "nexus_orchestrator.ui.crash_report._config_dir",
            return_value=config_dir,
        ):
            exc = RuntimeError("lifecycle test")
            save_crash_report(exc)
            data = load_crash_report()
            assert data is not None
            assert data["message"] == "lifecycle test"
            clear_crash_report()
            assert load_crash_report() is None
