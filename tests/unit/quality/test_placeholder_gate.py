"""
nexus-orchestrator — unit tests for placeholder gate wrapper

File: tests/unit/quality/test_placeholder_gate.py
Last updated: 2026-02-13

Purpose
- Verify deterministic wrapper exit codes and policy behavior for `scripts/placeholder_gate.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "placeholder_gate.py"


def _write(root: Path, rel_path: str, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_gate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("placeholder_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_returns_zero_for_warning_only_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "tests/unit/test_warning.py", "# TODO: test fixture follow-up\n")
    monkeypatch.chdir(tmp_path)
    gate_module = _load_gate_module()

    exit_code = gate_module.main(["--paths", "tests", "--exclude", "tests/meta"])

    assert exit_code == 0


def test_gate_blocks_test_warnings_with_tests_blocking_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, "tests/unit/test_warning.py", "# TODO: test fixture follow-up\n")
    monkeypatch.chdir(tmp_path)
    gate_module = _load_gate_module()

    exit_code = gate_module.main(
        [
            "--paths",
            "tests",
            "--exclude",
            "tests/meta",
            "--scan-tests-as-blocking",
        ]
    )

    assert exit_code == 1


def test_gate_returns_one_when_blocking_src_findings_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, "src/blocking.py", "def run() -> None:\n    raise NotImplementedError\n")
    monkeypatch.chdir(tmp_path)
    gate_module = _load_gate_module()

    exit_code = gate_module.main(["--paths", "src", "--format", "text"])
    output = capsys.readouterr().out

    assert exit_code == 1
    assert "raise_not_implemented_error" in output


def test_gate_returns_gt_one_when_tool_runtime_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate_module = _load_gate_module()

    def _crash(**_: object) -> object:
        raise RuntimeError("forced crash boundary")

    class _BrokenModule:
        DEFAULT_SELF_REFERENCE_ALLOWLIST = ()

        @staticmethod
        def run_placeholder_audit(**kwargs: object) -> object:
            return _crash(**kwargs)

        @staticmethod
        def format_json(_: object) -> str:
            return "{}\n"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gate_module, "_load_module", lambda: _BrokenModule())

    exit_code = gate_module.main(["--paths", "src"])
    stderr = capsys.readouterr().err

    assert exit_code > 1
    assert "placeholder-gate crashed: forced crash boundary" in stderr
