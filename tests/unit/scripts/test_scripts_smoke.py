"""
nexus-orchestrator — script subprocess smoke tests

File: tests/unit/scripts/test_scripts_smoke.py
Last updated: 2026-02-14

Purpose
- Keep script entrypoints executable and deterministic at a smoke-test level.
- Verify `--help`, `--json` output structure, and `--dry-run` non-destructive safety behavior.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PATH = REPO_ROOT / "src"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_pythonpath = str(SRC_PATH)
    env["PYTHONPATH"] = (
        src_pythonpath if not existing_pythonpath else f"{src_pythonpath}:{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _render_failure(label: str, completed: subprocess.CompletedProcess[str]) -> str:
    return (
        f"{label} failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}\n"
    )


@pytest.mark.unit
def test_placeholder_gate_help_smoke() -> None:
    result = _run_script("scripts/placeholder_gate.py", "--help")

    assert result.returncode == 0, _render_failure("placeholder_gate --help", result)
    lowered_output = result.stdout.lower()
    assert "usage" in lowered_output
    assert "--format" in lowered_output
    assert "--scan-tests-as-blocking" in lowered_output


@pytest.mark.unit
def test_repo_audit_json_smoke_has_expected_top_level_keys() -> None:
    result = _run_script("scripts/repo_audit.py", "--json", "--repo-root", str(REPO_ROOT))

    assert result.returncode == 0, _render_failure("repo_audit --json", result)
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert {"metadata", "tracked_files", "headers", "placeholder_scan"} <= set(payload.keys())

    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("source_strategy") in {"git_ls_files", "filesystem_fallback"}


@pytest.mark.unit
def test_gc_workspaces_dry_run_is_non_destructive(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    candidate = workspace_root / "dry-run-item" / "attempt-1"
    candidate.mkdir(parents=True, exist_ok=True)
    artifact = candidate / "artifact.txt"
    artifact.write_text("keep me\n", encoding="utf-8")

    result = _run_script(
        "scripts/gc_workspaces.py",
        "--workspace-root",
        str(workspace_root),
        "--dry-run",
    )

    assert result.returncode == 0, _render_failure("gc_workspaces --dry-run", result)
    assert candidate.exists()
    assert artifact.exists()
