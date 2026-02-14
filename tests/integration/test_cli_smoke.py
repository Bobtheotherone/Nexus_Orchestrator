"""
nexus-orchestrator — CLI subprocess smoke contracts

File: tests/integration/test_cli_smoke.py
Last updated: 2026-02-14

Purpose
- Enforce hard-to-cheat CLI behavior for `python -m nexus_orchestrator` plan/run/status.
- Verify exit codes, command output signals, and persistent run/evidence side effects.
- Ensure model routing under mock runs follows config + model catalog resolution (no hard-coded model IDs).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from nexus_orchestrator.synthesis_plane.model_catalog import load_model_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"


def _run_cli(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_pythonpath = str(SRC_PATH)
    env["PYTHONPATH"] = (
        src_pythonpath if not existing_pythonpath else f"{src_pythonpath}:{existing_pythonpath}"
    )
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    return subprocess.run(
        [sys.executable, "-m", "nexus_orchestrator", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def _git(repo_root: Path, *args: str) -> None:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        command = "git " + " ".join(args)
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git command failed: {command}: {detail}")


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _seed_repo(repo_root: Path) -> None:
    _write(
        repo_root / "src" / "a.py",
        'def greet(name: str) -> str:\n    return "TODO"\n',
    )
    _write(
        repo_root / "src" / "b.py",
        'def farewell(name: str) -> str:\n    return "TODO"\n',
    )
    _write(
        repo_root / "src" / "c.py",
        'def conversation(name: str) -> str:\n    return "TODO"\n',
    )
    _write(
        repo_root / "tests" / "unit" / "test_a.py",
        "def test_greet() -> None:\n    assert True\n",
    )
    _write(
        repo_root / "tests" / "unit" / "test_b.py",
        "def test_farewell() -> None:\n    assert True\n",
    )
    _write(
        repo_root / "tests" / "unit" / "test_c.py",
        "def test_conversation() -> None:\n    assert True\n",
    )
    sample_spec_source = PROJECT_ROOT / "samples" / "specs" / "minimal_design_doc.md"
    _write(
        repo_root / "samples" / "specs" / "minimal_design_doc.md",
        sample_spec_source.read_text(encoding="utf-8"),
    )


def _resolve_openai_code_model_override() -> tuple[str, str]:
    catalog = load_model_catalog()
    default_code_model = catalog.default_model_for_profile(
        provider="openai",
        capability_profile="code",
    )
    openai_entries = sorted(
        (entry for entry in catalog.models if entry.provider == "openai"),
        key=lambda entry: entry.model,
    )
    for entry in openai_entries:
        if entry.model == default_code_model:
            continue
        if entry.aliases:
            configured_token = entry.aliases[0]
            resolved_model = catalog.resolve_model_for_profile(
                provider="openai",
                capability_profile="code",
                configured_model=configured_token,
            )
            return configured_token, resolved_model
    for entry in openai_entries:
        if entry.model == default_code_model:
            continue
        configured_token = entry.model
        resolved_model = catalog.resolve_model_for_profile(
            provider="openai",
            capability_profile="code",
            configured_model=configured_token,
        )
        return configured_token, resolved_model

    fallback = catalog.resolve_model_for_profile(
        provider="openai",
        capability_profile="code",
        configured_model=default_code_model,
    )
    return default_code_model, fallback


def _write_local_config(repo_root: Path, *, openai_code_model: str) -> None:
    catalog = load_model_catalog()
    openai_architect_model = catalog.default_model_for_profile(
        provider="openai",
        capability_profile="architect",
    )
    constraint_registry = (PROJECT_ROOT / "constraints" / "registry").resolve().as_posix()
    config_text = (
        "[meta]\n"
        "schema_version = 1\n"
        "\n"
        "[providers]\n"
        'default = "openai"\n'
        "\n"
        "[providers.openai]\n"
        'api_key_env = "NEXUS_OPENAI_API_KEY"\n'
        f'model_code = "{openai_code_model}"\n'
        f'model_architect = "{openai_architect_model}"\n'
        "\n"
        "[providers.anthropic]\n"
        'api_key_env = "NEXUS_ANTHROPIC_API_KEY"\n'
        "\n"
        "[providers.local]\n"
        "\n"
        "[paths]\n"
        'workspace_root = "workspaces/"\n'
        'evidence_root = "evidence/"\n'
        'state_db = "state/nexus.sqlite"\n'
        f'constraint_registry = "{constraint_registry}"\n'
    )
    _write(repo_root / "orchestrator.toml", config_text)


def _prepare_workspace(tmp_path: Path) -> tuple[Path, str]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _seed_repo(repo_root)
    configured_model_token, resolved_model = _resolve_openai_code_model_override()
    _write_local_config(repo_root, openai_code_model=configured_model_token)

    _git(repo_root, "init", "--initial-branch=main", "--quiet")
    _git(repo_root, "config", "user.name", "CLI Smoke")
    _git(repo_root, "config", "user.email", "cli-smoke@example.com")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "--no-gpg-sign", "--quiet", "-m", "seed workspace")
    _git(repo_root, "branch", "-f", "integration", "main")
    return repo_root, resolved_model


def _render_failure(command_name: str, completed: subprocess.CompletedProcess[str]) -> str:
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    return (
        f"{command_name} failed with exit code {completed.returncode}\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}\n"
    )


def _assert_mock_run_side_effects(
    *,
    repo_root: Path,
    expected_model: str,
) -> str:
    state_db = repo_root / "state" / "nexus.sqlite"
    assert state_db.exists(), "mock run must create state DB at state/nexus.sqlite"

    with sqlite3.connect(state_db) as conn:
        run_row = conn.execute(
            """
            SELECT id, status, metadata_json
            FROM runs
            ORDER BY started_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
        assert run_row is not None, "mock run must persist at least one row in runs"
        run_id = str(run_row[0])
        assert str(run_row[1]).lower() == "completed", run_row
        metadata = json.loads(str(run_row[2]))
        assert metadata.get("mock_mode") is True

        attempt_model_rows = conn.execute(
            "SELECT DISTINCT model FROM attempts WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        persisted_models = {
            str(model)
            for (model,) in attempt_model_rows
            if isinstance(model, str) and model.strip()
        }
        assert persisted_models == {expected_model}, persisted_models

        evidence_rows = conn.execute(
            "SELECT payload_json FROM evidence WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        assert evidence_rows, "mock run must persist evidence rows"

    for (payload_json,) in evidence_rows:
        payload = json.loads(str(payload_json))
        artifact_paths = payload.get("artifact_paths")
        assert isinstance(artifact_paths, list) and artifact_paths, payload
        for artifact_path in artifact_paths:
            assert isinstance(artifact_path, str) and artifact_path
            assert (repo_root / artifact_path).exists(), artifact_path

    return run_id


@pytest.mark.integration
def test_cli_plan_run_status_subprocess_contract(tmp_path: Path) -> None:
    repo_root, expected_model = _prepare_workspace(tmp_path)
    spec_path = "samples/specs/minimal_design_doc.md"
    failures: list[str] = []

    plan_result = _run_cli(repo_root, "plan", spec_path)
    if plan_result.returncode != 0:
        failures.append(_render_failure("plan", plan_result))
    else:
        if not plan_result.stdout.strip():
            failures.append("plan produced empty stdout")
        plan_stdout = plan_result.stdout.lower()
        if "plan" not in plan_stdout and "work" not in plan_stdout:
            failures.append(f"plan output missing plan/work signal:\n{plan_result.stdout}")

    run_result = _run_cli(repo_root, "run", spec_path, "--mock")
    if run_result.returncode != 0:
        failures.append(_render_failure("run", run_result))
    else:
        if not run_result.stdout.strip():
            failures.append("run produced empty stdout")
        run_stdout = run_result.stdout.lower()
        if "run" not in run_stdout and "completed" not in run_stdout:
            failures.append(f"run output missing run/completed signal:\n{run_result.stdout}")

        run_id = _assert_mock_run_side_effects(repo_root=repo_root, expected_model=expected_model)

        status_result = _run_cli(repo_root, "status")
        if status_result.returncode != 0:
            failures.append(_render_failure("status", status_result))
        else:
            if not status_result.stdout.strip():
                failures.append("status produced empty stdout")
            if run_id not in status_result.stdout:
                failures.append(f"status output missing run id {run_id}:\n{status_result.stdout}")
            if "completed" not in status_result.stdout.lower():
                failures.append(f"status output missing completed state:\n{status_result.stdout}")

    assert not failures, "\n\n".join(failures)
