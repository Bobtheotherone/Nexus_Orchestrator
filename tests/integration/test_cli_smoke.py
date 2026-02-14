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


# ---------------------------------------------------------------------------
# New CLI commands and UX tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_doctor_exits_zero(tmp_path: Path) -> None:
    """Doctor command must exit 0 and produce diagnostic output."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "doctor")
    assert result.returncode == 0, _render_failure("doctor", result)
    stdout = result.stdout.lower()
    assert "config" in stdout, f"doctor output missing config check:\n{result.stdout}"
    assert "git" in stdout, f"doctor output missing git check:\n{result.stdout}"


@pytest.mark.integration
def test_cli_doctor_json_has_checks(tmp_path: Path) -> None:
    """Doctor --json must produce parseable JSON with checks array."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "doctor", "--json")
    assert result.returncode == 0, _render_failure("doctor --json", result)
    payload = json.loads(result.stdout)
    assert payload["command"] == "doctor"
    assert isinstance(payload["checks"], list)
    assert len(payload["checks"]) > 0
    check_names = {check["name"] for check in payload["checks"]}
    assert "config" in check_names
    assert "git" in check_names


@pytest.mark.integration
def test_cli_config_shows_redacted_config(tmp_path: Path) -> None:
    """Config command must exit 0 and print effective config (redacted)."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "config")
    assert result.returncode == 0, _render_failure("config", result)
    assert "profile" in result.stdout.lower(), f"config output missing profile:\n{result.stdout}"


@pytest.mark.integration
def test_cli_config_json_has_stable_keys(tmp_path: Path) -> None:
    """Config --json must produce parseable JSON with expected keys."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "config", "--json")
    assert result.returncode == 0, _render_failure("config --json", result)
    payload = json.loads(result.stdout)
    assert payload["command"] == "config"
    assert "config" in payload
    assert "active_profile" in payload


@pytest.mark.integration
def test_cli_completion_bash(tmp_path: Path) -> None:
    """Completion command must produce a shell script for bash."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "completion", "bash")
    assert result.returncode == 0, _render_failure("completion bash", result)
    assert "complete" in result.stdout.lower(), "completion bash must produce a completion script"
    assert "nexus" in result.stdout, "completion must reference the nexus command"


@pytest.mark.integration
def test_cli_completion_zsh(tmp_path: Path) -> None:
    """Completion command must produce a shell script for zsh."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "completion", "zsh")
    assert result.returncode == 0, _render_failure("completion zsh", result)
    assert "compdef" in result.stdout, "completion zsh must produce a zsh completion script"


@pytest.mark.integration
def test_cli_plan_json_has_stable_keys(tmp_path: Path) -> None:
    """Plan --json must produce parseable JSON with required keys."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "plan", "samples/specs/minimal_design_doc.md", "--json")
    assert result.returncode == 0, _render_failure("plan --json", result)
    payload = json.loads(result.stdout)
    assert payload["command"] == "plan"
    assert isinstance(payload["work_items"], list)
    assert "warnings" in payload
    assert "errors" in payload
    assert "task_graph" in payload
    assert "spec_path" in payload


@pytest.mark.integration
def test_cli_plan_shows_next_steps(tmp_path: Path) -> None:
    """Plan human output must include Next steps hint."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "plan", "samples/specs/minimal_design_doc.md")
    assert result.returncode == 0, _render_failure("plan", result)
    assert "next steps" in result.stdout.lower(), f"plan missing next steps:\n{result.stdout}"
    assert "nexus run" in result.stdout, f"plan next steps missing run hint:\n{result.stdout}"


@pytest.mark.integration
def test_cli_plan_shows_work_item_table(tmp_path: Path) -> None:
    """Plan human output must include a work-item table."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "plan", "samples/specs/minimal_design_doc.md")
    assert result.returncode == 0, _render_failure("plan", result)
    stdout = result.stdout
    assert "TITLE" in stdout or "title" in stdout.lower(), f"plan missing table header:\n{stdout}"
    assert "RISK" in stdout or "risk" in stdout.lower(), f"plan missing risk column:\n{stdout}"


@pytest.mark.integration
def test_cli_run_shows_next_steps(tmp_path: Path) -> None:
    """Run human output must include Next steps hint."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    assert result.returncode == 0, _render_failure("run --mock", result)
    assert "next steps" in result.stdout.lower(), f"run missing next steps:\n{result.stdout}"
    assert "nexus status" in result.stdout, f"run next steps missing status hint:\n{result.stdout}"


@pytest.mark.integration
def test_cli_status_json_has_routing_and_warnings(tmp_path: Path) -> None:
    """Status --json must include routing_ladder and model_catalog_warnings."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    result = _run_cli(repo_root, "status", "--json")
    assert result.returncode == 0, _render_failure("status --json", result)
    payload = json.loads(result.stdout)
    assert "routing_ladder" in payload, "status JSON must include routing_ladder"
    assert isinstance(payload["routing_ladder"], list)
    assert "model_catalog_warnings" in payload, "status JSON must include model_catalog_warnings"
    if payload["routing_ladder"]:
        first_role = payload["routing_ladder"][0]
        assert "role" in first_role
        assert "steps" in first_role


@pytest.mark.integration
def test_cli_inspect_json_has_routing_and_warnings(tmp_path: Path) -> None:
    """Inspect --json must include routing_ladder and model_catalog_warnings."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    result = _run_cli(repo_root, "inspect", "--json")
    assert result.returncode == 0, _render_failure("inspect --json", result)
    payload = json.loads(result.stdout)
    assert "routing_ladder" in payload, "inspect JSON must include routing_ladder"
    assert "model_catalog_warnings" in payload, "inspect JSON must include model_catalog_warnings"


@pytest.mark.integration
def test_cli_clean_dry_run_is_non_destructive(tmp_path: Path) -> None:
    """Clean with no flags must perform auto-dry-run and not delete anything."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    state_db = repo_root / "state" / "nexus.sqlite"
    assert state_db.exists(), "state DB must exist before clean"
    result = _run_cli(repo_root, "clean")
    assert result.returncode == 0, _render_failure("clean", result)
    assert "dry-run" in result.stdout.lower(), (
        f"clean without flags must be dry-run:\n{result.stdout}"
    )
    assert state_db.exists(), "clean without explicit flags must NOT delete state DB"


@pytest.mark.integration
def test_cli_clean_explicit_dry_run_flag(tmp_path: Path) -> None:
    """Clean --dry-run must not delete anything."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    state_db = repo_root / "state" / "nexus.sqlite"
    assert state_db.exists()
    result = _run_cli(repo_root, "clean", "--dry-run")
    assert result.returncode == 0, _render_failure("clean --dry-run", result)
    assert state_db.exists(), "clean --dry-run must NOT delete state DB"


@pytest.mark.integration
def test_cli_clean_explicit_flags_actually_delete(tmp_path: Path) -> None:
    """Clean --workspaces must actually remove the workspaces directory."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    workspaces = repo_root / "workspaces"
    assert workspaces.exists(), "workspaces must exist after run"
    result = _run_cli(repo_root, "clean", "--workspaces")
    assert result.returncode == 0, _render_failure("clean --workspaces", result)
    assert not workspaces.exists(), "clean --workspaces must delete workspaces directory"
    state_db = repo_root / "state" / "nexus.sqlite"
    assert state_db.exists(), "clean --workspaces must NOT delete state DB (not selected)"


@pytest.mark.integration
def test_cli_clean_json_has_stable_keys(tmp_path: Path) -> None:
    """Clean --json must produce parseable JSON with expected keys."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "clean", "--json")
    assert result.returncode == 0, _render_failure("clean --json", result)
    payload = json.loads(result.stdout)
    assert payload["command"] == "clean"
    assert "dry_run" in payload
    assert "removed" in payload
    assert "missing" in payload
    assert "errors" in payload


@pytest.mark.integration
def test_cli_help_includes_new_commands(tmp_path: Path) -> None:
    """--help must list doctor, config, and completion commands."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "--help")
    assert result.returncode == 0, _render_failure("--help", result)
    stdout = result.stdout.lower()
    assert "doctor" in stdout, "--help missing doctor command"
    assert "config" in stdout, "--help missing config command"
    assert "completion" in stdout, "--help missing completion command"


@pytest.mark.integration
def test_cli_routing_ladder_not_hardcoded(tmp_path: Path) -> None:
    """Routing ladder in status must be derived from config + model catalog, not hard-coded."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    result = _run_cli(repo_root, "status", "--json")
    assert result.returncode == 0, _render_failure("status --json", result)
    payload = json.loads(result.stdout)
    ladder = payload.get("routing_ladder", [])
    assert len(ladder) > 0, "routing ladder must have roles"
    for role_entry in ladder:
        steps = role_entry.get("steps", [])
        for step in steps:
            assert "provider" in step, f"step missing provider: {step}"
            assert "model" in step, f"step missing model: {step}"
            assert step["provider"] and step["model"], f"empty provider/model in step: {step}"


@pytest.mark.integration
def test_cli_tui_missing_deps_exit_code(tmp_path: Path) -> None:
    """nexus tui returns exit code 2 with install hint when textual is missing."""

    repo_root, _ = _prepare_workspace(tmp_path)
    # Force textual to be "not found" by running in a subprocess that hides it
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_pythonpath = str(SRC_PATH)
    env["PYTHONPATH"] = (
        src_pythonpath if not existing_pythonpath else f"{src_pythonpath}:{existing_pythonpath}"
    )
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GIT_CONFIG_NOSYSTEM", "1")
    # Run with a script that patches find_spec to hide textual
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, unittest.mock, sys; "
                "orig = importlib.util.find_spec; "
                "unittest.mock.patch("
                "'importlib.util.find_spec', "
                "side_effect=lambda name, *a, **k: None if name == 'textual' else orig(name, *a, **k)"
                ").start(); "
                "from nexus_orchestrator.ui.tui import run_tui; "
                "sys.exit(run_tui())"
            ),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert result.returncode == 2, _render_failure("tui (no textual)", result)
    assert "install" in result.stderr.lower(), (
        f"tui missing-deps message must mention install:\n{result.stderr}"
    )


@pytest.mark.integration
def test_cli_tui_help(tmp_path: Path) -> None:
    """nexus tui --help must work and show TUI description."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "tui", "--help")
    assert result.returncode == 0, _render_failure("tui --help", result)
    stdout = result.stdout.lower()
    assert "tui" in stdout, "--help missing tui reference"


@pytest.mark.integration
def test_cli_help_includes_tui(tmp_path: Path) -> None:
    """--help must list the tui command."""

    repo_root, _ = _prepare_workspace(tmp_path)
    result = _run_cli(repo_root, "--help")
    assert result.returncode == 0, _render_failure("--help", result)
    assert "tui" in result.stdout.lower(), "--help missing tui command"


@pytest.mark.integration
def test_cli_export_json_has_sha256(tmp_path: Path) -> None:
    """Export --json must include bundle_sha256."""

    repo_root, _ = _prepare_workspace(tmp_path)
    _run_cli(repo_root, "run", "samples/specs/minimal_design_doc.md", "--mock")
    result = _run_cli(repo_root, "export", "--json")
    assert result.returncode == 0, _render_failure("export --json", result)
    payload = json.loads(result.stdout)
    assert payload["command"] == "export"
    assert "bundle_sha256" in payload, "export JSON must include bundle_sha256"
    assert len(payload["bundle_sha256"]) == 64, "bundle_sha256 must be 64-char hex string"
    assert "bundle_path" in payload
    assert "member_names" in payload
