"""
nexus-orchestrator — end-to-end smoke integration test

File: tests/smoke/test_end_to_end.py
Last updated: 2026-02-14

Purpose
- Validate deterministic crash/restart behavior and persistence/evidence side effects in a full mocked orchestration run.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from nexus_orchestrator.config.schema import default_config
from nexus_orchestrator.control_plane import OrchestratorController, SimulatedCrashError
from nexus_orchestrator.domain.models import RunStatus
from nexus_orchestrator.integration_plane import GitEngine
from nexus_orchestrator.persistence.repositories import (
    AttemptRepo,
    EvidenceRepo,
    RunRepo,
    WorkItemRepo,
)
from nexus_orchestrator.persistence.state_db import StateDB


@pytest.mark.smoke
def test_end_to_end_smoke_crash_resume_deterministic(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _seed_repo(repo_root)

    project_root = Path(__file__).resolve().parents[2]
    sample_spec_source = project_root / "samples" / "specs" / "minimal_design_doc.md"
    sample_spec_target = repo_root / "samples" / "specs" / "minimal_design_doc.md"
    sample_spec_target.parent.mkdir(parents=True, exist_ok=True)
    sample_spec_target.write_text(sample_spec_source.read_text(encoding="utf-8"), encoding="utf-8")

    git_engine = GitEngine(repo_root)
    git_engine.init_or_open()
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "--no-gpg-sign", "-m", "seed workspace")
    _git(repo_root, "branch", "-f", "integration", "main")

    config = default_config()
    state_db_path = repo_root / "state" / "nexus.sqlite"
    config["paths"]["workspace_root"] = str((repo_root / "workspaces").as_posix())
    config["paths"]["evidence_root"] = str((repo_root / "evidence").as_posix())
    config["paths"]["state_db"] = str(state_db_path.as_posix())
    config["paths"]["constraint_registry"] = str(
        (project_root / "constraints" / "registry").as_posix()
    )

    crashing_controller = OrchestratorController(
        repo_root=repo_root,
        state_db_path=state_db_path,
        crash_after_attempts=2,
    )
    with pytest.raises(SimulatedCrashError):
        crashing_controller.run(
            spec_path="samples/specs/minimal_design_doc.md",
            config=config,
            mode="run",
            mock=True,
        )

    mid_db = StateDB(state_db_path)
    mid_run_repo = RunRepo(mid_db)
    active_runs = mid_run_repo.list(status=RunStatus.RUNNING, limit=10)
    assert len(active_runs) == 1
    crashed_run_id = active_runs[0].id

    resumed_controller = OrchestratorController(
        repo_root=repo_root,
        state_db_path=state_db_path,
    )
    result = resumed_controller.run(
        spec_path="samples/specs/minimal_design_doc.md",
        config=config,
        mode="resume",
        mock=True,
    )

    assert result.run_id == crashed_run_id
    assert result.status is RunStatus.COMPLETED
    assert result.resumed_from_crash is True
    assert result.failed_work_item_ids == ()
    assert len(result.merged_work_item_ids) == 3
    assert result.budget_tokens_used > 0
    assert result.provider_calls >= 3

    state_db = StateDB(state_db_path)
    run_repo = RunRepo(state_db)
    work_item_repo = WorkItemRepo(state_db)
    attempt_repo = AttemptRepo(state_db)
    evidence_repo = EvidenceRepo(state_db)

    persisted_run = run_repo.get(result.run_id)
    assert persisted_run is not None
    assert persisted_run.status is RunStatus.COMPLETED

    work_items = work_item_repo.list_for_run(result.run_id, limit=100)
    by_title = {item.title: item for item in work_items}
    a_item = by_title["Implement module A"]
    b_item = by_title["Implement module B"]
    c_item = by_title["Implement module C"]
    assert c_item.dependencies == (a_item.id, b_item.id)

    a_attempt = attempt_repo.list_for_work_item(a_item.id, limit=10)[0]
    b_attempt = attempt_repo.list_for_work_item(b_item.id, limit=10)[0]
    c_attempt = attempt_repo.list_for_work_item(c_item.id, limit=10)[0]
    assert c_attempt.created_at >= a_attempt.created_at
    assert c_attempt.created_at >= b_attempt.created_at

    assert len(result.dispatch_batches) >= 2
    first_batch = set(result.dispatch_batches[0])
    assert {a_item.id, b_item.id} <= first_batch
    assert result.dispatch_batches[1] == (c_item.id,)

    for work_item in work_items:
        records = evidence_repo.list_for_work_item(work_item.id, limit=100)
        assert records
        for record in records:
            for artifact_path in record.artifact_paths:
                assert (repo_root / artifact_path).exists()

    with sqlite3.connect(state_db_path) as conn:
        row = conn.execute(
            """
            SELECT queue_json, completed_work_items_json
            FROM merge_queue_states
            WHERE integration_branch = ?
            """,
            ("integration",),
        ).fetchone()
    assert row is not None
    queue_payload = json.loads(row[0])
    completed_payload = json.loads(row[1])
    assert queue_payload == []
    assert set(completed_payload) == set(result.merged_work_item_ids)


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


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _git(repo_root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_root, check=True, text=True, capture_output=True)
