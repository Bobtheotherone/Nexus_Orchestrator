"""
nexus-orchestrator — test skeleton

File: tests/unit/domain/test_persistence_roundtrip.py
Last updated: 2026-02-11

Purpose
- Round-trip domain objects through persistence adapters (in-memory or temp SQLite).

What this test file should cover
- Saving and loading core entities (Run, WorkItem, EvidenceRecord).
- Migration compatibility checks (when introduced).

Functional requirements
- Use temp directories / temp DB; no network.

Non-functional requirements
- Isolated and repeatable.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from nexus_orchestrator.persistence.repositories import (
    AttemptRepo,
    ConstraintRepo,
    EvidenceRepo,
    IncidentRepo,
    MergeRepo,
    ProviderCallRecord,
    ProviderCallRepo,
    RunRepo,
    TaskGraphRepo,
    ToolInstallRecord,
    ToolInstallRepo,
    WorkItemRepo,
)
from nexus_orchestrator.persistence.state_db import StateDB

from ..persistence import (
    make_attempt,
    make_constraint,
    make_evidence,
    make_incident,
    make_merge,
    make_run,
    make_task_graph,
    make_work_item,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_roundtrip_major_persistence_models(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)
    constraint_repo = ConstraintRepo(db)
    attempt_repo = AttemptRepo(db)
    evidence_repo = EvidenceRepo(db)
    merge_repo = MergeRepo(db)
    incident_repo = IncidentRepo(db)
    task_graph_repo = TaskGraphRepo(db)
    provider_call_repo = ProviderCallRepo(db)
    tool_install_repo = ToolInstallRepo(db)

    work_item = make_work_item(101)
    run = make_run(100, work_item_ids=(work_item.id,))
    constraint = make_constraint(101)
    attempt = make_attempt(102, run_id=run.id, work_item_id=work_item.id)
    evidence = make_evidence(
        103,
        run_id=run.id,
        work_item_id=work_item.id,
        constraint_ids=(constraint.id,),
    )
    merge = make_merge(104, run_id=run.id, work_item_id=work_item.id, evidence_ids=(evidence.id,))
    incident = make_incident(105, run_id=run.id, work_item_id=work_item.id)
    graph = make_task_graph(run.id, (work_item,), ())

    provider_call = ProviderCallRecord(
        id="pc-0001",
        attempt_id=attempt.id,
        provider="openai",
        tokens=800,
        cost_usd=0.42,
        latency_ms=250,
        created_at=attempt.created_at,
        model=attempt.model,
        request_id="req-0001",
        metadata={"phase": "unit"},
    )
    tool_install = ToolInstallRecord(
        id="tool-0001",
        tool="ruff",
        version="0.14.14",
        checksum="a" * 64,
        approved=True,
        installed_at=attempt.created_at,
        installed_by="ci",
        metadata={"channel": "pinned"},
    )

    assert run_repo.add(run) == run
    assert constraint_repo.add(constraint) == constraint
    assert work_item_repo.add(run.id, work_item) == work_item
    assert attempt_repo.add(attempt) == attempt
    assert evidence_repo.add(evidence) == evidence
    assert merge_repo.add(merge) == merge
    assert incident_repo.add(incident) == incident
    assert task_graph_repo.add(graph) == graph
    assert provider_call_repo.add(provider_call) == provider_call
    assert tool_install_repo.add(tool_install) == tool_install

    assert run_repo.get(run.id) == run
    loaded_work_item = work_item_repo.get(work_item.id)
    assert loaded_work_item is not None
    # Evidence/merge updates work-item state atomically.
    assert loaded_work_item.status.value == "merged"
    assert loaded_work_item.commit_sha == merge.commit_sha
    assert loaded_work_item.evidence_ids == (evidence.id,)

    assert constraint_repo.get(constraint.id) == constraint
    assert attempt_repo.get(attempt.id) == attempt
    assert evidence_repo.get(evidence.id) == evidence
    assert merge_repo.get(merge.id) == merge
    assert incident_repo.get(incident.id) == incident
    assert task_graph_repo.get(run.id) == graph
    assert provider_call_repo.get(provider_call.id) == provider_call
    assert tool_install_repo.get(tool_install.id) == tool_install


def test_deterministic_json_storage_for_equivalent_upserts(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)

    work_item = make_work_item(202)
    run = make_run(201, work_item_ids=(work_item.id,), metadata={"z": 1, "a": {"b": 2, "a": 1}})

    run_repo.upsert(run)
    work_item_repo.upsert(run.id, work_item)

    row_before_run = db.query_one("SELECT payload_json FROM runs WHERE id = ?", (run.id,))
    row_before_work_item = db.query_one(
        "SELECT payload_json FROM work_items WHERE id = ?",
        (work_item.id,),
    )
    assert row_before_run is not None
    assert row_before_work_item is not None

    payload_before_run = str(row_before_run["payload_json"])
    payload_before_work_item = str(row_before_work_item["payload_json"])

    run_repo.upsert(run)
    work_item_repo.upsert(run.id, work_item)

    row_after_run = db.query_one("SELECT payload_json FROM runs WHERE id = ?", (run.id,))
    row_after_work_item = db.query_one(
        "SELECT payload_json FROM work_items WHERE id = ?", (work_item.id,)
    )
    assert row_after_run is not None
    assert row_after_work_item is not None

    payload_after_run = str(row_after_run["payload_json"])
    payload_after_work_item = str(row_after_work_item["payload_json"])

    assert payload_before_run == payload_after_run
    assert payload_before_work_item == payload_after_work_item

    parsed_run = json.loads(payload_after_run)
    parsed_work_item = json.loads(payload_after_work_item)
    assert payload_after_run == json.dumps(parsed_run, sort_keys=True, separators=(",", ":"))
    assert payload_after_work_item == json.dumps(
        parsed_work_item,
        sort_keys=True,
        separators=(",", ":"),
    )
