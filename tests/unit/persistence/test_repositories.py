"""Repository behavior tests for transactions, invariants, scheduling, and security."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import JSONValue, RunStatus, WorkItem, WorkItemStatus
from nexus_orchestrator.persistence.repositories import (
    AttemptRepo,
    ConstraintRepo,
    EvidenceRepo,
    ProviderCallRecord,
    ProviderCallRepo,
    RunRepo,
    TaskGraphRepo,
    ToolInstallRecord,
    ToolInstallRepo,
    WorkItemRepo,
)
from nexus_orchestrator.persistence.state_db import StateDB

from . import (
    collect_text_cells,
    make_attempt,
    make_evidence,
    make_run,
    make_task_graph,
    make_work_item,
)

if TYPE_CHECKING:
    from pathlib import Path

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
except ModuleNotFoundError:
    _HYPOTHESIS_AVAILABLE = False
else:
    _HYPOTHESIS_AVAILABLE = True


def test_transaction_atomicity_with_fk_failure_rolls_back_partial_rows(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)

    run = make_run(10_000)
    run_repo.add(run)

    missing_parent = ids.generate_work_item_id(
        timestamp_ms=1_800_000_000_000,
        randbytes=lambda size: b"\x7f" * size,
    )
    child = make_work_item(10_001, dependencies=(missing_parent,))

    with pytest.raises(sqlite3.IntegrityError):
        work_item_repo.add(run.id, child)

    assert work_item_repo.get(child.id) is None

    parent = make_work_item(10_002, status=WorkItemStatus.MERGED)
    work_item_repo.add(run.id, parent)

    valid_child = make_work_item(10_003, dependencies=(parent.id,))
    work_item_repo.add(run.id, valid_child)

    loaded_child = work_item_repo.get(valid_child.id)
    assert loaded_child is not None
    assert loaded_child.dependencies == (parent.id,)

    edge_row = db.query_one(
        """
        SELECT COUNT(*) AS edge_count
        FROM task_graph_edges
        WHERE run_id = ? AND parent_id = ? AND child_id = ?
        """,
        (run.id, parent.id, valid_child.id),
    )
    assert edge_row is not None
    edge_count = edge_row["edge_count"]
    assert edge_count is not None
    assert int(edge_count) == 1


def test_fk_and_status_invariants_are_enforced(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)
    attempt_repo = AttemptRepo(db)
    evidence_repo = EvidenceRepo(db)

    run = make_run(20_000)
    work_item = make_work_item(20_001)
    run_repo.add(run)
    work_item_repo.add(run.id, work_item)

    orphan_attempt = make_attempt(
        20_002,
        run_id=run.id,
        work_item_id=ids.generate_work_item_id(
            timestamp_ms=1_900_000_000_000,
            randbytes=lambda size: b"\x66" * size,
        ),
    )
    with pytest.raises(ValueError):
        attempt_repo.add(orphan_attempt)

    wrong_run_evidence = make_evidence(
        20_003,
        run_id=ids.generate_run_id(
            timestamp_ms=1_900_000_100_000,
            randbytes=lambda size: b"\x65" * size,
        ),
        work_item_id=work_item.id,
    )
    with pytest.raises(ValueError):
        evidence_repo.add(wrong_run_evidence)

    with pytest.raises(ValueError):
        work_item_repo.set_status(work_item.id, "__invalid_status__")

    with db.connection() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE work_items SET status = ? WHERE id = ?",
            ("__invalid_status__", work_item.id),
        )


def test_constraint_parameters_with_secrets_are_rejected(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    constraint_repo = ConstraintRepo(db)
    secret_openai = "sk-FAKEOPENAIKEY12345678901234567890"

    safe_constraint = make_work_item(25_000).constraint_envelope.constraints[0]
    assert constraint_repo.add(safe_constraint) == safe_constraint

    secret_constraint = make_work_item(
        25_001,
        constraint_parameters={"provider_secret": secret_openai},
    ).constraint_envelope.constraints[0]

    with pytest.raises(ValueError, match=r"Constraint\.parameters\.provider_secret"):
        constraint_repo.add(secret_constraint)

    with db.connection() as conn:
        all_text = collect_text_cells(conn)
    assert secret_openai not in all_text


def test_get_next_runnable_respects_dependency_and_status_gates(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)

    item_a = make_work_item(30_001, status=WorkItemStatus.READY)
    item_b = make_work_item(30_002, status=WorkItemStatus.READY)
    item_c = make_work_item(
        30_003,
        status=WorkItemStatus.READY,
        dependencies=(item_a.id, item_b.id),
    )
    item_d = make_work_item(30_004, status=WorkItemStatus.FAILED)

    run = make_run(
        30_000,
        status=RunStatus.RUNNING,
        work_item_ids=(item_a.id, item_b.id, item_c.id, item_d.id),
    )

    run_repo.add(run)
    work_item_repo.add(run.id, item_a)
    work_item_repo.add(run.id, item_b)
    work_item_repo.add(run.id, item_c)
    work_item_repo.add(run.id, item_d)

    initial = {item.id for item in work_item_repo.get_next_runnable(run.id, limit=16)}
    assert initial == {item_a.id, item_b.id}

    work_item_repo.set_status(item_a.id, WorkItemStatus.MERGED)
    after_a = {item.id for item in work_item_repo.get_next_runnable(run.id, limit=16)}
    assert after_a == {item_b.id}

    work_item_repo.set_status(item_b.id, WorkItemStatus.MERGED)
    after_b = {item.id for item in work_item_repo.get_next_runnable(run.id, limit=16)}
    assert after_b == {item_c.id}

    work_item_repo.set_status(item_c.id, WorkItemStatus.MERGED)
    final = {item.id for item in work_item_repo.get_next_runnable(run.id, limit=16)}
    assert final == set()


def test_provider_call_and_tool_install_roundtrip(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)
    attempt_repo = AttemptRepo(db)
    provider_repo = ProviderCallRepo(db)
    tool_repo = ToolInstallRepo(db)

    run = make_run(40_000)
    work_item = make_work_item(40_001)
    attempt = make_attempt(40_002, run_id=run.id, work_item_id=work_item.id)

    run_repo.add(run)
    work_item_repo.add(run.id, work_item)
    attempt_repo.add(attempt)

    provider_call = ProviderCallRecord(
        id="pc-40003",
        attempt_id=attempt.id,
        provider="openai",
        tokens=128,
        cost_usd=0.03,
        latency_ms=150,
        created_at=attempt.created_at,
        metadata={"source": "unit"},
    )
    tool_install = ToolInstallRecord(
        id="tool-40004",
        tool="mypy",
        version="1.19.1",
        checksum="b" * 64,
        approved=True,
        installed_at=attempt.created_at,
        metadata={"scope": "repo"},
    )

    provider_repo.add(provider_call)
    tool_repo.add(tool_install)

    assert provider_repo.get(provider_call.id) == provider_call
    assert tool_repo.get(tool_install.id) == tool_install


def test_semantic_secret_values_are_rejected_without_mutation(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state" / "nexus.sqlite3")
    db.migrate()

    run_repo = RunRepo(db)
    work_item_repo = WorkItemRepo(db)
    evidence_repo = EvidenceRepo(db)

    secret_openai = "sk-FAKEOPENAIKEY12345678901234567890"
    secret_github = "ghp_abcdefghijklmnopqrstuvwxyz0123456789"

    safe_run = make_run(50_000)
    safe_work_item = make_work_item(50_001)
    run_repo.add(safe_run)
    work_item_repo.add(safe_run.id, safe_work_item)

    run_with_secret = make_run(
        50_010,
        metadata={"operator": "local", "api_key": secret_openai},
    )
    with pytest.raises(ValueError, match=r"Run\.metadata\.api_key"):
        run_repo.add(run_with_secret)

    work_item_with_secret = make_work_item(
        50_011,
        constraint_parameters={"provider_token": secret_github},
    )
    with pytest.raises(
        ValueError,
        match=r"WorkItem\.constraint_envelope\.constraints\[0\]\.parameters\.provider_token",
    ):
        work_item_repo.add(safe_run.id, work_item_with_secret)

    evidence_with_secret = make_evidence(
        50_002,
        run_id=safe_run.id,
        work_item_id=safe_work_item.id,
        metadata={"prompt_blob": secret_openai},
    )
    with pytest.raises(ValueError, match=r"EvidenceRecord\.metadata\.prompt_blob"):
        evidence_repo.add(evidence_with_secret)

    with db.connection() as conn:
        all_text = collect_text_cells(conn)

    assert secret_openai not in all_text
    assert secret_github not in all_text


if _HYPOTHESIS_AVAILABLE:
    _SAFE_KEY = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=1,
        max_size=8,
    ).filter(lambda key: key not in {"token", "secret", "password", "api_key", "authorization"})

    _SCALAR: st.SearchStrategy[JSONValue] = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-100, max_value=100),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=0, max_size=20),
    )

    _JSON_OBJECT: st.SearchStrategy[JSONValue] = st.recursive(
        _SCALAR,
        lambda child: st.one_of(
            st.lists(child, max_size=4),
            st.dictionaries(_SAFE_KEY, child, max_size=4),
        ),
        max_leaves=20,
    )

    @st.composite
    def _dag_case(draw: st.DrawFn) -> tuple[int, tuple[tuple[int, int], ...]]:
        node_count = draw(st.integers(min_value=3, max_value=6))
        candidates = [(i, j) for i in range(node_count) for j in range(i + 1, node_count)]
        max_edges = min(len(candidates), node_count + 3)
        edges = draw(st.sets(st.sampled_from(candidates), max_size=max_edges))
        return node_count, tuple(sorted(edges))

    @given(metadata=st.dictionaries(_SAFE_KEY, _JSON_OBJECT, max_size=4))
    @settings(
        max_examples=25,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_work_item_constraint_parameters_roundtrip(
        metadata: dict[str, JSONValue],
        tmp_path: Path,
    ) -> None:
        digest = hashlib.sha1(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        db_path = tmp_path / "state" / f"meta-{digest[:12]}.sqlite3"
        if db_path.exists():
            db_path.unlink()

        db = StateDB(db_path)
        db.migrate()
        run_repo = RunRepo(db)
        work_item_repo = WorkItemRepo(db)

        work_item = make_work_item(60_001, constraint_parameters=metadata)
        run = make_run(60_000, work_item_ids=(work_item.id,))

        run_repo.add(run)
        work_item_repo.add(run.id, work_item)

        loaded = work_item_repo.get(work_item.id)
        assert loaded is not None
        assert loaded.constraint_envelope.constraints[0].parameters == metadata

    @given(case=_dag_case())
    @settings(
        max_examples=25,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_property_task_graph_roundtrip(
        case: tuple[int, tuple[tuple[int, int], ...]], tmp_path: Path
    ) -> None:
        node_count, edges = case
        digest = hashlib.sha1(repr(case).encode("utf-8")).hexdigest()
        db_path = tmp_path / "state" / f"dag-{digest[:12]}.sqlite3"
        if db_path.exists():
            db_path.unlink()

        db = StateDB(db_path)
        db.migrate()

        run_repo = RunRepo(db)
        work_item_repo = WorkItemRepo(db)
        task_graph_repo = TaskGraphRepo(db)

        base_items = [
            make_work_item(70_000 + idx, status=WorkItemStatus.READY) for idx in range(node_count)
        ]
        id_by_index = [item.id for item in base_items]

        items: list[WorkItem] = []
        for idx, item in enumerate(base_items):
            dependencies = [id_by_index[parent] for parent, child in edges if child == idx]
            payload = item.to_dict()
            dependency_values: list[JSONValue] = [dependency for dependency in dependencies]
            payload["dependencies"] = dependency_values
            items.append(WorkItem.from_dict(payload))

        run = make_run(69_999, work_item_ids=tuple(id_by_index), status=RunStatus.RUNNING)
        run_repo.add(run)
        for item in items:
            work_item_repo.add(run.id, item)

        graph_edges = tuple((id_by_index[parent], id_by_index[child]) for parent, child in edges)
        graph = make_task_graph(run.id, tuple(items), graph_edges)
        task_graph_repo.add(graph)

        loaded_graph = task_graph_repo.get(run.id)
        assert loaded_graph is not None
        assert set(loaded_graph.edges) == set(graph_edges)

else:

    def test_property_work_item_constraint_parameters_roundtrip() -> None:
        pytest.skip("hypothesis is not installed")

    def test_property_task_graph_roundtrip() -> None:
        pytest.skip("hypothesis is not installed")
