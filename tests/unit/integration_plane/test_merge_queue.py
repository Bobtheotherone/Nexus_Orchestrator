"""
nexus-orchestrator — test suite for deterministic merge queue semantics.

File: tests/unit/integration_plane/test_merge_queue.py
Last updated: 2026-02-13

Purpose
- Validate serialization, ordering, rollback/requeue, and restart safety invariants.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.domain import RiskTier
from nexus_orchestrator.integration_plane import merge_queue as mq
from nexus_orchestrator.integration_plane.conflict_resolution import (
    ConflictAuditEntry,
    ConflictClassification,
    ConflictInput,
    ConflictResolutionResult,
    ConflictResolver,
    ResolutionStatus,
    TrivialConflictProof,
)
from nexus_orchestrator.integration_plane.merge_queue import (
    MergeQueue,
    MergeQueueStateError,
    MergeStatus,
    QueueCandidate,
)
from nexus_orchestrator.integration_plane.merge_queue import (
    MergeResult as QueueMergeResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(slots=True)
class MergeCheck:
    source_branch: str
    target_head_before: str
    source_head_before: str
    is_fast_forward: bool


@dataclass(slots=True)
class FakeGitEngine:
    """In-memory deterministic Git model for merge queue tests."""

    branch_heads: dict[str, str] = field(default_factory=dict)
    commit_parents: dict[str, str | None] = field(default_factory=dict)
    operations: list[str] = field(default_factory=list)
    merge_checks: list[MergeCheck] = field(default_factory=list)
    rebase_parent_by_branch: dict[str, str] = field(default_factory=dict)

    _counter: int = 0

    async def get_head(self, branch: str) -> str:
        return self.branch_heads[branch]

    async def rebase(self, branch: str, onto_branch: str) -> dict[str, object]:
        onto_head = self.branch_heads[onto_branch]
        previous_head = self.branch_heads[branch]

        self._counter += 1
        rebased_head = f"{branch.replace('/', '_')}_r{self._counter}"
        self.commit_parents[rebased_head] = onto_head
        self.branch_heads[branch] = rebased_head
        self.rebase_parent_by_branch[branch] = onto_head

        self.operations.append(f"rebase:{branch}->{onto_branch}:{previous_head}->{rebased_head}")
        return {"success": True}

    async def fast_forward_merge(self, source_branch: str, target_branch: str) -> dict[str, object]:
        source_head = self.branch_heads[source_branch]
        target_head = self.branch_heads[target_branch]
        is_fast_forward = self._is_ancestor(target_head, source_head)

        self.operations.append(f"merge:{source_branch}->{target_branch}")
        self.merge_checks.append(
            MergeCheck(
                source_branch=source_branch,
                target_head_before=target_head,
                source_head_before=source_head,
                is_fast_forward=is_fast_forward,
            )
        )

        if not is_fast_forward:
            return {
                "success": False,
                "message": "not a fast-forward",
                "conflicts": ["non-fast-forward ancestry"],
            }

        self.branch_heads[target_branch] = source_head
        return {"success": True}

    async def reset_hard(self, branch: str, commit_sha: str) -> None:
        self.operations.append(f"reset:{branch}->{commit_sha}")
        self.branch_heads[branch] = commit_sha

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        cursor: str | None = descendant
        visited: set[str] = set()
        while cursor is not None and cursor not in visited:
            if cursor == ancestor:
                return True
            visited.add(cursor)
            cursor = self.commit_parents.get(cursor)
        return False


class NonRollbackGitEngine(FakeGitEngine):
    async def reset_hard(self, branch: str, commit_sha: str) -> None:
        self.operations.append(f"reset-failed:{branch}->{commit_sha}")
        raise RuntimeError("reset is unavailable")


class FailingMergeGitEngine(FakeGitEngine):
    mutate_integration_head_on_failure: bool = True

    async def fast_forward_merge(self, source_branch: str, target_branch: str) -> dict[str, object]:
        self.operations.append(f"merge-forced-failure:{source_branch}->{target_branch}")
        if self.mutate_integration_head_on_failure:
            self.branch_heads[target_branch] = f"{target_branch}_drifted"
        return {
            "success": False,
            "message": "forced merge failure",
            "conflicts": ["simulated merge failure"],
        }


class FailingMergeWithoutMutationGitEngine(FailingMergeGitEngine):
    mutate_integration_head_on_failure = False


class NonRollbackFailingMergeGitEngine(FailingMergeGitEngine):
    async def reset_hard(self, branch: str, commit_sha: str) -> None:
        self.operations.append(f"reset-failed:{branch}->{commit_sha}")
        raise RuntimeError("reset is unavailable")


def _build_engine(branches: tuple[str, ...]) -> FakeGitEngine:
    commit_parents = {
        "root": None,
        "integration_base": "root",
    }
    branch_heads = {"integration": "integration_base"}

    for index, branch in enumerate(branches, start=1):
        branch_head = f"{branch.replace('/', '_')}_head_{index}"
        commit_parents[branch_head] = "root"
        branch_heads[branch] = branch_head

    return FakeGitEngine(branch_heads=branch_heads, commit_parents=commit_parents)


def _build_engine_without_reset(branches: tuple[str, ...]) -> NonRollbackGitEngine:
    base = _build_engine(branches)
    return NonRollbackGitEngine(
        branch_heads=dict(base.branch_heads),
        commit_parents=dict(base.commit_parents),
    )


def _build_failing_merge_engine(
    engine_type: type[FailingMergeGitEngine], branches: tuple[str, ...]
) -> FailingMergeGitEngine:
    base = _build_engine(branches)
    return engine_type(
        branch_heads=dict(base.branch_heads),
        commit_parents=dict(base.commit_parents),
    )


def _candidate(
    work_item_id: str,
    branch: str,
    *,
    risk: RiskTier,
    dependencies: tuple[str, ...] = (),
) -> QueueCandidate:
    return QueueCandidate(
        branch=branch,
        work_item_id=work_item_id,
        evidence_ids=(f"ev-{work_item_id}",),
        risk_tier=risk,
        dependencies=dependencies,
    )


@dataclass(slots=True)
class _WorkItemObject:
    work_item_id: str
    title: str
    status: str = "ready"
    risk_tier: str = "low"
    scope: tuple[str, ...] = ("src/a.py",)
    dependencies: tuple[str, ...] = ()
    requirement_links: tuple[str, ...] = ("REQ-1",)


@dataclass(slots=True)
class _RebaseFailGitEngine(FakeGitEngine):
    async def rebase(self, branch: str, onto_branch: str) -> dict[str, object]:
        self.operations.append(f"rebase-failed:{branch}->{onto_branch}")
        return {
            "success": False,
            "message": "forced rebase failure",
            "conflicts": ["src/conflicted.py"],
        }


@dataclass(slots=True)
class _MissingGitMethodsEngine:
    async def get_head(self, branch: str) -> str:
        return "integration_base"


async def test_one_merge_at_a_time_uses_lock(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.LOW))
    queue.enqueue(_candidate("w2", "work/w2", risk=RiskTier.LOW))

    engine = _build_engine(("work/w1", "work/w2"))

    first_started = asyncio.Event()
    second_started = asyncio.Event()
    allow_first_to_finish = asyncio.Event()

    starts: dict[str, float] = {}
    ends: dict[str, float] = {}

    async def compositional(candidate: QueueCandidate) -> bool:
        starts[candidate.work_item_id] = time.monotonic()
        if candidate.work_item_id == "w1":
            first_started.set()
            await allow_first_to_finish.wait()
        else:
            second_started.set()
        ends[candidate.work_item_id] = time.monotonic()
        return True

    task1 = asyncio.create_task(queue.process_next(engine, compositional))
    await first_started.wait()

    task2 = asyncio.create_task(queue.process_next(engine, compositional))
    await asyncio.sleep(0)
    assert not second_started.is_set()

    allow_first_to_finish.set()
    result1, result2 = await asyncio.gather(task1, task2)

    assert result1.status == MergeStatus.MERGED
    assert result2.status == MergeStatus.MERGED
    assert starts["w2"] >= ends["w1"]


async def test_process_next_rebases_before_fast_forward_merge(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.MEDIUM))

    engine = _build_engine(("work/w1",))

    result = await queue.process_next(engine, lambda _candidate: True)

    assert result.status == MergeStatus.MERGED

    rebase_index = next(
        index for index, op in enumerate(engine.operations) if op.startswith("rebase:")
    )
    merge_index = next(
        index for index, op in enumerate(engine.operations) if op.startswith("merge:")
    )
    assert rebase_index < merge_index

    merge_check = engine.merge_checks[0]
    assert merge_check.is_fast_forward
    assert engine.commit_parents[merge_check.source_head_before] == merge_check.target_head_before


async def test_compositional_failure_requeues_and_preserves_integration_head(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queued = queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.MEDIUM))

    engine = _build_engine(("work/w1",))
    integration_before = await engine.get_head("integration")

    async def fail_check(_: QueueCandidate) -> dict[str, object]:
        return {
            "passed": False,
            "requeue": True,
            "message": "integration checks failed",
            "diagnostics": {"reason": "integration tests"},
        }

    result = await queue.process_next(engine, fail_check)

    assert result.status == MergeStatus.REQUEUED
    assert result.requeued
    assert result.diagnostics["reason"] == "integration tests"
    assert await engine.get_head("integration") == integration_before

    pending = queue.pending_candidates
    assert len(pending) == 1
    assert pending[0].work_item_id == "w1"
    assert pending[0].attempts == queued.attempts + 1
    assert pending[0].arrival > queued.arrival


async def test_ordering_dependency_then_risk_then_fifo(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")

    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.HIGH))
    queue.enqueue(_candidate("w2", "work/w2", risk=RiskTier.LOW, dependencies=("w4",)))
    queue.enqueue(_candidate("w3", "work/w3", risk=RiskTier.LOW))
    queue.enqueue(_candidate("w4", "work/w4", risk=RiskTier.MEDIUM))
    queue.enqueue(_candidate("w5", "work/w5", risk=RiskTier.LOW))

    engine = _build_engine(("work/w1", "work/w2", "work/w3", "work/w4", "work/w5"))

    merged_order: list[str] = []
    while queue.has_pending:
        result = await queue.process_next(engine, lambda _candidate: True)
        assert result.status == MergeStatus.MERGED
        assert result.candidate is not None
        merged_order.append(result.candidate.work_item_id)

    assert merged_order == ["w3", "w5", "w4", "w2", "w1"]


async def test_restart_safety_reconstructs_remaining_queue_without_duplication(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "merge-queue.json"

    queue1 = MergeQueue(state_file)
    queue1.enqueue(_candidate("a", "work/a", risk=RiskTier.MEDIUM))
    queue1.enqueue(_candidate("b", "work/b", risk=RiskTier.HIGH))
    queue1.enqueue(_candidate("c", "work/c", risk=RiskTier.LOW, dependencies=("a",)))

    engine = _build_engine(("work/a", "work/b", "work/c"))

    first = await queue1.process_next(engine, lambda _candidate: True)
    assert first.status == MergeStatus.MERGED
    assert first.candidate is not None
    assert first.candidate.work_item_id == "a"

    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    pending_ids = [entry["work_item_id"] for entry in persisted["queue"]]
    assert pending_ids == ["b", "c"]

    queue2 = MergeQueue(state_file)

    rebuilt_pending_ids = [candidate.work_item_id for candidate in queue2.pending_candidates]
    assert rebuilt_pending_ids == ["b", "c"]

    processed_after_restart: list[str] = []
    while queue2.has_pending:
        result = await queue2.process_next(engine, lambda _candidate: True)
        assert result.status == MergeStatus.MERGED
        assert result.candidate is not None
        processed_after_restart.append(result.candidate.work_item_id)

    assert processed_after_restart == ["c", "b"]
    assert queue2.pending_candidates == ()
    assert queue2.completed_work_items == ("a", "b", "c")


async def test_process_next_no_candidate_returns_no_candidate_and_records_hook(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    captured: list[MergeStatus] = []

    def hook(result: QueueMergeResult) -> None:
        captured.append(result.status)

    engine = _build_engine(())
    result = await queue.process_next(engine, lambda: True, persistence_hooks=hook)
    assert result.status == MergeStatus.NO_CANDIDATE
    assert captured == [MergeStatus.NO_CANDIDATE]


async def test_process_next_rebase_failure_marks_candidate_failed(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.MEDIUM))
    base = _build_engine(("work/w1",))
    engine = _RebaseFailGitEngine(
        branch_heads=dict(base.branch_heads),
        commit_parents=dict(base.commit_parents),
    )

    result = await queue.process_next(engine, lambda _candidate: True)
    assert result.status == MergeStatus.FAILED
    assert result.conflict_details == ("src/conflicted.py",)
    assert queue.failed_work_items == ("w1",)
    assert queue.pending_candidates == ()


async def test_process_next_with_unsatisfied_dependency_returns_no_candidate(
    tmp_path: Path,
) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(
        QueueCandidate(
            branch="work/w2",
            work_item_id="w2",
            evidence_ids=("ev-w2",),
            risk_tier=RiskTier.LOW,
            dependencies=("w1",),
        )
    )
    engine = _build_engine(("work/w2",))
    result = await queue.process_next(engine, lambda _candidate: True)
    assert result.status == MergeStatus.NO_CANDIDATE
    assert queue.pending_candidates[0].work_item_id == "w2"


async def test_process_next_supports_zero_arg_compositional_callback(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.LOW))
    engine = _build_engine(("work/w1",))

    result = await queue.process_next(engine, lambda: True)
    assert result.status == MergeStatus.MERGED


async def test_process_next_records_result_via_mapping_hook(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.LOW))
    engine = _build_engine(("work/w1",))
    recorded: list[MergeStatus] = []

    async def on_result(result: QueueMergeResult) -> None:
        recorded.append(result.status)

    result = await queue.process_next(
        engine,
        lambda _candidate: True,
        persistence_hooks={"record_merge_result": on_result},
    )
    assert result.status == MergeStatus.MERGED
    assert recorded == [MergeStatus.MERGED]


async def test_process_next_records_result_via_object_hook(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.LOW))
    engine = _build_engine(("work/w1",))

    class HookRecorder:
        def __init__(self) -> None:
            self.seen: list[MergeStatus] = []

        def record_merge_result(self, result: QueueMergeResult) -> None:
            self.seen.append(result.status)

    recorder = HookRecorder()
    result = await queue.process_next(
        engine,
        lambda _candidate: True,
        persistence_hooks=recorder,
    )
    assert result.status == MergeStatus.MERGED
    assert recorder.seen == [MergeStatus.MERGED]


async def test_process_next_handles_missing_git_methods_as_failure(tmp_path: Path) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.LOW))

    result = await queue.process_next(_MissingGitMethodsEngine(), lambda _candidate: True)
    assert result.status == MergeStatus.FAILED
    assert result.message is not None
    assert "rebase failed" in result.message


def test_state_file_invalid_json_hard_fails_with_explicit_exception(tmp_path: Path) -> None:
    state_file = tmp_path / "merge-queue.json"
    state_file.write_text('{"schema_version":1,', encoding="utf-8")

    with pytest.raises(MergeQueueStateError, match="invalid JSON in MergeQueue state file"):
        MergeQueue(state_file)


def test_state_file_missing_required_keys_hard_fails_with_explicit_exception(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "merge-queue.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "queue": [],
                "completed_work_items": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MergeQueueStateError, match="missing required keys"):
        MergeQueue(state_file)


def test_state_file_schema_version_mismatch_hard_fails(tmp_path: Path) -> None:
    state_file = tmp_path / "merge-queue.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "next_arrival": 0,
                "completed_work_items": [],
                "failed_work_items": [],
                "queue": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MergeQueueStateError, match="unsupported MergeQueue state schema version"):
        MergeQueue(state_file)


def test_state_file_duplicate_work_item_entries_hard_fail(tmp_path: Path) -> None:
    state_file = tmp_path / "merge-queue.json"
    entry = _candidate("w1", "work/w1", risk=RiskTier.LOW).to_dict()
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "next_arrival": 2,
                "completed_work_items": [],
                "failed_work_items": [],
                "queue": [entry, entry],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MergeQueueStateError, match="contains duplicate work item"):
        MergeQueue(state_file)


def test_state_file_completed_failed_overlap_hard_fails(tmp_path: Path) -> None:
    state_file = tmp_path / "merge-queue.json"
    state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "next_arrival": 0,
                "completed_work_items": ["w1"],
                "failed_work_items": ["w1"],
                "queue": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MergeQueueStateError, match="both completed and failed"):
        MergeQueue(state_file)


def test_queue_candidate_helpers_and_aliases() -> None:
    candidate = QueueCandidate(
        branch="work/w1",
        work_item="w1",
        evidence=("ev1", "ev2"),
        risk="low",
        dependencies=("dep1",),
        arrival=1,
        attempts=2,
    )
    assert candidate.work_item == "w1"
    assert candidate.evidence == ("ev1", "ev2")
    assert candidate.risk == RiskTier.LOW
    assert candidate.with_arrival(3).arrival == 3
    assert candidate.with_attempts(4).attempts == 4
    roundtrip = QueueCandidate.from_dict(candidate.to_dict())
    assert roundtrip == candidate

    with pytest.raises(TypeError, match="requires work_item_id/work_item"):
        QueueCandidate(branch="work/missing", work_item_id=None)
    with pytest.raises(ValueError, match="arrival must be >= 0"):
        QueueCandidate(branch="work/w1", work_item_id="w1", arrival=-1)
    with pytest.raises(ValueError, match="attempts must be >= 0"):
        QueueCandidate(branch="work/w1", work_item_id="w1", attempts=-1)


def test_merge_result_and_queue_metadata_helpers(tmp_path: Path) -> None:
    result = mq.MergeResult(
        status=MergeStatus.MERGED, candidate=None, conflict_details=(" a ", "a")
    )
    assert result.merged is True
    assert result.conflict_details == ("a",)

    queue = MergeQueue.from_json_file(tmp_path / "state.json", integration_branch="integration")
    assert queue.integration_branch == "integration"
    queued = queue.enqueue_candidate(_candidate("w1", "work/w1", risk=RiskTier.LOW))
    assert queued.work_item_id == "w1"


def test_merge_queue_private_helper_branches() -> None:
    assert mq._coerce_risk_tier("LOW") == RiskTier.LOW
    with pytest.raises(ValueError):
        mq._coerce_risk_tier("unknown")

    assert mq._mapping_truth({"status": "passed"}, keys=("status",), default=False) is True
    assert mq._mapping_truth({"status": "failed"}, keys=("status",), default=True) is False
    assert mq._mapping_optional_text({"message": " ok "}, keys=("message",)) == "ok"
    assert mq._mapping_optional_text({"message": None}, keys=("message",)) is None
    assert mq._mapping_truth({}, keys=("status",), default=True) is True
    assert mq._mapping_optional_text({"note": "x"}, keys=("message",)) is None

    coerced = mq._coerce_json_value({"a": [1, 2, {"b": object()}]})
    assert isinstance(coerced, dict)
    assert isinstance(coerced["a"], list)
    assert isinstance(coerced["a"][2], dict)
    assert isinstance(coerced["a"][2]["b"], str)
    deep: object = 0
    for _ in range(11):
        deep = {"v": deep}
    assert isinstance(mq._coerce_json_value(deep), dict)

    class Carrier:
        conflicts = ("x.py", "x.py", "y.py")

    assert mq._extract_conflict_details({"conflicts": ["x.py", "y.py"]}) == ("x.py", "y.py")
    assert mq._extract_conflict_details(Carrier()) == ("x.py", "y.py")

    assert mq._expect_int(1, field_name="value") == 1
    with pytest.raises(ValueError):
        mq._expect_int(-1, field_name="value")
    with pytest.raises(ValueError):
        mq._expect_str_sequence("bad", field_name="items")


def _assert_bisect_diagnostics(
    result_status: MergeStatus,
    *,
    expected_status: MergeStatus,
    expected_head_changed: bool,
    expected_rollback_performed: bool,
    expected_bisect_recommended: bool,
    integration_head_before: str | None,
    integration_head_after: str | None,
    diagnostics: Mapping[str, object],
) -> None:
    assert result_status == expected_status
    assert integration_head_before is not None
    assert integration_head_after is not None

    head_changed = integration_head_before != integration_head_after
    rollback_performed = diagnostics.get("rollback_performed")
    bisect_recommended = diagnostics.get("bisect_recommended")
    assert isinstance(rollback_performed, bool)
    assert isinstance(bisect_recommended, bool)

    assert head_changed is expected_head_changed
    assert rollback_performed is expected_rollback_performed
    assert bisect_recommended is expected_bisect_recommended
    assert bisect_recommended is (head_changed and not rollback_performed)


@pytest.mark.parametrize(
    (
        "mutate_integration_head",
        "rollback_supported",
        "expected_head_changed",
        "expected_rollback_performed",
        "expected_bisect_recommended",
    ),
    [
        (True, True, False, True, False),
        (True, False, True, False, True),
        (False, False, False, False, False),
    ],
)
async def test_bisect_recommended_truth_table_for_compositional_failures(
    tmp_path: Path,
    mutate_integration_head: bool,
    rollback_supported: bool,
    expected_head_changed: bool,
    expected_rollback_performed: bool,
    expected_bisect_recommended: bool,
) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.MEDIUM))

    engine = (
        _build_engine(("work/w1",))
        if rollback_supported
        else _build_engine_without_reset(("work/w1",))
    )

    async def fail_check(_: QueueCandidate, git_engine: FakeGitEngine) -> dict[str, object]:
        if mutate_integration_head:
            git_engine.branch_heads["integration"] = "integration_drifted"
        return {
            "passed": False,
            "requeue": True,
            "message": "forced compositional failure",
        }

    result = await queue.process_next(engine, fail_check)

    _assert_bisect_diagnostics(
        result.status,
        expected_status=MergeStatus.REQUEUED,
        expected_head_changed=expected_head_changed,
        expected_rollback_performed=expected_rollback_performed,
        expected_bisect_recommended=expected_bisect_recommended,
        integration_head_before=result.integration_head_before,
        integration_head_after=result.integration_head_after,
        diagnostics=result.diagnostics,
    )


@pytest.mark.parametrize(
    ("engine_type", "expected_head_changed", "expected_rollback_performed", "expected_bisect"),
    [
        (FailingMergeGitEngine, False, True, False),
        (NonRollbackFailingMergeGitEngine, True, False, True),
        (FailingMergeWithoutMutationGitEngine, False, False, False),
    ],
)
async def test_bisect_recommended_truth_table_for_merge_failures(
    tmp_path: Path,
    engine_type: type[FailingMergeGitEngine],
    expected_head_changed: bool,
    expected_rollback_performed: bool,
    expected_bisect: bool,
) -> None:
    queue = MergeQueue(tmp_path / "merge-queue.json")
    queue.enqueue(_candidate("w1", "work/w1", risk=RiskTier.MEDIUM))

    engine = _build_failing_merge_engine(engine_type, ("work/w1",))

    result = await queue.process_next(engine, lambda _candidate: True)

    _assert_bisect_diagnostics(
        result.status,
        expected_status=MergeStatus.FAILED,
        expected_head_changed=expected_head_changed,
        expected_rollback_performed=expected_rollback_performed,
        expected_bisect_recommended=expected_bisect,
        integration_head_before=result.integration_head_before,
        integration_head_after=result.integration_head_after,
        diagnostics=result.diagnostics,
    )


def test_conflict_resolver_whitespace_change_is_trivial() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/module.py",
        "ours_content": "def f():\n    return 1\n",
        "theirs_content": "def f():\n\treturn 1\n",
    }
    assert resolver.classify_conflict(conflict) == ConflictClassification.TRIVIAL
    result = resolver.auto_resolve_trivial(conflict)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.proof == TrivialConflictProof.WHITESPACE_ONLY


def test_conflict_resolver_contract_path_requires_replan() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "contracts/api.py",
        "ours_content": "class API: ...\n",
        "theirs_content": "class API2: ...\n",
    }
    assert resolver.classify_conflict(conflict) == ConflictClassification.CONTRACT_LEVEL
    assert resolver.requires_replan(conflict) is True
    result = resolver.auto_resolve_trivial(conflict)
    assert result.status == ResolutionStatus.UNRESOLVED


def test_conflict_resolver_non_trivial_is_not_auto_resolved() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/logic.py",
        "ours_content": "def calc(x):\n    return x + 1\n",
        "theirs_content": "def calc(x):\n    return x * 2\n",
    }
    assert resolver.classify_conflict(conflict) == ConflictClassification.NON_TRIVIAL
    result = resolver.auto_resolve_trivial(conflict)
    assert result.status == ResolutionStatus.UNRESOLVED


def test_conflict_input_from_mapping_aliases_and_prepare_context() -> None:
    resolver = ConflictResolver()
    conflict = {
        "file_path": "src/context.py",
        "local": "import a\nimport b\n\nx = 1\n",
        "incoming": "import b\nimport a\n\nx = 1\n",
        "ancestor": "import a\nimport b\n\nx = 1\n",
        "source_branch": "work/a",
        "target_branch": "work/b",
        "metadata": {"requires_replan": "false"},
    }
    context = resolver.prepare_integrator_context(
        conflict,
        relevant_work_items=[
            _WorkItemObject(work_item_id="w1", title="First"),
            {"id": "w2", "title": "Second", "scope": ["src/context.py"]},
        ],
    )

    assert context["conflict_path"] == "src/context.py"
    assert context["classification"] == "trivial"
    assert context["requires_replan"] is False
    relevant_work_items = context["relevant_work_items"]
    assert isinstance(relevant_work_items, list)
    assert len(relevant_work_items) == 2


def test_conflict_resolver_metadata_contract_flag_requires_replan() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/service.py",
        "ours_content": "x = 1\n",
        "theirs_content": "x = 2\n",
        "metadata": {"requires_replan": "true"},
    }
    assert resolver.classify_conflict(conflict) == ConflictClassification.CONTRACT_LEVEL
    assert resolver.requires_replan(conflict) is True


def test_prepare_integrator_context_for_contract_level_conflict() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "contracts/public_api.py",
        "ours_content": "class API: ...\n",
        "theirs_content": "class APIv2: ...\n",
    }
    context = resolver.prepare_integrator_context(conflict, relevant_work_items=[])
    assert context["classification"] == "contract_level"
    assert context["requires_replan"] is True
    recommended_actions = context["recommended_actions"]
    assert isinstance(recommended_actions, tuple)
    assert len(recommended_actions) == 3


def test_conflict_resolver_context_rejects_invalid_work_items_type() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/service.py",
        "ours_content": "x = 1\n",
        "theirs_content": "x = 2\n",
    }
    with pytest.raises(TypeError):
        resolver.prepare_integrator_context(conflict, relevant_work_items="invalid")


def test_conflict_resolver_custom_contract_prefixes_must_not_be_empty() -> None:
    with pytest.raises(ValueError):
        ConflictResolver(contract_path_prefixes=())


def test_conflict_resolver_exact_text_match_trivial_resolution() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/same.py",
        "ours_content": "x = 1\n",
        "theirs_content": "x = 1\n",
    }
    result = resolver.auto_resolve_trivial(conflict)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.proof == TrivialConflictProof.EXACT_TEXT_MATCH
    assert result.resolved_content == "x = 1\n"


def test_conflict_resolver_python_ast_equivalent_trivial_resolution() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/calc.py",
        "ours_content": "def calc(a, b):\n    return a + b\n",
        "theirs_content": "def calc(a,b):\n    return a+b\n",
    }
    result = resolver.auto_resolve_trivial(conflict)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.proof in {
        TrivialConflictProof.WHITESPACE_ONLY,
        TrivialConflictProof.PYTHON_AST_EQUIVALENT,
    }


def test_conflict_resolver_import_order_only_proof() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/imports.py",
        "ours_content": "import a\nimport b\n\nx = 1\n",
        "theirs_content": "import b\nimport a\n\nx = 1\n",
    }
    result = resolver.auto_resolve_trivial(conflict)
    assert result.status == ResolutionStatus.RESOLVED
    assert result.proof in {
        TrivialConflictProof.PYTHON_IMPORT_ORDER_ONLY,
        TrivialConflictProof.PYTHON_AST_EQUIVALENT,
    }


def test_conflict_resolution_result_validation_guards() -> None:
    with pytest.raises(ValueError):
        ConflictResolutionResult(
            path="src/x.py",
            classification=ConflictClassification.TRIVIAL,
            proof=TrivialConflictProof.WHITESPACE_ONLY,
            status=ResolutionStatus.RESOLVED,
            reason="missing resolved content",
            requires_replan=False,
            resolved_content=None,
        )

    with pytest.raises(ValueError):
        ConflictResolutionResult(
            path="src/x.py",
            classification=ConflictClassification.NON_TRIVIAL,
            proof=TrivialConflictProof.NONE,
            status=ResolutionStatus.RESOLVED,
            reason="non-trivial cannot be resolved",
            requires_replan=False,
            resolved_content="x = 1\n",
        )


def test_conflict_audit_entry_redacts_sensitive_details() -> None:
    entry = ConflictAuditEntry(
        sequence=1,
        action="classify",
        path="src/a.py",
        classification=ConflictClassification.NON_TRIVIAL,
        proof=TrivialConflictProof.NONE,
        details={"api_token": "super-secret-token"},
    )
    assert entry.details["api_token"] == "***REDACTED***"


def test_conflict_resolver_audit_log_sequence_is_monotonic() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/a.py",
        "ours_content": "x = 1\n",
        "theirs_content": "x = 2\n",
    }
    resolver.classify_conflict(conflict)
    resolver.requires_replan(conflict)
    log = resolver.audit_log
    assert len(log) == 2
    assert [entry.sequence for entry in log] == [1, 2]


def test_conflict_input_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        ConflictInput(
            path="../escape.py",
            ours_content="x = 1\n",
            theirs_content="x = 2\n",
        )


def test_conflict_resolver_invalid_python_syntax_is_conservative() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/broken.py",
        "ours_content": "def broken(:\n    return 1\n",
        "theirs_content": "def broken(:\n\treturn 1\n",
    }

    assert resolver.classify_conflict(conflict) == ConflictClassification.NON_TRIVIAL
    result = resolver.auto_resolve_trivial(conflict)
    assert result.classification == ConflictClassification.NON_TRIVIAL
    assert result.proof == TrivialConflictProof.NONE
    assert result.status == ResolutionStatus.UNRESOLVED


def test_conflict_resolver_surrogate_like_content_does_not_crash() -> None:
    resolver = ConflictResolver()
    conflict = {
        "path": "src/non_utf8_like.txt",
        "ours_content": "left-\udcff\n",
        "theirs_content": "right-\udcff\n",
    }

    assert resolver.classify_conflict(conflict) == ConflictClassification.NON_TRIVIAL
    result = resolver.auto_resolve_trivial(conflict)
    assert result.classification == ConflictClassification.NON_TRIVIAL
    assert result.proof == TrivialConflictProof.NONE
    assert result.status == ResolutionStatus.UNRESOLVED


def test_conflict_resolver_binary_like_content_is_conservative() -> None:
    resolver = ConflictResolver()
    long_binary_like_line = ("A" * 12000) + "\x00payload"
    conflict = {
        "path": "artifacts/blob.dat",
        "ours_content": f"{long_binary_like_line}\n",
        "theirs_content": f"{long_binary_like_line} \n",
    }

    assert resolver.classify_conflict(conflict) == ConflictClassification.NON_TRIVIAL
    result = resolver.auto_resolve_trivial(conflict)
    assert result.classification == ConflictClassification.NON_TRIVIAL
    assert result.proof == TrivialConflictProof.NONE
    assert result.status == ResolutionStatus.UNRESOLVED
