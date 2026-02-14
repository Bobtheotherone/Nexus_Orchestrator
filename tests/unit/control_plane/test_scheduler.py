"""Unit tests for control-plane scheduler runnable selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from nexus_orchestrator.control_plane.scheduler import (
    Scheduler,
    SchedulerLimits,
    WorkItemRuntimeState,
)
from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    RiskTier,
    TaskGraph,
    WorkItem,
    WorkItemStatus,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

_BASE_TS = datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC)


def _fixed_now(seed: int) -> datetime:
    return _BASE_TS + timedelta(seconds=seed)


def _randbytes(seed: int) -> callable:
    byte_value = (seed % 251) + 1

    def _provider(size: int) -> bytes:
        return bytes([byte_value]) * size

    return _provider


def _make_work_item(
    seed: int,
    *,
    dependencies: tuple[str, ...] = (),
    status: WorkItemStatus = WorkItemStatus.READY,
    risk_tier: RiskTier = RiskTier.MEDIUM,
) -> WorkItem:
    work_item_id = ids.generate_work_item_id(
        timestamp_ms=1_700_000_000_000 + seed,
        randbytes=_randbytes(seed),
    )
    constraint = Constraint(
        id=f"CON-SEC-{(seed % 9_000) + 1:04d}",
        severity=ConstraintSeverity.MUST,
        category="security",
        description="Constraint for scheduler tests",
        checker_binding="scheduler_test",
        parameters={},
        requirement_links=("REQ-0001",),
        source=ConstraintSource.MANUAL,
        created_at=_fixed_now(seed),
    )
    envelope = ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=(constraint,),
        inherited_constraint_ids=(),
        compiled_at=_fixed_now(seed),
    )
    return WorkItem(
        id=work_item_id,
        title=f"Work item {seed}",
        description="Scheduler test item",
        scope=(f"src/module_{seed}.py",),
        constraint_envelope=envelope,
        dependencies=dependencies,
        status=status,
        risk_tier=risk_tier,
        requirement_links=("REQ-0001",),
        constraint_ids=(constraint.id,),
        created_at=_fixed_now(seed),
        updated_at=_fixed_now(seed + 1),
    )


def _make_task_graph(
    *,
    work_items: Iterable[WorkItem],
    edges: tuple[tuple[str, str], ...] = (),
    critical_path: tuple[str, ...] = (),
    seed: int = 1,
) -> TaskGraph:
    run_id = ids.generate_run_id(
        timestamp_ms=1_700_000_100_000 + seed,
        randbytes=_randbytes(seed),
    )
    return TaskGraph(
        run_id=run_id,
        work_items=tuple(work_items),
        edges=edges,
        critical_path=critical_path,
    )


def test_scheduler_prioritizes_explicit_critical_path() -> None:
    critical_root = _make_work_item(101, status=WorkItemStatus.READY)
    alternate_root = _make_work_item(102, status=WorkItemStatus.READY)
    critical_mid = _make_work_item(
        103,
        status=WorkItemStatus.PENDING,
        dependencies=(critical_root.id,),
    )
    critical_leaf = _make_work_item(
        104,
        status=WorkItemStatus.PENDING,
        dependencies=(critical_mid.id,),
    )
    alternate_leaf = _make_work_item(
        105,
        status=WorkItemStatus.PENDING,
        dependencies=(alternate_root.id,),
    )

    graph = _make_task_graph(
        work_items=(alternate_leaf, critical_leaf, critical_mid, alternate_root, critical_root),
        edges=(
            (critical_root.id, critical_mid.id),
            (critical_mid.id, critical_leaf.id),
            (alternate_root.id, alternate_leaf.id),
        ),
        critical_path=(critical_root.id, critical_mid.id, critical_leaf.id),
    )
    scheduler = Scheduler(limits=SchedulerLimits(max_dispatch_per_tick=2, max_in_flight=5))

    decision = scheduler.schedule(graph)

    assert decision.runnable == (critical_root.id, alternate_root.id)
    assert decision.selected == (critical_root.id, alternate_root.id)


def test_scheduler_derives_critical_path_priority_when_not_provided() -> None:
    long_root = _make_work_item(111, status=WorkItemStatus.READY)
    short_root = _make_work_item(112, status=WorkItemStatus.READY)
    long_mid = _make_work_item(
        113,
        status=WorkItemStatus.PENDING,
        dependencies=(long_root.id,),
    )
    long_leaf = _make_work_item(
        114,
        status=WorkItemStatus.PENDING,
        dependencies=(long_mid.id,),
    )
    short_leaf = _make_work_item(
        115,
        status=WorkItemStatus.PENDING,
        dependencies=(short_root.id,),
    )

    graph = _make_task_graph(
        work_items=(short_leaf, long_leaf, long_mid, short_root, long_root),
        edges=(
            (long_root.id, long_mid.id),
            (long_mid.id, long_leaf.id),
            (short_root.id, short_leaf.id),
        ),
    )
    scheduler = Scheduler(limits=SchedulerLimits(max_dispatch_per_tick=2, max_in_flight=5))

    decision = scheduler.schedule(graph)

    assert decision.selected == (long_root.id, short_root.id)


def test_scheduler_uses_wait_cycles_for_anti_starvation_fairness() -> None:
    critical = _make_work_item(201, status=WorkItemStatus.READY)
    long_tail = _make_work_item(202, status=WorkItemStatus.READY)
    graph = _make_task_graph(
        work_items=(long_tail, critical),
        critical_path=(critical.id,),
    )
    scheduler = Scheduler(
        limits=SchedulerLimits(
            max_dispatch_per_tick=1,
            max_in_flight=5,
            fairness_starvation_cycles=3,
        )
    )

    decision = scheduler.schedule(
        graph,
        states={
            critical.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=0),
            long_tail.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=7),
        },
    )

    assert decision.selected == (long_tail.id,)


def test_scheduler_throttles_by_global_and_per_risk_tier_limits() -> None:
    active_critical = _make_work_item(
        301,
        status=WorkItemStatus.DISPATCHED,
        risk_tier=RiskTier.CRITICAL,
    )
    candidate_critical_a = _make_work_item(
        302,
        status=WorkItemStatus.READY,
        risk_tier=RiskTier.CRITICAL,
    )
    candidate_critical_b = _make_work_item(
        303,
        status=WorkItemStatus.READY,
        risk_tier=RiskTier.CRITICAL,
    )
    candidate_medium = _make_work_item(
        304,
        status=WorkItemStatus.READY,
        risk_tier=RiskTier.MEDIUM,
    )

    graph = _make_task_graph(
        work_items=(
            candidate_critical_b,
            candidate_medium,
            active_critical,
            candidate_critical_a,
        ),
    )
    scheduler = Scheduler(
        limits=SchedulerLimits(
            max_dispatch_per_tick=3,
            max_in_flight=3,
            per_risk_tier_in_flight={
                RiskTier.CRITICAL: 1,
                RiskTier.MEDIUM: 2,
            },
        )
    )

    decision = scheduler.schedule(graph)

    assert decision.runnable == (
        candidate_critical_a.id,
        candidate_critical_b.id,
        candidate_medium.id,
    )
    assert decision.selected == (candidate_medium.id,)
    assert decision.blocked_by_limits == (candidate_critical_a.id, candidate_critical_b.id)


def test_scheduler_disables_speculative_execution_under_backpressure() -> None:
    parent = _make_work_item(401, status=WorkItemStatus.DISPATCHED)
    child = _make_work_item(
        402,
        status=WorkItemStatus.READY,
        dependencies=(parent.id,),
    )
    graph = _make_task_graph(
        work_items=(parent, child),
        edges=((parent.id, child.id),),
    )
    scheduler = Scheduler(
        limits=SchedulerLimits(
            max_dispatch_per_tick=1,
            max_in_flight=4,
            allow_speculative_execution=True,
        )
    )

    without_backpressure = scheduler.schedule(graph, backpressure=False)
    with_backpressure = scheduler.schedule(graph, backpressure=True)

    assert without_backpressure.selected == (child.id,)
    assert without_backpressure.speculative_runnable == (child.id,)
    assert without_backpressure.speculative_selected == (child.id,)

    assert with_backpressure.speculative_enabled is False
    assert with_backpressure.runnable == ()
    assert with_backpressure.selected == ()


def test_scheduler_selection_is_deterministic_for_identical_inputs() -> None:
    item_a = _make_work_item(501, status=WorkItemStatus.READY)
    item_b = _make_work_item(502, status=WorkItemStatus.READY)
    item_c = _make_work_item(503, status=WorkItemStatus.READY)
    graph = _make_task_graph(work_items=(item_c, item_a, item_b))
    scheduler = Scheduler(limits=SchedulerLimits(max_dispatch_per_tick=3, max_in_flight=5))

    first = scheduler.schedule(
        graph,
        states={
            item_c.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=1),
            item_a.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=3),
            item_b.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=2),
        },
    )
    second = scheduler.schedule(
        graph,
        states={
            item_b.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=2),
            item_c.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=1),
            item_a.id: WorkItemRuntimeState(status=WorkItemStatus.READY, wait_cycles=3),
        },
    )

    assert first.runnable == second.runnable
    assert first.selected == second.selected == (item_a.id, item_b.id, item_c.id)
