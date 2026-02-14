"""Deterministic scheduler for control-plane runnable work-item selection."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING, cast

from nexus_orchestrator.domain.models import RiskTier, TaskGraph, WorkItem, WorkItemStatus

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

_DISPATCHABLE_STATUSES = frozenset({WorkItemStatus.PENDING, WorkItemStatus.READY})
_ACTIVE_STATUSES = frozenset({WorkItemStatus.DISPATCHED, WorkItemStatus.VERIFYING})
_COMPLETED_DEPENDENCY_STATUSES = frozenset({WorkItemStatus.PASSED, WorkItemStatus.MERGED})
_TERMINAL_BLOCKING_STATUSES = frozenset({WorkItemStatus.FAILED})


@dataclass(frozen=True, slots=True)
class WorkItemRuntimeState:
    """Runtime lifecycle details used by scheduling policy."""

    status: WorkItemStatus
    wait_cycles: int = 0

    def __post_init__(self) -> None:
        if self.wait_cycles < 0:
            raise ValueError("wait_cycles must be >= 0")

    @classmethod
    def coerce(
        cls,
        value: WorkItemRuntimeState | WorkItemStatus | str,
        *,
        path: str,
    ) -> WorkItemRuntimeState:
        if isinstance(value, cls):
            return value
        return cls(status=_coerce_work_item_status(cast("WorkItemStatus | str", value), path=path))


@dataclass(frozen=True, slots=True)
class SchedulerLimits:
    """Resource-governor-like dispatch limits."""

    max_dispatch_per_tick: int = 8
    max_in_flight: int = 8
    per_risk_tier_in_flight: Mapping[RiskTier, int] = field(default_factory=dict)
    fairness_starvation_cycles: int = 3
    allow_speculative_execution: bool = True

    def __post_init__(self) -> None:
        if self.max_dispatch_per_tick <= 0:
            raise ValueError("max_dispatch_per_tick must be > 0")
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be > 0")
        if self.fairness_starvation_cycles < 0:
            raise ValueError("fairness_starvation_cycles must be >= 0")
        for tier, limit in self.per_risk_tier_in_flight.items():
            if not isinstance(tier, RiskTier):
                raise ValueError(
                    "per_risk_tier_in_flight keys must be RiskTier values, "
                    f"got {type(tier).__name__}"
                )
            if limit <= 0:
                raise ValueError(f"per_risk_tier_in_flight[{tier.value}] must be > 0")


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    """Deterministic scheduler output for one dispatch tick."""

    selected: tuple[str, ...]
    runnable: tuple[str, ...]
    speculative_selected: tuple[str, ...]
    speculative_runnable: tuple[str, ...]
    blocked_by_limits: tuple[str, ...]
    backpressure: bool
    speculative_enabled: bool


@dataclass(frozen=True, slots=True)
class _Candidate:
    work_item_id: str
    risk_tier: RiskTier
    wait_cycles: int
    speculative: bool
    on_critical_path: bool
    critical_position: int
    critical_depth: int
    created_at: datetime


class Scheduler:
    """Critical-path-first scheduler with anti-starvation and throttling."""

    __slots__ = ("_limits",)

    def __init__(self, *, limits: SchedulerLimits | None = None) -> None:
        self._limits = limits if limits is not None else SchedulerLimits()

    @property
    def limits(self) -> SchedulerLimits:
        return self._limits

    def schedule(
        self,
        task_graph: TaskGraph,
        *,
        states: Mapping[str, WorkItemRuntimeState | WorkItemStatus | str] | None = None,
        backpressure: bool = False,
    ) -> ScheduleDecision:
        work_items_by_id = _index_work_items(task_graph.work_items)
        children_by_id, parents_by_id = _build_dependency_maps(task_graph, work_items_by_id)
        runtime_states = _resolve_runtime_states(work_items_by_id, states)
        in_flight_by_tier, total_in_flight = _in_flight_counts(work_items_by_id, runtime_states)

        topo_order = _topological_order(work_items_by_id, children_by_id, parents_by_id)
        critical_depth = _downstream_depth(topo_order, children_by_id)
        critical_path = _resolve_critical_path(
            explicit_path=task_graph.critical_path,
            topo_order=topo_order,
            children_by_id=children_by_id,
            critical_depth=critical_depth,
        )
        critical_position = {
            work_item_id: index for index, work_item_id in enumerate(critical_path)
        }

        speculative_enabled = self._limits.allow_speculative_execution and not backpressure
        candidates: list[_Candidate] = []
        speculative_runnable: list[str] = []

        for work_item_id in sorted(work_items_by_id):
            state = runtime_states[work_item_id]
            if state.status not in _DISPATCHABLE_STATUSES:
                continue

            dependency_ids = parents_by_id[work_item_id]
            if _dependencies_completed(dependency_ids, runtime_states):
                speculative = False
            elif speculative_enabled and _dependencies_allow_speculation(
                dependency_ids, runtime_states
            ):
                speculative = True
                speculative_runnable.append(work_item_id)
            else:
                continue

            item = work_items_by_id[work_item_id]
            candidates.append(
                _Candidate(
                    work_item_id=work_item_id,
                    risk_tier=item.risk_tier,
                    wait_cycles=state.wait_cycles,
                    speculative=speculative,
                    on_critical_path=work_item_id in critical_position,
                    critical_position=critical_position.get(
                        work_item_id, len(work_items_by_id) + 1
                    ),
                    critical_depth=critical_depth[work_item_id],
                    created_at=item.created_at,
                )
            )

        ordered_candidates = tuple(sorted(candidates, key=self._candidate_sort_key))
        runnable = tuple(candidate.work_item_id for candidate in ordered_candidates)

        dispatch_capacity = min(
            self._limits.max_dispatch_per_tick,
            max(0, self._limits.max_in_flight - total_in_flight),
        )
        selected: list[str] = []
        speculative_selected: list[str] = []
        blocked_by_limits: list[str] = []
        tier_counts = defaultdict(int, in_flight_by_tier)

        for candidate in ordered_candidates:
            if len(selected) >= dispatch_capacity:
                blocked_by_limits.append(candidate.work_item_id)
                continue

            tier_limit = self._limits.per_risk_tier_in_flight.get(
                candidate.risk_tier,
                self._limits.max_in_flight,
            )
            if tier_counts[candidate.risk_tier] >= tier_limit:
                blocked_by_limits.append(candidate.work_item_id)
                continue

            selected.append(candidate.work_item_id)
            tier_counts[candidate.risk_tier] += 1
            if candidate.speculative:
                speculative_selected.append(candidate.work_item_id)

        return ScheduleDecision(
            selected=tuple(selected),
            runnable=runnable,
            speculative_selected=tuple(speculative_selected),
            speculative_runnable=tuple(speculative_runnable),
            blocked_by_limits=tuple(blocked_by_limits),
            backpressure=backpressure,
            speculative_enabled=speculative_enabled,
        )

    def _candidate_sort_key(self, candidate: _Candidate) -> tuple[object, ...]:
        starved = candidate.wait_cycles >= self._limits.fairness_starvation_cycles
        return (
            0 if starved else 1,
            -candidate.wait_cycles,
            0 if candidate.on_critical_path else 1,
            candidate.critical_position,
            -candidate.critical_depth,
            0 if not candidate.speculative else 1,
            candidate.created_at,
            candidate.work_item_id,
        )


def schedule_runnable_work_items(
    task_graph: TaskGraph,
    *,
    states: Mapping[str, WorkItemRuntimeState | WorkItemStatus | str] | None = None,
    limits: SchedulerLimits | None = None,
    backpressure: bool = False,
) -> ScheduleDecision:
    """Convenience wrapper around :class:`Scheduler`."""

    return Scheduler(limits=limits).schedule(task_graph, states=states, backpressure=backpressure)


def select_runnable_work_items(
    task_graph: TaskGraph,
    *,
    states: Mapping[str, WorkItemRuntimeState | WorkItemStatus | str] | None = None,
    limits: SchedulerLimits | None = None,
    backpressure: bool = False,
) -> tuple[str, ...]:
    """Return only the selected work-item IDs for one scheduler tick."""

    decision = schedule_runnable_work_items(
        task_graph,
        states=states,
        limits=limits,
        backpressure=backpressure,
    )
    return decision.selected


def _index_work_items(work_items: tuple[WorkItem, ...]) -> dict[str, WorkItem]:
    indexed: dict[str, WorkItem] = {}
    for item in work_items:
        indexed[item.id] = item
    return indexed


def _build_dependency_maps(
    task_graph: TaskGraph,
    work_items_by_id: Mapping[str, WorkItem],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    children_by_id: dict[str, set[str]] = {work_item_id: set() for work_item_id in work_items_by_id}
    parents_by_id: dict[str, set[str]] = {work_item_id: set() for work_item_id in work_items_by_id}

    for parent_id, child_id in task_graph.edges:
        children_by_id[parent_id].add(child_id)
        parents_by_id[child_id].add(parent_id)

    for work_item_id, item in work_items_by_id.items():
        for dependency_id in item.dependencies:
            if dependency_id not in work_items_by_id:
                continue
            children_by_id[dependency_id].add(work_item_id)
            parents_by_id[work_item_id].add(dependency_id)

    return children_by_id, parents_by_id


def _resolve_runtime_states(
    work_items_by_id: Mapping[str, WorkItem],
    states: Mapping[str, WorkItemRuntimeState | WorkItemStatus | str] | None,
) -> dict[str, WorkItemRuntimeState]:
    runtime_states = {
        work_item_id: WorkItemRuntimeState(status=item.status)
        for work_item_id, item in work_items_by_id.items()
    }
    if states is None:
        return runtime_states

    for work_item_id, value in states.items():
        if work_item_id not in work_items_by_id:
            raise ValueError(f"states contains unknown work_item_id: {work_item_id}")
        runtime_states[work_item_id] = WorkItemRuntimeState.coerce(
            value,
            path=f"states[{work_item_id!r}]",
        )
    return runtime_states


def _in_flight_counts(
    work_items_by_id: Mapping[str, WorkItem],
    runtime_states: Mapping[str, WorkItemRuntimeState],
) -> tuple[dict[RiskTier, int], int]:
    per_tier: dict[RiskTier, int] = defaultdict(int)
    total_in_flight = 0
    for work_item_id, state in runtime_states.items():
        if state.status not in _ACTIVE_STATUSES:
            continue
        total_in_flight += 1
        per_tier[work_items_by_id[work_item_id].risk_tier] += 1
    return per_tier, total_in_flight


def _topological_order(
    work_items_by_id: Mapping[str, WorkItem],
    children_by_id: Mapping[str, set[str]],
    parents_by_id: Mapping[str, set[str]],
) -> tuple[str, ...]:
    indegree = {work_item_id: len(parents_by_id[work_item_id]) for work_item_id in work_items_by_id}
    ready = [work_item_id for work_item_id, degree in indegree.items() if degree == 0]
    heapify(ready)

    ordered: list[str] = []
    while ready:
        work_item_id = heappop(ready)
        ordered.append(work_item_id)
        for child_id in sorted(children_by_id[work_item_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heappush(ready, child_id)

    if len(ordered) != len(work_items_by_id):
        raise ValueError("Task graph contains at least one cycle.")
    return tuple(ordered)


def _downstream_depth(
    topo_order: tuple[str, ...],
    children_by_id: Mapping[str, set[str]],
) -> dict[str, int]:
    depth = {work_item_id: 1 for work_item_id in topo_order}
    for work_item_id in reversed(topo_order):
        child_depth = [depth[child_id] for child_id in children_by_id[work_item_id]]
        if child_depth:
            depth[work_item_id] = 1 + max(child_depth)
    return depth


def _resolve_critical_path(
    *,
    explicit_path: tuple[str, ...],
    topo_order: tuple[str, ...],
    children_by_id: Mapping[str, set[str]],
    critical_depth: Mapping[str, int],
) -> tuple[str, ...]:
    if explicit_path:
        return explicit_path
    if not topo_order:
        return ()

    start_id = min(
        topo_order, key=lambda work_item_id: (-critical_depth[work_item_id], work_item_id)
    )
    path = [start_id]
    cursor = start_id
    while children_by_id[cursor]:
        next_id = min(
            children_by_id[cursor],
            key=lambda work_item_id: (-critical_depth[work_item_id], work_item_id),
        )
        path.append(next_id)
        cursor = next_id
    return tuple(path)


def _dependencies_completed(
    dependency_ids: set[str],
    runtime_states: Mapping[str, WorkItemRuntimeState],
) -> bool:
    for dependency_id in dependency_ids:
        if runtime_states[dependency_id].status not in _COMPLETED_DEPENDENCY_STATUSES:
            return False
    return True


def _dependencies_allow_speculation(
    dependency_ids: set[str],
    runtime_states: Mapping[str, WorkItemRuntimeState],
) -> bool:
    unresolved_count = 0
    for dependency_id in dependency_ids:
        status = runtime_states[dependency_id].status
        if status in _COMPLETED_DEPENDENCY_STATUSES:
            continue
        if status in _TERMINAL_BLOCKING_STATUSES or status not in _ACTIVE_STATUSES:
            return False
        unresolved_count += 1
    return unresolved_count > 0


def _coerce_work_item_status(
    value: WorkItemStatus | str,
    *,
    path: str,
) -> WorkItemStatus:
    if isinstance(value, WorkItemStatus):
        return value
    if isinstance(value, str):
        try:
            return WorkItemStatus(value)
        except ValueError as exc:
            allowed = ", ".join(sorted(status.value for status in WorkItemStatus))
            raise ValueError(f"{path} must be one of: {allowed}") from exc
    raise ValueError(f"{path} must be WorkItemStatus or str, got {type(value).__name__}")


__all__ = [
    "ScheduleDecision",
    "Scheduler",
    "SchedulerLimits",
    "WorkItemRuntimeState",
    "schedule_runnable_work_items",
    "select_runnable_work_items",
]
