"""
nexus-orchestrator — verification pipeline engine

File: src/nexus_orchestrator/verification_plane/pipeline.py
Last updated: 2026-02-12

Purpose
- Define the normative verification stage model and execute checker pipelines deterministically.

Normative behavior
- Stage order is authoritative: build runs first, then downstream stages according to dependencies.
- Risk-tier policy decides required stages:
  - low: stages 1..5
  - medium: stages 1..6
  - high: stages 1..7 + adversarial required
  - critical: stages 1..8 + adversarial required
- Selection hooks support incremental/full runs while preserving deterministic ordering.
- Parallel stage execution is bounded by max concurrency.
- Per-stage timeouts and cancellation token propagation are enforced.
- Results are normalized and sorted into a deterministic ``PipelineResult``.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, TypeAlias

from nexus_orchestrator.domain import RiskTier
from nexus_orchestrator.utils.concurrency import CancellationToken, WorkerPool, run_with_timeout

BUILD_STAGE_ID: Final[str] = "build"
LINT_STAGE_ID: Final[str] = "lint_format"
TYPECHECK_STAGE_ID: Final[str] = "type_check"
UNIT_TEST_STAGE_ID: Final[str] = "unit_tests"
SECURITY_STAGE_ID: Final[str] = "security_scan"
INTEGRATION_STAGE_ID: Final[str] = "integration_tests"
PERFORMANCE_STAGE_ID: Final[str] = "performance"
ADVERSARIAL_STAGE_ID: Final[str] = "adversarial_tests"

DEFAULT_STAGE_IDS_IN_ORDER: Final[tuple[str, ...]] = (
    BUILD_STAGE_ID,
    LINT_STAGE_ID,
    TYPECHECK_STAGE_ID,
    UNIT_TEST_STAGE_ID,
    SECURITY_STAGE_ID,
    INTEGRATION_STAGE_ID,
    PERFORMANCE_STAGE_ID,
    ADVERSARIAL_STAGE_ID,
)

DEFAULT_STAGE_CHECKER_IDS: Final[Mapping[str, tuple[str, ...]]] = {
    BUILD_STAGE_ID: ("build_checker",),
    LINT_STAGE_ID: ("lint_checker",),
    TYPECHECK_STAGE_ID: ("typecheck_checker",),
    UNIT_TEST_STAGE_ID: ("test_checker",),
    SECURITY_STAGE_ID: ("security_checker",),
    INTEGRATION_STAGE_ID: ("test_checker",),
    PERFORMANCE_STAGE_ID: ("performance_checker",),
    ADVERSARIAL_STAGE_ID: ("adversarial/test_generator",),
}


class VerificationSelectionMode(StrEnum):
    """Pipeline scope selection mode."""

    INCREMENTAL = "incremental"
    FULL = "full"


class EarlyExitPolicy(StrEnum):
    """Policy for stopping after must-stage failures."""

    NEVER = "never"
    STOP_ON_MUST_FAILURE = "stop_on_must_failure"


class CheckerStatus(StrEnum):
    """Normalized checker/stage status values."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


_FAILURE_STATUSES: Final[frozenset[CheckerStatus]] = frozenset(
    {
        CheckerStatus.FAIL,
        CheckerStatus.ERROR,
        CheckerStatus.TIMEOUT,
        CheckerStatus.CANCELLED,
    }
)

_MUST_FAILURE_STATUSES: Final[frozenset[CheckerStatus]] = _FAILURE_STATUSES | frozenset(
    {CheckerStatus.SKIP}
)


@dataclass(frozen=True, slots=True)
class PipelineStage:
    """Stage definition for planning and execution."""

    stage_id: str
    order: int
    checker_ids: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    parallelizable: bool = True
    must_pass: bool = True
    default_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("PipelineStage.stage_id must be non-empty")
        if self.order <= 0:
            raise ValueError("PipelineStage.order must be > 0")

        normalized_checker_ids = _normalize_identifier_tuple(self.checker_ids)
        if not normalized_checker_ids:
            raise ValueError("PipelineStage.checker_ids must contain at least one checker id")
        object.__setattr__(self, "checker_ids", normalized_checker_ids)

        normalized_deps = _normalize_identifier_tuple(self.depends_on)
        if self.stage_id in normalized_deps:
            raise ValueError(f"PipelineStage {self.stage_id!r} cannot depend on itself")
        object.__setattr__(self, "depends_on", normalized_deps)

        if self.default_timeout_seconds is not None and self.default_timeout_seconds <= 0:
            raise ValueError("PipelineStage.default_timeout_seconds must be > 0 when provided")


@dataclass(frozen=True, slots=True)
class RiskTierPolicy:
    """Risk-tier requirements for stage coverage."""

    required_stage_ids: tuple[str, ...]
    adversarial_required: bool = False

    def __post_init__(self) -> None:
        normalized_items: list[str] = []
        seen: set[str] = set()
        for item in self.required_stage_ids:
            if not isinstance(item, str):
                raise TypeError("RiskTierPolicy.required_stage_ids must contain strings")
            candidate = item.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            normalized_items.append(candidate)
        normalized = tuple(normalized_items)
        if not normalized:
            raise ValueError("RiskTierPolicy.required_stage_ids must not be empty")
        object.__setattr__(self, "required_stage_ids", normalized)


DEFAULT_RISK_TIER_POLICY: Final[Mapping[RiskTier, RiskTierPolicy]] = {
    RiskTier.LOW: RiskTierPolicy(
        required_stage_ids=DEFAULT_STAGE_IDS_IN_ORDER[:5],
        adversarial_required=False,
    ),
    RiskTier.MEDIUM: RiskTierPolicy(
        required_stage_ids=DEFAULT_STAGE_IDS_IN_ORDER[:6],
        adversarial_required=False,
    ),
    RiskTier.HIGH: RiskTierPolicy(
        required_stage_ids=DEFAULT_STAGE_IDS_IN_ORDER[:7] + (ADVERSARIAL_STAGE_ID,),
        adversarial_required=True,
    ),
    RiskTier.CRITICAL: RiskTierPolicy(
        required_stage_ids=DEFAULT_STAGE_IDS_IN_ORDER[:8],
        adversarial_required=True,
    ),
}


@dataclass(frozen=True, slots=True)
class PipelineSelectionContext:
    """Context provided to incremental/full selection hooks."""

    risk_tier: RiskTier
    mode: VerificationSelectionMode
    changed_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CheckerContext:
    """Invocation context passed to checker callables."""

    stage_id: str
    stage_order: int
    checker_id: str
    risk_tier: RiskTier
    mode: VerificationSelectionMode
    timeout_seconds: float
    changed_paths: tuple[str, ...]
    cancel_token: CancellationToken


@dataclass(frozen=True, slots=True)
class CheckerResult:
    """Normalized checker execution result."""

    stage_id: str
    checker_id: str
    status: CheckerStatus
    duration_ms: int
    summary: str | None = None
    details: str | None = None
    artifact_paths: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("CheckerResult.stage_id must be non-empty")
        if not self.checker_id:
            raise ValueError("CheckerResult.checker_id must be non-empty")
        object.__setattr__(self, "status", _coerce_checker_status(self.status))
        if self.duration_ms < 0:
            raise ValueError("CheckerResult.duration_ms must be >= 0")
        object.__setattr__(self, "artifact_paths", _normalize_identifier_tuple(self.artifact_paths))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_failure(self) -> bool:
        return self.status in _FAILURE_STATUSES


@dataclass(frozen=True, slots=True)
class StageResult:
    """Aggregated stage-level result."""

    stage_id: str
    stage_order: int
    status: CheckerStatus
    checker_results: tuple[CheckerResult, ...]
    duration_ms: int
    must_pass: bool
    timeout_seconds: float
    skipped: bool = False
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("StageResult.stage_id must be non-empty")
        if self.stage_order <= 0:
            raise ValueError("StageResult.stage_order must be > 0")
        object.__setattr__(self, "status", _coerce_checker_status(self.status))
        if self.duration_ms < 0:
            raise ValueError("StageResult.duration_ms must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("StageResult.timeout_seconds must be > 0")
        ordered_checkers = tuple(
            sorted(
                self.checker_results,
                key=lambda item: (item.checker_id, item.status.value, item.duration_ms),
            )
        )
        object.__setattr__(self, "checker_results", ordered_checkers)

    @property
    def must_failure(self) -> bool:
        if not self.must_pass:
            return False
        if self.skipped and self.skip_reason != "no_checkers_selected":
            return False
        return self.status in _MUST_FAILURE_STATUSES


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Deterministic, normalized pipeline result."""

    risk_tier: RiskTier
    mode: VerificationSelectionMode
    early_exit_policy: EarlyExitPolicy
    required_stage_ids: tuple[str, ...]
    selected_stage_ids: tuple[str, ...]
    adversarial_required: bool
    stage_results: tuple[StageResult, ...]
    stopped_early: bool
    cancelled: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_stage_ids", _normalize_identifier_tuple(self.required_stage_ids)
        )
        object.__setattr__(
            self, "selected_stage_ids", _normalize_identifier_tuple(self.selected_stage_ids)
        )

        ordered_stage_results = tuple(
            sorted(self.stage_results, key=lambda item: (item.stage_order, item.stage_id))
        )
        object.__setattr__(self, "stage_results", ordered_stage_results)

    @property
    def checker_results(self) -> tuple[CheckerResult, ...]:
        flattened: list[CheckerResult] = []
        for stage_result in self.stage_results:
            flattened.extend(stage_result.checker_results)
        return tuple(flattened)

    @property
    def must_failure_stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stage_results if stage.must_failure)

    @property
    def timed_out(self) -> bool:
        return any(
            checker_result.status == CheckerStatus.TIMEOUT
            for checker_result in self.checker_results
        )

    @property
    def passed(self) -> bool:
        return not self.cancelled and not self.must_failure_stage_ids


@dataclass(frozen=True, slots=True)
class PlannedStage:
    """Concrete stage execution plan entry."""

    stage: PipelineStage
    checker_ids: tuple[str, ...]
    timeout_seconds: float

    def __post_init__(self) -> None:
        normalized_checker_ids = _normalize_selected_checker_ids(self.stage, self.checker_ids)
        object.__setattr__(self, "checker_ids", normalized_checker_ids)
        if self.timeout_seconds <= 0:
            raise ValueError("PlannedStage.timeout_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """Deterministic stage plan for a pipeline execution."""

    risk_tier: RiskTier
    mode: VerificationSelectionMode
    changed_paths: tuple[str, ...]
    required_stage_ids: tuple[str, ...]
    selected_stage_ids: tuple[str, ...]
    adversarial_required: bool
    stages: tuple[PlannedStage, ...]
    waves: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_paths", _normalize_identifier_tuple(self.changed_paths))
        object.__setattr__(
            self, "required_stage_ids", _normalize_identifier_tuple(self.required_stage_ids)
        )
        object.__setattr__(
            self, "selected_stage_ids", _normalize_identifier_tuple(self.selected_stage_ids)
        )


CheckerOutput: TypeAlias = CheckerResult | Mapping[str, object] | bool
CheckerRunner: TypeAlias = Callable[[CheckerContext], Awaitable[CheckerOutput] | CheckerOutput]
SelectionHook: TypeAlias = Callable[[PipelineStage, PipelineSelectionContext], Sequence[str] | None]


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Pipeline invocation inputs."""

    risk_tier: RiskTier | str
    selection_mode: VerificationSelectionMode | str = VerificationSelectionMode.INCREMENTAL
    changed_paths: tuple[str, ...] = ()
    include_stage_ids: tuple[str, ...] = ()
    exclude_stage_ids: tuple[str, ...] = ()
    stage_timeouts: Mapping[str, float] = field(default_factory=dict)
    early_exit_policy: EarlyExitPolicy | str | None = None
    incremental_selection_hook: SelectionHook | None = None
    full_selection_hook: SelectionHook | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_paths", _normalize_identifier_tuple(self.changed_paths))
        object.__setattr__(
            self, "include_stage_ids", _normalize_identifier_tuple(self.include_stage_ids)
        )
        object.__setattr__(
            self, "exclude_stage_ids", _normalize_identifier_tuple(self.exclude_stage_ids)
        )
        object.__setattr__(self, "stage_timeouts", dict(self.stage_timeouts))


class VerificationPipelineEngine:
    """
    Async verification pipeline engine.

    Pluggability
    - Stages/checker bindings can be replaced via constructor inputs and register helpers.
    - Checker logic is fully externalized via the checker registry.
    - Incremental/full selection hooks can choose checker subsets per stage.
    """

    def __init__(
        self,
        *,
        stages: Sequence[PipelineStage] = (),
        checkers: Mapping[str, CheckerRunner] | None = None,
        risk_tier_policy: Mapping[RiskTier | str, RiskTierPolicy] | None = None,
        max_parallel_stages: int = 4,
        default_stage_timeout_seconds: float = 600.0,
        stage_timeout_defaults: Mapping[str, float] | None = None,
        default_early_exit_policy: EarlyExitPolicy | str = EarlyExitPolicy.STOP_ON_MUST_FAILURE,
        incremental_selection_hook: SelectionHook | None = None,
        full_selection_hook: SelectionHook | None = None,
    ) -> None:
        if max_parallel_stages <= 0:
            raise ValueError("max_parallel_stages must be > 0")
        if default_stage_timeout_seconds <= 0:
            raise ValueError("default_stage_timeout_seconds must be > 0")

        stage_catalog = tuple(stages) if stages else _default_stage_catalog()
        ordered_stages, stages_by_id = _normalize_stage_catalog(stage_catalog)

        self._stages: tuple[PipelineStage, ...] = ordered_stages
        self._stages_by_id: dict[str, PipelineStage] = stages_by_id
        self._checkers: dict[str, CheckerRunner] = dict(checkers or {})
        self._max_parallel_stages = max_parallel_stages
        self._default_stage_timeout_seconds = float(default_stage_timeout_seconds)
        self._stage_timeout_defaults = _normalize_stage_timeout_defaults(
            stage_timeout_defaults or {}
        )
        self._default_early_exit_policy = _coerce_early_exit_policy(default_early_exit_policy)
        self._incremental_selection_hook = incremental_selection_hook
        self._full_selection_hook = full_selection_hook
        configured_policy: (
            Mapping[RiskTier | str, RiskTierPolicy] | Mapping[RiskTier, RiskTierPolicy]
        ) = risk_tier_policy if risk_tier_policy is not None else DEFAULT_RISK_TIER_POLICY
        self._risk_tier_policy = _normalize_risk_tier_policy(
            configured_policy,
            self._stages_by_id,
        )

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    @property
    def max_parallel_stages(self) -> int:
        return self._max_parallel_stages

    @property
    def checkers(self) -> Mapping[str, CheckerRunner]:
        return dict(self._checkers)

    def register_checker(self, checker_id: str, checker: CheckerRunner) -> None:
        if not checker_id:
            raise ValueError("checker_id must be non-empty")
        self._checkers[checker_id] = checker

    def register_stage(self, stage: PipelineStage, *, replace: bool = False) -> None:
        if stage.stage_id in self._stages_by_id and not replace:
            raise ValueError(f"stage already exists: {stage.stage_id!r}")

        updated = dict(self._stages_by_id)
        updated[stage.stage_id] = stage
        ordered_stages, stages_by_id = _normalize_stage_catalog(tuple(updated.values()))
        self._stages = ordered_stages
        self._stages_by_id = stages_by_id
        self._risk_tier_policy = _normalize_risk_tier_policy(
            self._risk_tier_policy, self._stages_by_id
        )

    def required_stage_ids_for_risk_tier(self, risk_tier: RiskTier | str) -> tuple[str, ...]:
        normalized_tier = _coerce_risk_tier(risk_tier)
        policy = self._risk_tier_policy[normalized_tier]
        return _ordered_stage_subset(policy.required_stage_ids, self._stages)

    def adversarial_required_for_risk_tier(self, risk_tier: RiskTier | str) -> bool:
        normalized_tier = _coerce_risk_tier(risk_tier)
        policy = self._risk_tier_policy[normalized_tier]
        return policy.adversarial_required

    def plan(self, request: PipelineRequest) -> PipelinePlan:
        risk_tier = _coerce_risk_tier(request.risk_tier)
        mode = _coerce_selection_mode(request.selection_mode)
        policy = self._risk_tier_policy[risk_tier]

        required_stage_ids = _ordered_stage_subset(policy.required_stage_ids, self._stages)
        required_set = set(required_stage_ids)

        candidate_set = set(required_stage_ids)
        candidate_set.update(request.include_stage_ids)
        for stage_id in request.exclude_stage_ids:
            if stage_id in required_set:
                raise ValueError(
                    f"cannot exclude required stage {stage_id!r} for risk tier {risk_tier.value}"
                )
            candidate_set.discard(stage_id)

        unknown_candidates = sorted(
            stage_id for stage_id in candidate_set if stage_id not in self._stages_by_id
        )
        if unknown_candidates:
            raise ValueError(f"unknown stage ids requested: {unknown_candidates}")

        selection_context = PipelineSelectionContext(
            risk_tier=risk_tier,
            mode=mode,
            changed_paths=request.changed_paths,
        )
        selection_hook = self._select_hook_for_mode(mode, request)

        planned: list[PlannedStage] = []
        for stage in self._stages:
            if stage.stage_id not in candidate_set:
                continue

            selected_checker_ids = stage.checker_ids
            if selection_hook is not None:
                selected_from_hook = selection_hook(stage, selection_context)
                if selected_from_hook is not None:
                    selected_checker_ids = _normalize_selected_checker_ids(
                        stage, selected_from_hook
                    )

            timeout_seconds = self._resolve_stage_timeout(stage.stage_id, request.stage_timeouts)
            planned.append(
                PlannedStage(
                    stage=stage,
                    checker_ids=selected_checker_ids,
                    timeout_seconds=timeout_seconds,
                )
            )

        selected_stage_ids = tuple(stage.stage.stage_id for stage in planned)
        missing_required = sorted(required_set - set(selected_stage_ids))
        if missing_required:
            raise ValueError(f"required stages missing from plan: {missing_required}")

        waves = _build_execution_waves(planned)
        return PipelinePlan(
            risk_tier=risk_tier,
            mode=mode,
            changed_paths=request.changed_paths,
            required_stage_ids=required_stage_ids,
            selected_stage_ids=selected_stage_ids,
            adversarial_required=policy.adversarial_required,
            stages=tuple(planned),
            waves=waves,
        )

    async def run(
        self,
        request: PipelineRequest,
        *,
        cancel_token: CancellationToken | None = None,
    ) -> PipelineResult:
        plan = self.plan(request)
        effective_early_exit_policy = (
            _coerce_early_exit_policy(request.early_exit_policy)
            if request.early_exit_policy is not None
            else self._default_early_exit_policy
        )
        token = cancel_token or CancellationToken()

        stage_results_by_id: dict[str, StageResult] = {}
        plan_by_id = {planned.stage.stage_id: planned for planned in plan.stages}
        cancelled = False
        stopped_early = False

        for wave in plan.waves:
            if token.is_cancelled:
                cancelled = True
                break

            wave_planned_stages = [plan_by_id[stage_id] for stage_id in wave]
            try:
                wave_results = await self._run_wave(
                    planned_stages=wave_planned_stages,
                    plan=plan,
                    cancel_token=token,
                )
            except asyncio.CancelledError:
                cancelled = True
                break

            for stage_result in wave_results:
                stage_results_by_id[stage_result.stage_id] = stage_result

            if effective_early_exit_policy is EarlyExitPolicy.STOP_ON_MUST_FAILURE and any(
                stage_result.must_failure for stage_result in wave_results
            ):
                stopped_early = True
                break

        if token.is_cancelled:
            cancelled = True

        for planned_stage in plan.stages:
            stage_id = planned_stage.stage.stage_id
            if stage_id in stage_results_by_id:
                continue

            skip_reason = (
                "cancelled" if cancelled else "stopped_early" if stopped_early else "not_executed"
            )
            stage_results_by_id[stage_id] = StageResult(
                stage_id=planned_stage.stage.stage_id,
                stage_order=planned_stage.stage.order,
                status=CheckerStatus.SKIP,
                checker_results=(),
                duration_ms=0,
                must_pass=planned_stage.stage.must_pass,
                timeout_seconds=planned_stage.timeout_seconds,
                skipped=True,
                skip_reason=skip_reason,
            )

        ordered_stage_results = tuple(
            stage_results_by_id[planned_stage.stage.stage_id] for planned_stage in plan.stages
        )
        return PipelineResult(
            risk_tier=plan.risk_tier,
            mode=plan.mode,
            early_exit_policy=effective_early_exit_policy,
            required_stage_ids=plan.required_stage_ids,
            selected_stage_ids=plan.selected_stage_ids,
            adversarial_required=plan.adversarial_required,
            stage_results=ordered_stage_results,
            stopped_early=stopped_early,
            cancelled=cancelled,
        )

    async def _run_wave(
        self,
        *,
        planned_stages: Sequence[PlannedStage],
        plan: PipelinePlan,
        cancel_token: CancellationToken,
    ) -> tuple[StageResult, ...]:
        if not planned_stages:
            return ()

        if len(planned_stages) == 1 or self._max_parallel_stages == 1:
            result = await self._run_stage(
                planned_stage=planned_stages[0],
                plan=plan,
                cancel_token=cancel_token,
            )
            return (result,)

        max_concurrency = min(self._max_parallel_stages, len(planned_stages))
        pool: WorkerPool[StageResult] = WorkerPool(
            max_concurrency=max_concurrency,
            cancel_token=cancel_token,
        )

        outcomes: list[StageResult] = []
        coroutines = [
            self._run_stage(
                planned_stage=planned_stage,
                plan=plan,
                cancel_token=cancel_token,
            )
            for planned_stage in planned_stages
        ]
        async for outcome in pool.run(coroutines):
            outcomes.append(outcome)

        stage_order_index = {
            planned_stage.stage.stage_id: index
            for index, planned_stage in enumerate(planned_stages)
        }
        outcomes.sort(key=lambda stage_result: stage_order_index[stage_result.stage_id])
        return tuple(outcomes)

    async def _run_stage(
        self,
        *,
        planned_stage: PlannedStage,
        plan: PipelinePlan,
        cancel_token: CancellationToken,
    ) -> StageResult:
        if not planned_stage.checker_ids:
            return StageResult(
                stage_id=planned_stage.stage.stage_id,
                stage_order=planned_stage.stage.order,
                status=CheckerStatus.SKIP,
                checker_results=(),
                duration_ms=0,
                must_pass=planned_stage.stage.must_pass,
                timeout_seconds=planned_stage.timeout_seconds,
                skipped=True,
                skip_reason="no_checkers_selected",
            )

        start = time.perf_counter()
        checker_results: list[CheckerResult] = []

        for checker_id in planned_stage.checker_ids:
            cancel_token.raise_if_cancelled()
            checker_results.append(
                await self._run_checker(
                    stage=planned_stage.stage,
                    checker_id=checker_id,
                    timeout_seconds=planned_stage.timeout_seconds,
                    plan=plan,
                    cancel_token=cancel_token,
                )
            )

        duration_ms = _duration_ms(start)
        stage_status = _aggregate_stage_status(tuple(checker_results))
        return StageResult(
            stage_id=planned_stage.stage.stage_id,
            stage_order=planned_stage.stage.order,
            status=stage_status,
            checker_results=tuple(checker_results),
            duration_ms=duration_ms,
            must_pass=planned_stage.stage.must_pass,
            timeout_seconds=planned_stage.timeout_seconds,
            skipped=False,
            skip_reason=None,
        )

    async def _run_checker(
        self,
        *,
        stage: PipelineStage,
        checker_id: str,
        timeout_seconds: float,
        plan: PipelinePlan,
        cancel_token: CancellationToken,
    ) -> CheckerResult:
        start = time.perf_counter()
        checker = self._checkers.get(checker_id)
        if checker is None:
            return CheckerResult(
                stage_id=stage.stage_id,
                checker_id=checker_id,
                status=CheckerStatus.ERROR,
                duration_ms=_duration_ms(start),
                summary="checker_not_registered",
                details=f"checker {checker_id!r} is not registered",
            )

        context = CheckerContext(
            stage_id=stage.stage_id,
            stage_order=stage.order,
            checker_id=checker_id,
            risk_tier=plan.risk_tier,
            mode=plan.mode,
            timeout_seconds=timeout_seconds,
            changed_paths=plan.changed_paths,
            cancel_token=cancel_token,
        )

        try:
            raw_output = await self._invoke_checker(
                checker=checker,
                context=context,
                timeout_seconds=timeout_seconds,
                cancel_token=cancel_token,
            )
        except TimeoutError:
            return CheckerResult(
                stage_id=stage.stage_id,
                checker_id=checker_id,
                status=CheckerStatus.TIMEOUT,
                duration_ms=_duration_ms(start),
                summary="checker_timeout",
                details=f"checker timed out after {timeout_seconds:.3f}s",
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            return CheckerResult(
                stage_id=stage.stage_id,
                checker_id=checker_id,
                status=CheckerStatus.ERROR,
                duration_ms=_duration_ms(start),
                summary="checker_error",
                details=f"{type(exc).__name__}: {exc}",
            )

        return _normalize_checker_output(
            output=raw_output,
            stage_id=stage.stage_id,
            checker_id=checker_id,
            default_duration_ms=_duration_ms(start),
        )

    async def _invoke_checker(
        self,
        *,
        checker: CheckerRunner,
        context: CheckerContext,
        timeout_seconds: float,
        cancel_token: CancellationToken,
    ) -> CheckerOutput:
        candidate = checker(context)
        awaitable: Awaitable[CheckerOutput]
        if inspect.isawaitable(candidate):
            awaitable = candidate
        else:
            awaitable = _immediate_checker_output(candidate)
        return await run_with_timeout(awaitable, timeout_seconds, cancel_token)

    def _resolve_stage_timeout(self, stage_id: str, overrides: Mapping[str, float]) -> float:
        if stage_id in overrides:
            timeout = float(overrides[stage_id])
        elif stage_id in self._stage_timeout_defaults:
            timeout = self._stage_timeout_defaults[stage_id]
        else:
            stage = self._stages_by_id[stage_id]
            timeout = (
                stage.default_timeout_seconds
                if stage.default_timeout_seconds is not None
                else self._default_stage_timeout_seconds
            )

        if timeout <= 0:
            raise ValueError(f"timeout for stage {stage_id!r} must be > 0")
        return timeout

    def _select_hook_for_mode(
        self,
        mode: VerificationSelectionMode,
        request: PipelineRequest,
    ) -> SelectionHook | None:
        if mode is VerificationSelectionMode.INCREMENTAL:
            return request.incremental_selection_hook or self._incremental_selection_hook
        return request.full_selection_hook or self._full_selection_hook


def _default_stage_catalog() -> tuple[PipelineStage, ...]:
    return (
        PipelineStage(
            stage_id=BUILD_STAGE_ID,
            order=1,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[BUILD_STAGE_ID],
            depends_on=(),
            parallelizable=False,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=LINT_STAGE_ID,
            order=2,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[LINT_STAGE_ID],
            depends_on=(BUILD_STAGE_ID,),
            parallelizable=True,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=TYPECHECK_STAGE_ID,
            order=3,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[TYPECHECK_STAGE_ID],
            depends_on=(BUILD_STAGE_ID,),
            parallelizable=True,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=UNIT_TEST_STAGE_ID,
            order=4,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[UNIT_TEST_STAGE_ID],
            depends_on=(BUILD_STAGE_ID,),
            parallelizable=True,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=SECURITY_STAGE_ID,
            order=5,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[SECURITY_STAGE_ID],
            depends_on=(BUILD_STAGE_ID,),
            parallelizable=True,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=INTEGRATION_STAGE_ID,
            order=6,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[INTEGRATION_STAGE_ID],
            depends_on=(BUILD_STAGE_ID, UNIT_TEST_STAGE_ID),
            parallelizable=True,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=PERFORMANCE_STAGE_ID,
            order=7,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[PERFORMANCE_STAGE_ID],
            depends_on=(BUILD_STAGE_ID, UNIT_TEST_STAGE_ID),
            parallelizable=True,
            must_pass=True,
        ),
        PipelineStage(
            stage_id=ADVERSARIAL_STAGE_ID,
            order=8,
            checker_ids=DEFAULT_STAGE_CHECKER_IDS[ADVERSARIAL_STAGE_ID],
            depends_on=(BUILD_STAGE_ID,),
            parallelizable=True,
            must_pass=True,
        ),
    )


def _normalize_stage_catalog(
    stages: Sequence[PipelineStage],
) -> tuple[tuple[PipelineStage, ...], dict[str, PipelineStage]]:
    if not stages:
        raise ValueError("at least one stage is required")

    ordered = tuple(sorted(stages, key=lambda stage: (stage.order, stage.stage_id)))
    stage_by_id: dict[str, PipelineStage] = {}
    seen_orders: set[int] = set()
    for stage in ordered:
        if stage.stage_id in stage_by_id:
            raise ValueError(f"duplicate stage id: {stage.stage_id!r}")
        if stage.order in seen_orders:
            raise ValueError(f"duplicate stage order: {stage.order}")
        stage_by_id[stage.stage_id] = stage
        seen_orders.add(stage.order)

    build_stage = stage_by_id.get(BUILD_STAGE_ID)
    if build_stage is None:
        raise ValueError(f"required stage {BUILD_STAGE_ID!r} is missing")
    if build_stage.order != ordered[0].order:
        raise ValueError("build stage must be first in normative order")

    stage_order_index = {stage.stage_id: stage.order for stage in ordered}
    for stage in ordered:
        for dependency in stage.depends_on:
            if dependency not in stage_by_id:
                raise ValueError(
                    f"stage {stage.stage_id!r} depends on unknown stage {dependency!r}"
                )
            if stage_order_index[dependency] >= stage.order:
                raise ValueError(
                    "stage dependencies must point to earlier stages: "
                    f"{stage.stage_id!r} -> {dependency!r}"
                )
    return ordered, stage_by_id


def _normalize_risk_tier_policy(
    policy: Mapping[RiskTier | str, RiskTierPolicy] | Mapping[RiskTier, RiskTierPolicy],
    stage_catalog: Mapping[str, PipelineStage],
) -> dict[RiskTier, RiskTierPolicy]:
    normalized: dict[RiskTier, RiskTierPolicy] = {}
    for raw_tier, tier_policy in policy.items():
        tier = _coerce_risk_tier(raw_tier)
        unknown = sorted(
            stage_id for stage_id in tier_policy.required_stage_ids if stage_id not in stage_catalog
        )
        if unknown:
            raise ValueError(f"risk policy for {tier.value!r} references unknown stages: {unknown}")
        normalized[tier] = tier_policy

    missing = sorted(tier.value for tier in RiskTier if tier not in normalized)
    if missing:
        raise ValueError(f"risk tier policy is missing tiers: {missing}")
    return normalized


def _build_execution_waves(planned_stages: Sequence[PlannedStage]) -> tuple[tuple[str, ...], ...]:
    if not planned_stages:
        return ()

    planned_by_id = {planned.stage.stage_id: planned for planned in planned_stages}
    remaining = set(planned_by_id)
    completed: set[str] = set()
    waves: list[tuple[str, ...]] = []

    while remaining:
        ready = [
            planned_by_id[stage_id]
            for stage_id in remaining
            if _dependencies_satisfied(
                planned_by_id[stage_id].stage.depends_on, completed, planned_by_id
            )
        ]
        if not ready:
            unresolved = sorted(remaining)
            raise ValueError(f"unable to resolve stage dependencies for: {unresolved}")

        ready.sort(key=lambda planned: (planned.stage.order, planned.stage.stage_id))
        first_ready = ready[0]
        if first_ready.stage.parallelizable:
            wave_planned = [planned for planned in ready if planned.stage.parallelizable]
        else:
            wave_planned = [first_ready]

        wave_stage_ids = tuple(
            stage_id
            for stage_id, _ in sorted(
                ((planned.stage.stage_id, planned.stage.order) for planned in wave_planned),
                key=lambda item: (item[1], item[0]),
            )
        )
        waves.append(wave_stage_ids)
        completed.update(wave_stage_ids)
        remaining.difference_update(wave_stage_ids)

    return tuple(waves)


def _dependencies_satisfied(
    dependencies: Sequence[str],
    completed: set[str],
    planned_by_id: Mapping[str, PlannedStage],
) -> bool:
    for dependency in dependencies:
        if dependency in planned_by_id and dependency not in completed:
            return False
    return True


def _aggregate_stage_status(checker_results: Sequence[CheckerResult]) -> CheckerStatus:
    if not checker_results:
        return CheckerStatus.SKIP

    statuses = {checker_result.status for checker_result in checker_results}
    for candidate in (
        CheckerStatus.CANCELLED,
        CheckerStatus.TIMEOUT,
        CheckerStatus.ERROR,
        CheckerStatus.FAIL,
    ):
        if candidate in statuses:
            return candidate
    if CheckerStatus.WARN in statuses:
        return CheckerStatus.WARN
    if statuses == {CheckerStatus.SKIP}:
        return CheckerStatus.SKIP
    if CheckerStatus.PASS in statuses and CheckerStatus.SKIP in statuses:
        return CheckerStatus.WARN
    return CheckerStatus.PASS


def _normalize_checker_output(
    *,
    output: CheckerOutput,
    stage_id: str,
    checker_id: str,
    default_duration_ms: int,
) -> CheckerResult:
    if isinstance(output, CheckerResult):
        return CheckerResult(
            stage_id=stage_id,
            checker_id=checker_id,
            status=output.status,
            duration_ms=output.duration_ms,
            summary=output.summary,
            details=output.details,
            artifact_paths=output.artifact_paths,
            metadata=output.metadata,
        )

    if isinstance(output, bool):
        status = CheckerStatus.PASS if output else CheckerStatus.FAIL
        return CheckerResult(
            stage_id=stage_id,
            checker_id=checker_id,
            status=status,
            duration_ms=default_duration_ms,
        )

    if isinstance(output, Mapping):
        status = _coerce_checker_status(output.get("status", CheckerStatus.PASS))
        duration_ms = _coerce_duration_ms(output.get("duration_ms"), default_duration_ms)
        summary = _coerce_optional_string(output.get("summary"))
        details = _coerce_optional_string(output.get("details"))
        artifact_paths = _coerce_string_tuple(output.get("artifact_paths", ()))
        metadata = _coerce_metadata_mapping(output.get("metadata", {}))
        return CheckerResult(
            stage_id=stage_id,
            checker_id=checker_id,
            status=status,
            duration_ms=duration_ms,
            summary=summary,
            details=details,
            artifact_paths=artifact_paths,
            metadata=metadata,
        )

    raise TypeError(
        "checker output must be CheckerResult, bool, or Mapping[str, object]; "
        f"got {type(output).__name__}"
    )


def _coerce_checker_status(value: object) -> CheckerStatus:
    if isinstance(value, CheckerStatus):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        return CheckerStatus(normalized)
    raise ValueError(f"invalid checker status value: {value!r}")


def _coerce_risk_tier(risk_tier: object) -> RiskTier:
    if isinstance(risk_tier, RiskTier):
        return risk_tier
    if isinstance(risk_tier, str):
        return RiskTier(risk_tier)
    raise TypeError(f"invalid risk tier value: {risk_tier!r}")


def _coerce_selection_mode(selection_mode: object) -> VerificationSelectionMode:
    if isinstance(selection_mode, VerificationSelectionMode):
        return selection_mode
    if isinstance(selection_mode, str):
        return VerificationSelectionMode(selection_mode)
    raise TypeError(f"invalid selection mode value: {selection_mode!r}")


def _coerce_early_exit_policy(early_exit_policy: object) -> EarlyExitPolicy:
    if isinstance(early_exit_policy, EarlyExitPolicy):
        return early_exit_policy
    if isinstance(early_exit_policy, str):
        return EarlyExitPolicy(early_exit_policy)
    raise TypeError(f"invalid early-exit policy value: {early_exit_policy!r}")


def _normalize_stage_timeout_defaults(
    stage_timeout_defaults: Mapping[str, float],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for stage_id, timeout_seconds in stage_timeout_defaults.items():
        if not stage_id:
            raise ValueError("stage timeout default key must be non-empty")
        timeout = float(timeout_seconds)
        if timeout <= 0:
            raise ValueError(f"timeout for stage {stage_id!r} must be > 0")
        normalized[stage_id] = timeout
    return normalized


def _normalize_identifier_tuple(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"identifier values must be strings, got {type(value).__name__}")
        candidate = value.strip()
        if not candidate:
            raise ValueError("identifier values must be non-empty")
        if candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return tuple(normalized)


def _normalize_selected_checker_ids(
    stage: PipelineStage, selected: Sequence[str]
) -> tuple[str, ...]:
    normalized_selected = _normalize_identifier_tuple(selected)
    order_index = {checker_id: index for index, checker_id in enumerate(stage.checker_ids)}
    return tuple(
        sorted(
            normalized_selected,
            key=lambda checker_id: (0, order_index[checker_id])
            if checker_id in order_index
            else (1, checker_id),
        )
    )


def _ordered_stage_subset(
    stage_ids: Sequence[str], catalog: Sequence[PipelineStage]
) -> tuple[str, ...]:
    requested = set(stage_ids)
    return tuple(stage.stage_id for stage in catalog if stage.stage_id in requested)


def _duration_ms(start: float) -> int:
    elapsed_seconds = max(time.perf_counter() - start, 0.0)
    return int(round(elapsed_seconds * 1000))


def _coerce_duration_ms(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("duration_ms must be an int/float when provided")
    duration_ms = int(round(float(value)))
    if duration_ms < 0:
        raise ValueError("duration_ms must be >= 0")
    return duration_ms


def _coerce_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"expected optional string, got {type(value).__name__}")
    return value


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return _normalize_identifier_tuple(value)
    if isinstance(value, list):
        return _normalize_identifier_tuple(value)
    raise TypeError(f"artifact_paths must be a sequence of strings, got {type(value).__name__}")


def _coerce_metadata_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(value).__name__}")
    metadata: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        metadata[key] = item
    return metadata


async def _immediate_checker_output(value: CheckerOutput) -> CheckerOutput:
    return value


__all__ = [
    "ADVERSARIAL_STAGE_ID",
    "BUILD_STAGE_ID",
    "CheckerContext",
    "CheckerOutput",
    "CheckerResult",
    "CheckerRunner",
    "CheckerStatus",
    "DEFAULT_RISK_TIER_POLICY",
    "DEFAULT_STAGE_CHECKER_IDS",
    "DEFAULT_STAGE_IDS_IN_ORDER",
    "EarlyExitPolicy",
    "INTEGRATION_STAGE_ID",
    "LINT_STAGE_ID",
    "PERFORMANCE_STAGE_ID",
    "PipelinePlan",
    "PipelineRequest",
    "PipelineResult",
    "PipelineSelectionContext",
    "PipelineStage",
    "PlannedStage",
    "RiskTierPolicy",
    "SECURITY_STAGE_ID",
    "SelectionHook",
    "StageResult",
    "TYPECHECK_STAGE_ID",
    "UNIT_TEST_STAGE_ID",
    "VerificationPipelineEngine",
    "VerificationSelectionMode",
]
