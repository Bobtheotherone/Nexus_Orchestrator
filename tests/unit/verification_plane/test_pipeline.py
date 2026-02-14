"""
nexus-orchestrator — test skeleton

File: tests/unit/verification_plane/test_pipeline.py
Last updated: 2026-02-11

Purpose
- Validate verification pipeline stage ordering and selection by risk tier.

What this test file should cover
- Risk tier mapping to required stages.
- Incremental vs full verification decisions.
- Artifact/evidence bundling.

Functional requirements
- No provider calls; use fixtures.

Non-functional requirements
- Deterministic.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Final

import pytest

from nexus_orchestrator.domain import RiskTier
from nexus_orchestrator.utils.concurrency import CancellationToken
from nexus_orchestrator.verification_plane.pipeline import (
    ADVERSARIAL_STAGE_ID,
    BUILD_STAGE_ID,
    DEFAULT_STAGE_IDS_IN_ORDER,
    INTEGRATION_STAGE_ID,
    LINT_STAGE_ID,
    SECURITY_STAGE_ID,
    TYPECHECK_STAGE_ID,
    UNIT_TEST_STAGE_ID,
    CheckerContext,
    EarlyExitPolicy,
    PipelineRequest,
    PipelineSelectionContext,
    PipelineStage,
    VerificationPipelineEngine,
    VerificationSelectionMode,
)


@dataclass(slots=True)
class ConcurrencyProbe:
    in_flight: int = 0
    max_in_flight: int = 0

    async def enter(self) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)

    async def exit(self) -> None:
        self.in_flight -= 1


@pytest.mark.asyncio
async def test_build_stage_runs_first_before_parallel_stages() -> None:
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}

    async def make_checker(name: str, delay: float) -> dict[str, object]:
        starts[name] = time.monotonic()
        await asyncio.sleep(delay)
        ends[name] = time.monotonic()
        return {"status": "pass", "duration_ms": int(delay * 1000)}

    async def build_checker(_: CheckerContext) -> dict[str, object]:
        return await make_checker(BUILD_STAGE_ID, 0.05)

    async def lint_checker(_: CheckerContext) -> dict[str, object]:
        return await make_checker(LINT_STAGE_ID, 0.01)

    async def type_checker(_: CheckerContext) -> dict[str, object]:
        return await make_checker(TYPECHECK_STAGE_ID, 0.01)

    async def test_checker(_: CheckerContext) -> dict[str, object]:
        return await make_checker(UNIT_TEST_STAGE_ID, 0.01)

    async def security_checker(_: CheckerContext) -> dict[str, object]:
        return await make_checker(SECURITY_STAGE_ID, 0.01)

    engine = VerificationPipelineEngine(
        checkers={
            "build_checker": build_checker,
            "lint_checker": lint_checker,
            "typecheck_checker": type_checker,
            "test_checker": test_checker,
            "security_checker": security_checker,
        },
        max_parallel_stages=4,
    )

    result = await engine.run(PipelineRequest(risk_tier=RiskTier.LOW))

    assert result.passed
    build_end = ends[BUILD_STAGE_ID]
    for stage_id in (LINT_STAGE_ID, TYPECHECK_STAGE_ID, UNIT_TEST_STAGE_ID, SECURITY_STAGE_ID):
        assert starts[stage_id] >= build_end


@pytest.mark.asyncio
async def test_parallel_stage_execution_is_bounded_by_max_concurrency() -> None:
    probe = ConcurrencyProbe()

    async def build_checker(_: CheckerContext) -> dict[str, object]:
        await asyncio.sleep(0.01)
        return {"status": "pass"}

    async def parallel_checker(_: CheckerContext) -> dict[str, object]:
        await probe.enter()
        try:
            await asyncio.sleep(0.05)
            return {"status": "pass"}
        finally:
            await probe.exit()

    engine = VerificationPipelineEngine(
        checkers={
            "build_checker": build_checker,
            "lint_checker": parallel_checker,
            "typecheck_checker": parallel_checker,
            "test_checker": parallel_checker,
            "security_checker": parallel_checker,
        },
        max_parallel_stages=2,
    )

    result = await engine.run(PipelineRequest(risk_tier=RiskTier.LOW))

    assert result.passed
    assert probe.max_in_flight == 2


@pytest.mark.asyncio
async def test_cancellation_stops_remaining_stages_deterministically() -> None:
    async def build_checker(_: CheckerContext) -> dict[str, object]:
        await asyncio.sleep(0.02)
        return {"status": "pass"}

    async def slow_checker(_: CheckerContext) -> dict[str, object]:
        await asyncio.sleep(0.25)
        return {"status": "pass"}

    token = CancellationToken()

    async def cancel_soon() -> None:
        await asyncio.sleep(0.06)
        token.cancel()

    engine = VerificationPipelineEngine(
        checkers={
            "build_checker": build_checker,
            "lint_checker": slow_checker,
            "typecheck_checker": slow_checker,
            "test_checker": slow_checker,
            "security_checker": slow_checker,
        },
        max_parallel_stages=4,
    )

    cancel_task = asyncio.create_task(cancel_soon())
    try:
        result = await engine.run(PipelineRequest(risk_tier=RiskTier.LOW), cancel_token=token)
    finally:
        await cancel_task

    assert result.cancelled
    assert any(stage.skipped and stage.skip_reason == "cancelled" for stage in result.stage_results)


@pytest.mark.asyncio
async def test_early_exit_policy_stops_after_must_failure() -> None:
    async def build_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    async def lint_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    async def type_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    async def security_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    async def test_checker(ctx: CheckerContext) -> dict[str, object]:
        if ctx.stage_id == UNIT_TEST_STAGE_ID:
            return {"status": "fail", "summary": "unit tests failed"}
        return {"status": "pass"}

    engine = VerificationPipelineEngine(
        checkers={
            "build_checker": build_checker,
            "lint_checker": lint_checker,
            "typecheck_checker": type_checker,
            "test_checker": test_checker,
            "security_checker": security_checker,
        },
    )

    result = await engine.run(
        PipelineRequest(
            risk_tier=RiskTier.MEDIUM,
            early_exit_policy=EarlyExitPolicy.STOP_ON_MUST_FAILURE,
        )
    )

    assert result.stopped_early

    stage_by_id = {stage.stage_id: stage for stage in result.stage_results}
    assert stage_by_id[UNIT_TEST_STAGE_ID].status.value == "fail"
    assert stage_by_id[INTEGRATION_STAGE_ID].skipped
    assert stage_by_id[INTEGRATION_STAGE_ID].skip_reason == "stopped_early"


@pytest.mark.asyncio
async def test_incremental_vs_full_selection_hooks_and_artifact_payloads() -> None:
    incremental_calls: list[str] = []
    full_calls: list[str] = []

    def incremental_hook(
        stage: PipelineStage,
        _context: PipelineSelectionContext,
    ) -> tuple[str, ...]:
        incremental_calls.append(stage.stage_id)
        if stage.stage_id == SECURITY_STAGE_ID:
            return ()
        return stage.checker_ids

    def full_hook(stage: PipelineStage, _context: PipelineSelectionContext) -> tuple[str, ...]:
        full_calls.append(stage.stage_id)
        return stage.checker_ids

    async def build_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass", "artifact_paths": ["artifacts/build.json"]}

    async def lint_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass", "artifact_paths": ["artifacts/lint.json", "artifacts/log.txt"]}

    async def type_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    async def test_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    async def security_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    engine = VerificationPipelineEngine(
        checkers={
            "build_checker": build_checker,
            "lint_checker": lint_checker,
            "typecheck_checker": type_checker,
            "test_checker": test_checker,
            "security_checker": security_checker,
        },
    )

    incremental_result = await engine.run(
        PipelineRequest(
            risk_tier=RiskTier.LOW,
            selection_mode=VerificationSelectionMode.INCREMENTAL,
            incremental_selection_hook=incremental_hook,
        )
    )

    assert SECURITY_STAGE_ID in incremental_calls
    stage_by_id = {stage.stage_id: stage for stage in incremental_result.stage_results}
    assert stage_by_id[SECURITY_STAGE_ID].skipped
    assert stage_by_id[SECURITY_STAGE_ID].skip_reason == "no_checkers_selected"

    full_result = await engine.run(
        PipelineRequest(
            risk_tier=RiskTier.LOW,
            selection_mode=VerificationSelectionMode.FULL,
            full_selection_hook=full_hook,
        )
    )
    assert full_result.passed
    assert set(full_calls) == set(full_result.selected_stage_ids)

    lint_stage = next(
        stage for stage in full_result.stage_results if stage.stage_id == LINT_STAGE_ID
    )
    lint_checker_result = lint_stage.checker_results[0]
    assert lint_checker_result.artifact_paths == (
        "artifacts/lint.json",
        "artifacts/log.txt",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("risk_tier", [RiskTier.HIGH, RiskTier.CRITICAL])
async def test_high_and_critical_risk_cannot_silently_skip_adversarial_stage(
    risk_tier: RiskTier,
) -> None:
    def deselect_adversarial(
        stage: PipelineStage,
        _context: PipelineSelectionContext,
    ) -> tuple[str, ...]:
        if stage.stage_id == ADVERSARIAL_STAGE_ID:
            return ()
        return stage.checker_ids

    async def pass_checker(_: CheckerContext) -> dict[str, object]:
        return {"status": "pass"}

    engine = VerificationPipelineEngine(
        checkers={
            "build_checker": pass_checker,
            "lint_checker": pass_checker,
            "typecheck_checker": pass_checker,
            "test_checker": pass_checker,
            "security_checker": pass_checker,
            "performance_checker": pass_checker,
            "adversarial/test_generator": pass_checker,
        },
        incremental_selection_hook=deselect_adversarial,
    )

    result = await engine.run(
        PipelineRequest(
            risk_tier=risk_tier,
            selection_mode=VerificationSelectionMode.INCREMENTAL,
        )
    )

    stage_by_id = {stage.stage_id: stage for stage in result.stage_results}
    adversarial_stage = stage_by_id[ADVERSARIAL_STAGE_ID]
    assert adversarial_stage.skipped
    assert adversarial_stage.skip_reason == "no_checkers_selected"
    assert adversarial_stage.must_failure
    assert ADVERSARIAL_STAGE_ID in result.must_failure_stage_ids
    assert not result.passed


def test_risk_tier_stage_mapping_matches_verification_pipeline_spec() -> None:
    engine = VerificationPipelineEngine(checkers={})

    low = engine.required_stage_ids_for_risk_tier(RiskTier.LOW)
    medium = engine.required_stage_ids_for_risk_tier(RiskTier.MEDIUM)
    high = engine.required_stage_ids_for_risk_tier(RiskTier.HIGH)
    critical = engine.required_stage_ids_for_risk_tier(RiskTier.CRITICAL)

    assert low == DEFAULT_STAGE_IDS_IN_ORDER[:5]
    assert medium == DEFAULT_STAGE_IDS_IN_ORDER[:6]
    assert high == DEFAULT_STAGE_IDS_IN_ORDER[:7] + (ADVERSARIAL_STAGE_ID,)
    assert critical == DEFAULT_STAGE_IDS_IN_ORDER[:8]

    assert not engine.adversarial_required_for_risk_tier(RiskTier.LOW)
    assert not engine.adversarial_required_for_risk_tier(RiskTier.MEDIUM)
    assert engine.adversarial_required_for_risk_tier(RiskTier.HIGH)
    assert engine.adversarial_required_for_risk_tier(RiskTier.CRITICAL)


def test_pipeline_result_stage_order_is_deterministic() -> None:
    engine = VerificationPipelineEngine(checkers={})
    plan = engine.plan(PipelineRequest(risk_tier=RiskTier.MEDIUM))

    expected_order: Final[tuple[str, ...]] = (
        BUILD_STAGE_ID,
        LINT_STAGE_ID,
        TYPECHECK_STAGE_ID,
        UNIT_TEST_STAGE_ID,
        SECURITY_STAGE_ID,
        INTEGRATION_STAGE_ID,
    )
    assert plan.selected_stage_ids == expected_order
