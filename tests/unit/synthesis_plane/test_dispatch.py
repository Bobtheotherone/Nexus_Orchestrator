"""
nexus-orchestrator — test skeleton

File: tests/unit/synthesis_plane/test_dispatch.py
Last updated: 2026-02-11

Purpose
- Validate dispatch controller behavior with mocked providers.

What this test file should cover
- Rate limiting/backoff behavior.
- Budget enforcement (tokens/cost/iterations).
- Provider routing decisions.

Functional requirements
- No real network calls.

Non-functional requirements
- Deterministic with seeded randomness.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import Attempt, AttemptResult
from nexus_orchestrator.synthesis_plane.dispatch import (
    BudgetExceededError,
    DispatchBudget,
    DispatchController,
    DispatchFailedError,
    DispatchRequest,
    ProviderBinding,
    ProviderCallError,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)
from nexus_orchestrator.utils.concurrency import CancellationToken

if TYPE_CHECKING:
    from collections.abc import Callable

    from nexus_orchestrator.persistence.repositories import ProviderCallRecord


def _request(
    index: int,
    *,
    route_key: str = "unit",
    provider_allowlist: tuple[str, ...] | None = None,
    idempotency_key: str | None = None,
) -> DispatchRequest:
    return DispatchRequest(
        run_id=f"run-{index}",
        work_item_id=f"wi-{index}",
        role="implementer",
        prompt=f"prompt-{index}",
        routing_key=route_key,
        provider_allowlist=provider_allowlist,
        idempotency_key=idempotency_key,
    )


def _response(content: str = "ok", *, tokens: int = 1, cost_usd: float = 0.01) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        usage=ProviderUsage(tokens=tokens, cost_usd=cost_usd, latency_ms=1),
        model="mock-model",
        request_id="req-1",
    )


@dataclass(slots=True)
class FakeClock:
    current: float = 0.0
    sleep_calls: list[float] = field(default_factory=list)

    def now(self) -> float:
        return self.current

    async def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current += seconds
        await asyncio.sleep(0)


@dataclass(slots=True)
class BlockingProvider:
    release_event: asyncio.Event
    started_event: asyncio.Event
    threshold: int
    in_flight: int = 0
    max_in_flight: int = 0
    calls_started: int = 0

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        _ = request
        self.calls_started += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.calls_started >= self.threshold:
            self.started_event.set()
        try:
            await self.release_event.wait()
            return _response(content="done")
        finally:
            self.in_flight -= 1


@dataclass(slots=True)
class ImmediateProvider:
    call_count: int = 0

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        _ = request
        self.call_count += 1
        return _response(content="immediate")


@dataclass(slots=True)
class ScriptedProvider:
    outcomes: deque[ProviderResponse | Exception]
    requests: list[ProviderRequest] = field(default_factory=list)

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if not self.outcomes:
            raise RuntimeError("scripted outcomes exhausted")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass(slots=True)
class NeverEndingProvider:
    started: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        _ = request
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return _response(content="unreachable")


@dataclass(slots=True)
class FailingProvider:
    name: str
    call_count: int = 0

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        _ = request
        self.call_count += 1
        raise ProviderCallError(provider=self.name, message="hard failure", retryable=False)


@dataclass(slots=True)
class RecordingAttemptRepo:
    attempts: list[Attempt] = field(default_factory=list)

    def save(self, attempt: Attempt) -> Attempt:
        self.attempts.append(attempt)
        return attempt


@dataclass(slots=True)
class RecordingProviderCallRepo:
    records: list[ProviderCallRecord] = field(default_factory=list)

    def save(self, record: ProviderCallRecord) -> ProviderCallRecord:
        self.records.append(record)
        return record


def _randbytes(seed: int) -> Callable[[int], bytes]:
    value = seed % 255

    def factory(size: int) -> bytes:
        return bytes([value]) * size

    return factory


def _valid_ids(seed: int) -> tuple[str, str]:
    timestamp = 1_750_000_000_000 + seed
    run_id = ids.generate_run_id(timestamp_ms=timestamp, randbytes=_randbytes(seed + 1))
    work_item_id = ids.generate_work_item_id(
        timestamp_ms=timestamp + 1, randbytes=_randbytes(seed + 2)
    )
    return run_id, work_item_id


async def test_dispatch_enforces_per_provider_concurrency_cap() -> None:
    release_event = asyncio.Event()
    started_event = asyncio.Event()
    provider = BlockingProvider(
        release_event=release_event,
        started_event=started_event,
        threshold=2,
    )
    controller = DispatchController(
        [
            ProviderBinding(
                name="mock",
                model="mock-model",
                provider=provider,
                max_concurrency=2,
            )
        ]
    )

    tasks = [asyncio.create_task(controller.dispatch(_request(index))) for index in range(4)]
    await asyncio.wait_for(started_event.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert provider.max_in_flight == 2
    assert provider.calls_started == 2

    release_event.set()
    results = await asyncio.gather(*tasks)
    assert len(results) == 4
    assert provider.max_in_flight == 2


async def test_dispatch_waits_for_deterministic_rate_limit_window() -> None:
    fake_clock = FakeClock()
    provider = ImmediateProvider()
    controller = DispatchController(
        [
            ProviderBinding(
                name="mock",
                model="mock-model",
                provider=provider,
                rate_limit_calls=1,
                rate_limit_period_seconds=5.0,
            )
        ],
        clock=fake_clock.now,
        sleep=fake_clock.sleep,
    )

    await controller.dispatch(_request(1))
    await controller.dispatch(_request(2))

    assert provider.call_count == 2
    assert fake_clock.sleep_calls == [5.0]


async def test_dispatch_retries_with_backoff_and_stable_idempotency_key() -> None:
    fake_clock = FakeClock()
    provider = ScriptedProvider(
        outcomes=deque(
            [
                ProviderCallError(provider="mock", message="retry one", retryable=True),
                ProviderCallError(provider="mock", message="retry two", retryable=True),
                _response(content="final", tokens=7, cost_usd=0.03),
            ]
        )
    )
    controller = DispatchController(
        [
            ProviderBinding(
                name="mock",
                model="mock-model",
                provider=provider,
                max_retries=2,
                retry_backoff_seconds=1.0,
                retry_backoff_multiplier=2.0,
            )
        ],
        clock=fake_clock.now,
        sleep=fake_clock.sleep,
    )

    result = await controller.dispatch(_request(3, idempotency_key="idem-stable"))

    assert result.content == "final"
    assert result.attempts == 3
    assert fake_clock.sleep_calls == [1.0, 2.0]
    assert [request.idempotency_key for request in provider.requests] == [
        "idem-stable",
        "idem-stable",
        "idem-stable",
    ]


async def test_dispatch_short_circuits_when_budget_disallows_call() -> None:
    provider = ImmediateProvider()
    controller = DispatchController(
        [ProviderBinding(name="mock", model="mock-model", provider=provider)]
    )
    request = DispatchRequest(
        run_id="run-budget",
        work_item_id="wi-budget",
        role="implementer",
        prompt="budget test",
        budget=DispatchBudget(max_cost_usd=0.0, reserved_cost_usd_per_call=0.25),
    )

    with pytest.raises(BudgetExceededError):
        await controller.dispatch(request)
    assert provider.call_count == 0


async def test_dispatch_cancels_in_flight_provider_call() -> None:
    provider = NeverEndingProvider()
    controller = DispatchController(
        [ProviderBinding(name="mock", model="mock-model", provider=provider)]
    )
    token = CancellationToken()

    task = asyncio.create_task(controller.dispatch(_request(4), cancel_token=token))
    await asyncio.wait_for(provider.started.wait(), timeout=1.0)

    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert provider.cancelled.is_set()


async def test_dispatch_adaptive_routing_switches_provider_after_failure() -> None:
    alpha = FailingProvider(name="alpha")
    beta = ImmediateProvider()
    controller = DispatchController(
        [
            ProviderBinding(name="alpha", model="alpha-model", provider=alpha),
            ProviderBinding(name="beta", model="beta-model", provider=beta),
        ]
    )

    with pytest.raises(DispatchFailedError):
        await controller.dispatch(_request(5, route_key="routing", provider_allowlist=("alpha",)))

    result = await controller.dispatch(
        _request(6, route_key="routing", provider_allowlist=("alpha", "beta"))
    )

    assert result.provider == "beta"
    assert alpha.call_count == 1
    assert beta.call_count == 1


async def test_dispatch_persists_attempts_and_provider_calls_when_repos_exist() -> None:
    secret = "sk-FAKEOPENAIKEY1234567890ABCDE"
    provider = ScriptedProvider(
        outcomes=deque(
            [
                _response(content=f"response contains {secret}", tokens=9, cost_usd=0.04),
                _response(content=f"second response {secret}", tokens=3, cost_usd=0.02),
            ]
        )
    )
    attempt_repo = RecordingAttemptRepo()
    provider_call_repo = RecordingProviderCallRepo()
    controller = DispatchController(
        [ProviderBinding(name="mock", model="mock-model", provider=provider)],
        transcript_retention=1,
        attempt_repo=attempt_repo,
        provider_call_repo=provider_call_repo,
    )

    run_id, first_work_item_id = _valid_ids(10)
    _, second_work_item_id = _valid_ids(20)

    await controller.dispatch(
        DispatchRequest(
            run_id=run_id,
            work_item_id=first_work_item_id,
            role="implementer",
            prompt=f"first prompt {secret}",
        )
    )
    await controller.dispatch(
        DispatchRequest(
            run_id=run_id,
            work_item_id=second_work_item_id,
            role="implementer",
            prompt=f"second prompt {secret}",
        )
    )

    assert len(attempt_repo.attempts) == 2
    assert all(attempt.result is AttemptResult.SUCCESS for attempt in attempt_repo.attempts)
    assert len(provider_call_repo.records) == 2
    assert provider_call_repo.records[0].attempt_id == attempt_repo.attempts[0].id
    assert provider_call_repo.records[1].attempt_id == attempt_repo.attempts[1].id

    transcripts = controller.transcripts()
    assert len(transcripts) == 1
    transcript = transcripts[0]
    assert transcript.work_item_id == second_work_item_id
    assert secret not in transcript.prompt
    assert transcript.response is not None
    assert secret not in transcript.response
    assert "***REDACTED***" in transcript.prompt
    assert "***REDACTED***" in transcript.response
