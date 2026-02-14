"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/dispatch.py
Last updated: 2026-02-11

Purpose
- Dispatch Controller: manages concurrent agent calls, retries, backoff, and routing to providers.

What should be included in this file
- Concurrency control and rate limiting per provider.
- Adaptive routing based on success/failure history.
- Retry policies and idempotency keys for provider calls.
- Transcript storage policy (redaction and retention).

Functional requirements
- Must support mocked providers for offline tests.
- Must support cancellation (stop in-flight work when run is paused).

Non-functional requirements
- Must not overwhelm local machine (prompt assembly may be heavy).
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar, runtime_checkable

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import Attempt, AttemptResult
from nexus_orchestrator.persistence.repositories import ProviderCallRecord
from nexus_orchestrator.security.redaction import redact_text
from nexus_orchestrator.synthesis_plane.providers.base import (
    ProviderError,
    ProviderRequest,
    ProviderResponse,
    ProviderServiceError,
    ProviderUsage,
)
from nexus_orchestrator.synthesis_plane.providers.base import (
    ProviderProtocol as BaseProviderProtocol,
)
from nexus_orchestrator.utils.hashing import sha256_text

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

if TYPE_CHECKING:
    from nexus_orchestrator.utils.concurrency import CancellationToken

Clock = Callable[[], float]
SleepFn = Callable[[float], Awaitable[None]]
_TaskT = TypeVar("_TaskT")

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class DispatchError(RuntimeError):
    """Base error for dispatch controller failures."""


class BudgetExceededError(DispatchError):
    """Raised when the budget envelope disallows another provider call."""


class DispatchFailedError(DispatchError):
    """Raised when all candidate providers fail."""


class ProviderCallError(ProviderServiceError):
    """Deprecated compatibility shim for pre-canonical dispatch errors."""

    def __init__(
        self,
        *,
        provider: str,
        message: str,
        retryable: bool,
        code: str | None = None,
    ) -> None:
        warnings.warn(
            "ProviderCallError is deprecated; raise ProviderError subclasses from "
            "nexus_orchestrator.synthesis_plane.providers.base",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            provider=provider,
            detail=message,
            retryable=retryable,
            provider_code=code,
        )


@dataclass(slots=True)
class DispatchBudget:
    """Simple preflight budget envelope for one dispatch call."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_attempts: int = 8
    reserved_tokens_per_call: int = 0
    reserved_cost_usd_per_call: float = 0.0

    def __post_init__(self) -> None:
        if self.max_tokens is not None and self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("max_cost_usd must be >= 0")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")
        if self.reserved_tokens_per_call < 0:
            raise ValueError("reserved_tokens_per_call must be >= 0")
        if self.reserved_cost_usd_per_call < 0:
            raise ValueError("reserved_cost_usd_per_call must be >= 0")


@dataclass(slots=True)
class DispatchRequest:
    """Dispatch request for a single work-item/provider attempt cycle."""

    run_id: str
    work_item_id: str
    role: str
    prompt: str
    routing_key: str | None = None
    provider_allowlist: tuple[str, ...] | None = None
    budget: DispatchBudget = field(default_factory=DispatchBudget)
    idempotency_key: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_id = _validate_non_empty_str(self.run_id, "run_id")
        self.work_item_id = _validate_non_empty_str(self.work_item_id, "work_item_id")
        self.role = _validate_non_empty_str(self.role, "role")
        self.prompt = _validate_non_empty_str(self.prompt, "prompt", strip=False)
        if self.routing_key is not None:
            self.routing_key = _validate_non_empty_str(self.routing_key, "routing_key")
        if self.provider_allowlist is not None:
            if not self.provider_allowlist:
                raise ValueError("provider_allowlist cannot be empty")
            self.provider_allowlist = tuple(
                _validate_non_empty_str(name, "provider_allowlist")
                for name in self.provider_allowlist
            )
        if self.idempotency_key is not None:
            self.idempotency_key = _validate_non_empty_str(self.idempotency_key, "idempotency_key")
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class DispatchResult:
    """Successful dispatch outcome."""

    provider: str
    model: str
    content: str
    idempotency_key: str
    attempts: int
    tokens_used: int
    cost_usd: float
    latency_ms: int
    request_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        self.provider = _validate_non_empty_str(self.provider, "provider")
        self.model = _validate_non_empty_str(self.model, "model")
        self.content = _validate_non_empty_str(self.content, "content", strip=False)
        self.idempotency_key = _validate_non_empty_str(self.idempotency_key, "idempotency_key")
        if self.attempts <= 0:
            raise ValueError("attempts must be > 0")
        if self.tokens_used < 0:
            raise ValueError("tokens_used must be >= 0")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.request_id is not None:
            self.request_id = _validate_non_empty_str(self.request_id, "request_id")


@dataclass(slots=True, frozen=True)
class TranscriptEntry:
    """Redacted transcript entry retained by the controller."""

    run_id: str
    work_item_id: str
    provider: str
    attempt_number: int
    idempotency_key: str
    prompt: str
    response: str | None
    error: str | None
    created_at: datetime


@dataclass(slots=True)
class ProviderBinding:
    """Provider runtime configuration for dispatch routing/execution."""

    name: str
    model: str
    provider: ProviderProtocol | LegacyGenerateProviderProtocol
    max_concurrency: int = 1
    rate_limit_calls: int = 0
    rate_limit_period_seconds: float = 1.0
    max_retries: int = 0
    retry_backoff_seconds: float = 0.0
    retry_backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        self.name = _validate_non_empty_str(self.name, "name")
        self.model = _validate_non_empty_str(self.model, "model")
        if not isinstance(self.provider, ProviderProtocol) and not isinstance(
            self.provider, LegacyGenerateProviderProtocol
        ):
            raise TypeError("provider must implement send(request) or legacy generate(request)")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        if self.rate_limit_calls < 0:
            raise ValueError("rate_limit_calls must be >= 0")
        if self.rate_limit_period_seconds < 0:
            raise ValueError("rate_limit_period_seconds must be >= 0")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        if self.retry_backoff_multiplier < 1.0:
            raise ValueError("retry_backoff_multiplier must be >= 1.0")


ProviderProtocol = BaseProviderProtocol


@runtime_checkable
class LegacyGenerateProviderProtocol(Protocol):
    """Deprecated provider protocol maintained as a compatibility shim."""

    async def generate(self, request: ProviderRequest) -> ProviderResponse: ...


@runtime_checkable
class AttemptRepoSaveProtocol(Protocol):
    def save(self, attempt: Attempt) -> Attempt: ...


@runtime_checkable
class AttemptRepoAddProtocol(Protocol):
    def add(self, attempt: Attempt) -> Attempt: ...


AttemptRepoLike = AttemptRepoSaveProtocol | AttemptRepoAddProtocol


@runtime_checkable
class ProviderCallRepoSaveProtocol(Protocol):
    def save(self, record: ProviderCallRecord) -> ProviderCallRecord: ...


@runtime_checkable
class ProviderCallRepoAddProtocol(Protocol):
    def add(self, record: ProviderCallRecord) -> ProviderCallRecord: ...


ProviderCallRepoLike = ProviderCallRepoSaveProtocol | ProviderCallRepoAddProtocol


@runtime_checkable
class DispatchPersistenceProtocol(Protocol):
    """Explicit persistence wiring used by dispatch."""

    @property
    def persist_in_progress(self) -> bool: ...

    def save_attempt(self, attempt: Attempt) -> Attempt: ...

    def save_provider_call(self, record: ProviderCallRecord) -> ProviderCallRecord: ...


@dataclass(slots=True)
class RepositoryDispatchPersistence:
    """Repository-backed dispatch persistence wiring."""

    attempt_repo: AttemptRepoLike
    provider_call_repo: ProviderCallRepoLike
    persist_in_progress: bool = False

    def __post_init__(self) -> None:
        self.persist_in_progress = bool(self.persist_in_progress)

    def save_attempt(self, attempt: Attempt) -> Attempt:
        return _save_attempt(self.attempt_repo, attempt)

    def save_provider_call(self, record: ProviderCallRecord) -> ProviderCallRecord:
        return _save_provider_call(self.provider_call_repo, record)


@dataclass(slots=True)
class _RoutingStats:
    successes: int = 0
    failures: int = 0

    @property
    def attempts(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.attempts == 0:
            return 0.5
        return self.successes / self.attempts


@dataclass(slots=True)
class _ProviderRuntime:
    binding: ProviderBinding
    semaphore: asyncio.Semaphore
    rate_limiter: DeterministicRateLimiter


class DeterministicRateLimiter:
    """Deterministic sliding-window rate limiter with injected clock/sleep."""

    def __init__(
        self,
        *,
        max_calls: int,
        period_seconds: float,
        clock: Clock,
    ) -> None:
        if max_calls < 0:
            raise ValueError("max_calls must be >= 0")
        if period_seconds < 0:
            raise ValueError("period_seconds must be >= 0")
        self._max_calls = max_calls
        self._period_seconds = period_seconds
        self._clock = clock
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self, *, sleep: SleepFn, cancel_token: CancellationToken | None) -> None:
        if self._max_calls == 0 or self._period_seconds == 0:
            return

        while True:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()

            wait_seconds = 0.0
            async with self._lock:
                now = self._clock()
                cutoff = now - self._period_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return

                wait_seconds = (self._timestamps[0] + self._period_seconds) - now

            await _sleep_with_cancellation(
                max(wait_seconds, 0.0),
                sleep=sleep,
                cancel_token=cancel_token,
            )


class DispatchController:
    """Deterministic offline dispatch controller with routing/retry controls."""

    def __init__(
        self,
        providers: Sequence[ProviderBinding],
        *,
        clock: Clock = time.monotonic,
        sleep: SleepFn = asyncio.sleep,
        transcript_retention: int = 256,
        persistence: DispatchPersistenceProtocol | None = None,
        attempt_repo: AttemptRepoLike | None = None,
        provider_call_repo: ProviderCallRepoLike | None = None,
    ) -> None:
        if not providers:
            raise ValueError("providers cannot be empty")
        if transcript_retention < 0:
            raise ValueError("transcript_retention must be >= 0")

        self._clock = clock
        self._sleep = sleep
        self._clock_anchor = clock()
        self._datetime_anchor = datetime.now(tz=UTC)

        self._provider_order: dict[str, int] = {}
        self._providers: dict[str, _ProviderRuntime] = {}
        for index, binding in enumerate(providers):
            if binding.name in self._providers:
                raise ValueError(f"duplicate provider name: {binding.name}")
            self._provider_order[binding.name] = index
            self._providers[binding.name] = _ProviderRuntime(
                binding=binding,
                semaphore=asyncio.Semaphore(binding.max_concurrency),
                rate_limiter=DeterministicRateLimiter(
                    max_calls=binding.rate_limit_calls,
                    period_seconds=binding.rate_limit_period_seconds,
                    clock=clock,
                ),
            )

        self._route_stats: dict[str, dict[str, _RoutingStats]] = {}
        self._transcripts: deque[TranscriptEntry] = deque(maxlen=transcript_retention)
        self._persistence = _resolve_dispatch_persistence(
            persistence=persistence,
            attempt_repo=attempt_repo,
            provider_call_repo=provider_call_repo,
        )
        self._attempt_sequence = 0

    async def dispatch(
        self,
        request: DispatchRequest,
        *,
        cancel_token: CancellationToken | None = None,
    ) -> DispatchResult:
        token = cancel_token
        if token is not None:
            token.raise_if_cancelled()

        route_key = request.routing_key or request.role
        provider_candidates = self._ranked_provider_candidates(request, route_key=route_key)
        dispatch_idempotency_key = _derive_dispatch_idempotency_key(request)
        attempt_id = self._build_attempt_id(
            request=request,
            idempotency_key=dispatch_idempotency_key,
        )

        started_at = self._clock_to_datetime(self._clock())
        attempts_made = 0
        spent_tokens = 0
        spent_cost = 0.0
        selected_provider = "unassigned"
        selected_model = "unassigned"
        final_result = AttemptResult.ERROR
        final_feedback: str | None = None
        last_error: Exception | None = None

        if self._persistence is not None and self._persistence.persist_in_progress:
            self._persist_attempt(
                attempt_id=attempt_id,
                request=request,
                provider=selected_provider,
                model=selected_model,
                attempts=1,
                tokens_used=0,
                cost_usd=0.0,
                result=AttemptResult.ERROR,
                started_at=started_at,
                finished_at=started_at,
                feedback="dispatch_in_progress",
            )

        try:
            for provider_name in provider_candidates:
                runtime = self._providers[provider_name]
                selected_provider = provider_name
                selected_model = runtime.binding.model
                provider_idempotency_key = _derive_provider_idempotency_key(
                    request=request,
                    dispatch_idempotency_key=dispatch_idempotency_key,
                    provider_name=provider_name,
                    model=runtime.binding.model,
                )
                retry_index = 0

                while True:
                    next_attempt = attempts_made + 1
                    self._enforce_budget_preflight(
                        budget=request.budget,
                        attempts_so_far=attempts_made,
                        spent_tokens=spent_tokens,
                        spent_cost=spent_cost,
                    )

                    attempts_made = next_attempt
                    call_started = self._clock()
                    call_started_at = self._clock_to_datetime(call_started)
                    response: ProviderResponse | None = None
                    error_text: str | None = None
                    retryable_error = False

                    try:
                        await runtime.rate_limiter.acquire(sleep=self._sleep, cancel_token=token)
                        await _acquire_semaphore_with_cancellation(
                            runtime.semaphore,
                            cancel_token=token,
                        )
                        try:
                            provider_metadata = dict(request.metadata)
                            provider_metadata.setdefault("run_id", request.run_id)
                            provider_metadata.setdefault("work_item_id", request.work_item_id)
                            provider_request = ProviderRequest(
                                model=runtime.binding.model,
                                role_id=request.role,
                                user_prompt=request.prompt,
                                idempotency_key=provider_idempotency_key,
                                metadata=provider_metadata,
                            )
                            response = await _await_with_cancellation(
                                _call_provider(
                                    provider=runtime.binding.provider,
                                    request=provider_request,
                                ),
                                cancel_token=token,
                            )
                        finally:
                            runtime.semaphore.release()

                    except asyncio.CancelledError:
                        final_result = AttemptResult.TIMEOUT
                        final_feedback = "dispatch cancelled"
                        raise
                    except ProviderError as exc:
                        last_error = exc
                        error_text = str(exc)
                        retryable_error = exc.retryable
                        self._mark_route_failure(route_key=route_key, provider_name=provider_name)
                        final_feedback = _truncate_feedback(error_text)
                    except Exception as exc:  # noqa: BLE001
                        wrapped = ProviderServiceError(
                            provider=provider_name,
                            detail=str(exc),
                            retryable=False,
                        )
                        last_error = wrapped
                        error_text = str(wrapped)
                        self._mark_route_failure(route_key=route_key, provider_name=provider_name)
                        final_feedback = _truncate_feedback(error_text)

                    call_finished = self._clock()
                    latency_ms = int(round(max(0.0, call_finished - call_started) * 1000))

                    if response is not None:
                        usage_tokens = _usage_total_tokens(response.usage)
                        usage_cost = _usage_cost_usd(response.usage)
                        usage_latency_ms = (
                            response.usage.latency_ms
                            if response.usage.latency_ms is not None
                            else latency_ms
                        )
                        spent_tokens += usage_tokens
                        spent_cost += usage_cost
                        final_result = AttemptResult.SUCCESS
                        final_feedback = None
                        selected_model = response.model
                        self._mark_route_success(route_key=route_key, provider_name=provider_name)
                        self._record_transcript(
                            run_id=request.run_id,
                            work_item_id=request.work_item_id,
                            provider_name=provider_name,
                            attempt_number=attempts_made,
                            idempotency_key=provider_idempotency_key,
                            prompt=request.prompt,
                            response=response.raw_text,
                            error=None,
                            created_at=call_started_at,
                        )
                        self._persist_provider_call(
                            attempt_id=attempt_id,
                            provider_name=provider_name,
                            model=selected_model,
                            attempts_made=attempts_made,
                            idempotency_key=provider_idempotency_key,
                            tokens=usage_tokens,
                            cost_usd=usage_cost,
                            latency_ms=usage_latency_ms,
                            request_id=response.request_id,
                            error=None,
                            created_at=call_started_at,
                        )
                        return DispatchResult(
                            provider=provider_name,
                            model=selected_model,
                            content=response.raw_text,
                            idempotency_key=provider_idempotency_key,
                            attempts=attempts_made,
                            tokens_used=spent_tokens,
                            cost_usd=spent_cost,
                            latency_ms=usage_latency_ms,
                            request_id=response.request_id,
                            attempt_id=attempt_id,
                        )

                    self._record_transcript(
                        run_id=request.run_id,
                        work_item_id=request.work_item_id,
                        provider_name=provider_name,
                        attempt_number=attempts_made,
                        idempotency_key=provider_idempotency_key,
                        prompt=request.prompt,
                        response=None,
                        error=error_text,
                        created_at=call_started_at,
                    )
                    self._persist_provider_call(
                        attempt_id=attempt_id,
                        provider_name=provider_name,
                        model=runtime.binding.model,
                        attempts_made=attempts_made,
                        idempotency_key=provider_idempotency_key,
                        tokens=0,
                        cost_usd=0.0,
                        latency_ms=latency_ms,
                        request_id=None,
                        error=error_text,
                        created_at=call_started_at,
                    )

                    if retryable_error and retry_index < runtime.binding.max_retries:
                        retry_index += 1
                        backoff_seconds = runtime.binding.retry_backoff_seconds * (
                            runtime.binding.retry_backoff_multiplier ** (retry_index - 1)
                        )
                        await _sleep_with_cancellation(
                            backoff_seconds,
                            sleep=self._sleep,
                            cancel_token=token,
                        )
                        continue

                    break

            final_result = AttemptResult.ERROR
            if last_error is None:
                raise DispatchFailedError("dispatch failed: no providers could be selected")
            raise DispatchFailedError(
                f"dispatch failed after provider attempts: {', '.join(provider_candidates)}"
            ) from last_error
        except BudgetExceededError as exc:
            final_result = AttemptResult.ERROR
            final_feedback = _truncate_feedback(str(exc))
            raise
        finally:
            self._persist_attempt(
                attempt_id=attempt_id,
                request=request,
                provider=selected_provider,
                model=selected_model,
                attempts=attempts_made,
                tokens_used=spent_tokens,
                cost_usd=spent_cost,
                result=final_result,
                started_at=started_at,
                finished_at=self._clock_to_datetime(self._clock()),
                feedback=final_feedback,
            )

    def transcripts(self, *, work_item_id: str | None = None) -> tuple[TranscriptEntry, ...]:
        """Return retained redacted transcript entries."""
        if work_item_id is None:
            return tuple(self._transcripts)
        normalized = _validate_non_empty_str(work_item_id, "work_item_id")
        return tuple(item for item in self._transcripts if item.work_item_id == normalized)

    def routing_snapshot(self, route_key: str) -> dict[str, tuple[int, int]]:
        """Return provider success/failure counts for a route key."""
        normalized = _validate_non_empty_str(route_key, "route_key")
        stats_by_provider = self._route_stats.get(normalized, {})
        return {
            provider: (stats.successes, stats.failures)
            for provider, stats in sorted(stats_by_provider.items())
        }

    def _ranked_provider_candidates(
        self,
        request: DispatchRequest,
        *,
        route_key: str,
    ) -> tuple[str, ...]:
        if request.provider_allowlist is None:
            candidates = tuple(self._providers.keys())
        else:
            missing = sorted(set(request.provider_allowlist) - set(self._providers))
            if missing:
                raise ValueError(f"unknown provider(s): {', '.join(missing)}")
            candidates = request.provider_allowlist

        if not candidates:
            raise ValueError("no candidate providers")

        stats_by_provider = self._route_stats.setdefault(route_key, {})
        sorted_candidates = sorted(
            candidates,
            key=lambda provider_name: self._routing_sort_key(
                provider_name=provider_name,
                stats=stats_by_provider.get(provider_name),
            ),
        )
        return tuple(sorted_candidates)

    def _routing_sort_key(
        self,
        *,
        provider_name: str,
        stats: _RoutingStats | None,
    ) -> tuple[float, int, int, int, str]:
        item = stats or _RoutingStats()
        return (
            -item.success_rate,
            -item.successes,
            item.failures,
            self._provider_order[provider_name],
            provider_name,
        )

    def _mark_route_success(self, *, route_key: str, provider_name: str) -> None:
        stats_by_provider = self._route_stats.setdefault(route_key, {})
        stats = stats_by_provider.setdefault(provider_name, _RoutingStats())
        stats.successes += 1

    def _mark_route_failure(self, *, route_key: str, provider_name: str) -> None:
        stats_by_provider = self._route_stats.setdefault(route_key, {})
        stats = stats_by_provider.setdefault(provider_name, _RoutingStats())
        stats.failures += 1

    def _enforce_budget_preflight(
        self,
        *,
        budget: DispatchBudget,
        attempts_so_far: int,
        spent_tokens: int,
        spent_cost: float,
    ) -> None:
        next_attempt = attempts_so_far + 1
        if next_attempt > budget.max_attempts:
            raise BudgetExceededError(
                f"attempt budget exceeded: next={next_attempt} max={budget.max_attempts}"
            )

        projected_tokens = spent_tokens + budget.reserved_tokens_per_call
        if budget.max_tokens is not None and projected_tokens > budget.max_tokens:
            raise BudgetExceededError(
                f"token budget exceeded: projected={projected_tokens} max={budget.max_tokens}"
            )

        projected_cost = spent_cost + budget.reserved_cost_usd_per_call
        if budget.max_cost_usd is not None and projected_cost > budget.max_cost_usd:
            raise BudgetExceededError(
                f"cost budget exceeded: projected={projected_cost:.6f} max={budget.max_cost_usd:.6f}"
            )

    def _record_transcript(
        self,
        *,
        run_id: str,
        work_item_id: str,
        provider_name: str,
        attempt_number: int,
        idempotency_key: str,
        prompt: str,
        response: str | None,
        error: str | None,
        created_at: datetime,
    ) -> None:
        self._transcripts.append(
            TranscriptEntry(
                run_id=run_id,
                work_item_id=work_item_id,
                provider=provider_name,
                attempt_number=attempt_number,
                idempotency_key=idempotency_key,
                prompt=redact_text(prompt),
                response=redact_text(response) if response is not None else None,
                error=redact_text(error) if error is not None else None,
                created_at=created_at,
            )
        )

    def _persist_attempt(
        self,
        *,
        attempt_id: str,
        request: DispatchRequest,
        provider: str,
        model: str,
        attempts: int,
        tokens_used: int,
        cost_usd: float,
        result: AttemptResult,
        started_at: datetime,
        finished_at: datetime,
        feedback: str | None,
    ) -> None:
        if self._persistence is None:
            return

        attempt = Attempt(
            id=attempt_id,
            work_item_id=request.work_item_id,
            run_id=request.run_id,
            iteration=max(1, attempts),
            provider=provider,
            model=model,
            role=request.role,
            prompt_hash=sha256_text(request.prompt),
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            result=result,
            created_at=started_at,
            finished_at=finished_at,
            feedback=feedback,
        )
        self._persistence.save_attempt(attempt)

    def _persist_provider_call(
        self,
        *,
        attempt_id: str,
        provider_name: str,
        model: str,
        attempts_made: int,
        idempotency_key: str,
        tokens: int,
        cost_usd: float,
        latency_ms: int,
        request_id: str | None,
        error: str | None,
        created_at: datetime,
    ) -> None:
        if self._persistence is None:
            return
        record_id = _provider_call_record_id(
            attempt_id=attempt_id,
            provider_name=provider_name,
            attempts_made=attempts_made,
        )
        metadata: dict[str, JSONValue] = {
            "attempt_number": attempts_made,
            "idempotency_key": idempotency_key,
            "model": model,
        }
        record = ProviderCallRecord(
            id=record_id,
            attempt_id=attempt_id,
            provider=provider_name,
            tokens=tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            created_at=created_at,
            model=model,
            request_id=request_id,
            error=_truncate_feedback(error) if error is not None else None,
            metadata=metadata,
        )
        self._persistence.save_provider_call(record)

    def _build_attempt_id(self, *, request: DispatchRequest, idempotency_key: str) -> str:
        self._attempt_sequence += 1
        sequence = self._attempt_sequence
        timestamp_ms = max(0, int(round(self._clock() * 1000)))
        seed_material = (
            f"{idempotency_key}|{request.run_id}|{request.work_item_id}|{sequence}".encode()
        )
        digest = sha256(seed_material).digest()

        def randbytes(size: int) -> bytes:
            repeats = (size + len(digest) - 1) // len(digest)
            return (digest * repeats)[:size]

        return ids.generate_attempt_id(timestamp_ms=timestamp_ms, randbytes=randbytes)

    def _clock_to_datetime(self, clock_value: float) -> datetime:
        delta = clock_value - self._clock_anchor
        return self._datetime_anchor + timedelta(seconds=delta)


def _save_attempt(repo: AttemptRepoLike, attempt: Attempt) -> Attempt:
    if isinstance(repo, AttemptRepoSaveProtocol):
        return repo.save(attempt)
    if isinstance(repo, AttemptRepoAddProtocol):
        return repo.add(attempt)
    raise TypeError("attempt_repo must implement save(attempt) or add(attempt)")


def _save_provider_call(
    repo: ProviderCallRepoLike, record: ProviderCallRecord
) -> ProviderCallRecord:
    if isinstance(repo, ProviderCallRepoSaveProtocol):
        return repo.save(record)
    if isinstance(repo, ProviderCallRepoAddProtocol):
        return repo.add(record)
    raise TypeError("provider_call_repo must implement save(record) or add(record)")


def _resolve_dispatch_persistence(
    *,
    persistence: DispatchPersistenceProtocol | None,
    attempt_repo: AttemptRepoLike | None,
    provider_call_repo: ProviderCallRepoLike | None,
) -> DispatchPersistenceProtocol | None:
    if persistence is not None:
        if attempt_repo is not None or provider_call_repo is not None:
            raise ValueError(
                "use either persistence=... or attempt_repo/provider_call_repo, not both"
            )
        if not isinstance(persistence, DispatchPersistenceProtocol):
            raise TypeError("persistence must implement DispatchPersistenceProtocol")
        return persistence

    if attempt_repo is None and provider_call_repo is None:
        return None
    if attempt_repo is None or provider_call_repo is None:
        raise ValueError(
            "attempt_repo and provider_call_repo must either both be provided or both be omitted"
        )

    warnings.warn(
        "DispatchController(... attempt_repo=..., provider_call_repo=...) is deprecated; "
        "use persistence=RepositoryDispatchPersistence(...)",
        DeprecationWarning,
        stacklevel=3,
    )
    return RepositoryDispatchPersistence(
        attempt_repo=attempt_repo,
        provider_call_repo=provider_call_repo,
    )


async def _call_provider(
    *,
    provider: ProviderProtocol | LegacyGenerateProviderProtocol,
    request: ProviderRequest,
) -> ProviderResponse:
    if isinstance(provider, ProviderProtocol):
        return await provider.send(request)

    if isinstance(provider, LegacyGenerateProviderProtocol):
        warnings.warn(
            "provider.generate(...) is deprecated; implement provider.send(...)",
            DeprecationWarning,
            stacklevel=3,
        )
        return await provider.generate(request)

    raise TypeError("provider must implement send(request)")


def _usage_total_tokens(usage: ProviderUsage) -> int:
    if usage.total_tokens > 0:
        return usage.total_tokens
    derived_total = usage.input_tokens + usage.output_tokens
    if derived_total > 0:
        return derived_total
    return 0


def _usage_cost_usd(usage: ProviderUsage) -> float:
    if usage.cost_estimate_usd is None:
        return 0.0
    return usage.cost_estimate_usd


def _derive_dispatch_idempotency_key(request: DispatchRequest) -> str:
    if request.idempotency_key is not None:
        return request.idempotency_key
    attempt_number_raw = request.metadata.get("attempt_number", 1)
    attempt_number = int(attempt_number_raw) if isinstance(attempt_number_raw, int) else 1
    prompt_hash = sha256_text(request.prompt)
    seed = f"{request.work_item_id}|{attempt_number}|{prompt_hash}|{request.run_id}|{request.role}"
    return sha256_text(seed)


def _derive_provider_idempotency_key(
    *,
    request: DispatchRequest,
    dispatch_idempotency_key: str,
    provider_name: str,
    model: str,
) -> str:
    if request.idempotency_key is not None:
        return request.idempotency_key
    return sha256_text(f"{dispatch_idempotency_key}|{provider_name}|{model}")


def _provider_call_record_id(*, attempt_id: str, provider_name: str, attempts_made: int) -> str:
    digest = sha256_text(f"{attempt_id}|{provider_name}|{attempts_made}")
    return f"pc-{digest[:24]}"


async def _sleep_with_cancellation(
    delay_seconds: float,
    *,
    sleep: SleepFn,
    cancel_token: CancellationToken | None,
) -> None:
    if delay_seconds <= 0:
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        return

    if cancel_token is None:
        await sleep(delay_seconds)
        return

    cancel_token.raise_if_cancelled()
    sleep_task: asyncio.Task[None] = asyncio.create_task(_await_sleep(delay_seconds, sleep=sleep))
    cancel_task = asyncio.create_task(cancel_token.wait())
    done, pending = await asyncio.wait(
        {sleep_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    try:
        if cancel_task in done and cancel_token.is_cancelled:
            sleep_task.cancel()
            await _await_cancelled(sleep_task)
            raise asyncio.CancelledError("dispatch cancelled")
        await sleep_task
    finally:
        for task in pending:
            task.cancel()
        await _await_cancelled(cancel_task)


async def _acquire_semaphore_with_cancellation(
    semaphore: asyncio.Semaphore,
    *,
    cancel_token: CancellationToken | None,
) -> None:
    if cancel_token is None:
        await semaphore.acquire()
        return

    cancel_token.raise_if_cancelled()
    acquire_task = asyncio.create_task(semaphore.acquire())
    cancel_task = asyncio.create_task(cancel_token.wait())
    done, pending = await asyncio.wait(
        {acquire_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    try:
        if cancel_task in done and cancel_token.is_cancelled:
            acquire_task.cancel()
            await _await_cancelled(acquire_task)
            raise asyncio.CancelledError("dispatch cancelled")
        await acquire_task
    finally:
        for task in pending:
            task.cancel()
        await _await_cancelled(cancel_task)


async def _await_with_cancellation(
    awaitable: Awaitable[ProviderResponse],
    *,
    cancel_token: CancellationToken | None,
) -> ProviderResponse:
    if cancel_token is None:
        return await awaitable

    cancel_token.raise_if_cancelled()
    value_task: asyncio.Task[ProviderResponse] = asyncio.create_task(
        _await_provider_response(awaitable)
    )
    cancel_task = asyncio.create_task(cancel_token.wait())
    done, pending = await asyncio.wait(
        {value_task, cancel_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    try:
        if cancel_task in done and cancel_token.is_cancelled:
            value_task.cancel()
            await _await_cancelled(value_task)
            raise asyncio.CancelledError("dispatch cancelled")
        return await value_task
    finally:
        for task in pending:
            task.cancel()
        await _await_cancelled(cancel_task)


async def _await_cancelled(task: asyncio.Task[_TaskT]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return


async def _await_sleep(delay_seconds: float, *, sleep: SleepFn) -> None:
    await sleep(delay_seconds)


async def _await_provider_response(awaitable: Awaitable[ProviderResponse]) -> ProviderResponse:
    return await awaitable


def _validate_non_empty_str(value: str | None, field_name: str, *, strip: bool = True) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be a non-empty string")
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip() if strip else value
    if not normalized.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _truncate_feedback(message: str | None) -> str | None:
    if message is None:
        return None
    return message[:4096]


__all__ = [
    "AttemptRepoLike",
    "BudgetExceededError",
    "DispatchPersistenceProtocol",
    "DeterministicRateLimiter",
    "DispatchBudget",
    "DispatchController",
    "DispatchError",
    "DispatchFailedError",
    "DispatchRequest",
    "DispatchResult",
    "ProviderBinding",
    "ProviderCallError",
    "ProviderCallRepoLike",
    "ProviderError",
    "ProviderProtocol",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderUsage",
    "RepositoryDispatchPersistence",
    "TranscriptEntry",
]
