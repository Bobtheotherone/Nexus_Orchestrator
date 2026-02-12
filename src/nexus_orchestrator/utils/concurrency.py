"""Async concurrency primitives used across orchestration planes."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Iterable

T = TypeVar("T")


class CancellationToken:
    """Cooperative cancellation token backed by ``asyncio.Event``."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise asyncio.CancelledError("operation cancelled")


class BoundedSemaphore:
    """Small wrapper over ``asyncio.Semaphore`` with usage diagnostics."""

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("limit must be > 0")
        self._limit = limit
        self._semaphore = asyncio.Semaphore(limit)
        self._in_use = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def available(self) -> int:
        return self._limit - self._in_use

    async def acquire(self) -> None:
        # Cancellation while waiting here does not acquire a permit.
        await self._semaphore.acquire()
        self._in_use += 1

    def release(self) -> None:
        if self._in_use <= 0:
            raise RuntimeError("release called more times than acquire")
        self._in_use -= 1
        self._semaphore.release()

    @asynccontextmanager
    async def permit(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def snapshot(self) -> dict[str, int]:
        return {
            "limit": self._limit,
            "in_use": self._in_use,
            "available": self.available,
        }


@dataclass(slots=True)
class WorkerPool(Generic[T]):
    """Run coroutines with bounded concurrency and yield results as they finish."""

    max_concurrency: int
    cancel_token: CancellationToken | None = None
    _token: CancellationToken = field(init=False, repr=False)
    _semaphore: BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")
        self._token = self.cancel_token or CancellationToken()
        self._semaphore = BoundedSemaphore(self.max_concurrency)

    async def run(self, coroutines: Iterable[Awaitable[T]]) -> AsyncIterator[T]:
        tasks: set[asyncio.Task[T]] = set()

        for coroutine in coroutines:
            self._token.raise_if_cancelled()
            task: asyncio.Task[T] = asyncio.create_task(self._run_one(coroutine))
            tasks.add(task)

        try:
            while tasks:
                self._token.raise_if_cancelled()
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = set(pending)

                for task in done:
                    if task.cancelled():
                        raise asyncio.CancelledError("worker task cancelled")
                    exc = task.exception()
                    if exc is not None:
                        await self._cancel_all(tasks)
                        raise exc
                    yield task.result()
        except asyncio.CancelledError:
            await self._cancel_all(tasks)
            raise

    async def _run_one(self, coroutine: Awaitable[T]) -> T:
        async with self._semaphore.permit():
            self._token.raise_if_cancelled()
            return await coroutine

    async def _cancel_all(self, tasks: set[asyncio.Task[T]]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            with suppress(Exception):
                await asyncio.gather(*tasks, return_exceptions=True)


async def run_with_timeout(
    coroutine: Awaitable[T],
    timeout_seconds: float,
    cancel_token: CancellationToken | None = None,
) -> T:
    """Run ``coroutine`` with timeout and cooperative cancellation support."""
    if timeout_seconds <= 0:
        _close_unscheduled_coroutine(coroutine)
        raise ValueError("timeout_seconds must be > 0")

    token = cancel_token or CancellationToken()
    if token.is_cancelled:
        _close_unscheduled_coroutine(coroutine)
        raise asyncio.CancelledError("operation cancelled")

    task: asyncio.Task[T] = asyncio.create_task(_await_value(coroutine))
    cancel_wait_task = asyncio.create_task(token.wait())

    try:
        done, _ = await asyncio.wait(
            {task, cancel_wait_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )

        if task in done:
            return await task

        if cancel_wait_task in done and token.is_cancelled:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise asyncio.CancelledError("operation cancelled")

        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        raise TimeoutError(f"operation timed out after {timeout_seconds} seconds")
    finally:
        cancel_wait_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_wait_task


async def _await_value(awaitable: Awaitable[T]) -> T:
    return await awaitable


def _close_unscheduled_coroutine(awaitable: Awaitable[object]) -> None:
    # If cancellation/validation fails before scheduling, close raw coroutine objects
    # so CPython does not emit "coroutine was never awaited" at GC time.
    if inspect.iscoroutine(awaitable):
        awaitable.close()


__all__ = [
    "BoundedSemaphore",
    "CancellationToken",
    "WorkerPool",
    "run_with_timeout",
]
