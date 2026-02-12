"""Regression tests for concurrency utility edge cases."""

from __future__ import annotations

import asyncio
import gc
import sys
import warnings
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.utils.concurrency import CancellationToken, run_with_timeout

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _capture_unraisable() -> Iterator[list[SimpleNamespace]]:
    captured: list[SimpleNamespace] = []
    original = sys.unraisablehook

    def hook(unraisable: object) -> None:
        entry = cast_unraisable(unraisable)
        captured.append(entry)

    sys.unraisablehook = hook
    try:
        yield captured
    finally:
        sys.unraisablehook = original


def cast_unraisable(unraisable: object) -> SimpleNamespace:
    if isinstance(unraisable, SimpleNamespace):
        return unraisable

    namespace = SimpleNamespace(
        exc_type=getattr(unraisable, "exc_type", None),
        exc_value=getattr(unraisable, "exc_value", None),
        err_msg=getattr(unraisable, "err_msg", None),
        object=getattr(unraisable, "object", None),
    )
    return namespace


async def _slow() -> int:
    await asyncio.sleep(0.01)
    return 1


async def _slower() -> int:
    await asyncio.sleep(0.05)
    return 1


async def test_run_with_timeout_does_not_leak_coroutine_on_early_cancel() -> None:
    token = CancellationToken()
    token.cancel()

    with _capture_unraisable() as leaked, warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        coro = _slow()
        with pytest.raises(asyncio.CancelledError):
            await run_with_timeout(coro, 1.0, token)
        del coro
        gc.collect()

    assert leaked == []


async def test_run_with_timeout_timeout_path_does_not_leak_coroutine() -> None:
    with _capture_unraisable() as leaked, warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(TimeoutError):
            await run_with_timeout(_slower(), 0.001, None)
        gc.collect()

    assert leaked == []
