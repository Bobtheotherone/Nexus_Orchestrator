"""
nexus-orchestrator — unit tests for observability event bus

File: tests/unit/observability/test_events.py
Last updated: 2026-02-12

Purpose
- Validate event bus fanout resilience, replay semantics, and persistence hooks.

What this test file should cover
- Sync+async subscriber support.
- Subscriber exception isolation.
- Ring-buffer replay ordering.
- Critical event persistence failure handling.

Functional requirements
- Offline and deterministic.

Non-functional requirements
- No sleep-based synchronization.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import nexus_orchestrator.observability as observability_pkg
from nexus_orchestrator.observability.events import EventBus

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def test_subscribers_receive_events_in_publish_order() -> None:
    bus = EventBus(buffer_size=10)
    sub_a: list[str] = []
    sub_b: list[str] = []

    bus.subscribe(None, lambda event: sub_a.append(event.event_type.value))
    bus.subscribe(None, lambda event: sub_b.append(event.event_type.value))

    _, errors_1 = bus.emit("RunStarted", {"x": 1})
    _, errors_2 = bus.emit("RunCompleted", {"x": 2})

    assert errors_1 == ()
    assert errors_2 == ()
    assert sub_a == ["RunStarted", "RunCompleted"]
    assert sub_b == ["RunStarted", "RunCompleted"]


async def test_async_subscriber_supports_publish_async_and_sync_publish() -> None:
    bus = EventBus(buffer_size=10)
    received: list[str] = []

    async def async_sub(event: object) -> None:
        # type narrowed by bus internals; list append is sync and deterministic.
        received.append(event.event_type.value)

    bus.subscribe(None, async_sub)

    _, async_errors = await bus.emit_async("WorkItemDispatched", {"n": 1})
    assert async_errors == ()

    _, sync_errors = bus.emit("WorkItemPassed", {"n": 2})
    assert sync_errors == ()

    await bus.drain_async()
    assert received == ["WorkItemDispatched", "WorkItemPassed"]


def test_subscriber_exception_does_not_break_other_subscribers() -> None:
    bus = EventBus(buffer_size=10)
    received: list[str] = []

    def broken(_event: object) -> None:
        raise RuntimeError("boom")

    def healthy(event: object) -> None:
        received.append(event.event_type.value)

    bus.subscribe(None, broken)
    bus.subscribe(None, healthy)

    _, errors = bus.emit("RunFailed", {"reason": "x"})

    assert received == ["RunFailed"]
    assert len(errors) == 1
    assert errors[0].stage == "subscriber"


def test_replay_ring_buffer_is_deterministic() -> None:
    bus = EventBus(buffer_size=2)
    bus.emit("RunStarted", {"v": 1})
    bus.emit("WorkItemDispatched", {"v": 2})
    bus.emit("RunCompleted", {"v": 3})

    replay = bus.replay()

    assert [event.event_type.value for event in replay] == ["WorkItemDispatched", "RunCompleted"]


def test_replay_since_filter() -> None:
    bus = EventBus(buffer_size=5)
    first, _ = bus.emit("RunStarted", {"v": 1})
    bus.emit("RunCompleted", {"v": 2})

    replay = bus.replay(since=first.timestamp)

    assert [event.event_type.value for event in replay] == ["RunCompleted"]


def test_critical_event_persistence_callback_failure_is_recorded_and_event_replayable() -> None:
    persisted: list[str] = []

    def persist(event: object) -> None:
        persisted.append(event.event_id)
        raise ValueError("db down")

    bus = EventBus(buffer_size=10, persist_event=persist)

    event, errors = bus.emit("RunFailed", {"reason": "failure"})

    assert persisted == [event.event_id]
    assert len(errors) == 1
    assert errors[0].stage == "persistence"
    assert any(item.event_id == event.event_id for item in bus.dispatch_errors())
    replay = bus.replay()
    assert replay[-1].event_id == event.event_id


def test_replay_accepts_iso_since_strings() -> None:
    bus = EventBus(buffer_size=10)
    event, _ = bus.emit("RunStarted", {"ok": True})
    future = (event.timestamp + timedelta(seconds=1)).astimezone(UTC)

    replay = bus.replay(since=future.isoformat().replace("+00:00", "Z"))

    assert replay == ()


def test_observability_package_exports_event_bus() -> None:
    bus = observability_pkg.EventBus(buffer_size=2)
    event, errors = bus.emit("RunStarted", {"ok": True})
    assert errors == ()
    replay = bus.replay()
    assert replay[-1].event_id == event.event_id
