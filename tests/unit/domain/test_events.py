"""Unit tests for domain events and redaction behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.events import EventType, NexusEvent, redact_sensitive

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _fixed_bytes(size: int) -> bytes:
    return b"\x02" * size


def test_event_type_exhaustiveness_for_lifecycle() -> None:
    expected = {
        "RunStarted",
        "RunCompleted",
        "RunFailed",
        "RunPaused",
        "RunResumed",
        "WorkItemCreated",
        "WorkItemReady",
        "WorkItemDispatched",
        "WorkItemVerifying",
        "WorkItemPassed",
        "WorkItemFailed",
        "WorkItemMerged",
        "AttemptStarted",
        "AttemptCompleted",
        "AttemptFailed",
        "VerificationStarted",
        "VerificationPassed",
        "VerificationFailed",
        "MergeQueued",
        "MergeSucceeded",
        "MergeFailed",
        "MergeReverted",
        "ConstraintAdded",
        "ConstraintViolated",
        "BudgetExhausted",
        "BackpressureActivated",
        "BackpressureReleased",
        "IncidentRecorded",
    }
    actual = {member.value for member in EventType}
    assert expected.issubset(actual)


def test_event_json_roundtrip() -> None:
    timestamp = datetime(2026, 2, 1, 10, 30, 0, tzinfo=UTC)
    event = NexusEvent(
        event_id=ids.generate_event_id(timestamp_ms=10, randbytes=_fixed_bytes),
        event_type=EventType.WORK_ITEM_DISPATCHED,
        timestamp=timestamp,
        correlation_id="run-abc123",
        payload={
            "work_item_id": ids.generate_work_item_id(timestamp_ms=11, randbytes=_fixed_bytes),
            "attempt": 1,
        },
    )

    raw = event.to_json()
    restored = NexusEvent.from_json(raw)
    assert restored == event


def test_redact_sensitive_is_deep_and_non_mutating() -> None:
    timestamp = datetime(2026, 2, 1, 10, 30, 0, tzinfo=UTC)
    original = NexusEvent(
        event_id=ids.generate_event_id(timestamp_ms=12, randbytes=_fixed_bytes),
        event_type=EventType.ATTEMPT_FAILED,
        timestamp=timestamp,
        correlation_id=None,
        payload={
            "token": "abc",
            "nested": {
                "apiKey": "secret-value",
                "items": [
                    {"password": "p1", "safe": "ok"},
                    {"session": {"client_secret": "s2"}},
                ],
            },
            "safe_value": "visible",
        },
    )

    redacted = redact_sensitive(original)

    assert redacted.payload["token"] == "***REDACTED***"
    nested = redacted.payload["nested"]
    assert isinstance(nested, dict)
    assert nested["apiKey"] == "***REDACTED***"
    nested_items = nested["items"]
    assert isinstance(nested_items, list)
    first_item = nested_items[0]
    assert isinstance(first_item, dict)
    assert first_item["password"] == "***REDACTED***"
    second_item = nested_items[1]
    assert isinstance(second_item, dict)
    session = second_item["session"]
    assert isinstance(session, dict)
    assert session["client_secret"] == "***REDACTED***"
    assert redacted.payload["safe_value"] == "visible"

    assert original.payload["token"] == "abc"
    original_nested = original.payload["nested"]
    assert isinstance(original_nested, dict)
    assert original_nested["apiKey"] == "secret-value"


def test_timestamp_validation_and_default_presence() -> None:
    timestamp = datetime(2026, 2, 2, 0, 0, 0, tzinfo=UTC)
    event = NexusEvent(
        event_id=ids.generate_event_id(timestamp_ms=14, randbytes=_fixed_bytes),
        event_type=EventType.RUN_STARTED,
        timestamp=timestamp,
        correlation_id=None,
        payload={},
    )
    assert event.timestamp == timestamp

    with pytest.raises(ValueError, match="timestamp"):
        NexusEvent(
            event_id=ids.generate_event_id(timestamp_ms=15, randbytes=_fixed_bytes),
            event_type=EventType.RUN_STARTED,
            timestamp=datetime(2026, 2, 2, 0, 0, 0),
            correlation_id=None,
            payload={},
        )
