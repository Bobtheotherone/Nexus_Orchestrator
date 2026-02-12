"""Domain event definitions, serialization, and payload redaction helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from nexus_orchestrator.domain import ids

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

if TYPE_CHECKING:
    from enum import StrEnum
else:
    try:
        from enum import StrEnum
    except ImportError:

        class StrEnum(str, Enum):
            """Compatibility fallback for Python < 3.11."""


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_SENSITIVE_KEY_TERMS = ("secret", "key", "password", "token")
_REDACTED_VALUE = "***REDACTED***"


class EventType(StrEnum):
    """Lifecycle events emitted by the orchestrator."""

    RUN_STARTED = "RunStarted"
    RUN_COMPLETED = "RunCompleted"
    RUN_FAILED = "RunFailed"
    RUN_PAUSED = "RunPaused"
    RUN_RESUMED = "RunResumed"

    WORK_ITEM_CREATED = "WorkItemCreated"
    WORK_ITEM_READY = "WorkItemReady"
    WORK_ITEM_DISPATCHED = "WorkItemDispatched"
    WORK_ITEM_VERIFYING = "WorkItemVerifying"
    WORK_ITEM_PASSED = "WorkItemPassed"
    WORK_ITEM_FAILED = "WorkItemFailed"
    WORK_ITEM_MERGED = "WorkItemMerged"

    ATTEMPT_STARTED = "AttemptStarted"
    ATTEMPT_COMPLETED = "AttemptCompleted"
    ATTEMPT_FAILED = "AttemptFailed"

    VERIFICATION_STARTED = "VerificationStarted"
    VERIFICATION_PASSED = "VerificationPassed"
    VERIFICATION_FAILED = "VerificationFailed"

    MERGE_QUEUED = "MergeQueued"
    MERGE_SUCCEEDED = "MergeSucceeded"
    MERGE_FAILED = "MergeFailed"
    MERGE_REVERTED = "MergeReverted"

    CONSTRAINT_ADDED = "ConstraintAdded"
    CONSTRAINT_VIOLATED = "ConstraintViolated"

    BUDGET_EXHAUSTED = "BudgetExhausted"

    BACKPRESSURE_ACTIVATED = "BackpressureActivated"
    BACKPRESSURE_RELEASED = "BackpressureReleased"

    INCIDENT_RECORDED = "IncidentRecorded"


@dataclass(slots=True)
class NexusEvent:
    """Serializable event envelope for observability across orchestration planes."""

    event_id: str
    event_type: EventType
    timestamp: datetime
    correlation_id: str | None
    payload: dict[str, JSONValue]

    def __post_init__(self) -> None:
        ids.validate_event_id(self.event_id)
        self.event_type = _as_event_type(self.event_type, "NexusEvent.event_type")
        self.timestamp = _as_utc_datetime(self.timestamp, "NexusEvent.timestamp")
        self.correlation_id = _as_optional_str(self.correlation_id, "NexusEvent.correlation_id")
        self.payload = _as_json_object(self.payload, "NexusEvent.payload")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": _datetime_to_iso8601z(self.timestamp),
            "correlation_id": self.correlation_id,
            "payload": _as_json_object(self.payload, "NexusEvent.payload"),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NexusEvent:
        parsed = _expect_object(
            data,
            "NexusEvent",
            required={"event_id", "event_type", "timestamp", "payload"},
            optional={"correlation_id"},
        )
        return cls(
            event_id=_as_str(parsed["event_id"], "NexusEvent.event_id", max_len=128),
            event_type=_as_event_type(parsed["event_type"], "NexusEvent.event_type"),
            timestamp=_as_utc_datetime(parsed["timestamp"], "NexusEvent.timestamp"),
            correlation_id=_as_optional_str(
                parsed.get("correlation_id"), "NexusEvent.correlation_id"
            ),
            payload=_as_json_object(parsed["payload"], "NexusEvent.payload"),
        )

    @classmethod
    def from_json(cls, raw: str) -> NexusEvent:
        if not isinstance(raw, str):
            raise ValueError(f"NexusEvent: expected JSON string, got {type(raw).__name__}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"NexusEvent: invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("NexusEvent: JSON root must be an object")
        return cls.from_dict(parsed)


def redact_sensitive(event: NexusEvent) -> NexusEvent:
    """Return a new event with sensitive payload keys deeply redacted."""
    redacted_payload = _redact_value(event.payload, key_context=None)
    if not isinstance(redacted_payload, dict):
        raise ValueError("redacted payload must remain a JSON object")
    return NexusEvent(
        event_id=event.event_id,
        event_type=event.event_type,
        timestamp=event.timestamp,
        correlation_id=event.correlation_id,
        payload=redacted_payload,
    )


def _expect_object(
    value: object,
    path: str,
    *,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object, got {type(value).__name__}")

    for key in value:
        if not isinstance(key, str):
            raise ValueError(f"{path}: object keys must be strings")

    unknown = sorted(key for key in value if key not in required and key not in optional)
    if unknown:
        raise ValueError(f"{path}: unexpected fields: {unknown}")

    missing = sorted(key for key in required if key not in value)
    if missing:
        raise ValueError(f"{path}: missing required fields: {missing}")

    return value


def _as_str(value: object, path: str, *, max_len: int = 4096) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected string, got {type(value).__name__}")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{path}: must not be empty")
    if len(parsed) > max_len:
        raise ValueError(f"{path}: must be <= {max_len} characters")
    return parsed


def _as_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, path, max_len=256)


def _as_event_type(value: object, path: str) -> EventType:
    if isinstance(value, EventType):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected string event type, got {type(value).__name__}")
    try:
        return EventType(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in EventType)
        raise ValueError(f"{path}: unsupported event type {value!r}; allowed: {allowed}") from exc


def _as_utc_datetime(value: object, path: str) -> datetime:
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid ISO-8601 datetime: {value!r}") from exc
    else:
        raise ValueError(
            f"{path}: expected datetime or ISO-8601 string, got {type(value).__name__}"
        )

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{path}: datetime must be timezone-aware UTC")
    return parsed.astimezone(UTC)


def _datetime_to_iso8601z(value: datetime) -> str:
    normalized = _as_utc_datetime(value, "NexusEvent.timestamp")
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _as_json_value(value: object, path: str, *, depth: int = 0) -> JSONValue:
    if depth > 16:
        raise ValueError(f"{path}: JSON nesting too deep")

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: float value must be finite")
        return value
    if isinstance(value, str):
        if len(value) > 8192:
            raise ValueError(f"{path}: string too long")
        return value
    if isinstance(value, list):
        return [
            _as_json_value(item, f"{path}[{idx}]", depth=depth + 1)
            for idx, item in enumerate(value)
        ]
    if isinstance(value, dict):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object keys must be strings")
            out[key] = _as_json_value(item, f"{path}.{key}", depth=depth + 1)
        return out
    raise ValueError(f"{path}: value is not JSON-serializable ({type(value).__name__})")


def _as_json_object(value: object, path: str) -> dict[str, JSONValue]:
    parsed = _as_json_value(value, path)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected object")
    return parsed


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(term in lowered for term in _SENSITIVE_KEY_TERMS)


def _redact_value(value: JSONValue, key_context: str | None) -> JSONValue:
    if key_context is not None and _is_sensitive_key(key_context):
        return _REDACTED_VALUE

    if isinstance(value, list):
        return [_redact_value(item, key_context=None) for item in value]

    if isinstance(value, dict):
        redacted: dict[str, JSONValue] = {}
        for key, item in value.items():
            redacted[key] = _redact_value(item, key_context=key)
        return redacted

    return value


__all__ = ["EventType", "NexusEvent", "redact_sensitive"]
