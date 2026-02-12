"""In-process event bus with replay and critical-event persistence hooks."""

from __future__ import annotations

import asyncio
import inspect
import math
import threading
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, cast

from nexus_orchestrator.domain import EventType, NexusEvent
from nexus_orchestrator.domain.ids import generate_event_id

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

Subscriber = Callable[[NexusEvent], object]
PersistenceCallback = Callable[[NexusEvent], object]

_MAX_JSON_DEPTH: Final[int] = 16
_DEFAULT_ERROR_BUFFER: Final[int] = 1024
_DEFAULT_CRITICAL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        EventType.RUN_STARTED.value,
        EventType.RUN_COMPLETED.value,
        EventType.RUN_FAILED.value,
        EventType.VERIFICATION_FAILED.value,
        EventType.MERGE_FAILED.value,
        EventType.INCIDENT_RECORDED.value,
    }
)


@dataclass(frozen=True, slots=True)
class DispatchError:
    """Dispatch/persistence failure captured without interrupting publishers."""

    stage: str
    event_id: str
    target: str
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class _Subscription:
    token: int
    event_type: str | None
    callback: Subscriber
    is_async: bool


class EventBus:
    """Resilient event bus with sync+async subscribers and deterministic replay."""

    def __init__(
        self,
        *,
        buffer_size: int = 512,
        persist_event: PersistenceCallback | None = None,
        persistence_callback: PersistenceCallback | None = None,
        critical_event_types: Sequence[str | EventType] | None = None,
    ) -> None:
        if not isinstance(buffer_size, int):
            raise ValueError(f"buffer_size must be an integer, got {type(buffer_size).__name__}")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        persistence = _resolve_persistence_callback(persist_event, persistence_callback)

        self._buffer = deque[NexusEvent](maxlen=buffer_size)
        self._subscriptions: dict[int, _Subscription] = {}
        self._pending_async_tasks: set[asyncio.Task[None]] = set()
        self._dispatch_errors = deque[DispatchError](maxlen=_DEFAULT_ERROR_BUFFER)
        self._next_token = 1
        self._lock = threading.RLock()
        self._persist_event = persistence
        self._critical_event_types = _normalize_critical_event_types(critical_event_types)

    def set_persistence_callback(self, callback: PersistenceCallback | None) -> None:
        """Replace persistence callback used for critical events."""

        if callback is not None and not callable(callback):
            raise ValueError("persistence callback must be callable")
        with self._lock:
            self._persist_event = callback

    def subscribe(self, event_type: str | EventType | None, callback: Subscriber) -> int:
        """Subscribe callback to an event type or all events when ``event_type`` is ``None``."""

        if not callable(callback):
            raise ValueError("callback must be callable")

        normalized_event_type = _normalize_event_type_filter(event_type)
        is_async = inspect.iscoroutinefunction(callback)

        with self._lock:
            token = self._next_token
            self._next_token += 1
            self._subscriptions[token] = _Subscription(
                token=token,
                event_type=normalized_event_type,
                callback=callback,
                is_async=is_async,
            )
        return token

    def unsubscribe(self, token: int) -> bool:
        """Unsubscribe callback token. Returns ``True`` when token existed."""

        if not isinstance(token, int):
            raise ValueError(f"token must be an integer, got {type(token).__name__}")
        with self._lock:
            return self._subscriptions.pop(token, None) is not None

    def publish(self, event: NexusEvent) -> tuple[DispatchError, ...]:
        """Publish an event from synchronous code."""

        _ensure_event(event)
        with self._lock:
            self._buffer.append(event)
            subscriptions = tuple(self._subscriptions.values())
            persistence = self._persist_event

        errors: list[DispatchError] = []
        running_loop = _current_running_loop()

        if _is_critical_event(event, self._critical_event_types) and persistence is not None:
            persistence_error = self._invoke_callback(
                callback=persistence,
                callback_name=_callback_name(persistence),
                event=event,
                stage="persistence",
                running_loop=running_loop,
            )
            if persistence_error is not None:
                errors.append(persistence_error)

        for subscription in subscriptions:
            if not _subscription_matches(subscription, event):
                continue
            dispatch_error = self._invoke_callback(
                callback=subscription.callback,
                callback_name=_callback_name(subscription.callback),
                event=event,
                stage="subscriber",
                running_loop=running_loop,
            )
            if dispatch_error is not None:
                errors.append(dispatch_error)

        if errors:
            with self._lock:
                self._dispatch_errors.extend(errors)

        return tuple(errors)

    async def publish_async(self, event: NexusEvent) -> tuple[DispatchError, ...]:
        """Publish an event from async code and await async subscribers."""

        _ensure_event(event)
        with self._lock:
            self._buffer.append(event)
            subscriptions = tuple(self._subscriptions.values())
            persistence = self._persist_event

        errors: list[DispatchError] = []

        if _is_critical_event(event, self._critical_event_types) and persistence is not None:
            error = await self._invoke_callback_async(
                callback=persistence,
                callback_name=_callback_name(persistence),
                event=event,
                stage="persistence",
            )
            if error is not None:
                errors.append(error)

        for subscription in subscriptions:
            if not _subscription_matches(subscription, event):
                continue
            error = await self._invoke_callback_async(
                callback=subscription.callback,
                callback_name=_callback_name(subscription.callback),
                event=event,
                stage="subscriber",
            )
            if error is not None:
                errors.append(error)

        if errors:
            with self._lock:
                self._dispatch_errors.extend(errors)

        return tuple(errors)

    def emit(
        self,
        event_type: str | EventType,
        payload: Mapping[str, object],
        *,
        correlation_id: str | None = None,
    ) -> tuple[NexusEvent, tuple[DispatchError, ...]]:
        """Create and publish an event from sync code."""

        event = _build_event(event_type=event_type, payload=payload, correlation_id=correlation_id)
        return event, self.publish(event)

    async def emit_async(
        self,
        event_type: str | EventType,
        payload: Mapping[str, object],
        *,
        correlation_id: str | None = None,
    ) -> tuple[NexusEvent, tuple[DispatchError, ...]]:
        """Create and publish an event from async code."""

        event = _build_event(event_type=event_type, payload=payload, correlation_id=correlation_id)
        return event, await self.publish_async(event)

    async def drain_async(self) -> tuple[DispatchError, ...]:
        """Await queued async subscriber tasks created by synchronous ``publish``."""

        with self._lock:
            pending = tuple(self._pending_async_tasks)
            self._pending_async_tasks.clear()

        if not pending:
            return ()

        await asyncio.gather(*pending, return_exceptions=True)

        with self._lock:
            return tuple(self._dispatch_errors)

    def replay(
        self,
        *,
        since: datetime | str | None = None,
        event_type: str | EventType | None = None,
        limit: int | None = None,
    ) -> tuple[NexusEvent, ...]:
        """Replay buffered events in deterministic publish order."""

        since_dt = _normalize_since(since)
        type_filter = _normalize_event_type_filter(event_type)

        with self._lock:
            events = tuple(self._buffer)

        filtered = [
            event
            for event in events
            if _event_matches_replay_filter(event, since=since_dt, event_type=type_filter)
        ]

        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError(f"limit must be an integer, got {type(limit).__name__}")
            if limit <= 0:
                return ()
            filtered = filtered[-limit:]

        return tuple(filtered)

    def history(
        self,
        *,
        since: datetime | str | None = None,
        event_type: str | EventType | None = None,
        limit: int | None = None,
    ) -> tuple[NexusEvent, ...]:
        """Alias for ``replay``."""

        return self.replay(since=since, event_type=event_type, limit=limit)

    def dispatch_errors(self, *, limit: int | None = None) -> tuple[DispatchError, ...]:
        """Return recorded subscriber/persistence failures."""

        with self._lock:
            errors = tuple(self._dispatch_errors)

        if limit is None:
            return errors
        if not isinstance(limit, int):
            raise ValueError(f"limit must be an integer, got {type(limit).__name__}")
        if limit <= 0:
            return ()
        return errors[-limit:]

    def _invoke_callback(
        self,
        *,
        callback: Callable[[NexusEvent], object],
        callback_name: str,
        event: NexusEvent,
        stage: str,
        running_loop: asyncio.AbstractEventLoop | None,
    ) -> DispatchError | None:
        try:
            result = callback(event)
            if inspect.isawaitable(result):
                coroutine = _as_coroutine(result)
                if running_loop is None:
                    _run_awaitable_sync(coroutine)
                    return None

                task = running_loop.create_task(coroutine)
                with self._lock:
                    self._pending_async_tasks.add(task)
                task.add_done_callback(
                    lambda done: self._on_async_callback_done(
                        done,
                        stage=stage,
                        target=callback_name,
                        event=event,
                    )
                )
            return None
        except Exception as exc:  # noqa: BLE001
            return _dispatch_error(stage=stage, event=event, target=callback_name, exc=exc)

    async def _invoke_callback_async(
        self,
        *,
        callback: Callable[[NexusEvent], object],
        callback_name: str,
        event: NexusEvent,
        stage: str,
    ) -> DispatchError | None:
        try:
            result = callback(event)
            if inspect.isawaitable(result):
                await _as_coroutine(result)
            return None
        except Exception as exc:  # noqa: BLE001
            return _dispatch_error(stage=stage, event=event, target=callback_name, exc=exc)

    def _on_async_callback_done(
        self,
        task: asyncio.Task[None],
        *,
        stage: str,
        target: str,
        event: NexusEvent,
    ) -> None:
        with self._lock:
            self._pending_async_tasks.discard(task)

        try:
            task.result()
        except Exception as exc:  # noqa: BLE001
            error = _dispatch_error(stage=stage, event=event, target=target, exc=exc)
            with self._lock:
                self._dispatch_errors.append(error)


def _resolve_persistence_callback(
    persist_event: PersistenceCallback | None,
    persistence_callback: PersistenceCallback | None,
) -> PersistenceCallback | None:
    if (
        persist_event is not None
        and persistence_callback is not None
        and persist_event is not persistence_callback
    ):
        raise ValueError("provide only one of persist_event or persistence_callback")
    callback = persistence_callback if persistence_callback is not None else persist_event
    if callback is not None and not callable(callback):
        raise ValueError("persistence callback must be callable")
    return callback


def _normalize_critical_event_types(
    critical_event_types: Sequence[str | EventType] | None,
) -> frozenset[str]:
    if critical_event_types is None:
        return _DEFAULT_CRITICAL_EVENT_TYPES

    normalized: set[str] = set()
    for item in critical_event_types:
        normalized.add(_normalize_event_type(item))
    return frozenset(normalized)


def _ensure_event(event: object) -> None:
    if not isinstance(event, NexusEvent):
        raise ValueError(f"event must be NexusEvent, got {type(event).__name__}")


def _build_event(
    *,
    event_type: str | EventType,
    payload: Mapping[str, object],
    correlation_id: str | None,
) -> NexusEvent:
    return NexusEvent(
        event_id=generate_event_id(),
        event_type=_as_event_type(event_type),
        timestamp=datetime.now(tz=UTC),
        correlation_id=correlation_id,
        payload=_as_json_object(payload, "payload"),
    )


def _as_event_type(value: str | EventType) -> EventType:
    if isinstance(value, EventType):
        return value
    if not isinstance(value, str):
        raise ValueError(f"event_type must be string/EventType, got {type(value).__name__}")
    try:
        return EventType(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in EventType)
        raise ValueError(f"invalid event_type {value!r}; allowed: {allowed}") from exc


def _normalize_event_type(value: str | EventType) -> str:
    if isinstance(value, EventType):
        return value.value
    if not isinstance(value, str):
        raise ValueError(f"event type must be string/EventType, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError("event type must not be empty")
    return normalized


def _normalize_event_type_filter(value: str | EventType | None) -> str | None:
    if value is None:
        return None
    return _normalize_event_type(value)


def _normalize_since(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("since datetime must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid since datetime string: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("since datetime string must include timezone")
        return parsed.astimezone(UTC)
    raise ValueError(f"since must be datetime/str/None, got {type(value).__name__}")


def _event_matches_replay_filter(
    event: NexusEvent,
    *,
    since: datetime | None,
    event_type: str | None,
) -> bool:
    return not (
        (since is not None and event.timestamp <= since)
        or (event_type is not None and event.event_type.value != event_type)
    )


def _is_critical_event(event: NexusEvent, critical_event_types: frozenset[str]) -> bool:
    return event.event_type.value in critical_event_types


def _subscription_matches(subscription: _Subscription, event: NexusEvent) -> bool:
    if subscription.event_type is None:
        return True
    return subscription.event_type == event.event_type.value


def _callback_name(callback: object) -> str:
    name = getattr(callback, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return callback.__class__.__name__


def _current_running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _as_coroutine(value: object) -> Coroutine[Any, Any, None]:
    if inspect.iscoroutine(value):
        return cast("Coroutine[Any, Any, None]", value)
    if inspect.isawaitable(value):
        return _await_awaitable(cast("Awaitable[None]", value))
    raise TypeError("callback marked async did not return an awaitable")


async def _await_awaitable(awaitable: Awaitable[None]) -> None:
    await awaitable


def _run_awaitable_sync(coroutine: Coroutine[Any, Any, None]) -> None:
    loop = _current_running_loop()
    if loop is None:
        asyncio.run(coroutine)
        return

    error: BaseException | None = None

    def runner() -> None:
        nonlocal error
        try:
            asyncio.run(coroutine)
        except BaseException as exc:  # noqa: BLE001
            error = exc

    thread = threading.Thread(target=runner, name="nexus-eventbus-await", daemon=False)
    thread.start()
    thread.join()

    if error is not None:
        raise RuntimeError("failed to execute awaitable callback") from error


def _dispatch_error(*, stage: str, event: NexusEvent, target: str, exc: Exception) -> DispatchError:
    return DispatchError(
        stage=stage,
        event_id=event.event_id,
        target=target,
        error_type=exc.__class__.__name__,
        message=str(exc),
    )


def _as_json_object(value: object, path: str) -> dict[str, JSONValue]:
    parsed = _as_json_value(value, path)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected object")
    return parsed


def _as_json_value(value: object, path: str, *, depth: int = 0) -> JSONValue:
    if depth > _MAX_JSON_DEPTH:
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
    if isinstance(value, (list, tuple)):
        return [
            _as_json_value(item, f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: object keys must be strings")
            out[key] = _as_json_value(item, f"{path}.{key}", depth=depth + 1)
        return out

    raise ValueError(f"{path}: value is not JSON-serializable ({type(value).__name__})")


__all__ = [
    "DispatchError",
    "EventBus",
    "JSONScalar",
    "JSONValue",
    "PersistenceCallback",
    "Subscriber",
]
