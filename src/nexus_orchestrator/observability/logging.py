"""Structured logging setup with JSON-lines output and redaction support."""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import logging.handlers
import math
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, cast

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
LogRedactor = Callable[[JSONValue], JSONValue]

_REDACTED_VALUE: Final[str] = "***REDACTED***"
_DEFAULT_LOG_FILENAME: Final[str] = "orchestrator.jsonl"
_DEFAULT_LOGGER_NAME: Final[str] = "nexus_orchestrator"
_DEFAULT_QUEUE_SIZE: Final[int] = 4096

_CORRELATION_KEYS: Final[tuple[str, ...]] = (
    "run_id",
    "correlation_id",
    "work_item_id",
    "attempt_id",
    "event_id",
    "trace_id",
    "span_id",
)

_SENSITIVE_KEY_TERMS: Final[tuple[str, ...]] = (
    "secret",
    "token",
    "password",
    "passphrase",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "private_key",
    "client_secret",
)

_TRANSCRIPT_KEY_TERMS: Final[tuple[str, ...]] = (
    "transcript",
    "provider_prompt",
    "provider_response",
    "prompt_messages",
)

_SENSITIVE_ASSIGNMENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|client_secret|authorization)\b\s*([:=])\s*([^\s,;]+)"
)
_BEARER_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*")
_OPENAI_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bsk-[A-Za-z0-9]{12,}\b")
_ANTHROPIC_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}\b")

_STANDARD_LOG_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

_CorrelationState = tuple[tuple[str, str], ...]
_CORRELATION_CONTEXT: contextvars.ContextVar[_CorrelationState] = contextvars.ContextVar(
    "nexus_observability_correlation", default=()
)

_ACTIVE_HANDLE_LOCK = threading.Lock()
_ACTIVE_HANDLE: StructuredLoggingHandle | None = None
_ATEXIT_REGISTERED = False


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Configuration for queue-backed structured logging."""

    run_id: str
    base_log_dir: Path | str = Path("logs")
    logger_name: str = _DEFAULT_LOGGER_NAME
    level: int | str = "INFO"
    queue_size: int = _DEFAULT_QUEUE_SIZE
    log_filename: str = _DEFAULT_LOG_FILENAME
    log_to_stdout: bool = True
    rotating_file: bool = False
    max_bytes: int = 10_000_000
    backup_count: int = 5
    redactor: LogRedactor | None = None


def setup_logging(
    observability_config: Mapping[str, object] | None = None,
    *,
    run_id: str,
    log_dir: Path | str | None = None,
    logger_name: str = _DEFAULT_LOGGER_NAME,
) -> logging.Logger:
    """Compatibility wrapper: configure structured logging and return the logger.

    Parameters
    ----------
    observability_config:
        Mapping compatible with ``[observability]`` settings in ``orchestrator.toml``.
    run_id:
        Correlation run identifier used for per-run log directory and event fields.
    log_dir:
        Optional override for the base log directory.
    logger_name:
        Logger name to configure.
    """

    cfg = dict(observability_config or {})
    raw_level = cfg.get("log_level", "INFO")
    level: int | str = raw_level if isinstance(raw_level, (int, str)) else "INFO"
    redact_enabled = bool(cfg.get("redact_secrets", True))
    raw_base_log_dir: object = log_dir if log_dir is not None else cfg.get("log_dir", "logs")
    base_log_dir: Path | str = (
        raw_base_log_dir if isinstance(raw_base_log_dir, (Path, str)) else "logs"
    )

    redactor: LogRedactor | None = None if redact_enabled else _identity_redactor

    handle = setup_structured_logging(
        LoggingConfig(
            run_id=run_id,
            base_log_dir=base_log_dir,
            logger_name=logger_name,
            level=level,
            redactor=redactor,
        )
    )
    return handle.logger


class _DropCounter:
    """Thread-safe counter for dropped queue records."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def value(self) -> int:
        with self._lock:
            return self._value


class _NonBlockingQueueHandler(logging.handlers.QueueHandler):
    """Queue handler that drops records when the queue is full."""

    def __init__(self, log_queue: queue.Queue[object], drop_counter: _DropCounter) -> None:
        super().__init__(log_queue)
        self._drop_counter = drop_counter

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        context = get_correlation_context()
        existing = getattr(record, "correlation", None)
        if isinstance(existing, Mapping):
            for key, value in existing.items():
                if isinstance(key, str) and isinstance(value, str):
                    key_name = key.strip()
                    value_name = value.strip()
                    if key_name and value_name:
                        context[key_name] = value_name
        if context:
            record.correlation = context
        prepared = super().prepare(record)
        return cast("logging.LogRecord", prepared)

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self._drop_counter.increment()


class _JsonLineFormatter(logging.Formatter):
    """Formatter that emits canonical JSON objects per log line."""

    def __init__(
        self,
        *,
        redactor: LogRedactor,
        base_context: Mapping[str, str],
    ) -> None:
        super().__init__()
        self._redactor = redactor
        self._base_context = dict(base_context)

    def format(self, record: logging.LogRecord) -> str:
        event: dict[str, JSONValue] = {
            "timestamp": _iso8601z_from_epoch(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": _coerce_log_message(
                self._redactor(_normalize_json_value(record.getMessage()))
            ),
        }

        correlation = _merge_correlation_context(record, self._base_context)
        for key, value in sorted(correlation.items()):
            event[key] = value

        extras = _extract_extra_fields(record)
        if extras:
            event["fields"] = self._redactor(_normalize_json_value(extras))

        if record.exc_info is not None:
            event["exception"] = _coerce_log_message(
                self._redactor(_normalize_json_value(self.formatException(record.exc_info)))
            )
        if record.stack_info:
            event["stack"] = _coerce_log_message(
                self._redactor(_normalize_json_value(str(record.stack_info)))
            )

        return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class StructuredLoggingHandle:
    """Runtime handle for an active structured logging setup."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        run_id: str,
        run_log_dir: Path,
        log_path: Path,
        queue_size: int,
        log_queue: queue.Queue[object],
        queue_handler: _NonBlockingQueueHandler,
        sink_handlers: tuple[logging.Handler, ...],
        listener: logging.handlers.QueueListener,
        drop_counter: _DropCounter,
    ) -> None:
        self.logger = logger
        self.run_id = run_id
        self.run_log_dir = run_log_dir
        self.log_path = log_path
        self.queue_size = queue_size
        self._queue = log_queue
        self._queue_handler = queue_handler
        self._sink_handlers = sink_handlers
        self._listener = listener
        self._drop_counter = drop_counter
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    @property
    def dropped_records(self) -> int:
        return self._drop_counter.value()

    @property
    def is_shutdown(self) -> bool:
        return self._is_shutdown

    def flush(self, *, timeout_seconds: float = 2.0) -> None:
        timeout = max(timeout_seconds, 0.0)
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.01)

        for handler in self._sink_handlers:
            handler.flush()

    def shutdown(self, *, timeout_seconds: float = 2.0) -> None:
        with self._shutdown_lock:
            if self._is_shutdown:
                return

            self.flush(timeout_seconds=timeout_seconds)
            self._listener.stop()

            self.logger.removeHandler(self._queue_handler)
            self._queue_handler.close()

            for handler in self._sink_handlers:
                handler.flush()
                handler.close()

            self._is_shutdown = True


def setup_structured_logging(config: LoggingConfig) -> StructuredLoggingHandle:
    """Configure queue-backed structured logging for a single run."""
    _shutdown_previous_active_handle()

    run_id = _validate_run_id(config.run_id)
    queue_size = _validate_queue_size(config.queue_size)
    log_filename = _validate_log_filename(config.log_filename)
    logger_name = _validate_logger_name(config.logger_name)
    level = _parse_log_level(config.level)
    base_log_dir = Path(config.base_log_dir)
    run_log_dir = base_log_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_log_dir / log_filename

    redactor = _resolve_redactor(config.redactor)
    formatter = _JsonLineFormatter(redactor=redactor, base_context={"run_id": run_id})

    file_handler: logging.Handler
    if config.rotating_file:
        max_bytes = max(1, int(config.max_bytes))
        backup_count = max(1, int(config.backup_count))
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    sink_handlers: list[logging.Handler] = [file_handler]
    if config.log_to_stdout:
        stdout_handler = logging.StreamHandler()
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(formatter)
        sink_handlers.append(stdout_handler)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    log_queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
    drop_counter = _DropCounter()
    queue_handler = _NonBlockingQueueHandler(log_queue, drop_counter)
    queue_handler.setLevel(level)

    listener = logging.handlers.QueueListener(
        log_queue,
        *sink_handlers,
        respect_handler_level=True,
    )
    listener.start()

    logger.addHandler(queue_handler)

    handle = StructuredLoggingHandle(
        logger=logger,
        run_id=run_id,
        run_log_dir=run_log_dir,
        log_path=log_path,
        queue_size=queue_size,
        log_queue=log_queue,
        queue_handler=queue_handler,
        sink_handlers=tuple(sink_handlers),
        listener=listener,
        drop_counter=drop_counter,
    )

    with _ACTIVE_HANDLE_LOCK:
        global _ACTIVE_HANDLE
        _ACTIVE_HANDLE = handle

    _register_atexit_shutdown()
    return handle


def flush_logging(
    handle: StructuredLoggingHandle | None = None,
    *,
    timeout_seconds: float = 2.0,
) -> None:
    """Flush queued logs to configured sinks."""
    resolved = _resolve_handle(handle)
    if resolved is not None:
        resolved.flush(timeout_seconds=timeout_seconds)


def shutdown_logging(
    handle: StructuredLoggingHandle | None = None,
    *,
    timeout_seconds: float = 2.0,
) -> None:
    """Shutdown logging listener and close all sinks."""
    resolved = _resolve_handle(handle)
    if resolved is None:
        return

    resolved.shutdown(timeout_seconds=timeout_seconds)

    with _ACTIVE_HANDLE_LOCK:
        global _ACTIVE_HANDLE
        if _ACTIVE_HANDLE is resolved:
            _ACTIVE_HANDLE = None


def get_active_logging_handle() -> StructuredLoggingHandle | None:
    """Return the currently active handle, if one exists."""
    with _ACTIVE_HANDLE_LOCK:
        return _ACTIVE_HANDLE


def get_active_logger() -> logging.Logger | None:
    """Return the active configured logger, when available."""
    handle = get_active_logging_handle()
    if handle is None:
        return None
    return handle.logger


def get_correlation_context() -> dict[str, str]:
    """Return the current correlation context as a plain dictionary."""
    return dict(_CORRELATION_CONTEXT.get())


def set_correlation_fields(**fields: str | None) -> contextvars.Token[_CorrelationState]:
    """Set correlation fields for the active context and return a reset token."""
    state = get_correlation_context()
    for key, value in fields.items():
        key_name = _validate_correlation_key(key)
        if value is None:
            state.pop(key_name, None)
            continue
        state[key_name] = _validate_correlation_value(value)
    return _CORRELATION_CONTEXT.set(tuple(state.items()))


def reset_correlation_fields(token: contextvars.Token[_CorrelationState]) -> None:
    """Reset correlation context to a previous token."""
    _CORRELATION_CONTEXT.reset(token)


@contextmanager
def correlation_scope(**fields: str | None) -> Iterator[None]:
    """Temporarily bind correlation fields for log records in scope."""
    token = set_correlation_fields(**fields)
    try:
        yield
    finally:
        reset_correlation_fields(token)


def default_log_redactor(value: JSONValue) -> JSONValue:
    """Default deep redaction for secrets and provider transcript fields."""
    return _redact_value(value, key_context=None)


def _resolve_handle(handle: StructuredLoggingHandle | None) -> StructuredLoggingHandle | None:
    if handle is not None:
        return handle
    return get_active_logging_handle()


def _shutdown_previous_active_handle() -> None:
    existing = get_active_logging_handle()
    if existing is not None:
        existing.shutdown()

    with _ACTIVE_HANDLE_LOCK:
        global _ACTIVE_HANDLE
        _ACTIVE_HANDLE = None


def _register_atexit_shutdown() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    atexit.register(_shutdown_active_handle)
    _ATEXIT_REGISTERED = True


def _shutdown_active_handle() -> None:
    shutdown_logging()


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise ValueError(f"run_id must be a string, got {type(run_id).__name__}")
    normalized = run_id.strip()
    if not normalized:
        raise ValueError("run_id must not be empty")
    return normalized


def _validate_queue_size(queue_size: int) -> int:
    if not isinstance(queue_size, int):
        raise ValueError(f"queue_size must be an integer, got {type(queue_size).__name__}")
    if queue_size <= 0:
        raise ValueError("queue_size must be > 0")
    return queue_size


def _validate_log_filename(log_filename: str) -> str:
    if not isinstance(log_filename, str):
        raise ValueError(f"log_filename must be a string, got {type(log_filename).__name__}")
    normalized = log_filename.strip()
    if not normalized:
        raise ValueError("log_filename must not be empty")
    if Path(normalized).name != normalized:
        raise ValueError("log_filename must not include path separators")
    return normalized


def _validate_logger_name(logger_name: str) -> str:
    if not isinstance(logger_name, str):
        raise ValueError(f"logger_name must be a string, got {type(logger_name).__name__}")
    normalized = logger_name.strip()
    if not normalized:
        raise ValueError("logger_name must not be empty")
    return normalized


def _parse_log_level(value: int | str) -> int:
    if isinstance(value, int):
        return value

    if not isinstance(value, str):
        raise ValueError(f"level must be int or str, got {type(value).__name__}")

    normalized = value.strip().upper()
    parsed = logging.getLevelName(normalized)
    if isinstance(parsed, int):
        return parsed

    raise ValueError(f"unsupported logging level {value!r}")


def _validate_correlation_key(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"correlation key must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError("correlation key must not be empty")
    return normalized


def _validate_correlation_value(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"correlation value must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError("correlation value must not be empty")
    return normalized


def _iso8601z_from_epoch(epoch_seconds: float) -> str:
    timestamp = datetime.fromtimestamp(epoch_seconds, tz=UTC)
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _coerce_log_message(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return str(value)
        return _REDACTED_VALUE
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _merge_correlation_context(
    record: logging.LogRecord,
    base_context: Mapping[str, str],
) -> dict[str, str]:
    merged = dict(base_context)
    merged.update(get_correlation_context())

    for key in _CORRELATION_KEYS:
        value = getattr(record, key, None)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                merged[key] = stripped

    user_context = getattr(record, "correlation", None)
    if isinstance(user_context, Mapping):
        for key, value in user_context.items():
            if isinstance(key, str) and isinstance(value, str):
                key_name = key.strip()
                val = value.strip()
                if key_name and val:
                    merged[key_name] = val

    return merged


def _extract_extra_fields(record: logging.LogRecord) -> dict[str, JSONValue]:
    fields: dict[str, JSONValue] = {}
    for key, value in record.__dict__.items():
        if key in _STANDARD_LOG_RECORD_FIELDS:
            continue
        if key in _CORRELATION_KEYS or key == "correlation":
            continue
        if key.startswith("_"):
            continue
        fields[key] = _normalize_json_value(value)
    return fields


def _normalize_json_value(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return _REDACTED_VALUE
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            normalized = value.replace(tzinfo=UTC)
        else:
            normalized = value.astimezone(UTC)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        output: dict[str, JSONValue] = {}
        for key, item in value.items():
            output[str(key)] = _normalize_json_value(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_normalize_json_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )
    return repr(value)


def _resolve_redactor(configured_redactor: LogRedactor | None) -> LogRedactor:
    if configured_redactor is not None:
        return _compose_redactor(configured_redactor)

    try:
        from nexus_orchestrator.security import redaction as security_redaction
    except Exception:
        return default_log_redactor

    for attr_name in ("redact_for_logging", "redact_value", "redact"):
        candidate = getattr(security_redaction, attr_name, None)
        if callable(candidate):
            return _compose_redactor(cast("Callable[[JSONValue], JSONValue]", candidate))

    return default_log_redactor


def _compose_redactor(candidate: Callable[[JSONValue], JSONValue]) -> LogRedactor:
    def composed(value: JSONValue) -> JSONValue:
        try:
            external_output = candidate(value)
        except Exception:
            external_output = value
        normalized = _normalize_json_value(external_output)
        return default_log_redactor(normalized)

    return composed


def _identity_redactor(value: JSONValue) -> JSONValue:
    return value


def _redact_value(value: JSONValue, *, key_context: str | None) -> JSONValue:
    if key_context is not None and _requires_redaction_for_key(key_context):
        return _REDACTED_VALUE

    if isinstance(value, str):
        return _redact_string(value)

    if isinstance(value, list):
        return [_redact_value(item, key_context=None) for item in value]

    if isinstance(value, dict):
        output: dict[str, JSONValue] = {}
        for key, item in value.items():
            output[key] = _redact_value(item, key_context=key)
        return output

    return value


def _requires_redaction_for_key(key: str) -> bool:
    key_lower = key.lower()
    return any(term in key_lower for term in _SENSITIVE_KEY_TERMS) or any(
        term in key_lower for term in _TRANSCRIPT_KEY_TERMS
    )


def _redact_string(text: str) -> str:
    redacted = _SENSITIVE_ASSIGNMENT_PATTERN.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED_VALUE}", text
    )
    redacted = _BEARER_TOKEN_PATTERN.sub(f"Bearer {_REDACTED_VALUE}", redacted)
    redacted = _OPENAI_KEY_PATTERN.sub(_REDACTED_VALUE, redacted)
    redacted = _ANTHROPIC_KEY_PATTERN.sub(_REDACTED_VALUE, redacted)
    return redacted


__all__ = [
    "JSONScalar",
    "JSONValue",
    "LogRedactor",
    "LoggingConfig",
    "StructuredLoggingHandle",
    "correlation_scope",
    "default_log_redactor",
    "flush_logging",
    "get_active_logger",
    "get_active_logging_handle",
    "get_correlation_context",
    "reset_correlation_fields",
    "set_correlation_fields",
    "setup_logging",
    "setup_structured_logging",
    "shutdown_logging",
]
