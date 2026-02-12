"""Public observability primitives: structured logging, metrics, and event streaming."""

from nexus_orchestrator.observability.events import (
    DispatchError,
    EventBus,
    PersistenceCallback,
    Subscriber,
)
from nexus_orchestrator.observability.logging import (
    LoggingConfig,
    LogRedactor,
    StructuredLoggingHandle,
    correlation_scope,
    default_log_redactor,
    flush_logging,
    get_active_logger,
    get_active_logging_handle,
    get_correlation_context,
    reset_correlation_fields,
    set_correlation_fields,
    setup_logging,
    setup_structured_logging,
    shutdown_logging,
)
from nexus_orchestrator.observability.metrics import MetricsRegistry

__all__ = [
    "DispatchError",
    "EventBus",
    "LogRedactor",
    "LoggingConfig",
    "MetricsRegistry",
    "PersistenceCallback",
    "StructuredLoggingHandle",
    "Subscriber",
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
