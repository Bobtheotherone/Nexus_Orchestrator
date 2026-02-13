"""
nexus-orchestrator — unit tests for observability logging

File: tests/unit/observability/test_logging.py
Last updated: 2026-02-12

Purpose
- Validate structured JSON logging with redaction, correlation metadata, and queue-backed reliability.

What this test file should cover
- JSON line validity and redaction guarantees.
- Correlation field propagation.
- Multi-threaded logging stability.
- Queue drain/shutdown behavior.

Functional requirements
- Offline operation.

Non-functional requirements
- Deterministic and non-flaky.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import threading
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from nexus_orchestrator.observability.logging import (
    LoggingConfig,
    correlation_scope,
    setup_logging,
    setup_structured_logging,
    shutdown_logging,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Iterator[None]:
    yield
    shutdown_logging()


def _logger_name() -> str:
    return f"nexus_orchestrator.tests.logging.{uuid4().hex}"


def _read_json_lines(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_json_logging_redacts_secrets_and_preserves_correlation_fields(tmp_path: Path) -> None:
    logger_name = _logger_name()
    handle = setup_structured_logging(
        LoggingConfig(
            run_id="run-logging-redaction",
            base_log_dir=tmp_path,
            logger_name=logger_name,
            log_to_stdout=False,
        )
    )
    logger = logging.getLogger(logger_name)

    with correlation_scope(work_item_id="wi-123", attempt_id="att-1"):
        logger.info(
            "payload token=tok-FAKE and api_key=sk-FAKE123456789012345",
            extra={"nested": {"password": "hunter2", "safe": "ok"}},
        )

    shutdown_logging(handle)

    assert handle.log_path.exists()
    parsed = _read_json_lines(handle.log_path)
    assert len(parsed) == 1
    first = parsed[0]
    assert first["run_id"] == "run-logging-redaction"
    assert first["work_item_id"] == "wi-123"
    assert first["attempt_id"] == "att-1"

    line = handle.log_path.read_text(encoding="utf-8")
    assert "tok-FAKE" not in line
    assert "sk-FAKE" not in line
    assert "hunter2" not in line
    assert "***REDACTED***" in line


def test_setup_logging_wrapper_uses_observability_config(tmp_path: Path) -> None:
    logger_name = _logger_name()
    logger = setup_logging(
        {
            "log_level": "INFO",
            "log_dir": str(tmp_path),
            "redact_secrets": True,
        },
        run_id="run-wrapper",
        logger_name=logger_name,
    )

    logger.info("hello", extra={"token": "t-123"})
    shutdown_logging()

    run_dir = tmp_path / "run-wrapper"
    files = list(run_dir.glob("*.jsonl"))
    assert files
    content = files[0].read_text(encoding="utf-8")
    assert "t-123" not in content


def test_multithreaded_logging_produces_valid_json_lines(tmp_path: Path) -> None:
    logger_name = _logger_name()
    handle = setup_structured_logging(
        LoggingConfig(
            run_id="run-threaded",
            base_log_dir=tmp_path,
            logger_name=logger_name,
            log_to_stdout=False,
            queue_size=4096,
        )
    )
    logger = logging.getLogger(logger_name)

    total_threads = 8
    per_thread = 40

    def worker(thread_idx: int) -> None:
        for i in range(per_thread):
            logger.info(
                f"thread={thread_idx} index={i} token=tok-secret-{thread_idx}-{i}",
                extra={"api_key": f"sk-FAKE-{thread_idx}-{i}"},
            )

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(total_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    shutdown_logging(handle)

    lines = handle.log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == total_threads * per_thread
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert "message" in parsed
        assert "tok-secret" not in line
        assert "sk-FAKE" not in line


def test_queue_handler_non_blocking_and_shutdown_flushes(tmp_path: Path) -> None:
    logger_name = _logger_name()
    handle = setup_structured_logging(
        LoggingConfig(
            run_id="run-flush",
            base_log_dir=tmp_path,
            logger_name=logger_name,
            log_to_stdout=False,
            queue_size=10_000,
        )
    )
    logger = logging.getLogger(logger_name)

    queue_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.QueueHandler)]
    assert queue_handlers, "expected queue-backed non-blocking logging"

    expected = 300
    for i in range(expected):
        logger.info("message %s", i)

    shutdown_logging(handle)

    lines = handle.log_path.read_text(encoding="utf-8").splitlines()
    assert handle.dropped_records == 0
    assert len(lines) == expected
