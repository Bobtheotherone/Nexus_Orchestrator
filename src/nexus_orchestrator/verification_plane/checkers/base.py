"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/base.py
Last updated: 2026-02-11

Purpose
- Defines the checker interface: inputs (workspace, config, constraints) and outputs (EvidenceRecord + structured result).

What should be included in this file
- Standard output fields: status, violated constraints, logs path, artifact paths, tool versions.
- How to declare which constraints a checker can satisfy.

Functional requirements
- Must support deterministic result formatting for the feedback synthesizer.

Non-functional requirements
- Must support redaction and safe logging.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, NoReturn, Protocol, TypeVar, runtime_checkable

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_MAX_TEXT_LENGTH = 8192
_MAX_JSON_DEPTH = 16
_MAX_JSON_COLLECTION = 512
_MAX_ENV_ENTRIES = 256
_MAX_TOOL_VERSIONS = 256


CheckerSource = Literal["builtin", "external"]
CheckerFactory = Callable[[], "BaseChecker"]
TextRedactor = Callable[[str], str]
MetadataRedactor = Callable[[Mapping[str, JSONValue]], Mapping[str, JSONValue]]


def _identity_text_redactor(text: str) -> str:
    return text


def _identity_metadata_redactor(metadata: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    return metadata


class CheckStatus(StrEnum):
    """Canonical checker statuses for deterministic downstream handling."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"
    TIMEOUT = "timeout"
    SKIP = "skip"


@dataclass(slots=True)
class Violation:
    """Machine-readable description of one failed constraint."""

    constraint_id: str
    code: str
    message: str
    severity: str = "error"
    path: str | None = None
    line: int | None = None
    column: int | None = None
    details: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.constraint_id = _as_str(self.constraint_id, "Violation.constraint_id", max_len=256)
        self.code = _as_str(self.code, "Violation.code", max_len=128)
        self.message = _as_str(self.message, "Violation.message")
        self.severity = _as_str(self.severity, "Violation.severity", max_len=64)
        self.path = _as_optional_str(self.path, "Violation.path", max_len=2048)
        self.line = _as_positive_int_or_none(self.line, "Violation.line")
        self.column = _as_positive_int_or_none(self.column, "Violation.column")
        self.details = _canonicalize_json_object(self.details, "Violation.details")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Violation:
        parsed = _expect_object(
            payload,
            "Violation",
            required={"constraint_id", "code", "message"},
            optional={"severity", "path", "line", "column", "details"},
        )
        return cls(
            constraint_id=_as_str(parsed["constraint_id"], "Violation.constraint_id", max_len=256),
            code=_as_str(parsed["code"], "Violation.code", max_len=128),
            message=_as_str(parsed["message"], "Violation.message"),
            severity=_as_str(parsed.get("severity", "error"), "Violation.severity", max_len=64),
            path=_as_optional_str(parsed.get("path"), "Violation.path", max_len=2048),
            line=_as_positive_int_or_none(parsed.get("line"), "Violation.line"),
            column=_as_positive_int_or_none(parsed.get("column"), "Violation.column"),
            details=_canonicalize_json_object(parsed.get("details", {}), "Violation.details"),
        )

    def sort_key(self) -> tuple[str, str, str, str, int, int, str, str]:
        """Deterministic sort key used by ``normalize_violations``."""

        return (
            self.constraint_id,
            self.code,
            self.severity,
            self.path or "",
            self.line if self.line is not None else -1,
            self.column if self.column is not None else -1,
            self.message,
            _canonical_json_dumps(self.details),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """Stable-key JSON-safe export."""

        return {
            "constraint_id": self.constraint_id,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "column": self.column,
            "details": self.details,
        }


@dataclass(slots=True)
class CheckResult:
    """Deterministic result envelope emitted by every checker."""

    status: CheckStatus
    violations: tuple[Violation, ...] = ()
    covered_constraint_ids: tuple[str, ...] = ()
    tool_versions: dict[str, str] = field(default_factory=dict)
    artifact_paths: tuple[str, ...] = ()
    logs_path: str | None = None
    command_lines: tuple[str, ...] = ()
    duration_ms: int = 0
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    checker_id: str = ""
    stage: str = ""

    def __post_init__(self) -> None:
        self.status = _as_check_status(self.status, "CheckResult.status")
        self.violations = normalize_violations(self.violations)
        self.covered_constraint_ids = _normalize_unique_strings(
            self.covered_constraint_ids,
            "CheckResult.covered_constraint_ids",
            max_len=256,
        )
        self.tool_versions = _as_ordered_str_mapping(
            self.tool_versions,
            "CheckResult.tool_versions",
            max_entries=_MAX_TOOL_VERSIONS,
            value_max_len=256,
        )
        self.artifact_paths = normalize_artifact_paths(self.artifact_paths)
        self.logs_path = _as_optional_str(self.logs_path, "CheckResult.logs_path", max_len=2048)
        self.command_lines = normalize_command_lines(self.command_lines)
        self.duration_ms = _as_int(self.duration_ms, "CheckResult.duration_ms", minimum=0)
        self.metadata = _canonicalize_json_object(self.metadata, "CheckResult.metadata")
        self.checker_id = _as_str(self.checker_id, "CheckResult.checker_id", max_len=128)
        self.stage = _as_str(self.stage, "CheckResult.stage", max_len=128)

    def to_dict(self) -> dict[str, JSONValue]:
        """Stable-key JSON-safe export used by evidence writers."""

        return {
            "status": self.status.value,
            "violations": [item.to_dict() for item in self.violations],
            "covered_constraint_ids": list(self.covered_constraint_ids),
            "tool_versions": dict(self.tool_versions),
            "artifact_paths": list(self.artifact_paths),
            "logs_path": self.logs_path,
            "command_lines": list(self.command_lines),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "checker_id": self.checker_id,
            "stage": self.stage,
        }

    def to_json(self) -> str:
        """Canonical JSON string suitable for hashing/comparison."""

        return _canonical_json_dumps(self.to_dict())


@dataclass(slots=True, frozen=True)
class RedactionHooks:
    """Hooks for checker text/metadata redaction."""

    redact_text: TextRedactor = _identity_text_redactor
    redact_metadata: MetadataRedactor = _identity_metadata_redactor

    def apply_text(self, text: str) -> str:
        parsed = _as_text(text, "RedactionHooks.apply_text")
        redacted = self.redact_text(parsed)
        return _as_text(redacted, "RedactionHooks.redact_text")

    def apply_metadata(self, metadata: Mapping[str, object]) -> dict[str, JSONValue]:
        parsed = _canonicalize_json_object(metadata, "RedactionHooks.apply_metadata")
        redacted = self.redact_metadata(parsed)
        return _canonicalize_json_object(redacted, "RedactionHooks.redact_metadata")


@dataclass(slots=True)
class CommandSpec:
    """Portable command invocation contract used by checkers."""

    argv: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    stdin_text: str | None = None
    timeout_seconds: float | None = None
    allowed_exit_codes: tuple[int, ...] = (0,)
    inherit_env: bool = True

    def __post_init__(self) -> None:
        self.argv = _as_non_empty_str_tuple(self.argv, "CommandSpec.argv", max_len=2048)
        self.cwd = _as_optional_str(self.cwd, "CommandSpec.cwd", max_len=4096)
        self.env = _as_ordered_str_mapping(
            self.env,
            "CommandSpec.env",
            max_entries=_MAX_ENV_ENTRIES,
            value_max_len=4096,
        )
        self.stdin_text = _as_optional_text(self.stdin_text, "CommandSpec.stdin_text")
        self.timeout_seconds = _as_positive_float_or_none(
            self.timeout_seconds,
            "CommandSpec.timeout_seconds",
        )
        self.allowed_exit_codes = _as_exit_codes(
            self.allowed_exit_codes,
            "CommandSpec.allowed_exit_codes",
        )
        self.inherit_env = _as_bool(self.inherit_env, "CommandSpec.inherit_env")

    def resolved_timeout(self, default_timeout_seconds: float | None = None) -> float | None:
        if self.timeout_seconds is not None:
            return self.timeout_seconds
        return _as_positive_float_or_none(default_timeout_seconds, "default_timeout_seconds")

    def build_env(self) -> dict[str, str] | None:
        if not self.inherit_env and not self.env:
            return {}
        if self.inherit_env:
            env = dict(os.environ)
            env.update(self.env)
            return env
        if not self.env:
            return None
        return dict(self.env)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "env": dict(self.env),
            "stdin_text": self.stdin_text,
            "timeout_seconds": self.timeout_seconds,
            "allowed_exit_codes": list(self.allowed_exit_codes),
            "inherit_env": self.inherit_env,
        }


@dataclass(slots=True)
class CommandResult:
    """Deterministic command execution outcome."""

    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        self.argv = _as_non_empty_str_tuple(self.argv, "CommandResult.argv", max_len=2048)
        self.exit_code = _as_int_or_none(self.exit_code, "CommandResult.exit_code")
        self.stdout = _as_text(self.stdout, "CommandResult.stdout")
        self.stderr = _as_text(self.stderr, "CommandResult.stderr")
        self.duration_ms = _as_int(self.duration_ms, "CommandResult.duration_ms", minimum=0)
        self.timed_out = _as_bool(self.timed_out, "CommandResult.timed_out")
        self.error = _as_optional_text(self.error, "CommandResult.error")
        if self.timed_out and self.exit_code is not None:
            _fail("CommandResult.exit_code", "must be None when timed_out is true")

    def is_success(self, spec: CommandSpec | None = None) -> bool:
        if self.timed_out or self.error is not None or self.exit_code is None:
            return False
        if spec is None:
            return self.exit_code == 0
        return self.exit_code in spec.allowed_exit_codes

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "argv": list(self.argv),
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "error": self.error,
        }


@runtime_checkable
class CommandExecutor(Protocol):
    """Pluggable async command execution interface for checkers."""

    async def run(self, spec: CommandSpec) -> CommandResult: ...


class LocalSubprocessExecutor(CommandExecutor):
    """Async local subprocess executor with deterministic capture/timeout behavior."""

    def __init__(
        self,
        *,
        default_timeout_seconds: float | None = None,
        max_output_chars: int | None = 200_000,
        redaction: RedactionHooks | None = None,
    ) -> None:
        self._default_timeout_seconds = _as_positive_float_or_none(
            default_timeout_seconds,
            "LocalSubprocessExecutor.default_timeout_seconds",
        )
        self._max_output_chars = _as_positive_int_or_none(
            max_output_chars,
            "LocalSubprocessExecutor.max_output_chars",
        )
        self._redaction = redaction if redaction is not None else RedactionHooks()

    async def run(self, spec: CommandSpec) -> CommandResult:
        started_ns = time.monotonic_ns()
        timeout = spec.resolved_timeout(self._default_timeout_seconds)

        try:
            process = await asyncio.create_subprocess_exec(
                *spec.argv,
                cwd=spec.cwd,
                env=spec.build_env(),
                stdin=(
                    asyncio.subprocess.PIPE
                    if spec.stdin_text is not None
                    else asyncio.subprocess.DEVNULL
                ),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            return CommandResult(
                argv=spec.argv,
                exit_code=None,
                stdout="",
                stderr="",
                duration_ms=_elapsed_ms(started_ns),
                timed_out=False,
                error=self._redaction.apply_text(str(exc)),
            )

        stdin_bytes = spec.stdin_text.encode("utf-8") if spec.stdin_text is not None else None

        try:
            stdout_bytes, stderr_bytes = await _communicate_with_timeout(
                process=process,
                stdin_bytes=stdin_bytes,
                timeout_seconds=timeout,
            )
            timed_out = False
            error_text: str | None = None
            exit_code = process.returncode
        except _CommandTimeoutError as exc:
            stdout_bytes = exc.stdout
            stderr_bytes = exc.stderr
            timed_out = True
            timeout_value = timeout if timeout is not None else 0.0
            error_text = f"command timed out after {timeout_value:.3f}s"
            exit_code = None

        stdout_text = self._redaction.apply_text(
            _truncate_text(_normalize_output_text(stdout_bytes), self._max_output_chars)
        )
        stderr_text = self._redaction.apply_text(
            _truncate_text(_normalize_output_text(stderr_bytes), self._max_output_chars)
        )

        return CommandResult(
            argv=spec.argv,
            exit_code=exit_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=_elapsed_ms(started_ns),
            timed_out=timed_out,
            error=self._redaction.apply_text(error_text) if error_text is not None else None,
        )


@dataclass(slots=True)
class CheckerContext:
    """Normalized checker invocation context."""

    workspace_path: str
    config: dict[str, JSONValue] = field(default_factory=dict)
    constraint_ids: tuple[str, ...] = ()
    redaction: RedactionHooks = field(default_factory=RedactionHooks)
    command_executor: CommandExecutor | None = None

    def __post_init__(self) -> None:
        self.workspace_path = _as_str(
            self.workspace_path,
            "CheckerContext.workspace_path",
            max_len=4096,
        )
        self.config = _canonicalize_json_object(self.config, "CheckerContext.config")
        self.constraint_ids = _normalize_unique_strings(
            self.constraint_ids,
            "CheckerContext.constraint_ids",
            max_len=256,
        )
        if not isinstance(self.redaction, RedactionHooks):
            _fail("CheckerContext.redaction", "must be a RedactionHooks instance")
        if self.command_executor is not None and not isinstance(
            self.command_executor, CommandExecutor
        ):
            _fail("CheckerContext.command_executor", "must implement CommandExecutor")

    def require_executor(self) -> CommandExecutor:
        if self.command_executor is None:
            _fail("CheckerContext.command_executor", "command executor is required")
        return self.command_executor


@runtime_checkable
class BaseChecker(Protocol):
    """Checker protocol implemented by built-ins and external plugins."""

    checker_id: str
    stage: str
    covered_constraint_ids: tuple[str, ...]

    async def check(self, context: CheckerContext) -> CheckResult: ...


@dataclass(frozen=True, slots=True)
class CheckerRegistration:
    checker_id: str
    source: CheckerSource
    factory: CheckerFactory


class CheckerRegistry:
    """Deterministic checker factory registry."""

    def __init__(self) -> None:
        self._registrations: dict[str, CheckerRegistration] = {}

    def register(self, checker_id: str, factory: CheckerFactory, *, source: CheckerSource) -> None:
        normalized_id = _as_str(checker_id, "checker_id", max_len=128)
        if not callable(factory):
            _fail("factory", "must be callable")

        existing = self._registrations.get(normalized_id)
        if existing is not None:
            _fail(
                "checker_id",
                f"already registered by {existing.source} checker '{existing.checker_id}'",
            )

        self._registrations[normalized_id] = CheckerRegistration(
            checker_id=normalized_id,
            source=source,
            factory=factory,
        )

    def register_builtin(self, checker_id: str, factory: CheckerFactory) -> None:
        self.register(checker_id, factory, source="builtin")

    def register_external(self, checker_id: str, factory: CheckerFactory) -> None:
        self.register(checker_id, factory, source="external")

    def register_external_plugins(self, plugins: Mapping[str, CheckerFactory]) -> None:
        for checker_id in sorted(plugins):
            self.register_external(checker_id, plugins[checker_id])

    def contains(self, checker_id: str) -> bool:
        normalized_id = _as_str(checker_id, "checker_id", max_len=128)
        return normalized_id in self._registrations

    def get_registration(self, checker_id: str) -> CheckerRegistration:
        normalized_id = _as_str(checker_id, "checker_id", max_len=128)
        registration = self._registrations.get(normalized_id)
        if registration is None:
            known = ", ".join(self.registered_ids())
            _fail("checker_id", f"unknown checker {normalized_id!r}; registered: [{known}]")
        return registration

    def get_factory(self, checker_id: str) -> CheckerFactory:
        return self.get_registration(checker_id).factory

    def create(self, checker_id: str) -> BaseChecker:
        checker = self.get_factory(checker_id)()
        if not isinstance(checker, BaseChecker):
            _fail("factory", f"'{checker_id}' factory did not return a BaseChecker")
        return checker

    def registered_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def registrations(self) -> tuple[CheckerRegistration, ...]:
        return tuple(self._registrations[key] for key in sorted(self._registrations))


CheckerType = TypeVar("CheckerType", bound=BaseChecker)

DEFAULT_CHECKER_REGISTRY = CheckerRegistry()


def register_builtin_checker(
    checker_id: str,
    *,
    registry: CheckerRegistry | None = None,
) -> Callable[[type[CheckerType]], type[CheckerType]]:
    """Decorator that registers built-in checker classes in deterministic registry order."""

    target = registry if registry is not None else DEFAULT_CHECKER_REGISTRY
    normalized_id = _as_str(checker_id, "checker_id", max_len=128)

    def decorator(checker_cls: type[CheckerType]) -> type[CheckerType]:
        _validate_zero_arg_constructor(checker_cls, checker_id=normalized_id)
        target.register_builtin(normalized_id, factory=lambda: checker_cls())
        return checker_cls

    return decorator


def register_external_checker(
    checker_id: str,
    factory: CheckerFactory,
    *,
    registry: CheckerRegistry | None = None,
) -> None:
    """External plugin surface for dynamic checker registration."""

    target = registry if registry is not None else DEFAULT_CHECKER_REGISTRY
    target.register_external(checker_id, factory)


def register_external_plugins(
    plugins: Mapping[str, CheckerFactory],
    *,
    registry: CheckerRegistry | None = None,
) -> None:
    """Bulk external plugin registration in deterministic key order."""

    target = registry if registry is not None else DEFAULT_CHECKER_REGISTRY
    target.register_external_plugins(plugins)


def normalize_violations(violations: Iterable[Violation]) -> tuple[Violation, ...]:
    """Return violations sorted deterministically by semantic key."""

    parsed: list[Violation] = []
    for index, item in enumerate(violations):
        if not isinstance(item, Violation):
            _fail(
                f"violations[{index}]",
                f"expected Violation, got {type(item).__name__}",
            )
        parsed.append(item)
    parsed.sort(key=lambda item: item.sort_key())
    return tuple(parsed)


def normalize_artifact_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Return unique artifact paths with deterministic lexical ordering."""

    normalized: set[str] = set()
    for index, item in enumerate(paths):
        normalized.add(_as_str(item, f"artifact_paths[{index}]", max_len=2048))
    return tuple(sorted(normalized))


def normalize_command_lines(command_lines: Iterable[str]) -> tuple[str, ...]:
    """Return unique command line strings with deterministic lexical ordering."""

    normalized: set[str] = set()
    for index, item in enumerate(command_lines):
        normalized.add(_as_str(item, f"command_lines[{index}]", max_len=4096))
    return tuple(sorted(normalized))


@dataclass(slots=True)
class _CommandTimeoutError(Exception):
    stdout: bytes
    stderr: bytes


async def _communicate_with_timeout(
    *,
    process: asyncio.subprocess.Process,
    stdin_bytes: bytes | None,
    timeout_seconds: float | None,
) -> tuple[bytes, bytes]:
    try:
        if timeout_seconds is None:
            return await process.communicate(stdin_bytes)
        return await asyncio.wait_for(process.communicate(stdin_bytes), timeout=timeout_seconds)
    except TimeoutError as exc:
        with suppress(ProcessLookupError):
            process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        raise _CommandTimeoutError(stdout=stdout_bytes, stderr=stderr_bytes) from exc
    except asyncio.CancelledError:
        with suppress(ProcessLookupError):
            process.kill()
        await process.communicate()
        raise


def _elapsed_ms(started_ns: int) -> int:
    delta_ns = time.monotonic_ns() - started_ns
    if delta_ns < 0:
        return 0
    return delta_ns // 1_000_000


def _normalize_output_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} chars]"


def _validate_zero_arg_constructor(checker_cls: type[object], *, checker_id: str) -> None:
    signature = inspect.signature(checker_cls)
    for parameter in signature.parameters.values():
        if (
            parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
            and parameter.default is inspect.Signature.empty
        ):
            _fail(
                "checker_cls",
                (
                    f"{checker_id!r} checker decorator requires a zero-arg constructor; "
                    f"parameter '{parameter.name}' is required"
                ),
            )


def _as_check_status(value: object, path: str) -> CheckStatus:
    if isinstance(value, CheckStatus):
        return value
    if not isinstance(value, str):
        _fail(path, f"expected CheckStatus or string, got {type(value).__name__}")
    try:
        return CheckStatus(value)
    except ValueError:
        allowed = ", ".join(item.value for item in CheckStatus)
        _fail(path, f"invalid value {value!r}; expected one of: {allowed}")


def _expect_object(
    value: object,
    path: str,
    *,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected object, got {type(value).__name__}")

    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail(path, f"object key must be string, got {type(key).__name__}")
        out[key] = item

    unknown = sorted(key for key in out if key not in required and key not in optional)
    if unknown:
        _fail(path, f"unexpected fields: {unknown}")

    missing = sorted(key for key in required if key not in out)
    if missing:
        _fail(path, f"missing required fields: {missing}")

    return out


def _as_bool(value: object, path: str) -> bool:
    if isinstance(value, bool):
        return value
    _fail(path, f"expected boolean, got {type(value).__name__}")


def _as_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(path, f"expected integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    return value


def _as_int_or_none(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, path)


def _as_positive_int_or_none(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, path, minimum=1)


def _as_positive_float_or_none(value: object, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected number, got {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(path, "must be finite")
    if parsed <= 0.0:
        _fail(path, "must be > 0")
    return parsed


def _as_str(
    value: object,
    path: str,
    *,
    min_len: int = 1,
    max_len: int = _MAX_TEXT_LENGTH,
) -> str:
    if not isinstance(value, str):
        _fail(path, f"expected string, got {type(value).__name__}")
    parsed = value.strip()
    if len(parsed) < min_len:
        _fail(path, f"must be at least {min_len} character(s)")
    if len(parsed) > max_len:
        _fail(path, f"must be <= {max_len} characters")
    return parsed


def _as_optional_str(value: object, path: str, *, max_len: int = _MAX_TEXT_LENGTH) -> str | None:
    if value is None:
        return None
    return _as_str(value, path, max_len=max_len)


def _as_text(value: object, path: str, *, max_len: int = 10_000_000) -> str:
    if not isinstance(value, str):
        _fail(path, f"expected string, got {type(value).__name__}")
    if len(value) > max_len:
        _fail(path, f"must be <= {max_len} characters")
    return value


def _as_optional_text(value: object, path: str, *, max_len: int = 10_000_000) -> str | None:
    if value is None:
        return None
    return _as_text(value, path, max_len=max_len)


def _as_non_empty_str_tuple(value: object, path: str, *, max_len: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(path, f"expected sequence, got {type(value).__name__}")

    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_as_str(item, f"{path}[{index}]", max_len=max_len))

    if not parsed:
        _fail(path, "must not be empty")

    return tuple(parsed)


def _as_exit_codes(value: object, path: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        _fail(path, f"expected sequence, got {type(value).__name__}")

    codes: set[int] = set()
    for index, item in enumerate(value):
        codes.add(_as_int(item, f"{path}[{index}]"))

    if not codes:
        _fail(path, "must not be empty")

    return tuple(sorted(codes))


def _normalize_unique_strings(
    values: Iterable[str],
    path: str,
    *,
    max_len: int,
) -> tuple[str, ...]:
    unique: set[str] = set()
    for index, item in enumerate(values):
        unique.add(_as_str(item, f"{path}[{index}]", max_len=max_len))
    return tuple(sorted(unique))


def _as_ordered_str_mapping(
    value: Mapping[str, str],
    path: str,
    *,
    max_entries: int,
    value_max_len: int,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected object, got {type(value).__name__}")
    if len(value) > max_entries:
        _fail(path, f"contains too many entries (>{max_entries})")

    parsed: dict[str, str] = {}
    for key, item in value.items():
        parsed_key = _as_str(key, f"{path}.<key>", max_len=128)
        parsed[parsed_key] = _as_str(item, f"{path}.{parsed_key}", max_len=value_max_len)

    return {key: parsed[key] for key in sorted(parsed)}


def _as_json_value(value: object, path: str, *, depth: int = 0) -> JSONValue:
    if depth > _MAX_JSON_DEPTH:
        _fail(path, f"JSON nesting exceeds max depth {_MAX_JSON_DEPTH}")

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "float values must be finite")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_TEXT_LENGTH:
            _fail(path, f"string exceeds max length {_MAX_TEXT_LENGTH}")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_JSON_COLLECTION:
            _fail(path, f"array length exceeds {_MAX_JSON_COLLECTION}")
        return [
            _as_json_value(item, f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        if len(value) > _MAX_JSON_COLLECTION:
            _fail(path, f"object size exceeds {_MAX_JSON_COLLECTION}")
        parsed: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, f"object key must be string, got {type(key).__name__}")
            parsed[key] = _as_json_value(item, f"{path}.{key}", depth=depth + 1)
        return parsed

    _fail(path, f"value is not JSON-serializable ({type(value).__name__})")


def _canonicalize_json_value(value: JSONValue) -> JSONValue:
    if isinstance(value, list):
        return [_canonicalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonicalize_json_value(value[key]) for key in sorted(value)}
    return value


def _canonicalize_json_object(value: object, path: str) -> dict[str, JSONValue]:
    parsed = _as_json_value(value, path)
    if not isinstance(parsed, dict):
        _fail(path, "expected JSON object")
    canonicalized = _canonicalize_json_value(parsed)
    if not isinstance(canonicalized, dict):
        _fail(path, "expected canonical JSON object")
    return canonicalized


def _canonical_json_dumps(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


__all__ = [
    "BaseChecker",
    "CheckResult",
    "CheckStatus",
    "CheckerContext",
    "CheckerFactory",
    "CheckerRegistration",
    "CheckerRegistry",
    "CheckerSource",
    "CommandExecutor",
    "CommandResult",
    "CommandSpec",
    "DEFAULT_CHECKER_REGISTRY",
    "JSONScalar",
    "JSONValue",
    "LocalSubprocessExecutor",
    "MetadataRedactor",
    "RedactionHooks",
    "TextRedactor",
    "Violation",
    "normalize_command_lines",
    "normalize_artifact_paths",
    "normalize_violations",
    "register_builtin_checker",
    "register_external_checker",
    "register_external_plugins",
]
