"""Dataclass domain models with strict validation and canonical serialization."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, NoReturn, TypeVar, cast

from nexus_orchestrator.domain import ids as domain_ids

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

TModel = TypeVar("TModel", bound="CanonicalModel")
TEnum = TypeVar("TEnum", bound=Enum)

_SCHEMA_VERSION = 1
_MAX_TEXT = 8192
_MAX_JSON_DEPTH = 16
_MAX_JSON_COLLECTION = 512
_MAX_EVIDENCE_METADATA_BYTES = 16 * 1024
_MAX_EVIDENCE_RECORD_BYTES = 32 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConstraintSeverity(StrEnum):
    MUST = "must"
    SHOULD = "should"
    MAY = "may"


class WorkItemStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    DISPATCHED = "dispatched"
    VERIFYING = "verifying"
    PASSED = "passed"
    FAILED = "failed"
    MERGED = "merged"


class AttemptResult(StrEnum):
    SUCCESS = "success"
    CONSTRAINT_FAILURE = "constraint_failure"
    ERROR = "error"
    TIMEOUT = "timeout"


class EvidenceResult(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class ConstraintSource(StrEnum):
    SPEC_DERIVED = "spec_derived"
    FAILURE_DERIVED = "failure_derived"
    MANUAL = "manual"


class RunStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CanonicalModel:
    """Mixin for canonical dict/json serialization."""

    def to_dict(self) -> dict[str, JSONValue]:
        serialized = _serialize_value(self, self.__class__.__name__)
        if not isinstance(serialized, dict):
            _fail(self.__class__.__name__, "serialized model must be an object")
        return serialized

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls: type[TModel], raw: str) -> TModel:
        if not isinstance(raw, str):
            _fail(cls.__name__, f"expected JSON string, got {type(raw).__name__}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            _fail(cls.__name__, f"invalid JSON: {exc}")
        if not isinstance(parsed, dict):
            _fail(cls.__name__, "JSON root must be an object")
        return cls.from_dict(parsed)

    @classmethod
    def from_dict(cls: type[TModel], data: Mapping[str, object]) -> TModel:
        _fail(cls.__name__, "from_dict is not implemented for this model type")


def _fail(path: str, message: str) -> NoReturn:
    raise ValueError(f"{path}: {message}")


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _expect_object(
    value: object,
    path: str,
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected object, got {type(value).__name__}")

    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail(path, f"object keys must be strings, got {type(key).__name__}")
        parsed[key] = item

    allowed = required | (optional or set())
    unknown = sorted(key for key in parsed if key not in allowed)
    if unknown:
        _fail(path, f"unexpected fields: {unknown}")

    missing = sorted(key for key in required if key not in parsed)
    if missing:
        _fail(path, f"missing required fields: {missing}")

    return parsed


def _as_schema_version(value: object, path: str) -> int:
    version = _as_int(value, path, minimum=1)
    return version


def _as_str(
    value: object,
    path: str,
    *,
    min_len: int = 1,
    max_len: int = _MAX_TEXT,
    strip: bool = True,
) -> str:
    if not isinstance(value, str):
        _fail(path, f"expected string, got {type(value).__name__}")
    normalized = value.strip() if strip else value
    if len(normalized) < min_len:
        _fail(path, f"must be at least {min_len} character(s)")
    if len(normalized) > max_len:
        _fail(path, f"must be <= {max_len} characters")
    return normalized


def _as_optional_str(value: object, path: str, *, max_len: int = _MAX_TEXT) -> str | None:
    if value is None:
        return None
    return _as_str(value, path, max_len=max_len)


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


def _as_float(value: object, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, f"expected number, got {type(value).__name__}")
    parsed = float(value)
    if not math.isfinite(parsed):
        _fail(path, "must be finite")
    if minimum is not None and parsed < minimum:
        _fail(path, f"must be >= {minimum}")
    return parsed


def _as_datetime(value: object, path: str) -> datetime:
    parsed: datetime
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            _fail(path, f"invalid ISO-8601 datetime: {value!r} ({exc})")
    else:
        _fail(path, f"expected datetime or ISO-8601 string, got {type(value).__name__}")

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(path, "datetime must be timezone-aware UTC")
    return parsed.astimezone(UTC)


def _datetime_to_iso8601z(value: datetime) -> str:
    normalized = _as_datetime(value, "datetime")
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _as_enum(enum_type: type[TEnum], value: object, path: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        _fail(path, f"expected string enum value, got {type(value).__name__}")
    try:
        return enum_type(value)
    except ValueError:
        allowed = ", ".join(sorted(item.value for item in enum_type))
        _fail(path, f"invalid value {value!r}; expected one of: {allowed}")


def _as_sequence(value: object, path: str) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    _fail(path, f"expected array, got {type(value).__name__}")


def _as_str_tuple(
    value: object,
    path: str,
    *,
    allow_empty: bool,
    unique: bool,
    max_len: int = _MAX_TEXT,
) -> tuple[str, ...]:
    values = _as_sequence(value, path)
    if not allow_empty and not values:
        _fail(path, "must not be empty")
    if len(values) > _MAX_JSON_COLLECTION:
        _fail(path, f"too many items (>{_MAX_JSON_COLLECTION})")

    parsed: list[str] = []
    for index, item in enumerate(values):
        parsed.append(_as_str(item, f"{path}[{index}]", max_len=max_len))

    if unique and len(set(parsed)) != len(parsed):
        _fail(path, "contains duplicate values")
    return tuple(parsed)


def _as_relative_path(value: object, path: str) -> str:
    parsed = _as_str(value, path, max_len=1024)
    if "\x00" in parsed:
        _fail(path, "must not contain NUL bytes")

    pure = PurePosixPath(parsed)
    if pure.is_absolute():
        _fail(path, "must be a relative POSIX path")
    if any(part == ".." for part in pure.parts):
        _fail(path, "must not contain '..' traversal")
    return parsed


def _as_sha256(value: object, path: str) -> str:
    parsed = _as_str(value, path, min_len=64, max_len=64, strip=False).lower()
    if not _SHA256_RE.fullmatch(parsed):
        _fail(path, "must be a 64-character lowercase hex SHA-256 digest")
    return parsed


def _as_commit_sha(value: object, path: str) -> str:
    parsed = _as_str(value, path, min_len=7, max_len=40, strip=False)
    if not _COMMIT_SHA_RE.fullmatch(parsed):
        _fail(path, "must be a lowercase hex commit SHA (7..40 chars)")
    return parsed


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
        if len(value) > _MAX_TEXT:
            _fail(path, f"string exceeds max length {_MAX_TEXT}")
        return value
    if isinstance(value, list):
        if len(value) > _MAX_JSON_COLLECTION:
            _fail(path, f"list length exceeds {_MAX_JSON_COLLECTION}")
        return [
            _as_json_value(item, f"{path}[{idx}]", depth=depth + 1)
            for idx, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        if len(value) > _MAX_JSON_COLLECTION:
            _fail(path, f"list length exceeds {_MAX_JSON_COLLECTION}")
        return [
            _as_json_value(item, f"{path}[{idx}]", depth=depth + 1)
            for idx, item in enumerate(value)
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


def _as_json_object(value: object, path: str) -> dict[str, JSONValue]:
    parsed = _as_json_value(value, path)
    if not isinstance(parsed, dict):
        _fail(path, "expected JSON object")
    return parsed


def _enforce_json_string_limit(value: JSONValue, path: str, *, max_len: int) -> None:
    if isinstance(value, str):
        if len(value) > max_len:
            _fail(path, f"string exceeds max length {max_len}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _enforce_json_string_limit(item, f"{path}[{index}]", max_len=max_len)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _enforce_json_string_limit(item, f"{path}.{key}", max_len=max_len)


def _as_str_dict(value: object, path: str, *, max_entries: int = 128) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(path, f"expected object, got {type(value).__name__}")
    if len(value) > max_entries:
        _fail(path, f"contains too many entries (>{max_entries})")

    parsed: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            _fail(path, f"key must be string, got {type(key).__name__}")
        parsed_key = _as_str(key, f"{path}.<key>")
        parsed[parsed_key] = _as_str(item, f"{path}.{parsed_key}")
    return parsed


def _serialize_value(value: object, path: str) -> JSONValue:
    if value is None or isinstance(value, bool):
        return cast("JSONValue", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "float values must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        raw = value.value
        if not isinstance(raw, str):
            _fail(path, "enum value must be string")
        return raw
    if isinstance(value, datetime):
        return _datetime_to_iso8601z(value)
    if isinstance(value, tuple):
        return [_serialize_value(item, f"{path}[]") for item in value]
    if isinstance(value, list):
        return [_serialize_value(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(path, "dict keys must be strings")
            out[key] = _serialize_value(item, f"{path}.{key}")
        return out
    if is_dataclass(value):
        out_obj: dict[str, JSONValue] = {}
        for dataclass_field in fields(value):
            out_obj[dataclass_field.name] = _serialize_value(
                getattr(value, dataclass_field.name),
                f"{path}.{dataclass_field.name}",
            )
        return out_obj

    _fail(path, f"cannot serialize value of type {type(value).__name__}")


def _validate_requirement_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_requirement_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_constraint_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_constraint_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_work_item_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_work_item_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_run_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_run_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_attempt_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_attempt_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_evidence_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_evidence_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_merge_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_merge_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_incident_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_incident_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _validate_artifact_id(value: str, path: str) -> str:
    try:
        domain_ids.validate_artifact_id(value)
    except ValueError as exc:
        _fail(path, str(exc))
    return value


def _parse_constraint_links(value: object, path: str) -> tuple[str, ...]:
    parsed = _as_str_tuple(value, path, allow_empty=True, unique=True)
    for index, item in enumerate(parsed):
        _validate_constraint_id(item, f"{path}[{index}]")
    return parsed


def _parse_requirement_links(value: object, path: str) -> tuple[str, ...]:
    parsed = _as_str_tuple(value, path, allow_empty=True, unique=True)
    for index, item in enumerate(parsed):
        _validate_requirement_id(item, f"{path}[{index}]")
    return parsed


def _parse_evidence_ids(value: object, path: str) -> tuple[str, ...]:
    parsed = _as_str_tuple(value, path, allow_empty=True, unique=True)
    for index, item in enumerate(parsed):
        _validate_evidence_id(item, f"{path}[{index}]")
    return parsed


@dataclass(slots=True)
class Requirement(CanonicalModel):
    id: str
    statement: str
    acceptance_criteria: tuple[str, ...] = ()
    nfr_tags: tuple[str, ...] = ()
    source: str = "unknown"
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Requirement.schema_version")
        self.id = _validate_requirement_id(_as_str(self.id, "Requirement.id"), "Requirement.id")
        self.statement = _as_str(self.statement, "Requirement.statement")
        self.acceptance_criteria = _as_str_tuple(
            self.acceptance_criteria,
            "Requirement.acceptance_criteria",
            allow_empty=True,
            unique=False,
        )
        self.nfr_tags = _as_str_tuple(
            self.nfr_tags, "Requirement.nfr_tags", allow_empty=True, unique=True
        )
        self.source = _as_str(self.source, "Requirement.source")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Requirement:
        parsed = _expect_object(
            data,
            "Requirement",
            required={"id", "statement"},
            optional={"acceptance_criteria", "nfr_tags", "source", "schema_version"},
        )
        return cls(
            id=_as_str(parsed["id"], "Requirement.id"),
            statement=_as_str(parsed["statement"], "Requirement.statement"),
            acceptance_criteria=_as_str_tuple(
                parsed.get("acceptance_criteria", ()),
                "Requirement.acceptance_criteria",
                allow_empty=True,
                unique=False,
            ),
            nfr_tags=_as_str_tuple(
                parsed.get("nfr_tags", ()),
                "Requirement.nfr_tags",
                allow_empty=True,
                unique=True,
            ),
            source=_as_str(parsed.get("source", "unknown"), "Requirement.source"),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "Requirement.schema_version",
            ),
        )


@dataclass(slots=True)
class Budget(CanonicalModel):
    max_tokens: int
    max_cost_usd: float
    max_iterations: int
    max_wall_clock_seconds: int
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Budget.schema_version")
        self.max_tokens = _as_int(self.max_tokens, "Budget.max_tokens", minimum=1)
        self.max_cost_usd = _as_float(self.max_cost_usd, "Budget.max_cost_usd", minimum=0.0)
        self.max_iterations = _as_int(self.max_iterations, "Budget.max_iterations", minimum=1)
        self.max_wall_clock_seconds = _as_int(
            self.max_wall_clock_seconds,
            "Budget.max_wall_clock_seconds",
            minimum=1,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Budget:
        parsed = _expect_object(
            data,
            "Budget",
            required={"max_tokens", "max_cost_usd", "max_iterations", "max_wall_clock_seconds"},
            optional={"schema_version"},
        )
        return cls(
            max_tokens=_as_int(parsed["max_tokens"], "Budget.max_tokens", minimum=1),
            max_cost_usd=_as_float(parsed["max_cost_usd"], "Budget.max_cost_usd", minimum=0.0),
            max_iterations=_as_int(parsed["max_iterations"], "Budget.max_iterations", minimum=1),
            max_wall_clock_seconds=_as_int(
                parsed["max_wall_clock_seconds"],
                "Budget.max_wall_clock_seconds",
                minimum=1,
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION), "Budget.schema_version"
            ),
        )


@dataclass(slots=True)
class SandboxPolicy(CanonicalModel):
    allow_network: bool
    allow_privileged_tools: bool
    allowed_tools: tuple[str, ...] = ()
    read_only_paths: tuple[str, ...] = ()
    write_paths: tuple[str, ...] = ()
    max_cpu_seconds: int | None = None
    max_memory_mb: int | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(
            self.schema_version, "SandboxPolicy.schema_version"
        )
        self.allow_network = _as_bool(self.allow_network, "SandboxPolicy.allow_network")
        self.allow_privileged_tools = _as_bool(
            self.allow_privileged_tools,
            "SandboxPolicy.allow_privileged_tools",
        )
        self.allowed_tools = _as_str_tuple(
            self.allowed_tools,
            "SandboxPolicy.allowed_tools",
            allow_empty=True,
            unique=True,
            max_len=128,
        )
        self.read_only_paths = tuple(
            _as_relative_path(item, f"SandboxPolicy.read_only_paths[{idx}]")
            for idx, item in enumerate(
                _as_str_tuple(
                    self.read_only_paths,
                    "SandboxPolicy.read_only_paths",
                    allow_empty=True,
                    unique=True,
                    max_len=1024,
                )
            )
        )
        self.write_paths = tuple(
            _as_relative_path(item, f"SandboxPolicy.write_paths[{idx}]")
            for idx, item in enumerate(
                _as_str_tuple(
                    self.write_paths,
                    "SandboxPolicy.write_paths",
                    allow_empty=True,
                    unique=True,
                    max_len=1024,
                )
            )
        )
        if self.max_cpu_seconds is not None:
            self.max_cpu_seconds = _as_int(
                self.max_cpu_seconds, "SandboxPolicy.max_cpu_seconds", minimum=1
            )
        if self.max_memory_mb is not None:
            self.max_memory_mb = _as_int(
                self.max_memory_mb, "SandboxPolicy.max_memory_mb", minimum=1
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SandboxPolicy:
        parsed = _expect_object(
            data,
            "SandboxPolicy",
            required={"allow_network", "allow_privileged_tools"},
            optional={
                "allowed_tools",
                "read_only_paths",
                "write_paths",
                "max_cpu_seconds",
                "max_memory_mb",
                "schema_version",
            },
        )
        return cls(
            allow_network=_as_bool(parsed["allow_network"], "SandboxPolicy.allow_network"),
            allow_privileged_tools=_as_bool(
                parsed["allow_privileged_tools"],
                "SandboxPolicy.allow_privileged_tools",
            ),
            allowed_tools=_as_str_tuple(
                parsed.get("allowed_tools", ()),
                "SandboxPolicy.allowed_tools",
                allow_empty=True,
                unique=True,
                max_len=128,
            ),
            read_only_paths=_as_str_tuple(
                parsed.get("read_only_paths", ()),
                "SandboxPolicy.read_only_paths",
                allow_empty=True,
                unique=True,
                max_len=1024,
            ),
            write_paths=_as_str_tuple(
                parsed.get("write_paths", ()),
                "SandboxPolicy.write_paths",
                allow_empty=True,
                unique=True,
                max_len=1024,
            ),
            max_cpu_seconds=(
                _as_int(parsed["max_cpu_seconds"], "SandboxPolicy.max_cpu_seconds", minimum=1)
                if parsed.get("max_cpu_seconds") is not None
                else None
            ),
            max_memory_mb=(
                _as_int(parsed["max_memory_mb"], "SandboxPolicy.max_memory_mb", minimum=1)
                if parsed.get("max_memory_mb") is not None
                else None
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "SandboxPolicy.schema_version",
            ),
        )


@dataclass(slots=True)
class Constraint(CanonicalModel):
    id: str
    severity: ConstraintSeverity
    category: str
    description: str
    checker_binding: str
    parameters: dict[str, JSONValue]
    requirement_links: tuple[str, ...]
    source: ConstraintSource
    created_at: datetime
    failure_history: tuple[str, ...] = ()
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Constraint.schema_version")
        self.id = _validate_constraint_id(_as_str(self.id, "Constraint.id"), "Constraint.id")
        self.severity = _as_enum(ConstraintSeverity, self.severity, "Constraint.severity")
        self.category = _as_str(self.category, "Constraint.category", max_len=128)
        self.description = _as_str(self.description, "Constraint.description")
        self.checker_binding = _as_str(
            self.checker_binding, "Constraint.checker_binding", max_len=128
        )
        self.parameters = _as_json_object(self.parameters, "Constraint.parameters")
        self.requirement_links = _parse_requirement_links(
            self.requirement_links,
            "Constraint.requirement_links",
        )
        self.source = _as_enum(ConstraintSource, self.source, "Constraint.source")
        self.created_at = _as_datetime(self.created_at, "Constraint.created_at")
        self.failure_history = _as_str_tuple(
            self.failure_history,
            "Constraint.failure_history",
            allow_empty=True,
            unique=False,
            max_len=1024,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Constraint:
        parsed = _expect_object(
            data,
            "Constraint",
            required={
                "id",
                "severity",
                "category",
                "description",
                "checker_binding",
                "parameters",
                "requirement_links",
                "source",
                "created_at",
            },
            optional={"failure_history", "schema_version"},
        )
        return cls(
            id=_as_str(parsed["id"], "Constraint.id"),
            severity=_as_enum(ConstraintSeverity, parsed["severity"], "Constraint.severity"),
            category=_as_str(parsed["category"], "Constraint.category", max_len=128),
            description=_as_str(parsed["description"], "Constraint.description"),
            checker_binding=_as_str(
                parsed["checker_binding"], "Constraint.checker_binding", max_len=128
            ),
            parameters=_as_json_object(parsed["parameters"], "Constraint.parameters"),
            requirement_links=_parse_requirement_links(
                parsed["requirement_links"],
                "Constraint.requirement_links",
            ),
            source=_as_enum(ConstraintSource, parsed["source"], "Constraint.source"),
            created_at=_as_datetime(parsed["created_at"], "Constraint.created_at"),
            failure_history=_as_str_tuple(
                parsed.get("failure_history", ()),
                "Constraint.failure_history",
                allow_empty=True,
                unique=False,
                max_len=1024,
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "Constraint.schema_version",
            ),
        )


@dataclass(slots=True)
class ConstraintEnvelope(CanonicalModel):
    work_item_id: str
    constraints: tuple[Constraint, ...]
    inherited_constraint_ids: tuple[str, ...] = ()
    compiled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(
            self.schema_version,
            "ConstraintEnvelope.schema_version",
        )
        self.work_item_id = _validate_work_item_id(
            _as_str(self.work_item_id, "ConstraintEnvelope.work_item_id"),
            "ConstraintEnvelope.work_item_id",
        )

        raw_constraints = _as_sequence(self.constraints, "ConstraintEnvelope.constraints")
        if not raw_constraints:
            _fail("ConstraintEnvelope.constraints", "must include at least one constraint")

        parsed_constraints: list[Constraint] = []
        for index, item in enumerate(raw_constraints):
            if isinstance(item, Constraint):
                parsed_constraints.append(item)
            elif isinstance(item, Mapping):
                parsed_constraints.append(Constraint.from_dict(item))
            else:
                _fail(
                    f"ConstraintEnvelope.constraints[{index}]",
                    f"expected Constraint object, got {type(item).__name__}",
                )
        self.constraints = tuple(parsed_constraints)

        self.inherited_constraint_ids = _parse_constraint_links(
            self.inherited_constraint_ids,
            "ConstraintEnvelope.inherited_constraint_ids",
        )
        self.compiled_at = _as_datetime(self.compiled_at, "ConstraintEnvelope.compiled_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ConstraintEnvelope:
        parsed = _expect_object(
            data,
            "ConstraintEnvelope",
            required={"work_item_id", "constraints"},
            optional={"inherited_constraint_ids", "compiled_at", "schema_version"},
        )
        constraints_raw = _as_sequence(parsed["constraints"], "ConstraintEnvelope.constraints")
        constraints = tuple(
            item
            if isinstance(item, Constraint)
            else Constraint.from_dict(
                _expect_object(
                    item,
                    "ConstraintEnvelope.constraints[]",
                    required={
                        "id",
                        "severity",
                        "category",
                        "description",
                        "checker_binding",
                        "parameters",
                        "requirement_links",
                        "source",
                        "created_at",
                    },
                    optional={"failure_history", "schema_version"},
                )
            )
            for item in constraints_raw
        )
        return cls(
            work_item_id=_as_str(parsed["work_item_id"], "ConstraintEnvelope.work_item_id"),
            constraints=constraints,
            inherited_constraint_ids=_parse_constraint_links(
                parsed.get("inherited_constraint_ids", ()),
                "ConstraintEnvelope.inherited_constraint_ids",
            ),
            compiled_at=_as_datetime(
                parsed.get("compiled_at", datetime.now(UTC)),
                "ConstraintEnvelope.compiled_at",
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "ConstraintEnvelope.schema_version",
            ),
        )


def _default_budget() -> Budget:
    return Budget(
        max_tokens=10_000,
        max_cost_usd=5.0,
        max_iterations=3,
        max_wall_clock_seconds=300,
    )


def _default_sandbox_policy() -> SandboxPolicy:
    return SandboxPolicy(
        allow_network=False,
        allow_privileged_tools=False,
        allowed_tools=(),
        read_only_paths=(),
        write_paths=("src", "tests"),
    )


@dataclass(slots=True)
class WorkItem(CanonicalModel):
    id: str
    title: str
    description: str
    scope: tuple[str, ...]
    constraint_envelope: ConstraintEnvelope
    dependencies: tuple[str, ...] = ()
    status: WorkItemStatus = WorkItemStatus.PENDING
    risk_tier: RiskTier = RiskTier.MEDIUM
    budget: Budget = field(default_factory=_default_budget)
    sandbox_policy: SandboxPolicy = field(default_factory=_default_sandbox_policy)
    requirement_links: tuple[str, ...] = ()
    constraint_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    commit_sha: str | None = None
    expected_artifacts: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "WorkItem.schema_version")
        self.id = _validate_work_item_id(_as_str(self.id, "WorkItem.id"), "WorkItem.id")
        self.title = _as_str(self.title, "WorkItem.title", max_len=256)
        self.description = _as_str(self.description, "WorkItem.description")

        scope_items = _as_str_tuple(
            self.scope, "WorkItem.scope", allow_empty=False, unique=True, max_len=1024
        )
        self.scope = tuple(
            _as_relative_path(item, f"WorkItem.scope[{idx}]")
            for idx, item in enumerate(scope_items)
        )

        if not isinstance(self.constraint_envelope, ConstraintEnvelope):
            _fail("WorkItem.constraint_envelope", "must be ConstraintEnvelope")

        self.dependencies = _as_str_tuple(
            self.dependencies,
            "WorkItem.dependencies",
            allow_empty=True,
            unique=True,
        )
        for index, dep in enumerate(self.dependencies):
            _validate_work_item_id(dep, f"WorkItem.dependencies[{index}]")

        self.status = _as_enum(WorkItemStatus, self.status, "WorkItem.status")
        self.risk_tier = _as_enum(RiskTier, self.risk_tier, "WorkItem.risk_tier")

        if not isinstance(self.budget, Budget):
            _fail("WorkItem.budget", "must be Budget")
        if not isinstance(self.sandbox_policy, SandboxPolicy):
            _fail("WorkItem.sandbox_policy", "must be SandboxPolicy")

        self.requirement_links = _parse_requirement_links(
            self.requirement_links,
            "WorkItem.requirement_links",
        )
        self.constraint_ids = _parse_constraint_links(
            self.constraint_ids, "WorkItem.constraint_ids"
        )
        self.evidence_ids = _parse_evidence_ids(self.evidence_ids, "WorkItem.evidence_ids")

        if self.commit_sha is not None:
            self.commit_sha = _as_commit_sha(self.commit_sha, "WorkItem.commit_sha")

        artifact_paths = _as_str_tuple(
            self.expected_artifacts,
            "WorkItem.expected_artifacts",
            allow_empty=True,
            unique=True,
            max_len=1024,
        )
        self.expected_artifacts = tuple(
            _as_relative_path(item, f"WorkItem.expected_artifacts[{idx}]")
            for idx, item in enumerate(artifact_paths)
        )

        self.created_at = _as_datetime(self.created_at, "WorkItem.created_at")
        self.updated_at = _as_datetime(self.updated_at, "WorkItem.updated_at")
        if self.updated_at < self.created_at:
            _fail("WorkItem.updated_at", "must be >= WorkItem.created_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> WorkItem:
        parsed = _expect_object(
            data,
            "WorkItem",
            required={"id", "title", "description", "scope", "constraint_envelope"},
            optional={
                "dependencies",
                "status",
                "risk_tier",
                "budget",
                "sandbox_policy",
                "requirement_links",
                "constraint_ids",
                "evidence_ids",
                "commit_sha",
                "expected_artifacts",
                "created_at",
                "updated_at",
                "schema_version",
            },
        )

        envelope_raw = parsed["constraint_envelope"]
        if isinstance(envelope_raw, ConstraintEnvelope):
            envelope = envelope_raw
        else:
            envelope = ConstraintEnvelope.from_dict(
                _expect_object(
                    envelope_raw,
                    "WorkItem.constraint_envelope",
                    required={"work_item_id", "constraints"},
                    optional={"inherited_constraint_ids", "compiled_at", "schema_version"},
                )
            )

        budget_raw = parsed.get("budget", _default_budget())
        budget = (
            budget_raw
            if isinstance(budget_raw, Budget)
            else Budget.from_dict(
                _expect_object(
                    budget_raw,
                    "WorkItem.budget",
                    required={
                        "max_tokens",
                        "max_cost_usd",
                        "max_iterations",
                        "max_wall_clock_seconds",
                    },
                    optional={"schema_version"},
                )
            )
        )

        sandbox_raw = parsed.get("sandbox_policy", _default_sandbox_policy())
        sandbox_policy = (
            sandbox_raw
            if isinstance(sandbox_raw, SandboxPolicy)
            else SandboxPolicy.from_dict(
                _expect_object(
                    sandbox_raw,
                    "WorkItem.sandbox_policy",
                    required={"allow_network", "allow_privileged_tools"},
                    optional={
                        "allowed_tools",
                        "read_only_paths",
                        "write_paths",
                        "max_cpu_seconds",
                        "max_memory_mb",
                        "schema_version",
                    },
                )
            )
        )

        return cls(
            id=_as_str(parsed["id"], "WorkItem.id"),
            title=_as_str(parsed["title"], "WorkItem.title", max_len=256),
            description=_as_str(parsed["description"], "WorkItem.description"),
            scope=_as_str_tuple(
                parsed["scope"], "WorkItem.scope", allow_empty=False, unique=True, max_len=1024
            ),
            constraint_envelope=envelope,
            dependencies=_as_str_tuple(
                parsed.get("dependencies", ()),
                "WorkItem.dependencies",
                allow_empty=True,
                unique=True,
            ),
            status=_as_enum(
                WorkItemStatus, parsed.get("status", WorkItemStatus.PENDING), "WorkItem.status"
            ),
            risk_tier=_as_enum(
                RiskTier, parsed.get("risk_tier", RiskTier.MEDIUM), "WorkItem.risk_tier"
            ),
            budget=budget,
            sandbox_policy=sandbox_policy,
            requirement_links=_parse_requirement_links(
                parsed.get("requirement_links", ()),
                "WorkItem.requirement_links",
            ),
            constraint_ids=_parse_constraint_links(
                parsed.get("constraint_ids", ()), "WorkItem.constraint_ids"
            ),
            evidence_ids=_parse_evidence_ids(
                parsed.get("evidence_ids", ()), "WorkItem.evidence_ids"
            ),
            commit_sha=(
                _as_commit_sha(parsed["commit_sha"], "WorkItem.commit_sha")
                if parsed.get("commit_sha") is not None
                else None
            ),
            expected_artifacts=_as_str_tuple(
                parsed.get("expected_artifacts", ()),
                "WorkItem.expected_artifacts",
                allow_empty=True,
                unique=True,
                max_len=1024,
            ),
            created_at=_as_datetime(
                parsed.get("created_at", datetime.now(UTC)), "WorkItem.created_at"
            ),
            updated_at=_as_datetime(
                parsed.get("updated_at", datetime.now(UTC)), "WorkItem.updated_at"
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "WorkItem.schema_version",
            ),
        )


@dataclass(slots=True)
class TaskGraph(CanonicalModel):
    run_id: str
    work_items: tuple[WorkItem, ...]
    edges: tuple[tuple[str, str], ...]
    critical_path: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "TaskGraph.schema_version")
        self.run_id = _validate_run_id(_as_str(self.run_id, "TaskGraph.run_id"), "TaskGraph.run_id")

        raw_items = _as_sequence(self.work_items, "TaskGraph.work_items")
        if not raw_items:
            _fail("TaskGraph.work_items", "must include at least one work item")
        parsed_items: list[WorkItem] = []
        for index, item in enumerate(raw_items):
            if isinstance(item, WorkItem):
                parsed_items.append(item)
            elif isinstance(item, Mapping):
                parsed_items.append(WorkItem.from_dict(item))
            else:
                _fail(
                    f"TaskGraph.work_items[{index}]",
                    f"expected WorkItem, got {type(item).__name__}",
                )
        self.work_items = tuple(parsed_items)

        work_item_ids = {item.id for item in self.work_items}
        if len(work_item_ids) != len(self.work_items):
            _fail("TaskGraph.work_items", "duplicate work item IDs")

        edge_values = _as_sequence(self.edges, "TaskGraph.edges")
        parsed_edges: list[tuple[str, str]] = []
        for index, pair in enumerate(edge_values):
            pair_list = _as_sequence(pair, f"TaskGraph.edges[{index}]")
            if len(pair_list) != 2:
                _fail(f"TaskGraph.edges[{index}]", "edge must contain exactly 2 IDs")
            src = _validate_work_item_id(
                _as_str(pair_list[0], f"TaskGraph.edges[{index}][0]"),
                f"TaskGraph.edges[{index}][0]",
            )
            dst = _validate_work_item_id(
                _as_str(pair_list[1], f"TaskGraph.edges[{index}][1]"),
                f"TaskGraph.edges[{index}][1]",
            )
            if src not in work_item_ids or dst not in work_item_ids:
                _fail(
                    f"TaskGraph.edges[{index}]",
                    "edge references work item not present in graph",
                )
            parsed_edges.append((src, dst))
        self.edges = tuple(parsed_edges)

        self.critical_path = _as_str_tuple(
            self.critical_path,
            "TaskGraph.critical_path",
            allow_empty=True,
            unique=False,
        )
        for index, item in enumerate(self.critical_path):
            _validate_work_item_id(item, f"TaskGraph.critical_path[{index}]")
            if item not in work_item_ids:
                _fail(f"TaskGraph.critical_path[{index}]", "work item not found in graph")

        self.created_at = _as_datetime(self.created_at, "TaskGraph.created_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TaskGraph:
        parsed = _expect_object(
            data,
            "TaskGraph",
            required={"run_id", "work_items", "edges"},
            optional={"critical_path", "created_at", "schema_version"},
        )

        work_items_raw = _as_sequence(parsed["work_items"], "TaskGraph.work_items")
        work_items = tuple(
            item
            if isinstance(item, WorkItem)
            else WorkItem.from_dict(
                _expect_object(
                    item,
                    "TaskGraph.work_items[]",
                    required={"id", "title", "description", "scope", "constraint_envelope"},
                    optional={
                        "dependencies",
                        "status",
                        "risk_tier",
                        "budget",
                        "sandbox_policy",
                        "requirement_links",
                        "constraint_ids",
                        "evidence_ids",
                        "commit_sha",
                        "expected_artifacts",
                        "created_at",
                        "updated_at",
                        "schema_version",
                    },
                )
            )
            for item in work_items_raw
        )

        edge_values = _as_sequence(parsed["edges"], "TaskGraph.edges")
        edges: tuple[tuple[str, str], ...] = tuple(
            (
                _as_str(_as_sequence(edge, "TaskGraph.edges[]")[0], "TaskGraph.edges[][0]"),
                _as_str(_as_sequence(edge, "TaskGraph.edges[]")[1], "TaskGraph.edges[][1]"),
            )
            for edge in edge_values
        )

        return cls(
            run_id=_as_str(parsed["run_id"], "TaskGraph.run_id"),
            work_items=work_items,
            edges=edges,
            critical_path=_as_str_tuple(
                parsed.get("critical_path", ()),
                "TaskGraph.critical_path",
                allow_empty=True,
                unique=False,
            ),
            created_at=_as_datetime(
                parsed.get("created_at", datetime.now(UTC)), "TaskGraph.created_at"
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "TaskGraph.schema_version",
            ),
        )


@dataclass(slots=True)
class SpecMap(CanonicalModel):
    source_document: str
    requirements: tuple[Requirement, ...]
    created_at: datetime
    entities: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    glossary: dict[str, str] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "SpecMap.schema_version")
        self.source_document = _as_relative_path(self.source_document, "SpecMap.source_document")

        raw_reqs = _as_sequence(self.requirements, "SpecMap.requirements")
        if not raw_reqs:
            _fail("SpecMap.requirements", "must include at least one requirement")
        parsed_requirements: list[Requirement] = []
        for index, item in enumerate(raw_reqs):
            if isinstance(item, Requirement):
                parsed_requirements.append(item)
            elif isinstance(item, Mapping):
                parsed_requirements.append(Requirement.from_dict(item))
            else:
                _fail(
                    f"SpecMap.requirements[{index}]",
                    f"expected Requirement, got {type(item).__name__}",
                )
        self.requirements = tuple(parsed_requirements)

        self.created_at = _as_datetime(self.created_at, "SpecMap.created_at")
        self.entities = _as_str_tuple(
            self.entities, "SpecMap.entities", allow_empty=True, unique=True
        )
        self.interfaces = _as_str_tuple(
            self.interfaces, "SpecMap.interfaces", allow_empty=True, unique=True
        )
        self.glossary = _as_str_dict(self.glossary, "SpecMap.glossary", max_entries=256)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SpecMap:
        parsed = _expect_object(
            data,
            "SpecMap",
            required={"source_document", "requirements", "created_at"},
            optional={"entities", "interfaces", "glossary", "schema_version", "version"},
        )

        schema_version = parsed.get("schema_version", parsed.get("version", _SCHEMA_VERSION))

        requirements_raw = _as_sequence(parsed["requirements"], "SpecMap.requirements")
        requirements = tuple(
            item
            if isinstance(item, Requirement)
            else Requirement.from_dict(
                _expect_object(
                    item,
                    "SpecMap.requirements[]",
                    required={"id", "statement"},
                    optional={"acceptance_criteria", "nfr_tags", "source", "schema_version"},
                )
            )
            for item in requirements_raw
        )

        return cls(
            source_document=_as_str(
                parsed["source_document"], "SpecMap.source_document", max_len=1024
            ),
            requirements=requirements,
            created_at=_as_datetime(parsed["created_at"], "SpecMap.created_at"),
            entities=_as_str_tuple(
                parsed.get("entities", ()), "SpecMap.entities", allow_empty=True, unique=True
            ),
            interfaces=_as_str_tuple(
                parsed.get("interfaces", ()),
                "SpecMap.interfaces",
                allow_empty=True,
                unique=True,
            ),
            glossary=_as_str_dict(parsed.get("glossary", {}), "SpecMap.glossary", max_entries=256),
            schema_version=_as_schema_version(schema_version, "SpecMap.schema_version"),
        )


@dataclass(slots=True)
class Artifact(CanonicalModel):
    id: str
    work_item_id: str
    run_id: str
    kind: str
    path: str
    sha256: str
    size_bytes: int
    created_at: datetime
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Artifact.schema_version")
        self.id = _validate_artifact_id(_as_str(self.id, "Artifact.id"), "Artifact.id")
        self.work_item_id = _validate_work_item_id(
            _as_str(self.work_item_id, "Artifact.work_item_id"), "Artifact.work_item_id"
        )
        self.run_id = _validate_run_id(_as_str(self.run_id, "Artifact.run_id"), "Artifact.run_id")
        self.kind = _as_str(self.kind, "Artifact.kind", max_len=128)
        self.path = _as_relative_path(self.path, "Artifact.path")
        self.sha256 = _as_sha256(self.sha256, "Artifact.sha256")
        self.size_bytes = _as_int(self.size_bytes, "Artifact.size_bytes", minimum=0)
        self.created_at = _as_datetime(self.created_at, "Artifact.created_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Artifact:
        parsed = _expect_object(
            data,
            "Artifact",
            required={
                "id",
                "work_item_id",
                "run_id",
                "kind",
                "path",
                "sha256",
                "size_bytes",
                "created_at",
            },
            optional={"schema_version"},
        )
        return cls(
            id=_as_str(parsed["id"], "Artifact.id"),
            work_item_id=_as_str(parsed["work_item_id"], "Artifact.work_item_id"),
            run_id=_as_str(parsed["run_id"], "Artifact.run_id"),
            kind=_as_str(parsed["kind"], "Artifact.kind", max_len=128),
            path=_as_str(parsed["path"], "Artifact.path", max_len=1024),
            sha256=_as_str(parsed["sha256"], "Artifact.sha256", max_len=64, strip=False),
            size_bytes=_as_int(parsed["size_bytes"], "Artifact.size_bytes", minimum=0),
            created_at=_as_datetime(parsed["created_at"], "Artifact.created_at"),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION), "Artifact.schema_version"
            ),
        )


@dataclass(slots=True)
class EvidenceRecord(CanonicalModel):
    id: str
    work_item_id: str
    run_id: str
    stage: str
    result: EvidenceResult
    checker_id: str
    constraint_ids: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    tool_versions: dict[str, str]
    environment_hash: str
    duration_ms: int
    created_at: datetime
    summary: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(
            self.schema_version, "EvidenceRecord.schema_version"
        )
        self.id = _validate_evidence_id(_as_str(self.id, "EvidenceRecord.id"), "EvidenceRecord.id")
        self.work_item_id = _validate_work_item_id(
            _as_str(self.work_item_id, "EvidenceRecord.work_item_id"), "EvidenceRecord.work_item_id"
        )
        self.run_id = _validate_run_id(
            _as_str(self.run_id, "EvidenceRecord.run_id"), "EvidenceRecord.run_id"
        )
        self.stage = _as_str(self.stage, "EvidenceRecord.stage", max_len=128)
        self.result = _as_enum(EvidenceResult, self.result, "EvidenceRecord.result")
        self.checker_id = _as_str(self.checker_id, "EvidenceRecord.checker_id", max_len=128)
        self.constraint_ids = _parse_constraint_links(
            self.constraint_ids, "EvidenceRecord.constraint_ids"
        )

        artifact_paths = _as_str_tuple(
            self.artifact_paths,
            "EvidenceRecord.artifact_paths",
            allow_empty=True,
            unique=True,
            max_len=1024,
        )
        self.artifact_paths = tuple(
            _as_relative_path(item, f"EvidenceRecord.artifact_paths[{idx}]")
            for idx, item in enumerate(artifact_paths)
        )

        self.tool_versions = _as_str_dict(
            self.tool_versions,
            "EvidenceRecord.tool_versions",
            max_entries=128,
        )
        self.environment_hash = _as_sha256(self.environment_hash, "EvidenceRecord.environment_hash")
        self.duration_ms = _as_int(self.duration_ms, "EvidenceRecord.duration_ms", minimum=0)
        self.created_at = _as_datetime(self.created_at, "EvidenceRecord.created_at")
        self.summary = _as_optional_str(self.summary, "EvidenceRecord.summary", max_len=4096)
        self.metadata = _as_json_object(self.metadata, "EvidenceRecord.metadata")
        _enforce_json_string_limit(
            self.metadata,
            "EvidenceRecord.metadata",
            max_len=2048,
        )

        metadata_size = len(_canonical_json(self.metadata))
        if metadata_size > _MAX_EVIDENCE_METADATA_BYTES:
            _fail(
                "EvidenceRecord.metadata",
                f"metadata exceeds {_MAX_EVIDENCE_METADATA_BYTES} bytes (got {metadata_size})",
            )

        record_size = len(self.to_json())
        if record_size > _MAX_EVIDENCE_RECORD_BYTES:
            _fail(
                "EvidenceRecord",
                f"record exceeds {_MAX_EVIDENCE_RECORD_BYTES} bytes (got {record_size}); "
                "store large outputs in artifact files and keep references only",
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidenceRecord:
        parsed = _expect_object(
            data,
            "EvidenceRecord",
            required={
                "id",
                "work_item_id",
                "run_id",
                "stage",
                "result",
                "checker_id",
                "constraint_ids",
                "artifact_paths",
                "tool_versions",
                "environment_hash",
                "duration_ms",
                "created_at",
            },
            optional={"summary", "metadata", "schema_version"},
        )
        return cls(
            id=_as_str(parsed["id"], "EvidenceRecord.id"),
            work_item_id=_as_str(parsed["work_item_id"], "EvidenceRecord.work_item_id"),
            run_id=_as_str(parsed["run_id"], "EvidenceRecord.run_id"),
            stage=_as_str(parsed["stage"], "EvidenceRecord.stage", max_len=128),
            result=_as_enum(EvidenceResult, parsed["result"], "EvidenceRecord.result"),
            checker_id=_as_str(parsed["checker_id"], "EvidenceRecord.checker_id", max_len=128),
            constraint_ids=_parse_constraint_links(
                parsed["constraint_ids"], "EvidenceRecord.constraint_ids"
            ),
            artifact_paths=_as_str_tuple(
                parsed["artifact_paths"],
                "EvidenceRecord.artifact_paths",
                allow_empty=True,
                unique=True,
                max_len=1024,
            ),
            tool_versions=_as_str_dict(
                parsed["tool_versions"], "EvidenceRecord.tool_versions", max_entries=128
            ),
            environment_hash=_as_str(
                parsed["environment_hash"],
                "EvidenceRecord.environment_hash",
                max_len=64,
                strip=False,
            ),
            duration_ms=_as_int(parsed["duration_ms"], "EvidenceRecord.duration_ms", minimum=0),
            created_at=_as_datetime(parsed["created_at"], "EvidenceRecord.created_at"),
            summary=_as_optional_str(parsed.get("summary"), "EvidenceRecord.summary", max_len=4096),
            metadata=_as_json_object(parsed.get("metadata", {}), "EvidenceRecord.metadata"),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "EvidenceRecord.schema_version",
            ),
        )


@dataclass(slots=True)
class Attempt(CanonicalModel):
    id: str
    work_item_id: str
    run_id: str
    iteration: int
    provider: str
    model: str
    role: str
    prompt_hash: str
    tokens_used: int
    cost_usd: float
    result: AttemptResult
    created_at: datetime
    feedback: str | None = None
    finished_at: datetime | None = None
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Attempt.schema_version")
        self.id = _validate_attempt_id(_as_str(self.id, "Attempt.id"), "Attempt.id")
        self.work_item_id = _validate_work_item_id(
            _as_str(self.work_item_id, "Attempt.work_item_id"), "Attempt.work_item_id"
        )
        self.run_id = _validate_run_id(_as_str(self.run_id, "Attempt.run_id"), "Attempt.run_id")
        self.iteration = _as_int(self.iteration, "Attempt.iteration", minimum=1)
        self.provider = _as_str(self.provider, "Attempt.provider", max_len=128)
        self.model = _as_str(self.model, "Attempt.model", max_len=128)
        self.role = _as_str(self.role, "Attempt.role", max_len=128)
        self.prompt_hash = _as_sha256(self.prompt_hash, "Attempt.prompt_hash")
        self.tokens_used = _as_int(self.tokens_used, "Attempt.tokens_used", minimum=0)
        self.cost_usd = _as_float(self.cost_usd, "Attempt.cost_usd", minimum=0.0)
        self.result = _as_enum(AttemptResult, self.result, "Attempt.result")
        self.created_at = _as_datetime(self.created_at, "Attempt.created_at")
        self.feedback = _as_optional_str(self.feedback, "Attempt.feedback", max_len=4096)
        if self.finished_at is not None:
            self.finished_at = _as_datetime(self.finished_at, "Attempt.finished_at")
            if self.finished_at < self.created_at:
                _fail("Attempt.finished_at", "must be >= Attempt.created_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Attempt:
        parsed = _expect_object(
            data,
            "Attempt",
            required={
                "id",
                "work_item_id",
                "run_id",
                "iteration",
                "provider",
                "model",
                "role",
                "prompt_hash",
                "tokens_used",
                "cost_usd",
                "result",
                "created_at",
            },
            optional={"feedback", "finished_at", "schema_version"},
        )
        return cls(
            id=_as_str(parsed["id"], "Attempt.id"),
            work_item_id=_as_str(parsed["work_item_id"], "Attempt.work_item_id"),
            run_id=_as_str(parsed["run_id"], "Attempt.run_id"),
            iteration=_as_int(parsed["iteration"], "Attempt.iteration", minimum=1),
            provider=_as_str(parsed["provider"], "Attempt.provider", max_len=128),
            model=_as_str(parsed["model"], "Attempt.model", max_len=128),
            role=_as_str(parsed["role"], "Attempt.role", max_len=128),
            prompt_hash=_as_str(
                parsed["prompt_hash"], "Attempt.prompt_hash", max_len=64, strip=False
            ),
            tokens_used=_as_int(parsed["tokens_used"], "Attempt.tokens_used", minimum=0),
            cost_usd=_as_float(parsed["cost_usd"], "Attempt.cost_usd", minimum=0.0),
            result=_as_enum(AttemptResult, parsed["result"], "Attempt.result"),
            created_at=_as_datetime(parsed["created_at"], "Attempt.created_at"),
            feedback=_as_optional_str(parsed.get("feedback"), "Attempt.feedback", max_len=4096),
            finished_at=(
                _as_datetime(parsed["finished_at"], "Attempt.finished_at")
                if parsed.get("finished_at") is not None
                else None
            ),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "Attempt.schema_version",
            ),
        )


@dataclass(slots=True)
class MergeRecord(CanonicalModel):
    id: str
    work_item_id: str
    run_id: str
    commit_sha: str
    evidence_ids: tuple[str, ...]
    merged_at: datetime
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "MergeRecord.schema_version")
        self.id = _validate_merge_id(_as_str(self.id, "MergeRecord.id"), "MergeRecord.id")
        self.work_item_id = _validate_work_item_id(
            _as_str(self.work_item_id, "MergeRecord.work_item_id"), "MergeRecord.work_item_id"
        )
        self.run_id = _validate_run_id(
            _as_str(self.run_id, "MergeRecord.run_id"), "MergeRecord.run_id"
        )
        self.commit_sha = _as_commit_sha(self.commit_sha, "MergeRecord.commit_sha")
        self.evidence_ids = _parse_evidence_ids(self.evidence_ids, "MergeRecord.evidence_ids")
        self.merged_at = _as_datetime(self.merged_at, "MergeRecord.merged_at")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MergeRecord:
        parsed = _expect_object(
            data,
            "MergeRecord",
            required={"id", "work_item_id", "run_id", "commit_sha", "evidence_ids", "merged_at"},
            optional={"schema_version"},
        )
        return cls(
            id=_as_str(parsed["id"], "MergeRecord.id"),
            work_item_id=_as_str(parsed["work_item_id"], "MergeRecord.work_item_id"),
            run_id=_as_str(parsed["run_id"], "MergeRecord.run_id"),
            commit_sha=_as_str(
                parsed["commit_sha"], "MergeRecord.commit_sha", max_len=40, strip=False
            ),
            evidence_ids=_parse_evidence_ids(parsed["evidence_ids"], "MergeRecord.evidence_ids"),
            merged_at=_as_datetime(parsed["merged_at"], "MergeRecord.merged_at"),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "MergeRecord.schema_version",
            ),
        )


@dataclass(slots=True)
class Incident(CanonicalModel):
    id: str
    run_id: str
    category: str
    message: str
    created_at: datetime
    related_work_item_id: str | None = None
    constraint_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    details: dict[str, JSONValue] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Incident.schema_version")
        self.id = _validate_incident_id(_as_str(self.id, "Incident.id"), "Incident.id")
        self.run_id = _validate_run_id(_as_str(self.run_id, "Incident.run_id"), "Incident.run_id")
        self.category = _as_str(self.category, "Incident.category", max_len=128)
        self.message = _as_str(self.message, "Incident.message")
        self.created_at = _as_datetime(self.created_at, "Incident.created_at")
        if self.related_work_item_id is not None:
            self.related_work_item_id = _validate_work_item_id(
                _as_str(self.related_work_item_id, "Incident.related_work_item_id"),
                "Incident.related_work_item_id",
            )
        self.constraint_ids = _parse_constraint_links(
            self.constraint_ids, "Incident.constraint_ids"
        )
        self.evidence_ids = _parse_evidence_ids(self.evidence_ids, "Incident.evidence_ids")
        self.details = _as_json_object(self.details, "Incident.details")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Incident:
        parsed = _expect_object(
            data,
            "Incident",
            required={"id", "run_id", "category", "message", "created_at"},
            optional={
                "related_work_item_id",
                "constraint_ids",
                "evidence_ids",
                "details",
                "schema_version",
            },
        )
        return cls(
            id=_as_str(parsed["id"], "Incident.id"),
            run_id=_as_str(parsed["run_id"], "Incident.run_id"),
            category=_as_str(parsed["category"], "Incident.category", max_len=128),
            message=_as_str(parsed["message"], "Incident.message"),
            created_at=_as_datetime(parsed["created_at"], "Incident.created_at"),
            related_work_item_id=(
                _as_str(parsed["related_work_item_id"], "Incident.related_work_item_id")
                if parsed.get("related_work_item_id") is not None
                else None
            ),
            constraint_ids=_parse_constraint_links(
                parsed.get("constraint_ids", ()), "Incident.constraint_ids"
            ),
            evidence_ids=_parse_evidence_ids(
                parsed.get("evidence_ids", ()), "Incident.evidence_ids"
            ),
            details=_as_json_object(parsed.get("details", {}), "Incident.details"),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION),
                "Incident.schema_version",
            ),
        )


@dataclass(slots=True)
class Run(CanonicalModel):
    id: str
    spec_path: str
    status: RunStatus | str
    started_at: datetime
    finished_at: datetime | None = None
    work_item_ids: tuple[str, ...] = ()
    budget: Budget = field(default_factory=_default_budget)
    risk_tier: RiskTier = RiskTier.MEDIUM
    metadata: dict[str, JSONValue] = field(default_factory=dict)
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.schema_version = _as_schema_version(self.schema_version, "Run.schema_version")
        self.id = _validate_run_id(_as_str(self.id, "Run.id"), "Run.id")
        self.spec_path = _as_relative_path(self.spec_path, "Run.spec_path")
        self.status = _as_enum(RunStatus, self.status, "Run.status")
        self.started_at = _as_datetime(self.started_at, "Run.started_at")

        if self.finished_at is not None:
            self.finished_at = _as_datetime(self.finished_at, "Run.finished_at")
            if self.finished_at < self.started_at:
                _fail("Run.finished_at", "must be >= Run.started_at")

        self.work_item_ids = _as_str_tuple(
            self.work_item_ids,
            "Run.work_item_ids",
            allow_empty=True,
            unique=True,
        )
        for index, work_item_id in enumerate(self.work_item_ids):
            _validate_work_item_id(work_item_id, f"Run.work_item_ids[{index}]")

        if not isinstance(self.budget, Budget):
            _fail("Run.budget", "must be Budget")
        self.risk_tier = _as_enum(RiskTier, self.risk_tier, "Run.risk_tier")
        self.metadata = _as_json_object(self.metadata, "Run.metadata")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Run:
        parsed = _expect_object(
            data,
            "Run",
            required={"id", "spec_path", "status", "started_at"},
            optional={
                "finished_at",
                "work_item_ids",
                "budget",
                "risk_tier",
                "metadata",
                "schema_version",
            },
        )

        budget_raw = parsed.get("budget", _default_budget())
        budget = (
            budget_raw
            if isinstance(budget_raw, Budget)
            else Budget.from_dict(
                _expect_object(
                    budget_raw,
                    "Run.budget",
                    required={
                        "max_tokens",
                        "max_cost_usd",
                        "max_iterations",
                        "max_wall_clock_seconds",
                    },
                    optional={"schema_version"},
                )
            )
        )

        return cls(
            id=_as_str(parsed["id"], "Run.id"),
            spec_path=_as_str(parsed["spec_path"], "Run.spec_path", max_len=1024),
            status=_as_enum(RunStatus, parsed["status"], "Run.status"),
            started_at=_as_datetime(parsed["started_at"], "Run.started_at"),
            finished_at=(
                _as_datetime(parsed["finished_at"], "Run.finished_at")
                if parsed.get("finished_at") is not None
                else None
            ),
            work_item_ids=_as_str_tuple(
                parsed.get("work_item_ids", ()),
                "Run.work_item_ids",
                allow_empty=True,
                unique=True,
            ),
            budget=budget,
            risk_tier=_as_enum(RiskTier, parsed.get("risk_tier", RiskTier.MEDIUM), "Run.risk_tier"),
            metadata=_as_json_object(parsed.get("metadata", {}), "Run.metadata"),
            schema_version=_as_schema_version(
                parsed.get("schema_version", _SCHEMA_VERSION), "Run.schema_version"
            ),
        )


__all__ = [
    "Artifact",
    "Attempt",
    "AttemptResult",
    "Budget",
    "CanonicalModel",
    "Constraint",
    "ConstraintEnvelope",
    "ConstraintSeverity",
    "ConstraintSource",
    "EvidenceRecord",
    "EvidenceResult",
    "Incident",
    "MergeRecord",
    "Requirement",
    "RiskTier",
    "Run",
    "RunStatus",
    "SandboxPolicy",
    "SpecMap",
    "TaskGraph",
    "WorkItem",
    "WorkItemStatus",
]
