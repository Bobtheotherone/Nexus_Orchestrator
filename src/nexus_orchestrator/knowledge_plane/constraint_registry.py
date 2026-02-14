"""Deterministic constraint registry loader and query surface."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias, cast

import yaml

from nexus_orchestrator.domain import ids as domain_ids
from nexus_orchestrator.domain.models import Constraint, ConstraintSeverity, ConstraintSource

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc  # noqa: UP017

PathLike: TypeAlias = str | os.PathLike[str]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_REQUIRED_CONSTRAINT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "severity",
        "category",
        "description",
        "checker",
        "parameters",
        "requirement_links",
        "source",
    }
)
_OPTIONAL_CONSTRAINT_FIELDS: Final[frozenset[str]] = frozenset({"created_at"})
_ALLOWED_CONSTRAINT_FIELDS: Final[frozenset[str]] = (
    _REQUIRED_CONSTRAINT_FIELDS | _OPTIONAL_CONSTRAINT_FIELDS
)
_DEFAULT_REGISTRY_DIR: Final[Path] = Path("constraints") / "registry"
_DEFAULT_CREATED_AT: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ConstraintExemptionRecord:
    """Validated override/exemption record for one constraint."""

    constraint_id: str
    justification: str
    expiry: datetime | None = None
    approved: bool = False
    approved_by: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "constraint_id", _coerce_constraint_id(self.constraint_id, "constraint_id")
        )
        object.__setattr__(
            self,
            "justification",
            _coerce_non_empty_str(self.justification, "justification"),
        )
        if not isinstance(self.approved, bool):
            raise ValueError("approved: expected bool")
        if self.approved_by is not None:
            object.__setattr__(
                self,
                "approved_by",
                _coerce_non_empty_str(self.approved_by, "approved_by"),
            )
        if self.expiry is not None:
            object.__setattr__(
                self,
                "expiry",
                _coerce_datetime(self.expiry, "expiry"),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ConstraintExemptionRecord:
        """Parse one exemption record from a mapping payload."""

        parsed = _as_string_key_mapping(payload, "ConstraintExemptionRecord")
        allowed_fields = {
            "constraint_id",
            "justification",
            "expiry",
            "approved",
            "approved_by",
            "expires_at",
        }
        unknown = sorted(set(parsed) - allowed_fields)
        if unknown:
            raise ValueError(
                "ConstraintExemptionRecord: unexpected fields "
                f"{unknown}; allowed fields: {sorted(allowed_fields)}"
            )

        missing = sorted(
            field for field in ("constraint_id", "justification") if field not in parsed
        )
        if missing:
            raise ValueError(f"ConstraintExemptionRecord: missing required fields {missing}")
        if "expiry" in parsed and "expires_at" in parsed:
            raise ValueError("ConstraintExemptionRecord: set only one of expiry/expires_at")
        expiry_raw = parsed.get("expiry", parsed.get("expires_at"))
        approved_raw = parsed.get("approved", False)
        if not isinstance(approved_raw, bool):
            raise ValueError("ConstraintExemptionRecord.approved must be a bool")

        return cls(
            constraint_id=_coerce_non_empty_str(parsed["constraint_id"], "constraint_id"),
            justification=_coerce_non_empty_str(parsed["justification"], "justification"),
            expiry=(_coerce_datetime(expiry_raw, "expiry") if expiry_raw is not None else None),
            approved=approved_raw,
            approved_by=(
                _coerce_non_empty_str(parsed["approved_by"], "approved_by")
                if parsed.get("approved_by") is not None
                else None
            ),
        )

    def sort_key(self) -> tuple[str, int, str, str, str]:
        """Deterministic ordering key for stable tracking output."""

        expiry = self.expiry.isoformat(timespec="microseconds") if self.expiry else ""
        approved_by = self.approved_by if self.approved_by is not None else ""
        return (self.constraint_id, int(self.approved), expiry, approved_by, self.justification)

    def is_active(self, *, at: datetime | None = None) -> bool:
        """Return whether this exemption is active at ``at`` (default: now UTC)."""

        if not self.approved:
            return False
        if self.expiry is None:
            return True
        reference = datetime.now(UTC) if at is None else _coerce_datetime(at, "at")
        return self.expiry >= reference


@dataclass(frozen=True, slots=True)
class ConstraintExemptionTracker:
    """Deterministically ordered exemption tracker."""

    records: tuple[ConstraintExemptionRecord, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.records, key=lambda record: record.sort_key()))
        object.__setattr__(self, "records", ordered)

    @classmethod
    def from_records(
        cls,
        records: Iterable[ConstraintExemptionRecord | Mapping[str, object]],
    ) -> ConstraintExemptionTracker:
        """Build a tracker from typed or mapping-backed records."""

        parsed: list[ConstraintExemptionRecord] = []
        for index, item in enumerate(records):
            if isinstance(item, ConstraintExemptionRecord):
                parsed.append(item)
                continue
            if isinstance(item, Mapping):
                try:
                    parsed.append(ConstraintExemptionRecord.from_mapping(item))
                except ValueError as exc:
                    raise ValueError(f"exemptions[{index}]: {exc}") from exc
                continue
            raise ValueError(
                f"exemptions[{index}]: expected ConstraintExemptionRecord or mapping, "
                f"got {type(item).__name__}"
            )
        return cls(records=tuple(parsed))

    def exempted_constraint_ids(self, *, at: datetime | None = None) -> frozenset[str]:
        """Return IDs currently exempted at ``at`` (default: now UTC)."""

        return frozenset(record.constraint_id for record in self.records if record.is_active(at=at))


class ConstraintRegistry:
    """In-memory deterministic view of ``constraints/registry/*.yaml``."""

    __slots__ = (
        "_constraints",
        "_constraints_by_id",
        "_exemptions",
        "_registry_dir",
        "_source_files",
    )

    def __init__(
        self,
        *,
        registry_dir: Path,
        constraints: Sequence[Constraint],
        source_files: Sequence[Path],
        exemptions: ConstraintExemptionTracker | None = None,
    ) -> None:
        self._registry_dir = registry_dir
        self._constraints = tuple(constraints)
        self._source_files = tuple(source_files)
        self._exemptions = exemptions if exemptions is not None else ConstraintExemptionTracker()

        by_id: dict[str, Constraint] = {}
        for constraint in self._constraints:
            if constraint.id in by_id:
                raise ValueError(f"duplicate constraint id in memory: {constraint.id!r}")
            by_id[constraint.id] = constraint
        self._constraints_by_id = by_id

    @property
    def registry_dir(self) -> Path:
        return self._registry_dir

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return all loaded constraints in deterministic load order."""

        return self._constraints

    @property
    def source_files(self) -> tuple[Path, ...]:
        """Return source registry files in deterministic lexicographic order."""

        return self._source_files

    @property
    def exemptions(self) -> ConstraintExemptionTracker:
        return self._exemptions

    @property
    def overrides(self) -> ConstraintExemptionTracker:
        """Alias for override terminology used by planning callers."""

        return self._exemptions

    @classmethod
    def load(
        cls,
        registry_dir: PathLike = _DEFAULT_REGISTRY_DIR,
        *,
        overrides: Iterable[ConstraintExemptionRecord | Mapping[str, object]] = (),
        exemptions: Iterable[ConstraintExemptionRecord | Mapping[str, object]] | None = None,
    ) -> ConstraintRegistry:
        """Load and validate all ``*.yaml`` constraints from ``registry_dir``."""

        root = Path(registry_dir).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"constraint registry directory does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"constraint registry path is not a directory: {root}")

        files = tuple(sorted(root.glob("*.yaml"), key=lambda path: (path.name, path.as_posix())))
        constraints: list[Constraint] = []
        seen_ids: dict[str, Path] = {}

        for source_file in files:
            file_constraints = _load_constraints_file(source_file)
            for item in file_constraints:
                first_seen_path = seen_ids.get(item.id)
                if first_seen_path is not None:
                    raise ValueError(
                        "duplicate constraint id "
                        f"{item.id!r} across files: {first_seen_path.name} and {source_file.name}"
                    )
                seen_ids[item.id] = source_file
                constraints.append(item)

        active_overrides = overrides
        if exemptions is not None:
            if tuple(overrides):
                raise ValueError("ConstraintRegistry.load: pass only one of overrides/exemptions")
            active_overrides = exemptions

        return cls(
            registry_dir=root,
            constraints=constraints,
            source_files=files,
            exemptions=ConstraintExemptionTracker.from_records(active_overrides),
        )

    def with_overrides(
        self,
        overrides: Iterable[ConstraintExemptionRecord | Mapping[str, object]],
    ) -> ConstraintRegistry:
        """Return a registry copy with replacement override/exemption tracking."""

        return ConstraintRegistry(
            registry_dir=self._registry_dir,
            constraints=self._constraints,
            source_files=self._source_files,
            exemptions=ConstraintExemptionTracker.from_records(overrides),
        )

    def with_exemptions(
        self,
        exemptions: Iterable[ConstraintExemptionRecord | Mapping[str, object]],
    ) -> ConstraintRegistry:
        """Backward-compatible alias for ``with_overrides``."""

        return self.with_overrides(exemptions)

    def by_id(self, constraint_id: str, *, active_only: bool = False) -> Constraint | None:
        """Return one constraint by ID, optionally requiring active status."""

        candidate = self._constraints_by_id.get(constraint_id)
        if candidate is None:
            return None
        if active_only and not self._is_active(candidate):
            return None
        return candidate

    def by_category(self, category: str, *, active_only: bool = True) -> tuple[Constraint, ...]:
        """Return constraints matching ``category`` in deterministic order."""

        normalized = _coerce_non_empty_str(category, "category")
        return self._filter(
            lambda constraint: constraint.category == normalized,
            active_only=active_only,
        )

    def by_severity(
        self,
        severity: ConstraintSeverity | str,
        *,
        active_only: bool = True,
    ) -> tuple[Constraint, ...]:
        """Return constraints matching severity in deterministic order."""

        normalized = (
            severity
            if isinstance(severity, ConstraintSeverity)
            else _coerce_severity(severity, "severity")
        )
        return self._filter(
            lambda constraint: constraint.severity is normalized,
            active_only=active_only,
        )

    def by_checker(self, checker: str, *, active_only: bool = True) -> tuple[Constraint, ...]:
        """Return constraints bound to ``checker`` in deterministic order."""

        normalized = _coerce_non_empty_str(checker, "checker")
        return self._filter(
            lambda constraint: constraint.checker_binding == normalized,
            active_only=active_only,
        )

    def by_requirement_link(
        self,
        requirement_link: str,
        *,
        active_only: bool = True,
    ) -> tuple[Constraint, ...]:
        """Return constraints linked to one requirement ID."""

        normalized_requirement = _coerce_requirement_id(requirement_link, "requirement_link")
        return self._filter(
            lambda constraint: normalized_requirement in constraint.requirement_links,
            active_only=active_only,
        )

    def all_active(self) -> tuple[Constraint, ...]:
        """Return all active constraints in deterministic order."""

        exempted_ids = self._exemptions.exempted_constraint_ids()
        return tuple(item for item in self._constraints if item.id not in exempted_ids)

    def add_constraint(
        self,
        constraint: Constraint | Mapping[str, object],
        *,
        filename: str | None = None,
        current_date: date | None = None,
    ) -> Path:
        """Write a new deterministic YAML file for ``constraint`` and reload in-memory state.

        The target file must not already exist.
        """

        parsed_constraint = (
            constraint
            if isinstance(constraint, Constraint)
            else _parse_constraint_mapping(constraint, location="add_constraint")
        )
        if (
            not isinstance(constraint, Constraint)
            and isinstance(constraint, Mapping)
            and "created_at" not in constraint
        ):
            parsed_constraint = Constraint(
                id=parsed_constraint.id,
                severity=parsed_constraint.severity,
                category=parsed_constraint.category,
                description=parsed_constraint.description,
                checker_binding=parsed_constraint.checker_binding,
                parameters=parsed_constraint.parameters,
                requirement_links=parsed_constraint.requirement_links,
                source=parsed_constraint.source,
                created_at=datetime.now(UTC),
            )

        if parsed_constraint.id in self._constraints_by_id:
            raise ValueError(f"constraint id already exists in registry: {parsed_constraint.id}")

        if filename is None:
            resolved_filename = _next_available_auto_filename(self._registry_dir, current_date)
        else:
            resolved_filename = _validate_registry_filename(filename)
        destination = self._registry_dir / resolved_filename

        if filename is not None and destination.exists():
            raise FileExistsError(f"constraint registry file already exists: {destination}")

        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = [_constraint_to_yaml_record(parsed_constraint)]
        rendered = yaml.safe_dump(
            payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
            width=120,
        )
        if not rendered.endswith("\n"):
            rendered = rendered + "\n"

        with destination.open("x", encoding="utf-8") as handle:
            handle.write(rendered)

        reloaded = ConstraintRegistry.load(self._registry_dir, exemptions=self._exemptions.records)
        self._constraints = reloaded._constraints
        self._constraints_by_id = reloaded._constraints_by_id
        self._source_files = reloaded._source_files

        return destination

    def _is_active(self, constraint: Constraint) -> bool:
        return constraint.id not in self._exemptions.exempted_constraint_ids()

    def _filter(
        self,
        predicate: Callable[[Constraint], bool],
        *,
        active_only: bool,
    ) -> tuple[Constraint, ...]:
        if not active_only:
            return tuple(item for item in self._constraints if predicate(item))

        exempted_ids = self._exemptions.exempted_constraint_ids()
        return tuple(
            item for item in self._constraints if predicate(item) and item.id not in exempted_ids
        )


def _load_constraints_file(path: Path) -> list[Constraint]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = cast("object", yaml.safe_load(handle))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML ({exc})") from exc

    if not isinstance(loaded, list):
        raise ValueError(f"{path}: expected top-level YAML sequence, got {type(loaded).__name__}")

    parsed_constraints: list[Constraint] = []
    for index, item in enumerate(loaded):
        location = f"{path.name}[{index}]"
        parsed_constraints.append(_parse_constraint_mapping(item, location=location))
    return parsed_constraints


def _parse_constraint_mapping(value: object, *, location: str) -> Constraint:
    parsed = _as_string_key_mapping(value, location)
    parsed_keys = set(parsed)

    missing = sorted(_REQUIRED_CONSTRAINT_FIELDS - parsed_keys)
    if missing:
        raise ValueError(f"{location}: missing required fields: {missing}")

    unknown = sorted(parsed_keys - _ALLOWED_CONSTRAINT_FIELDS)
    if unknown:
        raise ValueError(
            f"{location}: unexpected fields: {unknown}; allowed fields: "
            f"{sorted(_ALLOWED_CONSTRAINT_FIELDS)}"
        )

    constraint_id = _coerce_constraint_id(parsed["id"], f"{location}.id")
    severity = _coerce_severity(parsed["severity"], f"{location}.severity")
    category = _coerce_non_empty_str(parsed["category"], f"{location}.category")
    description = _coerce_non_empty_str(parsed["description"], f"{location}.description")
    checker_binding = _coerce_non_empty_str(parsed["checker"], f"{location}.checker")
    parameters = _coerce_json_object(parsed["parameters"], f"{location}.parameters")
    requirement_links = _coerce_requirement_links(
        parsed["requirement_links"],
        f"{location}.requirement_links",
    )
    source = _coerce_source(parsed["source"], f"{location}.source")
    created_at = (
        _coerce_datetime(parsed["created_at"], f"{location}.created_at")
        if "created_at" in parsed
        else _DEFAULT_CREATED_AT
    )

    try:
        return Constraint(
            id=constraint_id,
            severity=severity,
            category=category,
            description=description,
            checker_binding=checker_binding,
            parameters=parameters,
            requirement_links=requirement_links,
            source=source,
            created_at=created_at,
        )
    except ValueError as exc:
        raise ValueError(f"{location}: {exc}") from exc


def _constraint_to_yaml_record(constraint: Constraint) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": constraint.id,
        "severity": constraint.severity.value,
        "category": constraint.category,
        "description": constraint.description,
        "checker": constraint.checker_binding,
        "parameters": _normalize_json_for_yaml(constraint.parameters),
        "requirement_links": list(sorted(constraint.requirement_links)),
        "source": constraint.source.value,
    }
    payload["created_at"] = constraint.created_at.isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    return payload


def _normalize_json_for_yaml(value: JSONValue) -> JSONValue:
    if isinstance(value, list):
        return [_normalize_json_for_yaml(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_json_for_yaml(value[key]) for key in sorted(value)}
    return value


def _as_string_key_mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: expected object, got {type(value).__name__}")

    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{path}: object keys must be strings, got {type(key).__name__}")
        parsed[key] = item
    return parsed


def _coerce_non_empty_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{path}: must not be empty")
    return normalized


def _coerce_constraint_id(value: object, path: str) -> str:
    parsed = _coerce_non_empty_str(value, path)
    try:
        domain_ids.validate_constraint_id(parsed)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return parsed


def _coerce_requirement_id(value: object, path: str) -> str:
    parsed = _coerce_non_empty_str(value, path)
    try:
        domain_ids.validate_requirement_id(parsed)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return parsed


def _coerce_requirement_links(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{path}: expected array of requirement IDs")

    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_coerce_requirement_id(item, f"{path}[{index}]"))

    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{path}: contains duplicate requirement IDs")
    return tuple(parsed)


def _coerce_severity(value: object, path: str) -> ConstraintSeverity:
    raw = _coerce_non_empty_str(value, path)
    try:
        return ConstraintSeverity(raw)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in ConstraintSeverity))
        raise ValueError(f"{path}: invalid severity {raw!r}; expected one of: {allowed}") from exc


def _coerce_source(value: object, path: str) -> ConstraintSource:
    raw = _coerce_non_empty_str(value, path)
    try:
        return ConstraintSource(raw)
    except ValueError as exc:
        allowed = ", ".join(sorted(item.value for item in ConstraintSource))
        raise ValueError(f"{path}: invalid source {raw!r}; expected one of: {allowed}") from exc


def _coerce_datetime(value: object, path: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid ISO-8601 datetime {value!r}") from exc
    else:
        raise ValueError(
            f"{path}: expected datetime or ISO-8601 string, got {type(value).__name__}"
        )

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{path}: datetime must be timezone-aware")

    return parsed.astimezone(UTC)


def _coerce_json_object(value: object, path: str) -> dict[str, JSONValue]:
    parsed = _coerce_json_value(value, path)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected JSON object")
    return parsed


def _coerce_json_value(value: object, path: str) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path}: float must be finite")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_coerce_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [_coerce_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        parsed: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: JSON object keys must be strings")
            parsed[key] = _coerce_json_value(item, f"{path}.{key}")
        return parsed

    raise ValueError(f"{path}: value is not JSON-serializable ({type(value).__name__})")


def _default_auto_filename(current_date: date | None) -> str:
    active_date = current_date if current_date is not None else datetime.now(UTC).date()
    return f"9xx_auto_{active_date.strftime('%Y%m%d')}.yaml"


def _next_available_auto_filename(registry_dir: Path, current_date: date | None) -> str:
    base = _default_auto_filename(current_date)
    base_path = registry_dir / base
    if not base_path.exists():
        return base

    stem = base_path.stem
    suffix = base_path.suffix
    for index in range(1, 10_000):
        candidate = f"{stem}_{index:02d}{suffix}"
        if not (registry_dir / candidate).exists():
            return candidate
    raise ValueError("unable to allocate auto registry filename after 9999 attempts")


def _validate_registry_filename(filename: str) -> str:
    raw = _coerce_non_empty_str(filename, "filename")
    normalized = PurePosixPath(raw)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError("filename: must be a safe relative path")
    if len(normalized.parts) != 1:
        raise ValueError("filename: must not include directory segments")
    if normalized.suffix != ".yaml":
        raise ValueError("filename: expected .yaml suffix")
    return normalized.as_posix()


__all__ = [
    "ConstraintExemptionRecord",
    "ConstraintExemptionTracker",
    "ConstraintRegistry",
]
