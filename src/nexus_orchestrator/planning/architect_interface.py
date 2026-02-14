"""Architect output contracts and parser/validator helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final

import yaml

if TYPE_CHECKING:
    from nexus_orchestrator.spec_ingestion.spec_map import SpecMap

_MAX_TEXT_LEN: Final[int] = 8_192
_ROOT_ALIASES: Final[dict[str, str]] = {
    "module_definitions": "modules",
    "work_item_proposals": "work_items",
    "adr_drafts": "adrs",
    "constraints": "constraint_suggestions",
}
_MODULE_ALIASES: Final[dict[str, str]] = {
    "module_name": "name",
    "owned_files": "owned_paths",
    "scope": "owned_paths",
    "depends_on": "dependencies",
    "interfaces": "interface_contract_refs",
}
_WORK_ITEM_ALIASES: Final[dict[str, str]] = {
    "scope": "owned_paths",
    "owned_files": "owned_paths",
    "depends_on": "dependencies",
    "interfaces": "interface_contract_refs",
}
_ADR_ALIASES: Final[dict[str, str]] = {
    "work_item_ids": "related_work_item_ids",
}
_CONSTRAINT_ALIASES: Final[dict[str, str]] = {
    "checker": "checker_binding",
}
_FENCED_BLOCK_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<fence>`{3,})(?P<lang>[^\n`]*)\n(?P<body>.*?)(?:\n(?P=fence))",
    flags=re.DOTALL,
)
_YAML_START_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:---\s*$|[A-Za-z_][A-Za-z0-9_-]*\s*:|-\s+[A-Za-z0-9_\[\]\"'])"
)


def _as_non_empty_str(value: object, field_name: str, *, max_len: int = _MAX_TEXT_LEN) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_len:
        raise ValueError(f"{field_name} must be <= {max_len} characters")
    return normalized


def _as_optional_str(value: object, field_name: str, *, max_len: int = _MAX_TEXT_LEN) -> str | None:
    if value is None:
        return None
    return _as_non_empty_str(value, field_name, max_len=max_len)


def _as_unique_str_tuple(
    value: object,
    field_name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if value is None:
        if allow_empty:
            return ()
        raise ValueError(f"{field_name} must not be empty")

    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a list of strings")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of strings")

    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        normalized = _as_non_empty_str(item, f"{field_name}[{index}]")
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)

    if not allow_empty and not out:
        raise ValueError(f"{field_name} must include at least one value")
    return tuple(out)


def _as_json_object(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} must use string keys")
        out[key] = item
    return out


def _apply_aliases(payload: Mapping[str, object], aliases: Mapping[str, str]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError("object keys must be strings")
        canonical = aliases.get(key, key)
        if canonical in normalized:
            raise ValueError(f"duplicate field {canonical!r} after alias normalization")
        normalized[canonical] = value
    return normalized


def _expect_object(
    value: object,
    object_name: str,
    *,
    required: set[str],
    optional: set[str],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{object_name} must be an object")
    normalized = _apply_aliases({str(key): item for key, item in value.items()}, {})

    allowed = required | optional
    unknown = sorted(key for key in normalized if key not in allowed)
    if unknown:
        raise ValueError(f"{object_name} has unknown fields: {unknown}")
    missing = sorted(key for key in required if key not in normalized)
    if missing:
        raise ValueError(f"{object_name} is missing required fields: {missing}")
    return normalized


def _normalize_path_tuple(value: object, field_name: str) -> tuple[str, ...]:
    raw = _as_unique_str_tuple(value, field_name, allow_empty=False)
    out: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        try:
            pure = PurePosixPath(item)
        except ValueError as exc:
            raise ValueError(f"{field_name}[{index}] is not a valid path: {exc}") from exc
        if pure.is_absolute():
            raise ValueError(f"{field_name}[{index}] must be a relative path")
        normalized = pure.as_posix().strip()
        if not normalized or normalized == ".":
            raise ValueError(f"{field_name}[{index}] must be a non-empty relative path")
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return tuple(out)


def _line_number(raw_text: str, offset: int) -> int:
    return raw_text.count("\n", 0, max(0, offset)) + 1


@dataclass(frozen=True, slots=True)
class ModuleDefinition:
    name: str
    summary: str
    owned_paths: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    requirement_links: tuple[str, ...] = ()
    interface_contract_refs: tuple[str, ...] = ()
    interface_guarantees: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_non_empty_str(self.name, "ModuleDefinition.name"))
        object.__setattr__(
            self, "summary", _as_non_empty_str(self.summary, "ModuleDefinition.summary")
        )
        object.__setattr__(
            self,
            "owned_paths",
            _normalize_path_tuple(self.owned_paths, "ModuleDefinition.owned_paths")
            if self.owned_paths
            else (),
        )
        object.__setattr__(
            self,
            "dependencies",
            _as_unique_str_tuple(self.dependencies, "ModuleDefinition.dependencies"),
        )
        object.__setattr__(
            self,
            "requirement_links",
            _as_unique_str_tuple(self.requirement_links, "ModuleDefinition.requirement_links"),
        )
        object.__setattr__(
            self,
            "interface_contract_refs",
            _as_unique_str_tuple(
                self.interface_contract_refs,
                "ModuleDefinition.interface_contract_refs",
            ),
        )
        object.__setattr__(
            self,
            "interface_guarantees",
            _as_unique_str_tuple(
                self.interface_guarantees,
                "ModuleDefinition.interface_guarantees",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ModuleDefinition:
        normalized = _apply_aliases(payload, _MODULE_ALIASES)
        parsed = _expect_object(
            normalized,
            "ModuleDefinition",
            required={"name"},
            optional={
                "summary",
                "owned_paths",
                "dependencies",
                "requirement_links",
                "interface_contract_refs",
                "interface_guarantees",
            },
        )
        return cls(
            name=_as_non_empty_str(parsed["name"], "ModuleDefinition.name"),
            summary=_as_non_empty_str(
                parsed.get("summary", parsed["name"]),
                "ModuleDefinition.summary",
            ),
            owned_paths=(
                _normalize_path_tuple(parsed["owned_paths"], "ModuleDefinition.owned_paths")
                if "owned_paths" in parsed
                else ()
            ),
            dependencies=_as_unique_str_tuple(
                parsed.get("dependencies", ()),
                "ModuleDefinition.dependencies",
            ),
            requirement_links=_as_unique_str_tuple(
                parsed.get("requirement_links", ()),
                "ModuleDefinition.requirement_links",
            ),
            interface_contract_refs=_as_unique_str_tuple(
                parsed.get("interface_contract_refs", ()),
                "ModuleDefinition.interface_contract_refs",
            ),
            interface_guarantees=_as_unique_str_tuple(
                parsed.get("interface_guarantees", ()),
                "ModuleDefinition.interface_guarantees",
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkItemProposal:
    id: str
    title: str
    description: str
    owned_paths: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    requirement_links: tuple[str, ...] = ()
    module: str | None = None
    interface_contract_refs: tuple[str, ...] = ()
    interface_guarantees: tuple[str, ...] = ()
    constraint_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _as_non_empty_str(self.id, "WorkItemProposal.id"))
        object.__setattr__(self, "title", _as_non_empty_str(self.title, "WorkItemProposal.title"))
        object.__setattr__(
            self,
            "description",
            _as_non_empty_str(self.description, "WorkItemProposal.description"),
        )
        object.__setattr__(
            self,
            "owned_paths",
            _normalize_path_tuple(self.owned_paths, "WorkItemProposal.owned_paths"),
        )
        object.__setattr__(
            self,
            "dependencies",
            _as_unique_str_tuple(self.dependencies, "WorkItemProposal.dependencies"),
        )
        object.__setattr__(
            self,
            "requirement_links",
            _as_unique_str_tuple(self.requirement_links, "WorkItemProposal.requirement_links"),
        )
        object.__setattr__(
            self,
            "module",
            _as_optional_str(self.module, "WorkItemProposal.module"),
        )
        object.__setattr__(
            self,
            "interface_contract_refs",
            _as_unique_str_tuple(
                self.interface_contract_refs,
                "WorkItemProposal.interface_contract_refs",
            ),
        )
        object.__setattr__(
            self,
            "interface_guarantees",
            _as_unique_str_tuple(
                self.interface_guarantees,
                "WorkItemProposal.interface_guarantees",
            ),
        )
        object.__setattr__(
            self,
            "constraint_ids",
            _as_unique_str_tuple(self.constraint_ids, "WorkItemProposal.constraint_ids"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> WorkItemProposal:
        normalized = _apply_aliases(payload, _WORK_ITEM_ALIASES)
        parsed = _expect_object(
            normalized,
            "WorkItemProposal",
            required={"id", "title", "description", "owned_paths"},
            optional={
                "dependencies",
                "requirement_links",
                "module",
                "interface_contract_refs",
                "interface_guarantees",
                "constraint_ids",
            },
        )
        return cls(
            id=_as_non_empty_str(parsed["id"], "WorkItemProposal.id"),
            title=_as_non_empty_str(parsed["title"], "WorkItemProposal.title"),
            description=_as_non_empty_str(
                parsed["description"],
                "WorkItemProposal.description",
            ),
            owned_paths=_normalize_path_tuple(
                parsed["owned_paths"], "WorkItemProposal.owned_paths"
            ),
            dependencies=_as_unique_str_tuple(
                parsed.get("dependencies", ()),
                "WorkItemProposal.dependencies",
            ),
            requirement_links=_as_unique_str_tuple(
                parsed.get("requirement_links", ()),
                "WorkItemProposal.requirement_links",
            ),
            module=_as_optional_str(parsed.get("module"), "WorkItemProposal.module"),
            interface_contract_refs=_as_unique_str_tuple(
                parsed.get("interface_contract_refs", ()),
                "WorkItemProposal.interface_contract_refs",
            ),
            interface_guarantees=_as_unique_str_tuple(
                parsed.get("interface_guarantees", ()),
                "WorkItemProposal.interface_guarantees",
            ),
            constraint_ids=_as_unique_str_tuple(
                parsed.get("constraint_ids", ()),
                "WorkItemProposal.constraint_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class ADRDraft:
    id: str
    title: str
    context: str
    decision: str
    consequences: tuple[str, ...] = ()
    related_work_item_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _as_non_empty_str(self.id, "ADRDraft.id"))
        object.__setattr__(self, "title", _as_non_empty_str(self.title, "ADRDraft.title"))
        object.__setattr__(self, "context", _as_non_empty_str(self.context, "ADRDraft.context"))
        object.__setattr__(
            self,
            "decision",
            _as_non_empty_str(self.decision, "ADRDraft.decision"),
        )
        object.__setattr__(
            self,
            "consequences",
            _as_unique_str_tuple(self.consequences, "ADRDraft.consequences"),
        )
        object.__setattr__(
            self,
            "related_work_item_ids",
            _as_unique_str_tuple(self.related_work_item_ids, "ADRDraft.related_work_item_ids"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ADRDraft:
        normalized = _apply_aliases(payload, _ADR_ALIASES)
        parsed = _expect_object(
            normalized,
            "ADRDraft",
            required={"id", "title", "context", "decision"},
            optional={"consequences", "related_work_item_ids"},
        )
        return cls(
            id=_as_non_empty_str(parsed["id"], "ADRDraft.id"),
            title=_as_non_empty_str(parsed["title"], "ADRDraft.title"),
            context=_as_non_empty_str(parsed["context"], "ADRDraft.context"),
            decision=_as_non_empty_str(parsed["decision"], "ADRDraft.decision"),
            consequences=_as_unique_str_tuple(
                parsed.get("consequences", ()), "ADRDraft.consequences"
            ),
            related_work_item_ids=_as_unique_str_tuple(
                parsed.get("related_work_item_ids", ()),
                "ADRDraft.related_work_item_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class ConstraintSuggestion:
    id: str
    severity: str
    category: str
    description: str
    checker_binding: str
    parameters: dict[str, object]
    requirement_links: tuple[str, ...] = ()
    source: str = "spec_derived"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _as_non_empty_str(self.id, "ConstraintSuggestion.id"))
        severity = _as_non_empty_str(self.severity, "ConstraintSuggestion.severity").lower()
        if severity not in {"must", "should", "may"}:
            raise ValueError("ConstraintSuggestion.severity must be one of: must, should, may")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(
            self,
            "category",
            _as_non_empty_str(self.category, "ConstraintSuggestion.category").lower(),
        )
        object.__setattr__(
            self,
            "description",
            _as_non_empty_str(self.description, "ConstraintSuggestion.description"),
        )
        object.__setattr__(
            self,
            "checker_binding",
            _as_non_empty_str(self.checker_binding, "ConstraintSuggestion.checker_binding"),
        )
        object.__setattr__(
            self,
            "parameters",
            _as_json_object(self.parameters, "ConstraintSuggestion.parameters"),
        )
        object.__setattr__(
            self,
            "requirement_links",
            _as_unique_str_tuple(
                self.requirement_links,
                "ConstraintSuggestion.requirement_links",
            ),
        )
        object.__setattr__(
            self,
            "source",
            _as_non_empty_str(self.source, "ConstraintSuggestion.source").lower(),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ConstraintSuggestion:
        normalized = _apply_aliases(payload, _CONSTRAINT_ALIASES)
        parsed = _expect_object(
            normalized,
            "ConstraintSuggestion",
            required={"id", "severity", "category", "description", "checker_binding"},
            optional={"parameters", "requirement_links", "source"},
        )
        return cls(
            id=_as_non_empty_str(parsed["id"], "ConstraintSuggestion.id"),
            severity=_as_non_empty_str(parsed["severity"], "ConstraintSuggestion.severity"),
            category=_as_non_empty_str(parsed["category"], "ConstraintSuggestion.category"),
            description=_as_non_empty_str(
                parsed["description"],
                "ConstraintSuggestion.description",
            ),
            checker_binding=_as_non_empty_str(
                parsed["checker_binding"],
                "ConstraintSuggestion.checker_binding",
            ),
            parameters=_as_json_object(
                parsed.get("parameters", {}),
                "ConstraintSuggestion.parameters",
            ),
            requirement_links=_as_unique_str_tuple(
                parsed.get("requirement_links", ()),
                "ConstraintSuggestion.requirement_links",
            ),
            source=_as_non_empty_str(
                parsed.get("source", "spec_derived"), "ConstraintSuggestion.source"
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchitectOutput:
    modules: tuple[ModuleDefinition, ...] = ()
    work_items: tuple[WorkItemProposal, ...] = ()
    adrs: tuple[ADRDraft, ...] = ()
    constraint_suggestions: tuple[ConstraintSuggestion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "work_items", tuple(self.work_items))
        object.__setattr__(self, "adrs", tuple(self.adrs))
        object.__setattr__(self, "constraint_suggestions", tuple(self.constraint_suggestions))
        if not self.work_items:
            raise ValueError("ArchitectOutput.work_items must include at least one work item")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ArchitectOutput:
        normalized = _apply_aliases(payload, _ROOT_ALIASES)
        parsed = _expect_object(
            normalized,
            "ArchitectOutput",
            required={"work_items"},
            optional={"modules", "adrs", "constraint_suggestions"},
        )
        modules_raw = parsed.get("modules", ())
        work_items_raw = parsed["work_items"]
        adrs_raw = parsed.get("adrs", ())
        constraints_raw = parsed.get("constraint_suggestions", ())

        if isinstance(modules_raw, (str, bytes, bytearray)) or not isinstance(
            modules_raw, Sequence
        ):
            raise ValueError("ArchitectOutput.modules must be a list")
        if isinstance(work_items_raw, (str, bytes, bytearray)) or not isinstance(
            work_items_raw, Sequence
        ):
            raise ValueError("ArchitectOutput.work_items must be a list")
        if isinstance(adrs_raw, (str, bytes, bytearray)) or not isinstance(adrs_raw, Sequence):
            raise ValueError("ArchitectOutput.adrs must be a list")
        if isinstance(constraints_raw, (str, bytes, bytearray)) or not isinstance(
            constraints_raw, Sequence
        ):
            raise ValueError("ArchitectOutput.constraint_suggestions must be a list")

        modules = tuple(
            ModuleDefinition.from_mapping(
                _as_json_object(item, f"ArchitectOutput.modules[{index}]"),
            )
            for index, item in enumerate(modules_raw)
        )
        work_items = tuple(
            WorkItemProposal.from_mapping(
                _as_json_object(item, f"ArchitectOutput.work_items[{index}]"),
            )
            for index, item in enumerate(work_items_raw)
        )
        adrs = tuple(
            ADRDraft.from_mapping(_as_json_object(item, f"ArchitectOutput.adrs[{index}]"))
            for index, item in enumerate(adrs_raw)
        )
        constraint_suggestions = tuple(
            ConstraintSuggestion.from_mapping(
                _as_json_object(
                    item,
                    f"ArchitectOutput.constraint_suggestions[{index}]",
                )
            )
            for index, item in enumerate(constraints_raw)
        )
        return cls(
            modules=modules,
            work_items=work_items,
            adrs=adrs,
            constraint_suggestions=constraint_suggestions,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    payload: dict[str, object]
    kind: str


def _normalize_candidate_set(candidates: Sequence[_Candidate]) -> list[_Candidate]:
    ordered = sorted(candidates, key=lambda item: (item.start, -(item.end - item.start)))
    top_level: list[_Candidate] = []
    for candidate in ordered:
        if any(
            existing.start <= candidate.start and candidate.end <= existing.end
            for existing in top_level
        ):
            continue
        top_level.append(candidate)
    return sorted(top_level, key=lambda item: item.start)


def _extract_json_candidates(raw_text: str) -> tuple[list[_Candidate], list[str], bool]:
    candidates: list[_Candidate] = []
    errors: list[str] = []
    explicit_json_seen = False

    for match in _FENCED_BLOCK_RE.finditer(raw_text):
        language = match.group("lang").strip().lower()
        body = match.group("body").strip()
        if language != "json":
            continue
        explicit_json_seen = True
        if not body:
            errors.append(
                f"Empty JSON fenced block near line {_line_number(raw_text, match.start())}."
            )
            continue
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            errors.append(
                "Invalid JSON fenced block near line "
                f"{_line_number(raw_text, match.start())}: {exc.msg}."
            )
            continue
        if not isinstance(parsed, Mapping):
            errors.append(
                "JSON fenced block near line "
                f"{_line_number(raw_text, match.start())} must decode to an object."
            )
            continue
        candidates.append(
            _Candidate(
                start=match.start("body"),
                end=match.end("body"),
                payload={str(key): value for key, value in parsed.items()},
                kind="json",
            )
        )

    decoder = json.JSONDecoder()
    seen_ranges: set[tuple[int, int]] = {
        (candidate.start, candidate.end) for candidate in candidates
    }
    for index, character in enumerate(raw_text):
        if character not in "{[":
            continue
        try:
            parsed, consumed = decoder.raw_decode(raw_text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, Mapping):
            continue
        end = index + consumed
        range_key = (index, end)
        if range_key in seen_ranges:
            continue
        seen_ranges.add(range_key)
        candidates.append(
            _Candidate(
                start=index,
                end=end,
                payload={str(key): value for key, value in parsed.items()},
                kind="json",
            )
        )

    return _normalize_candidate_set(candidates), errors, explicit_json_seen


def _line_offsets(lines: Sequence[str]) -> list[int]:
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _extract_plain_yaml_candidates(raw_text: str) -> list[_Candidate]:
    lines = raw_text.splitlines()
    offsets = _line_offsets(lines)
    candidates: list[_Candidate] = []

    for start_index, line in enumerate(lines):
        if not _YAML_START_RE.match(line):
            continue
        best: _Candidate | None = None
        for end_index in range(start_index + 1, len(lines) + 1):
            snippet = "\n".join(lines[start_index:end_index]).strip()
            if not snippet:
                continue
            try:
                parsed = yaml.safe_load(snippet)
            except yaml.YAMLError:
                continue
            if not isinstance(parsed, Mapping):
                continue
            best = _Candidate(
                start=offsets[start_index],
                end=offsets[end_index],
                payload={str(key): value for key, value in parsed.items()},
                kind="yaml",
            )
        if best is not None:
            candidates.append(best)

    return _normalize_candidate_set(candidates)


def _extract_yaml_candidates(raw_text: str) -> tuple[list[_Candidate], list[str], bool]:
    candidates: list[_Candidate] = []
    errors: list[str] = []
    explicit_yaml_seen = False

    for match in _FENCED_BLOCK_RE.finditer(raw_text):
        language = match.group("lang").strip().lower()
        if language not in {"yaml", "yml"}:
            continue
        explicit_yaml_seen = True
        body = match.group("body").strip()
        if not body:
            errors.append(
                f"Empty YAML fenced block near line {_line_number(raw_text, match.start())}."
            )
            continue
        try:
            parsed = yaml.safe_load(body)
        except yaml.YAMLError as exc:
            errors.append(
                "Invalid YAML fenced block near line "
                f"{_line_number(raw_text, match.start())}: {exc}."
            )
            continue
        if not isinstance(parsed, Mapping):
            errors.append(
                "YAML fenced block near line "
                f"{_line_number(raw_text, match.start())} must decode to an object."
            )
            continue
        candidates.append(
            _Candidate(
                start=match.start("body"),
                end=match.end("body"),
                payload={str(key): value for key, value in parsed.items()},
                kind="yaml",
            )
        )

    if candidates:
        return _normalize_candidate_set(candidates), errors, explicit_yaml_seen

    plain = _extract_plain_yaml_candidates(raw_text)
    return plain, errors, explicit_yaml_seen


def parse_architect_response(raw_text: str) -> ArchitectOutput:
    """
    Parse architect output from noisy model text.

    Parsing policy:
    1. Try JSON extraction first.
    2. Fall back to YAML only when no JSON object block is present.
    """

    if not isinstance(raw_text, str):
        raise ValueError("raw_text must be a string")
    if not raw_text.strip():
        raise ValueError("raw_text must not be empty")

    json_candidates, json_errors, explicit_json_seen = _extract_json_candidates(raw_text)
    if json_errors:
        details = " ".join(json_errors)
        raise ValueError(
            "Architect response includes malformed JSON. "
            f"Provide exactly one valid JSON object block. Details: {details}"
        )
    if len(json_candidates) > 1:
        locations = ", ".join(
            str(_line_number(raw_text, candidate.start)) for candidate in json_candidates
        )
        raise ValueError(
            "Ambiguous architect response: multiple top-level JSON objects detected "
            f"(starting lines: {locations}). Provide exactly one."
        )
    if json_candidates:
        try:
            return ArchitectOutput.from_mapping(json_candidates[0].payload)
        except ValueError as exc:
            raise ValueError(f"Invalid architect JSON payload: {exc}") from exc
    if explicit_json_seen:
        raise ValueError("JSON fenced block was present but no valid JSON object could be parsed.")

    yaml_candidates, yaml_errors, explicit_yaml_seen = _extract_yaml_candidates(raw_text)
    if yaml_errors:
        details = " ".join(yaml_errors)
        raise ValueError(
            "Architect response includes malformed YAML. "
            f"Provide exactly one valid YAML object block. Details: {details}"
        )
    if len(yaml_candidates) > 1:
        locations = ", ".join(
            str(_line_number(raw_text, candidate.start)) for candidate in yaml_candidates
        )
        raise ValueError(
            "Ambiguous architect response: multiple top-level YAML objects detected "
            f"(starting lines: {locations}). Provide exactly one."
        )
    if yaml_candidates:
        try:
            return ArchitectOutput.from_mapping(yaml_candidates[0].payload)
        except ValueError as exc:
            raise ValueError(f"Invalid architect YAML payload: {exc}") from exc
    if explicit_yaml_seen:
        raise ValueError("YAML fenced block was present but no valid YAML object could be parsed.")

    raise ValueError(
        "No machine-parseable architect output block found. "
        "Provide one top-level JSON object (preferred) or YAML object."
    )


def validate_architect_output(output: ArchitectOutput, spec_map: SpecMap) -> list[str]:
    """Validate architect output against requirement/interface and ownership invariants."""

    errors: list[str] = []

    requirement_ids = {requirement.id for requirement in spec_map.requirements}
    interface_modules = {interface.module_name for interface in spec_map.interfaces}

    module_names: set[str] = set()
    for module in output.modules:
        if module.name in module_names:
            errors.append(f"Duplicate module definition: {module.name}.")
        module_names.add(module.name)

    work_item_ids: set[str] = set()
    for proposal in output.work_items:
        if proposal.id in work_item_ids:
            errors.append(f"Duplicate work item ID: {proposal.id}.")
        work_item_ids.add(proposal.id)

    scope_owner: dict[str, str] = {}
    covered_requirements: set[str] = set()

    for module in output.modules:
        for requirement_id in module.requirement_links:
            if requirement_id not in requirement_ids:
                errors.append(
                    f"Module {module.name} references unknown requirement {requirement_id}."
                )
            covered_requirements.add(requirement_id)
        for interface_ref in module.interface_contract_refs:
            if interface_ref not in interface_modules:
                errors.append(
                    f"Module {module.name} references unknown interface contract {interface_ref}."
                )
        for dependency in module.dependencies:
            if dependency not in module_names and dependency not in work_item_ids:
                errors.append(
                    f"Module {module.name} dependency {dependency} is not a known module/work item."
                )

    for proposal in output.work_items:
        if proposal.module is not None and proposal.module not in module_names:
            errors.append(f"Work item {proposal.id} references unknown module {proposal.module}.")

        for path in proposal.owned_paths:
            owner = scope_owner.get(path)
            if owner is not None and owner != proposal.id:
                errors.append(
                    f"Duplicate file ownership: {path} owned by both {owner} and {proposal.id}."
                )
            else:
                scope_owner[path] = proposal.id

        for requirement_id in proposal.requirement_links:
            if requirement_id not in requirement_ids:
                errors.append(
                    f"Work item {proposal.id} references unknown requirement {requirement_id}."
                )
            covered_requirements.add(requirement_id)

        for dependency in proposal.dependencies:
            if dependency not in work_item_ids and dependency not in module_names:
                errors.append(
                    f"Work item {proposal.id} dependency {dependency} is not a known ID/name."
                )

        for interface_ref in proposal.interface_contract_refs:
            if interface_ref not in interface_modules:
                errors.append(
                    f"Work item {proposal.id} references unknown interface contract {interface_ref}."
                )

    missing_coverage = sorted(requirement_ids - covered_requirements)
    if missing_coverage:
        errors.append(
            f"Architect output does not cover requirements: {', '.join(missing_coverage)}."
        )

    return sorted(set(errors))


def _module_to_path_stem(module_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", module_name.strip()).strip("_").lower()
    return normalized or "module"


def build_deterministic_architect_output(spec_map: SpecMap) -> ArchitectOutput:
    """
    Build deterministic architect output directly from ingested interfaces.

    This helper is intentionally provider-free and deterministic to support
    integration tests and offline planning flows.
    """

    modules: list[ModuleDefinition] = []
    work_items: list[WorkItemProposal] = []

    for interface in sorted(spec_map.interfaces, key=lambda item: item.module_name):
        stem = _module_to_path_stem(interface.module_name)
        guarantees = (
            tuple(f"{symbol} contract remains stable" for symbol in interface.exposed_symbols)
            if interface.exposed_symbols
            else (f"{interface.module_name} interface contract remains stable",)
        )
        modules.append(
            ModuleDefinition(
                name=interface.module_name,
                summary=interface.summary or f"Implements module {interface.module_name}.",
                owned_paths=(f"src/{stem}.py",),
                dependencies=interface.dependencies,
                requirement_links=interface.requirement_links,
                interface_contract_refs=(interface.module_name,),
                interface_guarantees=guarantees,
            )
        )
        work_items.append(
            WorkItemProposal(
                id=interface.module_name,
                title=f"Implement module {interface.module_name}",
                description=interface.summary or f"Implement {interface.module_name}.",
                owned_paths=(f"src/{stem}.py", f"tests/unit/test_{stem}.py"),
                dependencies=interface.dependencies,
                requirement_links=interface.requirement_links,
                module=interface.module_name,
                interface_contract_refs=(interface.module_name,),
                interface_guarantees=guarantees,
                constraint_ids=(),
            )
        )

    if work_items:
        covered = {req_id for item in work_items for req_id in item.requirement_links}
        missing = [
            requirement.id for requirement in spec_map.requirements if requirement.id not in covered
        ]
        if missing:
            tail_work_item = work_items[-1]
            tail_module = modules[-1]
            merged_work_item_links = tuple([*tail_work_item.requirement_links, *missing])
            merged_module_links = tuple([*tail_module.requirement_links, *missing])
            work_items[-1] = WorkItemProposal(
                id=tail_work_item.id,
                title=tail_work_item.title,
                description=tail_work_item.description,
                owned_paths=tail_work_item.owned_paths,
                dependencies=tail_work_item.dependencies,
                requirement_links=merged_work_item_links,
                module=tail_work_item.module,
                interface_contract_refs=tail_work_item.interface_contract_refs,
                interface_guarantees=tail_work_item.interface_guarantees,
                constraint_ids=tail_work_item.constraint_ids,
            )
            modules[-1] = ModuleDefinition(
                name=tail_module.name,
                summary=tail_module.summary,
                owned_paths=tail_module.owned_paths,
                dependencies=tail_module.dependencies,
                requirement_links=merged_module_links,
                interface_contract_refs=tail_module.interface_contract_refs,
                interface_guarantees=tail_module.interface_guarantees,
            )

    return ArchitectOutput(
        modules=tuple(modules),
        work_items=tuple(work_items),
        adrs=(),
        constraint_suggestions=(),
    )


__all__ = [
    "ADRDraft",
    "ArchitectOutput",
    "ConstraintSuggestion",
    "ModuleDefinition",
    "WorkItemProposal",
    "build_deterministic_architect_output",
    "parse_architect_response",
    "validate_architect_output",
]
