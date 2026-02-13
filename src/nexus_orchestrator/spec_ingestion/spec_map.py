"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/spec_ingestion/spec_map.py
Last updated: 2026-02-11

Purpose
- SpecMap schema definition, serialization, and helpers.

What should be included in this file
- SpecMap versioning and JSON/TOML serialization guidance.
- Utilities for tracing from requirement -> planned work items.

Functional requirements
- Must support stable requirement IDs and updates without breaking references.

Non-functional requirements
- Must support incremental ingestion (diff-based updates).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from types import ModuleType

    from nexus_orchestrator.domain.models import WorkItem

_tomllib: ModuleType | None
try:
    import tomllib as _tomllib
except ImportError:  # pragma: no cover - Python 3.11+ ships tomllib.
    try:
        import tomli as _tomllib
    except ImportError:  # pragma: no cover
        _tomllib = None


def _strip_text(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _unique_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        text = raw.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return tuple(ordered)


def _as_string_sequence(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, tuple):
        items = list(value)
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError(f"{field_name} must be a list")
    parsed: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string")
        parsed.append(item)
    return tuple(parsed)


def _as_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} must use string keys")
        parsed[key] = item
    return parsed


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """Traceable location for parsed data in source markdown."""

    path: str
    section: str
    line: int
    column: int | None = None

    def __post_init__(self) -> None:
        normalized_path = _strip_text(self.path, "SourceLocation.path")
        normalized_section = _strip_text(self.section, "SourceLocation.section")
        if self.line < 1:
            raise ValueError("SourceLocation.line must be >= 1")
        if self.column is not None and self.column < 1:
            raise ValueError("SourceLocation.column must be >= 1")
        object.__setattr__(self, "path", normalized_path)
        object.__setattr__(self, "section", normalized_section)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "section": self.section,
            "line": self.line,
        }
        if self.column is not None:
            payload["column"] = self.column
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SourceLocation:
        path_raw = data.get("path")
        section_raw = data.get("section")
        line_raw = data.get("line")
        column_raw = data.get("column")

        if not isinstance(path_raw, str):
            raise ValueError("SourceLocation.path must be a string")
        if not isinstance(section_raw, str):
            raise ValueError("SourceLocation.section must be a string")
        if not isinstance(line_raw, int):
            raise ValueError("SourceLocation.line must be an integer")
        if column_raw is not None and not isinstance(column_raw, int):
            raise ValueError("SourceLocation.column must be an integer when set")
        return cls(path=path_raw, section=section_raw, line=line_raw, column=column_raw)


@dataclass(frozen=True, slots=True)
class Requirement:
    """Ingestion requirement record with source traceability."""

    id: str
    statement: str
    acceptance_criteria: tuple[str, ...] = ()
    nfr_tags: tuple[str, ...] = ()
    source: SourceLocation = field(
        default_factory=lambda: SourceLocation(path="unknown", section="unknown", line=1)
    )

    def __post_init__(self) -> None:
        req_id = _strip_text(self.id, "Requirement.id")
        statement = _strip_text(self.statement, "Requirement.statement")

        criteria = _unique_preserve_order(self.acceptance_criteria)
        tags = tuple(sorted({tag.strip().lower() for tag in self.nfr_tags if tag.strip()}))
        object.__setattr__(self, "id", req_id)
        object.__setattr__(self, "statement", statement)
        object.__setattr__(self, "acceptance_criteria", criteria)
        object.__setattr__(self, "nfr_tags", tags)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "statement": self.statement,
            "acceptance_criteria": list(self.acceptance_criteria),
            "nfr_tags": list(self.nfr_tags),
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Requirement:
        req_id = data.get("id")
        statement = data.get("statement")
        acceptance_raw = data.get("acceptance_criteria", [])
        tags_raw = data.get("nfr_tags", [])
        source_raw = data.get("source")

        if not isinstance(req_id, str):
            raise ValueError("Requirement.id must be a string")
        if not isinstance(statement, str):
            raise ValueError("Requirement.statement must be a string")

        acceptance = _as_string_sequence(acceptance_raw, "Requirement.acceptance_criteria")
        tags = _as_string_sequence(tags_raw, "Requirement.nfr_tags")
        source_mapping = _as_mapping(source_raw, "Requirement.source")
        return cls(
            id=req_id,
            statement=statement,
            acceptance_criteria=acceptance,
            nfr_tags=tags,
            source=SourceLocation.from_dict(source_mapping),
        )


@dataclass(frozen=True, slots=True)
class Entity:
    """Minimal extracted entity definition."""

    name: str
    description: str
    source: SourceLocation

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _strip_text(self.name, "Entity.name"))
        object.__setattr__(
            self,
            "description",
            self.description.strip(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Entity:
        name = data.get("name")
        description = data.get("description", "")
        source = data.get("source")
        if not isinstance(name, str):
            raise ValueError("Entity.name must be a string")
        if not isinstance(description, str):
            raise ValueError("Entity.description must be a string")
        source_mapping = _as_mapping(source, "Entity.source")
        return cls(
            name=name, description=description, source=SourceLocation.from_dict(source_mapping)
        )


@dataclass(frozen=True, slots=True)
class InterfaceContract:
    """Module/interface contract extracted from source docs."""

    module_name: str
    summary: str
    dependencies: tuple[str, ...] = ()
    requirement_links: tuple[str, ...] = ()
    exposed_symbols: tuple[str, ...] = ()
    source: SourceLocation = field(
        default_factory=lambda: SourceLocation(path="unknown", section="unknown", line=1)
    )

    def __post_init__(self) -> None:
        module_name = _strip_text(self.module_name, "InterfaceContract.module_name")
        summary = self.summary.strip()
        dependencies = tuple(sorted(_unique_preserve_order(self.dependencies)))
        requirement_links = tuple(sorted(_unique_preserve_order(self.requirement_links)))
        exposed_symbols = tuple(sorted(_unique_preserve_order(self.exposed_symbols)))
        object.__setattr__(self, "module_name", module_name)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "requirement_links", requirement_links)
        object.__setattr__(self, "exposed_symbols", exposed_symbols)

    def to_dict(self) -> dict[str, object]:
        return {
            "module_name": self.module_name,
            "summary": self.summary,
            "dependencies": list(self.dependencies),
            "requirement_links": list(self.requirement_links),
            "exposed_symbols": list(self.exposed_symbols),
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> InterfaceContract:
        module_name = data.get("module_name")
        summary = data.get("summary", "")
        source = data.get("source")
        if not isinstance(module_name, str):
            raise ValueError("InterfaceContract.module_name must be a string")
        if not isinstance(summary, str):
            raise ValueError("InterfaceContract.summary must be a string")
        dependencies = _as_string_sequence(
            data.get("dependencies", []), "InterfaceContract.dependencies"
        )
        requirement_links = _as_string_sequence(
            data.get("requirement_links", []),
            "InterfaceContract.requirement_links",
        )
        symbols = _as_string_sequence(
            data.get("exposed_symbols", []), "InterfaceContract.exposed_symbols"
        )
        source_mapping = _as_mapping(source, "InterfaceContract.source")
        return cls(
            module_name=module_name,
            summary=summary,
            dependencies=dependencies,
            requirement_links=requirement_links,
            exposed_symbols=symbols,
            source=SourceLocation.from_dict(source_mapping),
        )


@dataclass(frozen=True, slots=True)
class SpecMap:
    """Normalized and deterministic spec map for planning input."""

    version: int
    source_documents: tuple[str, ...]
    requirements: tuple[Requirement, ...]
    entities: tuple[Entity, ...] = ()
    interfaces: tuple[InterfaceContract, ...] = ()
    glossary: dict[str, str] = field(default_factory=dict)
    global_acceptance_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("SpecMap.version must be >= 1")

        source_docs = tuple(sorted(_unique_preserve_order(self.source_documents)))
        if not source_docs:
            raise ValueError("SpecMap.source_documents must include at least one path")
        object.__setattr__(self, "source_documents", source_docs)

        reqs = tuple(sorted(self.requirements, key=lambda req: req.id))
        if not reqs:
            raise ValueError("SpecMap.requirements must include at least one requirement")
        req_ids = [req.id for req in reqs]
        if len(req_ids) != len(set(req_ids)):
            raise ValueError("SpecMap.requirements contains duplicate requirement IDs")
        object.__setattr__(self, "requirements", reqs)

        entities = tuple(sorted(self.entities, key=lambda entity: entity.name))
        entity_names = [entity.name for entity in entities]
        if len(entity_names) != len(set(entity_names)):
            raise ValueError("SpecMap.entities contains duplicate entity names")
        object.__setattr__(self, "entities", entities)

        interfaces = tuple(sorted(self.interfaces, key=lambda interface: interface.module_name))
        module_names = [interface.module_name for interface in interfaces]
        if len(module_names) != len(set(module_names)):
            raise ValueError("SpecMap.interfaces contains duplicate module names")
        object.__setattr__(self, "interfaces", interfaces)

        normalized_glossary: dict[str, str] = {}
        for key in sorted(self.glossary):
            normalized_key = _strip_text(key, "SpecMap.glossary key")
            normalized_value = self.glossary[key].strip()
            if not normalized_value:
                raise ValueError(f"SpecMap.glossary[{normalized_key!r}] must not be empty")
            normalized_glossary[normalized_key] = normalized_value
        object.__setattr__(self, "glossary", normalized_glossary)

        object.__setattr__(
            self,
            "global_acceptance_criteria",
            _unique_preserve_order(self.global_acceptance_criteria),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source_documents": list(self.source_documents),
            "requirements": [requirement.to_dict() for requirement in self.requirements],
            "entities": [entity.to_dict() for entity in self.entities],
            "interfaces": [interface.to_dict() for interface in self.interfaces],
            "glossary": dict(self.glossary),
            "global_acceptance_criteria": list(self.global_acceptance_criteria),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> SpecMap:
        version_raw = data.get("version", 1)
        if not isinstance(version_raw, int):
            raise ValueError("SpecMap.version must be an integer")
        source_docs = _as_string_sequence(
            data.get("source_documents", []), "SpecMap.source_documents"
        )

        requirements_raw = data.get("requirements")
        if not isinstance(requirements_raw, list):
            raise ValueError("SpecMap.requirements must be a list")
        requirements: list[Requirement] = []
        for index, item in enumerate(requirements_raw):
            requirements.append(
                Requirement.from_dict(_as_mapping(item, f"SpecMap.requirements[{index}]"))
            )

        entities_raw = data.get("entities", [])
        if not isinstance(entities_raw, list):
            raise ValueError("SpecMap.entities must be a list")
        entities: list[Entity] = []
        for index, item in enumerate(entities_raw):
            entities.append(Entity.from_dict(_as_mapping(item, f"SpecMap.entities[{index}]")))

        interfaces_raw = data.get("interfaces", [])
        if not isinstance(interfaces_raw, list):
            raise ValueError("SpecMap.interfaces must be a list")
        interfaces: list[InterfaceContract] = []
        for index, item in enumerate(interfaces_raw):
            interfaces.append(
                InterfaceContract.from_dict(_as_mapping(item, f"SpecMap.interfaces[{index}]"))
            )

        glossary_raw = data.get("glossary", {})
        if not isinstance(glossary_raw, Mapping):
            raise ValueError("SpecMap.glossary must be an object")
        glossary: dict[str, str] = {}
        for key, value in glossary_raw.items():
            if not isinstance(key, str):
                raise ValueError("SpecMap.glossary keys must be strings")
            if not isinstance(value, str):
                raise ValueError(f"SpecMap.glossary[{key!r}] must be a string")
            glossary[key] = value

        global_acceptance = _as_string_sequence(
            data.get("global_acceptance_criteria", []),
            "SpecMap.global_acceptance_criteria",
        )
        return cls(
            version=version_raw,
            source_documents=source_docs,
            requirements=tuple(requirements),
            entities=tuple(entities),
            interfaces=tuple(interfaces),
            glossary=glossary,
            global_acceptance_criteria=global_acceptance,
        )


@dataclass(frozen=True, slots=True)
class RequirementDiff:
    """Requirement-level change."""

    requirement_id: str
    old: Requirement
    new: Requirement


@dataclass(frozen=True, slots=True)
class InterfaceDiff:
    """Interface-level change."""

    module_name: str
    old: InterfaceContract
    new: InterfaceContract


@dataclass(frozen=True, slots=True)
class EntityDiff:
    """Entity-level change."""

    entity_name: str
    old: Entity
    new: Entity


@dataclass(frozen=True, slots=True)
class SpecMapDiff:
    """Deterministic incremental diff between two SpecMap instances."""

    added_requirement_ids: tuple[str, ...] = ()
    removed_requirement_ids: tuple[str, ...] = ()
    modified_requirements: tuple[RequirementDiff, ...] = ()
    added_interface_modules: tuple[str, ...] = ()
    removed_interface_modules: tuple[str, ...] = ()
    modified_interfaces: tuple[InterfaceDiff, ...] = ()
    added_entities: tuple[str, ...] = ()
    removed_entities: tuple[str, ...] = ()
    modified_entities: tuple[EntityDiff, ...] = ()
    glossary_added_keys: tuple[str, ...] = ()
    glossary_removed_keys: tuple[str, ...] = ()
    glossary_modified_keys: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.added_requirement_ids,
                self.removed_requirement_ids,
                self.modified_requirements,
                self.added_interface_modules,
                self.removed_interface_modules,
                self.modified_interfaces,
                self.added_entities,
                self.removed_entities,
                self.modified_entities,
                self.glossary_added_keys,
                self.glossary_removed_keys,
                self.glossary_modified_keys,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "added_requirement_ids": list(self.added_requirement_ids),
            "removed_requirement_ids": list(self.removed_requirement_ids),
            "modified_requirements": [
                {
                    "requirement_id": item.requirement_id,
                    "old": item.old.to_dict(),
                    "new": item.new.to_dict(),
                }
                for item in self.modified_requirements
            ],
            "added_interface_modules": list(self.added_interface_modules),
            "removed_interface_modules": list(self.removed_interface_modules),
            "modified_interfaces": [
                {
                    "module_name": item.module_name,
                    "old": item.old.to_dict(),
                    "new": item.new.to_dict(),
                }
                for item in self.modified_interfaces
            ],
            "added_entities": list(self.added_entities),
            "removed_entities": list(self.removed_entities),
            "modified_entities": [
                {
                    "entity_name": item.entity_name,
                    "old": item.old.to_dict(),
                    "new": item.new.to_dict(),
                }
                for item in self.modified_entities
            ],
            "glossary_added_keys": list(self.glossary_added_keys),
            "glossary_removed_keys": list(self.glossary_removed_keys),
            "glossary_modified_keys": list(self.glossary_modified_keys),
        }


def serialize_spec_map(
    spec_map: SpecMap,
    *,
    format: Literal["json", "toml"] = "json",  # noqa: A002
) -> str:
    """Serialize a spec map deterministically as JSON or TOML."""

    payload = spec_map.to_dict()
    if format == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if format == "toml":
        return _serialize_toml(spec_map)
    raise ValueError(f"Unsupported format: {format!r}")


def deserialize_spec_map(
    data: str,
    *,
    format: Literal["json", "toml"] = "json",  # noqa: A002
) -> SpecMap:
    """Deserialize JSON/TOML into a validated SpecMap."""

    if format == "json":
        parsed = json.loads(data)
    elif format == "toml":
        if _tomllib is None:  # pragma: no cover
            raise RuntimeError("tomllib is unavailable")
        parsed = _tomllib.loads(data)
    else:
        raise ValueError(f"Unsupported format: {format!r}")

    if not isinstance(parsed, Mapping):
        raise ValueError("Serialized spec map root must be an object")
    parsed_mapping = _as_mapping(parsed, "SpecMap")
    return SpecMap.from_dict(parsed_mapping)


def diff_spec_maps(old: SpecMap, new: SpecMap) -> SpecMapDiff:
    """Compute a deterministic diff between two spec maps."""

    old_requirements = {requirement.id: requirement for requirement in old.requirements}
    new_requirements = {requirement.id: requirement for requirement in new.requirements}

    added_requirement_ids = tuple(sorted(set(new_requirements) - set(old_requirements)))
    removed_requirement_ids = tuple(sorted(set(old_requirements) - set(new_requirements)))
    modified_requirements = tuple(
        RequirementDiff(
            requirement_id=req_id,
            old=old_requirements[req_id],
            new=new_requirements[req_id],
        )
        for req_id in sorted(set(old_requirements) & set(new_requirements))
        if old_requirements[req_id] != new_requirements[req_id]
    )

    old_interfaces = {interface.module_name: interface for interface in old.interfaces}
    new_interfaces = {interface.module_name: interface for interface in new.interfaces}
    added_interface_modules = tuple(sorted(set(new_interfaces) - set(old_interfaces)))
    removed_interface_modules = tuple(sorted(set(old_interfaces) - set(new_interfaces)))
    modified_interfaces = tuple(
        InterfaceDiff(
            module_name=module_name,
            old=old_interfaces[module_name],
            new=new_interfaces[module_name],
        )
        for module_name in sorted(set(old_interfaces) & set(new_interfaces))
        if old_interfaces[module_name] != new_interfaces[module_name]
    )

    old_entities = {entity.name: entity for entity in old.entities}
    new_entities = {entity.name: entity for entity in new.entities}
    added_entities = tuple(sorted(set(new_entities) - set(old_entities)))
    removed_entities = tuple(sorted(set(old_entities) - set(new_entities)))
    modified_entities = tuple(
        EntityDiff(entity_name=name, old=old_entities[name], new=new_entities[name])
        for name in sorted(set(old_entities) & set(new_entities))
        if old_entities[name] != new_entities[name]
    )

    old_glossary_keys = set(old.glossary)
    new_glossary_keys = set(new.glossary)
    glossary_added_keys = tuple(sorted(new_glossary_keys - old_glossary_keys))
    glossary_removed_keys = tuple(sorted(old_glossary_keys - new_glossary_keys))
    glossary_modified_keys = tuple(
        sorted(
            key
            for key in old_glossary_keys & new_glossary_keys
            if old.glossary[key] != new.glossary[key]
        )
    )

    return SpecMapDiff(
        added_requirement_ids=added_requirement_ids,
        removed_requirement_ids=removed_requirement_ids,
        modified_requirements=modified_requirements,
        added_interface_modules=added_interface_modules,
        removed_interface_modules=removed_interface_modules,
        modified_interfaces=modified_interfaces,
        added_entities=added_entities,
        removed_entities=removed_entities,
        modified_entities=modified_entities,
        glossary_added_keys=glossary_added_keys,
        glossary_removed_keys=glossary_removed_keys,
        glossary_modified_keys=glossary_modified_keys,
    )


def trace_requirement_to_work_items(req_id: str, work_items: Sequence[WorkItem]) -> list[str]:
    """Return deterministic work-item IDs that reference a requirement ID."""

    normalized_req_id = req_id.strip()
    if not normalized_req_id:
        raise ValueError("req_id must not be empty")

    matches = {item.id for item in work_items if normalized_req_id in item.requirement_links}
    return sorted(matches)


def _serialize_toml(spec_map: SpecMap) -> str:
    lines: list[str] = []
    lines.append(f"version = {spec_map.version}")
    lines.append(f"source_documents = {_toml_value(list(spec_map.source_documents))}")
    lines.append(
        f"global_acceptance_criteria = {_toml_value(list(spec_map.global_acceptance_criteria))}"
    )

    if spec_map.glossary:
        lines.append("")
        lines.append("[glossary]")
        for key in sorted(spec_map.glossary):
            lines.append(f"{_toml_key(key)} = {_toml_value(spec_map.glossary[key])}")

    for requirement in spec_map.requirements:
        lines.append("")
        lines.append("[[requirements]]")
        lines.append(f"id = {_toml_value(requirement.id)}")
        lines.append(f"statement = {_toml_value(requirement.statement)}")
        lines.append(f"acceptance_criteria = {_toml_value(list(requirement.acceptance_criteria))}")
        lines.append(f"nfr_tags = {_toml_value(list(requirement.nfr_tags))}")
        lines.append(f"source = {_toml_value(requirement.source.to_dict())}")

    for entity in spec_map.entities:
        lines.append("")
        lines.append("[[entities]]")
        lines.append(f"name = {_toml_value(entity.name)}")
        lines.append(f"description = {_toml_value(entity.description)}")
        lines.append(f"source = {_toml_value(entity.source.to_dict())}")

    for interface in spec_map.interfaces:
        lines.append("")
        lines.append("[[interfaces]]")
        lines.append(f"module_name = {_toml_value(interface.module_name)}")
        lines.append(f"summary = {_toml_value(interface.summary)}")
        lines.append(f"dependencies = {_toml_value(list(interface.dependencies))}")
        lines.append(f"requirement_links = {_toml_value(list(interface.requirement_links))}")
        lines.append(f"exposed_symbols = {_toml_value(list(interface.exposed_symbols))}")
        lines.append(f"source = {_toml_value(interface.source.to_dict())}")

    return "\n".join(lines) + "\n"


def _toml_key(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items = [
            f"{_toml_key(key)} = {_toml_value(item)}"
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        ]
        return "{ " + ", ".join(items) + " }"
    raise ValueError(f"Unsupported TOML value type: {type(value).__name__}")


__all__ = [
    "Entity",
    "EntityDiff",
    "InterfaceContract",
    "InterfaceDiff",
    "Requirement",
    "RequirementDiff",
    "SourceLocation",
    "SpecMap",
    "SpecMapDiff",
    "deserialize_spec_map",
    "diff_spec_maps",
    "serialize_spec_map",
    "trace_requirement_to_work_items",
]
