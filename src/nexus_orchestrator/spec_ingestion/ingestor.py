"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/spec_ingestion/ingestor.py
Last updated: 2026-02-11

Purpose
- Parses the design_document.md (or arbitrary input spec) into a structured SpecMap.

What should be included in this file
- Markdown parsing strategy and guardrails (ignore code blocks intended to inject instructions).
- Requirement extraction: IDs, statements, acceptance criteria, NFR tags.
- Entity and interface extraction (modules, external integrations).
- Output validation against SpecMap schema.

Functional requirements
- Must fail with actionable errors if requirements are missing IDs or ambiguous.
- Must support merging multiple spec sources (design doc + ADRs + overrides).

Non-functional requirements
- Must be resilient to messy Markdown and headings.
- Must include prompt-injection defenses: treat spec as data, not instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

from nexus_orchestrator.domain import models as domain_models
from nexus_orchestrator.spec_ingestion.spec_map import (
    Entity,
    InterfaceContract,
    Requirement,
    SourceLocation,
    SpecMap,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python 3.11+ has UTC.
    UTC = timezone.utc  # noqa: UP017

_SECTION_ALIASES: Final[dict[str, str]] = {
    "requirements": "requirements",
    "functional requirements": "requirements",
    "reqs": "requirements",
    "interfaces": "interfaces",
    "interface contracts": "interfaces",
    "interface": "interfaces",
    "non-functional requirements": "nfr",
    "non functional requirements": "nfr",
    "nfr": "nfr",
    "nfrs": "nfr",
    "acceptance criteria": "acceptance",
    "acceptance": "acceptance",
    "entities": "entities",
    "domain entities": "entities",
    "glossary": "entities",
    "terms": "entities",
}

_ATX_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^\s{0,3}#{1,6}\s*(?P<text>.*?)\s*#*\s*$")
_SETEXT_UNDERLINE_RE: Final[re.Pattern[str]] = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
_LIST_ITEM_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>\s*)(?P<marker>[-*+]|\d+[.)])\s+(?P<text>\S.*)$"
)
_FENCE_START_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>[ ]{0,3})(?P<marker>`{3,}|~{3,}).*$"
)
_FENCE_CLOSE_RE: Final[re.Pattern[str]] = re.compile(r"^[ ]{0,3}(?P<marker>`{3,}|~{3,})\s*$")
_REQUIREMENT_LINE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<id>[A-Za-z][A-Za-z0-9-]*\d{4})\s*:\s*(?P<statement>\S.*)$"
)
_MODULE_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:(?:module|interface)\s+)?(?P<module>[A-Za-z0-9_.-]+)"
    r"(?:\s*\((?P<symbols>[^)]*)\))?\s*:\s*(?P<details>\S.*)$",
    flags=re.IGNORECASE,
)
_REQUIREMENT_REF_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9-]*\d{4}")
_VALIDATION_TIMESTAMP: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)


class SpecIngestError(Exception):
    """Structured ingestion validation failure."""

    path: Path
    line: int
    column: int | None
    section: str
    message: str
    hint: str

    def __init__(
        self,
        *,
        path: Path,
        line: int,
        section: str,
        message: str,
        hint: str,
        column: int | None = None,
    ) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.section = section
        self.message = message
        self.hint = hint
        location = f"{path}:{line}"
        if column is not None:
            location = f"{location}:{column}"
        super().__init__(f"{location} [{section}] {message} (hint: {hint})")


@dataclass(slots=True)
class _FenceState:
    marker_char: str
    marker_length: int


@dataclass(slots=True)
class _VisibleLine:
    line_number: int
    text: str


@dataclass(slots=True)
class _RequirementDraft:
    req_id: str
    statement: str
    acceptance_criteria: list[str]
    nfr_tags: list[str]
    source: SourceLocation


@dataclass(slots=True)
class _InterfaceDraft:
    module_name: str
    summary: str
    dependencies: list[str]
    requirement_links: list[str]
    exposed_symbols: list[str]
    source: SourceLocation


@dataclass(slots=True)
class _ParsedSource:
    source_document: str
    requirements: dict[str, _RequirementDraft] = field(default_factory=dict)
    interfaces: dict[str, _InterfaceDraft] = field(default_factory=dict)
    entities: dict[str, Entity] = field(default_factory=dict)
    glossary: dict[str, str] = field(default_factory=dict)
    global_acceptance: list[str] = field(default_factory=list)


def ingest_spec(spec_path: Path, additional_sources: Sequence[Path] | None = None) -> SpecMap:
    """
    Ingest one or more markdown specs into a deterministic ``SpecMap``.

    Merge order is deterministic: primary spec first, then ``additional_sources``
    sorted lexicographically by normalized path.
    """

    ordered_sources = _resolve_sources(spec_path, additional_sources)
    parsed_sources = [_parse_source(path) for path in ordered_sources]
    merged = _merge_sources(parsed_sources)
    _validate_cross_references(merged, ordered_sources[0])
    spec_map = _to_spec_map(merged, ordered_sources)

    try:
        _validate_against_domain_spec_map(spec_map)
    except ValueError as exc:
        raise SpecIngestError(
            path=ordered_sources[0],
            line=1,
            section="Validation",
            message="Failed domain-model compatibility validation",
            hint=str(exc),
        ) from exc

    return spec_map


def _resolve_sources(spec_path: Path, additional_sources: Sequence[Path] | None) -> list[Path]:
    primary = spec_path.resolve()
    if not primary.exists() or not primary.is_file():
        raise SpecIngestError(
            path=spec_path,
            line=1,
            section="I/O",
            message="Spec file does not exist or is not a file",
            hint="Pass a readable markdown file path",
        )

    extras = additional_sources if additional_sources is not None else ()
    resolved_extras: list[Path] = []
    seen: set[Path] = {primary}
    for candidate in extras:
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise SpecIngestError(
                path=candidate,
                line=1,
                section="I/O",
                message="Additional source does not exist or is not a file",
                hint="Pass only readable markdown file paths",
            )
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_extras.append(resolved)
    resolved_extras.sort(key=lambda path: _display_path(path))
    return [primary, *resolved_extras]


def _parse_source(path: Path) -> _ParsedSource:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecIngestError(
            path=path,
            line=1,
            section="I/O",
            message="Failed to read spec source",
            hint=str(exc),
        ) from exc

    visible_lines = _strip_non_semantic_lines(path, raw_text.splitlines())
    parsed = _ParsedSource(source_document=_display_path(path))

    section = ""
    previous_blank = True
    active_requirement_id: str | None = None
    active_requirement_indent = 0
    index = 0

    while index < len(visible_lines):
        line = visible_lines[index]

        heading = _parse_heading(visible_lines, index, previous_blank)
        if heading is not None:
            heading_text, consumed = heading
            section = _normalize_section_name(heading_text)
            active_requirement_id = None
            active_requirement_indent = 0
            index += consumed
            previous_blank = False
            continue

        stripped = line.text.strip()
        if not stripped:
            previous_blank = True
            index += 1
            continue

        list_item = _parse_list_item(line.text)
        if list_item is None:
            previous_blank = False
            index += 1
            continue

        indent, item_text = list_item
        if section in {"requirements", "nfr"}:
            is_nested_criterion = (
                active_requirement_id is not None and indent > active_requirement_indent
            )
            if not is_nested_criterion:
                req = _parse_requirement_item(path, line.line_number, section, item_text)
                if section == "nfr":
                    req.nfr_tags = _infer_nfr_tags(req.req_id, req.statement)
                existing = parsed.requirements.get(req.req_id)
                if existing is not None and existing.statement != req.statement:
                    raise SpecIngestError(
                        path=path,
                        line=line.line_number,
                        section="Requirements",
                        message=f"Duplicate requirement ID with conflicting statements: {req.req_id}",
                        hint="Keep one statement per requirement ID or use a distinct ID",
                    )
                parsed.requirements[req.req_id] = _merge_requirement_drafts(existing, req)
                active_requirement_id = req.req_id
                active_requirement_indent = indent
            else:
                criterion = item_text.strip()
                assert active_requirement_id is not None
                _append_unique(
                    parsed.requirements[active_requirement_id].acceptance_criteria,
                    criterion,
                )

        elif section == "acceptance":
            _append_unique(parsed.global_acceptance, item_text.strip())

        elif section == "interfaces":
            if indent > 3:
                index += 1
                previous_blank = False
                continue
            interface = _parse_interface_item(path, line.line_number, item_text)
            existing_interface = parsed.interfaces.get(interface.module_name)
            parsed.interfaces[interface.module_name] = _merge_interface_drafts(
                existing_interface,
                interface,
                path,
                line.line_number,
            )

        elif section == "entities":
            if indent > 3:
                index += 1
                previous_blank = False
                continue
            entity, glossary_entry = _parse_entity_item(path, line.line_number, item_text)
            existing_entity = parsed.entities.get(entity.name)
            if existing_entity is not None and existing_entity.description != entity.description:
                raise SpecIngestError(
                    path=path,
                    line=line.line_number,
                    section="Entities",
                    message=f"Entity definition conflict for {entity.name!r}",
                    hint="Use a single canonical definition per entity name",
                )
            parsed.entities[entity.name] = existing_entity or entity

            if glossary_entry is not None:
                key, value = glossary_entry
                existing_glossary = parsed.glossary.get(key)
                if existing_glossary is not None and existing_glossary != value:
                    raise SpecIngestError(
                        path=path,
                        line=line.line_number,
                        section="Glossary",
                        message=f"Glossary conflict for {key!r}",
                        hint="Use one canonical definition per glossary key",
                    )
                parsed.glossary[key] = value

        previous_blank = False
        index += 1

    return parsed


def _strip_non_semantic_lines(path: Path, lines: Sequence[str]) -> list[_VisibleLine]:
    del path  # Path is included for future scanner diagnostics.

    visible: list[_VisibleLine] = []
    fence_state: _FenceState | None = None
    in_comment = False

    for index, raw_line in enumerate(lines, start=1):
        if fence_state is not None:
            if _is_fence_close(raw_line, fence_state):
                fence_state = None
            continue

        fence_state = _parse_fence_start(raw_line)
        if fence_state is not None:
            continue

        sanitized_line, in_comment = _strip_html_comments(raw_line, in_comment)
        visible.append(_VisibleLine(line_number=index, text=sanitized_line))

    return visible


def _parse_fence_start(line: str) -> _FenceState | None:
    match = _FENCE_START_RE.match(line)
    if match is None:
        return None
    marker = match.group("marker")
    return _FenceState(marker_char=marker[0], marker_length=len(marker))


def _is_fence_close(line: str, state: _FenceState) -> bool:
    match = _FENCE_CLOSE_RE.match(line)
    if match is None:
        return False
    marker = match.group("marker")
    return marker[0] == state.marker_char and len(marker) >= state.marker_length


def _strip_html_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    output: list[str] = []
    cursor = 0

    while cursor < len(line):
        if in_comment:
            end = line.find("-->", cursor)
            if end < 0:
                return ("".join(output), True)
            cursor = end + 3
            in_comment = False
            continue

        start = line.find("<!--", cursor)
        if start < 0:
            output.append(line[cursor:])
            break
        output.append(line[cursor:start])
        cursor = start + 4
        in_comment = True

    return ("".join(output), in_comment)


def _parse_heading(
    lines: Sequence[_VisibleLine], index: int, previous_blank: bool
) -> tuple[str, int] | None:
    current = lines[index].text
    atx_match = _ATX_HEADING_RE.match(current)
    if atx_match is not None:
        heading_text = atx_match.group("text").strip()
        if heading_text:
            return (heading_text, 1)

    if index + 1 < len(lines):
        next_line = lines[index + 1].text
        if current.strip() and _SETEXT_UNDERLINE_RE.match(next_line) is not None:
            return (current.strip(), 2)

    normalized = _normalize_text(current)
    if previous_blank and normalized in _SECTION_ALIASES and _parse_list_item(current) is None:
        return (current.strip(), 1)
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _normalize_section_name(raw_heading: str) -> str:
    return _SECTION_ALIASES.get(_normalize_text(raw_heading), "")


def _parse_list_item(line: str) -> tuple[int, str] | None:
    match = _LIST_ITEM_RE.match(line)
    if match is None:
        return None
    indent = len(match.group("indent").expandtabs(4))
    text = match.group("text").strip()
    return (indent, text)


def _parse_requirement_item(path: Path, line: int, section: str, text: str) -> _RequirementDraft:
    match = _REQUIREMENT_LINE_RE.match(text)
    if match is None:
        raise SpecIngestError(
            path=path,
            line=line,
            section="Requirements" if section == "requirements" else "Non-Functional Requirements",
            message="Requirement bullet is missing an explicit ID prefix",
            hint="Expected 'REQ-...:' or 'NFR-...:' at the start of the bullet",
        )

    req_id = match.group("id").strip()
    statement = match.group("statement").strip()
    source = SourceLocation(
        path=_display_path(path),
        section="Requirements" if section == "requirements" else "Non-Functional Requirements",
        line=line,
    )
    return _RequirementDraft(
        req_id=req_id,
        statement=statement,
        acceptance_criteria=[],
        nfr_tags=[],
        source=source,
    )


def _merge_requirement_drafts(
    existing: _RequirementDraft | None,
    incoming: _RequirementDraft,
) -> _RequirementDraft:
    if existing is None:
        return incoming

    merged = _RequirementDraft(
        req_id=existing.req_id,
        statement=existing.statement,
        acceptance_criteria=list(existing.acceptance_criteria),
        nfr_tags=list(existing.nfr_tags),
        source=existing.source,
    )
    for criterion in incoming.acceptance_criteria:
        _append_unique(merged.acceptance_criteria, criterion)
    for tag in incoming.nfr_tags:
        _append_unique(merged.nfr_tags, tag)
    return merged


def _infer_nfr_tags(req_id: str, statement: str) -> list[str]:
    tags = ["nfr"]
    lowered = f"{req_id} {statement}".lower()

    if any(token in lowered for token in ("ms", "latency", "throughput", "performance", "fast")):
        tags.append("performance")
    if any(token in lowered for token in ("secret", "security", "auth", "encrypt")):
        tags.append("security")
    if any(token in lowered for token in ("reliab", "resum", "crash", "recover", "availability")):
        tags.append("reliability")
    return sorted(set(tags))


def _parse_interface_item(path: Path, line: int, text: str) -> _InterfaceDraft:
    match = _MODULE_HEADER_RE.match(text)
    if match is None:
        raise SpecIngestError(
            path=path,
            line=line,
            section="Interfaces",
            message="Interface bullet is ambiguous",
            hint="Use 'Module X: ... Depends on ...' format",
        )

    module_name = _clean_reference_token(match.group("module"))
    symbols = _split_reference_list(match.group("symbols") or "")
    details = match.group("details").strip()

    dependencies: list[str] = []
    if re.search(r"\bno dependencies\b", details, flags=re.IGNORECASE) is None:
        depends_match = re.search(r"\bdepends?\s+on\b(?P<deps>[^.]+)", details, flags=re.IGNORECASE)
        if depends_match is not None:
            dependencies = _split_reference_list(depends_match.group("deps"))

    req_links: list[str] = []
    implements_match = re.search(
        r"\bimplements?\b(?P<reqs>[^.]+)",
        details,
        flags=re.IGNORECASE,
    )
    if implements_match is not None:
        req_links = [
            token
            for token in _split_reference_list(implements_match.group("reqs"))
            if _REQUIREMENT_REF_RE.fullmatch(token) is not None
        ]

    return _InterfaceDraft(
        module_name=module_name,
        summary=details,
        dependencies=dependencies,
        requirement_links=req_links,
        exposed_symbols=symbols,
        source=SourceLocation(path=_display_path(path), section="Interfaces", line=line),
    )


def _merge_interface_drafts(
    existing: _InterfaceDraft | None,
    incoming: _InterfaceDraft,
    path: Path,
    line: int,
) -> _InterfaceDraft:
    if existing is None:
        return incoming

    summary = existing.summary if existing.summary else incoming.summary
    if existing.summary and incoming.summary and existing.summary != incoming.summary:
        raise SpecIngestError(
            path=path,
            line=line,
            section="Interfaces",
            message=f"Conflicting interface summary for module {incoming.module_name!r}",
            hint="Keep a single canonical interface summary per module",
        )

    merged = _InterfaceDraft(
        module_name=existing.module_name,
        summary=summary,
        dependencies=list(existing.dependencies),
        requirement_links=list(existing.requirement_links),
        exposed_symbols=list(existing.exposed_symbols),
        source=existing.source,
    )
    for dependency in incoming.dependencies:
        _append_unique(merged.dependencies, dependency)
    for req_id in incoming.requirement_links:
        _append_unique(merged.requirement_links, req_id)
    for symbol in incoming.exposed_symbols:
        _append_unique(merged.exposed_symbols, symbol)
    return merged


def _parse_entity_item(
    path: Path,
    line: int,
    text: str,
) -> tuple[Entity, tuple[str, str] | None]:
    name_part: str
    description_part: str
    if ":" in text:
        name_part, description_part = text.split(":", 1)
    elif " - " in text:
        name_part, description_part = text.split(" - ", 1)
    else:
        name_part, description_part = text, ""

    name = _clean_reference_token(name_part)
    description = description_part.strip()
    if not name:
        raise SpecIngestError(
            path=path,
            line=line,
            section="Entities",
            message="Entity bullet is missing a name",
            hint="Use 'EntityName: short definition'",
        )

    entity = Entity(
        name=name,
        description=description,
        source=SourceLocation(path=_display_path(path), section="Entities", line=line),
    )
    glossary_entry = (name, description) if description else None
    return (entity, glossary_entry)


def _split_reference_list(raw: str) -> list[str]:
    if not raw.strip():
        return []
    normalized = raw.replace("&", ",")
    normalized = re.sub(r"\band\b", ",", normalized, flags=re.IGNORECASE)
    pieces = [piece.strip() for piece in normalized.split(",")]
    values: list[str] = []
    for piece in pieces:
        token = _clean_reference_token(piece)
        if not token:
            continue
        _append_unique(values, token)
    return values


def _clean_reference_token(raw: str) -> str:
    stripped = raw.strip()
    stripped = re.sub(r"^\bmodule\b\s+", "", stripped, flags=re.IGNORECASE)
    stripped = stripped.strip("`'\"[](){}")
    stripped = stripped.strip(" .;")
    return stripped


def _append_unique(values: list[str], item: str) -> None:
    normalized = item.strip()
    if not normalized:
        return
    if normalized not in values:
        values.append(normalized)


def _merge_sources(parsed_sources: Sequence[_ParsedSource]) -> _ParsedSource:
    if not parsed_sources:
        raise ValueError("No parsed sources to merge")

    merged = _ParsedSource(source_document=parsed_sources[0].source_document)
    for source in parsed_sources:
        for req_id, req in source.requirements.items():
            existing_req = merged.requirements.get(req_id)
            if existing_req is None:
                merged.requirements[req_id] = _RequirementDraft(
                    req_id=req.req_id,
                    statement=req.statement,
                    acceptance_criteria=list(req.acceptance_criteria),
                    nfr_tags=list(req.nfr_tags),
                    source=req.source,
                )
                continue
            if existing_req.statement != req.statement:
                raise SpecIngestError(
                    path=Path(req.source.path),
                    line=req.source.line,
                    section=req.source.section,
                    message=f"Requirement conflict for ID {req_id}",
                    hint="Conflicting statements require explicit override support",
                )
            for criterion in req.acceptance_criteria:
                _append_unique(existing_req.acceptance_criteria, criterion)
            for tag in req.nfr_tags:
                _append_unique(existing_req.nfr_tags, tag)

        for module_name, interface in source.interfaces.items():
            existing_interface = merged.interfaces.get(module_name)
            if existing_interface is None:
                merged.interfaces[module_name] = _InterfaceDraft(
                    module_name=interface.module_name,
                    summary=interface.summary,
                    dependencies=list(interface.dependencies),
                    requirement_links=list(interface.requirement_links),
                    exposed_symbols=list(interface.exposed_symbols),
                    source=interface.source,
                )
                continue

            if (
                existing_interface.summary
                and interface.summary
                and existing_interface.summary != interface.summary
            ):
                raise SpecIngestError(
                    path=Path(interface.source.path),
                    line=interface.source.line,
                    section=interface.source.section,
                    message=f"Interface conflict for module {module_name}",
                    hint="Conflicting interface summaries must be reconciled",
                )
            if not existing_interface.summary:
                existing_interface.summary = interface.summary
            for dependency in interface.dependencies:
                _append_unique(existing_interface.dependencies, dependency)
            for req_id in interface.requirement_links:
                _append_unique(existing_interface.requirement_links, req_id)
            for symbol in interface.exposed_symbols:
                _append_unique(existing_interface.exposed_symbols, symbol)

        for name, entity in source.entities.items():
            existing_entity = merged.entities.get(name)
            if existing_entity is not None and existing_entity.description != entity.description:
                raise SpecIngestError(
                    path=Path(entity.source.path),
                    line=entity.source.line,
                    section=entity.source.section,
                    message=f"Entity conflict for {name!r}",
                    hint="Use one canonical entity definition per name",
                )
            merged.entities[name] = existing_entity or entity

        for key, value in source.glossary.items():
            existing_value = merged.glossary.get(key)
            if existing_value is not None and existing_value != value:
                raise SpecIngestError(
                    path=Path(source.source_document),
                    line=1,
                    section="Glossary",
                    message=f"Glossary conflict for key {key!r}",
                    hint="Glossary key definitions must match across merged sources",
                )
            merged.glossary[key] = value

        for criterion in source.global_acceptance:
            _append_unique(merged.global_acceptance, criterion)

    for req in merged.requirements.values():
        if "nfr" in {tag.lower() for tag in req.nfr_tags}:
            continue
        for global_criterion in merged.global_acceptance:
            _append_unique(req.acceptance_criteria, global_criterion)

    return merged


def _validate_cross_references(merged: _ParsedSource, primary_path: Path) -> None:
    if not merged.requirements:
        raise SpecIngestError(
            path=primary_path,
            line=1,
            section="Requirements",
            message="No requirements were extracted from the provided source(s)",
            hint="Add a 'Requirements' section with bullet items like 'REQ-...: ...'",
        )

    requirement_ids = set(merged.requirements)
    interface_modules = set(merged.interfaces)

    for interface in merged.interfaces.values():
        for dependency in interface.dependencies:
            if dependency not in interface_modules:
                raise SpecIngestError(
                    path=Path(interface.source.path),
                    line=interface.source.line,
                    section="Interfaces",
                    message=(
                        f"Interface {interface.module_name!r} depends on unknown module {dependency!r}"
                    ),
                    hint="Declare the dependency module in the Interfaces section",
                )
        for req_id in interface.requirement_links:
            if req_id not in requirement_ids:
                raise SpecIngestError(
                    path=Path(interface.source.path),
                    line=interface.source.line,
                    section="Interfaces",
                    message=f"Interface {interface.module_name!r} references unknown requirement {req_id!r}",
                    hint="Ensure interface requirement links match extracted requirement IDs",
                )


def _to_spec_map(merged: _ParsedSource, ordered_sources: Sequence[Path]) -> SpecMap:
    requirements = tuple(
        Requirement(
            id=req.req_id,
            statement=req.statement,
            acceptance_criteria=tuple(req.acceptance_criteria),
            nfr_tags=tuple(req.nfr_tags),
            source=req.source,
        )
        for req in sorted(merged.requirements.values(), key=lambda item: item.req_id)
    )
    interfaces = tuple(
        InterfaceContract(
            module_name=interface.module_name,
            summary=interface.summary,
            dependencies=tuple(interface.dependencies),
            requirement_links=tuple(interface.requirement_links),
            exposed_symbols=tuple(interface.exposed_symbols),
            source=interface.source,
        )
        for interface in sorted(merged.interfaces.values(), key=lambda item: item.module_name)
    )
    entities = tuple(merged.entities[name] for name in sorted(merged.entities))

    return SpecMap(
        version=1,
        source_documents=tuple(_display_path(path) for path in ordered_sources),
        requirements=requirements,
        entities=entities,
        interfaces=interfaces,
        glossary=dict(sorted(merged.glossary.items(), key=lambda item: item[0])),
        global_acceptance_criteria=tuple(merged.global_acceptance),
    )


def _validate_against_domain_spec_map(spec_map: SpecMap) -> None:
    req_id_map = {
        requirement.id: f"REQ-{index:04d}"
        for index, requirement in enumerate(spec_map.requirements, start=1)
    }
    if len(req_id_map) > 9999:
        raise ValueError("Too many requirements to map into domain REQ-0001 format")

    mapped_requirements = tuple(
        domain_models.Requirement(
            id=req_id_map[requirement.id],
            statement=requirement.statement,
            acceptance_criteria=requirement.acceptance_criteria,
            nfr_tags=requirement.nfr_tags,
            source=(
                f"{requirement.source.path}#{requirement.source.section.lower().replace(' ', '-')}"
                f":L{requirement.source.line}"
            ),
        )
        for requirement in spec_map.requirements
    )
    source_document = spec_map.source_documents[0]
    domain_source_document = (
        source_document if not Path(source_document).is_absolute() else Path(source_document).name
    )

    _ = domain_models.SpecMap(
        source_document=domain_source_document,
        requirements=mapped_requirements,
        created_at=_VALIDATION_TIMESTAMP,
        entities=tuple(entity.name for entity in spec_map.entities),
        interfaces=tuple(interface.module_name for interface in spec_map.interfaces),
        glossary=spec_map.glossary,
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved.is_relative_to(cwd):
        return resolved.relative_to(cwd).as_posix()
    return resolved.as_posix()


__all__ = ["SpecIngestError", "ingest_spec"]
