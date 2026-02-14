"""Deterministic constraint compilation from architect output into TaskGraph/WorkItems."""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml

from nexus_orchestrator.domain import ids as domain_ids
from nexus_orchestrator.domain.models import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    TaskGraph,
    WorkItem,
)
from nexus_orchestrator.planning.architect_interface import (
    ArchitectOutput,
    ModuleDefinition,
    WorkItemProposal,
    validate_architect_output,
)

if TYPE_CHECKING:
    from nexus_orchestrator.spec_ingestion.spec_map import SpecMap

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

_COMPILED_AT: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
_DEFAULT_REGISTRY_PATH: Final[str] = "constraints/registry"
_DEFAULT_RUN_ID: Final[str] = "run-00000000000000000000000000"
_PROPAGATED_CATEGORIES: Final[set[str]] = {"security", "style"}
_KNOWN_CHECKER_BINDINGS: Final[set[str]] = {
    "build_checker",
    "documentation_checker",
    "lint_checker",
    "performance_checker",
    "reliability_checker",
    "schema_checker",
    "scope_checker",
    "security_checker",
    "test_checker",
    "typecheck_checker",
}

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class _ConstraintTemplate:
    id: str
    severity: ConstraintSeverity
    category: str
    description: str
    checker_binding: str
    parameters: dict[str, JSONValue]
    requirement_links: tuple[str, ...]
    source: ConstraintSource


@dataclass(frozen=True, slots=True)
class CompilationResult:
    task_graph: TaskGraph | None
    work_items: tuple[WorkItem, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


def _as_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _as_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} must use string keys")
        out[key] = item
    return out


def _as_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a list of strings")
    if not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a list of strings")
    ordered: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        normalized = _as_non_empty_str(item, f"{field_name}[{index}]")
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _as_json_value(value: object, field_name: str) -> JSONValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite floats")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_as_json_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, tuple):
        return [_as_json_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} must use string keys")
            out[key] = _as_json_value(item, f"{field_name}.{key}")
        return out
    raise ValueError(f"{field_name} contains non-JSON-serializable value: {type(value).__name__}")


def _as_json_object(value: object, field_name: str) -> dict[str, JSONValue]:
    parsed = _as_json_value(value, field_name)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be an object")
    return parsed


def _parse_constraint_source(value: object, field_name: str) -> ConstraintSource:
    normalized = _as_non_empty_str(value, field_name).lower()
    if normalized == ConstraintSource.MANUAL.value:
        return ConstraintSource.MANUAL
    if normalized == ConstraintSource.SPEC_DERIVED.value:
        return ConstraintSource.SPEC_DERIVED
    if normalized == ConstraintSource.FAILURE_DERIVED.value:
        return ConstraintSource.FAILURE_DERIVED
    raise ValueError(
        f"{field_name} must be one of: {ConstraintSource.MANUAL.value}, "
        f"{ConstraintSource.SPEC_DERIVED.value}, {ConstraintSource.FAILURE_DERIVED.value}"
    )


def _parse_constraint_severity(value: object, field_name: str) -> ConstraintSeverity:
    normalized = _as_non_empty_str(value, field_name).lower()
    if normalized == ConstraintSeverity.MUST.value:
        return ConstraintSeverity.MUST
    if normalized == ConstraintSeverity.SHOULD.value:
        return ConstraintSeverity.SHOULD
    if normalized == ConstraintSeverity.MAY.value:
        return ConstraintSeverity.MAY
    raise ValueError(
        f"{field_name} must be one of: {ConstraintSeverity.MUST.value}, "
        f"{ConstraintSeverity.SHOULD.value}, {ConstraintSeverity.MAY.value}"
    )


def _coerce_constraint_template(payload: Mapping[str, object], path: str) -> _ConstraintTemplate:
    parsed = _as_mapping(payload, path)
    checker_raw = parsed.get("checker_binding", parsed.get("checker"))
    template = _ConstraintTemplate(
        id=_as_non_empty_str(parsed.get("id"), f"{path}.id"),
        severity=_parse_constraint_severity(parsed.get("severity"), f"{path}.severity"),
        category=_as_non_empty_str(parsed.get("category"), f"{path}.category").lower(),
        description=_as_non_empty_str(parsed.get("description"), f"{path}.description"),
        checker_binding=_as_non_empty_str(checker_raw, f"{path}.checker_binding"),
        parameters=_as_json_object(parsed.get("parameters", {}), f"{path}.parameters"),
        requirement_links=_as_string_tuple(
            parsed.get("requirement_links", ()),
            f"{path}.requirement_links",
        ),
        source=_parse_constraint_source(parsed.get("source", "manual"), f"{path}.source"),
    )
    domain_ids.validate_constraint_id(template.id)
    return template


def _load_registry_templates(
    registry_path: Path,
) -> tuple[dict[str, _ConstraintTemplate], list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    if not registry_path.exists():
        errors.append(f"Constraint registry path does not exist: {registry_path.as_posix()}.")
        return {}, warnings, errors
    if not registry_path.is_dir():
        errors.append(f"Constraint registry path is not a directory: {registry_path.as_posix()}.")
        return {}, warnings, errors

    yaml_files = sorted(
        path
        for path in registry_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not yaml_files:
        errors.append(f"Constraint registry has no YAML files: {registry_path.as_posix()}.")
        return {}, warnings, errors

    templates: dict[str, _ConstraintTemplate] = {}
    for file_path in yaml_files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle)
        except OSError as exc:
            errors.append(f"Failed to read constraint registry file {file_path.as_posix()}: {exc}.")
            continue
        except yaml.YAMLError as exc:
            errors.append(f"Invalid YAML in {file_path.as_posix()}: {exc}.")
            continue

        records: Sequence[object]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping):
            constraints_raw = payload.get("constraints")
            if isinstance(constraints_raw, Sequence) and not isinstance(
                constraints_raw, (str, bytes, bytearray)
            ):
                records = constraints_raw
            else:
                errors.append(
                    f"{file_path.as_posix()} must be a list or contain a 'constraints' list."
                )
                continue
        else:
            errors.append(f"{file_path.as_posix()} must be a list or contain a 'constraints' list.")
            continue

        for index, record in enumerate(records):
            entry_path = f"{file_path.as_posix()}[{index}]"
            if not isinstance(record, Mapping):
                errors.append(f"{entry_path} must be an object.")
                continue
            try:
                template = _coerce_constraint_template(record, entry_path)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if template.id in templates:
                errors.append(f"Duplicate constraint ID in registry: {template.id}.")
                continue
            templates[template.id] = template

    if not templates and not errors:
        warnings.append("No constraints were loaded from registry files.")
    return templates, warnings, errors


def _seed_byte(seed: int) -> int:
    return (seed % 251) + 1


def _randbytes(seed: int) -> Callable[[int], bytes]:
    byte = _seed_byte(seed)

    def provider(size: int) -> bytes:
        return bytes([byte]) * size

    return provider


def _run_seed(run_id: str) -> int:
    if run_id == _DEFAULT_RUN_ID:
        return 0
    digest = hashlib.blake2b(run_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _build_requirement_id_map(spec_map: SpecMap, errors: list[str]) -> dict[str, str]:
    if len(spec_map.requirements) > 9_999:
        errors.append("Spec has too many requirements to map into domain REQ-0001 IDs.")
        return {}
    return {
        requirement.id: f"REQ-{index:04d}"
        for index, requirement in enumerate(spec_map.requirements, start=1)
    }


def _merge_module_data(
    proposal: WorkItemProposal,
    module_by_name: Mapping[str, ModuleDefinition],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    module = module_by_name.get(proposal.module) if proposal.module is not None else None

    dependencies: list[str] = list(proposal.dependencies)
    requirement_links: list[str] = list(proposal.requirement_links)
    guarantees: list[str] = list(proposal.interface_guarantees)

    if module is not None:
        for dependency in module.dependencies:
            if dependency not in dependencies:
                dependencies.append(dependency)
        for requirement_id in module.requirement_links:
            if requirement_id not in requirement_links:
                requirement_links.append(requirement_id)
        for guarantee in module.interface_guarantees:
            if guarantee not in guarantees:
                guarantees.append(guarantee)

    return tuple(dependencies), tuple(requirement_links), tuple(guarantees)


def _resolve_dependencies(
    proposals_by_id: Mapping[str, WorkItemProposal],
    module_by_name: Mapping[str, ModuleDefinition],
    *,
    errors: list[str],
) -> tuple[
    dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], bool
]:
    module_to_work_items: dict[str, list[str]] = defaultdict(list)
    for proposal in proposals_by_id.values():
        if proposal.module is not None:
            module_to_work_items[proposal.module].append(proposal.id)
    for module_name in module_to_work_items:
        module_to_work_items[module_name].sort()

    resolved_dependencies: dict[str, tuple[str, ...]] = {}
    merged_requirement_links: dict[str, tuple[str, ...]] = {}
    merged_guarantees: dict[str, tuple[str, ...]] = {}
    fatal = False

    for proposal_id in sorted(proposals_by_id):
        proposal = proposals_by_id[proposal_id]
        merged_dependencies, req_links, guarantees = _merge_module_data(proposal, module_by_name)
        merged_requirement_links[proposal_id] = req_links
        merged_guarantees[proposal_id] = guarantees

        resolved: list[str] = []
        for dependency in merged_dependencies:
            resolved_dep: str | None = None
            if dependency in proposals_by_id:
                resolved_dep = dependency
            elif dependency in module_to_work_items:
                linked = module_to_work_items[dependency]
                if len(linked) == 1:
                    resolved_dep = linked[0]
                else:
                    errors.append(
                        "Dependency reference is ambiguous for "
                        f"{proposal_id}: module {dependency} maps to {linked}."
                    )
                    fatal = True
            elif dependency in module_by_name:
                errors.append(
                    f"Dependency reference for {proposal_id} points to module {dependency} "
                    "without a matching work item."
                )
                fatal = True
            else:
                errors.append(f"Unknown dependency for {proposal_id}: {dependency}.")
                fatal = True

            if resolved_dep is None:
                continue
            if resolved_dep == proposal_id:
                errors.append(f"Work item {proposal_id} cannot depend on itself.")
                fatal = True
                continue
            if resolved_dep not in resolved:
                resolved.append(resolved_dep)

        resolved_dependencies[proposal_id] = tuple(sorted(resolved))

    return resolved_dependencies, merged_requirement_links, merged_guarantees, fatal


def _topological_order(
    proposal_ids: Sequence[str],
    dependencies_by_item: Mapping[str, tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], bool]:
    indegree: dict[str, int] = {proposal_id: 0 for proposal_id in proposal_ids}
    adjacency: dict[str, list[str]] = {proposal_id: [] for proposal_id in proposal_ids}
    edges: set[tuple[str, str]] = set()

    for proposal_id in proposal_ids:
        for dependency in dependencies_by_item.get(proposal_id, ()):
            edges.add((dependency, proposal_id))
            adjacency[dependency].append(proposal_id)
            indegree[proposal_id] += 1

    for values in adjacency.values():
        values.sort()

    heap: list[str] = [proposal_id for proposal_id in proposal_ids if indegree[proposal_id] == 0]
    heapq.heapify(heap)

    ordered: list[str] = []
    while heap:
        current = heapq.heappop(heap)
        ordered.append(current)
        for child in adjacency[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)

    has_cycle = len(ordered) != len(proposal_ids)
    return tuple(ordered), tuple(sorted(edges)), has_cycle


def _build_constraint_index(
    templates: Mapping[str, _ConstraintTemplate],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    base_ids = tuple(
        sorted(template.id for template in templates.values() if not template.requirement_links)
    )
    by_requirement: dict[str, list[str]] = defaultdict(list)
    for template in templates.values():
        for requirement_id in template.requirement_links:
            by_requirement[requirement_id].append(template.id)
    return base_ids, {
        requirement_id: tuple(sorted(constraint_ids))
        for requirement_id, constraint_ids in by_requirement.items()
    }


def _map_requirement_links(
    requirement_links: Sequence[str],
    requirement_id_map: Mapping[str, str],
) -> tuple[str, ...]:
    mapped: list[str] = []
    seen: set[str] = set()
    for requirement_id in requirement_links:
        mapped_id = requirement_id_map.get(requirement_id)
        if mapped_id is None or mapped_id in seen:
            continue
        seen.add(mapped_id)
        mapped.append(mapped_id)
    return tuple(mapped)


def _materialize_constraint(
    template: _ConstraintTemplate,
    *,
    requirement_id_map: Mapping[str, str],
) -> Constraint:
    return Constraint(
        id=template.id,
        severity=template.severity,
        category=template.category,
        description=template.description,
        checker_binding=template.checker_binding,
        parameters=dict(template.parameters),
        requirement_links=_map_requirement_links(template.requirement_links, requirement_id_map),
        source=template.source,
        created_at=_COMPILED_AT,
    )


def _check_must_checker_bindings(
    *,
    proposal_id: str,
    constraints: Sequence[Constraint],
    checker_ids: set[str],
    errors: list[str],
) -> None:
    for constraint in constraints:
        if constraint.severity is not ConstraintSeverity.MUST:
            continue
        if constraint.checker_binding not in checker_ids:
            errors.append(
                "Missing checker binding for must constraint "
                f"{constraint.id} on work item {proposal_id}: "
                f"{constraint.checker_binding}."
            )


def _build_critical_path(
    ordered_ids: Sequence[str],
    dependencies_by_item: Mapping[str, tuple[str, ...]],
    *,
    domain_id_by_proposal_id: Mapping[str, str],
) -> tuple[str, ...]:
    if not ordered_ids:
        return ()

    distance: dict[str, int] = {proposal_id: 1 for proposal_id in ordered_ids}
    predecessor: dict[str, str | None] = {proposal_id: None for proposal_id in ordered_ids}

    children_by_node: dict[str, list[str]] = {proposal_id: [] for proposal_id in ordered_ids}
    for proposal_id in ordered_ids:
        for dependency in dependencies_by_item.get(proposal_id, ()):
            children_by_node[dependency].append(proposal_id)
    for children in children_by_node.values():
        children.sort()

    for node in ordered_ids:
        for child in children_by_node[node]:
            candidate = distance[node] + 1
            if candidate > distance[child]:
                distance[child] = candidate
                predecessor[child] = node
            elif candidate == distance[child]:
                current_prev = predecessor[child]
                if current_prev is None or node < current_prev:
                    predecessor[child] = node

    end_node = min(
        ordered_ids,
        key=lambda proposal_id: (-distance[proposal_id], proposal_id),
    )
    path: list[str] = []
    cursor: str | None = end_node
    while cursor is not None:
        path.append(domain_id_by_proposal_id[cursor])
        cursor = predecessor[cursor]
    path.reverse()
    return tuple(path)


def compile_constraints(
    spec_map: SpecMap,
    architect_output: ArchitectOutput,
    *,
    registry_path: str | Path = _DEFAULT_REGISTRY_PATH,
    run_id: str = _DEFAULT_RUN_ID,
) -> CompilationResult:
    """
    Deterministically compile work-item constraints and task graph.

    The result always includes collected warnings/errors; graph/work-item payloads
    are returned only when compilation can produce a valid DAG.
    """

    warnings: list[str] = []
    errors: list[str] = []
    fatal = False

    try:
        domain_ids.validate_run_id(run_id)
        normalized_run_id = run_id
    except ValueError:
        errors.append(f"Invalid run_id {run_id!r}; using default deterministic run id.")
        normalized_run_id = _DEFAULT_RUN_ID

    errors.extend(validate_architect_output(architect_output, spec_map))

    registry_templates, registry_warnings, registry_errors = _load_registry_templates(
        Path(registry_path)
    )
    warnings.extend(registry_warnings)
    errors.extend(registry_errors)

    for suggestion in sorted(architect_output.constraint_suggestions, key=lambda item: item.id):
        if suggestion.id in registry_templates:
            errors.append(f"Duplicate constraint ID from architect suggestion: {suggestion.id}.")
            continue
        try:
            domain_ids.validate_constraint_id(suggestion.id)
        except ValueError as exc:
            errors.append(f"Invalid architect constraint ID {suggestion.id}: {exc}.")
            continue
        registry_templates[suggestion.id] = _ConstraintTemplate(
            id=suggestion.id,
            severity=_parse_constraint_severity(suggestion.severity, suggestion.id),
            category=suggestion.category,
            description=suggestion.description,
            checker_binding=suggestion.checker_binding,
            parameters=_as_json_object(suggestion.parameters, f"{suggestion.id}.parameters"),
            requirement_links=tuple(suggestion.requirement_links),
            source=_parse_constraint_source(suggestion.source, suggestion.id),
        )

    proposals_by_id: dict[str, WorkItemProposal] = {}
    for proposal in sorted(architect_output.work_items, key=lambda item: item.id):
        proposals_by_id.setdefault(proposal.id, proposal)

    if not proposals_by_id:
        errors.append("Architect output produced no work items.")
        return CompilationResult(
            task_graph=None,
            work_items=(),
            warnings=tuple(sorted(set(warnings))),
            errors=tuple(sorted(set(errors))),
        )

    module_by_name: dict[str, ModuleDefinition] = {}
    for module in sorted(architect_output.modules, key=lambda item: item.name):
        module_by_name.setdefault(module.name, module)

    resolved_dependencies, merged_requirement_links, merged_guarantees, dep_fatal = (
        _resolve_dependencies(
            proposals_by_id,
            module_by_name,
            errors=errors,
        )
    )
    fatal = fatal or dep_fatal

    proposal_ids = tuple(sorted(proposals_by_id))
    topo_order, proposal_edges, has_cycle = _topological_order(
        proposal_ids,
        resolved_dependencies,
    )
    if has_cycle:
        errors.append(
            f"Dependency cycle detected in work-item graph: {', '.join(sorted(proposal_ids))}."
        )
        fatal = True

    requirement_ids = {requirement.id for requirement in spec_map.requirements}
    covered_requirements: set[str] = set()
    for requirement_links in merged_requirement_links.values():
        covered_requirements.update(requirement_links)
    missing_coverage = sorted(requirement_ids - covered_requirements)
    if missing_coverage:
        errors.append(
            f"Missing requirement coverage in compiled plan: {', '.join(missing_coverage)}."
        )

    scope_owners: dict[str, str] = {}
    for proposal in proposals_by_id.values():
        for path in proposal.owned_paths:
            owner = scope_owners.get(path)
            if owner is not None and owner != proposal.id:
                errors.append(
                    f"Duplicate file ownership detected: {path} owned by both {owner} and {proposal.id}."
                )
            else:
                scope_owners[path] = proposal.id

    requirement_id_map = _build_requirement_id_map(spec_map, errors)
    checker_ids = set(_KNOWN_CHECKER_BINDINGS)
    base_ids, ids_by_requirement = _build_constraint_index(registry_templates)

    domain_id_by_proposal_id: dict[str, str] = {}
    run_seed = _run_seed(normalized_run_id)
    timestamp_offset = run_seed % 10_000
    byte_seed_offset = run_seed % 10_000
    for index, proposal_id in enumerate(proposal_ids, start=1):
        work_item_id = domain_ids.generate_work_item_id(
            timestamp_ms=1_800_000_000_000 + timestamp_offset + index,
            randbytes=_randbytes(index + byte_seed_offset),
        )
        domain_id_by_proposal_id[proposal_id] = work_item_id

    if not topo_order:
        topo_order = proposal_ids

    generated_behavioral_ids: dict[tuple[str, str], str] = {}
    used_constraint_ids: set[str] = set(registry_templates)
    next_behavioral_index = 1

    proposal_constraints: dict[str, tuple[Constraint, ...]] = {}
    compiled_work_items: list[WorkItem] = []

    for proposal_id in topo_order:
        proposal = proposals_by_id[proposal_id]
        selected_constraint_ids: set[str] = set(base_ids)
        for requirement_id in merged_requirement_links.get(proposal_id, ()):
            for constraint_id in ids_by_requirement.get(requirement_id, ()):
                selected_constraint_ids.add(constraint_id)
        for constraint_id in proposal.constraint_ids:
            if constraint_id not in registry_templates:
                errors.append(
                    f"Work item {proposal_id} references unknown constraint {constraint_id}."
                )
                continue
            selected_constraint_ids.add(constraint_id)

        constraints_by_id: dict[str, Constraint] = {}
        for constraint_id in sorted(selected_constraint_ids):
            template = registry_templates.get(constraint_id)
            if template is None:
                errors.append(f"Constraint template {constraint_id} is missing.")
                continue
            try:
                constraints_by_id[constraint_id] = _materialize_constraint(
                    template,
                    requirement_id_map=requirement_id_map,
                )
            except ValueError as exc:
                errors.append(f"Failed to materialize constraint {constraint_id}: {exc}.")
                fatal = True

        inherited_ids: set[str] = set()
        for dependency_id in resolved_dependencies.get(proposal_id, ()):
            for dependency_constraint in proposal_constraints.get(dependency_id, ()):
                if (
                    dependency_constraint.severity is ConstraintSeverity.MUST
                    and dependency_constraint.category in _PROPAGATED_CATEGORIES
                ):
                    constraints_by_id.setdefault(dependency_constraint.id, dependency_constraint)
                    inherited_ids.add(dependency_constraint.id)

            for guarantee in merged_guarantees.get(dependency_id, ()):
                key = (dependency_id, guarantee)
                behavior_id = generated_behavioral_ids.get(key)
                if behavior_id is None:
                    while True:
                        candidate_id = f"CON-BHV-{next_behavioral_index:04d}"
                        next_behavioral_index += 1
                        if candidate_id in used_constraint_ids:
                            continue
                        used_constraint_ids.add(candidate_id)
                        behavior_id = candidate_id
                        break
                    generated_behavioral_ids[key] = behavior_id

                if behavior_id in constraints_by_id:
                    inherited_ids.add(behavior_id)
                    continue

                behavior_constraint = Constraint(
                    id=behavior_id,
                    severity=ConstraintSeverity.MUST,
                    category="behavioral",
                    description=f"Dependency {dependency_id} guarantee: {guarantee}",
                    checker_binding="test_checker",
                    parameters={"dependency": dependency_id, "guarantee": guarantee},
                    requirement_links=_map_requirement_links(
                        merged_requirement_links.get(dependency_id, ()),
                        requirement_id_map,
                    ),
                    source=ConstraintSource.SPEC_DERIVED,
                    created_at=_COMPILED_AT,
                )
                constraints_by_id[behavior_id] = behavior_constraint
                inherited_ids.add(behavior_id)

        ordered_constraints = tuple(
            constraints_by_id[constraint_id] for constraint_id in sorted(constraints_by_id)
        )
        if not ordered_constraints:
            errors.append(f"Work item {proposal_id} has an empty constraint envelope.")
            fatal = True
            continue

        _check_must_checker_bindings(
            proposal_id=proposal_id,
            constraints=ordered_constraints,
            checker_ids=checker_ids,
            errors=errors,
        )

        work_item_id = domain_id_by_proposal_id[proposal_id]
        try:
            envelope = ConstraintEnvelope(
                work_item_id=work_item_id,
                constraints=ordered_constraints,
                inherited_constraint_ids=tuple(sorted(inherited_ids)),
                compiled_at=_COMPILED_AT,
            )
            work_item = WorkItem(
                id=work_item_id,
                title=proposal.title,
                description=proposal.description,
                scope=proposal.owned_paths,
                constraint_envelope=envelope,
                dependencies=tuple(
                    domain_id_by_proposal_id[dependency]
                    for dependency in resolved_dependencies.get(proposal_id, ())
                ),
                requirement_links=_map_requirement_links(
                    merged_requirement_links.get(proposal_id, ()),
                    requirement_id_map,
                ),
                constraint_ids=tuple(constraint.id for constraint in ordered_constraints),
                created_at=_COMPILED_AT,
                updated_at=_COMPILED_AT,
            )
        except ValueError as exc:
            errors.append(f"Failed to build WorkItem for {proposal_id}: {exc}.")
            fatal = True
            continue

        compiled_work_items.append(work_item)
        proposal_constraints[proposal_id] = ordered_constraints

    if fatal:
        return CompilationResult(
            task_graph=None,
            work_items=tuple(compiled_work_items),
            warnings=tuple(sorted(set(warnings))),
            errors=tuple(sorted(set(errors))),
        )

    domain_edges = tuple(
        (
            domain_id_by_proposal_id[source],
            domain_id_by_proposal_id[target],
        )
        for source, target in proposal_edges
    )
    work_items_by_id = {work_item.id: work_item for work_item in compiled_work_items}
    ordered_domain_items = tuple(
        work_items_by_id[domain_id_by_proposal_id[proposal_id]]
        for proposal_id in topo_order
        if domain_id_by_proposal_id[proposal_id] in work_items_by_id
    )
    critical_path = _build_critical_path(
        topo_order,
        resolved_dependencies,
        domain_id_by_proposal_id=domain_id_by_proposal_id,
    )

    try:
        task_graph = TaskGraph(
            run_id=normalized_run_id,
            work_items=ordered_domain_items,
            edges=domain_edges,
            critical_path=critical_path,
            created_at=_COMPILED_AT,
        )
    except ValueError as exc:
        errors.append(f"Failed to build TaskGraph: {exc}.")
        task_graph = None

    return CompilationResult(
        task_graph=task_graph,
        work_items=ordered_domain_items,
        warnings=tuple(sorted(set(warnings))),
        errors=tuple(sorted(set(errors))),
    )


__all__ = ["CompilationResult", "compile_constraints"]
