"""Unit tests for spec_ingestion.spec_map helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from nexus_orchestrator.spec_ingestion import spec_map as spec_map_module
from nexus_orchestrator.spec_ingestion.spec_map import (
    Entity,
    InterfaceContract,
    Requirement,
    SourceLocation,
    SpecMap,
    deserialize_spec_map,
    diff_spec_maps,
    serialize_spec_map,
    trace_requirement_to_work_items,
)


def _source(path: str, section: str, line: int) -> SourceLocation:
    return SourceLocation(path=path, section=section, line=line)


def _spec_map_fixture() -> SpecMap:
    return SpecMap(
        version=1,
        source_documents=("b.md", "a.md"),
        requirements=(
            Requirement(
                id="REQ-SAMPLE-0002",
                statement="Second requirement",
                acceptance_criteria=("criterion b",),
                nfr_tags=(),
                source=_source("b.md", "Requirements", 10),
            ),
            Requirement(
                id="REQ-SAMPLE-0001",
                statement="First requirement",
                acceptance_criteria=("criterion a",),
                nfr_tags=(),
                source=_source("a.md", "Requirements", 5),
            ),
        ),
        entities=(
            Entity(
                name="Order", description="Order aggregate", source=_source("a.md", "Entities", 20)
            ),
        ),
        interfaces=(
            InterfaceContract(
                module_name="B",
                summary="Depends on A",
                dependencies=("A",),
                requirement_links=("REQ-SAMPLE-0002",),
                exposed_symbols=("farewell",),
                source=_source("b.md", "Interfaces", 30),
            ),
            InterfaceContract(
                module_name="A",
                summary="No dependencies",
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0001",),
                exposed_symbols=("greet",),
                source=_source("a.md", "Interfaces", 25),
            ),
        ),
        glossary={"ULID": "Sortable identifier"},
        global_acceptance_criteria=("global ac",),
    )


def test_serialize_deserialize_json_and_toml_are_deterministic() -> None:
    spec_map = _spec_map_fixture()

    json_a = serialize_spec_map(spec_map, format="json")
    json_b = serialize_spec_map(spec_map, format="json")
    assert json_a == json_b
    assert (
        hashlib.sha256(json_a.encode("utf-8")).hexdigest()
        == hashlib.sha256(json_b.encode("utf-8")).hexdigest()
    )

    toml_a = serialize_spec_map(spec_map, format="toml")
    toml_b = serialize_spec_map(spec_map, format="toml")
    assert toml_a == toml_b

    parsed_from_json = deserialize_spec_map(json_a, format="json")
    parsed_from_toml = deserialize_spec_map(toml_a, format="toml")
    assert parsed_from_json == spec_map
    assert parsed_from_toml == spec_map


def test_diff_spec_maps_reports_added_removed_and_modified_stably() -> None:
    old = SpecMap(
        version=1,
        source_documents=("spec.md",),
        requirements=(
            Requirement(
                id="REQ-SAMPLE-0001",
                statement="Old statement",
                acceptance_criteria=("criterion",),
                nfr_tags=(),
                source=_source("spec.md", "Requirements", 5),
            ),
        ),
        interfaces=(
            InterfaceContract(
                module_name="A",
                summary="No dependencies",
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0001",),
                exposed_symbols=("greet",),
                source=_source("spec.md", "Interfaces", 12),
            ),
        ),
    )
    new = SpecMap(
        version=1,
        source_documents=("spec.md",),
        requirements=(
            Requirement(
                id="REQ-SAMPLE-0001",
                statement="New statement",
                acceptance_criteria=("criterion", "criterion2"),
                nfr_tags=(),
                source=_source("spec.md", "Requirements", 5),
            ),
            Requirement(
                id="REQ-SAMPLE-0002",
                statement="Added statement",
                acceptance_criteria=(),
                nfr_tags=(),
                source=_source("spec.md", "Requirements", 6),
            ),
        ),
        interfaces=(
            InterfaceContract(
                module_name="B",
                summary="No dependencies",
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0002",),
                exposed_symbols=("farewell",),
                source=_source("spec.md", "Interfaces", 13),
            ),
        ),
        entities=(
            Entity(name="Order", description="Entity", source=_source("spec.md", "Entities", 20)),
        ),
    )

    diff = diff_spec_maps(old, new)
    assert diff.added_requirement_ids == ("REQ-SAMPLE-0002",)
    assert diff.removed_requirement_ids == ()
    assert tuple(item.requirement_id for item in diff.modified_requirements) == ("REQ-SAMPLE-0001",)
    assert diff.added_interface_modules == ("B",)
    assert diff.removed_interface_modules == ("A",)
    assert diff.modified_interfaces == ()
    assert diff.added_entities == ("Order",)
    assert diff.removed_entities == ()
    assert diff.modified_entities == ()

    diff_again = diff_spec_maps(old, new)
    assert diff == diff_again


@dataclass(slots=True)
class _WorkItemStub:
    id: str
    requirement_links: tuple[str, ...]


def test_trace_requirement_to_work_items_returns_sorted_ids() -> None:
    work_items = [
        _WorkItemStub(id="wi-3", requirement_links=("REQ-SAMPLE-0001",)),
        _WorkItemStub(id="wi-1", requirement_links=("REQ-SAMPLE-0002",)),
        _WorkItemStub(id="wi-2", requirement_links=("REQ-SAMPLE-0001", "REQ-SAMPLE-0002")),
    ]

    traced = trace_requirement_to_work_items("REQ-SAMPLE-0001", work_items)  # type: ignore[arg-type]
    assert traced == ["wi-2", "wi-3"]


def test_deserialize_rejects_invalid_root_type() -> None:
    with pytest.raises(ValueError, match="root must be an object"):
        deserialize_spec_map("[]", format="json")


def test_source_location_and_model_validation_errors() -> None:
    with pytest.raises(ValueError, match="SourceLocation.line"):
        SourceLocation(path="spec.md", section="Requirements", line=0)
    with pytest.raises(ValueError, match="SourceLocation.column"):
        SourceLocation(path="spec.md", section="Requirements", line=1, column=0)
    with pytest.raises(ValueError, match="SourceLocation.path must be a string"):
        SourceLocation.from_dict({"path": 1, "section": "Requirements", "line": 1})

    with pytest.raises(ValueError, match="Requirement.id must be a string"):
        Requirement.from_dict(
            {
                "id": 1,
                "statement": "statement",
                "acceptance_criteria": [],
                "nfr_tags": [],
                "source": {"path": "spec.md", "section": "Requirements", "line": 1},
            }
        )
    with pytest.raises(ValueError, match="Requirement.statement must be a string"):
        Requirement.from_dict(
            {
                "id": "REQ-SAMPLE-0001",
                "statement": 1,
                "acceptance_criteria": [],
                "nfr_tags": [],
                "source": {"path": "spec.md", "section": "Requirements", "line": 1},
            }
        )

    with pytest.raises(ValueError, match="Entity.name must be a string"):
        Entity.from_dict(
            {
                "name": 1,
                "description": "desc",
                "source": _source("spec.md", "Entities", 1).to_dict(),
            }
        )
    with pytest.raises(ValueError, match="Entity.description must be a string"):
        Entity.from_dict(
            {
                "name": "Order",
                "description": 1,
                "source": _source("spec.md", "Entities", 1).to_dict(),
            }
        )

    with pytest.raises(ValueError, match="InterfaceContract.module_name must be a string"):
        InterfaceContract.from_dict(
            {
                "module_name": 1,
                "summary": "summary",
                "dependencies": [],
                "requirement_links": [],
                "exposed_symbols": [],
                "source": _source("spec.md", "Interfaces", 1).to_dict(),
            }
        )
    with pytest.raises(ValueError, match="InterfaceContract.summary must be a string"):
        InterfaceContract.from_dict(
            {
                "module_name": "A",
                "summary": 1,
                "dependencies": [],
                "requirement_links": [],
                "exposed_symbols": [],
                "source": _source("spec.md", "Interfaces", 1).to_dict(),
            }
        )


def test_spec_map_constructor_validation_errors() -> None:
    req = Requirement(
        id="REQ-SAMPLE-0001",
        statement="statement",
        source=_source("spec.md", "Requirements", 1),
    )
    with pytest.raises(ValueError, match="version must be >= 1"):
        SpecMap(version=0, source_documents=("spec.md",), requirements=(req,))
    with pytest.raises(ValueError, match="source_documents"):
        SpecMap(version=1, source_documents=(), requirements=(req,))
    with pytest.raises(ValueError, match="must include at least one requirement"):
        SpecMap(version=1, source_documents=("spec.md",), requirements=())
    with pytest.raises(ValueError, match="duplicate requirement IDs"):
        SpecMap(version=1, source_documents=("spec.md",), requirements=(req, req))
    with pytest.raises(ValueError, match="duplicate entity names"):
        SpecMap(
            version=1,
            source_documents=("spec.md",),
            requirements=(req,),
            entities=(
                Entity(name="Order", description="d1", source=_source("spec.md", "Entities", 2)),
                Entity(name="Order", description="d2", source=_source("spec.md", "Entities", 3)),
            ),
        )
    with pytest.raises(ValueError, match="duplicate module names"):
        SpecMap(
            version=1,
            source_documents=("spec.md",),
            requirements=(req,),
            interfaces=(
                InterfaceContract(
                    module_name="A", summary="", source=_source("spec.md", "Interfaces", 3)
                ),
                InterfaceContract(
                    module_name="A", summary="", source=_source("spec.md", "Interfaces", 4)
                ),
            ),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        SpecMap(
            version=1,
            source_documents=("spec.md",),
            requirements=(req,),
            glossary={"term": "   "},
        )


def test_spec_map_from_dict_validation_errors() -> None:
    with pytest.raises(ValueError, match="version must be an integer"):
        SpecMap.from_dict({"version": "1", "source_documents": ["spec.md"], "requirements": []})
    with pytest.raises(ValueError, match="requirements must be a list"):
        SpecMap.from_dict({"version": 1, "source_documents": ["spec.md"], "requirements": {}})
    with pytest.raises(ValueError, match="entities must be a list"):
        SpecMap.from_dict(
            {
                "version": 1,
                "source_documents": ["spec.md"],
                "requirements": [
                    {
                        "id": "REQ-SAMPLE-0001",
                        "statement": "statement",
                        "source": _source("spec.md", "Requirements", 1).to_dict(),
                    }
                ],
                "entities": {},
            }
        )
    with pytest.raises(ValueError, match="interfaces must be a list"):
        SpecMap.from_dict(
            {
                "version": 1,
                "source_documents": ["spec.md"],
                "requirements": [
                    {
                        "id": "REQ-SAMPLE-0001",
                        "statement": "statement",
                        "source": _source("spec.md", "Requirements", 1).to_dict(),
                    }
                ],
                "interfaces": {},
            }
        )
    with pytest.raises(ValueError, match="glossary must be an object"):
        SpecMap.from_dict(
            {
                "version": 1,
                "source_documents": ["spec.md"],
                "requirements": [
                    {
                        "id": "REQ-SAMPLE-0001",
                        "statement": "statement",
                        "source": _source("spec.md", "Requirements", 1).to_dict(),
                    }
                ],
                "glossary": [],
            }
        )


def test_diff_is_empty_property_and_invalid_serializer_inputs() -> None:
    spec_map = _spec_map_fixture()
    diff = diff_spec_maps(spec_map, spec_map)
    assert diff.is_empty
    assert diff.to_dict()["added_requirement_ids"] == []

    with pytest.raises(ValueError, match="Unsupported format"):
        serialize_spec_map(spec_map, format="yaml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported format"):
        deserialize_spec_map("{}", format="yaml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="req_id must not be empty"):
        trace_requirement_to_work_items("   ", [])


def test_toml_value_helper_branches() -> None:
    assert spec_map_module._toml_value(True) == "true"
    assert spec_map_module._toml_value(1.5) == "1.5"
    assert spec_map_module._toml_value({"a": [1, "x"]}).startswith("{")
    with pytest.raises(ValueError, match="Unsupported TOML value type"):
        spec_map_module._toml_value(object())


def test_internal_helper_error_branches() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        spec_map_module._strip_text("   ", "field")

    assert spec_map_module._unique_preserve_order(["a", "", "a", "b"]) == ("a", "b")
    assert spec_map_module._as_string_sequence(("a", "b"), "field") == ("a", "b")
    with pytest.raises(ValueError, match="must be a list"):
        spec_map_module._as_string_sequence({}, "field")
    with pytest.raises(ValueError, match=r"\[0\] must be a string"):
        spec_map_module._as_string_sequence([1], "field")

    with pytest.raises(ValueError, match="must be an object"):
        spec_map_module._as_mapping([], "field")
    with pytest.raises(ValueError, match="must use string keys"):
        spec_map_module._as_mapping({1: "x"}, "field")

    source_with_column = SourceLocation(path="spec.md", section="Requirements", line=1, column=2)
    assert source_with_column.to_dict()["column"] == 2
    with pytest.raises(ValueError, match="SourceLocation.section must be a string"):
        SourceLocation.from_dict({"path": "spec.md", "section": 1, "line": 1})
    with pytest.raises(ValueError, match="SourceLocation.line must be an integer"):
        SourceLocation.from_dict({"path": "spec.md", "section": "Requirements", "line": "1"})
    with pytest.raises(ValueError, match="SourceLocation.column must be an integer"):
        SourceLocation.from_dict(
            {"path": "spec.md", "section": "Requirements", "line": 1, "column": "1"}
        )

    with pytest.raises(ValueError, match="keys must be strings"):
        SpecMap.from_dict(
            {
                "version": 1,
                "source_documents": ["spec.md"],
                "requirements": [
                    {
                        "id": "REQ-SAMPLE-0001",
                        "statement": "statement",
                        "source": _source("spec.md", "Requirements", 1).to_dict(),
                    }
                ],
                "glossary": {1: "value"},
            }
        )
    with pytest.raises(ValueError, match="must be a string"):
        SpecMap.from_dict(
            {
                "version": 1,
                "source_documents": ["spec.md"],
                "requirements": [
                    {
                        "id": "REQ-SAMPLE-0001",
                        "statement": "statement",
                        "source": _source("spec.md", "Requirements", 1).to_dict(),
                    }
                ],
                "glossary": {"term": 1},
            }
        )
