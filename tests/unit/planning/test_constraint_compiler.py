"""
nexus-orchestrator — unit tests for planning constraint compiler

File: tests/unit/planning/test_constraint_compiler.py
Last updated: 2026-02-14

Purpose
- Validate deterministic constraint compilation and dependency propagation for planned work items.

What this test file should cover
- Deterministic compile output (IDs, ordering, and edges) for a minimal three-module graph.
- Inherited security/correctness/style constraints on dependent work items.
- Duplicate file ownership detection in architect output.

Functional requirements
- Run fully offline with in-memory fixtures only.

Non-functional requirements
- Deterministic assertions and stable ordering semantics.
"""

from __future__ import annotations

from nexus_orchestrator.planning.architect_interface import (
    ArchitectOutput,
    ConstraintSuggestion,
    ModuleDefinition,
    WorkItemProposal,
)
from nexus_orchestrator.planning.constraint_compiler import compile_constraints
from nexus_orchestrator.spec_ingestion.spec_map import (
    InterfaceContract,
    Requirement,
    SourceLocation,
    SpecMap,
)


def _loc(*, section: str, line: int) -> SourceLocation:
    return SourceLocation(
        path="samples/specs/minimal_design_doc.md",
        section=section,
        line=line,
    )


def _minimal_spec_map() -> SpecMap:
    return SpecMap(
        version=1,
        source_documents=("samples/specs/minimal_design_doc.md",),
        requirements=(
            Requirement(
                id="REQ-SAMPLE-0001",
                statement="Expose greet(name) -> str.",
                source=_loc(section="Requirements", line=1),
            ),
            Requirement(
                id="REQ-SAMPLE-0002",
                statement="Expose farewell(name) -> str.",
                source=_loc(section="Requirements", line=2),
            ),
            Requirement(
                id="REQ-SAMPLE-0003",
                statement="Expose conversation(name) -> str calling greet + farewell.",
                source=_loc(section="Requirements", line=3),
            ),
        ),
        interfaces=(
            InterfaceContract(
                module_name="A",
                summary="Provides greet(name).",
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0001",),
                exposed_symbols=("greet",),
                source=_loc(section="Interfaces", line=1),
            ),
            InterfaceContract(
                module_name="B",
                summary="Provides farewell(name).",
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0002",),
                exposed_symbols=("farewell",),
                source=_loc(section="Interfaces", line=2),
            ),
            InterfaceContract(
                module_name="C",
                summary="Provides conversation(name) using A and B.",
                dependencies=("A", "B"),
                requirement_links=("REQ-SAMPLE-0003",),
                exposed_symbols=("conversation",),
                source=_loc(section="Interfaces", line=3),
            ),
        ),
    )


def _architect_output(*, duplicate_scope: bool = False) -> ArchitectOutput:
    second_scope = "src/a.py" if duplicate_scope else "src/b.py"
    return ArchitectOutput(
        modules=(
            ModuleDefinition(
                name="A",
                summary="Module A",
                owned_paths=("src/a.py",),
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0001",),
                interface_contract_refs=("A",),
                interface_guarantees=("greet(name) returns greeting text",),
            ),
            ModuleDefinition(
                name="B",
                summary="Module B",
                owned_paths=("src/b.py",),
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0002",),
                interface_contract_refs=("B",),
                interface_guarantees=("farewell(name) returns farewell text",),
            ),
            ModuleDefinition(
                name="C",
                summary="Module C",
                owned_paths=("src/c.py",),
                dependencies=("A", "B"),
                requirement_links=("REQ-SAMPLE-0003",),
                interface_contract_refs=("C",),
                interface_guarantees=("conversation(name) composes greet/farewell",),
            ),
        ),
        work_items=(
            WorkItemProposal(
                id="A",
                title="Implement A",
                description="Implement module A",
                owned_paths=("src/a.py", "tests/unit/test_a.py"),
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0001",),
                module="A",
                interface_contract_refs=("A",),
                interface_guarantees=("greet(name) returns greeting text",),
            ),
            WorkItemProposal(
                id="B",
                title="Implement B",
                description="Implement module B",
                owned_paths=(second_scope, "tests/unit/test_b.py"),
                dependencies=(),
                requirement_links=("REQ-SAMPLE-0002",),
                module="B",
                interface_contract_refs=("B",),
                interface_guarantees=("farewell(name) returns farewell text",),
            ),
            WorkItemProposal(
                id="C",
                title="Implement C",
                description="Implement module C",
                owned_paths=("src/c.py", "tests/unit/test_c.py"),
                dependencies=("A", "B"),
                requirement_links=("REQ-SAMPLE-0003",),
                module="C",
                interface_contract_refs=("C",),
                interface_guarantees=("conversation(name) composes greet/farewell",),
            ),
        ),
        adrs=(),
        constraint_suggestions=(
            ConstraintSuggestion(
                id="CON-STY-0901",
                severity="must",
                category="style",
                description="Module A must follow API naming style checks.",
                checker_binding="lint_checker",
                parameters={"style_profile": "strict_api"},
                requirement_links=("REQ-SAMPLE-0001",),
                source="spec_derived",
            ),
        ),
    )


def test_compile_constraints_deterministic_three_item_graph_and_envelopes() -> None:
    spec_map = _minimal_spec_map()
    architect_output = _architect_output()

    result = compile_constraints(spec_map, architect_output)
    repeat = compile_constraints(spec_map, architect_output)

    assert result.errors == ()
    assert result.task_graph is not None
    assert len(result.work_items) == 3

    # Deterministic IDs/order/edges across repeated compiles.
    assert tuple(item.id for item in result.work_items) == tuple(
        item.id for item in repeat.work_items
    )
    assert result.task_graph.edges == repeat.task_graph.edges
    assert tuple(item.title for item in result.task_graph.work_items) == (
        "Implement A",
        "Implement B",
        "Implement C",
    )

    work_items_by_title = {item.title: item for item in result.work_items}
    c_item = work_items_by_title["Implement C"]
    assert len(c_item.dependencies) == 2

    c_constraint_ids = {constraint.id for constraint in c_item.constraint_envelope.constraints}
    assert "CON-SEC-0001" in c_constraint_ids
    assert "CON-COR-0001" in c_constraint_ids
    assert "CON-STY-0901" in c_constraint_ids

    # Interface guarantees become behavioral constraints on dependents.
    assert any(
        constraint.category == "behavioral" for constraint in c_item.constraint_envelope.constraints
    )
    # Style MUST constraint from A propagates to dependent C as inherited.
    assert "CON-STY-0901" in c_item.constraint_envelope.inherited_constraint_ids


def test_compile_constraints_reports_duplicate_file_ownership() -> None:
    spec_map = _minimal_spec_map()
    architect_output = _architect_output(duplicate_scope=True)

    result = compile_constraints(spec_map, architect_output)

    assert any("Duplicate file ownership" in error for error in result.errors)
