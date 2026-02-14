"""
Integration test: spec ingestion through planning.

Purpose:
- Verify that minimal_design_doc.md produces a valid SpecMap.
- Verify deterministic architect decomposition and full constraint compilation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nexus_orchestrator.planning.architect_interface import build_deterministic_architect_output
from nexus_orchestrator.planning.constraint_compiler import compile_constraints
from nexus_orchestrator.spec_ingestion import ingest_spec


@pytest.mark.integration
def test_minimal_design_doc_ingestion_assertions() -> None:
    sample_spec = Path("samples/specs/minimal_design_doc.md")
    parsed = ingest_spec(sample_spec)

    req_by_id = {requirement.id: requirement for requirement in parsed.requirements}
    assert {"REQ-SAMPLE-0001", "REQ-SAMPLE-0002", "REQ-SAMPLE-0003"} <= set(req_by_id)

    expected_global_acceptance = (
        'Calling `conversation("World")` returns "Hello, World!\\nGoodbye, World!"',
        "All functions have unit tests with 100% branch coverage.",
    )
    for req_id in ("REQ-SAMPLE-0001", "REQ-SAMPLE-0002", "REQ-SAMPLE-0003"):
        assert req_by_id[req_id].acceptance_criteria == expected_global_acceptance
        assert req_by_id[req_id].source.path == "samples/specs/minimal_design_doc.md"
        assert req_by_id[req_id].source.section == "Requirements"

    assert req_by_id["NFR-SAMPLE-0001"].nfr_tags == ("nfr", "performance")
    assert req_by_id["NFR-SAMPLE-0002"].nfr_tags == ("nfr", "security")
    assert req_by_id["NFR-SAMPLE-0003"].nfr_tags == ("nfr", "reliability")
    assert req_by_id["NFR-SAMPLE-0001"].acceptance_criteria == ()

    interface_by_module = {interface.module_name: interface for interface in parsed.interfaces}
    assert set(interface_by_module) == {"A", "B", "C"}
    assert interface_by_module["A"].dependencies == ()
    assert interface_by_module["B"].dependencies == ()
    assert interface_by_module["C"].dependencies == ("A", "B")
    assert interface_by_module["A"].requirement_links == ("REQ-SAMPLE-0001",)
    assert interface_by_module["B"].requirement_links == ("REQ-SAMPLE-0002",)
    assert interface_by_module["C"].requirement_links == ("REQ-SAMPLE-0003",)


@pytest.mark.integration
def test_minimal_design_doc_ingestion_to_full_plan_compile() -> None:
    sample_spec = Path("samples/specs/minimal_design_doc.md")
    parsed = ingest_spec(sample_spec)
    architect_output = build_deterministic_architect_output(parsed)

    result = compile_constraints(parsed, architect_output)

    assert result.errors == ()
    assert result.task_graph is not None
    assert len(result.work_items) == 3

    titles = tuple(item.title for item in result.task_graph.work_items)
    assert titles == ("Implement module A", "Implement module B", "Implement module C")

    by_title = {item.title: item for item in result.work_items}
    a_item = by_title["Implement module A"]
    b_item = by_title["Implement module B"]
    c_item = by_title["Implement module C"]

    assert c_item.dependencies == (a_item.id, b_item.id)
    assert (a_item.id, c_item.id) in result.task_graph.edges
    assert (b_item.id, c_item.id) in result.task_graph.edges

    for work_item in result.work_items:
        constraint_ids = {constraint.id for constraint in work_item.constraint_envelope.constraints}
        assert constraint_ids
        assert "CON-SEC-0001" in constraint_ids
        assert "CON-COR-0001" in constraint_ids

    assert any(
        constraint.category == "behavioral" for constraint in c_item.constraint_envelope.constraints
    )
