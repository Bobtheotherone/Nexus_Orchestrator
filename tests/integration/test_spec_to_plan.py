"""
Integration test: spec ingestion through planning.

Purpose:
- Verify that minimal_design_doc.md produces a valid SpecMap.
- Verify ingestion outputs before planning compilation is implemented.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
