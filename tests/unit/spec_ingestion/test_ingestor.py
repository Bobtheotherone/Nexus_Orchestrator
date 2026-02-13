"""Unit tests for markdown spec ingestion."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nexus_orchestrator.spec_ingestion import SpecIngestError, ingest_spec, serialize_spec_map
from nexus_orchestrator.spec_ingestion import ingestor as ingestor_module
from nexus_orchestrator.spec_ingestion.spec_map import Requirement, SourceLocation, SpecMap


def _write_spec(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _find_requirement(spec_map: object, req_id: str) -> object:
    for requirement in spec_map.requirements:  # type: ignore[attr-defined]
        if requirement.id == req_id:  # type: ignore[attr-defined]
            return requirement
    raise AssertionError(f"Requirement not found: {req_id}")


@pytest.mark.unit
def test_ingest_ignores_backtick_and_tilde_code_fences(tmp_path: Path) -> None:
    spec = """
# Example

## Requirements
- REQ-REAL-0001: Outside fence requirement.
   ````md
- REQ-FAKE-0001: This must be ignored.
   ````
~~~python
- REQ-FAKE-0002: This must also be ignored.
~~~
- REQ-REAL-0002: Another outside requirement.

## Interfaces
- Module A (`greet`): implements REQ-REAL-0001. No dependencies.
- Module B (`farewell`): implements REQ-REAL-0002. No dependencies.
"""
    path = _write_spec(tmp_path, "fence.md", spec)
    parsed = ingest_spec(path)
    extracted_ids = [requirement.id for requirement in parsed.requirements]
    assert extracted_ids == ["REQ-REAL-0001", "REQ-REAL-0002"]
    assert "REQ-FAKE-0001" not in extracted_ids
    assert "REQ-FAKE-0002" not in extracted_ids


@pytest.mark.unit
def test_unclosed_fence_extends_to_eof(tmp_path: Path) -> None:
    spec = """
# Example

## Requirements
- REQ-REAL-0001: Survives because it appears before fence.
```md
- REQ-FAKE-0002: Ignored inside unclosed fence.
- REQ-FAKE-0003: Also ignored because unclosed fence extends to EOF.
"""
    path = _write_spec(tmp_path, "unclosed.md", spec)
    parsed = ingest_spec(path)
    assert [requirement.id for requirement in parsed.requirements] == ["REQ-REAL-0001"]


@pytest.mark.unit
def test_html_comments_are_non_semantic(tmp_path: Path) -> None:
    spec = """
# Example

## Requirements
<!-- - Missing ID should not be parsed: This line is ignored -->
- REQ-REAL-0001: Visible requirement.
<!--
- REQ-FAKE-0002: Also ignored.
-->

## Interfaces
- Module A: implements REQ-REAL-0001. No dependencies.
"""
    path = _write_spec(tmp_path, "comments.md", spec)
    parsed = ingest_spec(path)
    assert [requirement.id for requirement in parsed.requirements] == ["REQ-REAL-0001"]


@pytest.mark.unit
def test_messy_markdown_headings_and_bullets_are_supported(tmp_path: Path) -> None:
    spec = """
# Example

Requirements
------------
 * REQ-MESSY-0001: First requirement.
 + REQ-MESSY-0002: Second requirement.
 1. REQ-MESSY-0003: Third requirement.

###   INTERFACES
- Module A (`greet`): implements REQ-MESSY-0001. No dependencies.
- Module B (`farewell`): implements REQ-MESSY-0002. No dependencies.
- Module C (`conversation`): implements REQ-MESSY-0003. Depends on A and B.
"""
    path = _write_spec(tmp_path, "messy.md", spec)
    parsed = ingest_spec(path)

    assert [requirement.id for requirement in parsed.requirements] == [
        "REQ-MESSY-0001",
        "REQ-MESSY-0002",
        "REQ-MESSY-0003",
    ]
    modules = {interface.module_name: interface for interface in parsed.interfaces}
    assert tuple(modules["A"].dependencies) == ()
    assert tuple(modules["B"].dependencies) == ()
    assert tuple(modules["C"].dependencies) == ("A", "B")


@pytest.mark.unit
def test_missing_requirement_id_raises_actionable_error(tmp_path: Path) -> None:
    spec = """
# Example

## Requirements
- This line looks like a requirement but has no ID prefix.
"""
    path = _write_spec(tmp_path, "missing_id.md", spec)
    with pytest.raises(SpecIngestError) as exc_info:
        ingest_spec(path)
    err = exc_info.value
    assert err.path == path.resolve()
    assert err.line == 5
    assert err.section == "Requirements"
    assert "Expected 'REQ-...:'" in err.hint


@pytest.mark.unit
def test_duplicate_requirement_id_conflict_is_rejected(tmp_path: Path) -> None:
    base = _write_spec(
        tmp_path,
        "base.md",
        """
# Example
## Requirements
- REQ-DUP-0001: Original statement.
""",
    )
    override = _write_spec(
        tmp_path,
        "override.md",
        """
# Example
## Requirements
- REQ-DUP-0001: Conflicting statement.
""",
    )

    with pytest.raises(SpecIngestError, match="Requirement conflict for ID REQ-DUP-0001"):
        ingest_spec(base, additional_sources=[override])


@pytest.mark.unit
def test_unknown_interface_dependency_raises_actionable_error(tmp_path: Path) -> None:
    spec = """
# Example

## Requirements
- REQ-INT-0001: A requirement.

## Interfaces
- Module A: implements REQ-INT-0001. No dependencies.
- Module C: implements REQ-INT-0001. Depends on A and Z.
"""
    path = _write_spec(tmp_path, "unknown_dep.md", spec)
    with pytest.raises(SpecIngestError) as exc_info:
        ingest_spec(path)
    err = exc_info.value
    assert err.section == "Interfaces"
    assert "unknown module 'Z'" in err.message
    assert "Declare the dependency module" in err.hint


@pytest.mark.unit
def test_merge_policy_allows_acceptance_additions_but_not_statement_changes(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "primary.md",
        """
# Spec
## Requirements
- REQ-MERGE-0001: Deterministic statement.
  - local criterion one

## Acceptance Criteria
- global criterion
""",
    )
    supplemental = _write_spec(
        tmp_path,
        "supplemental.md",
        """
# Spec
## Requirements
- REQ-MERGE-0001: Deterministic statement.
  - local criterion two
""",
    )

    parsed = ingest_spec(primary, additional_sources=[supplemental])
    req = _find_requirement(parsed, "REQ-MERGE-0001")
    assert req.acceptance_criteria == (  # type: ignore[attr-defined]
        "local criterion one",
        "local criterion two",
        "global criterion",
    )

    conflicting = _write_spec(
        tmp_path,
        "conflicting.md",
        """
# Spec
## Requirements
- REQ-MERGE-0001: Different statement.
""",
    )
    with pytest.raises(SpecIngestError, match="Requirement conflict for ID REQ-MERGE-0001"):
        ingest_spec(primary, additional_sources=[conflicting])


@pytest.mark.unit
def test_ingestion_is_deterministic_for_repeated_runs(tmp_path: Path) -> None:
    spec = """
# Determinism

## Requirements
- REQ-DET-0001: Deterministic requirement one.
- REQ-DET-0002: Deterministic requirement two.

## Interfaces
- Module A: implements REQ-DET-0001. No dependencies.
- Module B: implements REQ-DET-0002. Depends on A.

## Acceptance Criteria
- deterministic criterion
"""
    path = _write_spec(tmp_path, "determinism.md", spec)

    first = ingest_spec(path)
    second = ingest_spec(path)
    first_json = serialize_spec_map(first, format="json")
    second_json = serialize_spec_map(second, format="json")

    assert first_json == second_json
    assert (
        hashlib.sha256(first_json.encode("utf-8")).hexdigest()
        == hashlib.sha256(second_json.encode("utf-8")).hexdigest()
    )


_noise_line = st.text(
    alphabet=" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_=+.,",
    min_size=0,
    max_size=25,
)


@pytest.mark.unit
@settings(max_examples=50, deadline=None)
@given(
    ids=st.lists(st.integers(min_value=1, max_value=9999), min_size=1, max_size=6, unique=True),
    noise=st.lists(_noise_line, min_size=0, max_size=8),
)
def test_fuzz_noise_markdown_extracts_only_declared_requirements(
    ids: list[int],
    noise: list[str],
) -> None:
    lines: list[str] = ["# Fuzz", "## Notes", *noise, "", "## Requirements"]
    expected_ids: list[str] = []
    for index, number in enumerate(ids):
        marker = f"{index + 1}." if index % 4 == 3 else ("-", "*", "+")[index % 3]
        req_id = f"REQ-FZZ-{number:04d}"
        expected_ids.append(req_id)
        lines.append(f"{marker} {req_id}: Fuzz requirement {index}.")
        lines.append("  - deterministic criterion")

    lines.extend(
        [
            "",
            "```txt",
            "- REQ-FAKE-9998: Must never be ingested.",
            "```",
            "~~~txt",
            "- REQ-FAKE-9999: Must never be ingested.",
            "~~~",
        ]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        path = _write_spec(Path(temp_dir), "fuzz.md", "\n".join(lines) + "\n")
        parsed = ingest_spec(path)
    extracted_ids = [requirement.id for requirement in parsed.requirements]
    assert extracted_ids == sorted(expected_ids)
    assert all(req_id.startswith("REQ-FZZ-") for req_id in extracted_ids)


@pytest.mark.unit
def test_ingest_reports_io_errors_for_missing_sources(tmp_path: Path) -> None:
    missing_primary = tmp_path / "missing.md"
    with pytest.raises(SpecIngestError, match="Spec file does not exist"):
        ingest_spec(missing_primary)

    primary = _write_spec(
        tmp_path,
        "primary.md",
        """
# Spec
## Requirements
- REQ-IO-0001: Requirement.
""",
    )
    missing_additional = tmp_path / "missing_additional.md"
    with pytest.raises(SpecIngestError, match="Additional source does not exist"):
        ingest_spec(primary, additional_sources=[missing_additional])


@pytest.mark.unit
def test_ingest_wraps_domain_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = _write_spec(
        tmp_path,
        "primary.md",
        """
# Spec
## Requirements
- REQ-VAL-0001: Requirement.
""",
    )

    def _boom(spec_map: SpecMap) -> None:
        del spec_map
        raise ValueError("forced domain failure")

    monkeypatch.setattr(ingestor_module, "_validate_against_domain_spec_map", _boom)
    with pytest.raises(SpecIngestError, match="Failed domain-model compatibility validation"):
        ingest_spec(primary)


@pytest.mark.unit
def test_parse_source_rejects_conflicting_duplicate_ids_in_same_file(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "dup.md",
        """
# Spec
## Requirements
- REQ-DUP-0001: Statement A.
- REQ-DUP-0001: Statement B.
""",
    )
    with pytest.raises(
        SpecIngestError, match="Duplicate requirement ID with conflicting statements"
    ):
        ingest_spec(primary)


@pytest.mark.unit
def test_interfaces_nested_bullets_are_ignored(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "interfaces_nested.md",
        """
# Spec
## Requirements
- REQ-NEST-0001: Requirement.

## Interfaces
- Module A: implements REQ-NEST-0001. No dependencies.
    - nested detail that should be ignored
""",
    )
    parsed = ingest_spec(primary)
    assert [interface.module_name for interface in parsed.interfaces] == ["A"]


@pytest.mark.unit
def test_interface_entry_must_match_expected_pattern(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "interfaces_bad.md",
        """
# Spec
## Requirements
- REQ-BAD-0001: Requirement.

## Interfaces
- this is not a module definition
""",
    )
    with pytest.raises(SpecIngestError, match="Interface bullet is ambiguous"):
        ingest_spec(primary)


@pytest.mark.unit
def test_interface_requirement_links_must_reference_known_requirements(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "interfaces_unknown_req.md",
        """
# Spec
## Requirements
- REQ-IF-0001: Requirement.

## Interfaces
- Module A: implements REQ-UNKNOWN-9999. No dependencies.
""",
    )
    with pytest.raises(SpecIngestError, match="references unknown requirement"):
        ingest_spec(primary)


@pytest.mark.unit
def test_entity_and_glossary_conflicts_raise_actionable_errors(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "entity_primary.md",
        """
# Spec
## Requirements
- REQ-ENT-0001: Requirement.

## Entities
- User: first definition
""",
    )
    entity_conflict = _write_spec(
        tmp_path,
        "entity_conflict.md",
        """
# Spec
## Requirements
- REQ-ENT-0001: Requirement.

## Entities
- User: conflicting definition
""",
    )
    with pytest.raises(SpecIngestError, match="Entity conflict"):
        ingest_spec(primary, additional_sources=[entity_conflict])

    req_source = SourceLocation(path="spec.md", section="Requirements", line=2)
    entity_source = SourceLocation(path="spec.md", section="Entities", line=4)
    parsed_a = ingestor_module._ParsedSource(
        source_document="spec.md",
        requirements={
            "REQ-GLO-0001": ingestor_module._RequirementDraft(
                req_id="REQ-GLO-0001",
                statement="Requirement.",
                acceptance_criteria=[],
                nfr_tags=[],
                source=req_source,
            )
        },
        entities={
            "Term": ingestor_module.Entity(name="Term", description="same", source=entity_source)
        },
        glossary={"Term": "first meaning"},
    )
    parsed_b = ingestor_module._ParsedSource(
        source_document="spec2.md",
        requirements={
            "REQ-GLO-0001": ingestor_module._RequirementDraft(
                req_id="REQ-GLO-0001",
                statement="Requirement.",
                acceptance_criteria=[],
                nfr_tags=[],
                source=req_source,
            )
        },
        entities={
            "Term": ingestor_module.Entity(name="Term", description="same", source=entity_source)
        },
        glossary={"Term": "second meaning"},
    )
    with pytest.raises(SpecIngestError, match="Glossary conflict"):
        ingestor_module._merge_sources([parsed_a, parsed_b])


@pytest.mark.unit
def test_entities_support_hyphen_definitions_and_missing_name_errors(tmp_path: Path) -> None:
    hyphen_entities = _write_spec(
        tmp_path,
        "entities_hyphen.md",
        """
# Spec
## Requirements
- REQ-ENT-0001: Requirement.

## Domain Entities
- Account - primary account aggregate
""",
    )
    parsed = ingest_spec(hyphen_entities)
    assert parsed.entities[0].name == "Account"

    bad_entities = _write_spec(
        tmp_path,
        "entities_bad.md",
        """
# Spec
## Requirements
- REQ-ENT-0001: Requirement.

## Entities
- : invalid
""",
    )
    with pytest.raises(SpecIngestError, match="Entity bullet is missing a name"):
        ingest_spec(bad_entities)


@pytest.mark.unit
def test_duplicate_additional_sources_are_deduplicated(tmp_path: Path) -> None:
    primary = _write_spec(
        tmp_path,
        "primary.md",
        """
# Spec
## Requirements
- REQ-ADD-0001: Requirement.
""",
    )
    extra = _write_spec(
        tmp_path,
        "extra.md",
        """
# Spec
## Requirements
- REQ-ADD-0002: Requirement.
""",
    )
    parsed = ingest_spec(primary, additional_sources=[extra, extra])
    assert len(parsed.source_documents) == 2
    assert any(path.endswith("extra.md") for path in parsed.source_documents)


@pytest.mark.unit
def test_parse_source_wraps_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "any.md"
    file_path.write_text("", encoding="utf-8")

    def _raise_os_error(self: Path, encoding: str = "utf-8") -> str:
        del self, encoding
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", _raise_os_error)
    with pytest.raises(SpecIngestError, match="Failed to read spec source"):
        ingestor_module._parse_source(file_path)


@pytest.mark.unit
def test_private_helper_branches() -> None:
    values = ["A"]
    ingestor_module._append_unique(values, "A")
    ingestor_module._append_unique(values, "   ")
    assert values == ["A"]
    assert ingestor_module._split_reference_list("") == []
    assert ingestor_module._split_reference_list("A, , module B and `C`") == ["A", "B", "C"]

    with pytest.raises(ValueError, match="No parsed sources to merge"):
        ingestor_module._merge_sources([])


@pytest.mark.unit
def test_private_interface_merge_conflict_and_fill_branches() -> None:
    source = SourceLocation(path="spec.md", section="Interfaces", line=1)
    existing = ingestor_module._InterfaceDraft(
        module_name="A",
        summary="",
        dependencies=["X"],
        requirement_links=["REQ-A-0001"],
        exposed_symbols=["foo"],
        source=source,
    )
    incoming = ingestor_module._InterfaceDraft(
        module_name="A",
        summary="filled summary",
        dependencies=["Y"],
        requirement_links=["REQ-A-0002"],
        exposed_symbols=["bar"],
        source=source,
    )
    merged = ingestor_module._merge_interface_drafts(existing, incoming, Path("spec.md"), 1)
    assert merged.summary == "filled summary"
    assert sorted(merged.dependencies) == ["X", "Y"]

    conflicting = ingestor_module._InterfaceDraft(
        module_name="A",
        summary="different",
        dependencies=[],
        requirement_links=[],
        exposed_symbols=[],
        source=source,
    )
    with pytest.raises(SpecIngestError, match="Conflicting interface summary"):
        ingestor_module._merge_interface_drafts(merged, conflicting, Path("spec.md"), 1)


@pytest.mark.unit
def test_private_cross_reference_validation_errors() -> None:
    with pytest.raises(SpecIngestError, match="No requirements were extracted"):
        ingestor_module._validate_cross_references(
            ingestor_module._ParsedSource(source_document="spec.md"),
            Path("spec.md"),
        )

    source = SourceLocation(path="spec.md", section="Interfaces", line=10)
    merged = ingestor_module._ParsedSource(
        source_document="spec.md",
        requirements={
            "REQ-A-0001": ingestor_module._RequirementDraft(
                req_id="REQ-A-0001",
                statement="statement",
                acceptance_criteria=[],
                nfr_tags=[],
                source=SourceLocation(path="spec.md", section="Requirements", line=5),
            )
        },
        interfaces={
            "A": ingestor_module._InterfaceDraft(
                module_name="A",
                summary="summary",
                dependencies=[],
                requirement_links=["REQ-UNKNOWN-9999"],
                exposed_symbols=[],
                source=source,
            )
        },
    )
    with pytest.raises(SpecIngestError, match="references unknown requirement"):
        ingestor_module._validate_cross_references(merged, Path("spec.md"))


@pytest.mark.unit
def test_domain_validation_limit_branch_for_large_requirement_count() -> None:
    requirements = tuple(
        Requirement(
            id=f"REQ-LARGE-{index:04d}",
            statement="statement",
            source=SourceLocation(path="spec.md", section="Requirements", line=1),
        )
        for index in range(1, 10002)
    )
    spec_map = SpecMap(
        version=1,
        source_documents=("spec.md",),
        requirements=requirements,
    )
    with pytest.raises(ValueError, match="Too many requirements"):
        ingestor_module._validate_against_domain_spec_map(spec_map)


@pytest.mark.unit
def test_private_parser_and_merge_branch_coverage(tmp_path: Path) -> None:
    error = SpecIngestError(
        path=Path("spec.md"),
        line=10,
        column=2,
        section="Requirements",
        message="bad",
        hint="fix it",
    )
    assert "spec.md:10:2" in str(error)

    heading_lines = [
        ingestor_module._VisibleLine(line_number=1, text="#"),
        ingestor_module._VisibleLine(line_number=2, text="not setext"),
    ]
    assert ingestor_module._parse_heading(heading_lines, 0, previous_blank=True) is None
    implicit_heading = [ingestor_module._VisibleLine(line_number=1, text="Requirements")]
    assert ingestor_module._parse_heading(implicit_heading, 0, previous_blank=True) == (
        "Requirements",
        1,
    )

    parsed_interface = ingestor_module._parse_interface_item(
        Path("spec.md"),
        1,
        "Module A (`func`): standalone contract",
    )
    assert parsed_interface.dependencies == []
    assert parsed_interface.requirement_links == []

    entity, glossary = ingestor_module._parse_entity_item(Path("spec.md"), 2, "Account")
    assert entity.name == "Account"
    assert glossary is None

    existing_req = ingestor_module._RequirementDraft(
        req_id="REQ-MRG-0001",
        statement="statement",
        acceptance_criteria=["a"],
        nfr_tags=["nfr"],
        source=SourceLocation(path="spec.md", section="Requirements", line=1),
    )
    incoming_req = ingestor_module._RequirementDraft(
        req_id="REQ-MRG-0001",
        statement="statement",
        acceptance_criteria=["b"],
        nfr_tags=["security"],
        source=SourceLocation(path="spec.md", section="Requirements", line=2),
    )
    merged_req = ingestor_module._merge_requirement_drafts(existing_req, incoming_req)
    assert merged_req.acceptance_criteria == ["a", "b"]
    assert merged_req.nfr_tags == ["nfr", "security"]

    source = SourceLocation(path="spec.md", section="Interfaces", line=1)
    parsed_a = ingestor_module._ParsedSource(
        source_document="spec.md",
        requirements={
            "REQ-MRG-0001": ingestor_module._RequirementDraft(
                req_id="REQ-MRG-0001",
                statement="statement",
                acceptance_criteria=[],
                nfr_tags=["nfr"],
                source=SourceLocation(path="spec.md", section="Requirements", line=1),
            )
        },
        interfaces={
            "A": ingestor_module._InterfaceDraft(
                module_name="A",
                summary="",
                dependencies=["X"],
                requirement_links=[],
                exposed_symbols=["foo"],
                source=source,
            )
        },
    )
    parsed_b = ingestor_module._ParsedSource(
        source_document="spec2.md",
        requirements={
            "REQ-MRG-0001": ingestor_module._RequirementDraft(
                req_id="REQ-MRG-0001",
                statement="statement",
                acceptance_criteria=[],
                nfr_tags=["security"],
                source=SourceLocation(path="spec2.md", section="Requirements", line=1),
            )
        },
        interfaces={
            "A": ingestor_module._InterfaceDraft(
                module_name="A",
                summary="merged summary",
                dependencies=["Y"],
                requirement_links=["REQ-MRG-0001"],
                exposed_symbols=["bar"],
                source=source,
            )
        },
    )
    merged = ingestor_module._merge_sources([parsed_a, parsed_b])
    merged_interface = merged.interfaces["A"]
    assert merged_interface.summary == "merged summary"
    assert sorted(merged_interface.dependencies) == ["X", "Y"]
    assert merged.requirements["REQ-MRG-0001"].nfr_tags == ["nfr", "security"]

    parsed_c = ingestor_module._ParsedSource(
        source_document="spec3.md",
        requirements={
            "REQ-MRG-0001": ingestor_module._RequirementDraft(
                req_id="REQ-MRG-0001",
                statement="statement",
                acceptance_criteria=[],
                nfr_tags=[],
                source=SourceLocation(path="spec3.md", section="Requirements", line=1),
            )
        },
        interfaces={
            "A": ingestor_module._InterfaceDraft(
                module_name="A",
                summary="conflict summary",
                dependencies=[],
                requirement_links=[],
                exposed_symbols=[],
                source=source,
            )
        },
    )
    with pytest.raises(SpecIngestError, match="Interface conflict"):
        ingestor_module._merge_sources([parsed_b, parsed_c])

    nested_entities = _write_spec(
        tmp_path,
        "nested_entities.md",
        """
# Spec
## Requirements
- REQ-ENT-0001: Requirement.

## Entities
- User: canonical
    - nested detail to ignore
""",
    )
    parsed_nested = ingest_spec(nested_entities)
    assert parsed_nested.entities[0].name == "User"

    conflicting_entities = _write_spec(
        tmp_path,
        "conflicting_entities.md",
        """
# Spec
## Requirements
- REQ-ENT-0001: Requirement.

## Entities
- User: first
- User: second
""",
    )
    with pytest.raises(SpecIngestError, match="Entity definition conflict"):
        ingest_spec(conflicting_entities)
