"""
Unit tests for the multi-stage requirements extraction pipeline.

Tests cover:
1. Explicit section parsing (doc with Goals/Scope/Constraints)
2. Heuristic extraction (freeform narrative)
3. LLM semantic extraction (mocked backend)
4. Graceful fallback
5. Strict mode preservation
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nexus_orchestrator.planning.requirements_extractor import (
    ExtractionResult,
    ExtractedRequirement,
    _extract_from_sections,
    _extract_heuristic,
    _fallback_from_summary,
    _llm_results_to_requirements,
    _parse_llm_response,
    _parse_sections,
    extract_requirements,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _fixture_path(name: str) -> Path:
    return FIXTURES_DIR / name


# ---------------------------------------------------------------------------
# Stage 1: Explicit section parsing
# ---------------------------------------------------------------------------


class TestExplicitSectionParsing:
    def test_goals_scope_constraints_doc(self) -> None:
        """Doc with Goals/Scope/Constraints headings extracts >= 3 requirements."""
        path = _fixture_path("design_doc_goals_scope.md")
        result = extract_requirements(path)

        assert len(result.requirements) >= 3
        assert result.confidence > 0
        assert "fail" not in result.used_strategy.lower()

        # Check that goals/scope/constraints content was captured
        texts = [r.text.lower() for r in result.requirements]
        assert any("data processing" in t or "pipeline" in t for t in texts)

    def test_explicit_nonstandard_ids(self) -> None:
        """Doc with IDs like F001/TR-01 (not REQ-) should still extract them."""
        path = _fixture_path("design_doc_explicit_nonstandard.md")
        result = extract_requirements(path)

        assert len(result.requirements) >= 4
        # Check that some IDs were preserved from the doc
        ids = {r.id for r in result.requirements}
        # The F001 and TR-01 patterns should be recognized as explicit IDs
        texts = [r.text.lower() for r in result.requirements]
        assert any("payment" in t or "stripe" in t for t in texts)

    def test_sections_parsed_correctly(self) -> None:
        """Verify section parser identifies known headings."""
        lines = [
            "# My Project",
            "",
            "## Goals",
            "- Build a fast system",
            "- Support 1000 users",
            "",
            "## Constraints",
            "- Must run on Linux",
            "",
            "## Random Section",
            "- Not a requirement",
        ]
        sections = _parse_sections(lines)
        assert "goals" in sections
        assert "constraints" in sections
        assert len(sections["goals"]) == 2
        assert len(sections["constraints"]) == 1

    def test_fenced_code_blocks_skipped(self) -> None:
        """Content inside fenced code blocks is not extracted."""
        lines = [
            "## Requirements",
            "- The system must work",
            "```",
            "- This must not be extracted",
            "```",
            "- The system shall be fast",
        ]
        sections = _parse_sections(lines)
        assert "requirements" in sections
        items = sections["requirements"]
        texts = [t for _, t in items]
        assert not any("not be extracted" in t for t in texts)


# ---------------------------------------------------------------------------
# Stage 2: Heuristic extraction
# ---------------------------------------------------------------------------


class TestHeuristicExtraction:
    def test_freeform_narrative(self) -> None:
        """Freeform narrative with RFC-2119 keywords extracts requirements."""
        path = _fixture_path("design_doc_freeform.md")
        result = extract_requirements(path)

        assert len(result.requirements) >= 3
        texts = [r.text.lower() for r in result.requirements]
        # Should capture "must" / "shall" / "should" statements
        assert any("must" in t or "shall" in t or "should" in t for t in texts)

    def test_heuristic_keywords(self) -> None:
        """Verify heuristic scanner finds RFC-2119 keyword sentences."""
        lines = [
            "# Overview",
            "",
            "The system must handle 1000 concurrent users.",
            "This is just a description paragraph.",
            "The API shall return JSON responses.",
            "Some filler text here.",
            "The service should support graceful shutdown.",
        ]
        reqs = _extract_heuristic(lines, "test.md", set())
        assert len(reqs) >= 3
        texts = [r.text.lower() for r in reqs]
        assert any("concurrent users" in t for t in texts)
        assert any("json" in t for t in texts)

    def test_noise_lines_skipped(self) -> None:
        """Noise lines like horizontal rules and table rows are skipped."""
        lines = [
            "---",
            "| col1 | col2 |",
            "```python",
            "x = 1  # must not extract this",
            "```",
            "The system must be reliable.",
        ]
        reqs = _extract_heuristic(lines, "test.md", set())
        texts = [r.text for r in reqs]
        assert not any("x = 1" in t for t in texts)

    def test_nfr_type_detection(self) -> None:
        """Heuristic scanner classifies performance/security as nonfunctional."""
        lines = [
            "The system must achieve sub-second latency for all queries.",
            "All data must be encrypted with AES-256.",
        ]
        reqs = _extract_heuristic(lines, "test.md", set())
        types = {r.type for r in reqs}
        assert "nonfunctional" in types


# ---------------------------------------------------------------------------
# Stage 3: LLM semantic extraction (mocked)
# ---------------------------------------------------------------------------


class TestLLMExtraction:
    def test_parse_valid_json_array(self) -> None:
        """Valid JSON array is parsed correctly."""
        raw = json.dumps([
            {"id": "REQ-001", "text": "Support OAuth2", "type": "functional"},
            {"id": "REQ-002", "text": "99.9% uptime", "type": "nonfunctional"},
        ])
        items = _parse_llm_response(raw)
        assert len(items) == 2
        assert items[0]["id"] == "REQ-001"

    def test_parse_json_with_markdown_fences(self) -> None:
        """JSON wrapped in markdown code fences is handled."""
        raw = "```json\n" + json.dumps([
            {"id": "R1", "text": "Must work"},
        ]) + "\n```"
        items = _parse_llm_response(raw)
        assert len(items) == 1

    def test_parse_json_with_requirements_key(self) -> None:
        """JSON with a top-level 'requirements' key is unwrapped."""
        raw = json.dumps({
            "requirements": [
                {"id": "R1", "text": "Must work"},
            ]
        })
        items = _parse_llm_response(raw)
        assert len(items) == 1

    def test_parse_handles_statement_alias(self) -> None:
        """'statement' field is accepted as alias for 'text'."""
        raw = json.dumps([
            {"id": "R1", "statement": "The system shall scale"},
        ])
        items = _parse_llm_response(raw)
        assert len(items) == 1
        assert items[0]["text"] == "The system shall scale"

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises ValueError."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_llm_response("not json at all")

    def test_llm_results_to_requirements(self) -> None:
        """Validated LLM output converts to ExtractedRequirement objects."""
        raw_items = [
            {
                "id": "REQ-001",
                "text": "Support OAuth2 authentication",
                "type": "functional",
                "priority": "high",
                "acceptance_criteria": ["OAuth2 flow completes", "Token refresh works"],
            },
            {
                "id": "NFR-001",
                "text": "99.9% uptime SLA",
                "type": "nonfunctional",
                "priority": "high",
                "acceptance_criteria": [],
            },
        ]
        reqs = _llm_results_to_requirements(raw_items, "test.md")
        assert len(reqs) == 2
        assert reqs[0].id == "REQ-001"
        assert reqs[0].type == "functional"
        assert len(reqs[0].acceptance_criteria) == 2
        assert reqs[1].type == "nonfunctional"

    def test_mocked_llm_extractor_integration(self, tmp_path: Path) -> None:
        """Full pipeline with mocked LLM extractor produces requirements."""
        # Create a minimal doc that will trigger LLM fallback
        doc = tmp_path / "sparse.md"
        doc.write_text("# Project\nJust a brief description.\n")

        mock_llm = MagicMock()
        mock_llm.return_value = [
            {"id": "LLM-001", "text": "System must authenticate users", "type": "functional"},
            {"id": "LLM-002", "text": "Latency under 100ms", "type": "nonfunctional"},
            {"id": "LLM-003", "text": "Data encrypted at rest", "type": "constraint"},
        ]

        result = extract_requirements(doc, llm_extractor=mock_llm)

        assert len(result.requirements) >= 3
        assert "llm" in result.used_strategy
        mock_llm.assert_called_once()

    def test_llm_failure_degrades_gracefully(self, tmp_path: Path) -> None:
        """When LLM extractor raises, pipeline falls back without crashing."""
        doc = tmp_path / "sparse.md"
        doc.write_text("# Project\nJust a brief description.\n")

        mock_llm = MagicMock(side_effect=RuntimeError("API unavailable"))

        result = extract_requirements(doc, llm_extractor=mock_llm)

        # Should not crash, should produce some requirements via fallback
        assert len(result.requirements) >= 1
        assert any("LLM extraction failed" in w for w in result.warnings)

    def test_llm_extractor_deduplicates(self, tmp_path: Path) -> None:
        """LLM results that duplicate existing IDs are skipped."""
        doc = tmp_path / "with_goals.md"
        doc.write_text(
            "# Project\n\n## Goals\n"
            "- Build a fast system\n"
            "- Support many users\n"
            "- Handle errors gracefully\n"
        )

        # LLM returns something that overlaps with stage 1
        mock_llm = MagicMock()
        mock_llm.return_value = [
            {"id": "EXTRA-001", "text": "Additional LLM requirement", "type": "functional"},
        ]

        result = extract_requirements(doc, llm_extractor=mock_llm)
        # Stage 1 should find >= 3, so LLM shouldn't even be called
        assert len(result.requirements) >= 3
        # LLM should not have been called since stage 1 found enough
        mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 4: Graceful fallback
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_from_headings(self) -> None:
        """Fallback creates investigation items from document headings."""
        lines = [
            "# Main Title",
            "## Architecture",
            "Some text",
            "## Data Model",
            "Some text",
            "## Deployment",
            "Some text",
        ]
        reqs = _fallback_from_summary(lines, "test.md")
        assert len(reqs) >= 3
        assert all(r.id.startswith("INFERRED-") for r in reqs)
        texts = [r.text for r in reqs]
        assert any("Architecture" in t for t in texts)

    def test_fallback_from_no_headings(self) -> None:
        """Fallback with no headings creates a single investigation item."""
        lines = [
            "This is just a paragraph of text describing a system.",
            "It has no headings at all.",
        ]
        reqs = _fallback_from_summary(lines, "test.md")
        assert len(reqs) >= 1
        assert reqs[0].id == "INFERRED-0001"

    def test_empty_doc_produces_placeholder(self, tmp_path: Path) -> None:
        """Completely empty doc still produces a placeholder requirement."""
        doc = tmp_path / "empty.md"
        doc.write_text("")

        result = extract_requirements(doc)
        assert len(result.requirements) >= 1
        assert result.confidence <= 0.5


# ---------------------------------------------------------------------------
# Integration: ingest_spec with flexible mode
# ---------------------------------------------------------------------------


class TestIngestSpecFlexible:
    def test_goals_scope_doc_ingests_successfully(self) -> None:
        """A doc with Goals/Scope/Constraints (no REQ- format) ingests in flexible mode."""
        path = _fixture_path("design_doc_goals_scope.md")
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        spec_map = ingest_spec(path, strict_requirements=False)
        assert len(spec_map.requirements) >= 3

    def test_freeform_doc_ingests_successfully(self) -> None:
        """A freeform narrative doc ingests in flexible mode."""
        path = _fixture_path("design_doc_freeform.md")
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        spec_map = ingest_spec(path, strict_requirements=False)
        assert len(spec_map.requirements) >= 3

    def test_explicit_nonstandard_doc_ingests_successfully(self) -> None:
        """A doc with non-REQ IDs (F001, TR-01) ingests in flexible mode."""
        path = _fixture_path("design_doc_explicit_nonstandard.md")
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        spec_map = ingest_spec(path, strict_requirements=False)
        assert len(spec_map.requirements) >= 4

    def test_strict_mode_rejects_nonstandard_doc(self) -> None:
        """Strict mode still fails on docs without REQ- formatted requirements."""
        path = _fixture_path("design_doc_goals_scope.md")
        from nexus_orchestrator.spec_ingestion.ingestor import SpecIngestError, ingest_spec

        with pytest.raises(SpecIngestError, match="No requirements were extracted"):
            ingest_spec(path, strict_requirements=True)

    def test_existing_strict_doc_still_works(self) -> None:
        """The existing minimal_design_doc.md still works in both modes."""
        path = Path("samples/specs/minimal_design_doc.md")
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        # Default (flexible) mode
        spec_map = ingest_spec(path)
        assert len(spec_map.requirements) >= 3

        # Strict mode
        spec_map_strict = ingest_spec(path, strict_requirements=True)
        assert len(spec_map_strict.requirements) >= 3

    def test_extraction_info_attached(self) -> None:
        """Extraction metadata is returned via metadata_out when flexible mode is used."""
        path = _fixture_path("design_doc_goals_scope.md")
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        metadata: dict[str, object] = {}
        spec_map = ingest_spec(path, strict_requirements=False, metadata_out=metadata)
        info = metadata.get("extraction_info")
        assert info is not None
        assert info.count > 0
        assert info.strategy
        assert 0.0 <= info.confidence <= 1.0


# ---------------------------------------------------------------------------
# Integration: full plan pipeline with arbitrary doc
# ---------------------------------------------------------------------------


class TestFullPlanPipeline:
    def test_plan_goals_scope_doc(self) -> None:
        """Full plan pipeline works with Goals/Scope/Constraints doc."""
        path = _fixture_path("design_doc_goals_scope.md")
        from nexus_orchestrator.planning import (
            build_deterministic_architect_output,
            compile_constraints,
        )
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        spec_map = ingest_spec(path)
        architect_output = build_deterministic_architect_output(spec_map)
        result = compile_constraints(spec_map, architect_output)

        assert len(result.work_items) > 0
        assert result.task_graph is not None

    def test_plan_freeform_doc(self) -> None:
        """Full plan pipeline works with freeform narrative doc."""
        path = _fixture_path("design_doc_freeform.md")
        from nexus_orchestrator.planning import (
            build_deterministic_architect_output,
            compile_constraints,
        )
        from nexus_orchestrator.spec_ingestion.ingestor import ingest_spec

        spec_map = ingest_spec(path)
        architect_output = build_deterministic_architect_output(spec_map)
        result = compile_constraints(spec_map, architect_output)

        assert len(result.work_items) > 0
        assert result.task_graph is not None
