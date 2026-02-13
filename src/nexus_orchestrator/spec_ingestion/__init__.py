"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/spec_ingestion/__init__.py
Last updated: 2026-02-11

Purpose
- Spec ingestion plane: transforms input design docs into a normalized Spec Map used by planning.

What should be included in this file
- Ingestor entrypoints and spec source formats supported (Markdown, plain text).
- Normalization outputs and schema versioning.

Functional requirements
- Must produce a SpecMap that covers all requirements with IDs and acceptance criteria.

Non-functional requirements
- Must be deterministic; same input yields same SpecMap (modulo timestamps).
"""

from nexus_orchestrator.spec_ingestion.ingestor import SpecIngestError, ingest_spec
from nexus_orchestrator.spec_ingestion.spec_map import (
    Entity,
    EntityDiff,
    InterfaceContract,
    InterfaceDiff,
    Requirement,
    RequirementDiff,
    SourceLocation,
    SpecMap,
    SpecMapDiff,
    deserialize_spec_map,
    diff_spec_maps,
    serialize_spec_map,
    trace_requirement_to_work_items,
)

__all__ = [
    "Entity",
    "EntityDiff",
    "InterfaceContract",
    "InterfaceDiff",
    "Requirement",
    "RequirementDiff",
    "SourceLocation",
    "SpecIngestError",
    "SpecMap",
    "SpecMapDiff",
    "deserialize_spec_map",
    "diff_spec_maps",
    "ingest_spec",
    "serialize_spec_map",
    "trace_requirement_to_work_items",
]
