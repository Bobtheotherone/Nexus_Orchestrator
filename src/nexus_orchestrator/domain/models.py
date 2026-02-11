"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/domain/models.py
Last updated: 2026-02-11

Purpose
- Core data structures for orchestration (no persistence logic).

What should be included in this file
- Definitions for: Requirement, SpecMap, WorkItem, TaskGraph, Constraint, ConstraintEnvelope, EvidenceRecord, Artifact, Attempt, MergeRecord, Incident.
- Fields for traceability: links between requirements, constraints, evidence, commits.
- Risk tiering model and severity enums.
- Policy objects: budgets, resource limits, sandbox policies.

Functional requirements
- Must be explicit about required vs optional fields, and defaults.
- Must support schema versioning for serialized forms.

Non-functional requirements
- Keep models small; avoid embedding large blobs (store references to on-disk artifacts instead).

Testing guidance
- Round-trip serialization tests for each model version.
- Validation tests for required fields and invariants.
"""
