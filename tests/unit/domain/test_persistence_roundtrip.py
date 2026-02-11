"""
nexus-orchestrator — test skeleton

File: tests/unit/domain/test_persistence_roundtrip.py
Last updated: 2026-02-11

Purpose
- Round-trip domain objects through persistence adapters (in-memory or temp SQLite).

What this test file should cover
- Saving and loading core entities (Run, WorkItem, EvidenceRecord).
- Migration compatibility checks (when introduced).

Functional requirements
- Use temp directories / temp DB; no network.

Non-functional requirements
- Isolated and repeatable.
"""
