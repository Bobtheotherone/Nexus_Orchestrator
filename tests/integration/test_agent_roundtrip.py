"""
nexus-orchestrator — test skeleton

File: tests/integration/test_agent_roundtrip.py
Last updated: 2026-02-11

Purpose
- Validate end-to-end agent roundtrip with mocked provider adapters.

What this test file should cover
- Prompt assembly → provider call → patch application → verification invocation.
- Budget and retry loops.
- Evidence ledger entry creation.

Functional requirements
- No real provider calls; mock network.

Non-functional requirements
- Deterministic; stable fixtures.
"""
