"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/context_assembler.py
Last updated: 2026-02-11

Purpose
- Builds the context package for an agent attempt: contracts, relevant code slices, constraints, and failure history.

What should be included in this file
- Deterministic selection rules: dependencies first, then similarity, then recency.
- Context size budgeting and truncation strategy with rationale.
- Content hygiene filters: exclude untrusted content that looks like prompt injection.
- Ability to include structured summaries instead of raw files.

Functional requirements
- Must guarantee inclusion of the work item's contract + scope + constraint envelope.
- Must support incremental updates as files change in integration branch.

Non-functional requirements
- Must be fast and cacheable; avoid re-indexing whole repo for each attempt.
"""
