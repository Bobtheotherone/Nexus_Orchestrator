"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/persistence/repositories.py
Last updated: 2026-02-11

Purpose
- Repository/DAO interfaces for reading/writing domain entities to the state DB.

What should be included in this file
- Repositories: RunRepo, WorkItemRepo, ConstraintRepo, EvidenceRepo, MergeRepo, ToolRepo, ProviderCallRepo.
- Query patterns needed by scheduler and UI (e.g., next runnable work items).
- Pagination and indexing considerations.

Functional requirements
- Must provide atomic updates for state transitions (e.g., work item status changes).
- Must support append-only evidence recording semantics.

Non-functional requirements
- Must be efficient; avoid loading entire run history into memory.
"""
