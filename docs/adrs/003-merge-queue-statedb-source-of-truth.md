# ADR-003: Merge Queue Persistence Uses StateDB as Source of Truth

## Status
Accepted

## Context
Merge queue durability relied on a standalone JSON state file, while the orchestrator already uses StateDB for durable state. This created split persistence semantics and brittle restart behavior.

## Decision
Persist merge queue state in StateDB as the authoritative source:
- Queue entries and queue state are stored in dedicated StateDB table(s).
- Restart reconstruction reads from StateDB, not JSON.
- Legacy JSON state is supported only for one-time import when DB state is absent.
- After import, runtime behavior ignores JSON for source-of-truth decisions.

## Consequences
- Positive: one durable store for orchestration state and merge queue state.
- Positive: crash/restart behavior is testable and deterministic.
- Negative: introduces DB migration complexity for queue schema evolution.

## Alternatives Considered
- Keep JSON file and mirror to DB asynchronously.
  Rejected because dual-write increases inconsistency risk and recovery ambiguity.

## Links
- Related modules: `src/nexus_orchestrator/integration_plane/merge_queue.py`
- Related modules: `src/nexus_orchestrator/persistence/state_db.py`
