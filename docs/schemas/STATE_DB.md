<!--
nexus-orchestrator — documentation skeleton

File: docs/schemas/STATE_DB.md
Last updated: 2026-02-11

Purpose
- Persistent state schema (SQLite-first) for runs, work items, attempts, merges, and queue state.

What should be included in this file
- Tables/collections: projects, runs, work_items, task_graph_edges, constraints, evidence_refs, provider_calls, tool_installs, sandboxes, merges, incidents.
- Indexing strategy for performance on a single machine.
- Migration/versioning strategy.

Functional requirements
- Must support safe resume after crash and idempotent operations.

Non-functional requirements
- Must remain fast and small; avoid unbounded growth without retention policies.

Suggested sections / outline
- Entities
- Indices
- Migrations
- Retention
- Backup/restore
-->

# State DB Schema (SQLite)

## Tables

| Table | Purpose | Key Fields |
|---|---|---|
| `runs` | Orchestration run lifecycle | id, spec_path, status, started_at, config_hash |
| `work_items` | Work item state and metadata | id, run_id, status, scope, risk_tier, constraint_envelope_hash |
| `task_graph_edges` | DAG edges between work items | parent_id, child_id |
| `constraints` | Constraint registry cache | id, severity, category, checker, active |
| `attempts` | Agent dispatch attempts | id, work_item_id, iteration, provider, model, result, cost |
| `evidence` | Evidence record metadata | id, work_item_id, stage, result, artifact_paths |
| `merges` | Merge queue records | id, work_item_id, commit_sha, evidence_ids |
| `provider_calls` | API call log (cost tracking) | id, attempt_id, provider, tokens, cost, latency |
| `tool_installs` | Tool provisioner audit log | id, tool, version, checksum, approved |
| `incidents` | Unexpected failures and errors | id, run_id, category, message, stack_trace |

## Design Decisions

- **SQLite WAL mode** for concurrent readers (UI + orchestrator).
- **Busy timeout** of 5 seconds to avoid lock contention.
- **Migrations** are sequential numbered files, idempotent, recorded in a `schema_versions` table.
- **No secrets** stored in the DB — only references.
- **Retention:** configurable per-table cleanup policies.

## Recovery

- State DB is the resume point after crashes.
- In-progress work items are marked `dispatched` and re-evaluated on restart.
- The merge queue is reconstructed from DB state, not held in memory.
