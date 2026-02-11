<!--
nexus-orchestrator — documentation skeleton

File: docs/architecture/DATA_MODEL.md
Last updated: 2026-02-11

Purpose
- Detailed data model specification (Artifact, Constraint, Evidence, WorkItem, TaskGraph, Tool, Sandbox, Run).

What should be included in this file
- Entity definitions and relationships (ERD-like description).
- Field-level semantics, required/optional fields, and versioning rules.
- IDs and traceability rules (requirement IDs, constraint IDs, evidence IDs).
- Serialization formats and backwards-compat policy.

Functional requirements
- Must be specific enough to implement persistent storage and APIs without guesswork.

Non-functional requirements
- Must avoid overfitting to one DB; support at least SQLite for local-first MVP.

Suggested sections / outline
- Entity catalog
- IDs and naming
- Versioning
- Persistence concerns
- Examples (non-code)
-->

# Data Model

## Entity Catalog

### Requirement
- `id` (str): Unique ID (e.g., REQ-0001). Stable once assigned.
- `statement` (str): The requirement text.
- `acceptance_criteria` (list[str]): Machine-checkable criteria.
- `nfr_tags` (list[str]): Non-functional tags (perf, security, reliability).
- `source` (str): Origin file and section.

### SpecMap
- `version` (int): Schema version.
- `requirements` (list[Requirement]): All extracted requirements.
- `entities` (list[Entity]): Domain entities and their relationships.
- `interfaces` (list[InterfaceContract]): Module boundaries and contracts.
- `glossary` (dict[str, str]): Canonical term definitions.

### WorkItem
- `id` (str): ULID. Globally unique.
- `scope` (list[str]): Owned file paths (non-overlapping).
- `constraint_envelope` (ConstraintEnvelope): Full set of applicable constraints.
- `dependencies` (list[str]): Work item IDs this depends on.
- `status` (enum): pending | ready | dispatched | verifying | passed | failed | merged
- `risk_tier` (enum): low | medium | high | critical
- `budget` (Budget): Token, cost, iteration, and time limits.
- `requirement_links` (list[str]): Requirement IDs this satisfies.

### Constraint
- `id` (str): Unique (e.g., CON-SEC-0001).
- `severity` (enum): must | should | may
- `category` (str): structural | behavioral | performance | security | style | ...
- `description` (str): Human-readable rule.
- `checker_binding` (str): Checker ID that verifies this constraint.
- `parameters` (dict): Checker-specific config.
- `requirement_links` (list[str]): Requirement IDs this enforces.
- `source` (enum): spec_derived | failure_derived | manual
- `created_at` (datetime): When added.
- `failure_history` (list[FailureRef]): Past violations.

### EvidenceRecord
- `id` (str): ULID.
- `work_item_id` (str): What this evidence is for.
- `run_id` (str): Which orchestration run.
- `stage` (str): Which pipeline stage produced it.
- `result` (enum): pass | fail | warn | skip
- `checker_id` (str): Which checker ran.
- `constraint_ids` (list[str]): Which constraints this covers.
- `tool_versions` (dict[str, str]): Tool name → version.
- `environment_hash` (str): Hash of execution environment.
- `artifact_paths` (list[str]): Paths to logs, reports, etc.
- `duration_ms` (int): Execution time.
- `created_at` (datetime): Timestamp.

### Attempt
- `id` (str): ULID.
- `work_item_id` (str): Parent work item.
- `iteration` (int): 1-based attempt number.
- `provider` (str): Which provider was used.
- `model` (str): Which model.
- `role` (str): Agent role.
- `prompt_hash` (str): Hash of the rendered prompt.
- `tokens_used` (int): Total tokens.
- `cost_usd` (float): Estimated cost.
- `result` (enum): success | constraint_failure | error | timeout
- `feedback` (str | None): Feedback synthesizer output if failed.

### MergeRecord
- `id` (str): ULID.
- `work_item_id` (str): What was merged.
- `commit_sha` (str): The merge commit.
- `evidence_ids` (list[str]): Evidence records proving constraints.
- `merged_at` (datetime): Timestamp.

## Persistence

- Primary store: SQLite (WAL mode for concurrent readers).
- Evidence artifacts: on-disk files under `evidence/` with hash-based integrity.
- Constraint YAML: files under `constraints/registry/` versioned in Git.
- Schema migrations: sequential, idempotent, recorded in state DB.
