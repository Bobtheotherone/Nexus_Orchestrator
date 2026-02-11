<!--
nexus-orchestrator — documentation skeleton

File: docs/BUILD_ORDER.md
Last updated: 2026-02-11

Purpose
- A prescriptive file-by-file implementation order for an agentic AI to build the orchestrator incrementally.

What should be included in this file
- Phase-based build order aligned with the roadmap (MVP -> contracts -> sandbox -> adversarial -> evolution -> UI).
- For each phase: required modules/files to implement, acceptance tests, and expected deliverables.
- Explicit dependency ordering (e.g., domain model -> persistence -> git integration -> verification -> agent dispatch).
- Minimal 'vertical slices' for early end-to-end functionality.

Functional requirements
- Must define the MVP that can ingest a tiny spec, create work items, dispatch one agent, run one verifier, and commit evidence metadata.
- Must define how to validate correctness at each phase (smoke tests and golden scenarios).

Non-functional requirements
- Must minimize rework: stabilize formats and contracts early.

Suggested sections / outline
- Phase 1 (MVP)
- Phase 2 (Contracts & scopes)
- Phase 3 (Sandbox & tools)
- Phase 4 (Adversarial & evolution)
- Phase 5 (UX)
-->

# Build Order — Incremental Implementation Guide for Agentic AI

This document defines the exact order in which to implement the orchestrator, phase by phase, file by file. Each phase produces a working vertical slice that can be tested before moving on. Dependencies flow downward — never implement a file before its dependencies are complete.

---

## Phase 1 — Domain Foundation (No IO, No Side Effects)

**Goal:** Establish the core data model, IDs, events, and constants that every other module depends on. This phase has zero external dependencies and zero IO.

**Build order:**

1. `src/nexus_orchestrator/constants.py` — branch names, schema versions, risk tiers, default paths
2. `src/nexus_orchestrator/domain/ids.py` — ID generation (ULIDs recommended), validation, formatting
3. `src/nexus_orchestrator/domain/models.py` — all core dataclasses: Requirement, SpecMap, WorkItem, Constraint, ConstraintEnvelope, EvidenceRecord, Artifact, Attempt, MergeRecord, Incident, Budget, SandboxPolicy
4. `src/nexus_orchestrator/domain/events.py` — event types for state transitions (RunStarted, WorkItemDispatched, VerificationFailed, MergeSucceeded, etc.)
5. `src/nexus_orchestrator/domain/__init__.py` — re-exports of public domain types
6. `src/nexus_orchestrator/utils/hashing.py` — SHA-256 helpers for evidence integrity
7. `src/nexus_orchestrator/utils/fs.py` — atomic writes, safe delete, path guards
8. `src/nexus_orchestrator/utils/concurrency.py` — cancellation tokens, bounded semaphores, async worker pools
9. `src/nexus_orchestrator/utils/__init__.py` — re-exports

**Acceptance test:** All domain models serialize to JSON and back without data loss. IDs are unique across 10,000 generations. Event types are exhaustive for the lifecycle.

**Test files to create:**
- `tests/unit/domain/test_ids.py`
- `tests/unit/domain/test_models.py`
- `tests/unit/domain/test_events.py`

---

## Phase 2 — Persistence and Config (Minimal IO)

**Goal:** Establish SQLite state storage and config loading so the system can persist and resume.

**Dependencies:** Phase 1 complete.

**Build order:**

1. `src/nexus_orchestrator/config/schema.py` — config schema with validation, profiles, versioning
2. `src/nexus_orchestrator/config/loader.py` — TOML + env var loader, precedence rules, secret reference resolution
3. `src/nexus_orchestrator/config/__init__.py` — public config types
4. `src/nexus_orchestrator/persistence/state_db.py` — SQLite schema, migrations, connection lifecycle, WAL mode
5. `src/nexus_orchestrator/persistence/repositories.py` — DAOs for all domain entities (RunRepo, WorkItemRepo, ConstraintRepo, EvidenceRepo, MergeRepo, ProviderCallRepo)
6. `src/nexus_orchestrator/persistence/__init__.py` — public persistence interfaces
7. `src/nexus_orchestrator/security/redaction.py` — secret pattern detection, redaction transforms
8. `src/nexus_orchestrator/observability/logging.py` — structured JSON logging with redaction and correlation IDs
9. `src/nexus_orchestrator/observability/metrics.py` — metric definitions and counters
10. `src/nexus_orchestrator/observability/events.py` — event bus (in-process pub/sub)
11. `src/nexus_orchestrator/observability/__init__.py` — setup helpers

**Acceptance test:** Config loads from orchestrator.toml + env overrides. State DB creates tables, inserts a run, and survives process restart. Secrets are never present in log output.

**Test files to create:**
- `tests/unit/config/test_schema.py`
- `tests/unit/config/test_loader.py`
- `tests/unit/domain/test_persistence_roundtrip.py`

---

## Phase 3 — Spec Ingestion and Planning

**Goal:** Parse a design document into a SpecMap, compile constraints, and produce a task graph.

**Dependencies:** Phases 1-2 complete.

**Build order:**

1. `src/nexus_orchestrator/spec_ingestion/spec_map.py` — SpecMap schema, serialization, traceability helpers
2. `src/nexus_orchestrator/spec_ingestion/ingestor.py` — Markdown parser, requirement extractor, entity extractor
3. `src/nexus_orchestrator/spec_ingestion/__init__.py`
4. `src/nexus_orchestrator/knowledge_plane/constraint_registry.py` — load/validate/query constraints from YAML
5. `src/nexus_orchestrator/planning/task_graph.py` — DAG representation, topo sort, critical path, next-runnable queries
6. `src/nexus_orchestrator/planning/architect_interface.py` — structured output format for architect agent decomposition
7. `src/nexus_orchestrator/planning/constraint_compiler.py` — validate DAG, compile envelopes, propagate constraints, detect conflicts
8. `src/nexus_orchestrator/planning/__init__.py`

**Acceptance test:** `samples/specs/minimal_design_doc.md` ingests into a SpecMap with 5 requirements. The constraint compiler produces 3 work items (A->B->C) with complete envelopes. The task graph computes correct topological ordering and critical path.

**Test files to create:**
- `tests/unit/planning/test_task_graph.py`
- `tests/unit/planning/test_constraint_compiler.py`
- `tests/integration/test_spec_to_plan.py`

---

## Phase 4 — Git Integration and Workspace Isolation

**Goal:** Create isolated branches, enforce file ownership, manage workspaces, implement the merge queue.

**Dependencies:** Phases 1-3 complete.

**Build order:**

1. `src/nexus_orchestrator/integration_plane/git_engine.py` — Git abstraction (init, branch, rebase, merge, revert, diff, scope check)
2. `src/nexus_orchestrator/integration_plane/workspace_manager.py` — create/populate/cleanup ephemeral workspaces
3. `src/nexus_orchestrator/integration_plane/merge_queue.py` — serialized queue, compositional checks, rollback
4. `src/nexus_orchestrator/integration_plane/conflict_resolution.py` — trivial auto-resolve, integrator dispatch
5. `src/nexus_orchestrator/integration_plane/__init__.py`

**Acceptance test:** Create 3 work branches with non-overlapping file ownership. Apply patches. Merge sequentially through the queue. Verify scope violation is blocked. Verify rollback on simulated compositional failure.

**Test files to create:**
- `tests/unit/integration_plane/test_git_engine.py`
- `tests/unit/integration_plane/test_merge_queue.py`
- `tests/integration/test_workspace_lifecycle.py`

---

## Phase 5 — Verification Pipeline and Constraint Gate

**Goal:** Build the multi-stage verification engine with pluggable checkers and evidence generation.

**Dependencies:** Phases 1-4 complete.

**Build order:**

1. `src/nexus_orchestrator/verification_plane/checkers/base.py` — abstract Checker interface and result model
2. `src/nexus_orchestrator/verification_plane/checkers/scope_checker.py` — enforce declared work-item scope boundaries
3. `src/nexus_orchestrator/verification_plane/checkers/schema_checker.py` — validate registries/config against schemas and invariants
4. `src/nexus_orchestrator/verification_plane/checkers/build_checker.py` — compilation/build verification
5. `src/nexus_orchestrator/verification_plane/checkers/lint_checker.py` — formatting and lint
6. `src/nexus_orchestrator/verification_plane/checkers/typecheck_checker.py` — static type checking
7. `src/nexus_orchestrator/verification_plane/checkers/test_checker.py` — unit/integration test runner
8. `src/nexus_orchestrator/verification_plane/checkers/security_checker.py` — secret scan, dependency audit, vulnerability scan
9. `src/nexus_orchestrator/verification_plane/checkers/documentation_checker.py` — doc completeness rules for public API/contract changes
10. `src/nexus_orchestrator/verification_plane/checkers/reliability_checker.py` — timeouts/retries/policy checks for external boundaries
11. `src/nexus_orchestrator/verification_plane/checkers/performance_checker.py` — benchmarks against thresholds
8. `src/nexus_orchestrator/verification_plane/checkers/__init__.py` — checker registry
9. `src/nexus_orchestrator/verification_plane/evidence.py` — evidence record creation, hashing, storage
10. `src/nexus_orchestrator/verification_plane/pipeline.py` — stage orchestration, parallelism within resource limits
11. `src/nexus_orchestrator/verification_plane/constraint_gate.py` — binary accept/reject with evidence aggregation
12. `src/nexus_orchestrator/verification_plane/__init__.py`

**Acceptance test:** A mock patch runs through all pipeline stages. Passing patch produces complete evidence bundle. Failing patch returns structured diagnostics per failing constraint. Pipeline respects concurrency limits.

**Test files to create:**
- `tests/unit/verification_plane/test_pipeline.py`
- `tests/unit/verification_plane/test_constraint_gate.py`
- `tests/unit/verification_plane/test_checkers.py`

---

## Phase 6 — Synthesis Plane (Agent Dispatch)

**Goal:** Build the agent runtime: provider adapters, role system, context assembly, prompt rendering, and dispatch controller.

**Dependencies:** Phases 1-5 complete.

**Build order:**

1. `src/nexus_orchestrator/synthesis_plane/providers/base.py` — abstract provider, request/response models, error taxonomy
2. `src/nexus_orchestrator/synthesis_plane/providers/openai_adapter.py` — OpenAI/Codex adapter
3. `src/nexus_orchestrator/synthesis_plane/providers/anthropic_adapter.py` — Anthropic/Claude adapter
4. `src/nexus_orchestrator/synthesis_plane/providers/__init__.py` — provider registry
5. `src/nexus_orchestrator/synthesis_plane/roles.py` — role definitions, tool permissions, budgets, model preferences
6. `src/nexus_orchestrator/synthesis_plane/prompt_templates.py` — template loader and renderer
7. `src/nexus_orchestrator/security/prompt_hygiene.py` — injection detection, content sanitization
8. `src/nexus_orchestrator/knowledge_plane/indexer.py` — repo indexing for context retrieval
9. `src/nexus_orchestrator/knowledge_plane/retrieval.py` — ranked context retrieval within token budgets
10. `src/nexus_orchestrator/synthesis_plane/context_assembler.py` — builds full context package for each attempt
11. `src/nexus_orchestrator/synthesis_plane/tools.py` — tool request protocol and approval workflow
12. `src/nexus_orchestrator/synthesis_plane/dispatch.py` — dispatch controller with concurrency, routing, retry
13. `src/nexus_orchestrator/synthesis_plane/__init__.py`

**Acceptance test:** Dispatch a work item to a mocked provider. Receive structured response. Context package includes contracts + constraints + relevant code. Provider routing prefers Codex for simple items, Claude for complex. Mocked dispatch completes full round-trip.

**Test files to create:**
- `tests/unit/synthesis_plane/test_context_assembler.py`
- `tests/unit/synthesis_plane/test_dispatch.py`
- `tests/unit/synthesis_plane/test_roles.py`
- `tests/integration/test_agent_roundtrip.py`

---

## Phase 7 — Control Plane and End-to-End Loop

**Goal:** Wire everything together: the controller runs the full lifecycle from spec to merged code.

**Dependencies:** Phases 1-6 complete.

**Build order:**

1. `src/nexus_orchestrator/control_plane/budgets.py` — budget enforcement and escalation
2. `src/nexus_orchestrator/control_plane/feedback.py` — feedback synthesizer for structured iteration diagnostics
3. `src/nexus_orchestrator/control_plane/scheduler.py` — critical-path scheduling, adaptive re-planning, speculative execution
4. `src/nexus_orchestrator/control_plane/controller.py` — run lifecycle state machine, pause/resume/cancel
5. `src/nexus_orchestrator/control_plane/__init__.py`
6. `src/nexus_orchestrator/sandbox/resource_governor.py` — CPU/RAM/disk monitoring and backpressure
7. `src/nexus_orchestrator/sandbox/network_policy.py` — egress controls
8. `src/nexus_orchestrator/sandbox/sandbox_manager.py` — container lifecycle
9. `src/nexus_orchestrator/sandbox/tool_provisioner.py` — tool install, pin, scan, record
10. `src/nexus_orchestrator/sandbox/__init__.py`

**Acceptance test:** Full end-to-end with `minimal_design_doc.md`: ingest -> plan -> dispatch (mocked) -> verify (mocked checkers) -> merge -> evidence recorded. Crash mid-run and resume successfully.

**Test files to create:**
- `tests/unit/control_plane/test_scheduler.py`
- `tests/unit/control_plane/test_feedback.py`
- `tests/unit/sandbox/test_resource_governor.py`
- `tests/smoke/test_end_to_end.py`

---

## Phase 8 — Adversarial Verification and Constraint Evolution

**Goal:** Add the quality amplification layer: adversarial testing, constraint mining, and the "never again" pipeline.

**Dependencies:** Phases 1-7 complete.

**Build order:**

1. `src/nexus_orchestrator/verification_plane/adversarial/test_generator.py` — independent test generation from contracts
2. `src/nexus_orchestrator/verification_plane/adversarial/__init__.py`
3. `src/nexus_orchestrator/knowledge_plane/failure_mining.py` — failure classification, constraint proposal, dedup
4. `src/nexus_orchestrator/knowledge_plane/evidence_ledger.py` — traceability queries, audit export
5. `src/nexus_orchestrator/knowledge_plane/personalization.py` — preference storage and influence on routing
6. `src/nexus_orchestrator/knowledge_plane/__init__.py`
7. `src/nexus_orchestrator/modes/greenfield.py` — greenfield mode orchestration
8. `src/nexus_orchestrator/modes/brownfield.py` — brownfield mode with existing repo indexing
9. `src/nexus_orchestrator/modes/hardening.py` — hardening mode focused on tests/security/perf
10. `src/nexus_orchestrator/modes/exploration.py` — parallel candidate evaluation
11. `src/nexus_orchestrator/modes/__init__.py` — mode registry and selection

**Acceptance test:** Adversarial generator produces tests that catch a known edge case. Failure mining turns a simulated failure into a new constraint YAML. Operating modes produce different task graphs for the same spec.

---

## Phase 9 — User Interface and Polish

**Goal:** Add CLI, optional TUI, and finalize documentation.

**Dependencies:** Phases 1-8 complete.

**Build order:**

1. `src/nexus_orchestrator/ui/cli.py` — subcommands: plan, run, status, inspect, export, clean
2. `src/nexus_orchestrator/ui/tui.py` — optional live dashboard
3. `src/nexus_orchestrator/ui/__init__.py`
4. `src/nexus_orchestrator/main.py` — entrypoint, bootstrap, shutdown, exit codes
5. `src/nexus_orchestrator/__init__.py` — package metadata

**Acceptance test:** `nexus plan samples/specs/minimal_design_doc.md` produces a task graph. `nexus run --mock` completes full cycle. `nexus status` shows run state. `nexus export` produces audit bundle.

---

## Invariants Across All Phases

- Every phase must have passing tests before the next phase begins.
- Every new public function must have at least one test.
- Domain models are never modified in a way that breaks serialization of existing data.
- Config schema changes require a migration note in CHANGELOG.md.
- No file outside the current phase's scope is modified (agents follow the same rules the orchestrator enforces on its targets).
