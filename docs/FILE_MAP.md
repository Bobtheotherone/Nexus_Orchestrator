<!--
nexus-orchestrator — repository map index

File: docs/FILE_MAP.md
Last updated: 2026-02-12

Purpose
- A map of the repo layout: which directory corresponds to which plane/component in the design doc, and where key schemas live.

What should be included in this file
- One section per top-level directory explaining its purpose.
- A per-plane mapping: control/synthesis/verification/integration/knowledge/sandbox/observability/ui.
- Pointers to the file-by-file build order and to the design document sections that justify each directory.

Functional requirements
- Must allow an agent to locate where to implement a given feature quickly.

Non-functional requirements
- Must stay up to date; treat as high-signal index doc.

Suggested sections / outline
- Top-level directories
- Plane-to-path mapping
- Schemas and formats
- Where to start
-->

# File Map — Repository Layout and Plane-to-Path Mapping

This document maps every directory and key file to its purpose, the design document section that justifies it, and the build phase in which it is implemented.

---

## Top-Level Directories

| Directory | Purpose | Design Doc Section |
|---|---|---|
| `src/nexus_orchestrator/` | All Python source code, organized by architectural plane | §5 System Architecture |
| `.github/workflows/` | CI and security policy gates for repository enforcement | §10 Quality Assurance |
| `tests/` | Unit, integration, and smoke tests mirroring src structure | §10 Quality Assurance |
| `docs/` | Architecture specs, schemas, runbooks, prompt templates, ADRs | §13 Observability |
| `constraints/` | Constraint registry YAML files and reusable constraint libraries | §6 Constraint-Based Program Synthesis |
| `evidence/` | On-disk evidence artifacts (gitignored, local-only) | §4.3 Evidence |
| `state/` | SQLite state DB (gitignored, local-only) | §5.2 Persistence |
| `workspaces/` | Ephemeral agent workspaces (gitignored, garbage-collected) | §8.2 Non-Overlapping File Ownership |
| `docker/` | Sandbox container images | §12 Security and Sandboxing |
| `tools/` | Tool registry (approved tools, versions, checksums) | §5.2 Tool Provisioner |
| `samples/` | Sample design docs and test fixtures for smoke testing | §14 Project Lifecycle |
| `profiles/` | Operator personalization profile(s) (preferences/house rules) | §5 Knowledge Plane |
| `scripts/` | Development utilities, migration helpers, diagnostics | — |

---

## Phase 0 Ownership — Repo Blueprint & Tooling Gates

Prompt 1 reserves the following files as the Phase 0 contract surface for repository tooling gates, docs ownership, and meta verification:

| Path | Contract Role |
|---|---|
| `pyproject.toml` | Dependency/version gate and local tool configuration source |
| `Makefile` | Local quality/security gate command surface |
| `.github/workflows/ci.yml` | CI quality gate definition |
| `.github/workflows/security.yml` | Security gate definition |
| `docs/quality/STYLE_AND_LINT.md` | Normative style/type standards and constraint mapping |
| `scripts/repo_audit.py` | Deterministic repo audit CLI for blueprint JSON |
| `src/nexus_orchestrator/repo_blueprint.py` | Deterministic blueprint extractor and validator |
| `docs/REPO_BLUEPRINT.md` | Generated human-readable blueprint artifact |
| `constraints/registry/000_base_constraints.yaml` | Base registry invariants and style/security constraint IDs |
| `tools/registry.toml` | Pinned tool versions and risk metadata |
| `docs/BUILD_ORDER.md` | Phase ownership and sequencing authority |
| `docs/FILE_MAP.md` | Repository ownership and path mapping authority |
| `tests/meta/__init__.py` | Phase 0 ownership anchor for meta contract tests |

---

## Plane-to-Path Mapping

### Control Plane — `src/nexus_orchestrator/control_plane/`

The brain. Coordinates all other planes.

| File | Component | Purpose |
|---|---|---|
| `controller.py` | Orchestrator Controller | Run lifecycle state machine, pause/resume/cancel, re-planning triggers |
| `scheduler.py` | Planner & Scheduler | Critical-path scheduling, speculative execution, adaptive re-planning |
| `budgets.py` | Budget Enforcement | Token/cost/iteration limits, escalation rules, runaway detection |
| `feedback.py` | Feedback Synthesizer | Structures verification failures into actionable diagnostics for agent retries |

### Synthesis Plane — `src/nexus_orchestrator/synthesis_plane/`

The swarm. Manages LLM agent dispatch and tool usage.

| File | Component | Purpose |
|---|---|---|
| `dispatch.py` | Dispatch Controller | Concurrent API session management, rate limiting, provider routing |
| `context_assembler.py` | Context Assembler | Builds token-budgeted context packages (contracts, deps, code, history) |
| `roles.py` | Agent Role System | Role definitions, tool permissions, model preferences, budgets |
| `prompt_templates.py` | Prompt Renderer | Loads and renders role templates with strict placeholders |
| `tools.py` | Tool Request Protocol | How agents request tools, approval workflow, audit records |
| `providers/base.py` | Provider Interface | Abstract provider, common request/response models |
| `providers/openai_adapter.py` | OpenAI Adapter | Codex/GPT API wrapper |
| `providers/anthropic_adapter.py` | Anthropic Adapter | Claude API wrapper |

### Verification Plane — `src/nexus_orchestrator/verification_plane/`

The gate. Nothing passes without evidence.

| File | Component | Purpose |
|---|---|---|
| `constraint_gate.py` | Constraint Gate | Binary accept/reject with evidence aggregation |
| `pipeline.py` | Pipeline Engine | Executes checkers in order, manages parallelism, collects results |
| `evidence.py` | Evidence Writer | Creates, hashes, and stores evidence records |
| `checkers/base.py` | Checker Interface | Abstract checker, result model, registration |
| `checkers/build_checker.py` | Build Checker | Compilation verification |
| `checkers/lint_checker.py` | Lint Checker | Format and lint verification |
| `checkers/typecheck_checker.py` | Type Checker | Static type analysis |
| `checkers/test_checker.py` | Test Checker | Unit and integration test execution |
| `checkers/security_checker.py` | Security Checker | Secret scanning, dependency audit, vulnerability scan |
| `checkers/performance_checker.py` | Performance Checker | Benchmark execution against thresholds |
| `adversarial/test_generator.py` | Adversarial Test Gen | Independent test generation from contracts (no implementation access) |

### Integration Plane — `src/nexus_orchestrator/integration_plane/`

The gatekeeper. Serializes all merges through Git.

| File | Component | Purpose |
|---|---|---|
| `git_engine.py` | Git Engine | Wraps Git operations with scope enforcement and audit metadata |
| `workspace_manager.py` | Workspace Manager | Ephemeral workspace lifecycle, file ownership enforcement, GC |
| `merge_queue.py` | Merge Queue | Serialized integration: rebase, compositional check, merge or rollback |
| `conflict_resolution.py` | Conflict Resolver | Trivial auto-resolve, integrator agent dispatch, re-plan triggers |

### Knowledge Plane — `src/nexus_orchestrator/knowledge_plane/`

The memory. Indexes, retrieves, and evolves knowledge.

| File | Component | Purpose |
|---|---|---|
| `indexer.py` | Codebase Indexer | Incremental file/symbol/dependency indexing |
| `retrieval.py` | Context Retrieval | Ranked retrieval within token budgets for context assembler |
| `constraint_registry.py` | Constraint Registry | Load/validate/query constraint YAML files |
| `evidence_ledger.py` | Evidence Ledger | Traceability queries (requirement→constraint→evidence→commit) |
| `failure_mining.py` | Constraint Miner | "Never again" pipeline: failures become new constraints |
| `personalization.py` | Personalization | User preferences as constraints and routing heuristics |

### Cross-Cutting Modules

| Path | Purpose |
|---|---|
| `src/nexus_orchestrator/domain/` | Core data model (no IO, no side effects) |
| `src/nexus_orchestrator/config/` | Config schema, loading, validation, profiles |
| `src/nexus_orchestrator/persistence/` | SQLite state DB, migrations, repositories |
| `src/nexus_orchestrator/sandbox/` | Sandbox lifecycle, tool provisioner, resource governor, network policy |
| `src/nexus_orchestrator/security/` | Redaction, prompt injection defense |
| `src/nexus_orchestrator/observability/` | Structured logging, metrics, event bus |
| `src/nexus_orchestrator/modes/` | Operating modes (greenfield, brownfield, hardening, exploration) |
| `src/nexus_orchestrator/ui/` | CLI subcommands and optional TUI dashboard |
| `src/nexus_orchestrator/utils/` | Hashing, filesystem helpers, concurrency primitives |
| `src/nexus_orchestrator/ext/` | Extension/plugin infrastructure |

---

## Key Schemas and Formats

| Schema | Documentation | Implementation |
|---|---|---|
| Config schema | `docs/schemas/CONFIG.md` | `src/.../config/schema.py` |
| Config JSON Schema | `docs/schemas/config.schema.jsonc` | `src/.../config/schema.py` |
| Constraint registry JSON Schema | `docs/schemas/constraint_registry.schema.jsonc` | `src/.../knowledge_plane/constraint_registry.py` |
| Evidence ledger JSON Schema | `docs/schemas/evidence_ledger.schema.jsonc` | `src/.../knowledge_plane/evidence_ledger.py` |
| Constraint records | `docs/schemas/CONSTRAINT_REGISTRY.md` | `constraints/registry/*.yaml` |
| Evidence records | `docs/schemas/EVIDENCE_LEDGER.md` | `src/.../verification_plane/evidence.py` |
| State DB tables | `docs/schemas/STATE_DB.md` | `src/.../persistence/state_db.py` |
| Git protocol | `docs/architecture/GIT_PROTOCOL.md` | `src/.../integration_plane/git_engine.py` |
| Verification pipeline | `docs/architecture/VERIFICATION_PIPELINE.md` | `src/.../verification_plane/pipeline.py` |
| Role definitions | `docs/prompts/ROLE_SYSTEM.md` | `src/.../synthesis_plane/roles.py` |
| Prompt templates | `docs/prompts/templates/*.md` | `src/.../synthesis_plane/prompt_templates.py` |

---

## Where to Start

1. Read `design_document.md` for the full architecture
2. Read this file for repo layout orientation
3. Read `docs/BUILD_ORDER.md` for the exact implementation sequence
4. Read the specific `docs/architecture/*.md` file for the plane you are implementing
5. Read the docstring of the file you are about to implement — it contains functional and non-functional requirements
