<!--
nexus-orchestrator — repository blueprint report

File: docs/REPO_BLUEPRINT.md
Last updated: 2026-02-11

Purpose
- Human-readable map of repository planes, phases, dependencies, and quality risks.

What should be included in this file
- Compact architecture overview.
- Source module table: path, plane, phase, dependencies, tests covering it.
- Missing/skeleton areas with exact file pointers.

Functional requirements
- Must be derived from deterministic audit extraction, not hand-maintained prose.

Non-functional requirements
- Keep this document concise enough for small context windows.
-->

# Repo Blueprint

## How The Repo Fits Together

NEXUS is organized into plane-oriented Python packages under `src/nexus_orchestrator/`, with implementation sequencing defined in `docs/BUILD_ORDER.md` and folder-level intent defined in `docs/FILE_MAP.md`.
This report is generated from repository state plus headers and doc references, then validated against deterministic invariants.

## Source Map

| Path | Plane | Phase | Dependencies | Tests Covering |
|---|---|---|---|---|
| `src/nexus_orchestrator/__init__.py` | `control` | `Phase 9 — User Interface and Polish` | `-` | `-` |
| `src/nexus_orchestrator/config/__init__.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/config/loader.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/config/schema.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/constants.py` | `control` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/control_plane/__init__.py` | `control` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/control_plane/budgets.py` | `control` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/control_plane/controller.py` | `control` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/control_plane/feedback.py` | `control` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/control_plane/scheduler.py` | `control` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/domain/__init__.py` | `domain` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/domain/events.py` | `domain` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/domain/ids.py` | `domain` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/domain/models.py` | `domain` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/ext/README.md` | `control` | `Unassigned` | `-` | `-` |
| `src/nexus_orchestrator/integration_plane/__init__.py` | `integration` | `Phase 4 — Git Integration and Workspace Isolation` | `-` | `tests/integration/test_workspace_lifecycle.py, tests/unit/integration_plane/test_git_engine.py, tests/unit/integration_plane/test_merge_queue.py` |
| `src/nexus_orchestrator/integration_plane/conflict_resolution.py` | `integration` | `Phase 4 — Git Integration and Workspace Isolation` | `-` | `tests/integration/test_workspace_lifecycle.py, tests/unit/integration_plane/test_git_engine.py, tests/unit/integration_plane/test_merge_queue.py` |
| `src/nexus_orchestrator/integration_plane/git_engine.py` | `integration` | `Phase 4 — Git Integration and Workspace Isolation` | `-` | `tests/integration/test_workspace_lifecycle.py, tests/unit/integration_plane/test_git_engine.py, tests/unit/integration_plane/test_merge_queue.py` |
| `src/nexus_orchestrator/integration_plane/merge_queue.py` | `integration` | `Phase 4 — Git Integration and Workspace Isolation` | `-` | `tests/integration/test_workspace_lifecycle.py, tests/unit/integration_plane/test_git_engine.py, tests/unit/integration_plane/test_merge_queue.py` |
| `src/nexus_orchestrator/integration_plane/workspace_manager.py` | `integration` | `Phase 4 — Git Integration and Workspace Isolation` | `-` | `tests/integration/test_workspace_lifecycle.py, tests/unit/integration_plane/test_git_engine.py, tests/unit/integration_plane/test_merge_queue.py` |
| `src/nexus_orchestrator/knowledge_plane/__init__.py` | `knowledge` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/knowledge_plane/constraint_registry.py` | `knowledge` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/knowledge_plane/evidence_ledger.py` | `knowledge` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/knowledge_plane/failure_mining.py` | `knowledge` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/knowledge_plane/indexer.py` | `knowledge` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/knowledge_plane/personalization.py` | `knowledge` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/knowledge_plane/retrieval.py` | `knowledge` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/main.py` | `control` | `Phase 9 — User Interface and Polish` | `-` | `-` |
| `src/nexus_orchestrator/modes/__init__.py` | `control` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/modes/brownfield.py` | `control` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/modes/exploration.py` | `control` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/modes/greenfield.py` | `control` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/modes/hardening.py` | `control` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/observability/__init__.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/observability/events.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/observability/logging.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/observability/metrics.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/persistence/__init__.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/persistence/repositories.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/persistence/state_db.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/planning/__init__.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/planning/architect_interface.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/planning/constraint_compiler.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/planning/task_graph.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/repo_blueprint.py` | `control` | `Unassigned` | `-` | `-` |
| `src/nexus_orchestrator/sandbox/__init__.py` | `sandbox` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/sandbox/network_policy.py` | `sandbox` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/sandbox/resource_governor.py` | `sandbox` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/sandbox/sandbox_manager.py` | `sandbox` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/sandbox/tool_provisioner.py` | `sandbox` | `Phase 7 — Control Plane and End-to-End Loop` | `-` | `tests/smoke/test_end_to_end.py, tests/unit/control_plane/test_feedback.py, tests/unit/control_plane/test_scheduler.py, tests/unit/sandbox/test_resource_governor.py` |
| `src/nexus_orchestrator/security/__init__.py` | `control` | `Unassigned` | `-` | `-` |
| `src/nexus_orchestrator/security/prompt_hygiene.py` | `control` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/security/redaction.py` | `control` | `Phase 2 — Persistence and Config (Minimal IO)` | `-` | `tests/unit/config/test_loader.py, tests/unit/config/test_schema.py, tests/unit/domain/test_persistence_roundtrip.py` |
| `src/nexus_orchestrator/spec_ingestion/__init__.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/spec_ingestion/ingestor.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/spec_ingestion/spec_map.py` | `control` | `Phase 3 — Spec Ingestion and Planning` | `-` | `tests/integration/test_spec_to_plan.py, tests/unit/planning/test_constraint_compiler.py, tests/unit/planning/test_task_graph.py` |
| `src/nexus_orchestrator/synthesis_plane/__init__.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/context_assembler.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/dispatch.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/prompt_templates.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/providers/__init__.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/providers/anthropic_adapter.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/providers/base.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/providers/openai_adapter.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/roles.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/synthesis_plane/tools.py` | `synthesis` | `Phase 6 — Synthesis Plane (Agent Dispatch)` | `-` | `tests/integration/test_agent_roundtrip.py, tests/unit/synthesis_plane/test_context_assembler.py, tests/unit/synthesis_plane/test_dispatch.py, tests/unit/synthesis_plane/test_roles.py` |
| `src/nexus_orchestrator/ui/__init__.py` | `ui` | `Phase 9 — User Interface and Polish` | `-` | `-` |
| `src/nexus_orchestrator/ui/cli.py` | `ui` | `Phase 9 — User Interface and Polish` | `-` | `-` |
| `src/nexus_orchestrator/ui/tui.py` | `ui` | `Phase 9 — User Interface and Polish` | `-` | `-` |
| `src/nexus_orchestrator/utils/__init__.py` | `utils` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/utils/concurrency.py` | `utils` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/utils/fs.py` | `utils` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/utils/hashing.py` | `utils` | `Phase 1 — Domain Foundation (No IO, No Side Effects)` | `-` | `tests/unit/domain/test_events.py, tests/unit/domain/test_ids.py, tests/unit/domain/test_models.py` |
| `src/nexus_orchestrator/verification_plane/__init__.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/adversarial/__init__.py` | `verification` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/verification_plane/adversarial/test_generator.py` | `verification` | `Phase 8 — Adversarial Verification and Constraint Evolution` | `-` | `-` |
| `src/nexus_orchestrator/verification_plane/checkers/__init__.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/base.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/build_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/documentation_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/lint_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/performance_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/reliability_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/schema_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/scope_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/security_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/test_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/checkers/typecheck_checker.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/constraint_gate.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/evidence.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |
| `src/nexus_orchestrator/verification_plane/pipeline.py` | `verification` | `Phase 5 — Verification Pipeline and Constraint Gate` | `-` | `tests/unit/verification_plane/test_checkers.py, tests/unit/verification_plane/test_constraint_gate.py, tests/unit/verification_plane/test_pipeline.py` |

## Missing / Skeleton Areas

The following files are still identified as skeleton placeholders:

- `src/nexus_orchestrator/__init__.py`
- `src/nexus_orchestrator/config/__init__.py`
- `src/nexus_orchestrator/config/loader.py`
- `src/nexus_orchestrator/config/schema.py`
- `src/nexus_orchestrator/constants.py`
- `src/nexus_orchestrator/control_plane/__init__.py`
- `src/nexus_orchestrator/control_plane/budgets.py`
- `src/nexus_orchestrator/control_plane/controller.py`
- `src/nexus_orchestrator/control_plane/feedback.py`
- `src/nexus_orchestrator/control_plane/scheduler.py`
- `src/nexus_orchestrator/domain/__init__.py`
- `src/nexus_orchestrator/domain/events.py`
- `src/nexus_orchestrator/domain/ids.py`
- `src/nexus_orchestrator/domain/models.py`
- `src/nexus_orchestrator/ext/README.md`
- `src/nexus_orchestrator/integration_plane/__init__.py`
- `src/nexus_orchestrator/integration_plane/conflict_resolution.py`
- `src/nexus_orchestrator/integration_plane/git_engine.py`
- `src/nexus_orchestrator/integration_plane/merge_queue.py`
- `src/nexus_orchestrator/integration_plane/workspace_manager.py`
- `src/nexus_orchestrator/knowledge_plane/__init__.py`
- `src/nexus_orchestrator/knowledge_plane/constraint_registry.py`
- `src/nexus_orchestrator/knowledge_plane/evidence_ledger.py`
- `src/nexus_orchestrator/knowledge_plane/failure_mining.py`
- `src/nexus_orchestrator/knowledge_plane/indexer.py`
- `src/nexus_orchestrator/knowledge_plane/personalization.py`
- `src/nexus_orchestrator/knowledge_plane/retrieval.py`
- `src/nexus_orchestrator/main.py`
- `src/nexus_orchestrator/observability/__init__.py`
- `src/nexus_orchestrator/observability/events.py`
- `src/nexus_orchestrator/observability/logging.py`
- `src/nexus_orchestrator/observability/metrics.py`
- `src/nexus_orchestrator/persistence/__init__.py`
- `src/nexus_orchestrator/persistence/repositories.py`
- `src/nexus_orchestrator/persistence/state_db.py`
- `src/nexus_orchestrator/planning/__init__.py`
- `src/nexus_orchestrator/planning/architect_interface.py`
- `src/nexus_orchestrator/planning/constraint_compiler.py`
- `src/nexus_orchestrator/planning/task_graph.py`
- `src/nexus_orchestrator/sandbox/__init__.py`
- `src/nexus_orchestrator/sandbox/network_policy.py`
- `src/nexus_orchestrator/sandbox/resource_governor.py`
- `src/nexus_orchestrator/sandbox/sandbox_manager.py`
- `src/nexus_orchestrator/sandbox/tool_provisioner.py`
- `src/nexus_orchestrator/security/__init__.py`
- `src/nexus_orchestrator/security/prompt_hygiene.py`
- `src/nexus_orchestrator/security/redaction.py`
- `src/nexus_orchestrator/spec_ingestion/__init__.py`
- `src/nexus_orchestrator/spec_ingestion/ingestor.py`
- `src/nexus_orchestrator/spec_ingestion/spec_map.py`
- `src/nexus_orchestrator/synthesis_plane/__init__.py`
- `src/nexus_orchestrator/synthesis_plane/context_assembler.py`
- `src/nexus_orchestrator/synthesis_plane/dispatch.py`
- `src/nexus_orchestrator/synthesis_plane/prompt_templates.py`
- `src/nexus_orchestrator/synthesis_plane/providers/__init__.py`
- `src/nexus_orchestrator/synthesis_plane/providers/anthropic_adapter.py`
- `src/nexus_orchestrator/synthesis_plane/providers/base.py`
- `src/nexus_orchestrator/synthesis_plane/providers/openai_adapter.py`
- `src/nexus_orchestrator/synthesis_plane/roles.py`
- `src/nexus_orchestrator/synthesis_plane/tools.py`
- `src/nexus_orchestrator/ui/__init__.py`
- `src/nexus_orchestrator/ui/cli.py`
- `src/nexus_orchestrator/ui/tui.py`
- `src/nexus_orchestrator/utils/__init__.py`
- `src/nexus_orchestrator/utils/concurrency.py`
- `src/nexus_orchestrator/utils/fs.py`
- `src/nexus_orchestrator/utils/hashing.py`
- `src/nexus_orchestrator/verification_plane/__init__.py`
- `src/nexus_orchestrator/verification_plane/adversarial/__init__.py`
- `src/nexus_orchestrator/verification_plane/adversarial/test_generator.py`
- `src/nexus_orchestrator/verification_plane/checkers/__init__.py`
- `src/nexus_orchestrator/verification_plane/checkers/base.py`
- `src/nexus_orchestrator/verification_plane/checkers/documentation_checker.py`
- `src/nexus_orchestrator/verification_plane/checkers/reliability_checker.py`
- `src/nexus_orchestrator/verification_plane/checkers/schema_checker.py`
- `src/nexus_orchestrator/verification_plane/checkers/scope_checker.py`
- `src/nexus_orchestrator/verification_plane/constraint_gate.py`
- `src/nexus_orchestrator/verification_plane/evidence.py`
- `src/nexus_orchestrator/verification_plane/pipeline.py`

## Risk Map

- Core modules: 13
- Security-sensitive modules: 13
- Correctness-critical modules: 31
