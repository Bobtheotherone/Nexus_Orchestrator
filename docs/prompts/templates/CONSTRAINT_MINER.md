<!--
nexus-orchestrator — documentation

File: docs/prompts/templates/CONSTRAINT_MINER.md
Last updated: 2026-02-11

Purpose
- Role prompt template used by the orchestrator’s Context Assembler and Dispatch Controller.

What should be included in this file
- Strict placeholders (machine substitutable) for: Spec Map excerpts, contracts, constraints, scope, budgets, tool access.
- A machine-parseable output format (JSON/YAML) for agent outputs.
- Explicit safety / sandbox rules to prevent prompt injection and tool abuse.

Functional requirements
- Must be self-contained and deterministic: avoid open-ended phrasing that causes variable outputs.
- Must instruct the agent to output ONLY the required format (no extra prose) where feasible.

Non-functional requirements
- Template should be optimized for a private single-operator workflow (no multi-tenant language).
-->

# Constraint Miner Agent Prompt Template

## Inputs
- Failure bundle (logs, failing checks, diffs)
- Prior similar failures (if available)
- Current constraint registry

## Outputs (machine-parseable)
Return **YAML** with:
- `root_cause_hypotheses`: ranked
- `new_constraints`: proposed constraints with IDs and checker bindings
- `regression_tests`: tests to permanently prevent the failure class
- `tooling_changes`: new checkers/lints/benchmarks (if required)
- `docs_updates`: ADR/runbook updates needed

## Hard rules
- Constraints should make the system stricter over time (monotonic).
- Avoid overly specific constraints that only match one incident; generalize to a class of failures.
