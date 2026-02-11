<!--
nexus-orchestrator — documentation

File: docs/prompts/templates/ADVERSARIAL_TESTER.md
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

# Adversarial Test Generator Prompt Template

## Inputs
- Work item’s interface contracts + constraint envelope
- No implementation rationale (treat as black box)
- Existing test suite summary

## Outputs (machine-parseable)
Return **YAML** with:
- `tests_to_add`: list of adversarial tests (unit/integration/property)
- `oracles`: expected behaviors and invariants each test asserts
- `edge_cases`: input boundary sets and weird cases
- `failure_modes`: concurrency/timeout/network/tooling failure scenarios

## Guidance
- Assume the implementer missed edge cases.
- Prefer tests that are deterministic and fast.
- Target error paths, boundary conditions, concurrency races, and malformed inputs.
