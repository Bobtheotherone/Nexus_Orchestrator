<!--
nexus-orchestrator — documentation

File: docs/prompts/templates/DOCUMENTATION.md
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

# Documentation Agent Prompt Template

## Inputs
- Work item summary and scope
- Public APIs/contracts changed (if any)
- Existing docs in docs/ and module READMEs

## Outputs (machine-parseable)
Return **YAML** with:
- `docs_to_update`: list of files and sections
- `new_docs`: proposed new docs files with purpose
- `examples`: runnable examples to include
- `approval`: `ready|needs_more_info|blocked`

## Guidance
- Keep docs oriented to *private operator workflows* (you).
- Prefer short runbooks with exact commands and troubleshooting steps.
