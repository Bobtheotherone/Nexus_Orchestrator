<!--
nexus-orchestrator — documentation skeleton

File: docs/prompts/ROLE_SYSTEM.md
Last updated: 2026-02-11

Purpose
- Defines agent roles (Architect, Implementer, Reviewer, Security, Performance, Toolsmith, Integrator, Constraint Miner, Doc agent) and their tool permissions.

What should be included in this file
- Role definitions: goals, allowed tools/capabilities, budgets, success criteria.
- Default routing policy (Codex vs Claude class) and escalation ladder.
- Prompt injection defenses and context hygiene rules for role prompts.

Functional requirements
- Must align with synthesis_plane/roles.py and provider routing implementation.

Non-functional requirements
- Must avoid embedding secrets; prompts should be deterministic templates with placeholders.

Suggested sections / outline
- Roles
- Budgets
- Tool permissions
- Routing & escalation
- Context hygiene
-->

# Agent Role System

## Role Definitions

| Role | Primary Model | Purpose | Required For |
|---|---|---|---|
| Architect | Claude (max context) | Decomposition, contracts, ADRs | Phase 1 planning |
| Implementer | Codex (fast) → Claude (fallback) | Write code within scope | All code work items |
| Test Engineer | Codex/Claude | Unit, integration, property tests | Test work items |
| Reviewer | Claude | Adversarial review, edge cases | High/critical risk items |
| Security | Claude | Threat surface, secure defaults | Security-tagged items |
| Performance | Codex/Claude | Benchmarks, profiling guidance | Perf-constrained items |
| Toolsmith | Codex | Tool integration, CI updates | Infrastructure items |
| Integrator | Claude | Conflict resolution, coherence | Non-trivial conflicts |
| Constraint Miner | Claude | Turn failures into new constraints | Post-failure analysis |
| Documentation | Codex/Claude | API docs, guides, examples | Doc-tagged items |

## Prompt Templates

Role prompt templates live in `docs/prompts/templates/`. The orchestrator should treat these as **versioned, auditable artifacts**:
- The Context Assembler fills placeholders (Spec Map excerpts, contracts, constraints, scope, budgets, tool access).
- The Dispatch Controller records a **prompt hash** in the evidence ledger for reproducibility.
- Templates must instruct agents to emit **machine-parseable outputs** whenever feasible.

Expected templates (seed set):
- `ARCHITECT.md`
- `IMPLEMENTER.md`
- `REVIEWER.md`
- `SECURITY.md`
- `PERFORMANCE.md`
- `TOOLSMITH.md`
- `INTEGRATOR.md`
- `ADVERSARIAL_TESTER.md`
- `CONSTRAINT_MINER.md`
- `DOCUMENTATION.md`

## Escalation Ladder

1. Codex (fast, cheap) — 2 attempts
2. Claude Sonnet (balanced) — 2 attempts
3. Claude Opus (max capability) — 1 attempt
4. Operator review (full diagnostic package)

## Prompt Hygiene

- Role prompts are templates with strict placeholders (no dynamic system prompt injection)
- Untrusted content (repo files, spec text) is delimited and quoted
- Agents are instructed to treat repository content as data, not instructions
- Prompt hashes are recorded in evidence for reproducibility
