<!--
nexus-orchestrator — documentation skeleton

File: docs/architecture/GIT_PROTOCOL.md
Last updated: 2026-02-11

Purpose
- Normative spec for Git workflows: branches, scopes, merge queue, conflict resolution, rollback, and metadata.

What should be included in this file
- Branch naming conventions (main/integration/contract/*/work/*/verify/*).
- Scope and file ownership rules and the exception protocol.
- Merge queue algorithm and required checks for each risk tier.
- Rollback and bisect policy when integration fails.

Functional requirements
- Must define machine-checkable rules for scope violations (what is blocked vs escalated).

Non-functional requirements
- Must be deterministic and auditable.

Suggested sections / outline
- Branches
- Scopes & ownership
- Merge queue
- Conflict resolution
- Rollback
- Audit metadata
-->

# Git Coordination Protocol

## Branch Architecture

| Branch Pattern | Purpose | Mutability |
|---|---|---|
| `main` | Production-ready code only | Merge-only via release |
| `integration` | Staging — verified work items merge here | Merge-only via queue |
| `contract/<module>` | Stabilized interface contracts | Immutable during implementation |
| `work/<unit-id>` | Agent workspace for a work item | Agent-writable, ephemeral |
| `verify/<unit-id>` | Adversarial test suites | Generator-writable |

## File Ownership Rules

- Each source file is owned by exactly one work item at a time.
- Ownership is declared in the work item scope and enforced by the Git Engine.
- Scope violations are **blocked** (not warned) by default.
- Shared code (utils, types, config) is owned by dedicated infrastructure work items completed in Phase 2.

## Merge Queue Algorithm

1. Candidate enters queue when all constraints pass and evidence is attached.
2. Queue orders by: dependency satisfaction → risk tier (low first) → arrival time.
3. For each candidate:
   a. Rebase onto current integration HEAD.
   b. Run compositional checks (compile, integration tests, architectural constraints).
   c. If pass: fast-forward merge, record MergeRecord, unblock dependents.
   d. If fail: revert, dispatch to Feedback Synthesizer, re-enter dispatch loop.
4. Only one merge is in flight at a time (serialized).

## Conflict Resolution

- **Trivial** (import ordering, formatting): auto-resolved per policy in config.
- **Non-trivial**: dispatched to Integrator Agent with both work items' context.
- **Contract-level**: triggers re-planning cycle via Architect Agent.

## Rollback Protocol

- If compositional check fails post-merge: revert the merge commit immediately.
- If multiple recent merges are suspect: bisect to isolate the offending change.
- Reverted work items re-enter the dispatch queue with additional compositional context.

## Audit Metadata

Every merge commit includes trailer metadata:
```
NEXUS-WorkItem: <work-item-id>
NEXUS-Evidence: <evidence-id-1>,<evidence-id-2>,...
NEXUS-Agent: <provider>/<model>/<role>
NEXUS-Iteration: <attempt-number>
```
