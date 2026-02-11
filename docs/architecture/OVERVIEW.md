<!--
nexus-orchestrator — documentation skeleton

File: docs/architecture/OVERVIEW.md
Last updated: 2026-02-11

Purpose
- High-level architecture overview that complements design_document.md with repo-specific details (paths, module boundaries, and diagrams).

What should be included in this file
- A clear explanation of the 5 planes and their responsibilities.
- A request/response-style walkthrough of a single work item lifecycle.
- Data flow: Spec -> SpecMap -> TaskGraph -> Dispatch -> Patch -> Verify -> Evidence -> Merge.
- Explicit serialized points (merge queue) vs parallel points (agent dispatch).

Functional requirements
- Must define the orchestrator’s 'critical invariants' in one place (no merge without evidence, scope enforcement, etc.).

Non-functional requirements
- Must be stable enough to serve as onboarding for new agents.

Suggested sections / outline
- Planes
- Key invariants
- Lifecycle walkthrough
- Parallelism model
- Where the bottlenecks are
-->

# Architecture Overview

## The Five Planes

1. **Control Plane** (`control_plane/`): The brain — scheduling, budgets, feedback, run lifecycle
2. **Synthesis Plane** (`synthesis_plane/`): The swarm — agent dispatch, context assembly, provider routing
3. **Verification Plane** (`verification_plane/`): The gate — checkers, evidence, adversarial testing
4. **Integration Plane** (`integration_plane/`): The gatekeeper — Git operations, merge queue, workspaces
5. **Knowledge Plane** (`knowledge_plane/`): The memory — indexing, retrieval, constraints, failure mining

## Critical Invariants

These invariants hold at all times and are never violated:

1. **No merge without evidence.** Every artifact entering integration has a complete evidence record.
2. **Scope enforcement.** No agent modifies files outside its declared scope.
3. **Serialized integration.** Only one merge is in flight at a time.
4. **Constraint completeness.** Every work item has a complete constraint envelope before dispatch.
5. **Non-overlapping ownership.** Each file is owned by exactly one work item at a time.
6. **Graceful degradation.** Resource exhaustion causes throttling, never crashes.

## Work Item Lifecycle

```
Spec → [Ingest] → SpecMap → [Plan] → TaskGraph → [Schedule] → Dispatch
  ↓
Agent receives: constraint envelope + contracts + context
  ↓
Agent produces: patch + tests + rationale
  ↓
[Self-verify] → [Constraint Gate] → pass? → [Merge Queue] → integration
                      ↓ fail
              [Feedback Synthesizer] → agent retries (up to budget)
                      ↓ budget exhausted
              [Escalate: other provider → stronger model → operator]
```

## Parallelism Model

- **Horizontal:** Independent work items run on separate agents simultaneously (API-bound)
- **Pipeline:** While tier-N items are implementing, tier-(N-1) is verifying, tier-(N+1) is planning
- **Speculative:** Critical-path items may dispatch to multiple agents; first passing wins

## Where the Bottlenecks Are

1. **API rate limits** — bound agent parallelism (50-200 concurrent sessions typical)
2. **Verification pool** — local CPU/RAM bound (3 heavy + 8 light concurrent jobs)
3. **Merge queue** — serialized by design (one at a time)
4. **Context assembly** — can become slow without caching on large repos
