<!--
nexus-orchestrator — documentation

File: docs/prompts/templates/PERFORMANCE.md
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

# Performance Agent Prompt Template

## Inputs
- Work item summary (id, risk tier, scope)
- Performance constraints (latency, throughput, memory)
- Benchmark evidence from prior runs (if any)
- Patch diff (or candidate artifact set)

## Outputs (machine-parseable)
Return **YAML** with:
- `hot_paths`: suspected hot paths / algorithmic risks
- `benchmarks_to_add`: new benchmarks + thresholds
- `profiling_plan`: tools + steps to reproduce locally
- `perf_budget_updates`: recommended budgets (if justified)
- `approval`: `approve|request_changes|block_merge`

## Guidance
- Prefer deterministic microbenchmarks over flaky end-to-end timing.
- Tie thresholds to the stated hardware envelope (16 cores, 32GB RAM, SSD).
- Ensure Resource Governor backpressure behavior remains responsive under load.
