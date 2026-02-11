<!--
nexus-orchestrator — documentation skeleton

File: docs/runbooks/OPERATIONS.md
Last updated: 2026-02-11

Purpose
- Operational guidance: resource tuning, backpressure, caches, and managing long runs.

What should be included in this file
- Resource governor knobs and recommended defaults for the given hardware envelope.
- How to tune verification concurrency vs agent dispatch concurrency.
- Disk cleanup policies and evidence retention settings.
- Cost controls: budgets, speculative execution toggles.

Functional requirements
- Must describe how to safely stop/resume runs without corrupting state.

Non-functional requirements
- Must prioritize not destabilizing the workstation.

Suggested sections / outline
- Resource governor
- Caches
- Backpressure
- Cost controls
- Stop/resume
- Monitoring
-->

# Operations Runbook

## Resource Governor Tuning

The governor monitors CPU, RAM, and disk continuously. Default thresholds:
- RAM headroom: 6 GB free minimum (prevents swap)
- Disk headroom: 100 GB free minimum (prevents build cache disasters)
- Heavy verification concurrency: 3 (build + full test)
- Light verification concurrency: 8 (lint, format, typecheck)

Adjust in `orchestrator.toml [resources]`.

## Backpressure Cascade

When resources are constrained, degradation is ordered:
1. Disable speculative execution
2. Reduce verification concurrency
3. Throttle dispatch rate
4. Shrink caches
5. Trigger aggressive GC

## Stop / Resume

- `Ctrl+C` during a run triggers graceful shutdown
- In-progress API calls complete or timeout
- State is persisted to SQLite
- `nexus run --resume` picks up where it left off
- Work items marked `dispatched` at crash time are re-evaluated

## Cost Monitoring

```bash
nexus status --costs    # Show cost breakdown by provider, module, role
```

Set cost ceiling in config: `budgets.max_total_cost_usd`
