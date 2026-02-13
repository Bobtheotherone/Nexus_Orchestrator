<!--
nexus-orchestrator — documentation skeleton

File: docs/runbooks/LOCAL_USAGE.md
Last updated: 2026-02-11

Purpose
- Step-by-step runbook for running the orchestrator on a local workstation.

What should be included in this file
- Pre-reqs (Python, container runtime, Git).
- Initial config: providers, budgets, sandbox strictness, paths.
- Running a demo: ingest sample spec, plan, dispatch, verify, integrate.
- How to inspect state, evidence, logs, and dashboards (TUI).
- Common failure triage and how to recover.

Functional requirements
- Must define how to run in offline mode (no providers) for smoke tests.

Non-functional requirements
- Must be copy/paste friendly when code exists; for now, descriptive.

Suggested sections / outline
- Prerequisites
- Configure
- Run a demo
- Inspect
- Troubleshooting
- Reset/cleanup
-->

# Local Usage Runbook

## Prerequisites

- Python 3.11+
- Docker or Podman (for sandboxed tool execution)
- Git 2.30+
- 32 GB RAM, 100 GB free disk space

## Configure

1. `cp .env.example .env` and add your API keys (optional for mock mode)
2. Review `orchestrator.toml` — adjust resource limits, sandbox policy, budgets
3. Review `profiles/operator_profile.toml` — set your private preferences/house rules (no secrets)
4. For strict mode: `nexus run --profile strict`

## Run a Demo (Mocked, No API Calls)

```bash
make test-smoke
```

This ingests `samples/specs/minimal_design_doc.md`, creates 3 work items, dispatches to mocked providers, runs mocked verification, and merges into a test repo.

## Run for Real

```bash
# Plan only (no dispatch)
nexus plan path/to/your/design_document.md

# Full run
nexus run path/to/your/design_document.md

# With exploration mode
nexus run --profile exploration path/to/your/design_document.md
```

## Inspect

```bash
nexus status              # Current run state
nexus inspect <work-item-id>  # Details for a specific work item
nexus export              # Produce audit bundle
```

## Troubleshooting

- **OOM / swap:** Reduce `max_heavy_verification` in orchestrator.toml
- **API rate limits:** Reduce `max_concurrent` for the affected provider
- **Disk full:** Run `make clean` or reduce evidence retention
- **Stuck work item:** Check `nexus inspect <id>` for feedback; may need manual constraint override

## Phase 4 Audit Snippet (Pipefail Safe)

Use this pattern when `set -euo pipefail` is enabled and `rg` returning exit code `1` means
"no matches" (not a command failure):

```bash
set -euo pipefail

# Fail only when the producer command fails; tolerate 'no matches' from rg.
git log --format='%H%n%B%n---' -n 20 \
  | rg 'NEXUS-(WorkItem|Evidence|Agent|Iteration):' \
  || { code=$?; [ "$code" -eq 1 ] && echo "no trailer matches"; exit 0; }

# Placeholder scan that does not fail when there are zero matches.
rg -n --pcre2 'TODO|FIXME|NotImplementedError|^[[:space:]]*pass[[:space:]]*$' \
  src/nexus_orchestrator/integration_plane \
  tests/unit/integration_plane \
  tests/integration/test_workspace_lifecycle.py \
  || { code=$?; [ "$code" -eq 1 ] && echo "no placeholders found"; exit 0; }
```
