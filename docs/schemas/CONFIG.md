<!--
nexus-orchestrator — documentation skeleton

File: docs/schemas/CONFIG.md
Last updated: 2026-02-11

Purpose
- Config schema documentation: keys, types, defaults, and examples (non-code).

What should be included in this file
- Config sections: providers, routing, budgets, resource governor, sandbox, git, observability, UI.
- Versioning and migration guidance.

Functional requirements
- Must match config/schema.py validation rules.

Non-functional requirements
- Avoid provider-specific coupling; keep adapters abstract.

Suggested sections / outline
- Schema version
- Sections
- Defaults
- Profiles
- Migration
-->

# Config Schema — orchestrator.toml Reference

See `orchestrator.toml` for the live config with comments.
See `docs/schemas/config.schema.jsonc` for the machine-readable JSON Schema (JSONC).
See `src/nexus_orchestrator/config/schema.py` for validation rules.

## Sections

| Section | Purpose |
|---|---|
| `[meta]` | Schema version for migration |
| `[providers]` | API provider routing, keys (env refs only), model selection |
| `[resources]` | CPU/RAM/disk/GPU allocation and headroom targets |
| `[budgets]` | Per-work-item and per-run token/cost/iteration limits |
| `[git]` | Branch naming, merge strategy, auto-resolve policy |
| `[sandbox]` | Container backend, network policy, tool approval |
| `[paths]` | Workspace, evidence, state, cache directories |
| `[observability]` | Log level/format, evidence retention, redaction |
| `[profiles.*]` | Named override sets activated with `--profile` |

## Config Precedence (highest wins)

1. CLI flags (`--budget-max-iterations 3`)
2. Environment variables (`NEXUS_BUDGETS_MAX_ITERATIONS=3`)
3. `orchestrator.toml`
4. Built-in defaults
