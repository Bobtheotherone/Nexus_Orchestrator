<!--
nexus-orchestrator — documentation skeleton

File: src/nexus_orchestrator/ext/README.md
Last updated: 2026-02-11

Purpose
- Explains extension points: provider adapters, checkers, UI plugins, constraint libraries.

What should be included in this file
- Stable plugin APIs and versioning rules.
- How plugins are discovered and loaded (config + entrypoints).
- Security considerations: plugins are code; treat as trusted only when reviewed.

Functional requirements
- Must document how to add a plugin without editing core modules.

Non-functional requirements
- Must keep plugin system simple for a single-user tool.

Suggested sections / outline
- Plugin types
- Discovery
- Versioning
- Security
-->

# Extension Points

## Plugin Types
- **Provider adapters:** New LLM backends (local models, alternative APIs)
- **Checkers:** New verification stages (custom linters, domain-specific validators)
- **Constraint libraries:** Reusable constraint sets for common patterns

## Discovery
Plugins are registered in config (orchestrator.toml) or via Python entry points.

## Security
Plugins are code — treat as trusted only when reviewed. Sandboxed execution applies.
