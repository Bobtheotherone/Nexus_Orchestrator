<!--
nexus-orchestrator — documentation skeleton

File: docs/schemas/CONSTRAINT_REGISTRY.md
Last updated: 2026-02-11

Purpose
- Schema/spec for constraints: how constraints are authored, versioned, and mapped to checkers.

What should be included in this file
- Constraint fields (id, severity, description, category, associated requirements, checker binding, parameters, exemptions).
- How constraints evolve and how to prevent regressions.
- How to represent project-specific vs reusable constraint libraries.

Functional requirements
- Must be specific enough to implement serialization + validation + registry queries.

Non-functional requirements
- Must support backward compatibility via schema versions.

Suggested sections / outline
- Constraint record
- Severity and overrides
- Binding to checkers
- Versioning
- Examples (non-code)
-->

# Constraint Registry Schema

Constraints are stored as YAML files in `constraints/registry/`.
See `docs/schemas/constraint_registry.schema.jsonc` for the machine-readable JSON Schema (JSONC).

## Constraint Record Fields

```yaml
- id: CON-SEC-0001
  severity: must          # must | should | may
  category: security      # structural | behavioral | performance | security | style | operational | documentation
  description: "No secrets in source files or log output"
  checker: security_checker
  parameters:
    patterns: ["API_KEY", "SECRET", "PASSWORD", "TOKEN"]
    exclude_paths: [".env.example"]
  requirement_links: [REQ-SEC-001]
  source: spec_derived    # spec_derived | failure_derived | manual
  created_at: "2026-02-11T00:00:00Z"
```

## Severity Levels

- **must:** Blocks merge. No exceptions without explicit override + justification + audit record.
- **should:** Generates warning. Override with justification in the work item.
- **may:** Advisory. Tracked but never blocking.

## Override Protocol

Overrides are rare and audited:
1. Agent or operator requests override with justification.
2. Override recorded in evidence ledger with constraint ID, justification, and expiry.
3. Overrides are time-bounded and re-audited at expiry.

## Evolution

New constraints are added by:
- Spec Ingestor (from requirements)
- Constraint Miner (from failures — the "never again" pipeline)
- Operator (manual additions)

All additions are versioned in Git alongside the code.