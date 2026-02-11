<!--
nexus-orchestrator — documentation skeleton

File: constraints/README.md
Last updated: 2026-02-11

Purpose
- Explains how constraints are authored, stored, versioned, and loaded by the orchestrator.

What should be included in this file
- Directory conventions: registry/, libraries/, checkers/ bindings.
- Naming conventions for constraint IDs.
- Override/exemption protocol (audited, rare).
- How constraint evolution adds new constraints from failures.

Functional requirements
- Must map directly to docs/schemas/CONSTRAINT_REGISTRY.md and the loader implementation.

Non-functional requirements
- Must keep constraints readable and reviewable (avoid over-parameterized complexity).

Suggested sections / outline
- Layout
- IDs
- Severities
- Overrides
- Evolution workflow
-->

# Constraints

## Layout
- `registry/` — Active constraint YAML files loaded by the orchestrator
- `libraries/` — Reusable constraint sets for common patterns (REST, DB, CLI, etc.)

## Constraint IDs
Format: `CON-<CATEGORY>-<NUMBER>` (e.g., `CON-SEC-0001`, `CON-STY-0002`)

Categories: SEC (security), STY (style), COR (correctness), ARC (architecture), PER (performance), REL (reliability), DOC (documentation), OPS (operational)

## Adding Constraints
1. Create or edit a YAML file in `registry/`
2. Follow the schema in `docs/schemas/CONSTRAINT_REGISTRY.md`
3. Bind to a checker in `verification_plane/checkers/`
4. Constraints from the "never again" pipeline are auto-generated with `source: failure_derived`
