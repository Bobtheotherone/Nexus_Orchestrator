<!--
nexus-orchestrator — documentation skeleton

File: constraints/libraries/README.md
Last updated: 2026-02-11

Purpose
- Reusable constraint libraries (e.g., REST API, DB access, CLI, security baselines) that can be applied to projects.

What should be included in this file
- How libraries are structured and versioned.
- How the planner selects libraries based on detected stack.

Functional requirements
- Must be loadable independently and composable with project constraints.

Non-functional requirements
- Avoid overfitting; keep libraries generic and well-tested.

Suggested sections / outline
- Layout
- Versioning
- Selection
- Examples
-->

# Reusable Constraint Libraries

Constraint libraries provide pre-validated constraint sets for common patterns.
The Architect Agent references these during decomposition to avoid deriving constraints from scratch.

## Available Libraries (add as needed)
- `rest_api.yaml` — constraints for REST API endpoints (validation, error codes, auth)
- `database.yaml` — constraints for DB access (connection pooling, migration safety, injection prevention)
- `cli_tool.yaml` — constraints for CLI tools (exit codes, help text, signal handling)
- `security_baseline.yaml` — baseline security constraints for any project

## Selection
The planner selects applicable libraries based on detected technology stack and Spec Map tags.
