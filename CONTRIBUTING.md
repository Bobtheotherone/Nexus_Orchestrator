<!--
nexus-orchestrator — documentation skeleton

File: CONTRIBUTING.md
Last updated: 2026-02-11

Purpose
- Contributor/agent guide: repo conventions, how to add constraints, how to add checkers, how to propose ADRs, and how to keep changes mergeable.

What should be included in this file
- Repo philosophy: constraint-first, evidence-led, Git-native isolation.
- How to run local verification suites and generate evidence artifacts.
- How to add/modify constraints (files in constraints/registry/ + docs).
- How to add a new checker plugin (verification_plane/checkers + docs/schemas).
- How to add a new provider adapter (synthesis_plane/providers + config).
- How to write ADRs (docs/adrs/) and when they are required.
- Coding standards and security rules (no secrets, supply-chain policy).
- Definition of Done for a work item implementation in this repo itself.

Functional requirements
- Must give step-by-step instructions an agent can follow without needing tribal knowledge.
- Must define the required evidence artifacts for changes to critical subsystems (security, sandbox, merge queue).

Non-functional requirements
- Must be concise but unambiguous; prefer checklists.
- Must be safe-by-default (explicitly forbid unsafe shortcuts like disabling gates).

Suggested sections / outline
- Local dev setup
- Repo layout (links to docs/FILE_MAP.md)
- Constraint changes
- Checker changes
- Provider adapter changes
- ADR workflow
- Security & supply chain rules
- Definition of Done
-->

# Contributing

This is a personal orchestrator. These conventions apply to both human and AI agent contributions.

## Repo Philosophy

- **Constraint-first:** Define what "correct" means before writing code.
- **Evidence-led:** Every merge carries machine-checkable proof.
- **Git-native isolation:** Work on isolated branches, merge through the queue.

## Local Dev Setup

1. `pip install -e ".[dev,providers]"` (or `make install`)
2. Copy `.env.example` to `.env` and add API keys
3. Run `make test` to verify setup

## How to Add/Modify Constraints

1. Create or edit a YAML file in `constraints/registry/`
2. Follow the schema in `docs/schemas/CONSTRAINT_REGISTRY.md`
3. Add corresponding checker binding in `verification_plane/checkers/`
4. Test: `make test-unit`

## How to Add a Provider Adapter

1. Create `src/nexus_orchestrator/synthesis_plane/providers/new_adapter.py`
2. Implement the abstract interface from `providers/base.py`
3. Register in `providers/__init__.py`
4. Add config section to `orchestrator.toml`
5. Test with mocked API responses

## How to Write ADRs

1. Copy `docs/adrs/000_TEMPLATE.md`
2. Number sequentially (001, 002, ...)
3. Required for: schema changes, new planes, security boundary changes

## Definition of Done

A change is merge-eligible when:
- All must-severity constraints pass with evidence
- New public functions have tests
- No files outside declared scope are modified
- Documentation updated if public API changed
- No secrets in code, logs, or prompts
