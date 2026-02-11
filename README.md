<!--
nexus-orchestrator — documentation skeleton

File: README.md
Last updated: 2026-02-11

Purpose
- Top-level orientation for humans and for agentic AIs. Explains what NEXUS is, how to run it locally, and where to start implementing.

What should be included in this file
- One-paragraph mission statement aligned to the design document (constraint-driven, Git-native, local-first).
- Quick-start instructions (install deps, configure providers, run a demo orchestration).
- Conceptual architecture map: planes/components and where to find them in the repo (link to docs/FILE_MAP.md).
- Operating modes (greenfield/brownfield/hardening/exploration) and when to use each.
- How constraints, evidence, and the merge queue work in practice.
- Security posture + sandboxing overview and how to configure strictness.
- Links to core docs: docs/design_document.md, docs/architecture/*.md, docs/runbooks/*.md, docs/schemas/*.md.

Functional requirements
- Must make it obvious how an agent should proceed file-by-file to implement the orchestrator (point to docs/BUILD_ORDER.md).
- Must define the expected user-facing workflows: ingest spec -> plan -> dispatch -> verify -> integrate -> audit.
- Must define the minimal 'hello world' demo scenario (tiny repo, few work items) used by tests and smoke runs.

Non-functional requirements
- Must remain accurate as the repo evolves (keep links/paths updated).
- Must avoid provider-specific marketing language; treat providers as adapters.
- Must be readable in under ~10 minutes; deeper details live in docs/.

Suggested sections / outline
- What this repo is
- Core ideas (constraints, evidence, Git isolation)
- Local-first architecture and hardware envelope
- Quick start (dev) — minimal demo
- How the orchestrator works (high level)
- Docs index
- Security + safety
- Contributing and conventions
-->

# NEXUS — Constraint-Driven Agentic LLM Orchestrator

A personal, local-first orchestrator that takes a single design document and produces a high-quality, modular, production-grade codebase by coordinating hundreds of parallel LLM agents through constraint-based program synthesis.

## Core Ideas

- **Constraints and Evidence:** Nothing merges without machine-checkable proof of correctness.
- **Massive Parallelism:** Hundreds of remote LLM agents work concurrently on isolated Git branches.
- **Agent Resourcefulness:** Agents install tools, fetch docs, try unconventional approaches — the constraint gate is the only judge.
- **Never Again:** Every failure becomes a permanent new constraint, test, or rule.
- **Single Machine:** Runs on one workstation (RTX 5090 / 32GB RAM / 16 cores / 1TB SSD).

## Quick Start

```bash
# Install
pip install -e ".[dev,providers]"

# Configure providers
cp .env.example .env
# Edit .env with your API keys

# Run smoke test (mocked providers, no API calls)
make test-smoke

# Plan from a spec
nexus plan samples/specs/minimal_design_doc.md

# Run full orchestration (mocked)
nexus run --mock
```

## How It Works

1. **Ingest** a design document → extract requirements, constraints, interfaces
2. **Plan** → decompose into work items with constraint envelopes, build a task graph
3. **Dispatch** → send work items to LLM agents (Codex, Claude) in parallel
4. **Verify** → run every artifact through the constraint gate, collect evidence
5. **Integrate** → merge passing artifacts through a serialized queue into the repo
6. **Evolve** → mine failures into new constraints that prevent recurrence

## Docs Index

- [Design Document](design_document.md) — canonical architecture spec
- [File Map](docs/FILE_MAP.md) — repo layout and component locations
- [Build Order](docs/BUILD_ORDER.md) — implementation sequence for agentic AI
- [Architecture Overview](docs/architecture/OVERVIEW.md) — planes, lifecycle, invariants
- [Data Model](docs/architecture/DATA_MODEL.md) — entities, IDs, relationships
- [Git Protocol](docs/architecture/GIT_PROTOCOL.md) — branches, merge queue, ownership
- [Verification Pipeline](docs/architecture/VERIFICATION_PIPELINE.md) — stages, evidence, adversarial
- [Config Schema](docs/schemas/CONFIG.md) — orchestrator.toml reference
- [Local Usage Runbook](docs/runbooks/LOCAL_USAGE.md) — step-by-step operation
- [Threat Model](docs/threat_model.md) — security boundaries and controls
