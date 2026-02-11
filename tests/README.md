<!--
nexus-orchestrator — documentation skeleton

File: tests/README.md
Last updated: 2026-02-11

Purpose
- Test strategy for the orchestrator repo.

What should be included in this file
- Unit test areas (domain models, planners, registry loader).
- Integration tests (sample spec end-to-end with mocked providers).
- Security tests (redaction, prompt hygiene).
- Performance/smoke tests (resource governor and queue behavior).

Functional requirements
- Must include offline tests for CI; provider calls must be mocked.

Non-functional requirements
- Tests must be deterministic; flaky tests are treated as defects.

Suggested sections / outline
- Unit
- Integration
- Golden runs
- Security
- Performance
-->

# Test Strategy

## Structure
```
tests/
  unit/           # Fast, isolated, no IO or external deps
    domain/       # Data model serialization, ID generation, events
    config/       # Schema validation, loader
    planning/     # Task graph, constraint compiler
    ...           # Mirrors src/ structure
  integration/    # Cross-module tests with real SQLite, real Git (local)
  smoke/          # End-to-end with mocked providers
  fixtures/       # Golden data, sample specs, mock transcripts
```

## Rules
- All tests are deterministic. Flaky tests are treated as defects.
- Provider API calls are ALWAYS mocked in CI.
- Unit tests have no IO side effects.
- Integration tests use temp directories and in-memory/temp SQLite.
- Smoke tests exercise the full pipeline with mocked providers.
