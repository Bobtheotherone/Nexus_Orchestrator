<!--
nexus-orchestrator — documentation skeleton

File: tests/fixtures/README.md
Last updated: 2026-02-11

Purpose
- Holds test fixtures like golden SpecMap JSON, sample constraint registries, and fake provider transcripts.

What should be included in this file
- Versioned fixtures and how to update them safely.
- Redaction policy for transcripts (no secrets).

Functional requirements
- Must keep fixtures small and stable.

Non-functional requirements
- Avoid embedding large binaries.

Suggested sections / outline
- Spec fixtures
- Registry fixtures
- Provider fixtures
-->

# Test Fixtures

## Contents (add as implemented)
- `sample_spec_map.json` — Expected SpecMap from `minimal_design_doc.md`
- `sample_task_graph.json` — Expected TaskGraph from sample spec
- `sample_constraint_envelope.json` — Example constraint envelope
- `mock_provider_responses/` — Canned LLM responses for smoke tests
- `sample_evidence/` — Example evidence records and artifacts

## Rules
- Fixtures are versioned and stable. Changing a fixture is a breaking change.
- No secrets in fixtures. Use fake/redacted data.
