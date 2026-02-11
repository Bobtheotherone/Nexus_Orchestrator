<!--
nexus-orchestrator — documentation skeleton

File: samples/README.md
Last updated: 2026-02-11

Purpose
- Holds sample input design documents and fixtures for smoke tests (spec ingestion, planning, and minimal orchestration).

What should be included in this file
- A tiny design doc that produces a small DAG of work items.
- Expected Spec Map output fixtures (serialized) once implemented.
- Golden run bundles for tests (no provider calls; use mocked providers).

Functional requirements
- Must allow running end-to-end demo offline for CI.

Non-functional requirements
- Keep samples small and stable; version them as contracts.

Suggested sections / outline
- Sample design docs
- Fixtures
- Golden runs
-->

# Samples

## Purpose
Sample design documents and test fixtures for smoke testing the orchestrator.

## Files
- `specs/minimal_design_doc.md` — Tiny spec producing 3 work items (A->B->C). Used by smoke tests.

## Golden Runs
Once implemented, this directory will also contain:
- Expected SpecMap JSON fixtures
- Expected TaskGraph fixtures
- Mocked provider response transcripts
- Expected evidence record structures
