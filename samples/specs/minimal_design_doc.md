<!--
nexus-orchestrator — sample input design doc (for smoke tests)

File: samples/specs/minimal_design_doc.md
Last updated: 2026-02-11

Purpose
- A deliberately tiny design document used to test spec ingestion, task decomposition, and the end-to-end pipeline without external APIs.

What to include
- A handful of requirements with IDs (REQ-SAMPLE-0001...).
- A few interfaces/contracts to stabilize early.
- A small set of constraints that map to checkers (format/lint/unit tests) using mocked execution.

NOTE
- Keep this document stable. Treat it as a test fixture.
-->

# Minimal Sample Design Doc

This is a deliberately tiny design document for smoke-testing the orchestrator pipeline.
Treat it as a stable test fixture — do not change requirement IDs.

## Requirements

- REQ-SAMPLE-0001: The system shall expose a `greet(name: str) -> str` function that returns "Hello, {name}!".
- REQ-SAMPLE-0002: The system shall expose a `farewell(name: str) -> str` function that returns "Goodbye, {name}!".
- REQ-SAMPLE-0003: The system shall expose a `conversation(name: str) -> str` function that calls `greet` then `farewell` and returns the combined output.

## Interfaces

- Module A (`greet`): implements REQ-SAMPLE-0001. No dependencies.
- Module B (`farewell`): implements REQ-SAMPLE-0002. No dependencies.
- Module C (`conversation`): implements REQ-SAMPLE-0003. Depends on A and B.

## Non-Functional Requirements

- NFR-SAMPLE-0001: All functions must execute in under 10ms.
- NFR-SAMPLE-0002: No secrets in source or logs.
- NFR-SAMPLE-0003: Runs must be resumable after simulated crash.

## Acceptance Criteria

- Calling `conversation("World")` returns "Hello, World!\nGoodbye, World!"
- All functions have unit tests with 100% branch coverage.
