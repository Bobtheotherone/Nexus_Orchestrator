# ADR-001: Canonical Provider Interface and Dispatch Migration

## Status
Accepted

## Context
The synthesis plane had two overlapping provider contracts:
- Legacy dispatch-local `ProviderRequest`/`ProviderResponse`/`ProviderCallError`.
- Canonical `synthesis_plane/providers/base.py` interface with `send()` and `ProviderError` taxonomy.

This split created drift in request fields, error handling, and adapter behavior.

## Decision
Use `synthesis_plane/providers/base.py` as the single canonical provider contract:
- Dispatch consumes canonical `ProviderRequest` and `ProviderResponse`.
- Dispatch classifies provider failures using canonical `ProviderError` types.
- Canonical invocation path is `provider.send(request)`.
- Legacy `generate(request)` is retained only as a compatibility shim with deprecation signaling.

## Consequences
- Positive: one request/response schema and one error taxonomy across adapters and dispatch.
- Positive: easier capability evolution (structured outputs, tool configs) without duplicate models.
- Negative: tests and mocks using `generate()` require migration or shim-aware assertions.

## Alternatives Considered
- Keep dual stacks and map between them at runtime.
  Rejected due to continued drift risk and duplicated validation logic.

## Links
- Related modules: `src/nexus_orchestrator/synthesis_plane/dispatch.py`
- Related modules: `src/nexus_orchestrator/synthesis_plane/providers/base.py`
