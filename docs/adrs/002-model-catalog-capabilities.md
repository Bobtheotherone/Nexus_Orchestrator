# ADR-002: File-Backed Model Capability Catalog

## Status
Accepted

## Context
Model capabilities (context limits, tool support, cost assumptions, reasoning controls) were hardcoded in provider adapters and role routing defaults, making updates risky and non-auditable.

## Decision
Introduce a repository-shipped, file-backed `ModelCatalog` as the single capability source:
- Catalog file includes explicit metadata: `version`, `last_updated`.
- Per-model records include:
  - `max_context_tokens`
  - `supports_tool_calling`
  - `supports_structured_outputs`
  - `reasoning_effort_allowed`
  - cost fields used for deterministic estimation
  - `availability` and `notes`
- Routing resolves capability profiles from config and catalog instead of hardcoded model IDs.
- Provider adapters query the catalog for capability and cost decisions.

## Consequences
- Positive: capability/cost updates can be audited and changed without adapter code edits.
- Positive: deterministic, offline-safe defaults remain in-repo.
- Negative: unknown models now require explicit catalog entries or explicit fallback policy.

## Alternatives Considered
- Keep adapter-local heuristics (prefix matching by model name).
  Rejected because behavior is implicit, difficult to audit, and prone to silent drift.

## Links
- Related modules: `src/nexus_orchestrator/synthesis_plane/model_catalog.py`
- Related config: `orchestrator.toml`
- Related modules: `src/nexus_orchestrator/synthesis_plane/providers/openai_adapter.py`
- Related modules: `src/nexus_orchestrator/synthesis_plane/providers/anthropic_adapter.py`
