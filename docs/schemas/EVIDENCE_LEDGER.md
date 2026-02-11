<!--
nexus-orchestrator — documentation skeleton

File: docs/schemas/EVIDENCE_LEDGER.md
Last updated: 2026-02-11

Purpose
- Schema/spec for evidence artifacts and how they link to constraints/work items/commits.

What should be included in this file
- Evidence types (logs, reports, coverage, benchmarks, SBOM, dependency audit).
- Required metadata: run ID, timestamps, tool versions, hashes, environment, file paths.
- Retention policy guidance and redaction rules.

Functional requirements
- Must enable answering: requirement -> constraint -> evidence -> commit.

Non-functional requirements
- Must be append-only and tamper-evident where feasible.

Suggested sections / outline
- Evidence record
- Linkage
- Storage
- Redaction
- Retention
-->

# Evidence Ledger Schema

## Evidence Record

See `docs/architecture/DATA_MODEL.md` for the full field list.
See `docs/schemas/evidence_ledger.schema.jsonc` for the machine-readable JSON Schema (JSONC).

## Linkage (Traceability Chain)

```
Requirement (REQ-0001)
  → Constraint (CON-COR-0001)
    → Checker (test_checker)
      → EvidenceRecord (evidence/run-001/wi-abc/test_results.json)
        → MergeRecord (commit abc123)
```

## Storage Layout

```
evidence/
  <run-id>/
    <work-item-id>/
      <stage>/
        result.json       # EvidenceRecord metadata
        output.log        # Raw checker output
        coverage.json     # Coverage data (if applicable)
```

## Retention Policy

- **Full records:** Keep last N runs (configurable, default 5).
- **Summaries:** Keep forever (pass/fail, duration, constraint coverage).
- **Failure artifacts:** Retain for 90 days (longer than success artifacts).
- **Redaction:** Applied before storage. Secrets never appear in evidence.

## Integrity

- Every evidence file is hashed (SHA-256) at creation time.
- Hashes stored in the EvidenceRecord metadata.
- Integrity verified on export and on traceability queries.