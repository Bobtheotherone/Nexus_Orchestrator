<!--
nexus-orchestrator — documentation skeleton

File: docs/architecture/VERIFICATION_PIPELINE.md
Last updated: 2026-02-11

Purpose
- Normative spec for the constraint gate: stages, evidence artifacts, incremental vs full verification, and risk-tier policy.

What should be included in this file
- Stage taxonomy (build, lint, typecheck, unit, integration, security scan, perf).
- Evidence format and required metadata (tool versions, hashes, environment).
- Adversarial tests: when and how they run; separation of concerns.
- Flaky test detection and quarantine protocol.

Functional requirements
- Must define how checkers are registered and how constraints map to checkers.

Non-functional requirements
- Must be reproducible; avoid nondeterminism or record it explicitly as evidence.

Suggested sections / outline
- Pipeline stages
- Evidence
- Risk tiers
- Adversarial
- Flakes
- Performance budgets
-->

# Verification Pipeline

## Pipeline Stages (Execution Order)

| Stage | Checker | Blocks Merge | Parallelizable |
|---|---|---|---|
| 1. Build | `build_checker` | Yes | No (prerequisite) |
| 2. Lint/Format | `lint_checker` | Yes (must) | Yes (after build) |
| 3. Type Check | `typecheck_checker` | Yes (must) | Yes (after build) |
| 4. Unit Tests | `test_checker` (unit) | Yes | Yes (after build) |
| 5. Security Scan | `security_checker` | Yes (must) | Yes (after build) |
| 6. Integration Tests | `test_checker` (integration) | Yes (high risk) | After merge-sim |
| 7. Performance | `performance_checker` | If envelope has perf constraints | After tests pass |
| 8. Adversarial Tests | `adversarial/test_generator` | Yes (high risk items) | Independent |

## Evidence Format

Every checker produces an `EvidenceRecord` (see DATA_MODEL.md) with:
- Tool name and exact version used
- Environment hash (OS, language runtime, key dependency versions)
- Full output artifact paths (logs, reports, coverage)
- Duration and resource usage

## Risk Tier Policy

| Risk Tier | Required Stages | Adversarial Required | Reviewer Required |
|---|---|---|---|
| Low | 1-5 | No | No |
| Medium | 1-6 | No | No |
| High | 1-7 | Yes | Yes |
| Critical | 1-8 | Yes | Yes + Security Agent |

## Flaky Test Protocol

1. Detection: test passes and fails across repeated runs on same input.
2. Quarantine: remove from blocking constraint set immediately.
3. Dispatch: high-priority work item to Test Engineer Agent for stabilization.
4. Constraint: add reproduction environment capture requirement.
5. Reinstate: only after N consecutive stable runs.

## Incremental vs Full Verification

- **Incremental** (default): run only checkers affected by changed files/modules.
- **Full** (periodic): run all checkers on full codebase. Required before:
  - Promoting integration → main
  - Hardening mode completion
  - After re-decomposition cycles
