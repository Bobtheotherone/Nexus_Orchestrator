<!--
nexus-orchestrator — documentation skeleton

File: docs/GLOSSARY.md
Last updated: 2026-02-11

Purpose
- Defines canonical meanings of terms used across docs and code (Constraint Envelope, Evidence Ledger, Work Item, etc.).

What should be included in this file
- Alphabetical list of terms and concise definitions.
- Cross-links to design_document.md and DATA_MODEL.md sections.

Functional requirements
- Must disambiguate overloaded terms like 'artifact' and 'contract'.

Non-functional requirements
- Keep definitions stable; changes require ADR if meaning shifts.

Suggested sections / outline
- Terms
-->

# Glossary

**Artifact** — Any deliverable that can be committed: code, config, docs, tests, schemas, CI definitions.

**Attempt** — A single agent invocation for a work item. A work item may have multiple attempts up to its iteration budget.

**Constraint** — A machine-checkable rule that must hold. Constraints are first-class, versioned, and enforced at merge time.

**Constraint Envelope** — The complete set of constraints applicable to a specific work item, including propagated constraints from dependencies.

**Constraint Gate** — The verification engine that determines whether an artifact satisfies all constraints in its envelope.

**Evidence** — A structured record proving that constraints are satisfied: test results, build logs, scan reports, benchmarks.

**Evidence Ledger** — The persistent store linking evidence to constraints, work items, and commits.

**Interface Contract** — A formal specification of a module boundary: types, functions, protocols, error conditions, guarantees.

**Merge Queue** — The serialized integration mechanism. One merge at a time, each verified against the current integration state.

**Never Again** — The constraint evolution principle: every failure becomes a new permanent constraint, test, or rule.

**Spec Map** — The normalized, structured representation of a design document's requirements, entities, and interfaces.

**Task Graph** — The DAG of work items with dependency edges, used by the scheduler to maximize parallelism.

**Work Item** — The smallest independently-executable unit of work, with defined scope, constraint envelope, and budget.
