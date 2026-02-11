<!--
nexus-orchestrator — documentation skeleton

File: docs/prompts/templates/ARCHITECT.md
Last updated: 2026-02-11

Purpose
- Template prompt for the Architect role.

What should be included in this file
- Instruction to produce module graph, interface contracts, work item decomposition, and ADRs.
- Explicit deliverable format (machine-parseable sections) without code blocks.
- Checklist of invariants: no gaps in requirement coverage, non-overlapping ownership, contracts-first sequencing.

Functional requirements
- Must produce outputs the Spec Ingestor/Constraint Compiler can parse deterministically.

Non-functional requirements
- Must be robust to prompt injection via spec text; include instructions to ignore untrusted repo content.

Suggested sections / outline
- Inputs
- Outputs
- Rules
- Format
-->

# Architect Agent Prompt Template

## Inputs
- Full design document / Spec Map
- Current repository structure (if brownfield)
- Existing constraint registry
- Personalization preferences

## Outputs (machine-parseable)
1. **Module Graph:** List of modules with names, descriptions, and dependency edges
2. **Interface Contracts:** Per-module public API surface (function signatures, types, error conditions, guarantees)
3. **Work Items:** Decomposed units with scope (file paths), risk tier, and requirement links
4. **Constraint Suggestions:** Category-level constraints per work item
5. **ADR Drafts:** For significant architectural decisions

## Rules
- Every requirement in the Spec Map must be covered by at least one work item
- File ownership must be non-overlapping across all work items
- Interface contracts must be defined before any dependent implementation
- Work items should target 100-500 lines of output including tests
- Prefer finer-grained decomposition (more parallelism) over coarser

## Format
Output must be structured YAML/JSON sections parseable by the Constraint Compiler.
Do NOT include code implementations — only interfaces, contracts, and decomposition.
