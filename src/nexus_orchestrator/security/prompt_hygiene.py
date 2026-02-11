"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/security/prompt_hygiene.py
Last updated: 2026-02-11

Purpose
- Prompt injection defenses and context sanitization for feeding repo content/specs into LLMs.

What should be included in this file
- Heuristics to detect instruction-like content in repo files and demote/exclude it.
- Delimiting and quoting strategy for untrusted content.
- Policy: which files are trusted (templates, schemas) vs untrusted (generated code).

Functional requirements
- Must be applied by context assembler and prompt rendering.

Non-functional requirements
- Must be transparent (log when content is excluded, with rationale).
"""
