"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/knowledge_plane/personalization.py
Last updated: 2026-02-11

Purpose
- Personalization memory: stores user preferences as constraints/heuristics (preferred stacks, style rules, dependency policy).

What should be included in this file
- Representation of preferences as data (not hidden prompts).
- How preferences influence planning and routing.
- Import/export of a 'profile' for reuse across projects.

Functional requirements
- Must allow turning off personalization for reproducible baseline runs.

Non-functional requirements
- Must keep private data local and avoid sending it to providers unless necessary and redacted.
"""
