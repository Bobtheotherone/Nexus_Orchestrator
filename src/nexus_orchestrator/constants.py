"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/constants.py
Last updated: 2026-02-11

Purpose
- Central place for stable constants: default branch names, directory names, schema versions, etc.

What should be included in this file
- Default branch names (main/integration/contract/work/verify).
- Schema version numbers for config/registry/evidence/state DB.
- Default paths for caches (overridable via config).
- Risk tier names and default mappings.

Functional requirements
- Must be imported safely without heavy deps.

Non-functional requirements
- Changing a constant that affects on-disk formats must be treated as a breaking change and recorded in CHANGELOG.
"""
