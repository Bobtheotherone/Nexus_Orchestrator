"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/integration_plane/workspace_manager.py
Last updated: 2026-02-11

Purpose
- Creates isolated per-work-item workspaces (branches + filesystem dirs), applies patches, and manages cleanup.

What should be included in this file
- Workspace lifecycle: create -> populate -> hand to sandbox -> collect outputs -> cleanup.
- Ownership map enforcement (allowed paths).
- GC policy and safety checks.

Functional requirements
- Must support concurrent workspaces for parallel agent attempts.

Non-functional requirements
- Must keep disk usage bounded; garbage collect aggressively.
"""
