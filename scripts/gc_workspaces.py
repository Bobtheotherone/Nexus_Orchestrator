"""
nexus-orchestrator — script skeleton

File: scripts/gc_workspaces.py
Last updated: 2026-02-11

Purpose
- Garbage-collect old workspaces/build artifacts to reclaim disk.

Expected CLI usage
- python scripts/gc_workspaces.py --workspace-root workspaces/ --dry-run

Functional requirements
- Must delete only safe-to-delete workspaces (not active).
- Must respect retention policy.
- Must support dry-run.

Non-functional requirements
- Safe and conservative; prefer leaving files over deleting wrongly.
- Fast; avoid scanning entire disk.
"""
