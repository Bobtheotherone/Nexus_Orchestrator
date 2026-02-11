"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/roles.py
Last updated: 2026-02-11

Purpose
- Defines agent roles, capabilities, budgets, and default prompts.

What should be included in this file
- Role enumeration and metadata (allowed tools, sandbox permissions, model preferences).
- Risk-tier mapping to required roles (e.g., critical requires Reviewer + Security).

Functional requirements
- Must be configurable via orchestrator.toml (enable/disable roles, budgets).

Non-functional requirements
- Roles must be explicit and auditable; avoid implicit permissions.
"""
