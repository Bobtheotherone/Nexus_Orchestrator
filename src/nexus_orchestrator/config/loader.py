"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/config/loader.py
Last updated: 2026-02-11

Purpose
- Loads runtime configuration from TOML + env vars and applies profile defaults.

What should be included in this file
- Search paths and precedence rules (CLI flag -> env -> local file -> defaults).
- Environment variable mapping (NEXUS_ prefix).
- Config normalization (paths expanded, durations parsed).

Functional requirements
- Must ensure secrets are referenced, not embedded (e.g., env var names).
- Must support dumping an effective config with secrets redacted.

Non-functional requirements
- Must be deterministic; avoid 'magic' host-dependent defaults that break reproducibility.
"""
