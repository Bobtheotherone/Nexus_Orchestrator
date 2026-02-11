"""
nexus-orchestrator — test skeleton

File: tests/unit/config/test_loader.py
Last updated: 2026-02-11

Purpose
- Validate config loader behavior (TOML + env var overrides).

What this test file should cover
- Load orchestrator.toml with defaults.
- Environment variable overrides with NEXUS_ prefix.
- Error handling for missing/invalid fields.

Functional requirements
- Must support mock/offline config.

Non-functional requirements
- Deterministic; no dependency on host-specific paths.
"""
