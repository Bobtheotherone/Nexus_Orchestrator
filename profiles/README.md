<!--
nexus-orchestrator — documentation

File: profiles/README.md
Last updated: 2026-02-11

Purpose
- Operator personalization profiles (private-first).

What should be included in this file
- How the orchestrator interprets this profile (constraints vs heuristics).
- What fields are safe to store (preferences) vs NOT safe (secrets).
- How profiles are versioned and imported/exported.

Functional requirements
- Must be understandable by an agentic AI implementing personalization memory.

Non-functional requirements
- Keep this operator-specific; do not design for multi-user accounts.
-->

# Profiles

This directory contains **operator profile** files that encode your preferences as data:
- preferred languages/frameworks
- dependency risk tolerance
- architectural “house rules”
- default verification strictness for `should` constraints (never for `must`)
- provider/model preferences and budget posture

Profiles are **not secrets**. They must never contain API keys.

## Files
- `operator_profile.toml` — your main profile (copy/modify locally)
