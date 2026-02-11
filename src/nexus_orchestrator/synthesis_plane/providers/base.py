"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/providers/base.py
Last updated: 2026-02-11

Purpose
- Abstract provider interface and common request/response models for LLM calls.

What should be included in this file
- Request fields: model, role, prompt, context docs, tool permissions, budget limits.
- Response fields: content, structured outputs, token usage, cost, latency, errors.
- Error taxonomy and retryability classification.

Functional requirements
- Must support idempotent retries with consistent attribution.

Non-functional requirements
- Must make it easy to add new providers without touching core logic.
"""
