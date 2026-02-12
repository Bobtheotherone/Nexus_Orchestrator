"""
nexus-orchestrator — public security utilities

File: src/nexus_orchestrator/security/__init__.py
Last updated: 2026-02-12

Purpose
- Security utilities: secret scanning hooks, prompt injection defenses, provenance tracking.

What should be included in this file
- Helpers used across planes to enforce safety boundaries.

Functional requirements
- Must provide consistent redaction and scanning utilities.

Non-functional requirements
- Must fail closed for critical safety checks (configurable exceptions require audit).
"""

from nexus_orchestrator.security.redaction import (
    DEFAULT_REDACTION_CONFIG,
    DEFAULT_SENSITIVE_KEY_DENYLIST,
    REDACTED_VALUE,
    RedactionConfig,
    RedactionError,
    SecretDetectionError,
    SecretFinding,
    is_sensitive_key,
    redact_structure,
    redact_text,
    redact_value,
    scan_for_secrets,
)

__all__ = [
    "DEFAULT_REDACTION_CONFIG",
    "DEFAULT_SENSITIVE_KEY_DENYLIST",
    "REDACTED_VALUE",
    "RedactionConfig",
    "RedactionError",
    "SecretDetectionError",
    "SecretFinding",
    "is_sensitive_key",
    "redact_structure",
    "redact_text",
    "redact_value",
    "scan_for_secrets",
]
