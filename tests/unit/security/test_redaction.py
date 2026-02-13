"""
nexus-orchestrator — unit tests for security redaction

File: tests/unit/security/test_redaction.py
Last updated: 2026-02-12

Purpose
- Verify deterministic secret detection/redaction behavior with policy controls.

What this test file should cover
- Key-based and pattern-based redaction.
- Idempotency and false-positive restraint.
- Allowlist/denylist policy behavior.
- Secret scanning helpers.

Functional requirements
- Offline only.

Non-functional requirements
- Deterministic and hard to bypass.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import pytest

import nexus_orchestrator.security as security_pkg
from nexus_orchestrator.security.redaction import (
    REDACTED_VALUE,
    RedactionConfig,
    SecretDetectionError,
    is_sensitive_key,
    redact_structure,
    redact_text,
    scan_for_secrets,
)


def _as_object_mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def test_key_based_redaction_masks_sensitive_values() -> None:
    value = {"api_key": "sk-THISISFAKE1234567890", "safe": "visible"}

    redacted = _as_object_mapping(redact_structure(value))

    assert redacted["api_key"] == REDACTED_VALUE
    assert redacted["safe"] == "visible"


def test_pattern_based_redaction_masks_private_key_block() -> None:
    text = """
-----BEGIN PRIVATE KEY-----
abc123
-----END PRIVATE KEY-----
""".strip()

    redacted = redact_text(text)

    assert REDACTED_VALUE in redacted
    assert "BEGIN PRIVATE KEY" not in redacted


def test_idempotency_for_text_and_structure() -> None:
    text = "token=abc123"
    once_text = redact_text(text)
    twice_text = redact_text(once_text)
    assert once_text == twice_text

    struct = {"token": "abc123", "nested": {"password": "p@ss"}}
    once_struct = redact_structure(struct)
    twice_struct = redact_structure(once_struct)
    assert once_struct == twice_struct


def test_false_positive_restraint_for_common_words() -> None:
    payload = {
        "keyboard": "enabled",
        "token_count": 2,
        "monkey": "banana",
    }

    redacted = redact_structure(payload)

    assert redacted == payload
    assert not is_sensitive_key("keyboard")
    assert not is_sensitive_key("token_count")


def test_allowlist_only_overrides_when_policy_allows() -> None:
    payload = {"api_key": "sk-THISISFAKE1234567890"}

    safe_default = _as_object_mapping(
        redact_structure(
            payload,
            config=RedactionConfig(key_allowlist=frozenset({"api_key"})),
        )
    )
    override_allowed = _as_object_mapping(
        redact_structure(
            payload,
            config=RedactionConfig(
                key_allowlist=frozenset({"api_key"}),
                allowlist_can_override=True,
            ),
        )
    )

    assert safe_default["api_key"] == REDACTED_VALUE
    assert override_allowed["api_key"] == "sk-THISISFAKE1234567890"


def test_custom_denylist_pattern_and_allowlist_pattern() -> None:
    config = RedactionConfig(
        text_denylist_patterns=(r"build-secret-\d+",),
        text_allowlist_patterns=(re.compile(r"^not-a-secret$"),),
        allowlist_can_override=True,
    )

    assert redact_text("build-secret-123", config=config) == REDACTED_VALUE
    assert redact_text("password=not-a-secret", config=config) == "password=not-a-secret"


def test_scan_for_secrets_finds_multiple_matches() -> None:
    text = "api_key=sk-THISISFAKE123456789012 and ghp_" + ("a" * 36)

    findings = scan_for_secrets(text)

    assert len(findings) >= 2
    rule_names = {item.rule for item in findings}
    assert "openai_api_key" in rule_names
    assert "github_token" in rule_names


def test_detection_raise_policy_fails_closed() -> None:
    config = RedactionConfig(on_detection="raise")

    with pytest.raises(SecretDetectionError):
        redact_text("password=abc123", config=config)


def test_security_package_exports_are_consistent() -> None:
    findings = security_pkg.scan_for_secrets("api_key=sk-THISISFAKE1234567890")
    assert findings
    redacted = security_pkg.redact_text("password=abc123")
    assert REDACTED_VALUE in redacted
