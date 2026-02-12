"""
nexus-orchestrator — security redaction utilities

File: src/nexus_orchestrator/security/redaction.py
Last updated: 2026-02-12

Purpose
- Implements redaction rules for logs, evidence, and provider transcripts.

What should be included in this file
- Secret patterns and configurable allowlists/denylists.
- Deterministic redaction transforms and tests.

Functional requirements
- Must ensure no secrets leak into prompts/logs by default.

Non-functional requirements
- Must minimize false positives while prioritizing safety.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

REDACTED_VALUE: Final[str] = "***REDACTED***"

KeyPath: TypeAlias = tuple[str, ...]
PatternLike: TypeAlias = str | re.Pattern[str]
KeyHook: TypeAlias = Callable[[str, KeyPath], bool]
TextHook: TypeAlias = Callable[[str, str], bool]

DEFAULT_SENSITIVE_KEY_DENYLIST: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "id_token",
        "password",
        "passwd",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "session_token",
        "token",
        "webhook_secret",
    }
)

_DEFAULT_SENSITIVE_KEY_SUFFIXES: Final[tuple[str, ...]] = (
    "_api_key",
    "_access_token",
    "_refresh_token",
    "_id_token",
    "_auth_token",
    "_client_secret",
    "_private_key",
    "_password",
    "_passwd",
    "_secret",
    "_token",
)

_DEFAULT_SENSITIVE_KEY_PREFIXES: Final[tuple[str, ...]] = (
    "api_key_",
    "access_token_",
    "refresh_token_",
    "password_",
    "private_key_",
    "secret_",
)

_CAMEL_CASE_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_MATCH_ALL = re.compile(r"[\s\S]+")


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """One secret-like match discovered during scanning."""

    rule: str
    start: int
    end: int
    sample: str


@dataclass(frozen=True, slots=True)
class _TextRule:
    name: str
    pattern: re.Pattern[str]
    sensitive_group: int | None = None


_DEFAULT_TEXT_RULES: Final[tuple[_TextRule, ...]] = (
    _TextRule(
        name="private_key_block",
        pattern=re.compile(
            r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----"
            r"[\s\S]+?"
            r"-----END(?: [A-Z0-9]+)* PRIVATE KEY-----"
        ),
    ),
    _TextRule(
        name="authorization_bearer",
        pattern=re.compile(r"(?i)(\bauthorization\s*:\s*bearer\s+)([A-Za-z0-9\-._~+/=]{8,})"),
        sensitive_group=2,
    ),
    _TextRule(
        name="explicit_secret_assignment",
        pattern=re.compile(
            r"(?i)(\b(?:password|passwd|secret|api[_-]?key|client[_-]?secret|"
            r"access[_-]?token|refresh[_-]?token|id[_-]?token)\b\s*[:=]\s*[\"']?)"
            r"([A-Za-z0-9._~+/=-]{6,})"
        ),
        sensitive_group=2,
    ),
    _TextRule(name="openai_api_key", pattern=re.compile(r"\bsk-[A-Za-z0-9]{20,255}\b")),
    _TextRule(name="anthropic_api_key", pattern=re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,255}\b")),
    _TextRule(name="aws_access_key", pattern=re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    _TextRule(name="github_token", pattern=re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b")),
    _TextRule(name="slack_token", pattern=re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{16,255}\b")),
    _TextRule(
        name="jwt",
        pattern=re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
)


class SecretDetectionError(ValueError):
    """Raised when secret detection policy requires hard failure."""


class RedactionError(RuntimeError):
    """Raised when fail-closed redaction policy uses ``raise`` mode."""


@dataclass(frozen=True, slots=True)
class RedactionConfig:
    """Policy that controls detection/redaction behavior."""

    replacement: str = REDACTED_VALUE
    fail_closed: bool = False
    on_failure: Literal["redact", "raise"] = "redact"
    on_detection: Literal["redact", "raise"] = "redact"

    allowlist_can_override: bool = False
    key_allowlist: frozenset[str] = frozenset()
    key_denylist: frozenset[str] = DEFAULT_SENSITIVE_KEY_DENYLIST

    key_allowlist_hook: KeyHook | None = None
    key_denylist_hook: KeyHook | None = None

    text_allowlist_patterns: tuple[PatternLike, ...] = ()
    text_denylist_patterns: tuple[PatternLike, ...] = ()
    text_allowlist_hook: TextHook | None = None
    text_denylist_hook: TextHook | None = None


DEFAULT_REDACTION_CONFIG: Final[RedactionConfig] = RedactionConfig()


@dataclass(frozen=True, slots=True)
class _ResolvedConfig:
    replacement: str
    fail_closed: bool
    on_failure: Literal["redact", "raise"]
    on_detection: Literal["redact", "raise"]
    allowlist_can_override: bool

    key_allowlist: frozenset[str]
    key_denylist: frozenset[str]
    key_allowlist_hook: KeyHook | None
    key_denylist_hook: KeyHook | None

    text_allowlist_patterns: tuple[re.Pattern[str], ...]
    text_denylist_patterns: tuple[re.Pattern[str], ...]
    text_allowlist_hook: TextHook | None
    text_denylist_hook: TextHook | None

    text_rules: tuple[_TextRule, ...]


def scan_for_secrets(
    text: str,
    *,
    config: RedactionConfig | None = None,
    fail_closed: bool | None = None,
) -> tuple[SecretFinding, ...]:
    """Scan text for secret-like patterns in deterministic rule order."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    resolved = _resolve_config(config, fail_closed)

    try:
        findings: list[SecretFinding] = []
        for rule in resolved.text_rules:
            for match in rule.pattern.finditer(text):
                sample = _extract_match_sample(match, rule.sensitive_group)
                if not _should_treat_match_as_sensitive(
                    rule_name=rule.name,
                    candidate=sample,
                    resolved=resolved,
                ):
                    continue
                start, end = match.span(rule.sensitive_group or 0)
                findings.append(SecretFinding(rule=rule.name, start=start, end=end, sample=sample))
        findings.sort(key=lambda item: (item.start, item.end, item.rule))
        return tuple(findings)
    except Exception as exc:  # noqa: BLE001
        return _handle_failure_scan(exc, resolved)


def is_sensitive_key(
    key: str,
    *,
    key_path: KeyPath = (),
    config: RedactionConfig | None = None,
    fail_closed: bool | None = None,
) -> bool:
    """Return whether ``key`` should be treated as sensitive by default policy."""

    resolved = _resolve_config(config, fail_closed)
    return _is_sensitive_key_resolved(key, key_path=key_path, resolved=resolved)


def redact_text(
    text: str,
    *,
    config: RedactionConfig | None = None,
    fail_closed: bool | None = None,
) -> str:
    """Redact secret-like text. Deterministic and idempotent for stable inputs."""

    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    resolved = _resolve_config(config, fail_closed)
    try:
        if resolved.on_detection == "raise":
            findings = scan_for_secrets(text, config=_to_public_config(resolved))
            if findings:
                rule_names = ", ".join(sorted({item.rule for item in findings}))
                raise SecretDetectionError(f"secret-like pattern detected: {rule_names}")

        redacted = text
        for rule in resolved.text_rules:
            redacted = _apply_text_rule(redacted, rule=rule, resolved=resolved)

        return redacted
    except SecretDetectionError:
        raise
    except RedactionError:
        raise
    except Exception as exc:  # noqa: BLE001
        return _handle_failure_text(exc, original=text, resolved=resolved)


def redact_structure(
    value: object,
    *,
    config: RedactionConfig | None = None,
    fail_closed: bool | None = None,
) -> object:
    """Return a deep-redacted copy of nested structures."""

    resolved = _resolve_config(config, fail_closed)
    try:
        return _redact_structure(value=value, key_path=(), seen=set(), resolved=resolved)
    except Exception as exc:  # noqa: BLE001
        return _handle_failure_structure(exc, original=value, resolved=resolved)


def redact_value(
    value: object,
    *,
    config: RedactionConfig | None = None,
    fail_closed: bool | None = None,
) -> object:
    """Alias for ``redact_structure`` used by other modules."""

    return redact_structure(value=value, config=config, fail_closed=fail_closed)


def _resolve_config(config: RedactionConfig | None, fail_closed: bool | None) -> _ResolvedConfig:
    base = config if config is not None else DEFAULT_REDACTION_CONFIG
    effective_fail_closed = base.fail_closed if fail_closed is None else bool(fail_closed)

    key_allowlist = frozenset(
        normalized
        for normalized in (_normalize_key(item) for item in base.key_allowlist)
        if normalized
    )
    key_denylist = frozenset(
        normalized
        for normalized in (_normalize_key(item) for item in base.key_denylist)
        if normalized
    )

    allow_patterns = _compile_patterns(
        base.text_allowlist_patterns, fail_closed=effective_fail_closed
    )
    deny_patterns = _compile_patterns(
        base.text_denylist_patterns, fail_closed=effective_fail_closed
    )
    custom_rules = tuple(
        _TextRule(name=f"custom_denylist_{index}", pattern=pattern)
        for index, pattern in enumerate(deny_patterns)
    )

    return _ResolvedConfig(
        replacement=base.replacement or REDACTED_VALUE,
        fail_closed=effective_fail_closed,
        on_failure=base.on_failure,
        on_detection=base.on_detection,
        allowlist_can_override=base.allowlist_can_override,
        key_allowlist=key_allowlist,
        key_denylist=key_denylist,
        key_allowlist_hook=base.key_allowlist_hook,
        key_denylist_hook=base.key_denylist_hook,
        text_allowlist_patterns=allow_patterns,
        text_denylist_patterns=deny_patterns,
        text_allowlist_hook=base.text_allowlist_hook,
        text_denylist_hook=base.text_denylist_hook,
        text_rules=_DEFAULT_TEXT_RULES + custom_rules,
    )


def _compile_patterns(
    patterns: tuple[PatternLike, ...],
    *,
    fail_closed: bool,
) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        if isinstance(pattern, re.Pattern):
            compiled.append(pattern)
            continue
        if isinstance(pattern, str):
            try:
                compiled.append(re.compile(pattern))
            except re.error:
                if fail_closed:
                    compiled.append(_MATCH_ALL)
            continue
        if fail_closed:
            compiled.append(_MATCH_ALL)
    return tuple(compiled)


def _extract_match_sample(match: re.Match[str], group: int | None) -> str:
    if group is None:
        return match.group(0)
    try:
        return match.group(group)
    except IndexError:
        return match.group(0)


def _apply_text_rule(text: str, *, rule: _TextRule, resolved: _ResolvedConfig) -> str:
    def repl(match: re.Match[str]) -> str:
        candidate = _extract_match_sample(match, rule.sensitive_group)
        if not _should_treat_match_as_sensitive(
            rule_name=rule.name,
            candidate=candidate,
            resolved=resolved,
        ):
            return match.group(0)
        return _replace_sensitive_group(
            match, replacement=resolved.replacement, group=rule.sensitive_group
        )

    return rule.pattern.sub(repl, text)


def _should_treat_match_as_sensitive(
    *,
    rule_name: str,
    candidate: str,
    resolved: _ResolvedConfig,
) -> bool:
    if resolved.text_denylist_hook is not None:
        try:
            if resolved.text_denylist_hook(rule_name, candidate):
                return True
        except Exception:
            if resolved.fail_closed:
                return True

    allowlisted = False

    for pattern in resolved.text_allowlist_patterns:
        try:
            if pattern.search(candidate):
                allowlisted = True
                break
        except Exception:
            if resolved.fail_closed:
                return True

    if resolved.text_allowlist_hook is not None:
        try:
            if resolved.text_allowlist_hook(rule_name, candidate):
                allowlisted = True
        except Exception:
            if resolved.fail_closed:
                return True

    return not (allowlisted and resolved.allowlist_can_override)


def _replace_sensitive_group(
    match: re.Match[str],
    *,
    replacement: str,
    group: int | None,
) -> str:
    if group is None:
        return replacement

    full = match.group(0)
    try:
        start, end = match.span(group)
    except IndexError:
        return replacement

    offset_start = start - match.start(0)
    offset_end = end - match.start(0)
    return f"{full[:offset_start]}{replacement}{full[offset_end:]}"


def _is_sensitive_key_resolved(key: str, *, key_path: KeyPath, resolved: _ResolvedConfig) -> bool:
    normalized = _normalize_key(key)
    if not normalized:
        return resolved.fail_closed

    if resolved.key_denylist_hook is not None:
        try:
            if resolved.key_denylist_hook(key, key_path):
                return True
        except Exception:
            if resolved.fail_closed:
                return True

    allowlisted = normalized in resolved.key_allowlist
    if resolved.key_allowlist_hook is not None:
        try:
            if resolved.key_allowlist_hook(key, key_path):
                allowlisted = True
        except Exception:
            if resolved.fail_closed:
                return True

    denylisted = normalized in resolved.key_denylist
    if not denylisted:
        denylisted = any(normalized.endswith(suffix) for suffix in _DEFAULT_SENSITIVE_KEY_SUFFIXES)
    if not denylisted:
        denylisted = any(
            normalized.startswith(prefix) for prefix in _DEFAULT_SENSITIVE_KEY_PREFIXES
        )

    if denylisted:
        return not (allowlisted and resolved.allowlist_can_override)

    return False


def _is_allowlisted_key_resolved(key: str, *, key_path: KeyPath, resolved: _ResolvedConfig) -> bool:
    normalized = _normalize_key(key)
    if not normalized:
        return False
    if normalized in resolved.key_allowlist:
        return True
    if resolved.key_allowlist_hook is None:
        return False
    try:
        return bool(resolved.key_allowlist_hook(key, key_path))
    except Exception:
        return False


def _redact_structure(
    *,
    value: object,
    key_path: KeyPath,
    seen: set[int],
    resolved: _ResolvedConfig,
) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return redact_text(value, config=_to_public_config(resolved))

    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in seen:
            return _fail_closed_value(value, resolved)
        seen.add(value_id)
        try:
            out: dict[object, object] = {}
            for key in sorted(value, key=lambda item: str(item)):
                item = value[key]
                key_name = str(key)
                child_path = (*key_path, key_name)
                if isinstance(key, str):
                    allowlisted = _is_allowlisted_key_resolved(
                        key, key_path=child_path, resolved=resolved
                    )
                    sensitive = _is_sensitive_key_resolved(
                        key,
                        key_path=child_path,
                        resolved=resolved,
                    )
                    if allowlisted and resolved.allowlist_can_override:
                        out[key] = item
                    elif sensitive:
                        out[key] = resolved.replacement
                    else:
                        out[key] = _redact_structure(
                            value=item,
                            key_path=child_path,
                            seen=seen,
                            resolved=resolved,
                        )
                else:
                    out[key] = _redact_structure(
                        value=item,
                        key_path=child_path,
                        seen=seen,
                        resolved=resolved,
                    )
            return out
        finally:
            seen.discard(value_id)

    if isinstance(value, list):
        value_id = id(value)
        if value_id in seen:
            return _fail_closed_value(value, resolved)
        seen.add(value_id)
        try:
            return [
                _redact_structure(
                    value=item,
                    key_path=(*key_path, f"[{index}]"),
                    seen=seen,
                    resolved=resolved,
                )
                for index, item in enumerate(value)
            ]
        finally:
            seen.discard(value_id)

    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in seen:
            return _fail_closed_value(value, resolved)
        seen.add(value_id)
        try:
            return tuple(
                _redact_structure(
                    value=item,
                    key_path=(*key_path, f"[{index}]"),
                    seen=seen,
                    resolved=resolved,
                )
                for index, item in enumerate(value)
            )
        finally:
            seen.discard(value_id)

    if isinstance(value, (set, frozenset)):
        items = [
            _redact_structure(value=item, key_path=(*key_path, "{}"), seen=seen, resolved=resolved)
            for item in value
        ]
        items.sort(key=lambda item: repr(item))
        return tuple(items)

    return _fail_closed_value(value, resolved)


def _fail_closed_value(original: object, resolved: _ResolvedConfig) -> object:
    if resolved.fail_closed:
        return resolved.replacement
    return original


def _normalize_key(key: str) -> str:
    with_boundaries = _CAMEL_CASE_BOUNDARY.sub(r"\1_\2", key.strip())
    return _NON_ALNUM.sub("_", with_boundaries.lower()).strip("_")


def _handle_failure_text(exc: Exception, *, original: str, resolved: _ResolvedConfig) -> str:
    if not resolved.fail_closed:
        return original
    if resolved.on_failure == "raise":
        raise RedactionError("redaction failed for text") from exc
    return resolved.replacement


def _handle_failure_structure(
    exc: Exception, *, original: object, resolved: _ResolvedConfig
) -> object:
    if not resolved.fail_closed:
        return original
    if resolved.on_failure == "raise":
        raise RedactionError("redaction failed for structure") from exc
    return resolved.replacement


def _handle_failure_scan(exc: Exception, resolved: _ResolvedConfig) -> tuple[SecretFinding, ...]:
    if not resolved.fail_closed:
        return ()
    if resolved.on_failure == "raise":
        raise RedactionError("secret scan failed") from exc
    return (SecretFinding(rule="scan_failure", start=0, end=0, sample=resolved.replacement),)


def _to_public_config(resolved: _ResolvedConfig) -> RedactionConfig:
    return RedactionConfig(
        replacement=resolved.replacement,
        fail_closed=resolved.fail_closed,
        on_failure=resolved.on_failure,
        on_detection=resolved.on_detection,
        allowlist_can_override=resolved.allowlist_can_override,
        key_allowlist=resolved.key_allowlist,
        key_denylist=resolved.key_denylist,
        key_allowlist_hook=resolved.key_allowlist_hook,
        key_denylist_hook=resolved.key_denylist_hook,
        text_allowlist_patterns=resolved.text_allowlist_patterns,
        text_denylist_patterns=resolved.text_denylist_patterns,
        text_allowlist_hook=resolved.text_allowlist_hook,
        text_denylist_hook=resolved.text_denylist_hook,
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
