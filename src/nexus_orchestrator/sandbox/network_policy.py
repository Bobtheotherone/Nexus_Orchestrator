"""Network egress policy primitives for sandbox and tool-provisioning workflows."""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python <3.11 compatibility.
    UTC = timezone.utc  # noqa: UP017

DecisionLogger = Callable[["NetworkDecision"], None]
AccessLogger = Callable[["NetworkAccessEvent"], None]


class NetworkPolicyMode(str, Enum):
    """Supported network policy modes."""

    DENY = "deny"
    ALLOWLIST = "allowlist"
    LOGGED_PERMISSIVE = "logged_permissive"


class NetworkPolicyViolationError(PermissionError):
    """Raised when a network request is denied by policy."""


@dataclass(frozen=True, slots=True)
class NetworkDecision:
    """Deterministic policy decision for one network request target."""

    mode: NetworkPolicyMode
    target: str
    host: str
    port: int | None
    scheme: str | None
    allowed: bool
    reason: str
    matched_rule: str | None
    context: dict[str, str] = field(default_factory=dict)
    decided_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _normalize_text(self.target, "target"))
        object.__setattr__(self, "host", _normalize_host(self.host, field_name="host"))
        if self.port is not None and (self.port <= 0 or self.port > 65_535):
            raise ValueError("port must be in [1, 65535]")
        if self.scheme is not None:
            object.__setattr__(self, "scheme", _normalize_scheme(self.scheme))
        object.__setattr__(self, "reason", _normalize_text(self.reason, "reason"))
        object.__setattr__(self, "decided_at", _coerce_datetime_utc(self.decided_at))
        normalized_context = {
            _normalize_text(key, "context key", max_len=128): _normalize_text(
                value, "context value", max_len=512
            )
            for key, value in self.context.items()
        }
        object.__setattr__(self, "context", normalized_context)


@dataclass(frozen=True, slots=True)
class NetworkAccessEvent:
    """Observed access telemetry associated with a policy decision."""

    decision: NetworkDecision
    bytes_sent: int = 0
    bytes_received: int = 0
    duration_ms: float | None = None
    error: str | None = None
    logged_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def __post_init__(self) -> None:
        if self.bytes_sent < 0:
            raise ValueError("bytes_sent must be >= 0")
        if self.bytes_received < 0:
            raise ValueError("bytes_received must be >= 0")
        if self.duration_ms is not None and (
            not math.isfinite(self.duration_ms) or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be finite and >= 0")
        if self.error is not None:
            object.__setattr__(self, "error", _normalize_text(self.error, "error", max_len=1024))
        object.__setattr__(self, "logged_at", _coerce_datetime_utc(self.logged_at))


@dataclass(frozen=True, slots=True)
class _AllowRule:
    raw: str
    exact_host: str | None = None
    suffix: str | None = None
    ip_network: ipaddress.IPv4Network | ipaddress.IPv6Network | None = None
    ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None


class NetworkPolicy:
    """Policy evaluator with deny, allowlist, and logged-permissive modes."""

    def __init__(
        self,
        *,
        mode: NetworkPolicyMode | str = NetworkPolicyMode.DENY,
        allowlist: Iterable[str] = (),
        decision_logger: DecisionLogger | None = None,
        access_logger: AccessLogger | None = None,
    ) -> None:
        self._mode = _coerce_mode(mode)
        self._rules = tuple(_parse_allow_rule(item) for item in allowlist)
        self._decision_logger = decision_logger
        self._access_logger = access_logger

    @property
    def mode(self) -> NetworkPolicyMode:
        return self._mode

    @property
    def allowlist(self) -> tuple[str, ...]:
        return tuple(rule.raw for rule in self._rules)

    def evaluate(
        self,
        target: str,
        *,
        port: int | None = None,
        scheme: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> NetworkDecision:
        parsed = _parse_target(target)
        effective_host = parsed.host
        effective_scheme = _normalize_scheme(scheme) if scheme is not None else parsed.scheme
        effective_port = port if port is not None else parsed.port

        if self._mode is NetworkPolicyMode.DENY:
            decision = NetworkDecision(
                mode=self._mode,
                target=target,
                host=effective_host,
                port=effective_port,
                scheme=effective_scheme,
                allowed=False,
                reason="network access denied by policy mode",
                matched_rule=None,
                context=dict(context or {}),
            )
            self._emit_decision(decision)
            return decision

        matched_rule = self._match_rule(effective_host)
        if self._mode is NetworkPolicyMode.ALLOWLIST:
            allowed = matched_rule is not None
            reason = (
                "network access allowed by allowlist rule"
                if allowed
                else "network access denied: host is not in allowlist"
            )
            decision = NetworkDecision(
                mode=self._mode,
                target=target,
                host=effective_host,
                port=effective_port,
                scheme=effective_scheme,
                allowed=allowed,
                reason=reason,
                matched_rule=matched_rule,
                context=dict(context or {}),
            )
            self._emit_decision(decision)
            return decision

        reason = "network access allowed in logged_permissive mode"
        if matched_rule is not None:
            reason = "network access allowed by allowlist rule"
        decision = NetworkDecision(
            mode=self._mode,
            target=target,
            host=effective_host,
            port=effective_port,
            scheme=effective_scheme,
            allowed=True,
            reason=reason,
            matched_rule=matched_rule,
            context=dict(context or {}),
        )
        self._emit_decision(decision)
        return decision

    def enforce(
        self,
        target: str,
        *,
        port: int | None = None,
        scheme: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> NetworkDecision:
        decision = self.evaluate(target, port=port, scheme=scheme, context=context)
        if decision.allowed:
            return decision
        raise NetworkPolicyViolationError(
            f"network request denied for host {decision.host!r}: {decision.reason}"
        )

    def log_access(
        self,
        decision: NetworkDecision,
        *,
        bytes_sent: int = 0,
        bytes_received: int = 0,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> NetworkAccessEvent:
        event = NetworkAccessEvent(
            decision=decision,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            duration_ms=duration_ms,
            error=error,
        )
        if self._access_logger is not None:
            self._access_logger(event)
        return event

    def _match_rule(self, host: str) -> str | None:
        host_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
        try:
            host_ip = ipaddress.ip_address(host)
        except ValueError:
            host_ip = None

        for rule in self._rules:
            if rule.exact_host is not None and host == rule.exact_host:
                return rule.raw
            if rule.suffix is not None and (
                host == rule.suffix or host.endswith(f".{rule.suffix}")
            ):
                return rule.raw
            if rule.ip_address is not None and host_ip is not None and host_ip == rule.ip_address:
                return rule.raw
            if rule.ip_network is not None and host_ip is not None and host_ip in rule.ip_network:
                return rule.raw
        return None

    def _emit_decision(self, decision: NetworkDecision) -> None:
        if self._decision_logger is not None:
            self._decision_logger(decision)


@dataclass(frozen=True, slots=True)
class _ParsedTarget:
    host: str
    port: int | None
    scheme: str | None


def _coerce_mode(value: NetworkPolicyMode | str) -> NetworkPolicyMode:
    if isinstance(value, NetworkPolicyMode):
        return value
    if not isinstance(value, str):
        raise ValueError("mode must be a string or NetworkPolicyMode")
    normalized = value.strip().lower()
    try:
        return NetworkPolicyMode(normalized)
    except ValueError as exc:  # pragma: no cover - defensive branch.
        allowed = ", ".join(item.value for item in NetworkPolicyMode)
        raise ValueError(
            f"unsupported network policy mode {value!r}; expected one of: {allowed}"
        ) from exc


def _normalize_text(value: str, field_name: str, *, max_len: int = 2048) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_len:
        raise ValueError(f"{field_name} must be <= {max_len} characters")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return normalized


def _normalize_host(value: str, *, field_name: str) -> str:
    normalized = _normalize_text(value, field_name, max_len=512).lower()
    if " " in normalized or "/" in normalized:
        raise ValueError(f"{field_name} must be a host or IP literal")
    return normalized


def _normalize_scheme(value: str) -> str:
    normalized = _normalize_text(value, "scheme", max_len=32).lower()
    return normalized.rstrip(":")


def _parse_target(target: str) -> _ParsedTarget:
    normalized_target = _normalize_text(target, "target", max_len=2048)
    if "://" in normalized_target:
        parsed = urlsplit(normalized_target)
        if parsed.hostname is None:
            raise ValueError(f"unable to parse host from target: {target!r}")
        scheme = parsed.scheme.lower() if parsed.scheme else None
        host = _normalize_host(parsed.hostname, field_name="target host")
        return _ParsedTarget(host=host, port=parsed.port, scheme=scheme)

    parsed = urlsplit(f"//{normalized_target}")
    if parsed.hostname is None:
        host = _normalize_host(normalized_target, field_name="target host")
        return _ParsedTarget(host=host, port=None, scheme=None)
    host = _normalize_host(parsed.hostname, field_name="target host")
    return _ParsedTarget(host=host, port=parsed.port, scheme=None)


def _parse_allow_rule(raw_rule: str) -> _AllowRule:
    normalized = _normalize_text(raw_rule, "allowlist rule", max_len=512).lower()

    if normalized.startswith("*."):
        suffix = normalized[2:]
        if not suffix:
            raise ValueError("allowlist wildcard rule must include a suffix")
        return _AllowRule(raw=normalized, suffix=suffix)

    try:
        network = ipaddress.ip_network(normalized, strict=False)
    except ValueError:
        network = None
    if network is not None:
        return _AllowRule(raw=normalized, ip_network=network)

    try:
        addr = ipaddress.ip_address(normalized)
    except ValueError:
        addr = None
    if addr is not None:
        return _AllowRule(raw=normalized, ip_address=addr)

    return _AllowRule(
        raw=normalized, exact_host=_normalize_host(normalized, field_name="allowlist rule")
    )


def _coerce_datetime_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "NetworkAccessEvent",
    "NetworkDecision",
    "NetworkPolicy",
    "NetworkPolicyMode",
    "NetworkPolicyViolation",
    "NetworkPolicyViolationError",
]

# Compatibility alias for callers that imported the old name.
NetworkPolicyViolation = NetworkPolicyViolationError
