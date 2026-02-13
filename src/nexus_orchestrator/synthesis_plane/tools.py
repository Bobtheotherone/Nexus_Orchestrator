"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/tools.py
Last updated: 2026-02-11

Purpose
- Tool request protocol: how agents ask for tools, data, or environment changes; how requests are approved/denied and recorded.

What should be included in this file
- ToolRequest model (purpose, scope, provenance, version, expected benefit).
- Approval workflow: automatic allowlist vs manual escalation.
- Auditing: record every install, command invocation, network access.

Functional requirements
- Must integrate with sandbox/tool provisioner and constraints (e.g., disallow risky tools in strict mode).

Non-functional requirements
- Must be tamper-evident and auditable.
"""

from __future__ import annotations

import json
import math
import tomllib as _toml_reader
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_DECISION_APPROVED = "approved"
_DECISION_DENIED = "denied"
_DECISION_REVIEW_REQUIRED = "review_required"
_VALID_DECISIONS = frozenset(
    {
        _DECISION_APPROVED,
        _DECISION_DENIED,
        _DECISION_REVIEW_REQUIRED,
    }
)


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_text(
    value: str,
    field_name: str,
    *,
    min_len: int = 1,
    max_len: int = 2048,
    strip: bool = True,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip() if strip else value
    if len(normalized) < min_len:
        raise ValueError(f"{field_name} must be at least {min_len} character(s)")
    if len(normalized) > max_len:
        raise ValueError(f"{field_name} must be <= {max_len} characters")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return normalized


def _normalize_tool_name(value: str, field_name: str) -> str:
    parsed = _validate_text(value, field_name, max_len=256).lower()
    parsed = parsed.replace("_", "-")
    if any(char.isspace() for char in parsed):
        raise ValueError(f"{field_name} must not contain whitespace")
    return parsed


def _coerce_json_value(value: object, path: str, *, depth: int = 0) -> JSONValue:
    if depth > 16:
        raise ValueError(f"{path} exceeds max JSON depth")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, str):
        return _validate_text(value, path, min_len=0, max_len=8192, strip=False)
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            result[key] = _coerce_json_value(item, f"{path}.{key}", depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_json_value(item, f"{path}[]", depth=depth + 1) for item in value]
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def _coerce_json_object(value: Mapping[str, object], path: str) -> dict[str, JSONValue]:
    return {
        _validate_text(key, f"{path}.<key>", max_len=256): _coerce_json_value(
            item,
            f"{path}.{key}",
        )
        for key, item in value.items()
    }


@dataclass(frozen=True, slots=True)
class ToolRegistryEntry:
    """Pinned allowlist entry loaded from tools/registry.toml."""

    name: str
    version: str
    source: str
    risk: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_tool_name(self.name, "ToolRegistryEntry.name"))
        object.__setattr__(
            self,
            "version",
            _validate_text(self.version, "ToolRegistryEntry.version", max_len=128),
        )
        object.__setattr__(
            self,
            "source",
            _validate_text(self.source, "ToolRegistryEntry.source", max_len=128),
        )
        object.__setattr__(
            self,
            "risk",
            _validate_text(self.risk, "ToolRegistryEntry.risk", max_len=64).lower(),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "risk": self.risk,
        }


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """Immutable request for tool access/install execution."""

    tool: str
    purpose: str
    scope: str
    provenance: str
    expected_benefit: str
    requested_by_role: str
    version: str | None = None
    version_constraint: str | None = None
    requested_by_work_item_id: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _normalize_tool_name(self.tool, "ToolRequest.tool"))
        object.__setattr__(self, "purpose", _validate_text(self.purpose, "ToolRequest.purpose"))
        object.__setattr__(self, "scope", _validate_text(self.scope, "ToolRequest.scope"))
        object.__setattr__(
            self,
            "provenance",
            _validate_text(self.provenance, "ToolRequest.provenance"),
        )
        object.__setattr__(
            self,
            "expected_benefit",
            _validate_text(self.expected_benefit, "ToolRequest.expected_benefit"),
        )
        object.__setattr__(
            self,
            "requested_by_role",
            _validate_text(self.requested_by_role, "ToolRequest.requested_by_role", max_len=128),
        )
        if self.version is not None:
            object.__setattr__(
                self,
                "version",
                _validate_text(self.version, "ToolRequest.version", max_len=128),
            )
        if self.version_constraint is not None:
            object.__setattr__(
                self,
                "version_constraint",
                _validate_text(
                    self.version_constraint,
                    "ToolRequest.version_constraint",
                    max_len=128,
                ),
            )
        if (
            self.version is not None
            and self.version_constraint is not None
            and self.version != self.version_constraint
        ):
            raise ValueError("ToolRequest.version and version_constraint cannot disagree")
        normalized_work_item_id = self.requested_by_work_item_id
        if normalized_work_item_id is not None:
            normalized_work_item_id = _validate_text(
                normalized_work_item_id,
                "ToolRequest.requested_by_work_item_id",
                max_len=128,
            )
        metadata_payload = _coerce_json_object(dict(self.metadata), "ToolRequest.metadata")
        if normalized_work_item_id is None:
            work_item_from_metadata = metadata_payload.get("work_item_id")
            if isinstance(work_item_from_metadata, str) and work_item_from_metadata.strip():
                normalized_work_item_id = work_item_from_metadata
        object.__setattr__(self, "requested_by_work_item_id", normalized_work_item_id)
        object.__setattr__(self, "metadata", MappingProxyType(metadata_payload))

    @property
    def tool_name(self) -> str:
        """Compatibility alias aligned with tool-request protocol naming."""

        return self.tool

    @property
    def desired_version(self) -> str | None:
        """Compatibility alias aligned with tool-request protocol naming."""

        return self.version

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "tool": self.tool,
            "tool_name": self.tool_name,
            "version": self.version,
            "desired_version": self.desired_version,
            "version_constraint": self.version_constraint or self.version,
            "purpose": self.purpose,
            "scope": self.scope,
            "provenance": self.provenance,
            "expected_benefit": self.expected_benefit,
            "requested_by_role": self.requested_by_role,
            "requested_by_work_item_id": self.requested_by_work_item_id,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ToolApprovalResult:
    """Deterministic, auditable decision for a tool request."""

    request: ToolRequest
    decision: str
    reason: str
    policy_source: str
    resolved_version: str | None = None
    registry_risk: str | None = None
    review_required: bool = field(init=False)
    audit_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, ToolRequest):
            raise ValueError("ToolApprovalResult.request must be ToolRequest")
        normalized_decision = _validate_text(
            self.decision,
            "ToolApprovalResult.decision",
            max_len=64,
        ).lower()
        if normalized_decision not in _VALID_DECISIONS:
            allowed = ", ".join(sorted(_VALID_DECISIONS))
            raise ValueError(f"ToolApprovalResult.decision must be one of: {allowed}")
        object.__setattr__(self, "decision", normalized_decision)
        object.__setattr__(self, "reason", _validate_text(self.reason, "ToolApprovalResult.reason"))
        object.__setattr__(
            self,
            "policy_source",
            _validate_text(self.policy_source, "ToolApprovalResult.policy_source", max_len=128),
        )
        if self.resolved_version is not None:
            object.__setattr__(
                self,
                "resolved_version",
                _validate_text(
                    self.resolved_version,
                    "ToolApprovalResult.resolved_version",
                    max_len=128,
                ),
            )
        if self.registry_risk is not None:
            object.__setattr__(
                self,
                "registry_risk",
                _validate_text(self.registry_risk, "ToolApprovalResult.registry_risk", max_len=64),
            )
        if self.decision == _DECISION_APPROVED and self.resolved_version is None:
            raise ValueError("approved decisions must include resolved_version")

        object.__setattr__(self, "review_required", self.decision == _DECISION_REVIEW_REQUIRED)
        payload = self._decision_payload()
        object.__setattr__(
            self, "audit_id", sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        )

    def _decision_payload(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "request": self.request.to_dict(),
            "decision": self.decision,
            "reason": self.reason,
            "policy_source": self.policy_source,
        }
        if self.resolved_version is not None:
            payload["resolved_version"] = self.resolved_version
        if self.registry_risk is not None:
            payload["registry_risk"] = self.registry_risk
        return payload

    def to_dict(self) -> dict[str, JSONValue]:
        payload = self._decision_payload()
        payload["review_required"] = self.review_required
        payload["audit_id"] = self.audit_id
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def status(self) -> str:
        """Compatibility alias aligned with tool approval protocol naming."""

        return self.decision

    @property
    def pinned_version(self) -> str | None:
        """Compatibility alias aligned with tool approval protocol naming."""

        return self.resolved_version


@runtime_checkable
class ToolDecisionAuditSink(Protocol):
    """Persistence/audit hook for tool decisions."""

    def record_tool_decision(self, decision: Mapping[str, JSONValue]) -> None: ...


@dataclass(frozen=True, slots=True)
class _RegistrySnapshot:
    allowlist: Mapping[str, ToolRegistryEntry]
    denylist: frozenset[str]


class ToolRequestHandler:
    """Deterministic policy evaluator for ToolRequest objects."""

    def __init__(
        self,
        *,
        registry_path: str | Path = "tools/registry.toml",
        audit_sink: ToolDecisionAuditSink | None = None,
        strictness: str = "strict",
    ) -> None:
        self._registry_path = Path(registry_path)
        self._audit_sink = audit_sink
        normalized_strictness = _validate_text(strictness, "strictness", max_len=32).lower()
        if normalized_strictness not in {"strict", "permissive"}:
            raise ValueError("strictness must be either 'strict' or 'permissive'")
        self._strictness = normalized_strictness
        snapshot = _load_registry(self._registry_path)
        self._allowlist = snapshot.allowlist
        self._denylist = snapshot.denylist

    @property
    def allowlist(self) -> Mapping[str, ToolRegistryEntry]:
        return self._allowlist

    @property
    def denylist(self) -> frozenset[str]:
        return self._denylist

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    @property
    def strictness(self) -> str:
        """Policy strictness dial for future profiles."""

        return self._strictness

    def handle_request(self, request: ToolRequest) -> ToolApprovalResult:
        if not isinstance(request, ToolRequest):
            raise ValueError("request must be ToolRequest")
        tool_name = request.tool

        if tool_name in self._denylist:
            decision = ToolApprovalResult(
                request=request,
                decision=_DECISION_DENIED,
                reason="tool is explicitly denylisted",
                policy_source="denylist",
            )
            self._emit_audit(decision)
            return decision

        allowlisted_entry = self._allowlist.get(tool_name)
        if allowlisted_entry is not None:
            if request.version is None or request.version == allowlisted_entry.version:
                decision = ToolApprovalResult(
                    request=request,
                    decision=_DECISION_APPROVED,
                    reason="tool approved by allowlist with pinned version",
                    policy_source="allowlist",
                    resolved_version=allowlisted_entry.version,
                    registry_risk=allowlisted_entry.risk,
                )
                self._emit_audit(decision)
                return decision
            decision = ToolApprovalResult(
                request=request,
                decision=_DECISION_REVIEW_REQUIRED,
                reason=(
                    f"requested version {request.version!r} does not match pinned "
                    f"version {allowlisted_entry.version!r}"
                ),
                policy_source="allowlist_version_mismatch",
                resolved_version=allowlisted_entry.version,
                registry_risk=allowlisted_entry.risk,
            )
            self._emit_audit(decision)
            return decision

        decision = ToolApprovalResult(
            request=request,
            decision=_DECISION_REVIEW_REQUIRED,
            reason="tool is not present in the allowlist",
            policy_source="unlisted_tool",
        )
        self._emit_audit(decision)
        return decision

    def evaluate(self, request: ToolRequest) -> ToolApprovalResult:
        return self.handle_request(request)

    def policy_snapshot(self) -> dict[str, JSONValue]:
        denylist_values: list[JSONValue] = [item for item in sorted(self._denylist)]
        return {
            "registry_path": str(self._registry_path),
            "strictness": self._strictness,
            "allowlist": {
                name: entry.to_dict()
                for name, entry in sorted(self._allowlist.items(), key=lambda item: item[0])
            },
            "denylist": denylist_values,
        }

    def _emit_audit(self, decision: ToolApprovalResult) -> None:
        if self._audit_sink is None:
            return
        self._audit_sink.record_tool_decision(decision.to_dict())


def _load_registry(path: Path) -> _RegistrySnapshot:
    if not path.exists():
        raise ValueError(f"tool registry file not found: {path}")
    try:
        with path.open("rb") as handle:
            payload = _toml_reader.load(handle)
    except OSError as exc:
        raise ValueError(f"unable to read tool registry {path}: {exc}") from exc
    except _toml_reader.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in tool registry {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("tool registry root must be a mapping")

    raw_tools = payload.get("tool")
    if not isinstance(raw_tools, Mapping):
        raise ValueError("tool registry must include [tool.<name>] entries")

    denylist = _parse_denylist(payload.get("denylist"))
    allowlist: dict[str, ToolRegistryEntry] = {}

    for raw_name, raw_entry in sorted(raw_tools.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_name, str):
            raise ValueError("tool registry [tool] keys must be strings")
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"tool registry entry for {raw_name!r} must be a mapping")

        canonical_name = _normalize_tool_name(raw_name, f"tool.{raw_name}")
        if canonical_name in allowlist:
            raise ValueError(f"duplicate tool entry: {raw_name!r}")

        raw_version = raw_entry.get("version")
        if raw_version is None:
            raise ValueError(f"tool {raw_name!r} is missing pinned version")
        if not isinstance(raw_version, str):
            raise ValueError(f"tool {raw_name!r} version must be a string")

        source = raw_entry.get("source", "unknown")
        risk = raw_entry.get("risk", "unknown")
        entry = ToolRegistryEntry(
            name=canonical_name,
            version=raw_version,
            source=str(source),
            risk=str(risk),
        )
        allowlist[canonical_name] = entry

        deny_flag = raw_entry.get("deny")
        if deny_flag is not None:
            if not isinstance(deny_flag, bool):
                raise ValueError(f"tool {raw_name!r} deny flag must be boolean")
            if deny_flag:
                denylist.add(canonical_name)

    return _RegistrySnapshot(
        allowlist=MappingProxyType(allowlist),
        denylist=frozenset(sorted(denylist)),
    )


def _parse_denylist(raw: object) -> set[str]:
    if raw is None:
        return set()

    values: object = raw
    if isinstance(raw, Mapping):
        values = raw.get("tools")
        if values is None:
            return set()

    if isinstance(values, str):
        raise ValueError("denylist must be an array of tool names")
    if not isinstance(values, Sequence):
        raise ValueError("denylist must be an array of tool names")

    parsed: set[str] = set()
    for index, item in enumerate(values):
        if not isinstance(item, str):
            raise ValueError(f"denylist[{index}] must be a string")
        parsed.add(_normalize_tool_name(item, f"denylist[{index}]"))
    return parsed


__all__ = [
    "ToolApprovalResult",
    "ToolDecisionAuditSink",
    "ToolRegistryEntry",
    "ToolRequest",
    "ToolRequestHandler",
]
