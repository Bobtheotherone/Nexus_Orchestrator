"""
Control-plane feedback synthesizer.

This module builds deterministic machine-readable retry feedback packages from:
- constraint gate decisions
- provider call accounting rows
- optional evidence/artifact path context

Feedback is always deep-redacted before return.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from nexus_orchestrator.security.redaction import RedactionConfig, redact_structure

if TYPE_CHECKING:
    from nexus_orchestrator.persistence.repositories import ProviderCallRecord
    from nexus_orchestrator.verification_plane.constraint_gate import ConstraintGateDecision

JSONValue: TypeAlias = object

_SCHEMA_VERSION = 1

_REASON_HINTS: dict[str, str] = {
    "missing_required_stage": "Execute all required verification stages before retry.",
    "failing_required_stage": "Fix required stage failures before requesting another attempt.",
    "must_constraint_unsatisfied": "Resolve all MUST constraints; merge is blocked until they pass.",
    "should_constraint_requires_override": (
        "Either satisfy SHOULD constraints or attach an approved override record."
    ),
    "stage_coverage_incomplete": (
        "Update checker mappings/output so required constraint coverage is reported."
    ),
}


@dataclass(frozen=True, slots=True)
class FeedbackPackage:
    """Machine-readable feedback package with deterministic ordering."""

    schema_version: int
    verdict: str
    accepted: bool
    reason_codes: tuple[str, ...]
    remediation_hints: tuple[str, ...]
    artifact_paths: tuple[str, ...]
    gate_feedback: dict[str, JSONValue]
    provider_calls: tuple[dict[str, JSONValue], ...]
    summary: dict[str, JSONValue]
    metadata: dict[str, JSONValue]

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": self.schema_version,
            "verdict": self.verdict,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "remediation_hints": list(self.remediation_hints),
            "artifact_paths": list(self.artifact_paths),
            "gate_feedback": self.gate_feedback,
            "provider_calls": list(self.provider_calls),
            "summary": self.summary,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class FeedbackSynthesizer:
    """Build deterministic, redacted feedback packages."""

    def __init__(self, *, redaction_config: RedactionConfig | None = None) -> None:
        self._redaction_config = redaction_config

    def synthesize(
        self,
        *,
        gate_decision: ConstraintGateDecision,
        provider_calls: Sequence[ProviderCallRecord] = (),
        artifact_paths: Sequence[str] = (),
        evidence_records: Sequence[object] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, JSONValue]:
        package = self.build_package(
            gate_decision=gate_decision,
            provider_calls=provider_calls,
            artifact_paths=artifact_paths,
            evidence_records=evidence_records,
            metadata=metadata,
        )
        redacted = redact_structure(package.to_dict(), config=self._redaction_config)
        if not isinstance(redacted, Mapping):
            raise TypeError("redacted feedback payload must be a mapping")
        return _as_json_object(redacted)

    def build_package(
        self,
        *,
        gate_decision: ConstraintGateDecision,
        provider_calls: Sequence[ProviderCallRecord] = (),
        artifact_paths: Sequence[str] = (),
        evidence_records: Sequence[object] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> FeedbackPackage:
        gate_payload = _normalize_gate_payload(gate_decision.to_feedback_payload())
        reason_codes = _sorted_unique_strings(gate_payload.get("reason_codes", ()))
        provider_payload = _provider_call_payload(provider_calls)
        resolved_artifacts = _collect_artifact_paths(
            explicit_paths=artifact_paths,
            provider_calls=provider_calls,
            evidence_records=evidence_records,
        )
        remediation_hints = _build_remediation_hints(
            gate_payload=gate_payload,
            provider_payload=provider_payload,
        )
        summary: dict[str, JSONValue] = {
            "reason_count": len(reason_codes),
            "hint_count": len(remediation_hints),
            "artifact_count": len(resolved_artifacts),
            "provider_call_count": len(provider_payload),
        }
        metadata_payload = _normalize_json_mapping(dict(metadata or {}))

        accepted = bool(gate_payload.get("accepted", gate_decision.accepted))
        verdict = str(gate_payload.get("verdict", gate_decision.verdict.value))
        return FeedbackPackage(
            schema_version=_SCHEMA_VERSION,
            verdict=verdict,
            accepted=accepted,
            reason_codes=tuple(reason_codes),
            remediation_hints=tuple(remediation_hints),
            artifact_paths=tuple(resolved_artifacts),
            gate_feedback=gate_payload,
            provider_calls=tuple(provider_payload),
            summary=summary,
            metadata=metadata_payload,
        )

    def render(
        self,
        *,
        gate_decision: ConstraintGateDecision,
        provider_calls: Sequence[ProviderCallRecord] = (),
        artifact_paths: Sequence[str] = (),
        evidence_records: Sequence[object] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, JSONValue]:
        """Compatibility alias for `synthesize()`."""

        return self.synthesize(
            gate_decision=gate_decision,
            provider_calls=provider_calls,
            artifact_paths=artifact_paths,
            evidence_records=evidence_records,
            metadata=metadata,
        )


def synthesize_feedback_package(
    *,
    gate_decision: ConstraintGateDecision,
    provider_calls: Sequence[ProviderCallRecord] = (),
    artifact_paths: Sequence[str] = (),
    evidence_records: Sequence[object] = (),
    metadata: Mapping[str, object] | None = None,
    redaction_config: RedactionConfig | None = None,
) -> dict[str, JSONValue]:
    """Functional wrapper for one-shot synthesis."""

    synthesizer = FeedbackSynthesizer(redaction_config=redaction_config)
    return synthesizer.synthesize(
        gate_decision=gate_decision,
        provider_calls=provider_calls,
        artifact_paths=artifact_paths,
        evidence_records=evidence_records,
        metadata=metadata,
    )


def _normalize_gate_payload(raw: Mapping[str, object]) -> dict[str, JSONValue]:
    payload = _normalize_json_mapping(dict(raw))
    payload["reason_codes"] = _sorted_unique_strings(payload.get("reason_codes", ()))
    payload["selected_stages"] = _sorted_unique_strings(payload.get("selected_stages", ()))
    payload["required_stages"] = _sorted_unique_strings(payload.get("required_stages", ()))
    payload["missing_required_stages"] = _sorted_unique_strings(
        payload.get("missing_required_stages", ())
    )
    payload["failing_required_stages"] = _sorted_unique_strings(
        payload.get("failing_required_stages", ())
    )
    payload["missing_checker_mappings"] = _sorted_unique_strings(
        payload.get("missing_checker_mappings", ())
    )
    payload["uncovered_must_constraints"] = _sorted_unique_strings(
        payload.get("uncovered_must_constraints", ())
    )
    payload["unresolved_should_constraints"] = _sorted_unique_strings(
        payload.get("unresolved_should_constraints", ())
    )
    payload["overridden_should_constraints"] = _sorted_unique_strings(
        payload.get("overridden_should_constraints", ())
    )
    payload["informative_may_constraints"] = _sorted_unique_strings(
        payload.get("informative_may_constraints", ())
    )
    payload["invalid_override_records"] = _sorted_unique_strings(
        payload.get("invalid_override_records", ())
    )
    payload["stage_coverage_gaps"] = _sort_mapping_list(
        payload.get("stage_coverage_gaps", ()),
        key_fields=("stage", "constraint_id"),
    )
    payload["stage_assessments"] = _sort_mapping_list(
        payload.get("stage_assessments", ()),
        key_fields=("stage", "check_count"),
    )
    payload["constraint_assessments"] = _sort_mapping_list(
        payload.get("constraint_assessments", ()),
        key_fields=("constraint_id", "checker_binding"),
    )
    payload["evidence_records"] = _sort_mapping_list(
        payload.get("evidence_records", ()),
        key_fields=("stage", "checker_id"),
    )
    payload["aggregated_violations"] = _sort_mapping_list(
        payload.get("aggregated_violations", ()),
        key_fields=("code", "constraint_id", "stage", "checker_id", "severity", "message"),
    )
    summary_raw = payload.get("summary", {})
    if isinstance(summary_raw, Mapping):
        payload["summary"] = _normalize_json_mapping(dict(summary_raw))
    else:
        payload["summary"] = {}
    return payload


def _provider_call_payload(records: Sequence[ProviderCallRecord]) -> list[dict[str, JSONValue]]:
    ordered = sorted(
        records,
        key=lambda item: (
            item.created_at.isoformat(),
            item.id,
            item.attempt_id,
            item.provider,
            item.model or "",
        ),
    )
    payload: list[dict[str, JSONValue]] = []
    for record in ordered:
        payload.append(
            {
                "id": record.id,
                "attempt_id": record.attempt_id,
                "provider": record.provider,
                "model": record.model,
                "request_id": record.request_id,
                "tokens": record.tokens,
                "cost_usd": float(round(record.cost_usd, 12)),
                "latency_ms": record.latency_ms,
                "error": record.error,
                "created_at": record.created_at.isoformat(),
                "metadata": _normalize_json_mapping(record.metadata),
            }
        )
    return payload


def _build_remediation_hints(
    *,
    gate_payload: Mapping[str, object],
    provider_payload: Sequence[Mapping[str, JSONValue]],
) -> list[str]:
    hints: set[str] = set()
    for reason_code in _sorted_unique_strings(gate_payload.get("reason_codes", ())):
        hint = _REASON_HINTS.get(reason_code)
        if hint is not None:
            hints.add(hint)

    missing_stages = _sorted_unique_strings(gate_payload.get("missing_required_stages", ()))
    if missing_stages:
        hints.add(f"Required stages still missing: {', '.join(missing_stages)}.")

    uncovered_constraints = _sorted_unique_strings(
        gate_payload.get("uncovered_must_constraints", ())
    )
    if uncovered_constraints:
        hints.add(
            "Unsatisfied MUST constraints: "
            + ", ".join(uncovered_constraints)
            + ". Address these before merge."
        )

    violations = gate_payload.get("aggregated_violations", ())
    if isinstance(violations, Sequence):
        for item in violations:
            if not isinstance(item, Mapping):
                continue
            code = _as_optional_non_empty_str(item.get("code"))
            stage = _as_optional_non_empty_str(item.get("stage"))
            checker_id = _as_optional_non_empty_str(item.get("checker_id"))
            constraint_id = _as_optional_non_empty_str(item.get("constraint_id"))
            if code is None and stage is None and checker_id is None and constraint_id is None:
                continue
            parts = [part for part in (code, stage, checker_id, constraint_id) if part is not None]
            hints.add("Investigate violation: " + " / ".join(parts) + ".")

    for provider_call in provider_payload:
        error = _as_optional_non_empty_str(provider_call.get("error"))
        provider = _as_optional_non_empty_str(provider_call.get("provider")) or "provider"
        if error is not None:
            hints.add(
                f"{provider} returned an error; inspect request metadata/artifacts and retry safely."
            )

    return sorted(hints)


def _collect_artifact_paths(
    *,
    explicit_paths: Sequence[str],
    provider_calls: Sequence[ProviderCallRecord],
    evidence_records: Sequence[object],
) -> list[str]:
    paths: set[str] = set()
    for item in explicit_paths:
        parsed = _normalize_path(item)
        if parsed is not None:
            paths.add(parsed)

    for evidence in evidence_records:
        if isinstance(evidence, Mapping):
            raw_paths = evidence.get("artifact_paths", ())
        else:
            raw_paths = getattr(evidence, "artifact_paths", ())
        if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes, bytearray)):
            for raw_item in raw_paths:
                parsed = _normalize_path(raw_item)
                if parsed is not None:
                    paths.add(parsed)

    for record in provider_calls:
        for candidate in _paths_from_metadata(record.metadata):
            parsed = _normalize_path(candidate)
            if parsed is not None:
                paths.add(parsed)

    return sorted(paths)


def _paths_from_metadata(metadata: Mapping[str, JSONValue]) -> tuple[str, ...]:
    paths: set[str] = set()
    for key, value in sorted(metadata.items(), key=lambda item: item[0]):
        key_lower = key.strip().lower()
        if key_lower.endswith("_path") and isinstance(value, str):
            paths.add(value)
            continue
        if (
            key_lower.endswith("_paths")
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
        ):
            for item in value:
                if isinstance(item, str):
                    paths.add(item)
    return tuple(sorted(paths))


def _normalize_json_mapping(value: Mapping[str, object]) -> dict[str, JSONValue]:
    normalized: dict[str, JSONValue] = {}
    for key in sorted(value):
        normalized[key] = _normalize_json_value(value[key])
    return normalized


def _normalize_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return cast("JSONValue", value)
    if isinstance(value, Mapping):
        nested: dict[str, JSONValue] = {}
        for key in sorted(value, key=lambda item: str(item)):
            nested[str(key)] = _normalize_json_value(value[key])
        return nested
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(item) for item in value]
    return str(value)


def _sort_mapping_list(
    value: object,
    *,
    key_fields: Sequence[str],
) -> list[dict[str, JSONValue]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    entries: list[dict[str, JSONValue]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        entries.append(_normalize_json_mapping(dict(item)))
    entries.sort(key=lambda item: _mapping_sort_key(item, key_fields=key_fields))
    return entries


def _mapping_sort_key(
    value: Mapping[str, JSONValue],
    *,
    key_fields: Sequence[str],
) -> tuple[str, ...]:
    parts: list[str] = []
    for key in key_fields:
        parts.append(str(value.get(key, "")))
    parts.append(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return tuple(parts)


def _sorted_unique_strings(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate:
            out.add(candidate)
    return sorted(out)


def _normalize_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    return candidate


def _as_json_object(value: Mapping[str, object]) -> dict[str, JSONValue]:
    return _normalize_json_mapping(dict(value))


def _as_optional_non_empty_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = value.strip()
    if not parsed:
        return None
    return parsed


__all__ = [
    "FeedbackPackage",
    "FeedbackSynthesizer",
    "synthesize_feedback_package",
]
