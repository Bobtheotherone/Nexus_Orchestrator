"""Deterministic offline failure mining for Phase 8 quality amplification."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Final, TypeAlias

from nexus_orchestrator.domain import (
    Attempt,
    Constraint,
    ConstraintSeverity,
    ConstraintSource,
    EvidenceResult,
    WorkItem,
)
from nexus_orchestrator.knowledge_plane.constraint_registry import ConstraintRegistry
from nexus_orchestrator.verification_plane.constraint_gate import (
    ConstraintGateDecision,
    PipelineCheckResult,
    PipelineOutput,
)

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc  # noqa: UP017

PathLike: TypeAlias = str | os.PathLike[str]
JSONValue: TypeAlias = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]

DEFAULT_MINER_VERSION: Final[str] = "phase8-constraint-miner-v1"
_EPOCH_UTC: Final[datetime] = datetime(1970, 1, 1, tzinfo=UTC)

_SECURITY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "credential",
        "cve",
        "dependency_audit",
        "injection",
        "secret",
        "security",
        "token",
        "vulnerab",
    }
)
_PERFORMANCE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "benchmark",
        "cpu",
        "latency",
        "memory",
        "perf",
        "performance",
        "regression",
        "throughput",
    }
)
_FLAKE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "flake",
        "flaky",
        "intermittent",
        "nondeterministic",
        "non-deterministic",
        "race",
        "retry_passed",
        "transient",
    }
)
_SPEC_GAP_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "missing_checker_mapping",
        "missing_required_stage",
        "required_stage_failed",
        "should_constraint_requires_override",
        "should_constraint_unsatisfied",
        "stage_coverage_incomplete",
        "gate.missing_checker_mapping",
        "gate.required_stage_failed",
        "gate.required_stage_missing",
        "gate.should_constraint_unsatisfied",
        "gate.stage_coverage_incomplete",
    }
)


class FailureTaxonomy(str, Enum):
    """Deterministic failure taxonomy for mined proposals."""

    BUG = "bug"
    SPEC_GAP = "spec_gap"
    FLAKE = "flake"
    PERF_REGRESSION = "perf_regression"
    SECURITY_FINDING = "security_finding"


@dataclass(frozen=True, slots=True)
class ConstraintProposal:
    """Proposed never-again constraint derived from one failure signal."""

    proposed_constraint: Constraint
    justification: str
    source_failure_id: str
    auto_accept: bool
    semantic_signature: str
    work_item_id: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.proposed_constraint, Constraint):
            raise TypeError("ConstraintProposal.proposed_constraint must be a Constraint")
        if not self.justification.strip():
            raise ValueError("ConstraintProposal.justification must be non-empty")
        if not self.source_failure_id.strip():
            raise ValueError("ConstraintProposal.source_failure_id must be non-empty")
        if not self.semantic_signature.strip():
            raise ValueError("ConstraintProposal.semantic_signature must be non-empty")
        if self.work_item_id is not None and not self.work_item_id.strip():
            raise ValueError("ConstraintProposal.work_item_id must be non-empty when provided")
        if self.attempt_id is not None and not self.attempt_id.strip():
            raise ValueError("ConstraintProposal.attempt_id must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class _ConstraintProfile:
    severity: ConstraintSeverity
    category: str
    checker_binding: str
    auto_accept: bool
    id_tag: str


_PROFILE_BY_FAILURE: Final[dict[FailureTaxonomy, _ConstraintProfile]] = {
    FailureTaxonomy.BUG: _ConstraintProfile(
        severity=ConstraintSeverity.MUST,
        category="correctness",
        checker_binding="test_checker",
        auto_accept=True,
        id_tag="BUG",
    ),
    FailureTaxonomy.SPEC_GAP: _ConstraintProfile(
        severity=ConstraintSeverity.SHOULD,
        category="documentation",
        checker_binding="documentation_checker",
        auto_accept=False,
        id_tag="SPEC",
    ),
    FailureTaxonomy.FLAKE: _ConstraintProfile(
        severity=ConstraintSeverity.SHOULD,
        category="reliability",
        checker_binding="reliability_checker",
        auto_accept=False,
        id_tag="FLK",
    ),
    FailureTaxonomy.PERF_REGRESSION: _ConstraintProfile(
        severity=ConstraintSeverity.SHOULD,
        category="performance",
        checker_binding="performance_checker",
        auto_accept=True,
        id_tag="PERF",
    ),
    FailureTaxonomy.SECURITY_FINDING: _ConstraintProfile(
        severity=ConstraintSeverity.MUST,
        category="security",
        checker_binding="security_checker",
        auto_accept=True,
        id_tag="SEC",
    ),
}


@dataclass(frozen=True, slots=True)
class _FailureSignal:
    origin: str
    stage: str
    checker_id: str
    result: EvidenceResult | None
    reason_code: str
    constraint_id: str
    summary: str
    message: str

    def sort_key(
        self,
    ) -> tuple[str, str, str, str, str, str, str, str]:
        return (
            self.origin,
            self.stage,
            self.checker_id,
            self.result.value if self.result is not None else "",
            self.reason_code,
            self.constraint_id,
            self.summary,
            self.message,
        )


class ConstraintMiner:
    """Offline deterministic miner with ergonomic ``mine`` + ``apply`` APIs."""

    def __init__(
        self,
        *,
        registry: ConstraintRegistry | None = None,
        registry_dir: PathLike = Path("constraints") / "registry",
        miner_version: str = DEFAULT_MINER_VERSION,
    ) -> None:
        self._registry = registry
        self._registry_dir = (
            registry.registry_dir if registry is not None else Path(registry_dir).expanduser()
        )
        normalized_version = miner_version.strip()
        if not normalized_version:
            raise ValueError("miner_version must be non-empty")
        self._miner_version = normalized_version

    @property
    def miner_version(self) -> str:
        return self._miner_version

    def mine(
        self,
        *,
        gate_decision: ConstraintGateDecision,
        pipeline_output: (
            PipelineOutput
            | Mapping[str, object]
            | Sequence[PipelineCheckResult | Mapping[str, object]]
        ),
        work_item: WorkItem | Mapping[str, object] | None = None,
        attempt: Attempt | Mapping[str, object] | None = None,
        registry: ConstraintRegistry | None = None,
    ) -> tuple[ConstraintProposal, ...]:
        """Mine deterministic proposals from a gate decision and pipeline output/results."""

        parsed_gate = _coerce_gate_decision(gate_decision)
        check_results = _coerce_pipeline_results(pipeline_output)
        parsed_work_item = _coerce_work_item(work_item)
        parsed_attempt = _coerce_attempt(attempt)
        target_registry = self._resolve_registry(registry)

        existing_constraint_ids = {constraint.id for constraint in target_registry.constraints}
        existing_signatures = _collect_existing_semantic_signatures(target_registry.constraints)

        seen_constraint_ids: set[str] = set()
        seen_signatures: set[str] = set()
        created_at = _resolve_created_at(parsed_work_item, parsed_attempt)
        requirement_links = _resolve_requirement_links(parsed_work_item)
        work_item_id = parsed_work_item.id if parsed_work_item is not None else None
        attempt_id = parsed_attempt.id if parsed_attempt is not None else None

        proposals: list[ConstraintProposal] = []
        for signal in _collect_failure_signals(parsed_gate, check_results):
            failure_type = _classify_failure(signal)
            profile = _PROFILE_BY_FAILURE[failure_type]
            semantic_signature = _build_semantic_signature(signal, failure_type)
            constraint_id = _build_constraint_id(
                id_tag=profile.id_tag,
                semantic_signature=semantic_signature,
            )

            if constraint_id in existing_constraint_ids or constraint_id in seen_constraint_ids:
                continue
            if semantic_signature in existing_signatures or semantic_signature in seen_signatures:
                continue

            checker_binding = _select_checker_binding(failure_type, signal, profile.checker_binding)
            proposed_constraint = Constraint(
                id=constraint_id,
                severity=profile.severity,
                category=profile.category,
                description=_build_description(failure_type, signal),
                checker_binding=checker_binding,
                parameters=_build_constraint_parameters(failure_type, signal),
                requirement_links=requirement_links,
                source=ConstraintSource.FAILURE_DERIVED,
                created_at=created_at,
            )

            source_failure_id = _build_source_failure_id(
                semantic_signature=semantic_signature,
                work_item_id=work_item_id,
                attempt_id=attempt_id,
            )
            proposal = ConstraintProposal(
                proposed_constraint=proposed_constraint,
                justification=_build_justification(signal),
                source_failure_id=source_failure_id,
                auto_accept=profile.auto_accept,
                semantic_signature=semantic_signature,
                work_item_id=work_item_id,
                attempt_id=attempt_id,
            )

            seen_constraint_ids.add(proposal.proposed_constraint.id)
            seen_signatures.add(proposal.semantic_signature)
            proposals.append(proposal)

        return tuple(
            sorted(
                proposals,
                key=lambda item: (
                    item.proposed_constraint.id,
                    item.semantic_signature,
                    item.source_failure_id,
                ),
            )
        )

    def apply(
        self,
        proposals: Sequence[ConstraintProposal],
        *,
        registry: ConstraintRegistry | None = None,
        include_non_auto_accept: bool = False,
        current_date: date | None = None,
    ) -> tuple[Path, ...]:
        """Persist deduplicated proposals as new registry YAML files."""

        target_registry = self._resolve_registry(registry)
        existing_constraint_ids = {constraint.id for constraint in target_registry.constraints}
        existing_signatures = _collect_existing_semantic_signatures(target_registry.constraints)

        seen_constraint_ids: set[str] = set()
        seen_signatures: set[str] = set()
        persisted_paths: list[Path] = []

        ordered = tuple(
            sorted(
                proposals,
                key=lambda item: (
                    item.proposed_constraint.id,
                    item.semantic_signature,
                    item.source_failure_id,
                ),
            )
        )
        for proposal in ordered:
            if not include_non_auto_accept and not proposal.auto_accept:
                continue

            constraint_id = proposal.proposed_constraint.id
            signature = proposal.semantic_signature
            if constraint_id in existing_constraint_ids or constraint_id in seen_constraint_ids:
                continue
            if signature in existing_signatures or signature in seen_signatures:
                continue

            with_provenance = _constraint_with_provenance(
                proposal=proposal,
                miner_version=self._miner_version,
            )
            try:
                destination = target_registry.add_constraint(
                    with_provenance,
                    current_date=current_date,
                )
            except ValueError as exc:
                if "constraint id already exists in registry" in str(exc):
                    existing_constraint_ids.add(constraint_id)
                    continue
                raise

            existing_constraint_ids.add(constraint_id)
            existing_signatures.add(signature)
            seen_constraint_ids.add(constraint_id)
            seen_signatures.add(signature)
            persisted_paths.append(destination)

        self._registry = target_registry
        return tuple(persisted_paths)

    def _resolve_registry(self, override: ConstraintRegistry | None) -> ConstraintRegistry:
        if override is not None:
            return override
        if self._registry is None:
            self._registry = ConstraintRegistry.load(self._registry_dir)
        return self._registry


def _coerce_gate_decision(value: ConstraintGateDecision) -> ConstraintGateDecision:
    if not isinstance(value, ConstraintGateDecision):
        raise TypeError(
            f"gate_decision: expected ConstraintGateDecision, got {type(value).__name__}"
        )
    return value


def _coerce_pipeline_results(
    value: PipelineOutput
    | Mapping[str, object]
    | Sequence[PipelineCheckResult | Mapping[str, object]],
) -> tuple[PipelineCheckResult, ...]:
    if isinstance(value, PipelineOutput):
        results = value.check_results
    elif isinstance(value, Mapping):
        results = PipelineOutput.from_mapping(value).check_results
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parsed: list[PipelineCheckResult] = []
        for index, item in enumerate(value):
            if isinstance(item, PipelineCheckResult):
                parsed.append(item)
            elif isinstance(item, Mapping):
                parsed.append(PipelineCheckResult.from_mapping(item))
            else:
                raise TypeError(
                    f"pipeline_output[{index}]: expected PipelineCheckResult or mapping, "
                    f"got {type(item).__name__}"
                )
        results = tuple(parsed)
    else:
        raise TypeError(
            "pipeline_output: expected PipelineOutput, mapping, or sequence of "
            f"PipelineCheckResult values (got {type(value).__name__})"
        )

    return tuple(sorted(results, key=_check_result_sort_key))


def _coerce_work_item(value: WorkItem | Mapping[str, object] | None) -> WorkItem | None:
    if value is None:
        return None
    if isinstance(value, WorkItem):
        return value
    if isinstance(value, Mapping):
        return WorkItem.from_dict(value)
    raise TypeError(f"work_item: expected WorkItem, mapping, or None (got {type(value).__name__})")


def _coerce_attempt(value: Attempt | Mapping[str, object] | None) -> Attempt | None:
    if value is None:
        return None
    if isinstance(value, Attempt):
        return value
    if isinstance(value, Mapping):
        return Attempt.from_dict(value)
    raise TypeError(f"attempt: expected Attempt, mapping, or None (got {type(value).__name__})")


def _check_result_sort_key(
    item: PipelineCheckResult,
) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...], str]:
    return (
        item.stage,
        item.checker_id,
        item.result.value,
        item.covered_constraint_ids,
        item.required_constraint_ids,
        item.summary or "",
    )


def _collect_failure_signals(
    gate_decision: ConstraintGateDecision,
    check_results: Sequence[PipelineCheckResult],
) -> tuple[_FailureSignal, ...]:
    collected: list[_FailureSignal] = []

    for result in check_results:
        if result.result is EvidenceResult.PASS:
            continue
        collected.append(
            _FailureSignal(
                origin="pipeline_result",
                stage=result.stage,
                checker_id=result.checker_id,
                result=result.result,
                reason_code="pipeline_result",
                constraint_id="",
                summary=(result.summary or "").strip(),
                message="",
            )
        )

    diagnostics = gate_decision.diagnostics
    for reason_code in gate_decision.reason_codes:
        collected.append(
            _FailureSignal(
                origin="gate_reason",
                stage="",
                checker_id="",
                result=None,
                reason_code=reason_code,
                constraint_id="",
                summary="",
                message="",
            )
        )

    for constraint_id in diagnostics.missing_checker_mappings:
        collected.append(
            _FailureSignal(
                origin="gate_diagnostic",
                stage="",
                checker_id="",
                result=None,
                reason_code="missing_checker_mapping",
                constraint_id=constraint_id,
                summary="",
                message="",
            )
        )
    for constraint_id in diagnostics.unresolved_should_constraints:
        collected.append(
            _FailureSignal(
                origin="gate_diagnostic",
                stage="",
                checker_id="",
                result=None,
                reason_code="should_constraint_unsatisfied",
                constraint_id=constraint_id,
                summary="",
                message="",
            )
        )
    for stage in diagnostics.missing_required_stages:
        collected.append(
            _FailureSignal(
                origin="gate_diagnostic",
                stage=stage,
                checker_id="",
                result=None,
                reason_code="missing_required_stage",
                constraint_id="",
                summary="",
                message="",
            )
        )
    for stage in diagnostics.failing_required_stages:
        collected.append(
            _FailureSignal(
                origin="gate_diagnostic",
                stage=stage,
                checker_id="",
                result=None,
                reason_code="required_stage_failed",
                constraint_id="",
                summary="",
                message="",
            )
        )
    for gap in diagnostics.stage_coverage_gaps:
        collected.append(
            _FailureSignal(
                origin="gate_diagnostic",
                stage=gap.stage,
                checker_id="",
                result=None,
                reason_code="stage_coverage_incomplete",
                constraint_id=gap.constraint_id,
                summary="",
                message="",
            )
        )

    for violation in sorted(
        gate_decision.aggregated_violations,
        key=lambda item: (
            item.code,
            item.constraint_id or "",
            item.stage or "",
            item.checker_id or "",
            item.message,
            item.severity,
        ),
    ):
        collected.append(
            _FailureSignal(
                origin="gate_violation",
                stage=(violation.stage or "").strip(),
                checker_id=(violation.checker_id or "").strip(),
                result=None,
                reason_code=violation.code.strip(),
                constraint_id=(violation.constraint_id or "").strip(),
                summary="",
                message=violation.message.strip(),
            )
        )

    deduped: dict[tuple[str, str, str, str, str, str, str, str], _FailureSignal] = {}
    for signal in collected:
        deduped[signal.sort_key()] = signal
    return tuple(deduped[key] for key in sorted(deduped))


def _classify_failure(signal: _FailureSignal) -> FailureTaxonomy:
    searchable = " ".join(
        (
            signal.origin,
            signal.stage,
            signal.checker_id,
            signal.reason_code,
            signal.constraint_id,
            signal.summary,
            signal.message,
        )
    ).lower()

    if _contains_any(searchable, _SECURITY_MARKERS):
        return FailureTaxonomy.SECURITY_FINDING
    if _contains_any(searchable, _PERFORMANCE_MARKERS):
        return FailureTaxonomy.PERF_REGRESSION
    if signal.reason_code in _SPEC_GAP_REASON_CODES:
        return FailureTaxonomy.SPEC_GAP
    if _contains_any(searchable, _FLAKE_MARKERS):
        return FailureTaxonomy.FLAKE
    if signal.result in {EvidenceResult.WARN, EvidenceResult.SKIP}:
        return FailureTaxonomy.FLAKE
    return FailureTaxonomy.BUG


def _contains_any(value: str, markers: Iterable[str]) -> bool:
    return any(marker in value for marker in markers)


def _build_semantic_signature(signal: _FailureSignal, failure_type: FailureTaxonomy) -> str:
    payload = {
        "failure_type": failure_type.value,
        "origin": signal.origin,
        "stage": signal.stage,
        "checker_id": signal.checker_id,
        "result": signal.result.value if signal.result is not None else "",
        "reason_code": signal.reason_code,
        "constraint_id": signal.constraint_id,
        "summary": signal.summary,
        "message": signal.message,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _build_constraint_id(*, id_tag: str, semantic_signature: str) -> str:
    sequence = (int(semantic_signature[:8], 16) % 9_999) + 1
    return f"CON-{id_tag}-{sequence:04d}"


def _select_checker_binding(
    failure_type: FailureTaxonomy,
    signal: _FailureSignal,
    default_checker: str,
) -> str:
    if (
        failure_type
        in {
            FailureTaxonomy.BUG,
            FailureTaxonomy.PERF_REGRESSION,
            FailureTaxonomy.SECURITY_FINDING,
        }
        and signal.checker_id
    ):
        return signal.checker_id
    return default_checker


def _build_description(failure_type: FailureTaxonomy, signal: _FailureSignal) -> str:
    prefix_by_type = {
        FailureTaxonomy.BUG: "Prevent recurrence of a bug detected by verification",
        FailureTaxonomy.SPEC_GAP: "Close a specification gap surfaced by gate diagnostics",
        FailureTaxonomy.FLAKE: "Stabilize flaky behavior observed in verification",
        FailureTaxonomy.PERF_REGRESSION: "Prevent performance regression from recurring",
        FailureTaxonomy.SECURITY_FINDING: "Prevent security finding from recurring",
    }
    context: list[str] = []
    if signal.stage:
        context.append(f"stage={signal.stage}")
    if signal.checker_id:
        context.append(f"checker={signal.checker_id}")
    if signal.reason_code and signal.reason_code != "pipeline_result":
        context.append(f"signal={signal.reason_code}")
    if signal.constraint_id:
        context.append(f"constraint={signal.constraint_id}")

    base = prefix_by_type[failure_type]
    if not context:
        return base
    return f"{base} ({', '.join(context)})"


def _build_justification(signal: _FailureSignal) -> str:
    parts: list[str] = []
    if signal.message:
        parts.append(signal.message)
    if signal.summary and signal.summary not in parts:
        parts.append(signal.summary)
    if signal.reason_code and signal.reason_code != "pipeline_result":
        parts.append(f"gate_signal={signal.reason_code}")
    if not parts:
        parts.append("Derived from deterministic offline failure mining.")
    return " | ".join(parts)


def _build_constraint_parameters(
    failure_type: FailureTaxonomy,
    signal: _FailureSignal,
) -> dict[str, JSONValue]:
    signal_payload: dict[str, JSONValue] = {
        "origin": signal.origin,
        "stage": signal.stage,
        "checker_id": signal.checker_id,
        "result": signal.result.value if signal.result is not None else "",
        "reason_code": signal.reason_code,
        "constraint_id": signal.constraint_id,
    }
    if signal.summary:
        signal_payload["summary"] = signal.summary
    if signal.message:
        signal_payload["message"] = signal.message
    return {
        "failure_category": failure_type.value,
        "signal": signal_payload,
    }


def _resolve_requirement_links(work_item: WorkItem | None) -> tuple[str, ...]:
    if work_item is None:
        return ()
    return tuple(sorted(set(work_item.requirement_links)))


def _resolve_created_at(work_item: WorkItem | None, attempt: Attempt | None) -> datetime:
    if attempt is not None:
        return attempt.created_at
    if work_item is not None:
        return work_item.updated_at
    return _EPOCH_UTC


def _build_source_failure_id(
    *,
    semantic_signature: str,
    work_item_id: str | None,
    attempt_id: str | None,
) -> str:
    payload = {
        "semantic_signature": semantic_signature,
        "work_item_id": work_item_id or "",
        "attempt_id": attempt_id or "",
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:12].upper()
    return f"FAIL-{digest}"


def _collect_existing_semantic_signatures(
    constraints: Sequence[Constraint],
) -> set[str]:
    signatures: set[str] = set()
    for constraint in constraints:
        signature = _extract_semantic_signature(constraint.parameters)
        if signature is not None:
            signatures.add(signature)
    return signatures


def _extract_semantic_signature(parameters: Mapping[str, object]) -> str | None:
    direct = parameters.get("semantic_signature")
    if isinstance(direct, str):
        normalized = direct.strip()
        if normalized:
            return normalized

    provenance_raw = parameters.get("provenance")
    if isinstance(provenance_raw, Mapping):
        nested = provenance_raw.get("semantic_signature")
        if isinstance(nested, str):
            normalized_nested = nested.strip()
            if normalized_nested:
                return normalized_nested
    return None


def _constraint_with_provenance(proposal: ConstraintProposal, miner_version: str) -> Constraint:
    parameters = _clone_json_object(proposal.proposed_constraint.parameters)

    provenance: dict[str, JSONValue] = {}
    existing_provenance = parameters.get("provenance")
    if isinstance(existing_provenance, Mapping):
        for key, value in sorted(existing_provenance.items(), key=lambda item: str(item[0])):
            if isinstance(key, str):
                provenance[key] = _clone_json_scalar_or_collection(value)

    provenance["source_failure_id"] = proposal.source_failure_id
    provenance["miner_version"] = miner_version
    provenance["semantic_signature"] = proposal.semantic_signature
    if proposal.work_item_id is not None:
        provenance["work_item_id"] = proposal.work_item_id
    if proposal.attempt_id is not None:
        provenance["attempt_id"] = proposal.attempt_id

    parameters["provenance"] = provenance

    return Constraint(
        id=proposal.proposed_constraint.id,
        severity=proposal.proposed_constraint.severity,
        category=proposal.proposed_constraint.category,
        description=proposal.proposed_constraint.description,
        checker_binding=proposal.proposed_constraint.checker_binding,
        parameters=parameters,
        requirement_links=proposal.proposed_constraint.requirement_links,
        source=proposal.proposed_constraint.source,
        created_at=proposal.proposed_constraint.created_at,
    )


def _clone_json_object(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    return {
        str(key): _clone_json_scalar_or_collection(item)
        for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
    }


def _clone_json_scalar_or_collection(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return {
            str(key): _clone_json_scalar_or_collection(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, list):
        return [_clone_json_scalar_or_collection(item) for item in value]
    if isinstance(value, tuple):
        return [_clone_json_scalar_or_collection(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


__all__ = [
    "ConstraintMiner",
    "ConstraintProposal",
    "DEFAULT_MINER_VERSION",
    "FailureTaxonomy",
]
