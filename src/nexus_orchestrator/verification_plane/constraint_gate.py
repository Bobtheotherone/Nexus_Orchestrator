"""
nexus-orchestrator — constraint gate.

File: src/nexus_orchestrator/verification_plane/constraint_gate.py
Last updated: 2026-02-12

Purpose
- Implements the binary accept/reject gate: maps a constraint envelope to pipeline output and
  determines merge eligibility.

What should be included in this file
- Deterministic gate policy over MUST/SHOULD/MAY constraints.
- Evidence completeness checks for required stage coverage.
- Incremental/full stage selection hooks and adversarial stage integration.
- FeedbackSynthesizer-friendly diagnostics payload with stable ordering.

Functional requirements
- Must support incremental verification (affected modules) and periodic full verification runs.
- Must integrate with adversarial tests when required.
- Must enforce no-silent-pass policy for MUST constraints.

Non-functional requirements
- Deterministic output ordering.
- Reliable hard reject behavior for missing checker mappings on MUST constraints.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from nexus_orchestrator.domain import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    EvidenceResult,
)

ADVERSARIAL_STAGE_ID: Final[str] = "adversarial_tests"


class VerificationSelectionMode(str, Enum):
    """Verification scope selection mode."""

    INCREMENTAL = "incremental"
    FULL = "full"


class ConstraintDisposition(str, Enum):
    """Gate disposition for one constraint after policy evaluation."""

    PASS = "pass"
    REJECT = "reject"
    OVERRIDDEN = "overridden"
    INFORMATIVE = "informative"


class GateVerdict(str, Enum):
    """Binary constraint gate verdict."""

    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ConstraintOverrideRecord:
    """Explicit override record for a non-must constraint."""

    constraint_id: str
    justification: str
    approved_by: str
    approved: bool = True
    approval_reference: str | None = None

    @property
    def is_explicit_approval(self) -> bool:
        """Return whether this override satisfies justification + approval requirements."""

        return self.approved and bool(self.justification) and bool(self.approved_by)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ConstraintOverrideRecord:
        """Parse override record from a mapping payload."""

        constraint_id = _require_non_empty_str(payload.get("constraint_id"), "constraint_id")
        justification = _require_non_empty_str(payload.get("justification"), "justification")
        approved_by_raw = payload.get(
            "approved_by",
            payload.get("approver", payload.get("approval")),
        )
        approved_by = _require_non_empty_str(approved_by_raw, "approved_by")
        approved = _coerce_optional_bool(payload.get("approved"), "approved", default=True)
        approval_reference = _coerce_optional_str(
            payload.get("approval_reference"),
            "approval_reference",
        )
        return cls(
            constraint_id=constraint_id,
            justification=justification,
            approved_by=approved_by,
            approved=approved,
            approval_reference=approval_reference,
        )


@dataclass(frozen=True, slots=True)
class PipelineCheckResult:
    """One checker outcome from the verification pipeline."""

    stage: str
    checker_id: str
    result: EvidenceResult
    covered_constraint_ids: tuple[str, ...] = ()
    required_constraint_ids: tuple[str, ...] = ()
    summary: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PipelineCheckResult:
        """Parse checker outcome from a mapping payload."""

        stage = _require_non_empty_str(payload.get("stage"), "stage")
        checker_id = _require_non_empty_str(
            payload.get("checker_id", payload.get("checker")),
            "checker_id",
        )

        result_raw = payload.get("result", payload.get("status"))
        if result_raw is None:
            raise ValueError("result: missing required value")
        result = _coerce_evidence_result(result_raw, "result")

        covered_constraint_ids = _coerce_string_tuple(
            payload.get(
                "covered_constraint_ids",
                payload.get("constraint_ids", payload.get("covered_constraints")),
            ),
            "covered_constraint_ids",
        )
        required_constraint_ids = _coerce_string_tuple(
            payload.get("required_constraint_ids", payload.get("required_constraints")),
            "required_constraint_ids",
        )
        summary = _coerce_optional_str(payload.get("summary"), "summary")

        return cls(
            stage=stage,
            checker_id=checker_id,
            result=result,
            covered_constraint_ids=covered_constraint_ids,
            required_constraint_ids=required_constraint_ids,
            summary=summary,
        )


@dataclass(frozen=True, slots=True)
class StageCoverageRequirement:
    """Required constraint coverage for a stage-level completeness check."""

    stage: str
    constraint_ids: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> StageCoverageRequirement:
        """Parse stage coverage requirements from mapping payload."""

        return cls(
            stage=_require_non_empty_str(payload.get("stage"), "stage"),
            constraint_ids=_coerce_string_tuple(
                payload.get("constraint_ids", payload.get("required_constraint_ids")),
                "constraint_ids",
            ),
        )


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """Normalized verification pipeline output consumed by the constraint gate."""

    check_results: tuple[PipelineCheckResult, ...]
    mode: VerificationSelectionMode = VerificationSelectionMode.INCREMENTAL
    selected_stages: tuple[str, ...] = ()
    required_stages: tuple[str, ...] = ()
    stage_coverage_requirements: tuple[StageCoverageRequirement, ...] = ()
    adversarial_required: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PipelineOutput:
        """Parse pipeline output from a mapping payload."""

        check_results = _coerce_check_results(
            payload.get("check_results", payload.get("results", payload.get("stage_results", ()))),
            "check_results",
        )

        mode = _coerce_mode(
            payload.get("mode", payload.get("selection_mode", "incremental")),
            "mode",
        )

        selected_stages = _coerce_string_tuple(payload.get("selected_stages"), "selected_stages")
        required_stages = _coerce_string_tuple(payload.get("required_stages"), "required_stages")
        adversarial_required = _coerce_optional_bool(
            payload.get("adversarial_required"),
            "adversarial_required",
            default=False,
        )

        stage_requirements = _coerce_stage_coverage_requirements(
            payload.get(
                "stage_coverage_requirements",
                payload.get("required_stage_constraints"),
            ),
            "stage_coverage_requirements",
        )

        return cls(
            check_results=check_results,
            mode=mode,
            selected_stages=selected_stages,
            required_stages=required_stages,
            stage_coverage_requirements=stage_requirements,
            adversarial_required=adversarial_required,
        )


@dataclass(frozen=True, slots=True)
class ConstraintEvidenceLink:
    """Stable pointer to pass evidence for one constraint."""

    stage: str
    checker_id: str


@dataclass(frozen=True, slots=True)
class GateViolation:
    """Aggregated gate-level violation used for strict diagnostics."""

    code: str
    message: str
    severity: str = "error"
    constraint_id: str | None = None
    stage: str | None = None
    checker_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConstraintAssessment:
    """Deterministic constraint-level gate assessment."""

    constraint_id: str
    severity: ConstraintSeverity
    checker_binding: str
    disposition: ConstraintDisposition
    blocking: bool
    has_checker_mapping: bool
    covered_by_pass: bool
    override_applied: bool
    override_approved_by: str | None
    override_justification: str | None
    reasons: tuple[str, ...]
    pass_evidence: tuple[ConstraintEvidenceLink, ...]


@dataclass(frozen=True, slots=True)
class StageCoverageGap:
    """Coverage gap for a passed stage that lacked required constraint evidence."""

    stage: str
    constraint_id: str


@dataclass(frozen=True, slots=True)
class StageAssessment:
    """Deterministic stage-level assessment for diagnostics."""

    stage: str
    required: bool
    passed: bool
    check_count: int
    covered_constraint_ids: tuple[str, ...]
    required_constraint_ids: tuple[str, ...]
    missing_constraint_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConstraintGateDiagnostics:
    """Machine-readable diagnostics payload for feedback synthesis."""

    mode: VerificationSelectionMode
    selected_stages: tuple[str, ...]
    required_stages: tuple[str, ...]
    missing_required_stages: tuple[str, ...]
    failing_required_stages: tuple[str, ...]
    stage_coverage_gaps: tuple[StageCoverageGap, ...]
    missing_checker_mappings: tuple[str, ...]
    uncovered_must_constraints: tuple[str, ...]
    unresolved_should_constraints: tuple[str, ...]
    overridden_should_constraints: tuple[str, ...]
    informative_may_constraints: tuple[str, ...]
    invalid_override_records: tuple[str, ...]
    stage_assessments: tuple[StageAssessment, ...]
    constraint_assessments: tuple[ConstraintAssessment, ...]

    def to_feedback_payload(self) -> dict[str, object]:
        """Return deterministic payload for FeedbackSynthesizer."""

        return {
            "mode": self.mode.value,
            "selected_stages": list(self.selected_stages),
            "required_stages": list(self.required_stages),
            "missing_required_stages": list(self.missing_required_stages),
            "failing_required_stages": list(self.failing_required_stages),
            "stage_coverage_gaps": [
                {
                    "stage": gap.stage,
                    "constraint_id": gap.constraint_id,
                }
                for gap in self.stage_coverage_gaps
            ],
            "missing_checker_mappings": list(self.missing_checker_mappings),
            "uncovered_must_constraints": list(self.uncovered_must_constraints),
            "unresolved_should_constraints": list(self.unresolved_should_constraints),
            "overridden_should_constraints": list(self.overridden_should_constraints),
            "informative_may_constraints": list(self.informative_may_constraints),
            "invalid_override_records": list(self.invalid_override_records),
            "stage_assessments": [
                {
                    "stage": stage.stage,
                    "required": stage.required,
                    "passed": stage.passed,
                    "check_count": stage.check_count,
                    "covered_constraint_ids": list(stage.covered_constraint_ids),
                    "required_constraint_ids": list(stage.required_constraint_ids),
                    "missing_constraint_ids": list(stage.missing_constraint_ids),
                }
                for stage in self.stage_assessments
            ],
            "constraint_assessments": [
                {
                    "constraint_id": assessment.constraint_id,
                    "severity": assessment.severity.value,
                    "checker_binding": assessment.checker_binding,
                    "disposition": assessment.disposition.value,
                    "blocking": assessment.blocking,
                    "has_checker_mapping": assessment.has_checker_mapping,
                    "covered_by_pass": assessment.covered_by_pass,
                    "override_applied": assessment.override_applied,
                    "override_approved_by": assessment.override_approved_by,
                    "override_justification": assessment.override_justification,
                    "reasons": list(assessment.reasons),
                    "pass_evidence": [
                        {"stage": link.stage, "checker_id": link.checker_id}
                        for link in assessment.pass_evidence
                    ],
                }
                for assessment in self.constraint_assessments
            ],
            "summary": {
                "constraints_total": len(self.constraint_assessments),
                "must_uncovered": len(self.uncovered_must_constraints),
                "should_unresolved": len(self.unresolved_should_constraints),
                "stage_coverage_gap_count": len(self.stage_coverage_gaps),
            },
        }


@dataclass(frozen=True, slots=True)
class ConstraintGateDecision:
    """Binary gate decision and deterministic diagnostics."""

    verdict: GateVerdict
    reason_codes: tuple[str, ...]
    diagnostics: ConstraintGateDiagnostics
    evidence_records: tuple[ConstraintEvidenceLink, ...] = ()
    aggregated_violations: tuple[GateViolation, ...] = ()

    @property
    def accepted(self) -> bool:
        """Return whether gate accepted the candidate."""

        return self.verdict is GateVerdict.ACCEPT

    def to_feedback_payload(self) -> dict[str, object]:
        """Return deterministic payload for feedback synthesis."""

        payload = self.diagnostics.to_feedback_payload()
        payload["verdict"] = self.verdict.value
        payload["accepted"] = self.accepted
        payload["reason_codes"] = list(self.reason_codes)
        payload["evidence_records"] = [
            {"stage": link.stage, "checker_id": link.checker_id} for link in self.evidence_records
        ]
        payload["aggregated_violations"] = [
            {
                "code": violation.code,
                "message": violation.message,
                "severity": violation.severity,
                "constraint_id": violation.constraint_id,
                "stage": violation.stage,
                "checker_id": violation.checker_id,
            }
            for violation in self.aggregated_violations
        ]
        return payload


def select_verification_mode(
    *,
    full_verification_requested: bool = False,
    periodic_full_due: bool = False,
) -> VerificationSelectionMode:
    """Select incremental vs full verification mode."""

    if full_verification_requested or periodic_full_due:
        return VerificationSelectionMode.FULL
    return VerificationSelectionMode.INCREMENTAL


def select_stage_plan(
    *,
    mode: VerificationSelectionMode | str,
    available_stages: Sequence[str],
    required_stages: Iterable[str] = (),
    incremental_stages: Iterable[str] | None = None,
    full_stages: Iterable[str] | None = None,
    adversarial_required: bool = False,
    adversarial_stage: str = ADVERSARIAL_STAGE_ID,
) -> tuple[str, ...]:
    """Build deterministic stage selection plan for incremental/full verification."""

    normalized_mode = _coerce_mode(mode, "mode")
    available = _coerce_stage_sequence(available_stages, "available_stages")
    available_lookup = set(available)

    if normalized_mode is VerificationSelectionMode.FULL:
        seed = (
            available if full_stages is None else _coerce_stage_sequence(full_stages, "full_stages")
        )
    else:
        seed = (
            available
            if incremental_stages is None
            else _coerce_stage_sequence(incremental_stages, "incremental_stages")
        )

    selected = set(seed)
    selected.update(_coerce_stage_sequence(required_stages, "required_stages"))
    if adversarial_required:
        selected.add(_require_non_empty_str(adversarial_stage, "adversarial_stage"))

    ordered: list[str] = [stage for stage in available if stage in selected]
    remainder = sorted(selected.difference(available_lookup))
    ordered.extend(remainder)
    return tuple(ordered)


def run_constraint_gate(
    constraint_envelope: ConstraintEnvelope | Mapping[str, object],
    pipeline_output: PipelineOutput | Mapping[str, object],
    *,
    override_records: Iterable[ConstraintOverrideRecord | Mapping[str, object]] = (),
) -> ConstraintGateDecision:
    """Run binary gate evaluation over constraints and pipeline output."""

    return evaluate_constraint_gate(
        constraint_envelope,
        pipeline_output,
        override_records=override_records,
    )


def evaluate_constraint_gate(
    constraint_envelope: ConstraintEnvelope | Mapping[str, object],
    pipeline_output: PipelineOutput | Mapping[str, object],
    *,
    override_records: Iterable[ConstraintOverrideRecord | Mapping[str, object]] = (),
) -> ConstraintGateDecision:
    """Evaluate constraints against pipeline evidence with deterministic diagnostics."""

    envelope = _coerce_constraint_envelope(constraint_envelope)
    normalized_output = _coerce_pipeline_output(pipeline_output)
    parsed_overrides, invalid_override_records = _coerce_override_records(override_records)
    override_map = _index_overrides(parsed_overrides)

    check_results = tuple(
        sorted(normalized_output.check_results, key=_check_result_sort_key),
    )

    checker_results: dict[str, list[PipelineCheckResult]] = defaultdict(list)
    stage_results: dict[str, list[PipelineCheckResult]] = defaultdict(list)
    for result in check_results:
        checker_results[result.checker_id].append(result)
        stage_results[result.stage].append(result)

    required_stages = _resolve_required_stages(normalized_output)
    selected_stage_set = set(normalized_output.selected_stages)
    selected_stage_set.update(stage_results.keys())
    selected_stages = _sorted_unique_strings(selected_stage_set)

    stage_required_constraints = _collect_stage_required_constraints(
        normalized_output, check_results
    )

    all_stage_ids = set(selected_stages)
    all_stage_ids.update(required_stages)
    all_stage_ids.update(stage_required_constraints.keys())
    stage_assessments, missing_required_stages, failing_required_stages, stage_coverage_gaps = (
        _evaluate_stages(
            all_stage_ids=all_stage_ids,
            required_stages=required_stages,
            stage_results=stage_results,
            stage_required_constraints=stage_required_constraints,
        )
    )

    constraint_assessments: list[ConstraintAssessment] = []
    missing_checker_mappings: set[str] = set()
    uncovered_must_constraints: list[str] = []
    unresolved_should_constraints: list[str] = []
    overridden_should_constraints: list[str] = []
    informative_may_constraints: list[str] = []

    for constraint in sorted(envelope.constraints, key=lambda item: item.id):
        assessment = _evaluate_constraint(
            constraint=constraint,
            checker_results=checker_results,
            override_record=override_map.get(constraint.id),
        )
        constraint_assessments.append(assessment)

        if not assessment.has_checker_mapping:
            missing_checker_mappings.add(constraint.id)

        if constraint.severity is ConstraintSeverity.MUST:
            if assessment.disposition is ConstraintDisposition.REJECT:
                uncovered_must_constraints.append(constraint.id)
        elif constraint.severity is ConstraintSeverity.SHOULD:
            if assessment.disposition is ConstraintDisposition.REJECT:
                unresolved_should_constraints.append(constraint.id)
            elif assessment.disposition is ConstraintDisposition.OVERRIDDEN:
                overridden_should_constraints.append(constraint.id)
        else:
            informative_may_constraints.append(constraint.id)

    reason_codes = _build_reason_codes(
        missing_required_stages=missing_required_stages,
        failing_required_stages=failing_required_stages,
        stage_coverage_gaps=stage_coverage_gaps,
        uncovered_must_constraints=tuple(uncovered_must_constraints),
        unresolved_should_constraints=tuple(unresolved_should_constraints),
    )

    verdict = GateVerdict.ACCEPT if not reason_codes else GateVerdict.REJECT

    diagnostics = ConstraintGateDiagnostics(
        mode=normalized_output.mode,
        selected_stages=selected_stages,
        required_stages=required_stages,
        missing_required_stages=missing_required_stages,
        failing_required_stages=failing_required_stages,
        stage_coverage_gaps=stage_coverage_gaps,
        missing_checker_mappings=_sorted_unique_strings(missing_checker_mappings),
        uncovered_must_constraints=tuple(uncovered_must_constraints),
        unresolved_should_constraints=tuple(unresolved_should_constraints),
        overridden_should_constraints=tuple(overridden_should_constraints),
        informative_may_constraints=tuple(informative_may_constraints),
        invalid_override_records=invalid_override_records,
        stage_assessments=stage_assessments,
        constraint_assessments=tuple(constraint_assessments),
    )
    evidence_records = _collect_evidence_records(tuple(constraint_assessments))
    aggregated_violations = _collect_gate_violations(
        diagnostics=diagnostics,
        constraint_assessments=tuple(constraint_assessments),
    )

    return ConstraintGateDecision(
        verdict=verdict,
        reason_codes=reason_codes,
        diagnostics=diagnostics,
        evidence_records=evidence_records,
        aggregated_violations=aggregated_violations,
    )


def _evaluate_stages(
    *,
    all_stage_ids: set[str],
    required_stages: tuple[str, ...],
    stage_results: Mapping[str, Sequence[PipelineCheckResult]],
    stage_required_constraints: Mapping[str, tuple[str, ...]],
) -> tuple[
    tuple[StageAssessment, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[StageCoverageGap, ...],
]:
    required_stage_lookup = set(required_stages)

    stage_assessments: list[StageAssessment] = []
    missing_required_stages: list[str] = []
    failing_required_stages: list[str] = []
    stage_coverage_gap_pairs: set[tuple[str, str]] = set()

    for stage_id in _sorted_unique_strings(all_stage_ids):
        stage_result_items = tuple(stage_results.get(stage_id, ()))
        stage_passed = bool(stage_result_items) and all(
            item.result is EvidenceResult.PASS for item in stage_result_items
        )

        covered_constraint_ids = _sorted_unique_strings(
            constraint_id
            for item in stage_result_items
            if item.result is EvidenceResult.PASS
            for constraint_id in item.covered_constraint_ids
        )

        required_constraint_ids = stage_required_constraints.get(stage_id, ())
        covered_lookup = set(covered_constraint_ids)
        missing_constraint_ids = tuple(
            constraint_id
            for constraint_id in required_constraint_ids
            if constraint_id not in covered_lookup
        )

        is_required = stage_id in required_stage_lookup
        if is_required:
            if not stage_result_items:
                missing_required_stages.append(stage_id)
            elif not stage_passed:
                failing_required_stages.append(stage_id)

        if stage_passed:
            for constraint_id in missing_constraint_ids:
                stage_coverage_gap_pairs.add((stage_id, constraint_id))

        stage_assessments.append(
            StageAssessment(
                stage=stage_id,
                required=is_required,
                passed=stage_passed,
                check_count=len(stage_result_items),
                covered_constraint_ids=covered_constraint_ids,
                required_constraint_ids=required_constraint_ids,
                missing_constraint_ids=missing_constraint_ids,
            )
        )

    stage_coverage_gaps = tuple(
        StageCoverageGap(stage=stage_id, constraint_id=constraint_id)
        for stage_id, constraint_id in sorted(stage_coverage_gap_pairs)
    )

    return (
        tuple(stage_assessments),
        tuple(missing_required_stages),
        tuple(failing_required_stages),
        stage_coverage_gaps,
    )


def _evaluate_constraint(
    *,
    constraint: Constraint,
    checker_results: Mapping[str, Sequence[PipelineCheckResult]],
    override_record: ConstraintOverrideRecord | None,
) -> ConstraintAssessment:
    mapped_results = tuple(checker_results.get(constraint.checker_binding, ()))
    has_checker_mapping = bool(mapped_results)

    pass_evidence_pairs = sorted(
        {
            (result.stage, result.checker_id)
            for result in mapped_results
            if result.result is EvidenceResult.PASS
            and constraint.id in result.covered_constraint_ids
        }
    )
    pass_evidence = tuple(
        ConstraintEvidenceLink(stage=stage_id, checker_id=checker_id)
        for stage_id, checker_id in pass_evidence_pairs
    )
    covered_by_pass = bool(pass_evidence)

    reasons: list[str] = []
    if not has_checker_mapping:
        reasons.append("missing_checker_mapping")
    elif not covered_by_pass:
        reasons.append("uncovered_by_pass")

    disposition: ConstraintDisposition
    blocking: bool
    override_applied = False
    override_approved_by: str | None = None
    override_justification: str | None = None

    if constraint.severity is ConstraintSeverity.MUST:
        if has_checker_mapping and covered_by_pass:
            disposition = ConstraintDisposition.PASS
            blocking = False
        else:
            disposition = ConstraintDisposition.REJECT
            blocking = True
    elif constraint.severity is ConstraintSeverity.SHOULD:
        if has_checker_mapping and covered_by_pass:
            disposition = ConstraintDisposition.PASS
            blocking = False
        elif override_record is not None and override_record.is_explicit_approval:
            disposition = ConstraintDisposition.OVERRIDDEN
            blocking = False
            override_applied = True
            override_approved_by = override_record.approved_by
            override_justification = override_record.justification
            reasons.append("override_applied")
        else:
            disposition = ConstraintDisposition.REJECT
            blocking = True
    else:
        if has_checker_mapping and covered_by_pass:
            disposition = ConstraintDisposition.PASS
        else:
            disposition = ConstraintDisposition.INFORMATIVE
            reasons.append("informative_only")
        blocking = False

    return ConstraintAssessment(
        constraint_id=constraint.id,
        severity=constraint.severity,
        checker_binding=constraint.checker_binding,
        disposition=disposition,
        blocking=blocking,
        has_checker_mapping=has_checker_mapping,
        covered_by_pass=covered_by_pass,
        override_applied=override_applied,
        override_approved_by=override_approved_by,
        override_justification=override_justification,
        reasons=tuple(sorted(set(reasons))),
        pass_evidence=pass_evidence,
    )


def _build_reason_codes(
    *,
    missing_required_stages: tuple[str, ...],
    failing_required_stages: tuple[str, ...],
    stage_coverage_gaps: tuple[StageCoverageGap, ...],
    uncovered_must_constraints: tuple[str, ...],
    unresolved_should_constraints: tuple[str, ...],
) -> tuple[str, ...]:
    reason_codes: list[str] = []
    if missing_required_stages:
        reason_codes.append("missing_required_stage")
    if failing_required_stages:
        reason_codes.append("required_stage_failed")
    if stage_coverage_gaps:
        reason_codes.append("stage_coverage_incomplete")
    if uncovered_must_constraints:
        reason_codes.append("must_constraint_unsatisfied")
    if unresolved_should_constraints:
        reason_codes.append("should_constraint_requires_override")
    return tuple(reason_codes)


def _collect_evidence_records(
    assessments: Sequence[ConstraintAssessment],
) -> tuple[ConstraintEvidenceLink, ...]:
    unique_pairs = sorted(
        {
            (link.stage, link.checker_id)
            for assessment in assessments
            for link in assessment.pass_evidence
        }
    )
    return tuple(
        ConstraintEvidenceLink(stage=stage, checker_id=checker_id)
        for stage, checker_id in unique_pairs
    )


def _collect_gate_violations(
    *,
    diagnostics: ConstraintGateDiagnostics,
    constraint_assessments: Sequence[ConstraintAssessment],
) -> tuple[GateViolation, ...]:
    collected: list[GateViolation] = []

    for stage_id in diagnostics.missing_required_stages:
        collected.append(
            GateViolation(
                code="gate.required_stage_missing",
                message=f"required stage missing: {stage_id}",
                stage=stage_id,
            )
        )
    for stage_id in diagnostics.failing_required_stages:
        collected.append(
            GateViolation(
                code="gate.required_stage_failed",
                message=f"required stage failed: {stage_id}",
                stage=stage_id,
            )
        )
    for gap in diagnostics.stage_coverage_gaps:
        collected.append(
            GateViolation(
                code="gate.stage_coverage_incomplete",
                message=(
                    "stage passed without required coverage: "
                    f"stage={gap.stage} constraint={gap.constraint_id}"
                ),
                stage=gap.stage,
                constraint_id=gap.constraint_id,
            )
        )

    for constraint_id in diagnostics.missing_checker_mappings:
        collected.append(
            GateViolation(
                code="gate.missing_checker_mapping",
                message=f"missing checker mapping evidence for {constraint_id}",
                constraint_id=constraint_id,
            )
        )
    for constraint_id in diagnostics.uncovered_must_constraints:
        collected.append(
            GateViolation(
                code="gate.must_constraint_unsatisfied",
                message=f"must constraint unsatisfied: {constraint_id}",
                constraint_id=constraint_id,
            )
        )
    for constraint_id in diagnostics.unresolved_should_constraints:
        collected.append(
            GateViolation(
                code="gate.should_constraint_unsatisfied",
                message=f"should constraint unsatisfied without override: {constraint_id}",
                constraint_id=constraint_id,
            )
        )

    assessment_by_id = {item.constraint_id: item for item in constraint_assessments}
    for constraint_id in diagnostics.overridden_should_constraints:
        assessment = assessment_by_id.get(constraint_id)
        checker_id = assessment.checker_binding if assessment is not None else None
        collected.append(
            GateViolation(
                code="gate.should_constraint_overridden",
                message=f"should constraint overridden by approved record: {constraint_id}",
                severity="warning",
                constraint_id=constraint_id,
                checker_id=checker_id,
            )
        )

    deduped: dict[tuple[str, str, str, str, str, str], GateViolation] = {}
    for item in collected:
        key = (
            item.code,
            item.message,
            item.severity,
            item.constraint_id or "",
            item.stage or "",
            item.checker_id or "",
        )
        deduped[key] = item
    return tuple(
        deduped[key]
        for key in sorted(
            deduped,
            key=lambda item: (item[2], item[0], item[3], item[4], item[5], item[1]),
        )
    )


def _resolve_required_stages(output: PipelineOutput) -> tuple[str, ...]:
    required_stages = set(output.required_stages)
    if output.adversarial_required:
        required_stages.add(ADVERSARIAL_STAGE_ID)
    return _sorted_unique_strings(required_stages)


def _collect_stage_required_constraints(
    output: PipelineOutput,
    check_results: Sequence[PipelineCheckResult],
) -> dict[str, tuple[str, ...]]:
    required: dict[str, set[str]] = defaultdict(set)

    for stage_requirement in output.stage_coverage_requirements:
        if stage_requirement.constraint_ids:
            required[stage_requirement.stage].update(stage_requirement.constraint_ids)

    for result in check_results:
        if result.required_constraint_ids:
            required[result.stage].update(result.required_constraint_ids)

    return {
        stage_id: _sorted_unique_strings(constraint_ids)
        for stage_id, constraint_ids in required.items()
    }


def _index_overrides(
    override_records: tuple[ConstraintOverrideRecord, ...],
) -> dict[str, ConstraintOverrideRecord]:
    indexed: dict[str, ConstraintOverrideRecord] = {}
    for record in override_records:
        indexed.setdefault(record.constraint_id, record)
    return indexed


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


def _coerce_constraint_envelope(
    value: ConstraintEnvelope | Mapping[str, object],
) -> ConstraintEnvelope:
    if isinstance(value, ConstraintEnvelope):
        return value
    if isinstance(value, Mapping):
        return ConstraintEnvelope.from_dict(value)
    raise TypeError(
        f"constraint_envelope: expected ConstraintEnvelope or mapping, got {type(value)}"
    )


def _coerce_pipeline_output(value: PipelineOutput | Mapping[str, object]) -> PipelineOutput:
    if isinstance(value, PipelineOutput):
        return value
    if isinstance(value, Mapping):
        return PipelineOutput.from_mapping(value)
    raise TypeError(f"pipeline_output: expected PipelineOutput or mapping, got {type(value)}")


def _coerce_override_records(
    value: Iterable[ConstraintOverrideRecord | Mapping[str, object]],
) -> tuple[tuple[ConstraintOverrideRecord, ...], tuple[str, ...]]:
    parsed: list[ConstraintOverrideRecord] = []
    invalid: list[str] = []

    for index, item in enumerate(value):
        try:
            if isinstance(item, ConstraintOverrideRecord):
                record = item
            elif isinstance(item, Mapping):
                record = ConstraintOverrideRecord.from_mapping(item)
            else:
                raise ValueError(f"override[{index}]: invalid override payload type {type(item)}")

            if not record.is_explicit_approval:
                invalid.append(
                    f"override[{index}]: missing explicit justification and approval metadata",
                )
                continue

            parsed.append(record)
        except (TypeError, ValueError) as exc:
            invalid.append(f"override[{index}]: {exc}")

    parsed_sorted = tuple(
        sorted(
            parsed,
            key=lambda item: (
                item.constraint_id,
                item.approved_by,
                item.justification,
                item.approval_reference or "",
            ),
        )
    )
    invalid_sorted = tuple(sorted(invalid))
    return parsed_sorted, invalid_sorted


def _coerce_stage_coverage_requirements(
    value: object,
    field_name: str,
) -> tuple[StageCoverageRequirement, ...]:
    if value is None:
        return ()

    parsed: list[StageCoverageRequirement] = []

    if isinstance(value, Mapping):
        for stage_id, constraint_values in sorted(value.items(), key=lambda item: str(item[0])):
            stage_name = _require_non_empty_str(stage_id, f"{field_name}.<key>")
            parsed.append(
                StageCoverageRequirement(
                    stage=stage_name,
                    constraint_ids=_coerce_string_tuple(
                        constraint_values,
                        f"{field_name}.{stage_name}",
                    ),
                )
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            if isinstance(item, StageCoverageRequirement):
                parsed.append(item)
            elif isinstance(item, Mapping):
                parsed.append(StageCoverageRequirement.from_mapping(item))
            else:
                raise ValueError(
                    f"{field_name}[{index}]: expected mapping or StageCoverageRequirement",
                )
    else:
        raise ValueError(f"{field_name}: expected mapping or array, got {type(value)}")

    parsed_sorted = sorted(parsed, key=lambda item: (item.stage, item.constraint_ids))
    return tuple(parsed_sorted)


def _coerce_check_results(value: object, field_name: str) -> tuple[PipelineCheckResult, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field_name}: expected array, got {type(value)}")

    parsed: list[PipelineCheckResult] = []
    for index, item in enumerate(value):
        if isinstance(item, PipelineCheckResult):
            parsed.append(item)
        elif isinstance(item, Mapping):
            parsed.append(PipelineCheckResult.from_mapping(item))
        else:
            raise ValueError(f"{field_name}[{index}]: expected mapping or PipelineCheckResult")

    return tuple(sorted(parsed, key=_check_result_sort_key))


def _coerce_mode(value: object, field_name: str) -> VerificationSelectionMode:
    if isinstance(value, VerificationSelectionMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == VerificationSelectionMode.INCREMENTAL.value:
            return VerificationSelectionMode.INCREMENTAL
        if normalized == VerificationSelectionMode.FULL.value:
            return VerificationSelectionMode.FULL
    raise ValueError(
        f"{field_name}: expected one of "
        f"{VerificationSelectionMode.INCREMENTAL.value!r}, "
        f"{VerificationSelectionMode.FULL.value!r}",
    )


def _coerce_evidence_result(value: object, field_name: str) -> EvidenceResult:
    if isinstance(value, EvidenceResult):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return EvidenceResult(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name}: invalid evidence result {value!r}") from exc
    raise ValueError(f"{field_name}: expected EvidenceResult or string")


def _coerce_optional_bool(value: object, field_name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name}: expected bool")


def _coerce_optional_str(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name}: expected string")
    normalized = value.strip()
    return normalized or None


def _coerce_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()

    if isinstance(value, str):
        return (_require_non_empty_str(value, field_name),)

    if not isinstance(value, (tuple, list, set, frozenset)):
        raise ValueError(f"{field_name}: expected string or array of strings")

    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_require_non_empty_str(item, f"{field_name}[{index}]"))

    return _sorted_unique_strings(parsed)


def _coerce_stage_sequence(
    value: Iterable[str] | Sequence[str], field_name: str
) -> tuple[str, ...]:
    parsed: list[str] = []
    for index, item in enumerate(value):
        parsed.append(_require_non_empty_str(item, f"{field_name}[{index}]"))
    return tuple(dict.fromkeys(parsed))


def _sorted_unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    normalized = {item.strip() for item in values if item.strip()}
    return tuple(sorted(normalized))


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}: expected non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name}: expected non-empty string")
    return normalized


__all__ = [
    "ADVERSARIAL_STAGE_ID",
    "ConstraintAssessment",
    "ConstraintDisposition",
    "ConstraintEvidenceLink",
    "ConstraintGateDecision",
    "ConstraintGateDiagnostics",
    "ConstraintOverrideRecord",
    "GateViolation",
    "GateVerdict",
    "PipelineCheckResult",
    "PipelineOutput",
    "StageAssessment",
    "StageCoverageGap",
    "StageCoverageRequirement",
    "VerificationSelectionMode",
    "evaluate_constraint_gate",
    "run_constraint_gate",
    "select_stage_plan",
    "select_verification_mode",
]
