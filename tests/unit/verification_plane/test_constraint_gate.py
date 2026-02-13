"""
nexus-orchestrator — test skeleton

File: tests/unit/verification_plane/test_constraint_gate.py
Last updated: 2026-02-11

Purpose
- Validate constraint gate binary accept/reject logic.

What this test file should cover
- Must fail on any must constraint violation.
- Should constraints require explicit override records.
- Evidence completeness requirements.

Functional requirements
- No real tool execution; use fake check results.

Non-functional requirements
- Deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nexus_orchestrator.domain import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    EvidenceResult,
)
from nexus_orchestrator.domain.ids import generate_work_item_id
from nexus_orchestrator.verification_plane.constraint_gate import (
    ConstraintOverrideRecord,
    GateVerdict,
    PipelineCheckResult,
    PipelineOutput,
    StageCoverageRequirement,
    VerificationSelectionMode,
    evaluate_constraint_gate,
)


def _fixed_work_item_id() -> str:
    return generate_work_item_id(timestamp_ms=1, randbytes=lambda size: b"\x00" * size)


def _constraint(
    *,
    constraint_id: str,
    severity: ConstraintSeverity,
    checker_binding: str,
) -> Constraint:
    return Constraint(
        id=constraint_id,
        severity=severity,
        category="correctness",
        description=f"constraint {constraint_id}",
        checker_binding=checker_binding,
        parameters={},
        requirement_links=(),
        source=ConstraintSource.MANUAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _envelope(*constraints: Constraint) -> ConstraintEnvelope:
    return ConstraintEnvelope(work_item_id=_fixed_work_item_id(), constraints=constraints)


def test_must_constraint_violation_rejects() -> None:
    must_constraint = _constraint(
        constraint_id="CON-COR-9001",
        severity=ConstraintSeverity.MUST,
        checker_binding="build_checker",
    )
    envelope = _envelope(must_constraint)

    pipeline_output = PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="build",
                checker_id="build_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=(must_constraint.id,),
            ),
        ),
        mode=VerificationSelectionMode.FULL,
        selected_stages=("build",),
        required_stages=("build",),
    )

    decision = evaluate_constraint_gate(envelope, pipeline_output)

    assert decision.verdict is GateVerdict.REJECT
    assert "must_constraint_unsatisfied" in decision.reason_codes
    assert decision.diagnostics.uncovered_must_constraints == (must_constraint.id,)


def test_should_constraint_requires_explicit_override() -> None:
    should_constraint = _constraint(
        constraint_id="CON-DOC-9001",
        severity=ConstraintSeverity.SHOULD,
        checker_binding="documentation_checker",
    )
    envelope = _envelope(should_constraint)

    output = PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="documentation",
                checker_id="documentation_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=(),
            ),
        ),
        selected_stages=("documentation",),
    )

    rejected = evaluate_constraint_gate(envelope, output)
    assert rejected.verdict is GateVerdict.REJECT
    assert "should_constraint_requires_override" in rejected.reason_codes

    overridden = evaluate_constraint_gate(
        envelope,
        output,
        override_records=(
            ConstraintOverrideRecord(
                constraint_id=should_constraint.id,
                justification="Docs update deferred to follow-up work item",
                approved_by="ops-reviewer",
                approved=True,
                approval_reference="CAB-123",
            ),
        ),
    )
    assert overridden.verdict is GateVerdict.ACCEPT
    assert overridden.diagnostics.overridden_should_constraints == (should_constraint.id,)


def test_missing_checker_mapping_rejects_no_silent_pass() -> None:
    must_constraint = _constraint(
        constraint_id="CON-STY-9001",
        severity=ConstraintSeverity.MUST,
        checker_binding="lint_checker",
    )
    envelope = _envelope(must_constraint)

    output = PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="build",
                checker_id="build_checker",
                result=EvidenceResult.PASS,
                covered_constraint_ids=(),
            ),
        ),
        selected_stages=("build",),
    )

    decision = evaluate_constraint_gate(envelope, output)

    assert decision.verdict is GateVerdict.REJECT
    assert must_constraint.id in decision.diagnostics.missing_checker_mappings
    assert "must_constraint_unsatisfied" in decision.reason_codes


def test_passed_stage_without_required_coverage_rejects() -> None:
    must_constraint = _constraint(
        constraint_id="CON-STY-9002",
        severity=ConstraintSeverity.MUST,
        checker_binding="lint_checker",
    )
    envelope = _envelope(must_constraint)

    output = PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="lint",
                checker_id="lint_checker",
                result=EvidenceResult.PASS,
                covered_constraint_ids=(),
                required_constraint_ids=(must_constraint.id,),
            ),
        ),
        selected_stages=("lint",),
        stage_coverage_requirements=(
            StageCoverageRequirement(stage="lint", constraint_ids=(must_constraint.id,)),
        ),
    )

    decision = evaluate_constraint_gate(envelope, output)

    assert decision.verdict is GateVerdict.REJECT
    assert "stage_coverage_incomplete" in decision.reason_codes
    assert decision.diagnostics.stage_coverage_gaps[0].stage == "lint"
    assert decision.diagnostics.stage_coverage_gaps[0].constraint_id == must_constraint.id


def test_diagnostics_payload_is_stably_sorted_and_informative() -> None:
    must_constraint = _constraint(
        constraint_id="CON-SEC-9001",
        severity=ConstraintSeverity.MUST,
        checker_binding="security_checker",
    )
    should_constraint = _constraint(
        constraint_id="CON-DOC-9002",
        severity=ConstraintSeverity.SHOULD,
        checker_binding="documentation_checker",
    )
    may_constraint = _constraint(
        constraint_id="CON-PER-9001",
        severity=ConstraintSeverity.MAY,
        checker_binding="performance_checker",
    )
    envelope = _envelope(should_constraint, may_constraint, must_constraint)

    output = PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="security",
                checker_id="security_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=(),
            ),
            PipelineCheckResult(
                stage="documentation",
                checker_id="documentation_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=(),
            ),
        ),
        mode=VerificationSelectionMode.INCREMENTAL,
        selected_stages=("documentation", "security"),
        required_stages=("security",),
    )

    decision = evaluate_constraint_gate(envelope, output)

    assert decision.verdict is GateVerdict.REJECT
    assert (
        tuple(sorted(decision.diagnostics.selected_stages)) == decision.diagnostics.selected_stages
    )
    assert tuple(
        sorted(decision.diagnostics.constraint_assessments, key=lambda item: item.constraint_id)
    ) == (decision.diagnostics.constraint_assessments)

    payload_a = decision.to_feedback_payload()
    payload_b = decision.to_feedback_payload()
    assert payload_a == payload_b

    failed_constraints = [item["constraint_id"] for item in payload_a["constraint_assessments"]]
    assert failed_constraints == sorted(failed_constraints)
    assert "must_constraint_unsatisfied" in payload_a["reason_codes"]
