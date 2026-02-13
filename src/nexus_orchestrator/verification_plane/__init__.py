"""
nexus-orchestrator — verification plane public API.

File: src/nexus_orchestrator/verification_plane/__init__.py
Last updated: 2026-02-12

Purpose
- Export stable entry points for verification gate evaluation and stage selection.

What should be included in this file
- Deterministic, side-effect-free interfaces used by control/integration planes.

Functional requirements
- Must expose binary constraint gate APIs and diagnostics payload helpers.

Non-functional requirements
- Keep import-time behavior deterministic and lightweight.
"""

from nexus_orchestrator.verification_plane.constraint_gate import (
    ADVERSARIAL_STAGE_ID,
    ConstraintAssessment,
    ConstraintDisposition,
    ConstraintEvidenceLink,
    ConstraintGateDecision,
    ConstraintGateDiagnostics,
    ConstraintOverrideRecord,
    GateVerdict,
    GateViolation,
    PipelineCheckResult,
    PipelineOutput,
    StageAssessment,
    StageCoverageGap,
    StageCoverageRequirement,
    VerificationSelectionMode,
    evaluate_constraint_gate,
    run_constraint_gate,
    select_stage_plan,
    select_verification_mode,
)

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
