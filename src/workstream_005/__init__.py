"""Workstream 005: equivalence novelty control and deployment integration readiness.

This module codifies four requirements from the design notes:
- if mapping to a baseline succeeds within tolerance, the candidate must not be labeled
  as a fundamentally new law and should be marked as a reformulation,
- adversarially equivalent syntactic variants (IBP/scaling/field transforms) must be
  merged by equivalence quotienting,
- deployment packages must include stable integrator settings with recommended
  timestepping guidance,
- implicit-solver integration must include Jacobian/tangent artifacts and adapter
  readiness evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

TimesteppingStrategy = Literal["fixed", "adaptive", "implicit_adaptive"]

_MISSING: object = object()


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    """Deterministic pass/fail result with stable reason codes."""

    accepted: bool
    reason_codes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", _sorted_unique(self.reason_codes))
        object.__setattr__(self, "notes", _sorted_unique(self.notes))
        if self.accepted and self.reason_codes:
            raise ValueError("accepted gate evaluations cannot include reason_codes")

    def merge(self, other: GateEvaluation) -> GateEvaluation:
        """Merge two evaluations, preserving deterministic ordering."""
        return GateEvaluation(
            accepted=self.accepted and other.accepted,
            reason_codes=self.reason_codes + other.reason_codes,
            notes=self.notes + other.notes,
        )


@dataclass(frozen=True, slots=True)
class TransformEquivalenceSpec:
    """Evidence for transform-equivalence novelty labeling obligations."""

    mapping_attempted: bool = False
    admissible_transforms_only: bool = True
    mapping_succeeds_within_tolerance: bool = False
    mapping_tolerance: float | None = None
    candidate_claimed_as_fundamentally_new: bool = False
    labeled_as_reformulation: bool = False

    def __post_init__(self) -> None:
        if self.mapping_tolerance is not None and not _is_nonnegative_finite(
            self.mapping_tolerance
        ):
            raise ValueError("mapping_tolerance must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class AdversarialEquivalenceSpec:
    """Coverage and quotienting checks for adversarial equivalence variants."""

    provides_ibp_variant: bool = False
    provides_scaling_variant: bool = False
    provides_field_transform_variant: bool = False
    equivalence_quotienting_enabled: bool = False
    equivalence_quotienting_merges_variants: bool = False


@dataclass(frozen=True, slots=True)
class IntegratorTimesteppingSpec:
    """Deployment readiness for stable integrator settings and recommended timestepping."""

    integrator_name: str | None = None
    stable_settings_verified: bool = False
    recommended_timestep: float | None = None
    maximum_stable_timestep: float | None = None
    timestepping_strategy: TimesteppingStrategy | None = None

    def __post_init__(self) -> None:
        normalized_name: str | None = None
        if self.integrator_name is not None:
            normalized_name = _normalize_tag(self.integrator_name, field_name="integrator_name")
        object.__setattr__(self, "integrator_name", normalized_name)

        if self.recommended_timestep is not None and not _is_positive_finite(
            self.recommended_timestep
        ):
            raise ValueError("recommended_timestep must be finite and > 0")

        if self.maximum_stable_timestep is not None and not _is_positive_finite(
            self.maximum_stable_timestep
        ):
            raise ValueError("maximum_stable_timestep must be finite and > 0")

        normalized_strategy: TimesteppingStrategy | None = None
        if self.timestepping_strategy is not None:
            normalized_strategy = _normalize_timestepping_strategy(
                self.timestepping_strategy,
                field_name="timestepping_strategy",
            )
        object.__setattr__(self, "timestepping_strategy", normalized_strategy)


@dataclass(frozen=True, slots=True)
class ImplicitSolverSupportSpec:
    """Evidence for Jacobian/tangent availability and implicit-solver integration readiness."""

    jacobians_or_tangents_provided: bool = False
    implicit_solver_integration_tested: bool = False
    solver_adapter_available: bool = False


def validate_transform_equivalence_obligation(
    spec: TransformEquivalenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate novelty labeling obligations under transform-equivalence mapping."""
    parsed = _coerce_transform_equivalence(spec)
    violations: list[str] = []
    notes: list[str] = []

    if not parsed.mapping_attempted:
        violations.append("equivalence.mapping_not_attempted")

    if not parsed.admissible_transforms_only:
        violations.append("equivalence.inadmissible_transform_used")

    if parsed.mapping_succeeds_within_tolerance:
        notes.append("equivalence.mapping_found_within_tolerance")
        if parsed.mapping_tolerance is None:
            violations.append("equivalence.missing_mapping_tolerance")
        if parsed.candidate_claimed_as_fundamentally_new:
            violations.append("equivalence.false_fundamental_novelty_claim")
        if not parsed.labeled_as_reformulation:
            violations.append("equivalence.missing_reformulation_label")
    else:
        notes.append("equivalence.no_within_tolerance_mapping_found")

    return _evaluation_from_violations(
        violations,
        pass_note="transform-equivalence obligation accepted",
        extra_notes=notes,
    )


def validate_adversarial_equivalence_test(
    spec: AdversarialEquivalenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate adversarial-equivalence variant coverage and quotienting behavior."""
    parsed = _coerce_adversarial_equivalence(spec)
    violations: list[str] = []
    notes: list[str] = []

    if not parsed.provides_ibp_variant:
        violations.append("adversarial_equivalence.missing_ibp_variant")
    if not parsed.provides_scaling_variant:
        violations.append("adversarial_equivalence.missing_scaling_variant")
    if not parsed.provides_field_transform_variant:
        violations.append("adversarial_equivalence.missing_field_transform_variant")

    if not parsed.equivalence_quotienting_enabled:
        violations.append("adversarial_equivalence.quotienting_disabled")
    elif not parsed.equivalence_quotienting_merges_variants:
        violations.append("adversarial_equivalence.quotienting_failed_to_merge_variants")
    else:
        notes.append("adversarial_equivalence.quotienting_merged_variants")

    return _evaluation_from_violations(
        violations,
        pass_note="adversarial equivalence test accepted",
        extra_notes=notes,
    )


def validate_integrator_timestepping_contract(
    spec: IntegratorTimesteppingSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate stable integrator settings and recommended timestepping artifacts."""
    parsed = _coerce_integrator_timestepping(spec)
    violations: list[str] = []
    notes: list[str] = []

    if parsed.integrator_name is None:
        violations.append("integrator.missing_integrator_name")

    if not parsed.stable_settings_verified:
        violations.append("integrator.settings_not_verified_stable")

    if parsed.recommended_timestep is None:
        violations.append("integrator.missing_recommended_timestep")

    if parsed.timestepping_strategy is None:
        violations.append("integrator.missing_timestepping_strategy")

    if (
        parsed.recommended_timestep is not None
        and parsed.maximum_stable_timestep is not None
        and parsed.recommended_timestep > parsed.maximum_stable_timestep
    ):
        violations.append("integrator.recommended_timestep_exceeds_stable_limit")

    if (
        parsed.recommended_timestep is not None
        and parsed.maximum_stable_timestep is not None
        and parsed.recommended_timestep <= parsed.maximum_stable_timestep
    ):
        notes.append("integrator.recommended_timestep_within_stable_limit")

    if parsed.timestepping_strategy == "adaptive":
        notes.append("integrator.adaptive_timestepping_recommended")
    if parsed.timestepping_strategy == "implicit_adaptive":
        notes.append("integrator.implicit_adaptive_timestepping_recommended")

    return _evaluation_from_violations(
        violations,
        pass_note="integrator timestepping contract accepted",
        extra_notes=notes,
    )


def validate_implicit_solver_support(
    spec: ImplicitSolverSupportSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate implicit-solver readiness artifacts for deployment."""
    parsed = _coerce_implicit_solver_support(spec)
    violations: list[str] = []

    if not parsed.jacobians_or_tangents_provided:
        violations.append("implicit_solver.missing_jacobians_or_tangents")
    if not parsed.implicit_solver_integration_tested:
        violations.append("implicit_solver.integration_not_tested")
    if not parsed.solver_adapter_available:
        violations.append("implicit_solver.missing_solver_adapter")

    return _evaluation_from_violations(
        violations,
        pass_note="implicit solver support accepted",
    )


def evaluate_workstream_005(
    *,
    transform_equivalence: TransformEquivalenceSpec | Mapping[str, object] | None = None,
    adversarial_equivalence: AdversarialEquivalenceSpec | Mapping[str, object] | None = None,
    integrator: IntegratorTimesteppingSpec | Mapping[str, object] | None = None,
    implicit_solver: ImplicitSolverSupportSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Evaluate configured Workstream 005 requirements."""
    if all(
        component is None
        for component in (
            transform_equivalence,
            adversarial_equivalence,
            integrator,
            implicit_solver,
        )
    ):
        raise ValueError(
            "at least one component (transform_equivalence, adversarial_equivalence, integrator, "
            "implicit_solver) is required"
        )

    decision = GateEvaluation(accepted=True)

    if transform_equivalence is not None:
        decision = decision.merge(validate_transform_equivalence_obligation(transform_equivalence))
    if adversarial_equivalence is not None:
        decision = decision.merge(validate_adversarial_equivalence_test(adversarial_equivalence))
    if integrator is not None:
        decision = decision.merge(validate_integrator_timestepping_contract(integrator))
    if implicit_solver is not None:
        decision = decision.merge(validate_implicit_solver_support(implicit_solver))

    return decision


def is_workstream_005_ready(
    *,
    transform_equivalence: TransformEquivalenceSpec | Mapping[str, object] | None = None,
    adversarial_equivalence: AdversarialEquivalenceSpec | Mapping[str, object] | None = None,
    integrator: IntegratorTimesteppingSpec | Mapping[str, object] | None = None,
    implicit_solver: ImplicitSolverSupportSpec | Mapping[str, object] | None = None,
) -> bool:
    """Boolean wrapper around :func:`evaluate_workstream_005`."""
    return evaluate_workstream_005(
        transform_equivalence=transform_equivalence,
        adversarial_equivalence=adversarial_equivalence,
        integrator=integrator,
        implicit_solver=implicit_solver,
    ).accepted


def validate_transform_mapping_obligation(
    spec: TransformEquivalenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_transform_equivalence_obligation`."""
    return validate_transform_equivalence_obligation(spec)


def validate_transform_equivalence(
    spec: TransformEquivalenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_transform_equivalence_obligation`."""
    return validate_transform_equivalence_obligation(spec)


def validate_adversarial_equivalence(
    spec: AdversarialEquivalenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_adversarial_equivalence_test`."""
    return validate_adversarial_equivalence_test(spec)


def validate_equivalence_quotienting(
    spec: AdversarialEquivalenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_adversarial_equivalence_test`."""
    return validate_adversarial_equivalence_test(spec)


def validate_stable_integrator_settings(
    spec: IntegratorTimesteppingSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_integrator_timestepping_contract`."""
    return validate_integrator_timestepping_contract(spec)


def validate_integrator_settings_and_timestepping(
    spec: IntegratorTimesteppingSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_integrator_timestepping_contract`."""
    return validate_integrator_timestepping_contract(spec)


def validate_timestepping_recommendation(
    spec: IntegratorTimesteppingSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_integrator_timestepping_contract`."""
    return validate_integrator_timestepping_contract(spec)


def validate_jacobians_and_tangents(
    spec: ImplicitSolverSupportSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_implicit_solver_support`."""
    return validate_implicit_solver_support(spec)


def validate_solver_integration_artifacts(
    spec: ImplicitSolverSupportSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_implicit_solver_support`."""
    return validate_implicit_solver_support(spec)


def check_workstream_005(
    *,
    transform_equivalence: TransformEquivalenceSpec | Mapping[str, object] | None = None,
    adversarial_equivalence: AdversarialEquivalenceSpec | Mapping[str, object] | None = None,
    integrator: IntegratorTimesteppingSpec | Mapping[str, object] | None = None,
    implicit_solver: ImplicitSolverSupportSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Alias for :func:`evaluate_workstream_005`."""
    return evaluate_workstream_005(
        transform_equivalence=transform_equivalence,
        adversarial_equivalence=adversarial_equivalence,
        integrator=integrator,
        implicit_solver=implicit_solver,
    )


def _coerce_transform_equivalence(
    spec: TransformEquivalenceSpec | Mapping[str, object],
) -> TransformEquivalenceSpec:
    if isinstance(spec, TransformEquivalenceSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a TransformEquivalenceSpec or mapping")

    return TransformEquivalenceSpec(
        mapping_attempted=_optional_bool(
            spec,
            (
                "mapping_attempted",
                "transform_mapping_attempted",
                "attempted_baseline_mapping",
            ),
            field_name="mapping_attempted",
            default=False,
        ),
        admissible_transforms_only=_optional_bool(
            spec,
            (
                "admissible_transforms_only",
                "uses_only_admissible_transforms",
                "transform_dsl_is_admissible",
            ),
            field_name="admissible_transforms_only",
            default=True,
        ),
        mapping_succeeds_within_tolerance=_optional_bool(
            spec,
            (
                "mapping_succeeds_within_tolerance",
                "mapping_success_within_tolerance",
                "equivalent_within_tolerance",
            ),
            field_name="mapping_succeeds_within_tolerance",
            default=False,
        ),
        mapping_tolerance=_optional_float(
            spec,
            (
                "mapping_tolerance",
                "equivalence_tolerance",
                "tolerance",
            ),
            field_name="mapping_tolerance",
            default=None,
        ),
        candidate_claimed_as_fundamentally_new=_optional_bool(
            spec,
            (
                "candidate_claimed_as_fundamentally_new",
                "claimed_fundamental_novelty",
                "claimed_new_law",
            ),
            field_name="candidate_claimed_as_fundamentally_new",
            default=False,
        ),
        labeled_as_reformulation=_optional_bool(
            spec,
            (
                "labeled_as_reformulation",
                "labeled_reformulation",
                "marked_as_reformulation",
            ),
            field_name="labeled_as_reformulation",
            default=False,
        ),
    )


def _coerce_adversarial_equivalence(
    spec: AdversarialEquivalenceSpec | Mapping[str, object],
) -> AdversarialEquivalenceSpec:
    if isinstance(spec, AdversarialEquivalenceSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an AdversarialEquivalenceSpec or mapping")

    return AdversarialEquivalenceSpec(
        provides_ibp_variant=_optional_bool(
            spec,
            (
                "provides_ibp_variant",
                "has_ibp_variant",
                "includes_integration_by_parts_variant",
            ),
            field_name="provides_ibp_variant",
            default=False,
        ),
        provides_scaling_variant=_optional_bool(
            spec,
            (
                "provides_scaling_variant",
                "has_scaling_variant",
                "includes_scaling_variant",
            ),
            field_name="provides_scaling_variant",
            default=False,
        ),
        provides_field_transform_variant=_optional_bool(
            spec,
            (
                "provides_field_transform_variant",
                "has_field_transform_variant",
                "includes_field_transform_variant",
            ),
            field_name="provides_field_transform_variant",
            default=False,
        ),
        equivalence_quotienting_enabled=_optional_bool(
            spec,
            (
                "equivalence_quotienting_enabled",
                "quotienting_enabled",
                "equivalence_merging_enabled",
            ),
            field_name="equivalence_quotienting_enabled",
            default=False,
        ),
        equivalence_quotienting_merges_variants=_optional_bool(
            spec,
            (
                "equivalence_quotienting_merges_variants",
                "quotienting_merges_variants",
                "equivalence_quotienting_passed",
            ),
            field_name="equivalence_quotienting_merges_variants",
            default=False,
        ),
    )


def _coerce_integrator_timestepping(
    spec: IntegratorTimesteppingSpec | Mapping[str, object],
) -> IntegratorTimesteppingSpec:
    if isinstance(spec, IntegratorTimesteppingSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an IntegratorTimesteppingSpec or mapping")

    timestepping_raw = _pick_first(
        spec,
        (
            "timestepping_strategy",
            "recommended_timestepping_strategy",
            "timestep_policy",
        ),
    )
    timestepping_strategy: TimesteppingStrategy | None = None
    if timestepping_raw is not _MISSING and timestepping_raw is not None:
        timestepping_strategy = _as_timestepping_strategy(
            timestepping_raw,
            field_name="timestepping_strategy",
        )

    return IntegratorTimesteppingSpec(
        integrator_name=_optional_str(
            spec,
            (
                "integrator_name",
                "recommended_integrator",
                "integrator",
            ),
            field_name="integrator_name",
            default=None,
        ),
        stable_settings_verified=_optional_bool(
            spec,
            (
                "stable_settings_verified",
                "integrator_settings_stable",
                "stable_integrator_settings",
            ),
            field_name="stable_settings_verified",
            default=False,
        ),
        recommended_timestep=_optional_float(
            spec,
            (
                "recommended_timestep",
                "recommended_dt",
                "recommended_timestep_size",
            ),
            field_name="recommended_timestep",
            default=None,
        ),
        maximum_stable_timestep=_optional_float(
            spec,
            (
                "maximum_stable_timestep",
                "max_stable_timestep",
                "stability_timestep_limit",
            ),
            field_name="maximum_stable_timestep",
            default=None,
        ),
        timestepping_strategy=timestepping_strategy,
    )


def _coerce_implicit_solver_support(
    spec: ImplicitSolverSupportSpec | Mapping[str, object],
) -> ImplicitSolverSupportSpec:
    if isinstance(spec, ImplicitSolverSupportSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an ImplicitSolverSupportSpec or mapping")

    return ImplicitSolverSupportSpec(
        jacobians_or_tangents_provided=_optional_bool(
            spec,
            (
                "jacobians_or_tangents_provided",
                "jacobians_available",
                "tangents_available",
            ),
            field_name="jacobians_or_tangents_provided",
            default=False,
        ),
        implicit_solver_integration_tested=_optional_bool(
            spec,
            (
                "implicit_solver_integration_tested",
                "solver_integration_tested",
                "implicit_solver_compatible",
            ),
            field_name="implicit_solver_integration_tested",
            default=False,
        ),
        solver_adapter_available=_optional_bool(
            spec,
            (
                "solver_adapter_available",
                "umat_or_closure_adapter_available",
                "solver_adapter_present",
            ),
            field_name="solver_adapter_available",
            default=False,
        ),
    )


def _evaluation_from_violations(
    violations: Iterable[str],
    *,
    pass_note: str,
    extra_notes: Iterable[str] = (),
) -> GateEvaluation:
    normalized_violations = _sorted_unique(violations)
    normalized_notes = _sorted_unique(extra_notes)
    if normalized_violations:
        return GateEvaluation(
            accepted=False,
            reason_codes=normalized_violations,
            notes=normalized_notes,
        )
    return GateEvaluation(accepted=True, notes=(pass_note,) + normalized_notes)


def _optional_bool(
    spec: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    field_name: str,
    default: bool,
) -> bool:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        return default
    return _as_bool(raw, field_name=field_name)


def _optional_float(
    spec: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    field_name: str,
    default: float | None,
) -> float | None:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        return default
    if raw is None:
        return None
    return _as_float(raw, field_name=field_name)


def _optional_str(
    spec: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    field_name: str,
    default: str | None,
) -> str | None:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        return default
    if raw is None:
        return None
    return _as_str(raw, field_name=field_name)


def _pick_first(spec: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in spec:
            return spec[key]
    return _MISSING


def _as_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a bool")


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite float")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalize_tag(value: str, *, field_name: str) -> str:
    return _as_str(value, field_name=field_name).lower()


def _as_timestepping_strategy(value: object, *, field_name: str) -> TimesteppingStrategy:
    text = _as_str(value, field_name=field_name)
    return _normalize_timestepping_strategy(text, field_name=field_name)


def _normalize_timestepping_strategy(value: str, *, field_name: str) -> TimesteppingStrategy:
    normalized = "".join(char for char in value.lower() if char.isalnum())
    if normalized in {"fixed", "constant"}:
        return "fixed"
    if normalized in {"adaptive", "adaptivetimestep", "adaptivestepping"}:
        return "adaptive"
    if normalized in {
        "implicitadaptive",
        "implicitadaptivetimestep",
        "semiimplicitadaptive",
        "stiffadaptive",
    }:
        return "implicit_adaptive"
    raise ValueError(f"{field_name} must be one of: fixed, adaptive, implicit_adaptive")


def _is_positive_finite(value: float) -> bool:
    return isfinite(value) and value > 0.0


def _is_nonnegative_finite(value: float) -> bool:
    return isfinite(value) and value >= 0.0


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    out = {value.strip() for value in values if value.strip()}
    return tuple(sorted(out))


__all__ = [
    "AdversarialEquivalenceSpec",
    "GateEvaluation",
    "ImplicitSolverSupportSpec",
    "IntegratorTimesteppingSpec",
    "TimesteppingStrategy",
    "TransformEquivalenceSpec",
    "check_workstream_005",
    "evaluate_workstream_005",
    "is_workstream_005_ready",
    "validate_adversarial_equivalence",
    "validate_adversarial_equivalence_test",
    "validate_equivalence_quotienting",
    "validate_implicit_solver_support",
    "validate_integrator_settings_and_timestepping",
    "validate_integrator_timestepping_contract",
    "validate_jacobians_and_tangents",
    "validate_solver_integration_artifacts",
    "validate_stable_integrator_settings",
    "validate_timestepping_recommendation",
    "validate_transform_equivalence",
    "validate_transform_equivalence_obligation",
    "validate_transform_mapping_obligation",
]
