"""Workstream 004: numerical contracts, weak-form robustness, and runtime fallbacks.

This module codifies five v3 requirements from the AI system notes:
- deployable theories must satisfy a solver-compatible numerical contract,
- weak-form residual engines must remain derivative-robust and FEM-native,
- experiments must support noisy measurements and sparse sensors,
- test-function coverage must include adaptive mode separation and reporting,
- runtime deployments must monitor extrapolation risk and provide safe fallbacks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

DifferentiabilityClass = Literal["c1", "c2"]
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
class NumericalContractSpec:
    """Solver compatibility declaration for deployable discovered theories."""

    differentiability_class: DifferentiabilityClass | None = None
    bounded_derivatives_in_envelope: bool = False
    stable_jacobians_or_tangents: bool = False
    stable_implicit_solves: bool = False
    fallback_behavior_defined: bool = False

    def __post_init__(self) -> None:
        normalized: DifferentiabilityClass | None = None
        if self.differentiability_class is not None:
            normalized = _normalize_differentiability_class(
                self.differentiability_class,
                field_name="differentiability_class",
            )
        object.__setattr__(self, "differentiability_class", normalized)


@dataclass(frozen=True, slots=True)
class WeakFormResidualSpec:
    """Weak-form residual engine capabilities for solver-integrated discovery."""

    computes_integral_residuals: bool = False
    uses_integration_by_parts: bool = False
    fem_or_irregular_mesh_compatible: bool = False
    constraints_evaluated_in_variational_form: bool = False
    supports_multiple_test_function_families: bool = False
    handles_boundary_terms_explicitly: bool = False
    supports_adjoint_or_ad_sensitivity: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentRobustnessSpec:
    """Capability declaration for noisy-measurement and sparse-sensor experiments."""

    supports_noisy_measurements: bool = False
    supports_sparse_sensors: bool = False


@dataclass(frozen=True, slots=True)
class TestFunctionCoverageSpec:
    """Weak-form mode-coverage and adaptive separation obligations."""

    coverage_library_available: bool = False
    covers_relevant_scales_and_modes: bool = False
    adaptive_enrichment_enabled: bool = False
    separates_indistinguishable_top_k: bool = False
    reports_mode_coverage: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeFallbackSpec:
    """Runtime monitoring and fallback strategy for deployment-time stability."""

    runtime_monitor_enabled: bool = False
    estimates_closure_error: bool = False
    estimates_extrapolation_risk: bool = False
    fallback_to_safer_baseline: bool = False
    fallback_to_local_mesh_or_dof_refinement: bool = False
    fallback_to_selective_high_fidelity_model: bool = False


def validate_numerical_contract(
    spec: NumericalContractSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate the solver-compatibility numerical contract."""
    parsed = _coerce_numerical_contract(spec)
    violations: list[str] = []
    notes: list[str] = []

    if parsed.differentiability_class is None:
        violations.append("numerical.missing_differentiability_class")
    elif parsed.differentiability_class == "c2":
        notes.append("numerical.c2_declared")
    else:
        notes.append("numerical.c1_declared")

    if not parsed.bounded_derivatives_in_envelope:
        violations.append("numerical.unbounded_derivatives")
    if not parsed.stable_jacobians_or_tangents:
        violations.append("numerical.missing_stable_jacobians_or_tangents")
    if not parsed.stable_implicit_solves:
        violations.append("numerical.unstable_implicit_solve_behavior")
    if not parsed.fallback_behavior_defined:
        violations.append("numerical.missing_extrapolation_fallback_behavior")

    return _evaluation_from_violations(
        violations,
        pass_note="numerical contract accepted",
        extra_notes=notes,
    )


def validate_weak_form_residual_engine(
    spec: WeakFormResidualSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate weak-form and solver-integrated residual capabilities."""
    parsed = _coerce_weak_form(spec)
    violations: list[str] = []

    if not parsed.computes_integral_residuals:
        violations.append("weak_form.missing_integral_residuals")
    if not parsed.uses_integration_by_parts:
        violations.append("weak_form.missing_integration_by_parts")
    if not parsed.fem_or_irregular_mesh_compatible:
        violations.append("weak_form.not_fem_or_irregular_mesh_compatible")
    if not parsed.constraints_evaluated_in_variational_form:
        violations.append("weak_form.constraints_not_evaluated_in_variational_form")
    if not parsed.supports_multiple_test_function_families:
        violations.append("weak_form.missing_multiple_test_function_families")
    if not parsed.handles_boundary_terms_explicitly:
        violations.append("weak_form.missing_boundary_term_handling")
    if not parsed.supports_adjoint_or_ad_sensitivity:
        violations.append("weak_form.missing_adjoint_or_ad_sensitivity_hooks")

    return _evaluation_from_violations(
        violations,
        pass_note="weak-form residual engine accepted",
    )


def validate_experiment_robustness(
    spec: ExperimentRobustnessSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate support for noisy measurements and sparse sensors."""
    parsed = _coerce_experiments(spec)
    violations: list[str] = []

    if not parsed.supports_noisy_measurements:
        violations.append("experiments.noisy_measurements_unsupported")
    if not parsed.supports_sparse_sensors:
        violations.append("experiments.sparse_sensors_unsupported")

    return _evaluation_from_violations(violations, pass_note="experiment robustness accepted")


def validate_test_function_coverage(
    spec: TestFunctionCoverageSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate weak-form test-function coverage and adaptive separation requirements."""
    parsed = _coerce_coverage(spec)
    violations: list[str] = []
    notes: list[str] = []

    if not parsed.coverage_library_available:
        violations.append("coverage.missing_test_function_library")
    if not parsed.covers_relevant_scales_and_modes:
        violations.append("coverage.missing_scale_mode_span")
    if not parsed.adaptive_enrichment_enabled:
        violations.append("coverage.adaptive_enrichment_disabled")
    if parsed.adaptive_enrichment_enabled and not parsed.separates_indistinguishable_top_k:
        violations.append("coverage.missing_top_k_mode_separation")
    if not parsed.reports_mode_coverage:
        violations.append("coverage.missing_mode_coverage_report")

    if parsed.adaptive_enrichment_enabled and parsed.separates_indistinguishable_top_k:
        notes.append("coverage.top_k_mode_separation_enabled")

    return _evaluation_from_violations(
        violations,
        pass_note="test-function coverage accepted",
        extra_notes=notes,
    )


def validate_runtime_error_fallback(
    spec: RuntimeFallbackSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate deployment-time monitor and fallback behavior."""
    parsed = _coerce_runtime(spec)
    violations: list[str] = []
    notes: list[str] = []

    if not parsed.runtime_monitor_enabled:
        violations.append("runtime.missing_error_monitor")
    if not parsed.estimates_closure_error:
        violations.append("runtime.missing_closure_error_estimation")
    if not parsed.estimates_extrapolation_risk:
        violations.append("runtime.missing_extrapolation_risk_estimation")

    fallback_count = sum(
        (
            parsed.fallback_to_safer_baseline,
            parsed.fallback_to_local_mesh_or_dof_refinement,
            parsed.fallback_to_selective_high_fidelity_model,
        )
    )
    if fallback_count < 1:
        violations.append("runtime.no_fallback_strategy")

    if parsed.fallback_to_safer_baseline:
        notes.append("runtime.fallback_safer_baseline_available")
    if parsed.fallback_to_local_mesh_or_dof_refinement:
        notes.append("runtime.fallback_local_refinement_available")
    if parsed.fallback_to_selective_high_fidelity_model:
        notes.append("runtime.fallback_selective_high_fidelity_available")

    return _evaluation_from_violations(
        violations,
        pass_note="runtime fallback accepted",
        extra_notes=notes,
    )


def evaluate_workstream_004(
    *,
    numerical_contract: NumericalContractSpec | Mapping[str, object] | None = None,
    weak_form: WeakFormResidualSpec | Mapping[str, object] | None = None,
    experiments: ExperimentRobustnessSpec | Mapping[str, object] | None = None,
    coverage: TestFunctionCoverageSpec | Mapping[str, object] | None = None,
    runtime: RuntimeFallbackSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Evaluate the configured Workstream 004 requirements."""
    if all(
        component is None
        for component in (numerical_contract, weak_form, experiments, coverage, runtime)
    ):
        raise ValueError(
            "at least one component (numerical_contract, weak_form, experiments, coverage, runtime) "
            "is required"
        )

    decision = GateEvaluation(accepted=True)

    if numerical_contract is not None:
        decision = decision.merge(validate_numerical_contract(numerical_contract))
    if weak_form is not None:
        decision = decision.merge(validate_weak_form_residual_engine(weak_form))
    if experiments is not None:
        decision = decision.merge(validate_experiment_robustness(experiments))
    if coverage is not None:
        decision = decision.merge(validate_test_function_coverage(coverage))
    if runtime is not None:
        decision = decision.merge(validate_runtime_error_fallback(runtime))

    return decision


def is_workstream_004_ready(
    *,
    numerical_contract: NumericalContractSpec | Mapping[str, object] | None = None,
    weak_form: WeakFormResidualSpec | Mapping[str, object] | None = None,
    experiments: ExperimentRobustnessSpec | Mapping[str, object] | None = None,
    coverage: TestFunctionCoverageSpec | Mapping[str, object] | None = None,
    runtime: RuntimeFallbackSpec | Mapping[str, object] | None = None,
) -> bool:
    """Boolean wrapper around :func:`evaluate_workstream_004`."""
    return evaluate_workstream_004(
        numerical_contract=numerical_contract,
        weak_form=weak_form,
        experiments=experiments,
        coverage=coverage,
        runtime=runtime,
    ).accepted


def validate_deployment_numerical_contract(
    spec: NumericalContractSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_numerical_contract`."""
    return validate_numerical_contract(spec)


def validate_weak_form_engine(spec: WeakFormResidualSpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_weak_form_residual_engine`."""
    return validate_weak_form_residual_engine(spec)


def validate_noisy_sparse_sensor_support(
    spec: ExperimentRobustnessSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_experiment_robustness`."""
    return validate_experiment_robustness(spec)


def validate_test_function_coverage_requirement(
    spec: TestFunctionCoverageSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_test_function_coverage`."""
    return validate_test_function_coverage(spec)


def validate_online_error_estimation_and_fallback(
    spec: RuntimeFallbackSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_runtime_error_fallback`."""
    return validate_runtime_error_fallback(spec)


def check_workstream_004(
    *,
    numerical_contract: NumericalContractSpec | Mapping[str, object] | None = None,
    weak_form: WeakFormResidualSpec | Mapping[str, object] | None = None,
    experiments: ExperimentRobustnessSpec | Mapping[str, object] | None = None,
    coverage: TestFunctionCoverageSpec | Mapping[str, object] | None = None,
    runtime: RuntimeFallbackSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Alias for :func:`evaluate_workstream_004`."""
    return evaluate_workstream_004(
        numerical_contract=numerical_contract,
        weak_form=weak_form,
        experiments=experiments,
        coverage=coverage,
        runtime=runtime,
    )


def _coerce_numerical_contract(
    spec: NumericalContractSpec | Mapping[str, object],
) -> NumericalContractSpec:
    if isinstance(spec, NumericalContractSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a NumericalContractSpec or mapping")

    differentiability_raw = _pick_first(
        spec,
        (
            "differentiability_class",
            "solver_differentiability_class",
            "required_solver_differentiability",
        ),
    )
    differentiability: DifferentiabilityClass | None = None
    if differentiability_raw is not _MISSING and differentiability_raw is not None:
        differentiability = _as_differentiability_class(
            differentiability_raw,
            field_name="differentiability_class",
        )

    return NumericalContractSpec(
        differentiability_class=differentiability,
        bounded_derivatives_in_envelope=_optional_bool(
            spec,
            (
                "bounded_derivatives_in_envelope",
                "derivatives_bounded_in_envelope",
                "bounded_derivatives",
            ),
            field_name="bounded_derivatives_in_envelope",
            default=False,
        ),
        stable_jacobians_or_tangents=_optional_bool(
            spec,
            (
                "stable_jacobians_or_tangents",
                "stable_jacobians",
                "stable_tangents",
            ),
            field_name="stable_jacobians_or_tangents",
            default=False,
        ),
        stable_implicit_solves=_optional_bool(
            spec,
            (
                "stable_implicit_solves",
                "stable_implicit_integration",
                "stable_nonlinear_solves",
            ),
            field_name="stable_implicit_solves",
            default=False,
        ),
        fallback_behavior_defined=_optional_bool(
            spec,
            (
                "fallback_behavior_defined",
                "extrapolation_fallback_defined",
                "monitor_triggered_fallback_defined",
            ),
            field_name="fallback_behavior_defined",
            default=False,
        ),
    )


def _coerce_weak_form(spec: WeakFormResidualSpec | Mapping[str, object]) -> WeakFormResidualSpec:
    if isinstance(spec, WeakFormResidualSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a WeakFormResidualSpec or mapping")

    return WeakFormResidualSpec(
        computes_integral_residuals=_optional_bool(
            spec,
            ("computes_integral_residuals", "integral_residuals", "weak_form_integral_residuals"),
            field_name="computes_integral_residuals",
            default=False,
        ),
        uses_integration_by_parts=_optional_bool(
            spec,
            ("uses_integration_by_parts", "integration_by_parts", "reduces_derivative_order"),
            field_name="uses_integration_by_parts",
            default=False,
        ),
        fem_or_irregular_mesh_compatible=_optional_bool(
            spec,
            (
                "fem_or_irregular_mesh_compatible",
                "fem_compatible",
                "supports_irregular_meshes",
            ),
            field_name="fem_or_irregular_mesh_compatible",
            default=False,
        ),
        constraints_evaluated_in_variational_form=_optional_bool(
            spec,
            (
                "constraints_evaluated_in_variational_form",
                "constraints_evaluated_in_conservative_or_variational_form",
                "variational_constraint_evaluation",
            ),
            field_name="constraints_evaluated_in_variational_form",
            default=False,
        ),
        supports_multiple_test_function_families=_optional_bool(
            spec,
            (
                "supports_multiple_test_function_families",
                "multiple_test_function_families",
                "supports_multiscale_test_functions",
            ),
            field_name="supports_multiple_test_function_families",
            default=False,
        ),
        handles_boundary_terms_explicitly=_optional_bool(
            spec,
            (
                "handles_boundary_terms_explicitly",
                "explicit_boundary_term_handling",
                "boundary_terms_explicit",
            ),
            field_name="handles_boundary_terms_explicitly",
            default=False,
        ),
        supports_adjoint_or_ad_sensitivity=_optional_bool(
            spec,
            (
                "supports_adjoint_or_ad_sensitivity",
                "adjoint_hooks_available",
                "ad_hooks_available",
            ),
            field_name="supports_adjoint_or_ad_sensitivity",
            default=False,
        ),
    )


def _coerce_experiments(
    spec: ExperimentRobustnessSpec | Mapping[str, object],
) -> ExperimentRobustnessSpec:
    if isinstance(spec, ExperimentRobustnessSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an ExperimentRobustnessSpec or mapping")

    return ExperimentRobustnessSpec(
        supports_noisy_measurements=_optional_bool(
            spec,
            (
                "supports_noisy_measurements",
                "noisy_measurements_supported",
                "robust_to_noisy_measurements",
            ),
            field_name="supports_noisy_measurements",
            default=False,
        ),
        supports_sparse_sensors=_optional_bool(
            spec,
            (
                "supports_sparse_sensors",
                "sparse_sensors_supported",
                "robust_to_sparse_sensor_layouts",
            ),
            field_name="supports_sparse_sensors",
            default=False,
        ),
    )


def _coerce_coverage(
    spec: TestFunctionCoverageSpec | Mapping[str, object],
) -> TestFunctionCoverageSpec:
    if isinstance(spec, TestFunctionCoverageSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a TestFunctionCoverageSpec or mapping")

    return TestFunctionCoverageSpec(
        coverage_library_available=_optional_bool(
            spec,
            (
                "coverage_library_available",
                "test_function_library_available",
                "has_coverage_library",
            ),
            field_name="coverage_library_available",
            default=False,
        ),
        covers_relevant_scales_and_modes=_optional_bool(
            spec,
            (
                "covers_relevant_scales_and_modes",
                "spans_relevant_scales_and_modes",
                "coverage_spans_modes",
            ),
            field_name="covers_relevant_scales_and_modes",
            default=False,
        ),
        adaptive_enrichment_enabled=_optional_bool(
            spec,
            (
                "adaptive_enrichment_enabled",
                "supports_adaptive_enrichment",
                "adaptive_test_function_enrichment",
            ),
            field_name="adaptive_enrichment_enabled",
            default=False,
        ),
        separates_indistinguishable_top_k=_optional_bool(
            spec,
            (
                "separates_indistinguishable_top_k",
                "top_k_mode_separation",
                "maximally_separates_top_k",
            ),
            field_name="separates_indistinguishable_top_k",
            default=False,
        ),
        reports_mode_coverage=_optional_bool(
            spec,
            (
                "reports_mode_coverage",
                "mode_coverage_reported",
                "explicit_mode_coverage_reporting",
            ),
            field_name="reports_mode_coverage",
            default=False,
        ),
    )


def _coerce_runtime(spec: RuntimeFallbackSpec | Mapping[str, object]) -> RuntimeFallbackSpec:
    if isinstance(spec, RuntimeFallbackSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a RuntimeFallbackSpec or mapping")

    return RuntimeFallbackSpec(
        runtime_monitor_enabled=_optional_bool(
            spec,
            (
                "runtime_monitor_enabled",
                "online_error_monitor_enabled",
                "runtime_risk_monitor_enabled",
            ),
            field_name="runtime_monitor_enabled",
            default=False,
        ),
        estimates_closure_error=_optional_bool(
            spec,
            (
                "estimates_closure_error",
                "closure_error_estimation",
                "estimates_closure_induced_error",
            ),
            field_name="estimates_closure_error",
            default=False,
        ),
        estimates_extrapolation_risk=_optional_bool(
            spec,
            (
                "estimates_extrapolation_risk",
                "extrapolation_risk_estimation",
                "monitors_extrapolation_risk",
            ),
            field_name="estimates_extrapolation_risk",
            default=False,
        ),
        fallback_to_safer_baseline=_optional_bool(
            spec,
            (
                "fallback_to_safer_baseline",
                "fallback_safer_baseline_closure",
                "fallback_baseline_closure",
            ),
            field_name="fallback_to_safer_baseline",
            default=False,
        ),
        fallback_to_local_mesh_or_dof_refinement=_optional_bool(
            spec,
            (
                "fallback_to_local_mesh_or_dof_refinement",
                "fallback_local_refinement",
                "fallback_mesh_refinement",
            ),
            field_name="fallback_to_local_mesh_or_dof_refinement",
            default=False,
        ),
        fallback_to_selective_high_fidelity_model=_optional_bool(
            spec,
            (
                "fallback_to_selective_high_fidelity_model",
                "fallback_selective_high_fidelity",
                "fallback_high_fidelity_model",
            ),
            field_name="fallback_to_selective_high_fidelity_model",
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


def _pick_first(spec: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in spec:
            return spec[key]
    return _MISSING


def _as_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{field_name} must be a bool")


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _as_differentiability_class(value: object, *, field_name: str) -> DifferentiabilityClass:
    text = _as_str(value, field_name=field_name)
    return _normalize_differentiability_class(text, field_name=field_name)


def _normalize_differentiability_class(
    value: str,
    *,
    field_name: str,
) -> DifferentiabilityClass:
    normalized = "".join(char for char in value.lower() if char.isalnum())
    if normalized in {"c1", "c2"}:
        return cast("DifferentiabilityClass", normalized)
    raise ValueError(f"{field_name} must be one of: c1, c2")


def _as_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite float")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    out = {value.strip() for value in values if value.strip()}
    return tuple(sorted(out))


__all__ = [
    "DifferentiabilityClass",
    "ExperimentRobustnessSpec",
    "GateEvaluation",
    "NumericalContractSpec",
    "RuntimeFallbackSpec",
    "TestFunctionCoverageSpec",
    "WeakFormResidualSpec",
    "check_workstream_004",
    "evaluate_workstream_004",
    "is_workstream_004_ready",
    "validate_deployment_numerical_contract",
    "validate_experiment_robustness",
    "validate_noisy_sparse_sensor_support",
    "validate_numerical_contract",
    "validate_online_error_estimation_and_fallback",
    "validate_runtime_error_fallback",
    "validate_test_function_coverage",
    "validate_test_function_coverage_requirement",
    "validate_weak_form_engine",
    "validate_weak_form_residual_engine",
]
