"""Workstream 002: symbolic-structure intent and representation capability gates.

This module codifies five design requirements:
- discovery targets must prioritize new symbolic structure over coefficient-only fitting,
- runtime assumptions should match the recommended hardware envelope,
- representations must support explicit discovery of new state variables,
- representations must support explicit discovery of new operator classes,
- representations should support principle-first discovery (energy/action/dissipation).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

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
class SymbolicDiscoveryTargetSpec:
    """Capability declaration for the structure-first discovery objective."""

    supports_new_closures: bool
    supports_new_constitutive_theories: bool
    supports_new_reduced_governing_pdes: bool
    supports_new_state_variables: bool
    supports_new_fields_and_potentials: bool
    supports_new_generative_principles: bool
    supports_new_operator_classes: bool
    novelty_is_defensible: bool
    coefficient_fit_only: bool = False


@dataclass(frozen=True, slots=True)
class HardwareEnvelopeSpec:
    """Hardware profile for the recommended 24GB-VRAM operating envelope."""

    gpu_vram_gb: float
    ram_gb: float
    fast_ssd: bool
    gpu_count: int = 1
    cpu_cluster_available: bool = False

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be >= 0")
        if not _is_nonnegative_finite(self.gpu_vram_gb):
            raise ValueError("gpu_vram_gb must be finite and >= 0")
        if not _is_nonnegative_finite(self.ram_gb):
            raise ValueError("ram_gb must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class StateVariableDiscoverySpec:
    """Explicit support for discovering closure coordinates/internal variables."""

    supports_new_state_variables: bool
    supports_internal_variable_evolution: bool
    evolution_includes_state: bool = True
    evolution_includes_invariants: bool = True
    evolution_includes_gradients: bool = True
    enforces_cncc: bool = True


@dataclass(frozen=True, slots=True)
class OperatorDiscoverySpec:
    """Explicit support for discovering non-standard operators."""

    supports_new_operators: bool
    supports_nonlocal_kernels: bool
    supports_fractional_operators: bool
    supports_memory_operators: bool
    supports_multiscale_homogenization: bool
    supports_operator_macros: bool
    supports_macro_symbolic_compression: bool


@dataclass(frozen=True, slots=True)
class PrincipleDiscoverySpec:
    """Support for discovering equations from generative principles."""

    supports_free_energy: bool
    supports_dissipation_potential: bool
    supports_action_functional: bool
    derives_governing_equations: bool
    supports_structural_conservation: bool
    supports_stability_admissibility: bool


def validate_symbolic_discovery_target(
    spec: SymbolicDiscoveryTargetSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate the structure-first discovery target."""
    parsed = _coerce_symbolic_target(spec)
    violations: list[str] = []

    if parsed.coefficient_fit_only:
        violations.append("target.coefficient_template_only")
    if not parsed.supports_new_closures:
        violations.append("target.missing_new_closures")
    if not parsed.supports_new_constitutive_theories:
        violations.append("target.missing_new_constitutive_theories")
    if not parsed.supports_new_reduced_governing_pdes:
        violations.append("target.missing_new_reduced_pdes")
    if not parsed.supports_new_state_variables:
        violations.append("target.missing_new_state_variables")
    if not parsed.supports_new_fields_and_potentials:
        violations.append("target.missing_new_fields_or_potentials")
    if not parsed.supports_new_generative_principles:
        violations.append("target.missing_new_generative_principles")
    if not parsed.supports_new_operator_classes:
        violations.append("target.missing_new_operator_classes")
    if not parsed.novelty_is_defensible:
        violations.append("target.novelty_not_defensible")

    return _evaluation_from_violations(
        violations,
        pass_note="symbolic discovery target accepted",
    )


def validate_hardware_envelope(
    spec: HardwareEnvelopeSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate whether hardware matches the recommended envelope."""
    parsed = _coerce_hardware(spec)
    violations: list[str] = []
    notes: list[str] = []

    if parsed.gpu_count < 1:
        violations.append("hardware.missing_gpu")
    if parsed.gpu_vram_gb < 24.0:
        violations.append("hardware.insufficient_vram")
    if parsed.ram_gb < 64.0:
        violations.append("hardware.insufficient_ram")
    if not parsed.fast_ssd:
        violations.append("hardware.missing_fast_ssd")

    if parsed.gpu_count > 1:
        notes.append("hardware.multi_gpu_available")
    if parsed.ram_gb > 256.0:
        notes.append("hardware.ram_above_recommended_max")
    if parsed.cpu_cluster_available:
        notes.append("hardware.cpu_cluster_available")

    return _evaluation_from_violations(
        violations,
        pass_note="hardware envelope accepted",
        extra_notes=notes,
    )


def validate_state_variable_discovery(
    spec: StateVariableDiscoverySpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate explicit support for discovering new state variables."""
    parsed = _coerce_state_variables(spec)
    violations: list[str] = []

    if not parsed.supports_new_state_variables:
        violations.append("state.discovery_disabled")
    if not parsed.supports_internal_variable_evolution:
        violations.append("state.missing_internal_variable_evolution")
    if not parsed.evolution_includes_state:
        violations.append("state.evolution_missing_state_term")
    if not parsed.evolution_includes_invariants:
        violations.append("state.evolution_missing_invariants")
    if not parsed.evolution_includes_gradients:
        violations.append("state.evolution_missing_gradient_terms")
    if not parsed.enforces_cncc:
        violations.append("state.cncc_not_enforced")

    return _evaluation_from_violations(
        violations,
        pass_note="state-variable discovery support accepted",
    )


def validate_operator_discovery(
    spec: OperatorDiscoverySpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate explicit support for discovering new operator classes."""
    parsed = _coerce_operators(spec)
    violations: list[str] = []

    if not parsed.supports_new_operators:
        violations.append("operator.discovery_disabled")
    if not parsed.supports_nonlocal_kernels:
        violations.append("operator.missing_nonlocal_kernel_support")
    if not (parsed.supports_fractional_operators or parsed.supports_memory_operators):
        violations.append("operator.missing_fractional_or_memory_support")
    if not parsed.supports_multiscale_homogenization:
        violations.append("operator.missing_multiscale_homogenization_support")
    if not parsed.supports_operator_macros:
        violations.append("operator.missing_macro_support")
    if parsed.supports_operator_macros and not parsed.supports_macro_symbolic_compression:
        violations.append("operator.macro_missing_symbolic_compression")

    return _evaluation_from_violations(
        violations,
        pass_note="operator discovery support accepted",
    )


def validate_principle_discovery(
    spec: PrincipleDiscoverySpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate support for principle-first equation discovery."""
    parsed = _coerce_principles(spec)
    violations: list[str] = []

    if not any(
        (
            parsed.supports_free_energy,
            parsed.supports_dissipation_potential,
            parsed.supports_action_functional,
        )
    ):
        violations.append("principle.no_principle_hypothesis_class")
    if not parsed.derives_governing_equations:
        violations.append("principle.missing_equation_derivation")
    if not parsed.supports_structural_conservation:
        violations.append("principle.missing_conservation_structure")
    if not parsed.supports_stability_admissibility:
        violations.append("principle.missing_stability_admissibility_path")

    return _evaluation_from_violations(
        violations,
        pass_note="principle discovery support accepted",
    )


def evaluate_workstream_002(
    *,
    target: SymbolicDiscoveryTargetSpec | Mapping[str, object] | None = None,
    hardware: HardwareEnvelopeSpec | Mapping[str, object] | None = None,
    state_variables: StateVariableDiscoverySpec | Mapping[str, object] | None = None,
    operators: OperatorDiscoverySpec | Mapping[str, object] | None = None,
    principles: PrincipleDiscoverySpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Evaluate the configured Workstream 002 capabilities."""
    if all(component is None for component in (target, hardware, state_variables, operators, principles)):
        raise ValueError(
            "at least one component (target, hardware, state_variables, operators, principles) "
            "is required"
        )

    decision = GateEvaluation(accepted=True)

    if target is not None:
        decision = decision.merge(validate_symbolic_discovery_target(target))
    if hardware is not None:
        decision = decision.merge(validate_hardware_envelope(hardware))
    if state_variables is not None:
        decision = decision.merge(validate_state_variable_discovery(state_variables))
    if operators is not None:
        decision = decision.merge(validate_operator_discovery(operators))
    if principles is not None:
        decision = decision.merge(validate_principle_discovery(principles))

    return decision


def is_workstream_002_ready(
    *,
    target: SymbolicDiscoveryTargetSpec | Mapping[str, object] | None = None,
    hardware: HardwareEnvelopeSpec | Mapping[str, object] | None = None,
    state_variables: StateVariableDiscoverySpec | Mapping[str, object] | None = None,
    operators: OperatorDiscoverySpec | Mapping[str, object] | None = None,
    principles: PrincipleDiscoverySpec | Mapping[str, object] | None = None,
) -> bool:
    """Boolean wrapper around :func:`evaluate_workstream_002`."""
    return evaluate_workstream_002(
        target=target,
        hardware=hardware,
        state_variables=state_variables,
        operators=operators,
        principles=principles,
    ).accepted


def recommended_hardware_envelope() -> HardwareEnvelopeSpec:
    """Return the baseline recommended hardware envelope."""
    return HardwareEnvelopeSpec(
        gpu_count=1,
        gpu_vram_gb=24.0,
        ram_gb=64.0,
        fast_ssd=True,
        cpu_cluster_available=False,
    )


def is_within_recommended_hardware_envelope(spec: HardwareEnvelopeSpec | Mapping[str, object]) -> bool:
    """Boolean wrapper around :func:`validate_hardware_envelope`."""
    return validate_hardware_envelope(spec).accepted


def validate_symbolic_structure_target(
    spec: SymbolicDiscoveryTargetSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_symbolic_discovery_target`."""
    return validate_symbolic_discovery_target(spec)


def validate_state_variable_support(
    spec: StateVariableDiscoverySpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_state_variable_discovery`."""
    return validate_state_variable_discovery(spec)


def validate_operator_support(spec: OperatorDiscoverySpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_operator_discovery`."""
    return validate_operator_discovery(spec)


def validate_principle_support(
    spec: PrincipleDiscoverySpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_principle_discovery`."""
    return validate_principle_discovery(spec)


def check_workstream_002(
    *,
    target: SymbolicDiscoveryTargetSpec | Mapping[str, object] | None = None,
    hardware: HardwareEnvelopeSpec | Mapping[str, object] | None = None,
    state_variables: StateVariableDiscoverySpec | Mapping[str, object] | None = None,
    operators: OperatorDiscoverySpec | Mapping[str, object] | None = None,
    principles: PrincipleDiscoverySpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Alias for :func:`evaluate_workstream_002`."""
    return evaluate_workstream_002(
        target=target,
        hardware=hardware,
        state_variables=state_variables,
        operators=operators,
        principles=principles,
    )


def _coerce_symbolic_target(
    spec: SymbolicDiscoveryTargetSpec | Mapping[str, object],
) -> SymbolicDiscoveryTargetSpec:
    if isinstance(spec, SymbolicDiscoveryTargetSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a SymbolicDiscoveryTargetSpec or mapping")

    return SymbolicDiscoveryTargetSpec(
        supports_new_closures=_required_bool(
            spec,
            ("supports_new_closures", "new_closures"),
            field_name="supports_new_closures",
        ),
        supports_new_constitutive_theories=_required_bool(
            spec,
            ("supports_new_constitutive_theories", "new_constitutive_theories"),
            field_name="supports_new_constitutive_theories",
        ),
        supports_new_reduced_governing_pdes=_required_bool(
            spec,
            ("supports_new_reduced_governing_pdes", "new_reduced_governing_pdes"),
            field_name="supports_new_reduced_governing_pdes",
        ),
        supports_new_state_variables=_required_bool(
            spec,
            ("supports_new_state_variables", "new_state_variables"),
            field_name="supports_new_state_variables",
        ),
        supports_new_fields_and_potentials=_required_bool(
            spec,
            ("supports_new_fields_and_potentials", "new_fields_and_potentials"),
            field_name="supports_new_fields_and_potentials",
        ),
        supports_new_generative_principles=_required_bool(
            spec,
            ("supports_new_generative_principles", "new_generative_principles"),
            field_name="supports_new_generative_principles",
        ),
        supports_new_operator_classes=_required_bool(
            spec,
            ("supports_new_operator_classes", "new_operator_classes"),
            field_name="supports_new_operator_classes",
        ),
        novelty_is_defensible=_required_bool(
            spec,
            ("novelty_is_defensible", "defensible_novelty"),
            field_name="novelty_is_defensible",
        ),
        coefficient_fit_only=_optional_bool(
            spec,
            ("coefficient_fit_only", "coefficient_template_fit_only", "template_fit_only"),
            field_name="coefficient_fit_only",
            default=False,
        ),
    )


def _coerce_hardware(spec: HardwareEnvelopeSpec | Mapping[str, object]) -> HardwareEnvelopeSpec:
    if isinstance(spec, HardwareEnvelopeSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a HardwareEnvelopeSpec or mapping")

    return HardwareEnvelopeSpec(
        gpu_count=_optional_int(spec, ("gpu_count", "num_gpus"), field_name="gpu_count", default=1),
        gpu_vram_gb=_required_float(
            spec,
            ("gpu_vram_gb", "vram_gb", "gpu_memory_gb"),
            field_name="gpu_vram_gb",
        ),
        ram_gb=_required_float(spec, ("ram_gb", "memory_gb", "system_ram_gb"), field_name="ram_gb"),
        fast_ssd=_optional_bool(spec, ("fast_ssd", "has_fast_ssd"), field_name="fast_ssd", default=False),
        cpu_cluster_available=_optional_bool(
            spec,
            ("cpu_cluster_available", "has_cpu_cluster"),
            field_name="cpu_cluster_available",
            default=False,
        ),
    )


def _coerce_state_variables(
    spec: StateVariableDiscoverySpec | Mapping[str, object],
) -> StateVariableDiscoverySpec:
    if isinstance(spec, StateVariableDiscoverySpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a StateVariableDiscoverySpec or mapping")

    return StateVariableDiscoverySpec(
        supports_new_state_variables=_required_bool(
            spec,
            ("supports_new_state_variables", "allows_new_state_variables"),
            field_name="supports_new_state_variables",
        ),
        supports_internal_variable_evolution=_required_bool(
            spec,
            ("supports_internal_variable_evolution", "supports_internal_variable_dynamics"),
            field_name="supports_internal_variable_evolution",
        ),
        evolution_includes_state=_optional_bool(
            spec,
            ("evolution_includes_state", "evolution_depends_on_state"),
            field_name="evolution_includes_state",
            default=True,
        ),
        evolution_includes_invariants=_optional_bool(
            spec,
            ("evolution_includes_invariants", "evolution_depends_on_invariants"),
            field_name="evolution_includes_invariants",
            default=True,
        ),
        evolution_includes_gradients=_optional_bool(
            spec,
            ("evolution_includes_gradients", "evolution_depends_on_gradients"),
            field_name="evolution_includes_gradients",
            default=True,
        ),
        enforces_cncc=_optional_bool(
            spec,
            ("enforces_cncc", "cncc_enforced"),
            field_name="enforces_cncc",
            default=True,
        ),
    )


def _coerce_operators(spec: OperatorDiscoverySpec | Mapping[str, object]) -> OperatorDiscoverySpec:
    if isinstance(spec, OperatorDiscoverySpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an OperatorDiscoverySpec or mapping")

    return OperatorDiscoverySpec(
        supports_new_operators=_required_bool(
            spec,
            ("supports_new_operators", "allows_non_enumerated_operators"),
            field_name="supports_new_operators",
        ),
        supports_nonlocal_kernels=_required_bool(
            spec,
            ("supports_nonlocal_kernels", "supports_nonlocal_operator_kernels"),
            field_name="supports_nonlocal_kernels",
        ),
        supports_fractional_operators=_optional_bool(
            spec,
            ("supports_fractional_operators", "supports_fractional"),
            field_name="supports_fractional_operators",
            default=False,
        ),
        supports_memory_operators=_optional_bool(
            spec,
            ("supports_memory_operators", "supports_memory_terms"),
            field_name="supports_memory_operators",
            default=False,
        ),
        supports_multiscale_homogenization=_required_bool(
            spec,
            ("supports_multiscale_homogenization", "supports_multiscale_operators"),
            field_name="supports_multiscale_homogenization",
        ),
        supports_operator_macros=_required_bool(
            spec,
            ("supports_operator_macros", "supports_macros"),
            field_name="supports_operator_macros",
        ),
        supports_macro_symbolic_compression=_optional_bool(
            spec,
            ("supports_macro_symbolic_compression", "supports_symbolic_macro_compression"),
            field_name="supports_macro_symbolic_compression",
            default=False,
        ),
    )


def _coerce_principles(spec: PrincipleDiscoverySpec | Mapping[str, object]) -> PrincipleDiscoverySpec:
    if isinstance(spec, PrincipleDiscoverySpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a PrincipleDiscoverySpec or mapping")

    return PrincipleDiscoverySpec(
        supports_free_energy=_optional_bool(
            spec,
            ("supports_free_energy", "supports_free_energy_principle"),
            field_name="supports_free_energy",
            default=False,
        ),
        supports_dissipation_potential=_optional_bool(
            spec,
            ("supports_dissipation_potential", "supports_dissipation_potentials"),
            field_name="supports_dissipation_potential",
            default=False,
        ),
        supports_action_functional=_optional_bool(
            spec,
            ("supports_action_functional", "supports_action_functionals"),
            field_name="supports_action_functional",
            default=False,
        ),
        derives_governing_equations=_required_bool(
            spec,
            ("derives_governing_equations", "derives_equations_from_principles"),
            field_name="derives_governing_equations",
        ),
        supports_structural_conservation=_required_bool(
            spec,
            ("supports_structural_conservation", "supports_conservation_structure"),
            field_name="supports_structural_conservation",
        ),
        supports_stability_admissibility=_required_bool(
            spec,
            ("supports_stability_admissibility", "supports_admissibility_certification"),
            field_name="supports_stability_admissibility",
        ),
    )


def _evaluation_from_violations(
    violations: list[str],
    *,
    pass_note: str,
    extra_notes: Iterable[str] = (),
) -> GateEvaluation:
    normalized_violations = _sorted_unique(violations)
    normalized_notes = _sorted_unique(extra_notes)
    if normalized_violations:
        return GateEvaluation(accepted=False, reason_codes=normalized_violations, notes=normalized_notes)
    return GateEvaluation(accepted=True, notes=(pass_note,) + normalized_notes)


def _required_bool(spec: Mapping[str, object], keys: tuple[str, ...], *, field_name: str) -> bool:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        raise TypeError(f"{field_name} must be a bool")
    return _as_bool(raw, field_name=field_name)


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


def _required_float(spec: Mapping[str, object], keys: tuple[str, ...], *, field_name: str) -> float:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        raise TypeError(f"{field_name} must be a finite float")
    return _as_float(raw, field_name=field_name)


def _optional_int(
    spec: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    field_name: str,
    default: int,
) -> int:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        return default
    return _as_int(raw, field_name=field_name)


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


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    return value


def _is_nonnegative_finite(value: float) -> bool:
    return isfinite(value) and value >= 0.0


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    out = {value.strip() for value in values if value.strip()}
    return tuple(sorted(out))


__all__ = [
    "GateEvaluation",
    "HardwareEnvelopeSpec",
    "OperatorDiscoverySpec",
    "PrincipleDiscoverySpec",
    "StateVariableDiscoverySpec",
    "SymbolicDiscoveryTargetSpec",
    "check_workstream_002",
    "evaluate_workstream_002",
    "is_within_recommended_hardware_envelope",
    "is_workstream_002_ready",
    "recommended_hardware_envelope",
    "validate_hardware_envelope",
    "validate_operator_discovery",
    "validate_operator_support",
    "validate_principle_discovery",
    "validate_principle_support",
    "validate_state_variable_discovery",
    "validate_state_variable_support",
    "validate_symbolic_discovery_target",
    "validate_symbolic_structure_target",
]
