"""Workstream 003: CNCC certificates and proposer-independence gates.

This module codifies five v3 requirements from the AI system notes:
- latent/operator proposals must provide observability-identifiability certificates,
- latent/operator proposals must satisfy minimality (MDL + out-of-regime gains),
- latent variables must be physically anchored,
- kernel/memory/fractional macros must compress symbolically or carry bounded error certificates,
- the system must remain functional when the neural proposer is replaced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal, cast

NormName = Literal["l1", "l2", "linf"]

_MISSING: object = object()
_CNCC_MACRO_KINDS: frozenset[str] = frozenset({"kernel", "memory", "fractional"})


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
class ObservabilityCertificateSpec:
    """Observability/identifiability evidence for a latent or macro proposal."""

    name: str = "candidate"
    identifiable_from_admissible_data: bool = False
    realistic_measurements: bool = False
    equivalence_or_gauge_declared: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_tag(self.name, field_name="name"))


@dataclass(frozen=True, slots=True)
class MinimalitySpec:
    """CNCC minimality declaration for latent/operator proposals."""

    name: str = "candidate"
    reduces_description_length: bool = False
    improves_out_of_regime_generalization: bool = False
    in_sample_only_improvement: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_tag(self.name, field_name="name"))


@dataclass(frozen=True, slots=True)
class PhysicalAnchoringSpec:
    """Physical anchoring declaration for latent variable proposals."""

    name: str = "candidate"
    measurable_microstructural_descriptor: bool = False
    constrained_coarse_graining: bool = False
    conservation_residual_or_defect_field: bool = False
    thermodynamic_internal_variable: bool = False
    bounded: bool = True
    has_dissipation_structure: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_tag(self.name, field_name="name"))


@dataclass(frozen=True, slots=True)
class ApproximationCertificate:
    """Certificate for a bounded approximation error in a target envelope."""

    envelope: str
    discretization: str
    norm: NormName
    bound: float
    relative_bound: float
    sample_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "envelope", _normalize_tag(self.envelope, field_name="envelope"))
        object.__setattr__(
            self,
            "discretization",
            _normalize_tag(self.discretization, field_name="discretization"),
        )

        if self.norm not in {"l1", "l2", "linf"}:
            raise ValueError("norm must be one of: l1, l2, linf")
        if not _is_nonnegative_finite(self.bound):
            raise ValueError("bound must be finite and >= 0")
        if not _is_nonnegative_finite(self.relative_bound):
            raise ValueError("relative_bound must be finite and >= 0")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be > 0")

    def within(self, tolerance: float) -> bool:
        """Return ``True`` if absolute error bound is within ``tolerance``."""
        _validate_nonnegative_finite(tolerance, field_name="tolerance")
        return self.bound <= tolerance

    def relative_within(self, tolerance: float) -> bool:
        """Return ``True`` if relative error bound is within ``tolerance``."""
        _validate_nonnegative_finite(tolerance, field_name="tolerance")
        return self.relative_bound <= tolerance


@dataclass(frozen=True, slots=True)
class MacroOperatorSpec:
    """Compression/certificate declaration for a macro operator proposal."""

    name: str = "candidate"
    kind: str = "kernel"
    symbolic_compression_available: bool = False
    approximation_certificate: ApproximationCertificate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_tag(self.name, field_name="name"))
        object.__setattr__(self, "kind", _normalize_tag(self.kind, field_name="kind"))


@dataclass(frozen=True, slots=True)
class ProposerIndependenceSpec:
    """Capability declaration for operating without a neural proposer."""

    system_functions_without_proposer: bool = False
    proposer_is_optional: bool = True
    proposer_used_as_prior_only: bool = True
    supports_probabilistic_grammar_prior: bool = False
    supports_lightweight_policy_network: bool = False
    supports_hand_built_heuristics: bool = False


def validate_observability_certificate(
    spec: ObservabilityCertificateSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate the CNCC observability/identifiability requirement."""
    parsed = _coerce_observability(spec)
    violations: list[str] = []

    if not parsed.identifiable_from_admissible_data:
        violations.append("observability.unidentifiable")
    if not parsed.realistic_measurements:
        violations.append("observability.unrealistic_measurement_requirements")
    if not parsed.equivalence_or_gauge_declared:
        violations.append("observability.missing_equivalence_or_gauge")

    return _evaluation_from_violations(
        violations,
        pass_note="observability certificate accepted",
    )


def validate_minimality(spec: MinimalitySpec | Mapping[str, object]) -> GateEvaluation:
    """Validate CNCC minimality (MDL + out-of-regime gains)."""
    parsed = _coerce_minimality(spec)
    violations: list[str] = []

    if not parsed.reduces_description_length:
        violations.append("minimality.fails_mdl_minimality")
    if not parsed.improves_out_of_regime_generalization:
        violations.append("minimality.fails_generalization_minimality")
    if parsed.in_sample_only_improvement:
        violations.append("minimality.in_sample_only_improvement")

    return _evaluation_from_violations(violations, pass_note="minimality certificate accepted")


def validate_physical_anchoring(spec: PhysicalAnchoringSpec | Mapping[str, object]) -> GateEvaluation:
    """Validate CNCC physical-anchoring obligations for latent variables."""
    parsed = _coerce_anchoring(spec)
    violations: list[str] = []

    if not any(
        (
            parsed.measurable_microstructural_descriptor,
            parsed.constrained_coarse_graining,
            parsed.conservation_residual_or_defect_field,
            parsed.thermodynamic_internal_variable,
        )
    ):
        violations.append("anchoring.missing_physical_anchor")

    if parsed.thermodynamic_internal_variable:
        if not parsed.bounded:
            violations.append("anchoring.thermodynamic_unbounded")
        if not parsed.has_dissipation_structure:
            violations.append("anchoring.thermodynamic_missing_dissipation")

    return _evaluation_from_violations(violations, pass_note="physical anchoring accepted")


def validate_macro_compression_or_bounded_error(
    spec: MacroOperatorSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate compression/certification obligations for macro operators."""
    parsed = _coerce_macro(spec)
    violations: list[str] = []
    notes: list[str] = []

    if parsed.kind in _CNCC_MACRO_KINDS:
        if not parsed.symbolic_compression_available and parsed.approximation_certificate is None:
            violations.append("macro.needs_compression_or_error_certificate")
        if parsed.symbolic_compression_available:
            notes.append("macro.symbolic_compression_available")
        if parsed.approximation_certificate is not None:
            notes.append("macro.certified_bounded_error")
    else:
        notes.append("macro.outside_cncc_scope")

    return _evaluation_from_violations(
        violations,
        pass_note="macro compression/certificate requirement accepted",
        extra_notes=notes,
    )


def validate_proposer_independence(
    spec: ProposerIndependenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate v3 proposer optionality/replacement requirements."""
    parsed = _coerce_proposer(spec)
    violations: list[str] = []
    notes: list[str] = []

    if not parsed.proposer_is_optional:
        violations.append("proposer.hard_dependency")
    if not parsed.system_functions_without_proposer:
        violations.append("proposer.no_proposer_fallback")
    if not parsed.proposer_used_as_prior_only:
        violations.append("proposer.not_prior_only")

    replacement_count = sum(
        (
            parsed.supports_probabilistic_grammar_prior,
            parsed.supports_lightweight_policy_network,
            parsed.supports_hand_built_heuristics,
        )
    )
    if replacement_count < 1:
        violations.append("proposer.no_supported_replacement")

    if parsed.supports_probabilistic_grammar_prior:
        notes.append("proposer.supports_probabilistic_grammar_prior")
    if parsed.supports_lightweight_policy_network:
        notes.append("proposer.supports_lightweight_policy_network")
    if parsed.supports_hand_built_heuristics:
        notes.append("proposer.supports_hand_built_heuristics")

    return _evaluation_from_violations(
        violations,
        pass_note="proposer independence accepted",
        extra_notes=notes,
    )


def evaluate_workstream_003(
    *,
    observability: ObservabilityCertificateSpec | Mapping[str, object] | None = None,
    minimality: MinimalitySpec | Mapping[str, object] | None = None,
    anchoring: PhysicalAnchoringSpec | Mapping[str, object] | None = None,
    macro: MacroOperatorSpec | Mapping[str, object] | None = None,
    proposer: ProposerIndependenceSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Evaluate the configured Workstream 003 requirements."""
    if all(component is None for component in (observability, minimality, anchoring, macro, proposer)):
        raise ValueError(
            "at least one component (observability, minimality, anchoring, macro, proposer) "
            "is required"
        )

    decision = GateEvaluation(accepted=True)

    if observability is not None:
        decision = decision.merge(validate_observability_certificate(observability))
    if minimality is not None:
        decision = decision.merge(validate_minimality(minimality))
    if anchoring is not None:
        decision = decision.merge(validate_physical_anchoring(anchoring))
    if macro is not None:
        decision = decision.merge(validate_macro_compression_or_bounded_error(macro))
    if proposer is not None:
        decision = decision.merge(validate_proposer_independence(proposer))

    return decision


def is_workstream_003_ready(
    *,
    observability: ObservabilityCertificateSpec | Mapping[str, object] | None = None,
    minimality: MinimalitySpec | Mapping[str, object] | None = None,
    anchoring: PhysicalAnchoringSpec | Mapping[str, object] | None = None,
    macro: MacroOperatorSpec | Mapping[str, object] | None = None,
    proposer: ProposerIndependenceSpec | Mapping[str, object] | None = None,
) -> bool:
    """Boolean wrapper around :func:`evaluate_workstream_003`."""
    return evaluate_workstream_003(
        observability=observability,
        minimality=minimality,
        anchoring=anchoring,
        macro=macro,
        proposer=proposer,
    ).accepted


def certify_bounded_approximation_error(
    reference: Sequence[float],
    approximation: Sequence[float],
    *,
    norm: NormName = "linf",
    envelope: str = "default",
    discretization: str = "default",
) -> ApproximationCertificate:
    """Create an approximation error certificate from paired sample values."""
    reference_values = _coerce_vector(reference, field_name="reference")
    approximation_values = _coerce_vector(approximation, field_name="approximation")

    if len(reference_values) != len(approximation_values):
        raise ValueError("reference and approximation must have the same length")
    if not reference_values:
        raise ValueError("reference and approximation must be non-empty")

    errors = tuple(
        abs(lhs - rhs) for lhs, rhs in zip(reference_values, approximation_values, strict=True)
    )
    bound = _norm_value(errors, norm=norm)

    reference_scale = max(
        _norm_value(tuple(abs(value) for value in reference_values), norm=norm),
        1.0,
    )
    relative_bound = bound / reference_scale

    return ApproximationCertificate(
        envelope=envelope,
        discretization=discretization,
        norm=norm,
        bound=bound,
        relative_bound=relative_bound,
        sample_size=len(reference_values),
    )


def certify_bounded_error(
    reference: Sequence[float],
    approximation: Sequence[float],
    *,
    norm: NormName = "linf",
    envelope: str = "default",
    discretization: str = "default",
) -> ApproximationCertificate:
    """Alias for :func:`certify_bounded_approximation_error`."""
    return certify_bounded_approximation_error(
        reference,
        approximation,
        norm=norm,
        envelope=envelope,
        discretization=discretization,
    )


def validate_identifiability_certificate(
    spec: ObservabilityCertificateSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_observability_certificate`."""
    return validate_observability_certificate(spec)


def validate_identifiability(spec: ObservabilityCertificateSpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_observability_certificate`."""
    return validate_observability_certificate(spec)


def validate_cncc_minimality(spec: MinimalitySpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_minimality`."""
    return validate_minimality(spec)


def validate_anchor_certificate(spec: PhysicalAnchoringSpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_physical_anchoring`."""
    return validate_physical_anchoring(spec)


def validate_macro_requirement(spec: MacroOperatorSpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_macro_compression_or_bounded_error`."""
    return validate_macro_compression_or_bounded_error(spec)


def validate_proposer_replacement(
    spec: ProposerIndependenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_proposer_independence`."""
    return validate_proposer_independence(spec)


def validate_proposer_optionality(
    spec: ProposerIndependenceSpec | Mapping[str, object],
) -> GateEvaluation:
    """Alias for :func:`validate_proposer_independence`."""
    return validate_proposer_independence(spec)


def check_workstream_003(
    *,
    observability: ObservabilityCertificateSpec | Mapping[str, object] | None = None,
    minimality: MinimalitySpec | Mapping[str, object] | None = None,
    anchoring: PhysicalAnchoringSpec | Mapping[str, object] | None = None,
    macro: MacroOperatorSpec | Mapping[str, object] | None = None,
    proposer: ProposerIndependenceSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Alias for :func:`evaluate_workstream_003`."""
    return evaluate_workstream_003(
        observability=observability,
        minimality=minimality,
        anchoring=anchoring,
        macro=macro,
        proposer=proposer,
    )


def _coerce_observability(
    spec: ObservabilityCertificateSpec | Mapping[str, object],
) -> ObservabilityCertificateSpec:
    if isinstance(spec, ObservabilityCertificateSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an ObservabilityCertificateSpec or mapping")

    return ObservabilityCertificateSpec(
        name=_optional_str(
            spec,
            ("name", "latent_name", "variable"),
            field_name="name",
            default="candidate",
        ),
        identifiable_from_admissible_data=_required_bool(
            spec,
            (
                "identifiable_from_admissible_data",
                "identifiable",
                "observable_from_admissible_data",
            ),
            field_name="identifiable_from_admissible_data",
        ),
        realistic_measurements=_required_bool(
            spec,
            ("realistic_measurements", "uses_realistic_measurements", "measurement_realism"),
            field_name="realistic_measurements",
        ),
        equivalence_or_gauge_declared=_optional_bool(
            spec,
            (
                "equivalence_or_gauge_declared",
                "declares_equivalence_or_gauge",
                "gauge_declared",
            ),
            field_name="equivalence_or_gauge_declared",
            default=True,
        ),
    )


def _coerce_minimality(spec: MinimalitySpec | Mapping[str, object]) -> MinimalitySpec:
    if isinstance(spec, MinimalitySpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a MinimalitySpec or mapping")

    return MinimalitySpec(
        name=_optional_str(spec, ("name", "candidate"), field_name="name", default="candidate"),
        reduces_description_length=_required_bool(
            spec,
            ("reduces_description_length", "reduces_mdl", "mdl_reduction"),
            field_name="reduces_description_length",
        ),
        improves_out_of_regime_generalization=_required_bool(
            spec,
            (
                "improves_out_of_regime_generalization",
                "improves_generalization",
                "improves_extrapolation",
            ),
            field_name="improves_out_of_regime_generalization",
        ),
        in_sample_only_improvement=_optional_bool(
            spec,
            ("in_sample_only_improvement", "only_in_sample_fit_improvement"),
            field_name="in_sample_only_improvement",
            default=False,
        ),
    )


def _coerce_anchoring(spec: PhysicalAnchoringSpec | Mapping[str, object]) -> PhysicalAnchoringSpec:
    if isinstance(spec, PhysicalAnchoringSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a PhysicalAnchoringSpec or mapping")

    return PhysicalAnchoringSpec(
        name=_optional_str(spec, ("name", "latent_name"), field_name="name", default="candidate"),
        measurable_microstructural_descriptor=_optional_bool(
            spec,
            (
                "measurable_microstructural_descriptor",
                "microstructural_descriptor",
                "anchor_microstructural_descriptor",
            ),
            field_name="measurable_microstructural_descriptor",
            default=False,
        ),
        constrained_coarse_graining=_optional_bool(
            spec,
            (
                "constrained_coarse_graining",
                "coarse_graining_anchor",
                "anchor_constrained_coarse_graining",
            ),
            field_name="constrained_coarse_graining",
            default=False,
        ),
        conservation_residual_or_defect_field=_optional_bool(
            spec,
            (
                "conservation_residual_or_defect_field",
                "conservation_residual_anchor",
                "defect_field_anchor",
            ),
            field_name="conservation_residual_or_defect_field",
            default=False,
        ),
        thermodynamic_internal_variable=_optional_bool(
            spec,
            (
                "thermodynamic_internal_variable",
                "thermodynamic_anchor",
                "anchor_thermodynamic_internal_variable",
            ),
            field_name="thermodynamic_internal_variable",
            default=False,
        ),
        bounded=_optional_bool(
            spec,
            ("bounded", "thermodynamic_bounded"),
            field_name="bounded",
            default=True,
        ),
        has_dissipation_structure=_optional_bool(
            spec,
            ("has_dissipation_structure", "thermodynamic_has_dissipation_structure"),
            field_name="has_dissipation_structure",
            default=True,
        ),
    )


def _coerce_macro(spec: MacroOperatorSpec | Mapping[str, object]) -> MacroOperatorSpec:
    if isinstance(spec, MacroOperatorSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a MacroOperatorSpec or mapping")

    certificate = _coerce_certificate(
        _pick_first(
            spec,
            (
                "approximation_certificate",
                "bounded_approximation_certificate",
                "error_certificate",
            ),
        )
    )

    return MacroOperatorSpec(
        name=_optional_str(spec, ("name", "macro_name"), field_name="name", default="candidate"),
        kind=_optional_str(spec, ("kind", "macro_kind", "operator_kind"), field_name="kind", default="kernel"),
        symbolic_compression_available=_required_bool(
            spec,
            (
                "symbolic_compression_available",
                "symbolically_compressible",
                "supports_symbolic_compression",
            ),
            field_name="symbolic_compression_available",
        ),
        approximation_certificate=certificate,
    )


def _coerce_proposer(spec: ProposerIndependenceSpec | Mapping[str, object]) -> ProposerIndependenceSpec:
    if isinstance(spec, ProposerIndependenceSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a ProposerIndependenceSpec or mapping")

    return ProposerIndependenceSpec(
        system_functions_without_proposer=_required_bool(
            spec,
            (
                "system_functions_without_proposer",
                "functions_without_proposer",
                "works_without_proposer",
            ),
            field_name="system_functions_without_proposer",
        ),
        proposer_is_optional=_optional_bool(
            spec,
            ("proposer_is_optional", "proposer_optional"),
            field_name="proposer_is_optional",
            default=True,
        ),
        proposer_used_as_prior_only=_optional_bool(
            spec,
            ("proposer_used_as_prior_only", "proposer_is_prior_only"),
            field_name="proposer_used_as_prior_only",
            default=True,
        ),
        supports_probabilistic_grammar_prior=_optional_bool(
            spec,
            ("supports_probabilistic_grammar_prior", "probabilistic_grammar_prior"),
            field_name="supports_probabilistic_grammar_prior",
            default=False,
        ),
        supports_lightweight_policy_network=_optional_bool(
            spec,
            ("supports_lightweight_policy_network", "lightweight_policy_network"),
            field_name="supports_lightweight_policy_network",
            default=False,
        ),
        supports_hand_built_heuristics=_optional_bool(
            spec,
            ("supports_hand_built_heuristics", "hand_built_heuristics"),
            field_name="supports_hand_built_heuristics",
            default=False,
        ),
    )


def _coerce_certificate(raw: object) -> ApproximationCertificate | None:
    if raw is _MISSING or raw is None:
        return None
    if isinstance(raw, ApproximationCertificate):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("approximation_certificate must be an ApproximationCertificate or mapping")

    certificate = cast("Mapping[str, object]", raw)
    norm_raw = _pick_first(certificate, ("norm", "error_norm"))
    norm_value: NormName = _as_norm(norm_raw, field_name="norm") if norm_raw is not _MISSING else "linf"

    return ApproximationCertificate(
        envelope=_optional_str(
            certificate,
            ("envelope", "target_envelope"),
            field_name="envelope",
            default="default",
        ),
        discretization=_optional_str(
            certificate,
            ("discretization", "target_discretization"),
            field_name="discretization",
            default="default",
        ),
        norm=norm_value,
        bound=_required_float(certificate, ("bound", "absolute_bound", "error_bound"), field_name="bound"),
        relative_bound=_optional_float(
            certificate,
            ("relative_bound", "rel_bound"),
            field_name="relative_bound",
            default=0.0,
        ),
        sample_size=_optional_int(
            certificate,
            ("sample_size", "n_samples"),
            field_name="sample_size",
            default=1,
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


def _optional_float(
    spec: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    field_name: str,
    default: float,
) -> float:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        return default
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


def _optional_str(
    spec: Mapping[str, object],
    keys: tuple[str, ...],
    *,
    field_name: str,
    default: str,
) -> str:
    raw = _pick_first(spec, keys)
    if raw is _MISSING:
        return default
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


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    return value


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalize_tag(value: str, *, field_name: str) -> str:
    return _as_str(value, field_name=field_name).lower()


def _as_norm(value: object, *, field_name: str) -> NormName:
    text = _as_str(value, field_name=field_name)
    if text in {"l1", "l2", "linf"}:
        return cast("NormName", text)
    raise ValueError(f"{field_name} must be one of: l1, l2, linf")


def _coerce_vector(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    out: list[float] = []
    for index, raw in enumerate(values):
        value = _as_float(raw, field_name=f"{field_name}[{index}]")
        if not isfinite(value):
            raise ValueError(f"{field_name}[{index}] must be finite")
        out.append(value)
    return tuple(out)


def _norm_value(values: Sequence[float], *, norm: NormName) -> float:
    if norm == "linf":
        return max(values, default=0.0)
    if norm == "l1":
        return sum(values)
    if norm == "l2":
        return sqrt(sum(value * value for value in values))
    raise ValueError("norm must be one of: l1, l2, linf")


def _is_nonnegative_finite(value: float) -> bool:
    return isfinite(value) and value >= 0.0


def _validate_nonnegative_finite(value: float, *, field_name: str) -> None:
    if not _is_nonnegative_finite(value):
        raise ValueError(f"{field_name} must be finite and >= 0")


def _sorted_unique(values: Iterable[str]) -> tuple[str, ...]:
    out = {value.strip() for value in values if value.strip()}
    return tuple(sorted(out))


__all__ = [
    "ApproximationCertificate",
    "GateEvaluation",
    "MacroOperatorSpec",
    "MinimalitySpec",
    "NormName",
    "ObservabilityCertificateSpec",
    "PhysicalAnchoringSpec",
    "ProposerIndependenceSpec",
    "certify_bounded_approximation_error",
    "certify_bounded_error",
    "check_workstream_003",
    "evaluate_workstream_003",
    "is_workstream_003_ready",
    "validate_anchor_certificate",
    "validate_cncc_minimality",
    "validate_identifiability",
    "validate_identifiability_certificate",
    "validate_macro_compression_or_bounded_error",
    "validate_macro_requirement",
    "validate_minimality",
    "validate_observability_certificate",
    "validate_physical_anchoring",
    "validate_proposer_independence",
    "validate_proposer_optionality",
    "validate_proposer_replacement",
]
