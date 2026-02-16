"""Workstream 001: constrained field/operator gates and CNCC checks.

This module codifies three requirements from the design notes:
- coarse-graining operators ``G`` must be constrained (bounded support/projection bounds),
- latent variables must pass CNCC-style anti-cheating checks,
- macro operators must either compress symbolically or provide a certified bounded error.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal, cast

NormName = Literal["l1", "l2", "linf"]

_ALLOWED_ANCHOR_TYPES: frozenset[str] = frozenset(
    {
        "measurable_microstructural_descriptor",
        "constrained_coarse_graining",
        "conservation_residual_or_defect_field",
        "thermodynamic_internal_variable",
    }
)

_BOUNDED_MACRO_KINDS: frozenset[str] = frozenset({"kernel", "memory", "fractional"})


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
class ConstrainedOperatorSpec:
    """Specification for a coarse-graining/transform operator ``G``."""

    kind: str
    bounded_support: bool = False
    support_radius: float | None = None
    projection_levels: int | None = None
    coefficient_bound: float | None = None

    def __post_init__(self) -> None:
        normalized_kind = _normalize_tag(self.kind, field_name="ConstrainedOperatorSpec.kind")
        object.__setattr__(self, "kind", normalized_kind)

        if self.support_radius is not None and not _is_positive_finite(self.support_radius):
            raise ValueError("ConstrainedOperatorSpec.support_radius must be > 0 when provided")

        if self.projection_levels is not None and self.projection_levels <= 0:
            raise ValueError("ConstrainedOperatorSpec.projection_levels must be > 0")

        if self.coefficient_bound is not None and not _is_positive_finite(self.coefficient_bound):
            raise ValueError("ConstrainedOperatorSpec.coefficient_bound must be > 0")


@dataclass(frozen=True, slots=True)
class LatentVariableSpec:
    """CNCC-relevant attributes for a latent variable proposal."""

    name: str
    identifiable_from_admissible_data: bool
    realistic_measurements: bool
    reduces_description_length: bool
    improves_out_of_regime_generalization: bool
    anchor_type: str
    bounded: bool = True
    has_dissipation_structure: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _normalize_tag(self.name, field_name="LatentVariableSpec.name")
        )
        anchor = _normalize_tag(self.anchor_type, field_name="LatentVariableSpec.anchor_type")
        object.__setattr__(self, "anchor_type", anchor)


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
class OperatorMacroSpec:
    """CNCC-relevant attributes for kernel/memory/fractional operator macros."""

    name: str
    kind: str
    constrained: bool
    symbolic_compression_available: bool
    approximation_certificate: ApproximationCertificate | None = None
    bounded_support: bool = False
    support_radius: float | None = None
    coefficient_bound: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _normalize_tag(self.name, field_name="OperatorMacroSpec.name")
        )
        object.__setattr__(
            self, "kind", _normalize_tag(self.kind, field_name="OperatorMacroSpec.kind")
        )

        if self.support_radius is not None and not _is_positive_finite(self.support_radius):
            raise ValueError("OperatorMacroSpec.support_radius must be > 0 when provided")

        if self.coefficient_bound is not None and not _is_positive_finite(self.coefficient_bound):
            raise ValueError("OperatorMacroSpec.coefficient_bound must be > 0 when provided")


def validate_constrained_operator(
    spec: ConstrainedOperatorSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate that a coarse-graining operator ``G`` is constrained."""
    parsed = _coerce_constrained_operator(spec)
    violations: list[str] = []

    if parsed.kind == "averaging_kernel":
        if not parsed.bounded_support:
            violations.append("g.unbounded_support")
        if parsed.support_radius is None:
            violations.append("g.missing_support_radius")
        if parsed.coefficient_bound is None:
            violations.append("g.unbounded_coefficients")
    elif parsed.kind == "multiscale_projection":
        if parsed.projection_levels is None:
            violations.append("g.missing_projection_levels")
    else:
        if not parsed.bounded_support and parsed.projection_levels is None:
            violations.append("g.unconstrained_operator")

    return _evaluation_from_violations(violations, pass_note="constrained operator accepted")


def validate_latent_variable(
    spec: LatentVariableSpec | Mapping[str, object],
) -> GateEvaluation:
    """Validate CNCC anti-cheating requirements for latent variables."""
    parsed = _coerce_latent(spec)
    violations: list[str] = []

    if not parsed.identifiable_from_admissible_data:
        violations.append("latent.unidentifiable")

    if not parsed.realistic_measurements:
        violations.append("latent.unrealistic_measurement_requirements")

    if not parsed.reduces_description_length:
        violations.append("latent.fails_mdl_minimality")

    if not parsed.improves_out_of_regime_generalization:
        violations.append("latent.fails_generalization_minimality")

    if parsed.anchor_type not in _ALLOWED_ANCHOR_TYPES:
        violations.append("latent.missing_physical_anchor")

    if parsed.anchor_type == "thermodynamic_internal_variable":
        if not parsed.bounded:
            violations.append("latent.thermodynamic_unbounded")
        if not parsed.has_dissipation_structure:
            violations.append("latent.thermodynamic_missing_dissipation")

    return _evaluation_from_violations(violations, pass_note="latent variable accepted")


def validate_operator_macro(spec: OperatorMacroSpec | Mapping[str, object]) -> GateEvaluation:
    """Validate CNCC requirements for nonlocal/memory/fractional operator macros."""
    parsed = _coerce_macro(spec)
    violations: list[str] = []

    if parsed.kind in _BOUNDED_MACRO_KINDS and not parsed.constrained:
        violations.append("macro.unconstrained")

    if parsed.kind == "kernel":
        if not parsed.bounded_support:
            violations.append("macro.kernel_unbounded_support")
        if parsed.support_radius is None:
            violations.append("macro.kernel_missing_support_radius")
        if parsed.coefficient_bound is None:
            violations.append("macro.kernel_unbounded_coefficients")

    if not parsed.symbolic_compression_available and parsed.approximation_certificate is None:
        violations.append("macro.needs_compression_or_error_certificate")

    return _evaluation_from_violations(violations, pass_note="operator macro accepted")


def evaluate_cncc(
    *,
    coarse_operator: ConstrainedOperatorSpec | Mapping[str, object] | None = None,
    latent: LatentVariableSpec | Mapping[str, object] | None = None,
    macro: OperatorMacroSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Evaluate CNCC gates across the provided components."""
    if coarse_operator is None and latent is None and macro is None:
        raise ValueError("at least one component (coarse_operator, latent, macro) is required")

    decision = GateEvaluation(accepted=True)

    if coarse_operator is not None:
        decision = decision.merge(validate_constrained_operator(coarse_operator))

    if latent is not None:
        decision = decision.merge(validate_latent_variable(latent))

    if macro is not None:
        decision = decision.merge(validate_operator_macro(macro))

    return decision


def is_cncc_compliant(
    *,
    coarse_operator: ConstrainedOperatorSpec | Mapping[str, object] | None = None,
    latent: LatentVariableSpec | Mapping[str, object] | None = None,
    macro: OperatorMacroSpec | Mapping[str, object] | None = None,
) -> bool:
    """Convenience boolean wrapper around :func:`evaluate_cncc`."""
    return evaluate_cncc(
        coarse_operator=coarse_operator,
        latent=latent,
        macro=macro,
    ).accepted


def certify_bounded_approximation_error(
    reference: Sequence[float],
    approximation: Sequence[float],
    *,
    norm: NormName = "linf",
    envelope: str = "default",
    discretization: str = "default",
) -> ApproximationCertificate:
    """Create an error certificate for a numerical approximation.

    The certificate is deterministic and uses a direct norm on pointwise error
    over the provided discrete samples.
    """
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
        _norm_value(tuple(abs(value) for value in reference_values), norm=norm), 1.0
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


def validate_macro_operator(spec: OperatorMacroSpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_operator_macro`."""
    return validate_operator_macro(spec)


def validate_constrained_g(spec: ConstrainedOperatorSpec | Mapping[str, object]) -> GateEvaluation:
    """Alias for :func:`validate_constrained_operator`."""
    return validate_constrained_operator(spec)


def check_cncc(
    *,
    coarse_operator: ConstrainedOperatorSpec | Mapping[str, object] | None = None,
    latent: LatentVariableSpec | Mapping[str, object] | None = None,
    macro: OperatorMacroSpec | Mapping[str, object] | None = None,
) -> GateEvaluation:
    """Alias for :func:`evaluate_cncc`."""
    return evaluate_cncc(coarse_operator=coarse_operator, latent=latent, macro=macro)


def _coerce_constrained_operator(
    spec: ConstrainedOperatorSpec | Mapping[str, object],
) -> ConstrainedOperatorSpec:
    if isinstance(spec, ConstrainedOperatorSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a ConstrainedOperatorSpec or mapping")

    return ConstrainedOperatorSpec(
        kind=_as_str(spec.get("kind"), field_name="kind"),
        bounded_support=_as_bool(spec.get("bounded_support", False), field_name="bounded_support"),
        support_radius=_as_optional_float(spec.get("support_radius"), field_name="support_radius"),
        projection_levels=_as_optional_int(
            spec.get("projection_levels"),
            field_name="projection_levels",
        ),
        coefficient_bound=_as_optional_float(
            spec.get("coefficient_bound"),
            field_name="coefficient_bound",
        ),
    )


def _coerce_latent(spec: LatentVariableSpec | Mapping[str, object]) -> LatentVariableSpec:
    if isinstance(spec, LatentVariableSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be a LatentVariableSpec or mapping")

    return LatentVariableSpec(
        name=_as_str(spec.get("name"), field_name="name"),
        identifiable_from_admissible_data=_as_bool(
            spec.get("identifiable_from_admissible_data"),
            field_name="identifiable_from_admissible_data",
        ),
        realistic_measurements=_as_bool(
            spec.get("realistic_measurements"),
            field_name="realistic_measurements",
        ),
        reduces_description_length=_as_bool(
            spec.get("reduces_description_length"),
            field_name="reduces_description_length",
        ),
        improves_out_of_regime_generalization=_as_bool(
            spec.get("improves_out_of_regime_generalization"),
            field_name="improves_out_of_regime_generalization",
        ),
        anchor_type=_as_str(spec.get("anchor_type"), field_name="anchor_type"),
        bounded=_as_bool(spec.get("bounded", True), field_name="bounded"),
        has_dissipation_structure=_as_bool(
            spec.get("has_dissipation_structure", True),
            field_name="has_dissipation_structure",
        ),
    )


def _coerce_macro(spec: OperatorMacroSpec | Mapping[str, object]) -> OperatorMacroSpec:
    if isinstance(spec, OperatorMacroSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("spec must be an OperatorMacroSpec or mapping")

    certificate_raw = spec.get("approximation_certificate")
    certificate: ApproximationCertificate | None
    if certificate_raw is None:
        certificate = None
    elif isinstance(certificate_raw, ApproximationCertificate):
        certificate = certificate_raw
    elif isinstance(certificate_raw, Mapping):
        certificate = ApproximationCertificate(
            envelope=_as_str(certificate_raw.get("envelope", "default"), field_name="envelope"),
            discretization=_as_str(
                certificate_raw.get("discretization", "default"),
                field_name="discretization",
            ),
            norm=_as_norm(certificate_raw.get("norm", "linf"), field_name="norm"),
            bound=_as_float(certificate_raw.get("bound"), field_name="bound"),
            relative_bound=_as_float(
                certificate_raw.get("relative_bound", 0.0),
                field_name="relative_bound",
            ),
            sample_size=_as_int(certificate_raw.get("sample_size", 1), field_name="sample_size"),
        )
    else:
        raise TypeError("approximation_certificate must be an ApproximationCertificate or mapping")

    return OperatorMacroSpec(
        name=_as_str(spec.get("name"), field_name="name"),
        kind=_as_str(spec.get("kind"), field_name="kind"),
        constrained=_as_bool(spec.get("constrained"), field_name="constrained"),
        symbolic_compression_available=_as_bool(
            spec.get("symbolic_compression_available"),
            field_name="symbolic_compression_available",
        ),
        approximation_certificate=certificate,
        bounded_support=_as_bool(spec.get("bounded_support", False), field_name="bounded_support"),
        support_radius=_as_optional_float(spec.get("support_radius"), field_name="support_radius"),
        coefficient_bound=_as_optional_float(
            spec.get("coefficient_bound"),
            field_name="coefficient_bound",
        ),
    )


def _evaluation_from_violations(violations: Sequence[str], *, pass_note: str) -> GateEvaluation:
    normalized = _sorted_unique(violations)
    if normalized:
        return GateEvaluation(accepted=False, reason_codes=normalized)
    return GateEvaluation(accepted=True, notes=(pass_note,))


def _norm_value(values: Sequence[float], *, norm: NormName) -> float:
    if norm == "linf":
        return max(values, default=0.0)
    if norm == "l1":
        return sum(values)
    if norm == "l2":
        return sqrt(sum(value * value for value in values))
    raise ValueError("norm must be one of: l1, l2, linf")


def _coerce_vector(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    out: list[float] = []
    for index, raw in enumerate(values):
        value = _as_float(raw, field_name=f"{field_name}[{index}]")
        if not isfinite(value):
            raise ValueError(f"{field_name}[{index}] must be finite")
        out.append(value)
    return tuple(out)


def _as_norm(value: object, *, field_name: str) -> NormName:
    text = _as_str(value, field_name=field_name)
    if text in {"l1", "l2", "linf"}:
        return cast("NormName", text)
    raise ValueError(f"{field_name} must be one of: l1, l2, linf")


def _as_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalize_tag(value: str, *, field_name: str) -> str:
    return _as_str(value, field_name=field_name).lower()


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


def _as_optional_float(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, field_name=field_name)


def _as_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int")
    return value


def _as_optional_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field_name=field_name)


def _is_positive_finite(value: float) -> bool:
    return isfinite(value) and value > 0.0


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
    "ConstrainedOperatorSpec",
    "GateEvaluation",
    "LatentVariableSpec",
    "NormName",
    "OperatorMacroSpec",
    "certify_bounded_approximation_error",
    "certify_bounded_error",
    "check_cncc",
    "evaluate_cncc",
    "is_cncc_compliant",
    "validate_constrained_g",
    "validate_constrained_operator",
    "validate_latent_variable",
    "validate_macro_operator",
    "validate_operator_macro",
]
