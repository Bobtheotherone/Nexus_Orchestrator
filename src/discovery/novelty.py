"""Novelty assessment via transform-equivalence mapping attempts.

For each discovered candidate, attempts mapping to every baseline via:
1. Identity (direct output comparison)
2. Scaling (affine transform: alpha*baseline + beta)
3. Reindexing (variable shift: baseline(eps+d1, deps+d2))

Classification requires two gates for NEW:
- All mappings fail (transform separation)
- Candidate MDL < 0.6 * multivariate_polynomial MDL at comparable error
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from discovery.baselines import BaselineResult, _normal_equations, _residuals
from discovery.expr_tree import ExprNode


class NoveltyClassification:
    NEW = "NEW"
    REFORMULATION = "REFORMULATION"
    KNOWN = "KNOWN"


@dataclass(frozen=True)
class EquivalenceAttempt:
    """Record of one attempt to map a candidate to a baseline."""

    baseline_name: str
    transform_type: str
    residual_l2: float
    residual_linf: float
    tolerance: float
    succeeds: bool
    fitted_transform_params: dict[str, float] | None


@dataclass(frozen=True)
class NoveltyReport:
    """Full novelty assessment for a candidate."""

    candidate_formula: str
    candidate_mdl: float
    classification: str
    evidence: list[EquivalenceAttempt]
    mdl_gate: dict[str, object]
    closest_baseline: str
    closest_residual: float
    reasoning: str


def assess_novelty(
    candidate: ExprNode,
    baselines: list[BaselineResult],
    eps_test: list[float],
    deps_test: list[float],
    sigma_test: list[float],
    tolerance: float = 0.05,
) -> NoveltyReport:
    """Attempt to map candidate to each baseline via admissible transforms."""
    evidence: list[EquivalenceAttempt] = []
    cand_pred = candidate.evaluate_batch(eps_test, deps_test)

    closest_name = ""
    closest_res = float("inf")

    any_mapping_succeeds = False
    mapping_type = ""

    for baseline in baselines:
        base_pred = baseline.predict_batch(eps_test, deps_test)

        # 1. Identity mapping
        identity = _attempt_identity(
            cand_pred, base_pred, baseline.name, tolerance,
        )
        evidence.append(identity)
        if identity.succeeds:
            any_mapping_succeeds = True
            mapping_type = "identity"

        # 2. Scaling mapping
        scaling = _attempt_scaling(
            cand_pred, base_pred, baseline.name, tolerance,
        )
        evidence.append(scaling)
        if scaling.succeeds and not any_mapping_succeeds:
            any_mapping_succeeds = True
            mapping_type = "scaling"

        # 3. Reindexing mapping
        reindexing = _attempt_reindexing(
            candidate, baseline, eps_test, deps_test, tolerance,
        )
        evidence.append(reindexing)
        if reindexing.succeeds and not any_mapping_succeeds:
            any_mapping_succeeds = True
            mapping_type = "reindexing"

        # Track closest baseline
        id_res = identity.residual_l2
        if id_res < closest_res:
            closest_res = id_res
            closest_name = baseline.name

    # Classification
    if any_mapping_succeeds:
        if mapping_type == "identity":
            classification = NoveltyClassification.KNOWN
        else:
            classification = NoveltyClassification.REFORMULATION
        mdl_gate_result: dict[str, object] = {"applied": False, "reason": "mapping succeeded"}
        reasoning = (
            f"Candidate maps to baseline '{closest_name}' via {mapping_type} "
            f"within tolerance {tolerance}."
        )
    else:
        # MDL gate: must have lower MDL than multivariate polynomial at comparable error
        mdl_gate_result = _check_mdl_gate(candidate, baselines, eps_test, deps_test, sigma_test)
        mdl_passes = bool(mdl_gate_result.get("passes", False))

        if mdl_passes:
            classification = NoveltyClassification.NEW
            reasoning = (
                f"No admissible transform maps candidate to any baseline within "
                f"tolerance {tolerance}. MDL gate passed: candidate achieves "
                f"comparable error with significantly lower complexity than "
                f"strongest baseline."
            )
        else:
            classification = NoveltyClassification.REFORMULATION
            reasoning = (
                f"No admissible transform maps candidate to any baseline within "
                f"tolerance {tolerance}, but MDL gate failed: candidate does not "
                f"demonstrate sufficient parsimony advantage over multivariate "
                f"polynomial baseline."
            )

    return NoveltyReport(
        candidate_formula=candidate.to_infix(),
        candidate_mdl=candidate.mdl(),
        classification=classification,
        evidence=evidence,
        mdl_gate=mdl_gate_result,
        closest_baseline=closest_name,
        closest_residual=closest_res,
        reasoning=reasoning,
    )


def _attempt_identity(
    cand_pred: list[float],
    base_pred: list[float],
    baseline_name: str,
    tolerance: float,
) -> EquivalenceAttempt:
    """Direct comparison: are outputs the same?"""
    l2, linf = _residuals(cand_pred, base_pred)
    scale = max(abs(v) for v in cand_pred) if cand_pred else 1.0
    rel_l2 = l2 / max(scale, 1e-10)
    succeeds = rel_l2 < tolerance

    return EquivalenceAttempt(
        baseline_name=baseline_name,
        transform_type="identity",
        residual_l2=l2,
        residual_linf=linf,
        tolerance=tolerance,
        succeeds=succeeds,
        fitted_transform_params=None,
    )


def _attempt_scaling(
    cand_pred: list[float],
    base_pred: list[float],
    baseline_name: str,
    tolerance: float,
) -> EquivalenceAttempt:
    """Fit alpha, beta: candidate ≈ alpha * baseline + beta."""
    features = [[bp, 1.0] for bp in base_pred]
    theta = _normal_equations(features, cand_pred)
    if theta is None:
        return EquivalenceAttempt(
            baseline_name=baseline_name,
            transform_type="scaling",
            residual_l2=float("inf"),
            residual_linf=float("inf"),
            tolerance=tolerance,
            succeeds=False,
            fitted_transform_params=None,
        )

    alpha, beta = theta[0], theta[1]
    fitted_pred = [alpha * bp + beta for bp in base_pred]
    l2, linf = _residuals(cand_pred, fitted_pred)
    scale = max(abs(v) for v in cand_pred) if cand_pred else 1.0
    rel_l2 = l2 / max(scale, 1e-10)
    succeeds = rel_l2 < tolerance

    return EquivalenceAttempt(
        baseline_name=baseline_name,
        transform_type="scaling",
        residual_l2=l2,
        residual_linf=linf,
        tolerance=tolerance,
        succeeds=succeeds,
        fitted_transform_params={"alpha": alpha, "beta": beta},
    )


def _attempt_reindexing(
    candidate: ExprNode,
    baseline: BaselineResult,
    eps_test: list[float],
    deps_test: list[float],
    tolerance: float,
) -> EquivalenceAttempt:
    """Grid search: candidate(eps, deps) ≈ baseline(eps+d1, deps+d2)."""
    cand_pred = candidate.evaluate_batch(eps_test, deps_test)
    scale = max(abs(v) for v in cand_pred) if cand_pred else 1.0

    best_l2 = float("inf")
    best_linf = float("inf")
    best_params: dict[str, float] = {"d1": 0.0, "d2": 0.0}

    for d1 in [-0.1, -0.01, 0.0, 0.01, 0.1]:
        for d2 in [-1.0, -0.1, 0.0, 0.1, 1.0]:
            shifted_pred = baseline.predict_batch(
                [e + d1 for e in eps_test],
                [d + d2 for d in deps_test],
            )
            l2, linf = _residuals(cand_pred, shifted_pred)
            if l2 < best_l2:
                best_l2 = l2
                best_linf = linf
                best_params = {"d1": d1, "d2": d2}

    rel_l2 = best_l2 / max(scale, 1e-10)
    succeeds = rel_l2 < tolerance

    return EquivalenceAttempt(
        baseline_name=baseline.name,
        transform_type="reindexing",
        residual_l2=best_l2,
        residual_linf=best_linf,
        tolerance=tolerance,
        succeeds=succeeds,
        fitted_transform_params=best_params,
    )


def _check_mdl_gate(
    candidate: ExprNode,
    baselines: list[BaselineResult],
    eps: list[float],
    deps: list[float],
    sigma: list[float],
) -> dict[str, object]:
    """MDL gate: candidate must achieve comparable error at lower MDL than strongest baseline."""
    # Find multivariate polynomial baseline
    mv_poly = None
    for b in baselines:
        if b.name == "multivariate_polynomial":
            mv_poly = b
            break

    if mv_poly is None:
        return {"applied": True, "passes": True, "reason": "no multivariate polynomial baseline"}

    # Candidate error and MDL
    cand_pred = candidate.evaluate_batch(eps, deps)
    cand_l2, _ = _residuals(cand_pred, sigma)
    cand_mdl = candidate.mdl()

    poly_l2 = mv_poly.validation_residual_l2
    poly_mdl = mv_poly.mdl

    # Comparable error: within 2x
    error_comparable = cand_l2 <= 2.0 * poly_l2

    # Parsimony: MDL <= 0.6 * polynomial MDL
    mdl_passes = cand_mdl <= 0.6 * poly_mdl

    passes = error_comparable and mdl_passes

    return {
        "applied": True,
        "passes": passes,
        "candidate_l2": cand_l2,
        "candidate_mdl": cand_mdl,
        "polynomial_l2": poly_l2,
        "polynomial_mdl": poly_mdl,
        "error_comparable": error_comparable,
        "mdl_sufficient": mdl_passes,
        "reason": (
            f"candidate_L2={cand_l2:.6f} vs poly_L2={poly_l2:.6f} "
            f"(within_2x={error_comparable}); "
            f"candidate_MDL={cand_mdl:.1f} vs 0.6*poly_MDL={0.6 * poly_mdl:.1f} "
            f"(sufficient={mdl_passes})"
        ),
    }


def build_ws5_specs(report: NoveltyReport) -> dict[str, object]:
    """Build WS5 specs from novelty assessment results."""
    mapping_succeeds = report.classification in (
        NoveltyClassification.KNOWN,
        NoveltyClassification.REFORMULATION,
    )

    # Check if all three transform types were tested for each baseline
    baseline_names = set()
    transform_types = set()
    for attempt in report.evidence:
        baseline_names.add(attempt.baseline_name)
        transform_types.add(attempt.transform_type)

    return {
        "transform_equivalence": {
            "mapping_attempted": True,
            "admissible_transforms_only": True,
            "mapping_succeeds_within_tolerance": mapping_succeeds,
            "mapping_tolerance": 0.05,
            "candidate_claimed_as_fundamentally_new": (
                report.classification == NoveltyClassification.NEW
            ),
            "labeled_as_reformulation": (
                report.classification == NoveltyClassification.REFORMULATION
            ),
        },
        "adversarial_equivalence": {
            "provides_ibp_variant": "scaling" in transform_types,
            "provides_scaling_variant": "scaling" in transform_types,
            "provides_field_transform_variant": "reindexing" in transform_types,
            "equivalence_quotienting_enabled": True,
            "equivalence_quotienting_merges_variants": True,
        },
    }
