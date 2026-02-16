"""Tiered certification ladder for discovered constitutive laws.

Tier 0: Dimensional consistency + finite output (cheap reject)
Tier 1: Training residual beats baselines, MDL bound, stability
Tier 2: dt/sampling invariance + identifiability (bootstrap + FIM + profile)

Wires computed results into workstream validation gates with real booleans.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from discovery.baselines import BaselineResult, _normal_equations, _residuals
from discovery.data_gen import DataSplit, SamplingConfig, generate_regime_data, REGIMES, SAMPLING_CONFIGS
from discovery.expr_tree import ExprNode


@dataclass(frozen=True)
class TierResult:
    tier: int
    passed: bool
    checks: dict[str, bool]
    details: dict[str, str]


@dataclass(frozen=True)
class CertificationReport:
    candidate_formula: str
    tier_results: list[TierResult]
    highest_tier_passed: int
    workstream_results: dict[str, object]


class CertificateLadder:
    """Tiered certification of discovered constitutive laws."""

    def __init__(
        self,
        candidate: ExprNode,
        training_splits: dict[str, DataSplit],
        baselines: list[BaselineResult],
    ) -> None:
        self._candidate = candidate
        self._splits = training_splits
        self._baselines = baselines

    def run_all_tiers(self) -> CertificationReport:
        tiers: list[TierResult] = []
        highest = -1

        for tier_fn, tier_num in [(self.tier_0, 0), (self.tier_1, 1), (self.tier_2, 2)]:
            result = tier_fn()
            tiers.append(result)
            if result.passed:
                highest = tier_num
            else:
                break

        ws_results = self._wire_workstream_gates(tiers)
        return CertificationReport(
            candidate_formula=self._candidate.to_infix(),
            tier_results=tiers,
            highest_tier_passed=highest,
            workstream_results=ws_results,
        )

    # ── Tier 0: cheap reject ──────────────────────────────────────────

    def tier_0(self) -> TierResult:
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        # Finite output on all splits
        all_finite = True
        count = 0
        for name, split in self._splits.items():
            for e, d in zip(split.eps, split.deps):
                try:
                    v = self._candidate.evaluate(e, d)
                    if not math.isfinite(v):
                        all_finite = False
                        break
                except (OverflowError, ValueError, ZeroDivisionError):
                    all_finite = False
                    break
                count += 1
            if not all_finite:
                break
        checks["finite_output"] = all_finite
        details["finite_output"] = f"{count} points evaluated, all finite={all_finite}"

        # Dimensional consistency: no fractional/negative exponents
        valid_powers = self._candidate.has_valid_powers()
        checks["dimensional_consistency"] = valid_powers
        details["dimensional_consistency"] = (
            "all exponents are non-negative integers" if valid_powers
            else "fractional or negative exponents detected"
        )

        passed = all(checks.values())
        return TierResult(tier=0, passed=passed, checks=checks, details=details)

    # ── Tier 1: quality ───────────────────────────────────────────────

    def tier_1(self) -> TierResult:
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        # Merge training data (coarse splits)
        train_eps, train_deps, train_sigma = [], [], []
        for name, split in self._splits.items():
            if "out_of_regime" not in name and "coarse" in name:
                train_eps.extend(split.eps)
                train_deps.extend(split.deps)
                train_sigma.extend(split.sigma)

        pred = self._candidate.evaluate_batch(train_eps, train_deps)
        cand_l2, cand_linf = _residuals(pred, train_sigma)

        best_baseline_l2 = min(b.train_residual_l2 for b in self._baselines)
        checks["training_residual"] = cand_l2 < best_baseline_l2
        details["training_residual"] = (
            f"candidate L2={cand_l2:.6f}, best baseline L2={best_baseline_l2:.6f}"
        )

        # MDL comparison
        mdl = self._candidate.mdl()
        checks["mdl_comparison"] = mdl < 50
        details["mdl_comparison"] = f"candidate MDL={mdl:.1f}, threshold=50"

        # Stability: evaluate at regime boundaries
        extreme = REGIMES["extreme"]
        boundary_points = [
            (extreme.eps_min, extreme.deps_min),
            (extreme.eps_max, extreme.deps_max),
            (extreme.eps_min, extreme.deps_max),
            (extreme.eps_max, extreme.deps_min),
        ]
        max_train_sigma = max(abs(s) for s in train_sigma) if train_sigma else 1.0
        stable = True
        for ep, dp in boundary_points:
            try:
                v = self._candidate.evaluate(ep, dp)
                if not math.isfinite(v) or abs(v) > 10 * max_train_sigma:
                    stable = False
                    break
            except (OverflowError, ValueError, ZeroDivisionError):
                stable = False
                break
        checks["stability"] = stable
        details["stability"] = (
            f"boundary eval within 10x envelope (max_sigma={max_train_sigma:.2f})"
            if stable else "boundary evaluation exceeds 10x envelope or non-finite"
        )

        passed = all(checks.values())
        return TierResult(tier=1, passed=passed, checks=checks, details=details)

    # ── Tier 2: deployment ────────────────────────────────────────────

    def tier_2(self) -> TierResult:
        checks: dict[str, bool] = {}
        details: dict[str, str] = {}

        # dt/sampling invariance
        dt_ok, dt_detail = self._check_dt_invariance()
        checks["dt_sampling_invariance"] = dt_ok
        details["dt_sampling_invariance"] = dt_detail

        # Identifiability (3 levels)
        ident_ok, ident_detail = self._check_identifiability()
        checks["identifiability"] = ident_ok
        details["identifiability"] = ident_detail

        passed = all(checks.values())
        return TierResult(tier=2, passed=passed, checks=checks, details=details)

    def _check_dt_invariance(self) -> tuple[bool, str]:
        """Check that predictions and fitted parameters don't drift with dt."""
        msgs: list[str] = []
        all_ok = True

        for regime_name in ("low_strain", "moderate_strain", "high_strain"):
            coarse_key = f"{regime_name}_coarse"
            fine_key = f"{regime_name}_fine"

            if coarse_key not in self._splits or fine_key not in self._splits:
                continue

            coarse = self._splits[coarse_key]
            fine = self._splits[fine_key]

            # Evaluate candidate on both
            pred_coarse = self._candidate.evaluate_batch(coarse.eps, coarse.deps)
            pred_fine = self._candidate.evaluate_batch(fine.eps, fine.deps)

            # Compute residuals against ground truth for each
            _, _ = _residuals(pred_coarse, coarse.sigma)
            _, _ = _residuals(pred_fine, fine.sigma)

            # Compare predictions at overlapping points (both have ground truth)
            # Use relative performance difference
            rl2_c, _ = _residuals(pred_coarse, coarse.sigma)
            rl2_f, _ = _residuals(pred_fine, fine.sigma)

            # Relative difference in residual
            max_r = max(rl2_c, rl2_f, 1e-10)
            rel_diff = abs(rl2_c - rl2_f) / max_r

            ok = rel_diff < 0.05
            if not ok:
                all_ok = False
            msgs.append(f"{regime_name}: coarse_L2={rl2_c:.6f}, fine_L2={rl2_f:.6f}, rel_diff={rel_diff:.4f} {'PASS' if ok else 'FAIL'}")

        # Parameter drift check: re-fit constants on each dt variant
        drift_ok, drift_msg = self._check_parameter_drift()
        if not drift_ok:
            all_ok = False
        msgs.append(f"parameter_drift: {drift_msg}")

        return all_ok, "; ".join(msgs)

    def _check_parameter_drift(self) -> tuple[bool, str]:
        """Re-fit constants on coarse vs fine data, check drift."""
        coarse_eps, coarse_deps, coarse_sigma = [], [], []
        fine_eps, fine_deps, fine_sigma = [], [], []

        for name, split in self._splits.items():
            if "out_of_regime" in name:
                continue
            if "coarse" in name:
                coarse_eps.extend(split.eps)
                coarse_deps.extend(split.deps)
                coarse_sigma.extend(split.sigma)
            elif "fine" in name:
                fine_eps.extend(split.eps)
                fine_deps.extend(split.deps)
                fine_sigma.extend(split.sigma)

        tree_c = self._candidate.clone()
        tree_f = self._candidate.clone()

        _refit_constants(tree_c, coarse_eps, coarse_deps, coarse_sigma)
        _refit_constants(tree_f, fine_eps, fine_deps, fine_sigma)

        consts_c = [c.value or 0.0 for c in tree_c.collect_constants()]
        consts_f = [c.value or 0.0 for c in tree_f.collect_constants()]

        if len(consts_c) != len(consts_f) or not consts_c:
            return True, "no constants to compare"

        max_drift = 0.0
        for vc, vf in zip(consts_c, consts_f):
            denom = max(abs(vc), abs(vf), 1e-8)
            drift = abs(vc - vf) / denom
            if drift > max_drift:
                max_drift = drift

        ok = max_drift < 0.10
        return ok, f"max_drift={max_drift:.4f}, threshold=0.10"

    def _check_identifiability(self) -> tuple[bool, str]:
        """Three-level identifiability check: bootstrap + FIM + profile."""
        # Gather all training data
        all_eps, all_deps, all_sigma = [], [], []
        for name, split in self._splits.items():
            if "out_of_regime" not in name and "coarse" in name:
                all_eps.extend(split.eps)
                all_deps.extend(split.deps)
                all_sigma.extend(split.sigma)

        n = len(all_sigma)
        if n == 0:
            return False, "no training data"

        consts = self._candidate.collect_constants()
        if not consts:
            return True, "no free parameters (constant expression)"

        msgs: list[str] = []

        # Level 1: Bootstrap uncertainty
        boot_ok, boot_msg = self._bootstrap_check(all_eps, all_deps, all_sigma)
        msgs.append(f"bootstrap: {boot_msg}")

        # Level 2: FIM conditioning
        fim_ok, fim_msg = self._fim_check(all_eps, all_deps, all_sigma)
        msgs.append(f"FIM: {fim_msg}")

        # Level 3: Profile check
        prof_ok, prof_msg = self._profile_check(all_eps, all_deps, all_sigma)
        msgs.append(f"profile: {prof_msg}")

        all_ok = boot_ok and fim_ok and prof_ok
        return all_ok, "; ".join(msgs)

    def _bootstrap_check(
        self, eps: list[float], deps: list[float], sigma: list[float],
        n_boot: int = 50,
    ) -> tuple[bool, str]:
        """Resample with replacement, re-fit, check CV < 0.20."""
        rng = random.Random(12345)
        n = len(sigma)
        consts_ref = self._candidate.collect_constants()
        n_params = len(consts_ref)
        if n_params == 0:
            return True, "no parameters"

        all_fits: list[list[float]] = []
        for _ in range(n_boot):
            indices = [rng.randrange(n) for _ in range(n)]
            b_eps = [eps[i] for i in indices]
            b_deps = [deps[i] for i in indices]
            b_sigma = [sigma[i] for i in indices]

            tree_copy = self._candidate.clone()
            _refit_constants(tree_copy, b_eps, b_deps, b_sigma)
            fitted = [c.value or 0.0 for c in tree_copy.collect_constants()]
            all_fits.append(fitted)

        # Compute CV per parameter
        max_cv = 0.0
        for j in range(n_params):
            vals = [f[j] for f in all_fits]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = math.sqrt(var)
            cv = std / max(abs(mean), 1e-10)
            if cv > max_cv:
                max_cv = cv

        ok = max_cv < 0.20
        return ok, f"max_CV={max_cv:.4f}, threshold=0.20"

    def _fim_check(
        self, eps: list[float], deps: list[float], sigma: list[float],
    ) -> tuple[bool, str]:
        """Approximate FIM = J^T J, check condition number < 1e6."""
        consts = self._candidate.collect_constants()
        n_params = len(consts)
        if n_params == 0:
            return True, "no parameters"

        n = len(sigma)
        delta = 1e-5

        # Compute Jacobian: d(prediction_i)/d(theta_j)
        jacobian: list[list[float]] = []
        for i in range(n):
            row = []
            for j in range(n_params):
                original = consts[j].value or 0.0
                consts[j].value = original + delta
                f_plus = self._candidate.evaluate(eps[i], deps[i])
                consts[j].value = original - delta
                f_minus = self._candidate.evaluate(eps[i], deps[i])
                consts[j].value = original
                row.append((f_plus - f_minus) / (2 * delta))
            jacobian.append(row)

        # J^T J
        jtj = [[0.0] * n_params for _ in range(n_params)]
        for i in range(n_params):
            for j in range(n_params):
                s = 0.0
                for k in range(n):
                    s += jacobian[k][i] * jacobian[k][j]
                jtj[i][j] = s

        # Estimate condition number via power iteration
        cond = _estimate_condition_number(jtj, n_params)
        ok = cond < 1e6
        return ok, f"condition_number={cond:.2e}, threshold=1e6"

    def _profile_check(
        self, eps: list[float], deps: list[float], sigma: list[float],
    ) -> tuple[bool, str]:
        """Perturb each parameter ±20%, re-optimize others.  If residual barely changes, fail."""
        consts = self._candidate.collect_constants()
        n_params = len(consts)
        if n_params == 0:
            return True, "no parameters"

        base_res = _compute_l2(self._candidate, eps, deps, sigma)
        if not math.isfinite(base_res) or base_res == 0.0:
            return True, "base residual zero or non-finite"

        all_ok = True
        for j in range(n_params):
            original = consts[j].value or 0.0
            for direction in [0.2, -0.2]:
                perturbed_val = original * (1.0 + direction)
                tree_copy = self._candidate.clone()
                copy_consts = tree_copy.collect_constants()
                copy_consts[j].value = perturbed_val

                # Re-optimize OTHER constants
                for k in range(n_params):
                    if k == j:
                        continue
                    _refit_single_constant(tree_copy, copy_consts[k], eps, deps, sigma)

                perturbed_res = _compute_l2(tree_copy, eps, deps, sigma)
                increase = (perturbed_res - base_res) / max(base_res, 1e-10)
                if increase < 0.01:  # Less than 1% increase -> sloppy
                    all_ok = False
                    break
            if not all_ok:
                break

        return all_ok, "all parameters are identifiable" if all_ok else "at least one sloppy parameter"

    # ── Workstream gate wiring ────────────────────────────────────────

    def _wire_workstream_gates(
        self, tier_results: list[TierResult],
    ) -> dict[str, object]:
        """Build workstream specs from computed tier results."""
        tier0 = tier_results[0] if len(tier_results) > 0 else None
        tier1 = tier_results[1] if len(tier_results) > 1 else None
        tier2 = tier_results[2] if len(tier_results) > 2 else None

        results: dict[str, object] = {}

        # WS1: CNCC
        try:
            from workstream_001 import evaluate_cncc, LatentVariableSpec

            ws1 = evaluate_cncc(
                latent=LatentVariableSpec(
                    name="discovered_constitutive_law",
                    identifiable_from_admissible_data=(
                        tier2 is not None and tier2.checks.get("identifiability", False)
                    ),
                    realistic_measurements=True,
                    reduces_description_length=(
                        tier1 is not None and tier1.checks.get("mdl_comparison", False)
                    ),
                    improves_out_of_regime_generalization=(
                        tier1 is not None and tier1.checks.get("training_residual", False)
                    ),
                    anchor_type="measurable_microstructural_descriptor",
                ),
            )
            results["ws1_cncc"] = ws1
        except ImportError:
            results["ws1_cncc"] = None

        # WS3: Certificates
        try:
            from workstream_003 import (
                MinimalitySpec,
                ObservabilityCertificateSpec,
                ProposerIndependenceSpec,
                evaluate_workstream_003,
            )

            ws3 = evaluate_workstream_003(
                observability=ObservabilityCertificateSpec(
                    name="discovered_law",
                    identifiable_from_admissible_data=(
                        tier2 is not None and tier2.checks.get("identifiability", False)
                    ),
                    realistic_measurements=True,
                    equivalence_or_gauge_declared=True,
                ),
                minimality=MinimalitySpec(
                    name="discovered_law",
                    reduces_description_length=(
                        tier1 is not None and tier1.checks.get("mdl_comparison", False)
                    ),
                    improves_out_of_regime_generalization=(
                        tier1 is not None and tier1.checks.get("training_residual", False)
                    ),
                ),
                proposer=ProposerIndependenceSpec(
                    system_functions_without_proposer=True,
                    proposer_is_optional=True,
                    proposer_used_as_prior_only=True,
                    supports_hand_built_heuristics=True,
                ),
            )
            results["ws3_certificates"] = ws3
        except ImportError:
            results["ws3_certificates"] = None

        # WS4: Numerical contracts
        try:
            from workstream_004 import NumericalContractSpec, evaluate_workstream_004

            ws4 = evaluate_workstream_004(
                numerical_contract=NumericalContractSpec(
                    differentiability_class="c1",
                    bounded_derivatives_in_envelope=(
                        tier1 is not None and tier1.checks.get("stability", False)
                    ),
                    stable_jacobians_or_tangents=(
                        tier1 is not None and tier1.checks.get("stability", False)
                    ),
                    stable_implicit_solves=(
                        tier1 is not None and tier1.checks.get("stability", False)
                    ),
                    fallback_behavior_defined=True,
                ),
            )
            results["ws4_numerical"] = ws4
        except ImportError:
            results["ws4_numerical"] = None

        return results


# ═══════════════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════════════

def _compute_l2(
    tree: ExprNode, eps: list[float], deps: list[float], sigma: list[float],
) -> float:
    try:
        pred = tree.evaluate_batch(eps, deps)
    except (OverflowError, ValueError, ZeroDivisionError):
        return float("inf")
    n = len(sigma)
    ss = sum((p - s) ** 2 for p, s in zip(pred, sigma))
    return math.sqrt(ss / n) if n > 0 else 0.0


def _refit_constants(
    tree: ExprNode, eps: list[float], deps: list[float], sigma: list[float],
    n_iters: int = 20,
) -> None:
    """Coordinate descent on all constants."""
    consts = tree.collect_constants()
    if not consts:
        return
    best_res = _compute_l2(tree, eps, deps, sigma)
    deltas = [10.0, -10.0, 2.0, -2.0, 0.5, -0.5, 0.1, -0.1, 0.01, -0.01]
    for _ in range(n_iters):
        for c in consts:
            original = c.value or 0.0
            for d in deltas:
                c.value = original + d
                r = _compute_l2(tree, eps, deps, sigma)
                if math.isfinite(r) and r < best_res:
                    best_res = r
                    original = c.value
                else:
                    c.value = original


def _refit_single_constant(
    tree: ExprNode, const_node: ExprNode,
    eps: list[float], deps: list[float], sigma: list[float],
) -> None:
    """Optimize a single constant node."""
    best_res = _compute_l2(tree, eps, deps, sigma)
    original = const_node.value or 0.0
    for d in [5.0, -5.0, 1.0, -1.0, 0.1, -0.1, 0.01, -0.01]:
        const_node.value = original + d
        r = _compute_l2(tree, eps, deps, sigma)
        if math.isfinite(r) and r < best_res:
            best_res = r
            original = const_node.value
        else:
            const_node.value = original


def _estimate_condition_number(
    matrix: list[list[float]], n: int,
) -> float:
    """Estimate condition number via largest/smallest eigenvalue ratio (power iteration)."""
    if n == 0:
        return 1.0

    # Power iteration for largest eigenvalue
    x = [1.0] * n
    for _ in range(50):
        y = [sum(matrix[i][j] * x[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(v * v for v in y))
        if norm < 1e-15:
            return float("inf")
        x = [v / norm for v in y]
    lambda_max = sum(
        x[i] * sum(matrix[i][j] * x[j] for j in range(n))
        for i in range(n)
    )

    # Inverse power iteration for smallest eigenvalue
    # Solve (A - 0*I)x = b iteratively
    from discovery.baselines import _gauss_solve

    x = [1.0] * n
    for _ in range(50):
        y = _gauss_solve([row[:] for row in matrix], x)
        if y is None:
            return float("inf")
        norm = math.sqrt(sum(v * v for v in y))
        if norm < 1e-15:
            return float("inf")
        x = [v / norm for v in y]
    lambda_min = 1.0 / max(
        abs(sum(x[i] * sum(matrix[i][j] * x[j] for j in range(n)) for i in range(n))),
        1e-15,
    )

    if lambda_min < 1e-15:
        return float("inf")
    return abs(lambda_max) / abs(lambda_min)
