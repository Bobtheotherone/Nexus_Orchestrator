"""Baseline constitutive models and pure-Python least-squares fitting.

Provides five strong baselines for comparison against discovered theories:
1. Hooke (linear elastic)
2. Kelvin-Voigt (linear viscoelastic)
3. Nonlinear polynomial (univariate in eps, degree 5)
4. Multivariate polynomial with cross terms (total degree 3 in eps, deps)
5. Prony series (internal-variable memory model, 2 branches)

All fitting is done in pure Python via Gaussian elimination -- no numpy
or scipy required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


# ═══════════════════════════════════════════════════════════════════════
# BaselineResult
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BaselineResult:
    """A fitted baseline with its parameters and residuals."""

    name: str
    formula: str
    parameters: dict[str, float]
    n_parameters: int
    formula_complexity: int
    train_residual_l2: float
    train_residual_linf: float
    validation_residual_l2: float
    validation_residual_linf: float
    _predict_fn: Callable[[float, float], float] = field(repr=False)

    @property
    def mdl(self) -> float:
        return float(self.formula_complexity + self.n_parameters)

    def predict(self, eps: float, deps: float) -> float:
        return self._predict_fn(eps, deps)

    def predict_batch(
        self, eps_list: list[float], deps_list: list[float],
    ) -> list[float]:
        return [self._predict_fn(e, d) for e, d in zip(eps_list, deps_list)]


# ═══════════════════════════════════════════════════════════════════════
# Pure-Python linear algebra
# ═══════════════════════════════════════════════════════════════════════

def _gauss_solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve Ax = b via Gaussian elimination with partial pivoting.

    Works for systems up to ~15x15.  Returns None if singular.
    """
    n = len(b)
    # Augmented matrix
    aug = [row[:] + [bi] for row, bi in zip(a, b)]

    for col in range(n):
        # Partial pivot
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, n):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        if max_val < 1e-14:
            return None  # Singular
        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / pivot
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]

    # Back-substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(aug[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (aug[i][n] - s) / aug[i][i]

    return x


def _normal_equations(
    features: list[list[float]], targets: list[float],
) -> list[float] | None:
    """Solve (X^T X) theta = X^T y."""
    n_samples = len(targets)
    n_features = len(features[0]) if features else 0

    # X^T X
    xtx = [[0.0] * n_features for _ in range(n_features)]
    for i in range(n_features):
        for j in range(n_features):
            s = 0.0
            for k in range(n_samples):
                s += features[k][i] * features[k][j]
            xtx[i][j] = s

    # X^T y
    xty = [0.0] * n_features
    for i in range(n_features):
        s = 0.0
        for k in range(n_samples):
            s += features[k][i] * targets[k]
        xty[i] = s

    return _gauss_solve(xtx, xty)


def _residuals(
    pred: list[float], actual: list[float],
) -> tuple[float, float]:
    """Compute L2 and L-inf residuals."""
    n = len(actual)
    if n == 0:
        return 0.0, 0.0
    ss = 0.0
    mx = 0.0
    for p, a in zip(pred, actual):
        diff = abs(p - a)
        ss += diff * diff
        if diff > mx:
            mx = diff
    return math.sqrt(ss / n), mx


# ═══════════════════════════════════════════════════════════════════════
# Baseline 1: Hooke's law
# ═══════════════════════════════════════════════════════════════════════

def fit_hooke(
    eps_train: list[float],
    deps_train: list[float],
    sigma_train: list[float],
    eps_val: list[float],
    deps_val: list[float],
    sigma_val: list[float],
) -> BaselineResult:
    """sigma = E * eps.  Fit E by least-squares."""
    features = [[e] for e in eps_train]
    theta = _normal_equations(features, sigma_train)
    if theta is None:
        theta = [0.0]
    E = theta[0]

    def predict(eps: float, deps: float) -> float:
        return E * eps

    train_pred = [predict(e, d) for e, d in zip(eps_train, deps_train)]
    val_pred = [predict(e, d) for e, d in zip(eps_val, deps_val)]
    tl2, tli = _residuals(train_pred, sigma_train)
    vl2, vli = _residuals(val_pred, sigma_val)

    return BaselineResult(
        name="hooke",
        formula="sigma = E * eps",
        parameters={"E": E},
        n_parameters=1,
        formula_complexity=3,
        train_residual_l2=tl2,
        train_residual_linf=tli,
        validation_residual_l2=vl2,
        validation_residual_linf=vli,
        _predict_fn=predict,
    )


# ═══════════════════════════════════════════════════════════════════════
# Baseline 2: Kelvin-Voigt
# ═══════════════════════════════════════════════════════════════════════

def fit_kelvin_voigt(
    eps_train: list[float],
    deps_train: list[float],
    sigma_train: list[float],
    eps_val: list[float],
    deps_val: list[float],
    sigma_val: list[float],
) -> BaselineResult:
    """sigma = E*eps + eta*deps.  Fit E, eta."""
    features = [[e, d] for e, d in zip(eps_train, deps_train)]
    theta = _normal_equations(features, sigma_train)
    if theta is None:
        theta = [0.0, 0.0]
    E, eta = theta[0], theta[1]

    def predict(eps: float, deps: float) -> float:
        return E * eps + eta * deps

    train_pred = [predict(e, d) for e, d in zip(eps_train, deps_train)]
    val_pred = [predict(e, d) for e, d in zip(eps_val, deps_val)]
    tl2, tli = _residuals(train_pred, sigma_train)
    vl2, vli = _residuals(val_pred, sigma_val)

    return BaselineResult(
        name="kelvin_voigt",
        formula="sigma = E*eps + eta*deps",
        parameters={"E": E, "eta": eta},
        n_parameters=2,
        formula_complexity=5,
        train_residual_l2=tl2,
        train_residual_linf=tli,
        validation_residual_l2=vl2,
        validation_residual_linf=vli,
        _predict_fn=predict,
    )


# ═══════════════════════════════════════════════════════════════════════
# Baseline 3: Nonlinear polynomial (univariate, degree 5)
# ═══════════════════════════════════════════════════════════════════════

def fit_polynomial(
    eps_train: list[float],
    deps_train: list[float],
    sigma_train: list[float],
    eps_val: list[float],
    deps_val: list[float],
    sigma_val: list[float],
    degree: int = 5,
) -> BaselineResult:
    """sigma = sum(a_n * eps^n, n=0..degree)."""
    features = [[e ** n for n in range(degree + 1)] for e in eps_train]
    theta = _normal_equations(features, sigma_train)
    if theta is None:
        theta = [0.0] * (degree + 1)
    coeffs = list(theta)

    def predict(eps: float, deps: float) -> float:
        return sum(c * eps ** n for n, c in enumerate(coeffs))

    train_pred = [predict(e, d) for e, d in zip(eps_train, deps_train)]
    val_pred = [predict(e, d) for e, d in zip(eps_val, deps_val)]
    tl2, tli = _residuals(train_pred, sigma_train)
    vl2, vli = _residuals(val_pred, sigma_val)

    params = {f"a{n}": c for n, c in enumerate(coeffs)}
    terms = " + ".join(f"a{n}*eps^{n}" for n in range(degree + 1))

    return BaselineResult(
        name="polynomial_univariate",
        formula=f"sigma = {terms}",
        parameters=params,
        n_parameters=degree + 1,
        formula_complexity=2 * (degree + 1) + degree,
        train_residual_l2=tl2,
        train_residual_linf=tli,
        validation_residual_l2=vl2,
        validation_residual_linf=vli,
        _predict_fn=predict,
    )


# ═══════════════════════════════════════════════════════════════════════
# Baseline 4: Multivariate polynomial (total degree 3 with cross terms)
# ═══════════════════════════════════════════════════════════════════════

def _multivar_monomials(
    eps: float, deps: float, total_degree: int,
) -> list[float]:
    """Generate all monomials eps^i * deps^j with i+j <= total_degree."""
    result: list[float] = []
    for i in range(total_degree + 1):
        for j in range(total_degree + 1 - i):
            result.append(eps ** i * deps ** j)
    return result


def _multivar_monomial_names(total_degree: int) -> list[str]:
    names: list[str] = []
    for i in range(total_degree + 1):
        for j in range(total_degree + 1 - i):
            if i == 0 and j == 0:
                names.append("1")
            elif j == 0:
                names.append(f"eps^{i}" if i > 1 else "eps")
            elif i == 0:
                names.append(f"deps^{j}" if j > 1 else "deps")
            else:
                ei = f"eps^{i}" if i > 1 else "eps"
                dj = f"deps^{j}" if j > 1 else "deps"
                names.append(f"{ei}*{dj}")
    return names


def fit_multivariate_polynomial(
    eps_train: list[float],
    deps_train: list[float],
    sigma_train: list[float],
    eps_val: list[float],
    deps_val: list[float],
    sigma_val: list[float],
    total_degree: int = 3,
) -> BaselineResult:
    """sigma = sum a_{ij} * eps^i * deps^j for i+j <= total_degree."""
    features = [
        _multivar_monomials(e, d, total_degree)
        for e, d in zip(eps_train, deps_train)
    ]
    theta = _normal_equations(features, sigma_train)
    n_mono = len(features[0]) if features else 0
    if theta is None:
        theta = [0.0] * n_mono
    coeffs = list(theta)
    names = _multivar_monomial_names(total_degree)

    def predict(eps: float, deps: float) -> float:
        monos = _multivar_monomials(eps, deps, total_degree)
        return sum(c * m for c, m in zip(coeffs, monos))

    train_pred = [predict(e, d) for e, d in zip(eps_train, deps_train)]
    val_pred = [predict(e, d) for e, d in zip(eps_val, deps_val)]
    tl2, tli = _residuals(train_pred, sigma_train)
    vl2, vli = _residuals(val_pred, sigma_val)

    params = {name: c for name, c in zip(names, coeffs)}
    formula = "sigma = " + " + ".join(
        f"a_{name}" for name in names
    )

    return BaselineResult(
        name="multivariate_polynomial",
        formula=formula,
        parameters=params,
        n_parameters=n_mono,
        formula_complexity=3 * n_mono,
        train_residual_l2=tl2,
        train_residual_linf=tli,
        validation_residual_l2=vl2,
        validation_residual_linf=vli,
        _predict_fn=predict,
    )


# ═══════════════════════════════════════════════════════════════════════
# Baseline 5: Prony series (internal-variable memory model)
# ═══════════════════════════════════════════════════════════════════════

def _prony_forward(
    eps_list: list[float],
    deps_list: list[float],
    dt: float,
    e_inf: float,
    branches: list[tuple[float, float]],  # [(E_k, tau_k), ...]
) -> list[float]:
    """Forward-integrate the Prony series ODE to predict stress.

    sigma(t) = E_inf * eps(t) + sum_k q_k(t)
    dq_k/dt = E_k * deps(t) - q_k / tau_k
    """
    n_branches = len(branches)
    q = [0.0] * n_branches  # internal variables start at zero
    sigma_list: list[float] = []

    for eps, deps in zip(eps_list, deps_list):
        # Update internal variables (implicit Euler for stability)
        for k in range(n_branches):
            e_k, tau_k = branches[k]
            if tau_k > 1e-15:
                # Implicit Euler: q_new = (q_old + dt * E_k * deps) / (1 + dt/tau_k)
                q[k] = (q[k] + dt * e_k * deps) / (1.0 + dt / tau_k)
            else:
                q[k] = 0.0

        sigma = e_inf * eps + sum(q)
        sigma_list.append(sigma)

    return sigma_list


def fit_prony_series(
    eps_train: list[float],
    deps_train: list[float],
    sigma_train: list[float],
    eps_val: list[float],
    deps_val: list[float],
    sigma_val: list[float],
    dt: float = 0.01,
    n_branches: int = 2,
    n_iters: int = 30,
) -> BaselineResult:
    """Fit a Prony series by coordinate descent.

    Optimizes E_inf and (E_k, tau_k) for each branch.
    """
    import random as _rng

    rng = _rng.Random(42)

    # Initial guesses
    e_inf = 50.0
    branches: list[list[float]] = [
        [rng.uniform(10.0, 100.0), rng.uniform(0.01, 1.0)]
        for _ in range(n_branches)
    ]

    def _eval_residual(
        ei: float, br: list[list[float]],
        eps: list[float], deps: list[float], sigma: list[float],
    ) -> float:
        br_tuples = [(b[0], b[1]) for b in br]
        pred = _prony_forward(eps, deps, dt, ei, br_tuples)
        return math.sqrt(sum((p - s) ** 2 for p, s in zip(pred, sigma)) / len(sigma))

    best_res = _eval_residual(e_inf, branches, eps_train, deps_train, sigma_train)

    for _iteration in range(n_iters):
        # Optimize E_inf
        for delta in [10.0, -10.0, 2.0, -2.0, 0.5, -0.5]:
            trial = e_inf + delta
            r = _eval_residual(trial, branches, eps_train, deps_train, sigma_train)
            if r < best_res:
                e_inf = trial
                best_res = r

        # Optimize each branch
        for k in range(n_branches):
            for param_idx in range(2):  # E_k, tau_k
                for delta in [10.0, -10.0, 2.0, -2.0, 0.5, -0.5, 0.1, -0.1]:
                    original = branches[k][param_idx]
                    branches[k][param_idx] = original + delta
                    if param_idx == 1 and branches[k][param_idx] <= 0.0:
                        branches[k][param_idx] = original
                        continue
                    r = _eval_residual(e_inf, branches, eps_train, deps_train, sigma_train)
                    if r < best_res:
                        best_res = r
                    else:
                        branches[k][param_idx] = original

    br_tuples = [(b[0], b[1]) for b in branches]
    train_pred = _prony_forward(eps_train, deps_train, dt, e_inf, br_tuples)
    val_pred = _prony_forward(eps_val, deps_val, dt, e_inf, br_tuples)
    tl2, tli = _residuals(train_pred, sigma_train)
    vl2, vli = _residuals(val_pred, sigma_val)

    params: dict[str, float] = {"E_inf": e_inf}
    for k, (ek, tk) in enumerate(br_tuples):
        params[f"E_{k + 1}"] = ek
        params[f"tau_{k + 1}"] = tk

    branch_terms = " + ".join(f"q{k + 1}(t)" for k in range(n_branches))
    formula = f"sigma = E_inf*eps + {branch_terms}; dq_k/dt = E_k*deps - q_k/tau_k"

    return BaselineResult(
        name="prony_series",
        formula=formula,
        parameters=params,
        n_parameters=1 + 2 * n_branches,
        formula_complexity=5 + 4 * n_branches,
        train_residual_l2=tl2,
        train_residual_linf=tli,
        validation_residual_l2=vl2,
        validation_residual_linf=vli,
        _predict_fn=lambda eps, deps, _ei=e_inf, _br=br_tuples: (
            _ei * eps + sum(ek * deps for ek, _ in _br)
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# Fit all baselines
# ═══════════════════════════════════════════════════════════════════════

def fit_all_baselines(
    eps_train: list[float],
    deps_train: list[float],
    sigma_train: list[float],
    eps_val: list[float],
    deps_val: list[float],
    sigma_val: list[float],
    dt: float = 0.01,
) -> list[BaselineResult]:
    """Fit all 5 baselines and return them sorted by validation L2."""
    baselines = [
        fit_hooke(eps_train, deps_train, sigma_train, eps_val, deps_val, sigma_val),
        fit_kelvin_voigt(eps_train, deps_train, sigma_train, eps_val, deps_val, sigma_val),
        fit_polynomial(eps_train, deps_train, sigma_train, eps_val, deps_val, sigma_val),
        fit_multivariate_polynomial(eps_train, deps_train, sigma_train, eps_val, deps_val, sigma_val),
        fit_prony_series(eps_train, deps_train, sigma_train, eps_val, deps_val, sigma_val, dt=dt),
    ]
    return sorted(baselines, key=lambda b: b.validation_residual_l2)
