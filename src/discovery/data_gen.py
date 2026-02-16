"""Synthetic viscoelastic data generator.

Generates stress-strain data from a ground truth constitutive law across
multiple strain regimes, sampling rates (dt), and noise levels.  The
ground truth is known ONLY to this module -- the search engine never
imports it.

Ground truth model:
    sigma = E1*eps + E3*eps^3 + eta*deps + beta*eps^2*deps

where deps = d(epsilon)/dt.  The beta*eps^2*deps cross-term is the
"novel" component that standard baselines cannot capture.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# ── Ground truth constants (private to this module) ───────────────────
_E1: float = 100.0
_E3: float = 50.0
_ETA: float = 10.0
_BETA: float = 5.0


def _ground_truth_stress(eps: float, deps: float) -> float:
    """Evaluate the true constitutive law."""
    return _E1 * eps + _E3 * eps ** 3 + _ETA * deps + _BETA * eps ** 2 * deps


# ── Data structures ──────────────────────────────────────────────────

@dataclass(frozen=True)
class DataRegime:
    """A named regime defining ranges for strain and strain rate."""

    name: str
    eps_min: float
    eps_max: float
    deps_min: float
    deps_max: float


@dataclass(frozen=True)
class SamplingConfig:
    """Defines the number of sample points and time-step size."""

    name: str
    n_points: int
    dt: float


@dataclass(frozen=True)
class DataSplit:
    """A dataset split with parallel lists of eps, deps, sigma."""

    name: str
    regime: str
    sampling: str
    eps: list[float]
    deps: list[float]
    sigma: list[float]
    # Time-series displacement data (for Prony baseline)
    time: list[float] | None = None
    displacement: list[float] | None = None


# ── Pre-defined regimes ──────────────────────────────────────────────

REGIMES: dict[str, DataRegime] = {
    "low_strain": DataRegime("low_strain", -0.01, 0.01, -0.1, 0.1),
    "moderate_strain": DataRegime("moderate_strain", -0.1, 0.1, -1.0, 1.0),
    "high_strain": DataRegime("high_strain", -0.5, 0.5, -5.0, 5.0),
    "extreme": DataRegime("extreme", -1.0, 1.0, -10.0, 10.0),
}

SAMPLING_CONFIGS: dict[str, SamplingConfig] = {
    "coarse": SamplingConfig("coarse", n_points=50, dt=0.01),
    "fine": SamplingConfig("fine", n_points=200, dt=0.001),
}


# ── Data generation ──────────────────────────────────────────────────

def _generate_displacement_timeseries(
    regime: DataRegime,
    n_steps: int,
    dt: float,
    rng: random.Random,
) -> tuple[list[float], list[float]]:
    """Generate a synthetic displacement time-series u(t).

    Uses a sum of sinusoids with random frequencies and amplitudes
    chosen so that the resulting strains stay within the regime bounds.
    """
    n_modes = 3
    freqs = [rng.uniform(0.5, 5.0) for _ in range(n_modes)]
    amps = [rng.uniform(0.2, 1.0) * (regime.eps_max - regime.eps_min) / (2 * n_modes)
            for _ in range(n_modes)]
    phases = [rng.uniform(0, 2 * math.pi) for _ in range(n_modes)]

    times: list[float] = []
    displacements: list[float] = []
    for i in range(n_steps):
        t = i * dt
        u = sum(a * math.sin(2 * math.pi * f * t + p)
                for a, f, p in zip(amps, freqs, phases))
        times.append(t)
        displacements.append(u)

    return times, displacements


def generate_regime_data(
    regime: DataRegime,
    sampling: SamplingConfig,
    seed: int = 42,
    noise_level: float = 0.0,
) -> DataSplit:
    """Generate data for one regime at one sampling config.

    The dt-sensitive path: generate a displacement time-series, compute
    eps(t) = u(t) and deps(t) = (u(t) - u(t-1)) / dt.  This means the
    same underlying physics produces slightly different (eps, deps) pairs
    at different dt, enabling real dt-drift detection.
    """
    rng = random.Random(seed)
    n_steps = sampling.n_points + 1  # extra point for finite difference

    times, displacements = _generate_displacement_timeseries(
        regime, n_steps, sampling.dt, rng,
    )

    eps_list: list[float] = []
    deps_list: list[float] = []
    sigma_list: list[float] = []

    for i in range(1, n_steps):
        eps = displacements[i]
        deps = (displacements[i] - displacements[i - 1]) / sampling.dt
        sigma = _ground_truth_stress(eps, deps)

        if noise_level > 0.0:
            sigma += rng.gauss(0.0, noise_level * max(abs(sigma), 1e-10))

        eps_list.append(eps)
        deps_list.append(deps)
        sigma_list.append(sigma)

    return DataSplit(
        name=f"{regime.name}_{sampling.name}",
        regime=regime.name,
        sampling=sampling.name,
        eps=eps_list,
        deps=deps_list,
        sigma=sigma_list,
        time=times[1:],
        displacement=displacements[1:],
    )


def generate_all_splits(
    seed: int = 42,
    noise_level: float = 0.0,
) -> dict[str, DataSplit]:
    """Generate all regime x sampling combinations plus out-of-regime validation.

    Returns a dictionary keyed by split name (e.g. "moderate_strain_coarse").
    """
    splits: dict[str, DataSplit] = {}

    for regime_name, regime in REGIMES.items():
        for samp_name, samp in SAMPLING_CONFIGS.items():
            split_seed = seed + hash(f"{regime_name}_{samp_name}") % 10000
            splits[f"{regime_name}_{samp_name}"] = generate_regime_data(
                regime, samp, seed=split_seed, noise_level=noise_level,
            )

    # Out-of-regime validation with a different seed
    extreme = REGIMES["extreme"]
    for samp_name, samp in SAMPLING_CONFIGS.items():
        val_seed = seed + 9999 + hash(samp_name) % 10000
        splits[f"out_of_regime_{samp_name}"] = generate_regime_data(
            extreme, samp, seed=val_seed, noise_level=noise_level,
        )

    return splits


def merge_training_splits(
    splits: dict[str, DataSplit],
    regimes: tuple[str, ...] = ("low_strain", "moderate_strain", "high_strain"),
    sampling: str = "coarse",
) -> tuple[list[float], list[float], list[float]]:
    """Merge specified regime splits into single training arrays."""
    eps: list[float] = []
    deps: list[float] = []
    sigma: list[float] = []
    for regime in regimes:
        key = f"{regime}_{sampling}"
        if key in splits:
            eps.extend(splits[key].eps)
            deps.extend(splits[key].deps)
            sigma.extend(splits[key].sigma)
    return eps, deps, sigma
