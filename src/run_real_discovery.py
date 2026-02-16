#!/usr/bin/env python3
"""Real equation discovery engine.

Performs genuine evolutionary program synthesis to discover constitutive
laws for viscoelastic materials, certifies them through a tiered ladder,
and assesses novelty against strong baselines.

Usage:
    python src/run_real_discovery.py
"""

from __future__ import annotations

import sys
import time

from discovery.artifacts import write_all_artifacts
from discovery.baselines import fit_all_baselines
from discovery.certification import CertificateLadder
from discovery.data_gen import (
    REGIMES,
    SAMPLING_CONFIGS,
    generate_all_splits,
    merge_training_splits,
)
from discovery.novelty import NoveltyClassification, assess_novelty, build_ws5_specs
from discovery.search import EvolutionarySearch, SearchConfig

SEP = "=" * 72


def main() -> int:
    print(SEP)
    print("  REAL EQUATION DISCOVERY ENGINE")
    print("  Domain: Viscoelastic constitutive laws")
    print("  Method: Evolutionary program synthesis")
    print(SEP)

    # ── 1. Generate synthetic data ────────────────────────────────────
    print("\n[1/6] Generating synthetic data...")
    t0 = time.monotonic()
    splits = generate_all_splits(seed=42)
    print(f"  {len(splits)} splits generated:")
    for name, split in sorted(splits.items()):
        print(f"    {name}: {len(split.eps)} points")

    # Merge training splits (coarse, non-extreme)
    train_eps, train_deps, train_sigma = merge_training_splits(splits)
    print(f"  Training set: {len(train_eps)} points")

    # Validation split
    val_split = splits.get("out_of_regime_fine") or splits.get("out_of_regime_coarse")
    if val_split is None:
        print("  ERROR: no validation split found")
        return 1
    print(f"  Validation set: {len(val_split.eps)} points ({val_split.name})")

    # ── 2. Fit baselines ──────────────────────────────────────────────
    print("\n[2/6] Fitting baselines...")
    baselines = fit_all_baselines(
        train_eps, train_deps, train_sigma,
        val_split.eps, val_split.deps, val_split.sigma,
    )
    print(f"  {'Name':<30s} {'Train L2':>10s} {'Val L2':>10s} {'MDL':>6s}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 6}")
    for b in baselines:
        print(
            f"  {b.name:<30s} {b.train_residual_l2:>10.4f} "
            f"{b.validation_residual_l2:>10.4f} {b.mdl:>6.0f}"
        )

    # ── 3. Run evolutionary search ────────────────────────────────────
    print("\n[3/6] Running evolutionary search...")
    config = SearchConfig(
        population_size=100,
        max_generations=50,
        min_unique_candidates=500,
        seed=42,
    )
    search = EvolutionarySearch(config, train_eps, train_deps, train_sigma)
    t_search = time.monotonic()
    result = search.run()
    search_elapsed = time.monotonic() - t_search

    print(f"  {len(result.all_candidates)} unique candidates in "
          f"{result.generations_run} generations ({search_elapsed:.1f}s)")
    print(f"  Rejections: non_finite={result.rejections.non_finite}, "
          f"tree_too_large={result.rejections.tree_too_large}, "
          f"duplicate={result.rejections.duplicate_hash}")
    print(f"\n  Top 5 candidates:")
    for i, cand in enumerate(result.top_k[:5], 1):
        print(f"    {i}. fitness={cand.fitness:.6f} MDL={cand.mdl:.0f} "
              f"L2={cand.train_residual_l2:.6f}")
        print(f"       {cand.tree.to_infix()}")

    # ── 4. Certify top candidate ──────────────────────────────────────
    print("\n[4/6] Running certificate ladder on top candidate...")
    best_tree = result.top_k[0].tree
    ladder = CertificateLadder(best_tree, splits, baselines)
    cert_report = ladder.run_all_tiers()

    for tier in cert_report.tier_results:
        status = "PASS" if tier.passed else "FAIL"
        print(f"  Tier {tier.tier}: {status}")
        for check_name, passed in tier.checks.items():
            s = "+" if passed else "x"
            detail = tier.details.get(check_name, "")
            print(f"    [{s}] {check_name}: {detail}")

    print(f"  Highest tier passed: {cert_report.highest_tier_passed}")

    # ── 5. Novelty assessment ─────────────────────────────────────────
    print("\n[5/6] Assessing novelty against baselines...")
    novelty_report = assess_novelty(
        best_tree,
        baselines,
        val_split.eps, val_split.deps, val_split.sigma,
    )

    print(f"  Classification: {novelty_report.classification}")
    print(f"  Closest baseline: {novelty_report.closest_baseline} "
          f"(residual={novelty_report.closest_residual:.4f})")
    print(f"  Reasoning: {novelty_report.reasoning}")

    if novelty_report.mdl_gate.get("applied"):
        print(f"  MDL gate: {novelty_report.mdl_gate.get('reason', '')}")

    # Show equivalence attempt summary
    baseline_results: dict[str, list[str]] = {}
    for attempt in novelty_report.evidence:
        if attempt.baseline_name not in baseline_results:
            baseline_results[attempt.baseline_name] = []
        status = "MATCH" if attempt.succeeds else "no match"
        baseline_results[attempt.baseline_name].append(
            f"{attempt.transform_type}={status}"
        )
    print(f"\n  Equivalence attempts:")
    for bname, attempts in baseline_results.items():
        print(f"    {bname}: {', '.join(attempts)}")

    # ── 6. Write artifacts ────────────────────────────────────────────
    print("\n[6/6] Writing artifacts...")
    artifact_config = {
        "search": {
            "population_size": config.population_size,
            "max_generations": config.max_generations,
            "min_unique_candidates": config.min_unique_candidates,
            "seed": config.seed,
            "mdl_penalty": config.mdl_penalty,
        },
        "regimes": list(REGIMES.keys()),
        "sampling_configs": {k: {"n_points": v.n_points, "dt": v.dt}
                             for k, v in SAMPLING_CONFIGS.items()},
    }

    write_all_artifacts(
        output_dir="artifacts",
        config=artifact_config,
        baselines=baselines,
        search_result=result,
        cert_report=cert_report,
        novelty_report=novelty_report,
    )
    print("  Artifacts written to artifacts/")

    total_elapsed = time.monotonic() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DISCOVERY COMPLETE")
    print(SEP)
    print(f"\n  Candidate: {best_tree.to_infix()}")
    print(f"  Classification: {novelty_report.classification}")
    print(f"  Highest tier: {cert_report.highest_tier_passed}")
    print(f"  Total time: {total_elapsed:.1f}s")

    if novelty_report.classification == NoveltyClassification.NEW:
        print(f"\n  NOVEL CONSTITUTIVE LAW DISCOVERED")
        print(f"  This law is structurally distinct from all tested baselines")
        print(f"  and achieves superior parsimony (lower MDL) at comparable error.")
    elif novelty_report.classification == NoveltyClassification.REFORMULATION:
        print(f"\n  Result is a REFORMULATION of known physics.")
        print(f"  The discovered form may be useful but is not fundamentally new.")
        print(f"  To identify novel physics, the following probes are needed:")
        print(f"    - Multi-scale experimental data with microstructural imaging")
        print(f"    - Non-equilibrium loading paths (cyclic, creep-recovery)")
        print(f"    - Temperature-dependent measurements")
    else:
        print(f"\n  Result matches KNOWN baseline: {novelty_report.closest_baseline}")

    print(f"\n{SEP}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
