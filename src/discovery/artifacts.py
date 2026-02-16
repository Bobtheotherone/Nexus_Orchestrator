"""Artifact generation for discovery runs.

Writes all required output files: config, baselines, top-K theories,
novelty report, certification report, and deploy directory.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from discovery.baselines import BaselineResult
from discovery.certification import CertificationReport
from discovery.novelty import NoveltyReport, build_ws5_specs
from discovery.search import SearchResult


def write_all_artifacts(
    output_dir: str,
    config: dict[str, object],
    baselines: list[BaselineResult],
    search_result: SearchResult,
    cert_report: CertificationReport,
    novelty_report: NoveltyReport,
) -> None:
    """Write all required artifact files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    _write_config(output_dir, config)
    _write_baselines(output_dir, baselines)
    _write_top_k(output_dir, search_result)
    _write_novelty_report(output_dir, novelty_report)
    _write_cert_report(output_dir, cert_report, novelty_report, baselines)
    if cert_report.highest_tier_passed >= 2:
        _write_deploy(output_dir, cert_report, novelty_report)


def _write_config(output_dir: str, config: dict[str, object]) -> None:
    """Write discovery_run_config.yaml."""
    path = os.path.join(output_dir, "discovery_run_config.yaml")
    lines = [
        "# Discovery Run Configuration",
        f"timestamp: '{time.strftime('%Y-%m-%dT%H:%M:%S')}'",
        "engine: evolutionary_program_synthesis",
        "domain: viscoelastic_constitutive_laws",
        "",
    ]
    for key, val in config.items():
        if isinstance(val, dict):
            lines.append(f"{key}:")
            for k, v in val.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{key}: {val}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_baselines(output_dir: str, baselines: list[BaselineResult]) -> None:
    """Write baselines/ directory with one JSON per baseline."""
    baselines_dir = os.path.join(output_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)

    for b in baselines:
        data = {
            "name": b.name,
            "formula": b.formula,
            "parameters": b.parameters,
            "n_parameters": b.n_parameters,
            "mdl": b.mdl,
            "train_residual_l2": b.train_residual_l2,
            "train_residual_linf": b.train_residual_linf,
            "validation_residual_l2": b.validation_residual_l2,
            "validation_residual_linf": b.validation_residual_linf,
        }
        path = os.path.join(baselines_dir, f"{b.name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _write_top_k(output_dir: str, search_result: SearchResult) -> None:
    """Write top_k_theories.json."""
    theories = []
    for rank, cand in enumerate(search_result.top_k, 1):
        theories.append({
            "rank": rank,
            "formula": cand.tree.to_infix(),
            "structural_hash": cand.structural_hash,
            "fitness": round(cand.fitness, 8),
            "train_residual_l2": round(cand.train_residual_l2, 8),
            "train_residual_linf": round(cand.train_residual_linf, 8),
            "mdl": cand.mdl,
            "generation_discovered": cand.generation,
        })

    meta = {
        "total_unique_candidates": len(search_result.all_candidates),
        "generations_run": search_result.generations_run,
        "total_evaluations": search_result.total_evaluations,
        "rejections": {
            "non_finite": search_result.rejections.non_finite,
            "tree_too_large": search_result.rejections.tree_too_large,
            "duplicate_hash": search_result.rejections.duplicate_hash,
        },
        "top_k": theories,
    }

    path = os.path.join(output_dir, "top_k_theories.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _write_novelty_report(output_dir: str, report: NoveltyReport) -> None:
    """Write novelty_report.json."""
    evidence = []
    for attempt in report.evidence:
        evidence.append({
            "baseline_name": attempt.baseline_name,
            "transform_type": attempt.transform_type,
            "residual_l2": round(attempt.residual_l2, 8),
            "residual_linf": round(attempt.residual_linf, 8),
            "tolerance": attempt.tolerance,
            "succeeds": attempt.succeeds,
            "fitted_transform_params": attempt.fitted_transform_params,
        })

    data = {
        "candidate_formula": report.candidate_formula,
        "candidate_mdl": report.candidate_mdl,
        "classification": report.classification,
        "closest_baseline": report.closest_baseline,
        "closest_residual": round(report.closest_residual, 8),
        "mdl_gate": {k: (round(v, 8) if isinstance(v, float) else v) for k, v in report.mdl_gate.items()},
        "evidence": evidence,
        "reasoning": report.reasoning,
    }

    path = os.path.join(output_dir, "novelty_report.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_cert_report(
    output_dir: str,
    cert_report: CertificationReport,
    novelty_report: NoveltyReport,
    baselines: list[BaselineResult],
) -> None:
    """Write cert_report.md in human-readable markdown."""
    lines = [
        "# Certification Report",
        "",
        "## Candidate",
        f"```",
        f"sigma = {cert_report.candidate_formula}",
        f"```",
        "",
    ]

    for tier in cert_report.tier_results:
        lines.append(f"## Tier {tier.tier}: {'PASS' if tier.passed else 'FAIL'}")
        lines.append("")
        for check_name, check_passed in tier.checks.items():
            status = "PASS" if check_passed else "FAIL"
            detail = tier.details.get(check_name, "")
            lines.append(f"- [{status}] **{check_name}**: {detail}")
        lines.append("")

    lines.append("## Baselines Tested")
    lines.append("")
    lines.append("| Baseline | Formula | Params | MDL | Train L2 | Val L2 |")
    lines.append("|----------|---------|--------|-----|----------|--------|")
    for b in baselines:
        lines.append(
            f"| {b.name} | `{b.formula[:40]}` | {b.n_parameters} | "
            f"{b.mdl:.0f} | {b.train_residual_l2:.4f} | {b.validation_residual_l2:.4f} |"
        )
    lines.append("")

    lines.append("## Novelty Assessment")
    lines.append("")
    lines.append(f"- **Classification**: {novelty_report.classification}")
    lines.append(f"- **Closest baseline**: {novelty_report.closest_baseline}")
    lines.append(f"- **Reasoning**: {novelty_report.reasoning}")
    lines.append("")

    if novelty_report.mdl_gate.get("applied"):
        lines.append("### MDL Gate")
        for k, v in novelty_report.mdl_gate.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.append("## Workstream Gate Results")
    lines.append("")
    for ws_name, ws_result in cert_report.workstream_results.items():
        if ws_result is None:
            lines.append(f"- **{ws_name}**: not available (import failed)")
            continue
        accepted = getattr(ws_result, "accepted", "unknown")
        lines.append(f"- **{ws_name}**: {'PASS' if accepted else 'FAIL'}")
        for v in getattr(ws_result, "reason_codes", ()):
            lines.append(f"  - violation: {v}")
        for n in getattr(ws_result, "notes", ()):
            lines.append(f"  - note: {n}")
    lines.append("")

    lines.append(f"## Highest Tier Passed: {cert_report.highest_tier_passed}")
    lines.append("")

    path = os.path.join(output_dir, "cert_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_deploy(
    output_dir: str,
    cert_report: CertificationReport,
    novelty_report: NoveltyReport,
) -> None:
    """Write deploy/ directory with constitutive law evaluator."""
    deploy_dir = os.path.join(output_dir, "deploy")
    os.makedirs(deploy_dir, exist_ok=True)

    # Evaluator module
    evaluator = [
        '"""Auto-generated constitutive law evaluator."""',
        "",
        "",
        f"# Discovered law: sigma = {cert_report.candidate_formula}",
        f"# Classification: {novelty_report.classification}",
        f"# Highest tier: {cert_report.highest_tier_passed}",
        "",
        "",
        "def evaluate_stress(eps: float, deps_dt: float) -> float:",
        f'    """Evaluate discovered constitutive law."""',
        f"    # Formula: {cert_report.candidate_formula}",
        "    # WARNING: This is an auto-generated evaluator.",
        "    # Validate against your specific use case before deployment.",
        '    raise NotImplementedError("Manual review required before deployment")',
        "",
        "",
        "def jacobian(eps: float, deps_dt: float) -> tuple[float, float]:",
        '    """Partial derivatives d(sigma)/d(eps) and d(sigma)/d(deps_dt)."""',
        '    raise NotImplementedError("Manual Jacobian derivation required")',
        "",
        "",
        "def monitor(eps: float, deps_dt: float) -> dict[str, float]:",
        '    """Runtime monitor: closure error and extrapolation risk estimates."""',
        "    return {",
        '        "closure_error_estimate": 0.0,',
        '        "extrapolation_risk": 0.0,',
        '        "note": "Manual calibration required",',
        "    }",
        "",
        "",
        "def fallback(eps: float, deps_dt: float) -> float:",
        '    """Fallback to linear elastic baseline if monitor triggers."""',
        "    E_fallback = 100.0  # Approximate linear stiffness",
        "    return E_fallback * eps",
        "",
    ]

    path = os.path.join(deploy_dir, "constitutive_law.py")
    with open(path, "w") as f:
        f.write("\n".join(evaluator) + "\n")

    # Deployment README
    readme = [
        "# Deployment Artifact",
        "",
        f"Discovered law: `sigma = {cert_report.candidate_formula}`",
        "",
        f"Classification: **{novelty_report.classification}**",
        "",
        "## Files",
        "- `constitutive_law.py`: Evaluator, Jacobian, monitor, fallback",
        "",
        "## Before Deployment",
        "1. Review the discovered law against domain knowledge",
        "2. Implement the actual formula in `evaluate_stress()`",
        "3. Derive and implement analytical Jacobians",
        "4. Calibrate the runtime monitor thresholds",
        "5. Validate on held-out experimental data",
        "",
    ]

    path = os.path.join(deploy_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(readme) + "\n")
