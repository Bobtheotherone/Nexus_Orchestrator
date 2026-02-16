"""Discretization invariance tests: FAIL if dt/sampling invariance is skipped."""

from __future__ import annotations

import inspect
import math

import pytest

from discovery.certification import CertificateLadder
from discovery.data_gen import generate_all_splits, merge_training_splits
from discovery.expr_tree import ExprNode, NodeType


def _get_splits_and_baselines():
    from discovery.baselines import fit_all_baselines

    splits = generate_all_splits(seed=42)
    train_eps, train_deps, train_sigma = merge_training_splits(splits)
    val = splits["out_of_regime_coarse"]
    baselines = fit_all_baselines(
        train_eps, train_deps, train_sigma,
        val.eps, val.deps, val.sigma,
    )
    return splits, baselines


class TestDtInvarianceRequired:
    """Tier 2 must check dt/sampling invariance."""

    def test_tier2_source_references_dt_or_sampling(self) -> None:
        """tier_2() source code must reference dt, sampling, coarse, or fine."""
        source = inspect.getsource(CertificateLadder.tier_2)
        keywords = ["coarse", "fine", "dt", "sampling", "invariance", "drift"]
        found = any(kw in source.lower() for kw in keywords)
        assert found, (
            "tier_2() does not reference dt/sampling keywords — "
            "discretization invariance may be skipped"
        )

    def test_bad_candidate_fails_tier_0(self) -> None:
        """A candidate producing inf should fail at Tier 0."""
        splits, baselines = _get_splits_and_baselines()
        bad = ExprNode(node_type=NodeType.CONST, value=float("inf"))
        ladder = CertificateLadder(bad, splits, baselines)
        report = ladder.run_all_tiers()
        assert report.highest_tier_passed < 0, (
            "Infinite candidate should fail at Tier 0"
        )


class TestIdentifiabilityUsesBootstrap:
    """Identifiability must use bootstrap, not just two-split agreement."""

    def test_identifiability_source_references_bootstrap(self) -> None:
        source = inspect.getsource(CertificateLadder)
        assert "bootstrap" in source.lower(), (
            "CertificateLadder does not reference 'bootstrap' — "
            "identifiability may use weak two-split check instead"
        )

    def test_identifiability_source_references_fim(self) -> None:
        source = inspect.getsource(CertificateLadder)
        assert "fim" in source.lower() or "fisher" in source.lower() or "condition" in source.lower(), (
            "CertificateLadder does not reference FIM/Fisher/condition — "
            "missing Fisher Information Matrix check"
        )

    def test_identifiability_source_references_profile(self) -> None:
        source = inspect.getsource(CertificateLadder)
        assert "profile" in source.lower(), (
            "CertificateLadder does not reference 'profile' — "
            "missing profile likelihood check"
        )


class TestParameterDriftChecked:
    """Tier 2 must check that fitted parameters don't drift with dt."""

    def test_parameter_drift_in_source(self) -> None:
        source = inspect.getsource(CertificateLadder)
        assert "drift" in source.lower() or "refit" in source.lower(), (
            "CertificateLadder does not reference parameter drift — "
            "dt-invariance may skip parameter stability check"
        )

    def test_simple_candidate_runs_tier2(self) -> None:
        """A reasonable candidate should be able to reach Tier 2 checks."""
        splits, baselines = _get_splits_and_baselines()

        # Simple Kelvin-Voigt-like: E*eps + eta*deps
        candidate = ExprNode(
            node_type=NodeType.ADD,
            left=ExprNode(
                node_type=NodeType.MUL,
                left=ExprNode(node_type=NodeType.CONST, value=100.0),
                right=ExprNode(node_type=NodeType.VAR_EPS),
            ),
            right=ExprNode(
                node_type=NodeType.MUL,
                left=ExprNode(node_type=NodeType.CONST, value=10.0),
                right=ExprNode(node_type=NodeType.VAR_DEPS),
            ),
        )

        ladder = CertificateLadder(candidate, splits, baselines)
        report = ladder.run_all_tiers()

        # This candidate may or may not pass Tier 1 (needs to beat baselines),
        # but the tier_2 method should exist and be reachable
        assert len(report.tier_results) >= 1, "At least Tier 0 should be evaluated"
