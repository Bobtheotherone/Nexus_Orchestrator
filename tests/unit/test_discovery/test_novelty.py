"""Novelty assessment tests: FAIL if equivalence attempts are missing."""

from __future__ import annotations

import pytest

from discovery.data_gen import generate_all_splits, merge_training_splits
from discovery.expr_tree import ExprNode, NodeType


def _make_simple_candidate() -> ExprNode:
    """Create a simple candidate: 50 * eps."""
    return ExprNode(
        node_type=NodeType.MUL,
        left=ExprNode(node_type=NodeType.CONST, value=50.0),
        right=ExprNode(node_type=NodeType.VAR_EPS),
    )


def _get_baselines_and_val():
    from discovery.baselines import fit_all_baselines

    splits = generate_all_splits(seed=42)
    train_eps, train_deps, train_sigma = merge_training_splits(splits)
    val = splits["out_of_regime_coarse"]
    baselines = fit_all_baselines(
        train_eps, train_deps, train_sigma,
        val.eps, val.deps, val.sigma,
    )
    return baselines, val


class TestEquivalenceAttemptCoverage:
    """Every baseline must have equivalence attempts with multiple transforms."""

    def test_all_baselines_have_attempts(self) -> None:
        from discovery.novelty import assess_novelty

        baselines, val = _get_baselines_and_val()
        candidate = _make_simple_candidate()

        report = assess_novelty(
            candidate, baselines, val.eps, val.deps, val.sigma,
        )

        baseline_names_in_evidence = {a.baseline_name for a in report.evidence}
        baseline_names_fitted = {b.name for b in baselines}

        assert baseline_names_fitted.issubset(baseline_names_in_evidence), (
            f"Missing equivalence attempts for baselines: "
            f"{baseline_names_fitted - baseline_names_in_evidence}"
        )

    def test_multiple_transform_types(self) -> None:
        from discovery.novelty import assess_novelty

        baselines, val = _get_baselines_and_val()
        candidate = _make_simple_candidate()

        report = assess_novelty(
            candidate, baselines, val.eps, val.deps, val.sigma,
        )

        transform_types = {a.transform_type for a in report.evidence}
        assert len(transform_types) >= 2, (
            f"Only {transform_types} transform types tested; need at least 2"
        )
        assert "identity" in transform_types
        assert "scaling" in transform_types


class TestKnownEquivalentClassification:
    """A candidate that IS a baseline should be classified as KNOWN or REFORMULATION."""

    def test_hooke_classified_correctly(self) -> None:
        from discovery.baselines import fit_hooke
        from discovery.novelty import assess_novelty

        baselines, val = _get_baselines_and_val()
        splits = generate_all_splits(seed=42)
        train_eps, train_deps, train_sigma = merge_training_splits(splits)

        hooke = fit_hooke(
            train_eps, train_deps, train_sigma,
            val.eps, val.deps, val.sigma,
        )

        candidate = ExprNode(
            node_type=NodeType.MUL,
            left=ExprNode(node_type=NodeType.CONST, value=hooke.parameters["E"]),
            right=ExprNode(node_type=NodeType.VAR_EPS),
        )

        report = assess_novelty(
            candidate, baselines, val.eps, val.deps, val.sigma,
        )

        assert report.classification in ("KNOWN", "REFORMULATION"), (
            f"Hooke-equivalent candidate classified as {report.classification}, "
            f"expected KNOWN or REFORMULATION"
        )


class TestMDLGateRequired:
    """NEW classification requires MDL gate to be checked."""

    def test_mdl_gate_applied(self) -> None:
        from discovery.novelty import assess_novelty

        baselines, val = _get_baselines_and_val()
        # Create a candidate that won't match any baseline
        candidate = ExprNode(
            node_type=NodeType.ADD,
            left=ExprNode(
                node_type=NodeType.MUL,
                left=ExprNode(node_type=NodeType.CONST, value=99.0),
                right=ExprNode(
                    node_type=NodeType.POW,
                    left=ExprNode(node_type=NodeType.VAR_EPS),
                    right=ExprNode(node_type=NodeType.CONST, value=3.0),
                ),
            ),
            right=ExprNode(
                node_type=NodeType.MUL,
                left=ExprNode(node_type=NodeType.CONST, value=7.0),
                right=ExprNode(
                    node_type=NodeType.MUL,
                    left=ExprNode(
                        node_type=NodeType.POW,
                        left=ExprNode(node_type=NodeType.VAR_EPS),
                        right=ExprNode(node_type=NodeType.CONST, value=2.0),
                    ),
                    right=ExprNode(node_type=NodeType.VAR_DEPS),
                ),
            ),
        )

        report = assess_novelty(
            candidate, baselines, val.eps, val.deps, val.sigma,
        )

        assert report.mdl_gate.get("applied") is True, (
            "MDL gate was not applied — required before labeling NEW"
        )
