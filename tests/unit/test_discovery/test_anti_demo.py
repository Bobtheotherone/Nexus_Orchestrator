"""Anti-demo tests: FAIL if hardcoded candidate theory is used."""

from __future__ import annotations

from pathlib import Path

import pytest


_DISCOVERY_DIR = Path(__file__).parent.parent.parent.parent / "src" / "discovery"
_SRC_DIR = Path(__file__).parent.parent.parent.parent / "src"


class TestNoHardcodedEquations:
    """Ensure the discovery engine does not contain the old demo's hardcoded equation."""

    FORBIDDEN_STRINGS = [
        "sigma(x,t) = int_Omega",
        "K(|x-y|) C(eps(y,t))",
        "Nonlocal Viscoelastic Constitutive Law",
        "int_Omega K",
    ]

    def test_no_forbidden_strings_in_discovery_package(self) -> None:
        for py_file in _DISCOVERY_DIR.glob("**/*.py"):
            source = py_file.read_text()
            for forbidden in self.FORBIDDEN_STRINGS:
                assert forbidden not in source, (
                    f"Forbidden hardcoded string '{forbidden}' found in {py_file}"
                )

    def test_old_demo_script_removed(self) -> None:
        old_demo = _SRC_DIR / "run_discovery_engine.py"
        assert not old_demo.exists(), (
            "Old demo script src/run_discovery_engine.py still exists — must be deleted"
        )


class TestSearchProducesRealCandidates:
    """Verify the search engine generates candidates from evolution, not hardcoding."""

    def test_search_produces_diverse_candidates(self) -> None:
        from discovery.data_gen import generate_all_splits, merge_training_splits
        from discovery.search import EvolutionarySearch, SearchConfig

        splits = generate_all_splits(seed=99)
        train_eps, train_deps, train_sigma = merge_training_splits(
            splits, regimes=("low_strain",), sampling="coarse",
        )

        config = SearchConfig(
            population_size=20,
            max_generations=5,
            min_unique_candidates=10,
            seed=99,
        )
        search = EvolutionarySearch(config, train_eps, train_deps, train_sigma)
        result = search.run()

        assert len(result.all_candidates) >= 10, (
            f"Search produced only {len(result.all_candidates)} unique candidates, need >= 10"
        )
        hashes = set(result.all_candidates.keys())
        assert len(hashes) >= 10, "Candidates do not have diverse structural hashes"

    def test_ground_truth_not_leaked_to_search(self) -> None:
        """No file in src/discovery/ other than data_gen.py should reference ground truth params."""
        for py_file in _DISCOVERY_DIR.glob("**/*.py"):
            if py_file.name == "data_gen.py":
                continue
            source = py_file.read_text()
            assert "_ground_truth_stress" not in source, (
                f"Ground truth function referenced in {py_file} — search must not see it"
            )
            assert "_E1" not in source and "_E3" not in source, (
                f"Ground truth constants referenced in {py_file}"
            )
