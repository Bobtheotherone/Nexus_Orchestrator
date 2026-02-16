"""Evolutionary program synthesis engine for constitutive law discovery.

Implements a steady-state evolutionary algorithm over expression trees
with tournament selection, subtree crossover/mutation, constant
re-optimization, and equivalence-based deduplication.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from discovery.expr_tree import (
    ExprNode,
    NodeType,
    random_terminal,
    random_tree,
    replace_subtree,
    subtree_at,
    tree_nodes_preorder,
    tree_size,
)


@dataclass
class SearchConfig:
    """Configuration for evolutionary search."""

    population_size: int = 100
    max_generations: int = 50
    tournament_size: int = 5
    elitism_count: int = 5
    max_tree_depth: int = 6
    max_tree_size: int = 30
    crossover_rate: float = 0.7
    mutation_rate: float = 0.3
    min_unique_candidates: int = 500
    mdl_penalty: float = 0.01
    constant_reopt_iters: int = 20
    seed: int = 42


@dataclass
class CandidateRecord:
    """A candidate with its fitness metrics."""

    tree: ExprNode
    structural_hash: str
    train_residual_l2: float
    train_residual_linf: float
    mdl: float
    fitness: float
    generation: int


@dataclass
class RejectionCounts:
    """Track reasons for candidate rejection."""

    non_finite: int = 0
    tree_too_large: int = 0
    duplicate_hash: int = 0
    total_evaluated: int = 0


@dataclass
class SearchResult:
    """Results of evolutionary search."""

    all_candidates: dict[str, CandidateRecord]
    top_k: list[CandidateRecord]
    generations_run: int
    total_evaluations: int
    rejections: RejectionCounts


class EvolutionarySearch:
    """Steady-state evolutionary search over expression trees."""

    def __init__(
        self,
        config: SearchConfig,
        eps_train: list[float],
        deps_train: list[float],
        sigma_train: list[float],
    ) -> None:
        self._config = config
        self._eps = eps_train
        self._deps = deps_train
        self._sigma = sigma_train
        self._rng = random.Random(config.seed)
        self._population: list[CandidateRecord] = []
        self._seen_hashes: dict[str, CandidateRecord] = {}
        self._generation = 0
        self._total_evals = 0
        self._rejections = RejectionCounts()

    def run(self) -> SearchResult:
        """Run the full evolutionary search."""
        self._initialize_population()

        while (
            len(self._seen_hashes) < self._config.min_unique_candidates
            and self._generation < self._config.max_generations
        ):
            self._generation += 1
            self._evolve_one_generation()

        top_k = sorted(self._seen_hashes.values(), key=lambda c: c.fitness)[:10]
        return SearchResult(
            all_candidates=dict(self._seen_hashes),
            top_k=top_k,
            generations_run=self._generation,
            total_evaluations=self._total_evals,
            rejections=self._rejections,
        )

    # ── population init ───────────────────────────────────────────────

    def _initialize_population(self) -> None:
        attempts = 0
        max_attempts = self._config.population_size * 10
        while len(self._population) < self._config.population_size and attempts < max_attempts:
            attempts += 1
            tree = random_tree(max_depth=4, rng=self._rng)
            tree = self._reoptimize_constants(tree)
            record = self._evaluate(tree)
            if record is None:
                continue
            if record.structural_hash not in self._seen_hashes:
                self._seen_hashes[record.structural_hash] = record
                self._population.append(record)
            elif len(self._population) < self._config.population_size:
                # Allow structurally duplicate trees with different constants
                self._population.append(record)

    # ── evolution ─────────────────────────────────────────────────────

    def _evolve_one_generation(self) -> None:
        new_children: list[CandidateRecord] = []
        target = self._config.population_size - self._config.elitism_count

        for _ in range(target):
            if self._rng.random() < self._config.crossover_rate:
                parent_a = self._tournament_select()
                parent_b = self._tournament_select()
                child_tree = self._crossover(parent_a.tree, parent_b.tree)
            else:
                parent = self._tournament_select()
                child_tree = self._mutate(parent.tree.clone())

            # Size guard
            if tree_size(child_tree) > self._config.max_tree_size:
                child_tree = self._prune_to_size(child_tree)
                self._rejections.tree_too_large += 1

            child_tree = self._reoptimize_constants(child_tree)
            record = self._evaluate(child_tree)
            if record is None:
                continue

            # Track for dedup counting
            if record.structural_hash not in self._seen_hashes:
                self._seen_hashes[record.structural_hash] = record
            else:
                self._rejections.duplicate_hash += 1

            new_children.append(record)

        # Sort: keep elite + best children
        all_candidates = sorted(
            self._population + new_children, key=lambda c: c.fitness,
        )
        self._population = all_candidates[: self._config.population_size]

    # ── evaluation ────────────────────────────────────────────────────

    def _evaluate(self, tree: ExprNode) -> CandidateRecord | None:
        """Evaluate a tree.  Returns None if output is non-finite."""
        self._total_evals += 1
        self._rejections.total_evaluated += 1

        try:
            predictions = tree.evaluate_batch(self._eps, self._deps)
        except (OverflowError, ValueError, ZeroDivisionError):
            self._rejections.non_finite += 1
            return None

        # Check finiteness
        for p in predictions:
            if isinstance(p, complex) or not math.isfinite(p):
                self._rejections.non_finite += 1
                return None

        n = len(self._sigma)
        ss = 0.0
        mx = 0.0
        for p, s in zip(predictions, self._sigma):
            diff = abs(p - s)
            ss += diff * diff
            if diff > mx:
                mx = diff
        l2 = math.sqrt(ss / n) if n > 0 else 0.0

        mdl = tree.mdl()
        fitness = l2 + self._config.mdl_penalty * mdl
        h = tree.equivalence_hash()

        return CandidateRecord(
            tree=tree,
            structural_hash=h,
            train_residual_l2=l2,
            train_residual_linf=mx,
            mdl=mdl,
            fitness=fitness,
            generation=self._generation,
        )

    # ── selection ─────────────────────────────────────────────────────

    def _tournament_select(self) -> CandidateRecord:
        contestants = self._rng.sample(
            self._population,
            min(self._config.tournament_size, len(self._population)),
        )
        return min(contestants, key=lambda c: c.fitness)

    # ── crossover ─────────────────────────────────────────────────────

    def _crossover(self, parent_a: ExprNode, parent_b: ExprNode) -> ExprNode:
        """Subtree swap crossover."""
        a = parent_a.clone()
        b = parent_b.clone()

        a_nodes = tree_nodes_preorder(a)
        b_nodes = tree_nodes_preorder(b)

        a_idx = self._rng.randrange(len(a_nodes))
        b_idx = self._rng.randrange(len(b_nodes))

        donor = subtree_at(b, b_idx).clone()
        return replace_subtree(a, a_idx, donor)

    # ── mutation ──────────────────────────────────────────────────────

    def _mutate(self, tree: ExprNode) -> ExprNode:
        mutation_type = self._rng.choice([
            "constant_perturbation",
            "operator_swap",
            "subtree_grow",
            "subtree_prune",
            "subtree_replacement",
        ])

        if mutation_type == "constant_perturbation":
            return self._mutate_constant(tree)
        if mutation_type == "operator_swap":
            return self._mutate_operator(tree)
        if mutation_type == "subtree_grow":
            return self._mutate_grow(tree)
        if mutation_type == "subtree_prune":
            return self._mutate_prune(tree)
        return self._mutate_replace_subtree(tree)

    def _mutate_constant(self, tree: ExprNode) -> ExprNode:
        consts = tree.collect_constants()
        if not consts:
            return tree
        c = self._rng.choice(consts)
        c.value = (c.value or 0.0) + self._rng.gauss(0.0, 5.0)
        return tree

    def _mutate_operator(self, tree: ExprNode) -> ExprNode:
        nodes = [n for n in tree_nodes_preorder(tree) if n.node_type in (NodeType.ADD, NodeType.MUL)]
        if not nodes:
            return tree
        n = self._rng.choice(nodes)
        n.node_type = NodeType.MUL if n.node_type is NodeType.ADD else NodeType.ADD
        return tree

    def _mutate_grow(self, tree: ExprNode) -> ExprNode:
        nodes = tree_nodes_preorder(tree)
        leaves = [i for i, n in enumerate(nodes) if n.left is None and n.right is None]
        if not leaves:
            return tree
        idx = self._rng.choice(leaves)
        new_subtree = random_tree(max_depth=2, rng=self._rng)
        return replace_subtree(tree, idx, new_subtree)

    def _mutate_prune(self, tree: ExprNode) -> ExprNode:
        nodes = tree_nodes_preorder(tree)
        internals = [i for i, n in enumerate(nodes) if n.left is not None]
        if not internals:
            return tree
        idx = self._rng.choice(internals)
        terminal = random_terminal(self._rng)
        return replace_subtree(tree, idx, terminal)

    def _mutate_replace_subtree(self, tree: ExprNode) -> ExprNode:
        nodes = tree_nodes_preorder(tree)
        idx = self._rng.randrange(len(nodes))
        new_sub = random_tree(max_depth=3, rng=self._rng)
        return replace_subtree(tree, idx, new_sub)

    # ── constant re-optimization ──────────────────────────────────────

    def _reoptimize_constants(self, tree: ExprNode) -> ExprNode:
        """Coordinate descent on constants to improve fit."""
        consts = tree.collect_constants()
        if not consts:
            return tree

        best_res = self._compute_residual_l2(tree)
        if not math.isfinite(best_res):
            return tree

        deltas = [20.0, -20.0, 5.0, -5.0, 1.0, -1.0, 0.1, -0.1, 0.01, -0.01]
        for _ in range(self._config.constant_reopt_iters):
            improved = False
            for const_node in consts:
                original = const_node.value or 0.0
                for delta in deltas:
                    const_node.value = original + delta
                    r = self._compute_residual_l2(tree)
                    if math.isfinite(r) and r < best_res:
                        best_res = r
                        original = const_node.value
                        improved = True
                    else:
                        const_node.value = original
            if not improved:
                break

        return tree

    def _compute_residual_l2(self, tree: ExprNode) -> float:
        try:
            preds = tree.evaluate_batch(self._eps, self._deps)
        except (OverflowError, ValueError, ZeroDivisionError):
            return float("inf")
        n = len(self._sigma)
        ss = 0.0
        for p, s in zip(preds, self._sigma):
            if isinstance(p, complex) or not math.isfinite(p):
                return float("inf")
            try:
                ss += (p - s) ** 2
            except OverflowError:
                return float("inf")
        return math.sqrt(ss / n) if n > 0 else 0.0

    # ── utilities ─────────────────────────────────────────────────────

    def _prune_to_size(self, tree: ExprNode) -> ExprNode:
        """Prune tree to max size by replacing deepest subtrees with terminals."""
        while tree_size(tree) > self._config.max_tree_size:
            nodes = tree_nodes_preorder(tree)
            # Find deepest internal node
            max_depth = -1
            deepest_idx = 0
            for i, n in enumerate(nodes):
                d = n.depth()
                if d > max_depth and n.left is not None:
                    max_depth = d
                    deepest_idx = i
            tree = replace_subtree(tree, deepest_idx, random_terminal(self._rng))
        return tree
