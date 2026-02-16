"""Expression tree AST for constitutive law discovery.

Each tree represents a candidate constitutive law sigma = f(eps, deps_dt).
Provides canonical forms, equivalence hashing, MDL scoring, evaluation,
and tree manipulation utilities for evolutionary search.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable


class NodeType(Enum):
    CONST = auto()
    VAR_EPS = auto()
    VAR_DEPS = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()


# ---------------------------------------------------------------------------
# ExprNode
# ---------------------------------------------------------------------------

@dataclass
class ExprNode:
    """A node in an expression tree representing sigma = f(eps, deps_dt)."""

    node_type: NodeType
    value: float | None = None
    left: ExprNode | None = None
    right: ExprNode | None = None

    # ── evaluation ────────────────────────────────────────────────────

    def evaluate(self, eps: float, deps: float) -> float:
        """Evaluate at a single (epsilon, d_epsilon/dt) point."""
        nt = self.node_type
        if nt is NodeType.CONST:
            return self.value  # type: ignore[return-value]
        if nt is NodeType.VAR_EPS:
            return eps
        if nt is NodeType.VAR_DEPS:
            return deps
        lv = self.left.evaluate(eps, deps)  # type: ignore[union-attr]
        rv = self.right.evaluate(eps, deps)  # type: ignore[union-attr]
        if nt is NodeType.ADD:
            return lv + rv
        if nt is NodeType.MUL:
            return lv * rv
        if nt is NodeType.POW:
            try:
                result = lv ** rv
                if isinstance(result, complex):
                    return float("nan")
                return result
            except (OverflowError, ValueError, ZeroDivisionError):
                return float("nan")
        raise ValueError(f"unknown node type {nt}")

    def evaluate_batch(
        self, eps_list: list[float], deps_list: list[float],
    ) -> list[float]:
        return [self.evaluate(e, d) for e, d in zip(eps_list, deps_list)]

    # ── tree metrics ──────────────────────────────────────────────────

    def node_count(self) -> int:
        c = 1
        if self.left is not None:
            c += self.left.node_count()
        if self.right is not None:
            c += self.right.node_count()
        return c

    def parameter_count(self) -> int:
        c = 1 if self.node_type is NodeType.CONST else 0
        if self.left is not None:
            c += self.left.parameter_count()
        if self.right is not None:
            c += self.right.parameter_count()
        return c

    def depth(self) -> int:
        if self.left is None and self.right is None:
            return 0
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def mdl(self) -> float:
        return float(self.node_count() + self.parameter_count())

    # ── canonical form ────────────────────────────────────────────────

    def canonical_form(self) -> ExprNode:
        """Return a normalized tree via multi-rule canonicalization.

        Rules applied bottom-up:
        1. Recursive canonicalization of children
        2. Constant folding
        3. Identity / annihilator rules
        4. Distributive normalization (expand MUL over ADD)
        5. Commutativity sort on ADD/MUL by canonical string repr
        6. Like-term collection under ADD
        """
        return _canonicalize(self)

    # ── equivalence hashing ───────────────────────────────────────────

    def equivalence_hash(self) -> str:
        """SHA-256 of structural canonical form (constants replaced by placeholder)."""
        canon = self.canonical_form()
        s = _structural_string(canon)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    # ── display ───────────────────────────────────────────────────────

    def to_infix(self) -> str:
        nt = self.node_type
        if nt is NodeType.CONST:
            v = self.value
            if v is None:
                return "None"
            if not math.isfinite(v):
                return str(v)
            if v == int(v) and abs(v) < 1e12:
                return str(int(v))
            return f"{v:.6g}"
        if nt is NodeType.VAR_EPS:
            return "eps"
        if nt is NodeType.VAR_DEPS:
            return "deps"
        ls = self.left.to_infix() if self.left else "?"  # type: ignore[union-attr]
        rs = self.right.to_infix() if self.right else "?"  # type: ignore[union-attr]
        if nt is NodeType.ADD:
            return f"({ls} + {rs})"
        if nt is NodeType.MUL:
            return f"({ls} * {rs})"
        if nt is NodeType.POW:
            return f"({ls} ** {rs})"
        return f"(?{nt}?)"

    # ── deep copy ─────────────────────────────────────────────────────

    def clone(self) -> ExprNode:
        left = self.left.clone() if self.left else None
        right = self.right.clone() if self.right else None
        return ExprNode(
            node_type=self.node_type,
            value=self.value,
            left=left,
            right=right,
        )

    # ── constant access ───────────────────────────────────────────────

    def collect_constants(self) -> list[ExprNode]:
        """Return mutable references to all CONST nodes (for fitting)."""
        result: list[ExprNode] = []
        if self.node_type is NodeType.CONST:
            result.append(self)
        if self.left is not None:
            result.extend(self.left.collect_constants())
        if self.right is not None:
            result.extend(self.right.collect_constants())
        return result

    def has_valid_powers(self) -> bool:
        """Check that all POW exponents are non-negative integers."""
        if self.node_type is NodeType.POW:
            if self.right is None or self.right.node_type is not NodeType.CONST:
                return False
            v = self.right.value
            if v is None or v < 0 or v != int(v):
                return False
        if self.left is not None and not self.left.has_valid_powers():
            return False
        if self.right is not None and not self.right.has_valid_powers():
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# Canonicalization internals
# ═══════════════════════════════════════════════════════════════════════

def _canonicalize(node: ExprNode) -> ExprNode:
    """Apply all canonical rules bottom-up."""
    nt = node.node_type

    # Leaf nodes are already canonical.
    if nt in (NodeType.CONST, NodeType.VAR_EPS, NodeType.VAR_DEPS):
        return ExprNode(node_type=nt, value=node.value)

    # Recursively canonicalize children first.
    left = _canonicalize(node.left) if node.left else None  # type: ignore[arg-type]
    right = _canonicalize(node.right) if node.right else None  # type: ignore[arg-type]

    # Constant folding: CONST op CONST -> CONST
    if (
        left is not None
        and right is not None
        and left.node_type is NodeType.CONST
        and right.node_type is NodeType.CONST
    ):
        lv, rv = left.value, right.value
        if lv is not None and rv is not None:
            try:
                if nt is NodeType.ADD:
                    return _const(lv + rv)
                if nt is NodeType.MUL:
                    return _const(lv * rv)
                if nt is NodeType.POW:
                    return _const(lv ** rv)
            except (OverflowError, ValueError, ZeroDivisionError):
                pass

    assert left is not None and right is not None

    # Identity / annihilator rules
    simplified = _apply_identity_rules(nt, left, right)
    if simplified is not None:
        return simplified

    # Distributive normalization: a*(b+c) -> (a*b)+(a*c)
    if nt is NodeType.MUL:
        distributed = _distribute(left, right)
        if distributed is not None:
            return _canonicalize(distributed)

    # Commutativity sort: smaller canonical string on the left
    if nt in (NodeType.ADD, NodeType.MUL):
        ls = _structural_string(left)
        rs = _structural_string(right)
        if rs < ls:
            left, right = right, left

    result = ExprNode(node_type=nt, left=left, right=right)

    # Like-term collection under ADD
    if nt is NodeType.ADD:
        collected = _collect_like_terms(result)
        if collected is not None:
            return collected

    return result


def _const(v: float) -> ExprNode:
    return ExprNode(node_type=NodeType.CONST, value=v)


def _apply_identity_rules(
    nt: NodeType, left: ExprNode, right: ExprNode,
) -> ExprNode | None:
    """Apply identity/annihilator simplification rules."""

    def _is_const_val(n: ExprNode, v: float) -> bool:
        return n.node_type is NodeType.CONST and n.value == v

    if nt is NodeType.ADD:
        if _is_const_val(left, 0.0):
            return right
        if _is_const_val(right, 0.0):
            return left

    if nt is NodeType.MUL:
        if _is_const_val(left, 0.0) or _is_const_val(right, 0.0):
            return _const(0.0)
        if _is_const_val(left, 1.0):
            return right
        if _is_const_val(right, 1.0):
            return left

    if nt is NodeType.POW:
        if _is_const_val(right, 0.0):
            return _const(1.0)
        if _is_const_val(right, 1.0):
            return left

    return None


def _distribute(left: ExprNode, right: ExprNode) -> ExprNode | None:
    """If either child is ADD, distribute: a*(b+c) -> (a*b)+(a*c)."""
    if left.node_type is NodeType.ADD and left.left and left.right:
        return ExprNode(
            node_type=NodeType.ADD,
            left=ExprNode(node_type=NodeType.MUL, left=left.left.clone(), right=right.clone()),
            right=ExprNode(node_type=NodeType.MUL, left=left.right.clone(), right=right.clone()),
        )
    if right.node_type is NodeType.ADD and right.left and right.right:
        return ExprNode(
            node_type=NodeType.ADD,
            left=ExprNode(node_type=NodeType.MUL, left=left.clone(), right=right.left.clone()),
            right=ExprNode(node_type=NodeType.MUL, left=left.clone(), right=right.right.clone()),
        )
    return None


def _collect_like_terms(node: ExprNode) -> ExprNode | None:
    """Flatten ADD tree, group by structural variable part, merge coefficients."""
    terms = _flatten_add(node)
    if len(terms) <= 1:
        return None

    # Decompose each term into (coefficient, variable_structure_string)
    grouped: dict[str, float] = {}
    for term in terms:
        coeff, var_part = _split_coeff_and_vars(term)
        key = _structural_string(var_part) if var_part is not None else "__const__"
        grouped[key] = grouped.get(key, 0.0) + coeff

    if len(grouped) >= len(terms):
        return None  # Nothing was merged

    # Rebuild tree from grouped terms
    rebuilt_terms: list[ExprNode] = []
    for key in sorted(grouped.keys()):
        coeff = grouped[key]
        if coeff == 0.0:
            continue
        if key == "__const__":
            rebuilt_terms.append(_const(coeff))
        else:
            # Reconstruct: coeff * var_part
            # We need the original var_part structure. Find any term with this key.
            var_part = None
            for term in terms:
                _, vp = _split_coeff_and_vars(term)
                if vp is not None and _structural_string(vp) == key:
                    var_part = vp.clone()
                    break
            if var_part is None:
                continue
            if coeff == 1.0:
                rebuilt_terms.append(var_part)
            else:
                rebuilt_terms.append(ExprNode(
                    node_type=NodeType.MUL,
                    left=_const(coeff),
                    right=var_part,
                ))

    if not rebuilt_terms:
        return _const(0.0)

    result = rebuilt_terms[0]
    for t in rebuilt_terms[1:]:
        result = ExprNode(node_type=NodeType.ADD, left=result, right=t)
    return result


def _flatten_add(node: ExprNode) -> list[ExprNode]:
    """Flatten a chain of ADD into a list of terms."""
    if node.node_type is not NodeType.ADD:
        return [node]
    terms: list[ExprNode] = []
    if node.left:
        terms.extend(_flatten_add(node.left))
    if node.right:
        terms.extend(_flatten_add(node.right))
    return terms


def _split_coeff_and_vars(
    term: ExprNode,
) -> tuple[float, ExprNode | None]:
    """Split a term into (numeric_coefficient, variable_structure).

    E.g. ``3 * (eps * deps)`` -> (3.0, eps*deps)
         ``eps`` -> (1.0, eps)
         ``5.0`` -> (5.0, None)
    """
    if term.node_type is NodeType.CONST:
        return (term.value if term.value is not None else 0.0, None)

    if term.node_type is NodeType.MUL and term.left and term.right:
        if term.left.node_type is NodeType.CONST:
            return (term.left.value or 0.0, term.right)
        if term.right.node_type is NodeType.CONST:
            return (term.right.value or 0.0, term.left)

    return (1.0, term)


def _structural_string(node: ExprNode) -> str:
    """Deterministic string ignoring constant values (for hashing)."""
    nt = node.node_type
    if nt is NodeType.CONST:
        return "C"
    if nt is NodeType.VAR_EPS:
        return "E"
    if nt is NodeType.VAR_DEPS:
        return "D"
    ls = _structural_string(node.left) if node.left else ""
    rs = _structural_string(node.right) if node.right else ""
    if nt is NodeType.ADD:
        return f"(+{ls}{rs})"
    if nt is NodeType.MUL:
        return f"(*{ls}{rs})"
    if nt is NodeType.POW:
        # For POW, include the exponent value since it's structural
        exp_val = ""
        if node.right and node.right.node_type is NodeType.CONST and node.right.value is not None:
            exp_val = str(int(node.right.value))
        return f"(^{ls}{exp_val})"
    return "?"


# ═══════════════════════════════════════════════════════════════════════
# Tree generation and manipulation helpers
# ═══════════════════════════════════════════════════════════════════════

_BINARY_OPS = (NodeType.ADD, NodeType.MUL)
_TERMINALS = (NodeType.CONST, NodeType.VAR_EPS, NodeType.VAR_DEPS)


def random_terminal(rng: random.Random) -> ExprNode:
    """Generate a random leaf node."""
    choice = rng.choice(_TERMINALS)
    if choice is NodeType.CONST:
        return ExprNode(node_type=NodeType.CONST, value=rng.uniform(-10.0, 10.0))
    return ExprNode(node_type=choice)


def random_tree(max_depth: int = 4, rng: random.Random | None = None) -> ExprNode:
    """Generate a random expression tree."""
    if rng is None:
        rng = random.Random()
    return _random_tree_impl(max_depth, rng)


def _random_tree_impl(max_depth: int, rng: random.Random) -> ExprNode:
    if max_depth <= 0 or rng.random() < 0.3:
        return random_terminal(rng)

    # Occasionally use POW
    if rng.random() < 0.15:
        base = _random_tree_impl(max_depth - 1, rng)
        exponent = ExprNode(node_type=NodeType.CONST, value=float(rng.choice([2, 3])))
        return ExprNode(node_type=NodeType.POW, left=base, right=exponent)

    op = rng.choice(_BINARY_OPS)
    left = _random_tree_impl(max_depth - 1, rng)
    right = _random_tree_impl(max_depth - 1, rng)
    return ExprNode(node_type=op, left=left, right=right)


def tree_nodes_preorder(root: ExprNode) -> list[ExprNode]:
    """Return all nodes in pre-order traversal."""
    result: list[ExprNode] = [root]
    if root.left is not None:
        result.extend(tree_nodes_preorder(root.left))
    if root.right is not None:
        result.extend(tree_nodes_preorder(root.right))
    return result


def tree_size(root: ExprNode) -> int:
    return root.node_count()


def subtree_at(root: ExprNode, index: int) -> ExprNode:
    """Return the subtree at pre-order index."""
    nodes = tree_nodes_preorder(root)
    return nodes[index % len(nodes)]


def replace_subtree(root: ExprNode, index: int, replacement: ExprNode) -> ExprNode:
    """Return a new tree with the subtree at pre-order index replaced."""
    counter = [0]

    def _replace(node: ExprNode) -> ExprNode:
        if counter[0] == index:
            counter[0] += node.node_count()
            return replacement.clone()
        counter[0] += 1
        left = _replace(node.left) if node.left else None
        right = _replace(node.right) if node.right else None
        return ExprNode(node_type=node.node_type, value=node.value, left=left, right=right)

    return _replace(root)
