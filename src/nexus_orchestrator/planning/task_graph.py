"""Deterministic adjacency-list task graph utilities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Set
from heapq import heapify, heappop, heappush
from typing import cast


class CycleError(ValueError):
    """Raised when a cycle is detected in the task graph."""

    cycles: tuple[tuple[str, ...], ...]

    def __init__(self, cycles: Iterable[Sequence[str]]) -> None:
        normalized: tuple[tuple[str, ...], ...] = tuple(tuple(path) for path in cycles)
        self.cycles = normalized

        if not normalized:
            message = "Task graph contains at least one cycle."
        else:
            preview = ", ".join(" -> ".join(path) for path in normalized[:3])
            suffix = "..." if len(normalized) > 3 else ""
            message = f"Task graph contains cycle(s): {preview}{suffix}"
        super().__init__(message)


class TaskGraph:
    """Directed graph with deterministic traversal and serialization."""

    __slots__ = ("_nodes", "_children", "_parents")

    def __init__(
        self,
        nodes: Iterable[str] | None = None,
        edges: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        self._nodes: set[str] = set()
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}

        if nodes is not None:
            for node_id in nodes:
                self.add_node(node_id)

        if edges is not None:
            for parent, child in edges:
                self.add_edge(parent, child)

    @property
    def nodes(self) -> tuple[str, ...]:
        """All node IDs in deterministic order."""
        return tuple(sorted(self._nodes))

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """All edges as ``(parent, child)`` pairs in deterministic order."""
        ordered_edges: list[tuple[str, str]] = []
        for parent in sorted(self._nodes):
            for child in sorted(self._children[parent]):
                ordered_edges.append((parent, child))
        return tuple(ordered_edges)

    def add_node(self, node_id: str) -> None:
        """Add a node if it does not already exist."""
        self._validate_node_id(node_id)
        if node_id in self._nodes:
            return

        self._nodes.add(node_id)
        self._children[node_id] = set()
        self._parents[node_id] = set()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all inbound/outbound edges."""
        self._assert_node_exists(node_id)

        for parent in tuple(self._parents[node_id]):
            self._children[parent].remove(node_id)
        for child in tuple(self._children[node_id]):
            self._parents[child].remove(node_id)

        del self._children[node_id]
        del self._parents[node_id]
        self._nodes.remove(node_id)

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge ``parent -> child``."""
        self._validate_node_id(parent)
        self._validate_node_id(child)

        if parent not in self._nodes:
            self.add_node(parent)
        if child not in self._nodes:
            self.add_node(child)

        if child in self._children[parent]:
            return

        self._children[parent].add(child)
        self._parents[child].add(parent)

    def topological_sort(self) -> tuple[str, ...]:
        """Return a deterministic topological ordering or raise ``CycleError``."""
        indegree: dict[str, int] = {node: len(self._parents[node]) for node in self._nodes}
        ready: list[str] = [node for node, degree in indegree.items() if degree == 0]
        heapify(ready)

        order: list[str] = []
        while ready:
            node = heappop(ready)
            order.append(node)

            for child in sorted(self._children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    heappush(ready, child)

        if len(order) != len(self._nodes):
            raise CycleError(self.detect_cycles())

        return tuple(order)

    def detect_cycles(self) -> tuple[tuple[str, ...], ...]:
        """
        Detect directed cycles.

        Returns cycle paths as closed paths, e.g. ``("A", "B", "C", "A")``.
        """
        state: dict[str, int] = {}
        stack: list[str] = []
        stack_index: dict[str, int] = {}
        cycles: dict[tuple[str, ...], None] = {}

        for start in sorted(self._nodes):
            if state.get(start, 0) != 0:
                continue

            state[start] = 1
            stack.append(start)
            stack_index[start] = len(stack) - 1
            frames: list[tuple[str, Iterator[str]]] = [(start, iter(sorted(self._children[start])))]

            while frames:
                node, child_iter = frames[-1]

                try:
                    child = next(child_iter)
                except StopIteration:
                    frames.pop()
                    state[node] = 2
                    stack.pop()
                    del stack_index[node]
                    continue

                child_state = state.get(child, 0)
                if child_state == 0:
                    state[child] = 1
                    stack_index[child] = len(stack)
                    stack.append(child)
                    frames.append((child, iter(sorted(self._children[child]))))
                    continue

                if child_state == 1:
                    start_index = stack_index[child]
                    cycle = tuple(stack[start_index:] + [child])
                    cycles[self._canonicalize_cycle(cycle)] = None

        return tuple(sorted(cycles))

    def get_dependencies(self, node_id: str, *, transitive: bool = False) -> tuple[str, ...]:
        """Return direct or transitive dependencies for ``node_id``."""
        self._assert_node_exists(node_id)
        if not transitive:
            return tuple(sorted(self._parents[node_id]))
        return self.get_transitive_dependencies(node_id)

    def get_dependents(self, node_id: str, *, transitive: bool = False) -> tuple[str, ...]:
        """Return direct or transitive dependents for ``node_id``."""
        self._assert_node_exists(node_id)
        if not transitive:
            return tuple(sorted(self._children[node_id]))
        return self.get_transitive_dependents(node_id)

    def get_transitive_dependencies(self, node_id: str) -> tuple[str, ...]:
        """Return all ancestors of ``node_id``."""
        return self._transitive_closure(node_id, upstream=True)

    def get_transitive_dependents(self, node_id: str) -> tuple[str, ...]:
        """Return all descendants of ``node_id``."""
        return self._transitive_closure(node_id, upstream=False)

    def get_runnable(self, completed: Set[str]) -> tuple[str, ...]:
        """
        Return nodes ready to run.

        A node is runnable when it is not already completed and all dependencies
        are present in ``completed``.
        """
        completed_nodes = set(completed)
        runnable: list[str] = []
        for node in sorted(self._nodes):
            if node in completed_nodes:
                continue
            if self._parents[node].issubset(completed_nodes):
                runnable.append(node)
        return tuple(runnable)

    def critical_path(self, weights: Mapping[str, float] | None = None) -> tuple[str, ...]:
        """
        Compute the longest path in the DAG.

        Weights are node weights; unspecified nodes default to ``1.0``.
        """
        ordered = self.topological_sort()
        if not ordered:
            return ()

        distances: dict[str, float] = {}
        predecessors: dict[str, str | None] = {}
        for node in ordered:
            distances[node] = self._weight_for(node, weights)
            predecessors[node] = None

        for parent in ordered:
            parent_distance = distances[parent]
            for child in sorted(self._children[parent]):
                candidate = parent_distance + self._weight_for(child, weights)
                current = distances[child]
                if candidate > current:
                    distances[child] = candidate
                    predecessors[child] = parent
                elif candidate == current:
                    existing_parent = predecessors[child]
                    if existing_parent is None or parent < existing_parent:
                        predecessors[child] = parent

        end_node = ordered[0]
        end_distance = distances[end_node]
        for node in ordered[1:]:
            candidate_distance = distances[node]
            if candidate_distance > end_distance:
                end_node = node
                end_distance = candidate_distance
            elif candidate_distance == end_distance and node < end_node:
                end_node = node

        path: list[str] = []
        cursor: str | None = end_node
        while cursor is not None:
            path.append(cursor)
            cursor = predecessors[cursor]
        path.reverse()
        return tuple(path)

    def serialize(self) -> dict[str, object]:
        """Serialize graph to a stable JSON-friendly mapping."""
        nodes = sorted(self._nodes)
        edges: list[list[str]] = []
        for parent in nodes:
            for child in sorted(self._children[parent]):
                edges.append([parent, child])

        return {
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def deserialize(cls, payload: Mapping[str, object]) -> TaskGraph:
        """Deserialize from :meth:`serialize` output."""
        nodes = cls._parse_nodes(payload.get("nodes", ()))
        edges = cls._parse_edges(payload.get("edges", ()))

        graph = cls(nodes=nodes)
        for parent, child in edges:
            graph.add_edge(parent, child)
        return graph

    def _transitive_closure(self, node_id: str, *, upstream: bool) -> tuple[str, ...]:
        self._assert_node_exists(node_id)

        adjacency = self._parents if upstream else self._children
        visited: set[str] = set()
        pending: list[str] = list(adjacency[node_id])

        while pending:
            node = pending.pop()
            if node in visited:
                continue

            visited.add(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    pending.append(neighbor)

        return tuple(sorted(visited))

    @staticmethod
    def _weight_for(node_id: str, weights: Mapping[str, float] | None) -> float:
        raw_value: float | int = 1.0
        if weights is not None and node_id in weights:
            raw_value = weights[node_id]

        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"Weight for node '{node_id}' must be numeric.")
        return float(raw_value)

    @staticmethod
    def _canonicalize_cycle(cycle: Sequence[str]) -> tuple[str, ...]:
        if len(cycle) < 2:
            raise ValueError("Cycle path must contain at least two nodes.")

        core = tuple(cycle[:-1])
        if len(core) == 1:
            return (core[0], core[0])

        best = core
        for offset in range(1, len(core)):
            rotated = core[offset:] + core[:offset]
            if rotated < best:
                best = rotated

        return best + (best[0],)

    @staticmethod
    def _parse_nodes(raw_nodes: object) -> tuple[str, ...]:
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes, bytearray)):
            raise TypeError("'nodes' must be a sequence of strings.")

        nodes: list[str] = []
        seen: set[str] = set()
        for index, raw_node in enumerate(raw_nodes):
            if not isinstance(raw_node, str):
                raise TypeError(f"'nodes[{index}]' must be a string.")
            TaskGraph._validate_node_id(raw_node)
            if raw_node in seen:
                raise ValueError(f"Duplicate node '{raw_node}' in 'nodes'.")
            seen.add(raw_node)
            nodes.append(raw_node)
        return tuple(nodes)

    @staticmethod
    def _parse_edges(raw_edges: object) -> tuple[tuple[str, str], ...]:
        if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes, bytearray)):
            raise TypeError("'edges' must be a sequence of [parent, child] pairs.")

        edges: list[tuple[str, str]] = []
        for index, raw_edge in enumerate(raw_edges):
            if not isinstance(raw_edge, Sequence) or isinstance(raw_edge, (str, bytes, bytearray)):
                raise TypeError(f"'edges[{index}]' must be a sequence of two strings.")
            pair = cast("Sequence[object]", raw_edge)
            if len(pair) != 2:
                raise ValueError(f"'edges[{index}]' must contain exactly two node IDs.")

            parent_raw = pair[0]
            child_raw = pair[1]
            if not isinstance(parent_raw, str) or not isinstance(child_raw, str):
                raise TypeError(f"'edges[{index}]' must contain only strings.")

            TaskGraph._validate_node_id(parent_raw)
            TaskGraph._validate_node_id(child_raw)
            edges.append((parent_raw, child_raw))

        return tuple(edges)

    @staticmethod
    def _validate_node_id(node_id: str) -> None:
        if not node_id:
            raise ValueError("Node ID must be non-empty.")

    def _assert_node_exists(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node: {node_id}")


__all__ = ["CycleError", "TaskGraph"]
