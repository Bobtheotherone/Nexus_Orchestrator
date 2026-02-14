"""Unit tests for planning.task_graph."""

from __future__ import annotations

import random

import pytest

from nexus_orchestrator.planning.task_graph import CycleError, TaskGraph


def test_diamond_graph_critical_path_correctness() -> None:
    graph = TaskGraph(
        edges=(
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("C", "D"),
        )
    )

    critical = graph.critical_path(weights={"A": 1.0, "B": 4.0, "C": 2.0, "D": 1.0})
    assert critical == ("A", "B", "D")


def test_cycle_detection_returns_cycle() -> None:
    graph = TaskGraph(
        edges=(
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("C", "D"),
        )
    )

    cycles = graph.detect_cycles()
    assert cycles
    assert any(path[0] == path[-1] and set(path[:-1]) == {"A", "B", "C"} for path in cycles)

    with pytest.raises(CycleError) as error:
        graph.topological_sort()
    assert any(set(path[:-1]) == {"A", "B", "C"} for path in error.value.cycles)


def test_dependency_queries_runnable_and_serialization_are_deterministic() -> None:
    graph = TaskGraph(
        nodes=("node-b", "node-a"),
        edges=(
            ("node-a", "node-c"),
            ("node-a", "node-b"),
            ("node-b", "node-d"),
            ("node-c", "node-d"),
        ),
    )

    assert graph.get_dependencies("node-d") == ("node-b", "node-c")
    assert graph.get_dependencies("node-d", transitive=True) == ("node-a", "node-b", "node-c")
    assert graph.get_dependents("node-a") == ("node-b", "node-c")
    assert graph.get_dependents("node-a", transitive=True) == ("node-b", "node-c", "node-d")
    assert graph.get_runnable(set()) == ("node-a",)
    assert graph.get_runnable({"node-a"}) == ("node-b", "node-c")

    payload = graph.serialize()
    assert payload == {
        "nodes": ["node-a", "node-b", "node-c", "node-d"],
        "edges": [
            ["node-a", "node-b"],
            ["node-a", "node-c"],
            ["node-b", "node-d"],
            ["node-c", "node-d"],
        ],
    }
    restored = TaskGraph.deserialize(payload)
    assert restored.serialize() == payload


def test_seeded_random_dag_with_1000_nodes_topological_sort_stress() -> None:
    rng = random.Random(2_026_021_4)
    node_count = 1_000
    node_ids = [f"task-{index:04d}" for index in range(node_count)]

    graph = TaskGraph(nodes=node_ids)

    for child_index in range(1, node_count):
        fan_in = min(4, child_index)
        for parent_index in rng.sample(range(child_index), fan_in):
            if rng.random() < 0.55:
                graph.add_edge(node_ids[parent_index], node_ids[child_index])

    order = graph.topological_sort()
    assert len(order) == node_count

    position = {node_id: index for index, node_id in enumerate(order)}
    assert len(position) == node_count
    for parent, child in graph.edges:
        assert position[parent] < position[child]
