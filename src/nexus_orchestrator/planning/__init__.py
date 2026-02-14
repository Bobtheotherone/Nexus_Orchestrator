"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/planning/__init__.py
Last updated: 2026-02-11

Purpose
- Planning layer: decomposition, constraint compilation, task graph generation, and re-planning.

What should be included in this file
- Planner entrypoints and high-level orchestration between architect agent and deterministic compiler.

Functional requirements
- Must output a valid DAG of work items and constraints.

Non-functional requirements
- Must produce repeatable plans given same spec/config (within nondeterminism constraints).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexus_orchestrator.planning.architect_interface import (
    ADRDraft,
    ArchitectOutput,
    ConstraintSuggestion,
    ModuleDefinition,
    WorkItemProposal,
    build_deterministic_architect_output,
    parse_architect_response,
    validate_architect_output,
)
from nexus_orchestrator.planning.constraint_compiler import CompilationResult, compile_constraints
from nexus_orchestrator.planning.task_graph import CycleError, TaskGraph

if TYPE_CHECKING:
    from pathlib import Path

    from nexus_orchestrator.spec_ingestion.spec_map import SpecMap


def plan_spec_map(
    spec_map: SpecMap,
    *,
    architect_output: ArchitectOutput | None = None,
    registry_path: str | Path = "constraints/registry",
    run_id: str = "run-00000000000000000000000000",
) -> CompilationResult:
    """
    Build a deterministic plan for a ``SpecMap``.

    When ``architect_output`` is omitted, a deterministic provider-free decomposition
    is built directly from ingested interface contracts.
    """

    output = (
        architect_output
        if architect_output is not None
        else build_deterministic_architect_output(spec_map)
    )
    return compile_constraints(
        spec_map,
        output,
        registry_path=registry_path,
        run_id=run_id,
    )


__all__ = [
    "ADRDraft",
    "ArchitectOutput",
    "CompilationResult",
    "ConstraintSuggestion",
    "CycleError",
    "ModuleDefinition",
    "TaskGraph",
    "WorkItemProposal",
    "build_deterministic_architect_output",
    "compile_constraints",
    "parse_architect_response",
    "plan_spec_map",
    "validate_architect_output",
]
