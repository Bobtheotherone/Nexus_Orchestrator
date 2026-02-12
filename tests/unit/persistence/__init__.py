"""Shared deterministic fixtures and builders for persistence tests."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Final

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Attempt,
    AttemptResult,
    Budget,
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    EvidenceRecord,
    EvidenceResult,
    Incident,
    MergeRecord,
    RiskTier,
    Run,
    RunStatus,
    SandboxPolicy,
    TaskGraph,
    WorkItem,
    WorkItemStatus,
)

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017

_BASE_TS: Final[datetime] = datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def fixed_now(seed: int) -> datetime:
    return _BASE_TS + timedelta(seconds=seed)


def _randbytes(seed: int):
    byte_value = (seed % 251) + 1

    def _provider(size: int) -> bytes:
        return bytes([byte_value]) * size

    return _provider


def _sha256(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def make_budget(seed: int) -> Budget:
    return Budget(
        max_tokens=10_000 + seed,
        max_cost_usd=5.0,
        max_iterations=3,
        max_wall_clock_seconds=600,
    )


def make_sandbox_policy() -> SandboxPolicy:
    return SandboxPolicy(
        allow_network=False,
        allow_privileged_tools=False,
        allowed_tools=("rg", "pytest"),
        read_only_paths=("docs",),
        write_paths=("src", "tests"),
        max_cpu_seconds=30,
        max_memory_mb=1024,
    )


def make_constraint(
    seed: int,
    *,
    parameters: dict[str, object] | None = None,
    requirement_links: tuple[str, ...] = ("REQ-0001",),
) -> Constraint:
    return Constraint(
        id=f"CON-SEC-{(seed % 9_000) + 1:04d}",
        severity=ConstraintSeverity.MUST,
        category="security",
        description="Constraint for persistence testing",
        checker_binding="security_checker",
        parameters={} if parameters is None else parameters,
        requirement_links=requirement_links,
        source=ConstraintSource.MANUAL,
        created_at=fixed_now(seed),
    )


def make_work_item(
    seed: int,
    *,
    dependencies: tuple[str, ...] = (),
    status: WorkItemStatus = WorkItemStatus.READY,
    risk_tier: RiskTier = RiskTier.MEDIUM,
    constraint_parameters: dict[str, object] | None = None,
) -> WorkItem:
    work_item_id = ids.generate_work_item_id(
        timestamp_ms=1_700_000_000_000 + seed,
        randbytes=_randbytes(seed),
    )
    created_at = fixed_now(seed)
    updated_at = fixed_now(seed + 1)
    constraint = make_constraint(seed, parameters=constraint_parameters)
    envelope = ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=(constraint,),
        inherited_constraint_ids=(),
        compiled_at=created_at,
    )
    return WorkItem(
        id=work_item_id,
        title=f"Work item {seed}",
        description="Implement persistence behavior",
        scope=(f"src/module_{seed}.py",),
        constraint_envelope=envelope,
        dependencies=dependencies,
        status=status,
        risk_tier=risk_tier,
        budget=make_budget(seed),
        sandbox_policy=make_sandbox_policy(),
        requirement_links=("REQ-0001",),
        constraint_ids=(constraint.id,),
        evidence_ids=(),
        expected_artifacts=(f"artifacts/{seed}.json",),
        created_at=created_at,
        updated_at=updated_at,
    )


def make_run(
    seed: int,
    *,
    status: RunStatus = RunStatus.RUNNING,
    work_item_ids: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> Run:
    return Run(
        id=ids.generate_run_id(
            timestamp_ms=1_700_000_100_000 + seed,
            randbytes=_randbytes(seed),
        ),
        spec_path="samples/specs/minimal_design_doc.md",
        status=status,
        started_at=fixed_now(seed),
        finished_at=None,
        work_item_ids=work_item_ids,
        budget=make_budget(seed),
        risk_tier=RiskTier.MEDIUM,
        metadata={} if metadata is None else metadata,
    )


def make_attempt(seed: int, *, run_id: str, work_item_id: str) -> Attempt:
    return Attempt(
        id=ids.generate_attempt_id(
            timestamp_ms=1_700_000_200_000 + seed,
            randbytes=_randbytes(seed),
        ),
        work_item_id=work_item_id,
        run_id=run_id,
        iteration=1,
        provider="openai",
        model="gpt-5",
        role="implementer",
        prompt_hash=_sha256(f"prompt-{seed}"),
        tokens_used=500,
        cost_usd=0.25,
        result=AttemptResult.SUCCESS,
        created_at=fixed_now(seed),
        finished_at=fixed_now(seed + 1),
    )


def make_evidence(
    seed: int,
    *,
    run_id: str,
    work_item_id: str,
    constraint_ids: tuple[str, ...] = ("CON-SEC-0001",),
    metadata: dict[str, object] | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        id=ids.generate_evidence_id(
            timestamp_ms=1_700_000_300_000 + seed,
            randbytes=_randbytes(seed),
        ),
        work_item_id=work_item_id,
        run_id=run_id,
        stage="tests",
        result=EvidenceResult.PASS,
        checker_id="pytest",
        constraint_ids=constraint_ids,
        artifact_paths=(f"artifacts/evidence-{seed}.json",),
        tool_versions={"pytest": "9.0.2"},
        environment_hash=_sha256(f"env-{seed}"),
        duration_ms=123,
        created_at=fixed_now(seed),
        summary="verification passed",
        metadata={} if metadata is None else metadata,
    )


def make_merge(
    seed: int, *, run_id: str, work_item_id: str, evidence_ids: tuple[str, ...]
) -> MergeRecord:
    return MergeRecord(
        id=ids.generate_merge_id(
            timestamp_ms=1_700_000_400_000 + seed,
            randbytes=_randbytes(seed),
        ),
        work_item_id=work_item_id,
        run_id=run_id,
        commit_sha=f"{seed:07x}",
        evidence_ids=evidence_ids,
        merged_at=fixed_now(seed),
    )


def make_incident(seed: int, *, run_id: str, work_item_id: str | None) -> Incident:
    return Incident(
        id=ids.generate_incident_id(
            timestamp_ms=1_700_000_500_000 + seed,
            randbytes=_randbytes(seed),
        ),
        run_id=run_id,
        category="runtime",
        message="simulated incident",
        created_at=fixed_now(seed),
        related_work_item_id=work_item_id,
        constraint_ids=("CON-SEC-0001",),
        evidence_ids=(),
        details={"retry": 1},
    )


def make_task_graph(
    run_id: str, work_items: tuple[WorkItem, ...], edges: tuple[tuple[str, str], ...]
) -> TaskGraph:
    return TaskGraph(
        run_id=run_id,
        work_items=work_items,
        edges=edges,
        critical_path=tuple(item.id for item in work_items),
        created_at=fixed_now(1),
    )


def collect_text_cells(conn: object) -> str:
    if not hasattr(conn, "execute"):
        raise TypeError("conn must expose execute()")

    cursor = conn.execute(  # type: ignore[call-arg]
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    table_names = [str(row[0]) for row in cursor.fetchall()]

    chunks: list[str] = []
    for table_name in table_names:
        if not _IDENTIFIER_RE.fullmatch(table_name):
            continue
        rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()  # type: ignore[call-arg]
        for row in rows:
            for value in row:
                if isinstance(value, str):
                    chunks.append(value)
    return "\n".join(chunks)


__all__ = [
    "collect_text_cells",
    "fixed_now",
    "make_attempt",
    "make_budget",
    "make_constraint",
    "make_evidence",
    "make_incident",
    "make_merge",
    "make_run",
    "make_sandbox_policy",
    "make_task_graph",
    "make_work_item",
]
