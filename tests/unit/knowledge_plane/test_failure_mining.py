"""Unit tests for deterministic offline failure mining."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from nexus_orchestrator.domain import ids as domain_ids
from nexus_orchestrator.domain.models import (
    Attempt,
    AttemptResult,
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    EvidenceResult,
    WorkItem,
)
from nexus_orchestrator.knowledge_plane.constraint_registry import ConstraintRegistry
from nexus_orchestrator.knowledge_plane.failure_mining import ConstraintMiner, FailureTaxonomy
from nexus_orchestrator.verification_plane.constraint_gate import (
    ConstraintGateDecision,
    PipelineCheckResult,
    PipelineOutput,
    VerificationSelectionMode,
    evaluate_constraint_gate,
)

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    UTC = timezone.utc  # noqa: UP017

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _randbytes(seed: int) -> Callable[[int], bytes]:
    byte = (seed % 251) + 1

    def factory(size: int) -> bytes:
        return bytes([byte]) * size

    return factory


def _constraint(
    *,
    constraint_id: str,
    severity: ConstraintSeverity,
    checker_binding: str,
    category: str,
) -> Constraint:
    return Constraint(
        id=constraint_id,
        severity=severity,
        category=category,
        description=f"constraint {constraint_id}",
        checker_binding=checker_binding,
        parameters={},
        requirement_links=(),
        source=ConstraintSource.MANUAL,
        created_at=datetime(2026, 2, 1, tzinfo=UTC),
    )


def _make_work_item(*, constraints: tuple[Constraint, ...], seed: int = 10) -> WorkItem:
    work_item_id = domain_ids.generate_work_item_id(
        timestamp_ms=1_760_000_100_000 + seed,
        randbytes=_randbytes(seed),
    )
    envelope = ConstraintEnvelope(work_item_id=work_item_id, constraints=constraints)
    return WorkItem(
        id=work_item_id,
        title="Failure mining fixture",
        description="Work item for failure mining unit tests",
        scope=("src/nexus_orchestrator/knowledge_plane/failure_mining.py",),
        constraint_envelope=envelope,
        requirement_links=("REQ-0001", "REQ-0002"),
        created_at=datetime(2026, 2, 10, 9, 0, tzinfo=UTC),
        updated_at=datetime(2026, 2, 10, 10, 0, tzinfo=UTC),
    )


def _make_attempt(*, work_item_id: str, seed: int = 10) -> Attempt:
    return Attempt(
        id=domain_ids.generate_attempt_id(
            timestamp_ms=1_760_000_200_000 + seed,
            randbytes=_randbytes(seed + 1),
        ),
        work_item_id=work_item_id,
        run_id=domain_ids.generate_run_id(
            timestamp_ms=1_760_000_300_000 + seed,
            randbytes=_randbytes(seed + 2),
        ),
        iteration=1,
        provider="openai",
        model="gpt-5-codex",
        role="constraint_miner",
        prompt_hash="a" * 64,
        tokens_used=100,
        cost_usd=0.01,
        result=AttemptResult.CONSTRAINT_FAILURE,
        created_at=datetime(2026, 2, 10, 10, 5, tzinfo=UTC),
    )


def _make_pipeline_output() -> PipelineOutput:
    return PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="security_scan",
                checker_id="security_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=("CON-SEC-9001",),
                summary="secret token found in logs",
            ),
            PipelineCheckResult(
                stage="performance",
                checker_id="performance_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=("CON-PERF-9001",),
                summary="latency regression exceeded threshold",
            ),
            PipelineCheckResult(
                stage="build",
                checker_id="build_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=("CON-BUG-9001",),
                summary="compilation failed",
            ),
            PipelineCheckResult(
                stage="unit_tests",
                checker_id="test_checker",
                result=EvidenceResult.WARN,
                covered_constraint_ids=("CON-FLK-9001",),
                summary="flaky intermittent timeout",
            ),
        ),
        mode=VerificationSelectionMode.INCREMENTAL,
        selected_stages=("build", "unit_tests", "security_scan", "performance"),
        required_stages=("build", "security_scan"),
    )


def _make_gate_and_inputs() -> tuple[WorkItem, Attempt, PipelineOutput, ConstraintGateDecision]:
    constraints = (
        _constraint(
            constraint_id="CON-SEC-9001",
            severity=ConstraintSeverity.MUST,
            checker_binding="security_checker",
            category="security",
        ),
        _constraint(
            constraint_id="CON-PERF-9001",
            severity=ConstraintSeverity.SHOULD,
            checker_binding="performance_checker",
            category="performance",
        ),
        _constraint(
            constraint_id="CON-BUG-9001",
            severity=ConstraintSeverity.MUST,
            checker_binding="build_checker",
            category="correctness",
        ),
        _constraint(
            constraint_id="CON-FLK-9001",
            severity=ConstraintSeverity.SHOULD,
            checker_binding="test_checker",
            category="reliability",
        ),
        _constraint(
            constraint_id="CON-SPEC-9001",
            severity=ConstraintSeverity.SHOULD,
            checker_binding="documentation_checker",
            category="documentation",
        ),
    )
    work_item = _make_work_item(constraints=constraints)
    attempt = _make_attempt(work_item_id=work_item.id)
    output = _make_pipeline_output()
    gate_decision = evaluate_constraint_gate(work_item.constraint_envelope, output)
    return work_item, attempt, output, gate_decision


def _empty_registry(tmp_path: Path) -> ConstraintRegistry:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    return ConstraintRegistry.load(registry_dir)


def test_mine_is_deterministic_and_covers_failure_taxonomy(tmp_path: Path) -> None:
    work_item, attempt, output, gate_decision = _make_gate_and_inputs()
    registry = _empty_registry(tmp_path)
    miner = ConstraintMiner(registry=registry)

    proposals_a = miner.mine(
        gate_decision=gate_decision,
        pipeline_output=output,
        work_item=work_item,
        attempt=attempt,
    )
    proposals_b = miner.mine(
        gate_decision=gate_decision,
        pipeline_output=output,
        work_item=work_item,
        attempt=attempt,
    )

    assert proposals_a == proposals_b
    assert proposals_a

    categories = {
        proposal.proposed_constraint.parameters["failure_category"] for proposal in proposals_a
    }
    assert isinstance(categories, set)
    assert {
        FailureTaxonomy.BUG.value,
        FailureTaxonomy.SPEC_GAP.value,
        FailureTaxonomy.FLAKE.value,
        FailureTaxonomy.PERF_REGRESSION.value,
        FailureTaxonomy.SECURITY_FINDING.value,
    }.issubset(categories)


def test_mine_deduplicates_by_id_and_semantic_signature(tmp_path: Path) -> None:
    work_item, attempt, output, gate_decision = _make_gate_and_inputs()
    registry = _empty_registry(tmp_path)
    miner = ConstraintMiner(registry=registry)

    duplicated_results = output.check_results + (output.check_results[0],)
    raw_proposals = miner.mine(
        gate_decision=gate_decision,
        pipeline_output=duplicated_results,
        work_item=work_item,
        attempt=attempt,
    )
    assert raw_proposals
    assert len(raw_proposals) == len({item.semantic_signature for item in raw_proposals})
    assert len(raw_proposals) >= 2

    duplicate_id = raw_proposals[0].proposed_constraint.id
    duplicate_signature = raw_proposals[1].semantic_signature
    registry.add_constraint(
        {
            "id": duplicate_id,
            "severity": "must",
            "category": "correctness",
            "description": "existing id duplicate",
            "checker": "build_checker",
            "parameters": {},
            "requirement_links": [],
            "source": "manual",
            "created_at": "2026-02-10T00:00:00Z",
        },
        filename="010_existing_id.yaml",
    )
    registry.add_constraint(
        {
            "id": "CON-DUP-9001",
            "severity": "should",
            "category": "reliability",
            "description": "existing semantic signature duplicate",
            "checker": "reliability_checker",
            "parameters": {
                "provenance": {
                    "semantic_signature": duplicate_signature,
                }
            },
            "requirement_links": [],
            "source": "failure_derived",
            "created_at": "2026-02-10T00:00:00Z",
        },
        filename="011_existing_signature.yaml",
    )
    refreshed = ConstraintRegistry.load(registry.registry_dir)
    filtered = ConstraintMiner(registry=refreshed).mine(
        gate_decision=gate_decision,
        pipeline_output=duplicated_results,
        work_item=work_item,
        attempt=attempt,
    )

    filtered_ids = {item.proposed_constraint.id for item in filtered}
    filtered_signatures = {item.semantic_signature for item in filtered}
    assert duplicate_id not in filtered_ids
    assert duplicate_signature not in filtered_signatures


def test_apply_writes_provenance_and_skips_non_auto_accept_by_default(tmp_path: Path) -> None:
    work_item, attempt, output, gate_decision = _make_gate_and_inputs()
    registry = _empty_registry(tmp_path)
    miner = ConstraintMiner(registry=registry)

    proposals = miner.mine(
        gate_decision=gate_decision,
        pipeline_output=output,
        work_item=work_item,
        attempt=attempt,
    )
    assert proposals
    assert any(not item.auto_accept for item in proposals)

    written_paths = miner.apply(
        proposals,
        registry=registry,
        current_date=date(2026, 2, 14),
    )
    assert written_paths

    reloaded = ConstraintRegistry.load(registry.registry_dir)
    applied_count = 0
    for proposal in proposals:
        applied_constraint = reloaded.by_id(proposal.proposed_constraint.id)
        if proposal.auto_accept:
            assert applied_constraint is not None
            provenance_raw = applied_constraint.parameters.get("provenance")
            assert isinstance(provenance_raw, dict)
            assert provenance_raw["source_failure_id"] == proposal.source_failure_id
            assert provenance_raw["miner_version"] == miner.miner_version
            assert provenance_raw["semantic_signature"] == proposal.semantic_signature
            assert provenance_raw["work_item_id"] == work_item.id
            assert provenance_raw["attempt_id"] == attempt.id
            applied_count += 1
        else:
            assert applied_constraint is None

    assert applied_count == len(written_paths)

    written_again = miner.apply(
        proposals,
        registry=reloaded,
        current_date=date(2026, 2, 14),
    )
    assert written_again == ()


def test_mine_accepts_pipeline_results_sequence(tmp_path: Path) -> None:
    work_item, attempt, output, gate_decision = _make_gate_and_inputs()
    registry = _empty_registry(tmp_path)
    miner = ConstraintMiner(registry=registry)

    from_output = miner.mine(
        gate_decision=gate_decision,
        pipeline_output=output,
        work_item=work_item,
        attempt=attempt,
    )
    from_results = miner.mine(
        gate_decision=gate_decision,
        pipeline_output=output.check_results,
        work_item=work_item,
        attempt=attempt,
    )

    assert from_results == from_output
