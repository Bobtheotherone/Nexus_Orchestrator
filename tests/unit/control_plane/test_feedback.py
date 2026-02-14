"""Unit tests for control-plane budget tracking and feedback synthesis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, cast

from nexus_orchestrator.control_plane.budgets import BudgetAction, BudgetTracker
from nexus_orchestrator.control_plane.feedback import FeedbackSynthesizer
from nexus_orchestrator.domain import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    EvidenceResult,
    ids,
)
from nexus_orchestrator.persistence.repositories import ProviderCallRecord
from nexus_orchestrator.security.redaction import REDACTED_VALUE
from nexus_orchestrator.synthesis_plane.roles import ROLE_IMPLEMENTER, RoleBudget, RoleRegistry
from nexus_orchestrator.verification_plane.constraint_gate import (
    ConstraintGateDecision,
    PipelineCheckResult,
    PipelineOutput,
    VerificationSelectionMode,
    evaluate_constraint_gate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


@dataclass(slots=True)
class RecordingLogger:
    events: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def info(self, event: str, **kwargs: object) -> None:
        self.events.append((event, dict(kwargs)))


def _randbytes(seed: int) -> Callable[[int], bytes]:
    byte_value = seed % 255

    def factory(size: int) -> bytes:
        return bytes([byte_value]) * size

    return factory


def _attempt_id(seed: int) -> str:
    return ids.generate_attempt_id(
        timestamp_ms=1_760_000_000_000 + seed, randbytes=_randbytes(seed)
    )


def _work_item_id(seed: int) -> str:
    return ids.generate_work_item_id(
        timestamp_ms=1_760_000_100_000 + seed, randbytes=_randbytes(seed + 1)
    )


def _provider_call(
    *,
    record_id: str,
    attempt_seed: int,
    provider: str = "openai",
    model: str | None = "gpt-5-codex",
    tokens: int = 100,
    cost_usd: float = 0.05,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
    created_offset_seconds: int = 0,
) -> ProviderCallRecord:
    return ProviderCallRecord(
        id=record_id,
        attempt_id=_attempt_id(attempt_seed),
        provider=provider,
        model=model,
        tokens=tokens,
        cost_usd=cost_usd,
        latency_ms=20,
        created_at=datetime(2026, 2, 1, tzinfo=UTC) + timedelta(seconds=created_offset_seconds),
        request_id=f"req-{record_id}",
        error=error,
        metadata=cast("dict[str, Any]", metadata or {}),
    )


def _constraint(
    *,
    constraint_id: str,
    severity: ConstraintSeverity,
    checker_binding: str,
) -> Constraint:
    return Constraint(
        id=constraint_id,
        severity=severity,
        category="quality",
        description=f"constraint {constraint_id}",
        checker_binding=checker_binding,
        parameters={},
        requirement_links=(),
        source=ConstraintSource.MANUAL,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _rejected_gate_decision() -> ConstraintGateDecision:
    must_constraint = _constraint(
        constraint_id="CON-SEC-1200",
        severity=ConstraintSeverity.MUST,
        checker_binding="security_checker",
    )
    should_constraint = _constraint(
        constraint_id="CON-DOC-1201",
        severity=ConstraintSeverity.SHOULD,
        checker_binding="documentation_checker",
    )
    envelope = ConstraintEnvelope(
        work_item_id=_work_item_id(42),
        constraints=(should_constraint, must_constraint),
    )
    output = PipelineOutput(
        check_results=(
            PipelineCheckResult(
                stage="security",
                checker_id="security_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=(),
            ),
            PipelineCheckResult(
                stage="documentation",
                checker_id="documentation_checker",
                result=EvidenceResult.FAIL,
                covered_constraint_ids=(),
            ),
        ),
        mode=VerificationSelectionMode.INCREMENTAL,
        selected_stages=("documentation", "security"),
        required_stages=("security",),
    )
    return evaluate_constraint_gate(envelope, output)


def test_budget_tracker_actions_are_deterministic_and_logged() -> None:
    logger = RecordingLogger()
    tracker = BudgetTracker(role_registry=RoleRegistry.default(), logger=logger)
    work_item_id = _work_item_id(1)
    records = (
        _provider_call(record_id="pc-1", attempt_seed=1, tokens=120, cost_usd=0.06),
        _provider_call(record_id="pc-2", attempt_seed=1, tokens=30, cost_usd=0.04),
    )

    decision_continue = tracker.decide(
        role_name=ROLE_IMPLEMENTER,
        work_item_id=work_item_id,
        next_attempt_number=2,
        work_item_records=records,
    )
    decision_continue_repeat = tracker.decide(
        role_name=ROLE_IMPLEMENTER,
        work_item_id=work_item_id,
        next_attempt_number=2,
        work_item_records=records,
    )
    decision_switch = tracker.decide(
        role_name=ROLE_IMPLEMENTER,
        work_item_id=work_item_id,
        next_attempt_number=3,
        work_item_records=records,
    )
    decision_stop = tracker.decide(
        role_name=ROLE_IMPLEMENTER,
        work_item_id=work_item_id,
        next_attempt_number=2,
        work_item_records=records,
        budget_override=RoleBudget(max_cost_per_work_item_usd=0.05),
    )

    assert decision_continue == decision_continue_repeat
    assert decision_continue.action is BudgetAction.CONTINUE
    assert decision_continue.provider == "openai"
    assert decision_switch.action is BudgetAction.SWITCH
    assert decision_switch.provider == "anthropic"
    assert decision_stop.action is BudgetAction.STOP
    assert "work_item_cost_cap_reached" in decision_stop.reason_codes

    assert len(logger.events) == 4
    assert all(event == "control_plane_budget_decision" for event, _ in logger.events)


def test_budget_tracker_enforces_optional_global_cap_with_catalog_cost_fallback() -> None:
    tracker = BudgetTracker(
        role_registry=RoleRegistry.default(),
        global_max_total_cost_usd=0.001,
    )
    work_item_id = _work_item_id(2)
    record = _provider_call(
        record_id="pc-global",
        attempt_seed=2,
        provider="openai",
        model="gpt-5-codex",
        tokens=1000,
        cost_usd=0.0,
    )

    decision = tracker.decide(
        role_name=ROLE_IMPLEMENTER,
        work_item_id=work_item_id,
        next_attempt_number=2,
        work_item_records=(record,),
        global_records=(record,),
    )

    assert decision.action is BudgetAction.STOP
    assert "global_cost_cap_reached" in decision.reason_codes


def test_feedback_synthesizer_is_deterministic_and_stably_sorted() -> None:
    gate_decision = _rejected_gate_decision()
    provider_calls = (
        _provider_call(
            record_id="pc-b",
            attempt_seed=4,
            created_offset_seconds=30,
            metadata={"artifact_paths": ["logs/z.log", "logs/a.log"]},
        ),
        _provider_call(
            record_id="pc-a",
            attempt_seed=3,
            created_offset_seconds=5,
            metadata={"trace_path": "logs/trace.log"},
        ),
    )
    synthesizer = FeedbackSynthesizer()

    payload_a = synthesizer.synthesize(
        gate_decision=gate_decision,
        provider_calls=provider_calls,
        artifact_paths=("logs/manual.log", "logs/a.log"),
    )
    payload_b = synthesizer.synthesize(
        gate_decision=gate_decision,
        provider_calls=provider_calls,
        artifact_paths=("logs/manual.log", "logs/a.log"),
    )

    assert payload_a == payload_b

    reason_codes = cast("list[str]", payload_a["reason_codes"])
    assert reason_codes == sorted(reason_codes)
    assert "must_constraint_unsatisfied" in reason_codes

    remediation_hints = cast("list[str]", payload_a["remediation_hints"])
    assert remediation_hints == sorted(remediation_hints)
    assert remediation_hints

    artifact_paths = cast("list[str]", payload_a["artifact_paths"])
    assert artifact_paths == sorted(artifact_paths)
    assert artifact_paths == ["logs/a.log", "logs/manual.log", "logs/trace.log", "logs/z.log"]

    provider_payload = cast("list[dict[str, object]]", payload_a["provider_calls"])
    assert [item["id"] for item in provider_payload] == ["pc-a", "pc-b"]


def test_feedback_synthesizer_redacts_secrets_in_provider_errors_and_metadata() -> None:
    gate_decision = _rejected_gate_decision()
    secret = "sk-THISISFAKE12345678901234567890"
    provider_call = _provider_call(
        record_id="pc-secret",
        attempt_seed=7,
        error=f"authorization: bearer {secret}",
        metadata={
            "api_key": secret,
            "notes": "safe",
            "auth_token": "token-abcdef123456",
        },
    )
    payload = FeedbackSynthesizer().synthesize(
        gate_decision=gate_decision,
        provider_calls=(provider_call,),
        artifact_paths=("logs/security.log",),
    )

    rendered = json.dumps(payload, sort_keys=True)
    assert secret not in rendered
    assert REDACTED_VALUE in rendered

    provider_payload = cast("list[dict[str, object]]", payload["provider_calls"])
    provider_metadata = cast("dict[str, object]", provider_payload[0]["metadata"])
    assert provider_metadata["api_key"] == REDACTED_VALUE
    assert provider_metadata["auth_token"] == REDACTED_VALUE
    assert provider_metadata["notes"] == "safe"
