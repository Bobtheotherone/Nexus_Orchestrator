"""
Budget tracking and deterministic control-plane decisions.

This module enforces budget envelopes for retry control:
- per-work-item caps (attempts, tokens per attempt, cost)
- optional global run cost cap
- deterministic next-action decisions (`continue`, `switch`, `stop`)

It integrates with:
- `RoleRegistry` for deterministic provider/model routing
- `load_model_catalog()` for cost fallback estimation
- `ProviderCallRecord` persistence rows for accounting
- `structlog` for machine-parseable decision logs
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog, load_model_catalog
from nexus_orchestrator.synthesis_plane.roles import EscalationDecision, RoleBudget, RoleRegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nexus_orchestrator.persistence.repositories import ProviderCallRecord

_ROUTE_STOP_SCAN_LIMIT = 10_000


class BudgetAction(StrEnum):
    """Deterministic control action for the next work-item attempt."""

    CONTINUE = "continue"
    SWITCH = "switch"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """Aggregated usage snapshot from provider call records."""

    attempt_count: int
    tokens_by_attempt: tuple[tuple[str, int], ...]
    max_tokens_in_single_attempt: int
    work_item_tokens_total: int
    work_item_cost_total_usd: float
    global_cost_total_usd: float

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_count": self.attempt_count,
            "tokens_by_attempt": [
                {"attempt_id": attempt_id, "tokens": tokens}
                for attempt_id, tokens in self.tokens_by_attempt
            ],
            "max_tokens_in_single_attempt": self.max_tokens_in_single_attempt,
            "work_item_tokens_total": self.work_item_tokens_total,
            "work_item_cost_total_usd": self.work_item_cost_total_usd,
            "global_cost_total_usd": self.global_cost_total_usd,
        }


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """Deterministic budget decision for a pending attempt."""

    action: BudgetAction
    reason_codes: tuple[str, ...]
    role_name: str
    work_item_id: str
    next_attempt_number: int
    provider: str | None
    model: str | None
    usage: BudgetUsage
    max_attempts: int | None
    max_tokens_per_attempt: int | None
    max_cost_per_work_item_usd: float | None
    max_total_cost_usd: float | None

    @property
    def should_stop(self) -> bool:
        return self.action is BudgetAction.STOP

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "reason_codes": list(self.reason_codes),
            "role_name": self.role_name,
            "work_item_id": self.work_item_id,
            "next_attempt_number": self.next_attempt_number,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "limits": {
                "max_attempts": self.max_attempts,
                "max_tokens_per_attempt": self.max_tokens_per_attempt,
                "max_cost_per_work_item_usd": self.max_cost_per_work_item_usd,
                "max_total_cost_usd": self.max_total_cost_usd,
            },
        }


class BudgetTracker:
    """
    Enforce budget envelopes and derive deterministic retry routing actions.

    Action semantics:
    - `continue`: proceed with the route for `next_attempt_number`
    - `switch`: continue but switch provider/model route
    - `stop`: halt retries for this work item
    """

    def __init__(
        self,
        *,
        role_registry: RoleRegistry | None = None,
        model_catalog: ModelCatalog | None = None,
        global_max_total_cost_usd: float | None = None,
        logger: Any | None = None,
    ) -> None:
        catalog = model_catalog if model_catalog is not None else load_model_catalog()
        if global_max_total_cost_usd is not None and global_max_total_cost_usd < 0:
            raise ValueError("global_max_total_cost_usd must be >= 0")

        self._model_catalog = catalog
        self._role_registry = (
            role_registry
            if role_registry is not None
            else RoleRegistry.default(model_catalog=catalog)
        )
        self._global_max_total_cost_usd = global_max_total_cost_usd
        self._logger = logger if logger is not None else structlog.get_logger(__name__)

    def summarize_usage(
        self,
        *,
        work_item_records: Sequence[ProviderCallRecord],
        global_records: Sequence[ProviderCallRecord] | None = None,
    ) -> BudgetUsage:
        ordered_work_item = _ordered_records(work_item_records)
        ordered_global = (
            _ordered_records(global_records) if global_records is not None else ordered_work_item
        )

        tokens_by_attempt = _tokens_by_attempt(ordered_work_item)
        max_tokens_single_attempt = 0
        work_item_tokens_total = 0
        for _, tokens in tokens_by_attempt:
            work_item_tokens_total += tokens
            if tokens > max_tokens_single_attempt:
                max_tokens_single_attempt = tokens

        work_item_cost = _round_cost(
            sum(self._effective_cost_usd(record) for record in ordered_work_item)
        )
        global_cost = _round_cost(
            sum(self._effective_cost_usd(record) for record in ordered_global)
        )

        return BudgetUsage(
            attempt_count=len(tokens_by_attempt),
            tokens_by_attempt=tokens_by_attempt,
            max_tokens_in_single_attempt=max_tokens_single_attempt,
            work_item_tokens_total=work_item_tokens_total,
            work_item_cost_total_usd=work_item_cost,
            global_cost_total_usd=global_cost,
        )

    def decide(
        self,
        *,
        role_name: str,
        work_item_id: str,
        next_attempt_number: int,
        work_item_records: Sequence[ProviderCallRecord],
        global_records: Sequence[ProviderCallRecord] | None = None,
        budget_override: RoleBudget | None = None,
        global_max_total_cost_usd: float | None = None,
    ) -> BudgetDecision:
        if not isinstance(role_name, str) or not role_name.strip():
            raise ValueError("role_name must be a non-empty string")
        if not isinstance(work_item_id, str) or not work_item_id.strip():
            raise ValueError("work_item_id must be a non-empty string")
        if next_attempt_number <= 0:
            raise ValueError("next_attempt_number must be > 0")

        role = self._role_registry.require(role_name)
        effective_budget = role.budget.merge(budget_override)
        global_cap = (
            self._global_max_total_cost_usd
            if global_max_total_cost_usd is None
            else global_max_total_cost_usd
        )
        if global_cap is not None and global_cap < 0:
            raise ValueError("global_max_total_cost_usd must be >= 0")

        usage = self.summarize_usage(
            work_item_records=work_item_records,
            global_records=global_records,
        )

        route = self._role_registry.route_attempt(
            role_name=role.name,
            attempt_number=next_attempt_number,
            budget_override=budget_override,
        )
        previous_route = (
            self._role_registry.route_attempt(
                role_name=role.name,
                attempt_number=next_attempt_number - 1,
                budget_override=budget_override,
            )
            if next_attempt_number > 1
            else None
        )

        reasons: list[str] = []
        hard_stop = False
        if (
            effective_budget.max_attempts is not None
            and next_attempt_number > effective_budget.max_attempts
        ):
            _append_reason(reasons, "max_attempts_reached")
            hard_stop = True
        if route is None:
            _append_reason(reasons, "no_route_for_attempt")
            hard_stop = True
        if (
            effective_budget.max_cost_per_work_item_usd is not None
            and usage.work_item_cost_total_usd >= effective_budget.max_cost_per_work_item_usd
        ):
            _append_reason(reasons, "work_item_cost_cap_reached")
            hard_stop = True
        if global_cap is not None and usage.global_cost_total_usd >= global_cap:
            _append_reason(reasons, "global_cost_cap_reached")
            hard_stop = True

        selected_route: EscalationDecision | None = route
        if hard_stop:
            action = BudgetAction.STOP
            selected_route = None
        elif (
            effective_budget.max_tokens_per_attempt is not None
            and usage.max_tokens_in_single_attempt > effective_budget.max_tokens_per_attempt
        ):
            _append_reason(reasons, "attempt_token_cap_exceeded")
            if route is None:
                action = BudgetAction.STOP
                selected_route = None
            else:
                switch_route = self._next_distinct_route(
                    role_name=role.name,
                    start_attempt_number=next_attempt_number,
                    budget_override=budget_override,
                    current_route=route,
                )
                if switch_route is None:
                    action = BudgetAction.STOP
                    selected_route = None
                    _append_reason(reasons, "no_alternative_route")
                else:
                    action = BudgetAction.SWITCH
                    selected_route = switch_route
                    _append_reason(reasons, "token_cap_requires_switch")
        else:
            if route is not None and _route_key(route) != _route_key(previous_route):
                action = BudgetAction.SWITCH
                _append_reason(reasons, "role_escalation")
            else:
                action = BudgetAction.CONTINUE
                _append_reason(reasons, "within_budget")

        decision = BudgetDecision(
            action=action,
            reason_codes=tuple(reasons),
            role_name=role.name,
            work_item_id=work_item_id.strip(),
            next_attempt_number=next_attempt_number,
            provider=selected_route.provider if selected_route is not None else None,
            model=selected_route.model if selected_route is not None else None,
            usage=usage,
            max_attempts=effective_budget.max_attempts,
            max_tokens_per_attempt=effective_budget.max_tokens_per_attempt,
            max_cost_per_work_item_usd=effective_budget.max_cost_per_work_item_usd,
            max_total_cost_usd=global_cap,
        )
        self._log_decision(decision)
        return decision

    def evaluate(self, **kwargs: Any) -> BudgetDecision:
        """Compatibility alias for `decide()`."""

        return self.decide(**kwargs)

    def decision_for(self, **kwargs: Any) -> BudgetDecision:
        """Compatibility alias for `decide()`."""

        return self.decide(**kwargs)

    def _next_distinct_route(
        self,
        *,
        role_name: str,
        start_attempt_number: int,
        budget_override: RoleBudget | None,
        current_route: EscalationDecision,
    ) -> EscalationDecision | None:
        for offset in range(_ROUTE_STOP_SCAN_LIMIT):
            attempt_number = start_attempt_number + offset
            candidate = self._role_registry.route_attempt(
                role_name=role_name,
                attempt_number=attempt_number,
                budget_override=budget_override,
            )
            if candidate is None:
                return None
            if _route_key(candidate) != _route_key(current_route):
                return candidate
        return None

    def _effective_cost_usd(self, record: ProviderCallRecord) -> float:
        if record.cost_usd > 0:
            return float(record.cost_usd)
        if record.tokens <= 0:
            return 0.0
        model = record.model
        if model is None:
            return 0.0
        try:
            return self._model_catalog.estimate_cost(
                provider=record.provider,
                model=model,
                total_tokens=record.tokens,
            )
        except (KeyError, ValueError):
            return 0.0

    def _log_decision(self, decision: BudgetDecision) -> None:
        self._logger.info(
            "control_plane_budget_decision",
            action=decision.action.value,
            reason_codes=list(decision.reason_codes),
            role_name=decision.role_name,
            work_item_id=decision.work_item_id,
            next_attempt_number=decision.next_attempt_number,
            provider=decision.provider,
            model=decision.model,
            usage=decision.usage.to_dict(),
            limits={
                "max_attempts": decision.max_attempts,
                "max_tokens_per_attempt": decision.max_tokens_per_attempt,
                "max_cost_per_work_item_usd": decision.max_cost_per_work_item_usd,
                "max_total_cost_usd": decision.max_total_cost_usd,
            },
        )


def _append_reason(reason_codes: list[str], reason_code: str) -> None:
    if reason_code not in reason_codes:
        reason_codes.append(reason_code)


def _ordered_records(
    records: Sequence[ProviderCallRecord] | None,
) -> tuple[ProviderCallRecord, ...]:
    if records is None:
        return ()
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.created_at.isoformat(),
                item.id,
                item.attempt_id,
                item.provider,
                item.model or "",
            ),
        )
    )


def _tokens_by_attempt(records: Sequence[ProviderCallRecord]) -> tuple[tuple[str, int], ...]:
    tokens_by_attempt: dict[str, int] = {}
    for record in records:
        tokens_by_attempt[record.attempt_id] = tokens_by_attempt.get(record.attempt_id, 0) + int(
            record.tokens
        )
    return tuple(sorted(tokens_by_attempt.items(), key=lambda item: item[0]))


def _route_key(route: EscalationDecision | None) -> tuple[str, str] | None:
    if route is None:
        return None
    return (route.provider, route.model)


def _round_cost(value: float) -> float:
    return float(round(value, 12))


__all__ = [
    "BudgetAction",
    "BudgetDecision",
    "BudgetTracker",
    "BudgetUsage",
]
