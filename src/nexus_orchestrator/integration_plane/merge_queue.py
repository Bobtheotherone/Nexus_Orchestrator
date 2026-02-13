"""Deterministic, restart-safe merge queue for serialized integration."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias, cast

from nexus_orchestrator.domain import RiskTier

if TYPE_CHECKING:
    from enum import StrEnum
else:
    try:
        from enum import StrEnum
    except ImportError:
        from enum import Enum

        class StrEnum(str, Enum):
            """Compatibility fallback for Python < 3.11."""


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_STATE_SCHEMA_VERSION: Final[int] = 1
_REQUIRED_STATE_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "next_arrival",
    "completed_work_items",
    "failed_work_items",
    "queue",
)

_RISK_ORDER: Final[dict[RiskTier, int]] = {
    RiskTier.LOW: 0,
    RiskTier.MEDIUM: 1,
    RiskTier.HIGH: 2,
    RiskTier.CRITICAL: 3,
}


class MergeStatus(StrEnum):
    """Terminal/observable result categories for one queue processing step."""

    NO_CANDIDATE = "no_candidate"
    MERGED = "merged"
    REQUEUED = "requeued"
    FAILED = "failed"


class MergeQueueStateError(ValueError):
    """Raised when persisted merge queue state is corrupted or incomplete."""


@dataclass(frozen=True, slots=True, init=False)
class QueueCandidate:
    """Serializable merge queue candidate entry."""

    branch: str
    work_item_id: str
    evidence_ids: tuple[str, ...] = ()
    risk_tier: RiskTier = RiskTier.MEDIUM
    dependencies: tuple[str, ...] = ()
    arrival: int = 0
    attempts: int = 0

    def __init__(
        self,
        branch: str,
        work_item_id: str | None = None,
        evidence_ids: Sequence[str] = (),
        risk_tier: RiskTier | str = RiskTier.MEDIUM,
        dependencies: Sequence[str] = (),
        arrival: int = 0,
        attempts: int = 0,
        *,
        work_item: str | None = None,
        evidence: Sequence[str] | None = None,
        risk: RiskTier | str | None = None,
    ) -> None:
        resolved_work_item = work_item if work_item is not None else work_item_id
        if resolved_work_item is None:
            raise TypeError("QueueCandidate requires work_item_id/work_item")

        resolved_evidence = evidence if evidence is not None else evidence_ids
        resolved_risk: RiskTier | str = risk if risk is not None else risk_tier

        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "work_item_id", resolved_work_item)
        object.__setattr__(self, "evidence_ids", tuple(resolved_evidence))
        object.__setattr__(self, "risk_tier", resolved_risk)
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "arrival", arrival)
        object.__setattr__(self, "attempts", attempts)
        self.__post_init__()

    def __post_init__(self) -> None:
        branch = _normalize_identifier(self.branch, field_name="QueueCandidate.branch")
        work_item = _normalize_identifier(
            self.work_item_id,
            field_name="QueueCandidate.work_item_id",
        )
        evidence_ids = _normalize_identifier_tuple(self.evidence_ids)
        dependencies = _normalize_identifier_tuple(self.dependencies)
        if self.arrival < 0:
            raise ValueError("QueueCandidate.arrival must be >= 0")
        if self.attempts < 0:
            raise ValueError("QueueCandidate.attempts must be >= 0")

        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "work_item_id", work_item)
        object.__setattr__(self, "evidence_ids", evidence_ids)
        object.__setattr__(self, "risk_tier", _coerce_risk_tier(self.risk_tier))
        object.__setattr__(self, "dependencies", dependencies)

    @property
    def work_item(self) -> str:
        return self.work_item_id

    @property
    def evidence(self) -> tuple[str, ...]:
        return self.evidence_ids

    @property
    def risk(self) -> RiskTier:
        return self.risk_tier

    def with_arrival(self, arrival: int) -> QueueCandidate:
        return QueueCandidate(
            branch=self.branch,
            work_item_id=self.work_item_id,
            evidence_ids=self.evidence_ids,
            risk_tier=self.risk_tier,
            dependencies=self.dependencies,
            arrival=arrival,
            attempts=self.attempts,
        )

    def with_attempts(self, attempts: int) -> QueueCandidate:
        return QueueCandidate(
            branch=self.branch,
            work_item_id=self.work_item_id,
            evidence_ids=self.evidence_ids,
            risk_tier=self.risk_tier,
            dependencies=self.dependencies,
            arrival=self.arrival,
            attempts=attempts,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "branch": self.branch,
            "work_item": self.work_item_id,
            "work_item_id": self.work_item_id,
            "evidence": list(self.evidence_ids),
            "evidence_ids": list(self.evidence_ids),
            "risk": self.risk_tier.value,
            "risk_tier": self.risk_tier.value,
            "dependencies": list(self.dependencies),
            "arrival": self.arrival,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> QueueCandidate:
        branch = _expect_string(payload.get("branch"), field_name="QueueCandidate.branch")
        work_item_id = _expect_string(
            payload.get("work_item_id", payload.get("work_item")),
            field_name="QueueCandidate.work_item_id",
        )
        evidence_ids = _expect_str_sequence(
            payload.get("evidence_ids", payload.get("evidence", ())),
            field_name="QueueCandidate.evidence_ids",
        )
        dependencies = _expect_str_sequence(
            payload.get("dependencies", ()),
            field_name="QueueCandidate.dependencies",
        )
        risk_raw = payload.get("risk_tier", payload.get("risk", RiskTier.MEDIUM.value))
        risk_tier = _coerce_risk_tier(risk_raw)
        arrival = _expect_int(payload.get("arrival", 0), field_name="QueueCandidate.arrival")
        attempts = _expect_int(payload.get("attempts", 0), field_name="QueueCandidate.attempts")
        return cls(
            branch=branch,
            work_item_id=work_item_id,
            evidence_ids=evidence_ids,
            risk_tier=risk_tier,
            dependencies=dependencies,
            arrival=arrival,
            attempts=attempts,
        )


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result payload for one ``MergeQueue.process_next`` call."""

    status: MergeStatus
    candidate: QueueCandidate | None
    integration_head_before: str | None = None
    integration_head_after: str | None = None
    message: str | None = None
    requeued: bool = False
    conflict_details: tuple[str, ...] = ()
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MergeStatus(self.status))
        object.__setattr__(
            self,
            "conflict_details",
            _normalize_identifier_tuple(self.conflict_details),
        )
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def merged(self) -> bool:
        return self.status == MergeStatus.MERGED


@dataclass(frozen=True, slots=True)
class _GitOperationOutcome:
    success: bool
    message: str | None = None
    conflict_details: tuple[str, ...] = ()
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _CompositionalOutcome:
    passed: bool
    requeue: bool
    message: str | None = None
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)


class MergeQueue:
    """Deterministic file-backed merge queue with serialized processing."""

    def __init__(self, state_file: str | Path, *, integration_branch: str = "integration") -> None:
        integration = _normalize_identifier(
            integration_branch,
            field_name="MergeQueue.integration_branch",
        )
        self._integration_branch = integration
        self._state_file = Path(state_file).expanduser().resolve()
        self._queue: list[QueueCandidate] = []
        self._completed_work_items: set[str] = set()
        self._failed_work_items: set[str] = set()
        self._next_arrival = 0
        self._process_lock = asyncio.Lock()
        self._load_state()

    @classmethod
    def from_json_file(
        cls, state_file: str | Path, *, integration_branch: str = "integration"
    ) -> MergeQueue:
        return cls(state_file, integration_branch=integration_branch)

    @property
    def integration_branch(self) -> str:
        return self._integration_branch

    @property
    def pending_candidates(self) -> tuple[QueueCandidate, ...]:
        ordered = sorted(self._queue, key=lambda item: (item.arrival, item.work_item_id))
        return tuple(ordered)

    @property
    def completed_work_items(self) -> tuple[str, ...]:
        return tuple(sorted(self._completed_work_items))

    @property
    def failed_work_items(self) -> tuple[str, ...]:
        return tuple(sorted(self._failed_work_items))

    @property
    def has_pending(self) -> bool:
        return bool(self._queue)

    def enqueue(self, candidate: QueueCandidate) -> QueueCandidate:
        """Insert one candidate with deterministic arrival assignment."""
        self._ensure_work_item_not_present(candidate.work_item_id)

        assigned_arrival = candidate.arrival
        if assigned_arrival < self._next_arrival:
            assigned_arrival = self._next_arrival

        normalized = candidate.with_arrival(assigned_arrival)
        self._next_arrival = max(self._next_arrival, normalized.arrival + 1)

        self._queue.append(normalized)
        self._persist_state()
        return normalized

    def enqueue_candidate(self, candidate: QueueCandidate) -> QueueCandidate:
        return self.enqueue(candidate)

    async def process_next(
        self,
        git_engine: object,
        compositional_check_callback: Callable[
            ..., bool | Mapping[str, object] | Awaitable[object]
        ],
        persistence_hooks: object | None = None,
    ) -> MergeResult:
        """Process one eligible candidate under a single async merge lock."""
        async with self._process_lock:
            candidate = self._select_next_ready_candidate()
            if candidate is None:
                result = MergeResult(status=MergeStatus.NO_CANDIDATE, candidate=None)
                await self._record_result(persistence_hooks, result)
                return result

            integration_head_before = await self._read_integration_head(git_engine)

            rebase_outcome = await self._perform_rebase(git_engine, candidate)
            if not rebase_outcome.success:
                self._mark_failed(candidate.work_item_id)
                self._persist_state()
                result = MergeResult(
                    status=MergeStatus.FAILED,
                    candidate=candidate,
                    integration_head_before=integration_head_before,
                    integration_head_after=await self._read_integration_head(git_engine),
                    message=rebase_outcome.message or "rebase failed",
                    conflict_details=rebase_outcome.conflict_details,
                    diagnostics=rebase_outcome.diagnostics,
                )
                await self._record_result(persistence_hooks, result)
                return result

            compositional_outcome = await self._run_compositional_checks(
                compositional_check_callback,
                candidate,
                git_engine,
            )
            if not compositional_outcome.passed:
                rollback_performed = await self._rollback_if_needed(
                    git_engine,
                    integration_head_before,
                )
                integration_head_after = await self._read_integration_head(git_engine)
                diagnostics = dict(compositional_outcome.diagnostics)
                diagnostics["rollback_performed"] = rollback_performed
                diagnostics["bisect_recommended"] = bool(
                    integration_head_before is not None
                    and integration_head_after is not None
                    and integration_head_after != integration_head_before
                    and not rollback_performed
                )

                if compositional_outcome.requeue:
                    requeued_candidate = self._requeue(candidate)
                    self._persist_state()
                    result = MergeResult(
                        status=MergeStatus.REQUEUED,
                        candidate=requeued_candidate,
                        integration_head_before=integration_head_before,
                        integration_head_after=integration_head_after,
                        message=compositional_outcome.message or "compositional checks failed",
                        requeued=True,
                        diagnostics=diagnostics,
                    )
                else:
                    self._mark_failed(candidate.work_item_id)
                    self._persist_state()
                    result = MergeResult(
                        status=MergeStatus.FAILED,
                        candidate=candidate,
                        integration_head_before=integration_head_before,
                        integration_head_after=integration_head_after,
                        message=compositional_outcome.message or "compositional checks failed",
                        diagnostics=diagnostics,
                    )

                await self._record_result(persistence_hooks, result)
                return result

            merge_outcome = await self._perform_fast_forward_merge(git_engine, candidate)
            if not merge_outcome.success:
                rollback_performed = await self._rollback_if_needed(
                    git_engine,
                    integration_head_before,
                )
                integration_head_after = await self._read_integration_head(git_engine)
                diagnostics = dict(merge_outcome.diagnostics)
                diagnostics["rollback_performed"] = rollback_performed
                diagnostics["bisect_recommended"] = bool(
                    integration_head_before is not None
                    and integration_head_after is not None
                    and integration_head_after != integration_head_before
                    and not rollback_performed
                )

                self._mark_failed(candidate.work_item_id)
                self._persist_state()
                result = MergeResult(
                    status=MergeStatus.FAILED,
                    candidate=candidate,
                    integration_head_before=integration_head_before,
                    integration_head_after=integration_head_after,
                    message=merge_outcome.message or "fast-forward merge failed",
                    conflict_details=merge_outcome.conflict_details,
                    diagnostics=diagnostics,
                )
                await self._record_result(persistence_hooks, result)
                return result

            self._mark_merged(candidate.work_item_id)
            integration_head_after = await self._read_integration_head(git_engine)
            self._persist_state()

            result = MergeResult(
                status=MergeStatus.MERGED,
                candidate=candidate,
                integration_head_before=integration_head_before,
                integration_head_after=integration_head_after,
                message="merge completed",
            )
            await self._record_result(persistence_hooks, result)
            return result

    def _ensure_work_item_not_present(self, work_item_id: str) -> None:
        if work_item_id in self._completed_work_items:
            raise ValueError(f"work item already merged: {work_item_id}")
        if work_item_id in self._failed_work_items:
            raise ValueError(f"work item already failed: {work_item_id}")
        for candidate in self._queue:
            if candidate.work_item_id == work_item_id:
                raise ValueError(f"work item already queued: {work_item_id}")

    def _select_next_ready_candidate(self) -> QueueCandidate | None:
        ready = [candidate for candidate in self._queue if self._deps_satisfied(candidate)]
        if not ready:
            return None
        return min(
            ready,
            key=lambda item: (_RISK_ORDER[item.risk_tier], item.arrival, item.work_item_id),
        )

    def _deps_satisfied(self, candidate: QueueCandidate) -> bool:
        return all(dep in self._completed_work_items for dep in candidate.dependencies)

    def _remove_candidate(self, work_item_id: str) -> QueueCandidate:
        for index, candidate in enumerate(self._queue):
            if candidate.work_item_id == work_item_id:
                return self._queue.pop(index)
        raise KeyError(f"candidate not found: {work_item_id}")

    def _mark_merged(self, work_item_id: str) -> None:
        self._remove_candidate(work_item_id)
        self._completed_work_items.add(work_item_id)

    def _mark_failed(self, work_item_id: str) -> None:
        self._remove_candidate(work_item_id)
        self._failed_work_items.add(work_item_id)

    def _requeue(self, candidate: QueueCandidate) -> QueueCandidate:
        removed = self._remove_candidate(candidate.work_item_id)
        requeued = removed.with_attempts(removed.attempts + 1).with_arrival(self._next_arrival)
        self._next_arrival += 1
        self._queue.append(requeued)
        return requeued

    async def _perform_rebase(
        self,
        git_engine: object,
        candidate: QueueCandidate,
    ) -> _GitOperationOutcome:
        try:
            raw_result = await _invoke_supported_method(
                git_engine,
                method_names=("rebase_onto_integration", "rebase_branch", "rebase"),
                arg_options=(
                    (candidate.branch, self._integration_branch),
                    (candidate.branch,),
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return _failure_from_exception("rebase failed", exc)
        return _normalize_git_outcome(raw_result)

    async def _perform_fast_forward_merge(
        self,
        git_engine: object,
        candidate: QueueCandidate,
    ) -> _GitOperationOutcome:
        try:
            raw_result = await _invoke_supported_method(
                git_engine,
                method_names=(
                    "merge_ff_only",
                    "fast_forward_merge",
                    "ff_merge",
                    "merge_fast_forward",
                    "merge",
                ),
                arg_options=(
                    (candidate.branch, self._integration_branch),
                    (candidate.branch,),
                ),
            )
        except Exception as exc:  # pragma: no cover - defensive path
            return _failure_from_exception("fast-forward merge failed", exc)
        return _normalize_git_outcome(raw_result)

    async def _run_compositional_checks(
        self,
        callback: Callable[..., bool | Mapping[str, object] | Awaitable[object]],
        candidate: QueueCandidate,
        git_engine: object,
    ) -> _CompositionalOutcome:
        args = _select_callback_args(callback, candidate, git_engine)
        try:
            raw_outcome = callback(*args)
            resolved = await _maybe_await(raw_outcome)
        except Exception as exc:  # pragma: no cover - defensive path
            return _CompositionalOutcome(
                passed=False,
                requeue=True,
                message=f"compositional check raised: {exc}",
                diagnostics={"exception": str(exc)},
            )
        return _normalize_compositional_outcome(resolved)

    async def _read_integration_head(self, git_engine: object) -> str | None:
        try:
            raw_head = await _invoke_supported_method(
                git_engine,
                method_names=(
                    "get_integration_head",
                    "get_branch_head",
                    "get_head",
                    "head",
                    "rev_parse",
                ),
                arg_options=((self._integration_branch,), ()),
            )
        except AttributeError:
            return None
        except Exception:
            return None

        if not isinstance(raw_head, str):
            return None
        normalized = raw_head.strip()
        return normalized or None

    async def _rollback_if_needed(self, git_engine: object, before_head: str | None) -> bool:
        if before_head is None:
            return False

        after_head = await self._read_integration_head(git_engine)
        if after_head is None or after_head == before_head:
            return False

        try:
            await _invoke_supported_method(
                git_engine,
                method_names=(
                    "reset_integration",
                    "reset_branch_hard",
                    "reset_hard",
                    "set_head",
                    "checkout_commit",
                ),
                arg_options=(
                    (self._integration_branch, before_head),
                    (before_head,),
                ),
            )
        except Exception:
            return False

        restored_head = await self._read_integration_head(git_engine)
        return restored_head == before_head

    async def _record_result(self, persistence_hooks: object | None, result: MergeResult) -> None:
        if persistence_hooks is None:
            return

        if callable(persistence_hooks):
            await _maybe_await(persistence_hooks(result))
            return

        hook_candidate = _resolve_hook_callable(
            persistence_hooks,
            names=("record_merge_result", "record", "on_merge_result", "on_result"),
        )
        if hook_candidate is not None:
            await _maybe_await(hook_candidate(result))

    def _load_state(self) -> None:
        if not self._state_file.exists():
            return

        try:
            payload_raw = self._state_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise MergeQueueStateError(
                f"failed to read MergeQueue state file {self._state_file}: {exc}"
            ) from exc

        try:
            parsed = json.loads(payload_raw)
        except json.JSONDecodeError as exc:
            raise MergeQueueStateError(
                f"invalid JSON in MergeQueue state file {self._state_file}: {exc.msg}"
            ) from exc

        try:
            self._apply_state_payload(parsed)
        except MergeQueueStateError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise MergeQueueStateError(
                f"invalid MergeQueue state file {self._state_file}: {exc}"
            ) from exc

    def _apply_state_payload(self, parsed: object) -> None:
        if not isinstance(parsed, dict):
            raise MergeQueueStateError("MergeQueue state file root must be an object")

        missing_keys = [key for key in _REQUIRED_STATE_KEYS if key not in parsed]
        if missing_keys:
            missing_summary = ", ".join(missing_keys)
            raise MergeQueueStateError(
                f"MergeQueue state file missing required keys: {missing_summary}"
            )

        schema_version = _expect_int(parsed["schema_version"], field_name="schema_version")
        if schema_version != _STATE_SCHEMA_VERSION:
            raise MergeQueueStateError(
                "unsupported MergeQueue state schema version "
                f"{schema_version}; expected {_STATE_SCHEMA_VERSION}"
            )

        queue_raw = parsed["queue"]
        if not isinstance(queue_raw, list):
            raise MergeQueueStateError("MergeQueue state file queue must be a list")

        completed_raw = parsed["completed_work_items"]
        failed_raw = parsed["failed_work_items"]

        loaded_candidates: list[QueueCandidate] = []
        seen_work_items: set[str] = set()

        for item in queue_raw:
            if not isinstance(item, dict):
                raise MergeQueueStateError("MergeQueue state file queue entries must be objects")
            candidate = QueueCandidate.from_dict(item)
            if candidate.work_item_id in seen_work_items:
                raise MergeQueueStateError(
                    f"MergeQueue state file contains duplicate work item: {candidate.work_item_id}"
                )
            seen_work_items.add(candidate.work_item_id)
            loaded_candidates.append(candidate)

        completed = set(_expect_str_sequence(completed_raw, field_name="completed_work_items"))
        failed = set(_expect_str_sequence(failed_raw, field_name="failed_work_items"))

        overlap = completed & failed
        if overlap:
            overlap_sorted = ", ".join(sorted(overlap))
            raise MergeQueueStateError(
                f"work items cannot be both completed and failed: {overlap_sorted}"
            )

        for candidate in loaded_candidates:
            if candidate.work_item_id in completed or candidate.work_item_id in failed:
                raise MergeQueueStateError(
                    "queued work item cannot also be terminal in persisted state: "
                    f"{candidate.work_item_id}"
                )

        next_arrival = _expect_int(parsed["next_arrival"], field_name="next_arrival")
        max_arrival = max((candidate.arrival for candidate in loaded_candidates), default=-1)

        self._queue = sorted(loaded_candidates, key=lambda item: (item.arrival, item.work_item_id))
        self._completed_work_items = completed
        self._failed_work_items = failed
        self._next_arrival = max(next_arrival, max_arrival + 1)

    def _persist_state(self) -> None:
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "next_arrival": self._next_arrival,
            "completed_work_items": sorted(self._completed_work_items),
            "failed_work_items": sorted(self._failed_work_items),
            "queue": [
                candidate.to_dict()
                for candidate in sorted(
                    self._queue, key=lambda item: (item.arrival, item.work_item_id)
                )
            ],
        }

        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_state_file = self._state_file.with_suffix(f"{self._state_file.suffix}.tmp")
        tmp_state_file.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_state_file.replace(self._state_file)


def _normalize_compositional_outcome(raw_outcome: object) -> _CompositionalOutcome:
    if isinstance(raw_outcome, bool):
        return _CompositionalOutcome(
            passed=raw_outcome,
            requeue=not raw_outcome,
            message=None if raw_outcome else "compositional checks failed",
            diagnostics={},
        )

    if isinstance(raw_outcome, Mapping):
        passed = _mapping_truth(
            raw_outcome, keys=("passed", "success", "ok", "status"), default=False
        )
        requeue_default = not passed
        requeue = _mapping_truth(raw_outcome, keys=("requeue",), default=requeue_default)
        message = _mapping_optional_text(raw_outcome, keys=("message", "summary", "details"))

        diagnostics_value = raw_outcome.get("diagnostics")
        diagnostics: dict[str, JSONValue]
        if isinstance(diagnostics_value, Mapping):
            diagnostics = _coerce_mapping_to_json_object(diagnostics_value)
        else:
            diagnostics = {
                key: _coerce_json_value(value)
                for key, value in raw_outcome.items()
                if key
                not in {
                    "passed",
                    "success",
                    "ok",
                    "status",
                    "requeue",
                    "message",
                    "summary",
                    "details",
                }
            }

        return _CompositionalOutcome(
            passed=passed,
            requeue=requeue,
            message=message,
            diagnostics=diagnostics,
        )

    return _CompositionalOutcome(
        passed=bool(raw_outcome),
        requeue=not bool(raw_outcome),
        message=None if raw_outcome else "compositional checks failed",
        diagnostics={},
    )


def _normalize_git_outcome(raw_outcome: object) -> _GitOperationOutcome:
    if isinstance(raw_outcome, bool):
        return _GitOperationOutcome(success=raw_outcome)

    if isinstance(raw_outcome, Mapping):
        conflict_details = _extract_conflict_details(raw_outcome)
        success_default = (
            not conflict_details and "error" not in raw_outcome and "exception" not in raw_outcome
        )
        success = _mapping_truth(
            raw_outcome,
            keys=("success", "ok", "passed", "status"),
            default=success_default,
        )
        message = _mapping_optional_text(raw_outcome, keys=("message", "reason", "error"))
        diagnostics = {
            key: _coerce_json_value(value)
            for key, value in raw_outcome.items()
            if key
            not in {
                "success",
                "ok",
                "passed",
                "status",
                "message",
                "reason",
                "error",
                "conflicts",
                "conflict_details",
                "conflict_files",
                "files",
            }
        }
        return _GitOperationOutcome(
            success=success,
            message=message,
            conflict_details=conflict_details,
            diagnostics=diagnostics,
        )

    return _GitOperationOutcome(success=bool(raw_outcome))


def _failure_from_exception(message: str, exc: Exception) -> _GitOperationOutcome:
    conflict_details = _extract_conflict_details(exc)
    diagnostics: dict[str, JSONValue] = {"exception": str(exc)}
    return _GitOperationOutcome(
        success=False,
        message=f"{message}: {exc}",
        conflict_details=conflict_details,
        diagnostics=diagnostics,
    )


async def _invoke_supported_method(
    target: object,
    *,
    method_names: Sequence[str],
    arg_options: Sequence[tuple[object, ...]],
) -> object:
    for method_name in method_names:
        method = getattr(target, method_name, None)
        if method is None or not callable(method):
            continue

        signature = _safe_signature(method)

        for args in arg_options:
            if signature is not None:
                try:
                    signature.bind(*args)
                except TypeError:
                    continue

            try:
                result = method(*args)
            except TypeError:
                if signature is None:
                    continue
                raise
            return await _maybe_await(result)

    joined = ", ".join(method_names)
    raise AttributeError(f"target does not expose a supported method; expected one of: {joined}")


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _safe_signature(callable_obj: Callable[..., object]) -> inspect.Signature | None:
    try:
        return inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None


def _select_callback_args(
    callback: Callable[..., object],
    candidate: QueueCandidate,
    git_engine: object,
) -> tuple[object, ...]:
    signature = _safe_signature(callback)
    if signature is None:
        return (candidate, git_engine)

    if _can_bind(signature, candidate, git_engine):
        return (candidate, git_engine)
    if _can_bind(signature, candidate):
        return (candidate,)
    if _can_bind(signature):
        return ()

    # Fallback path: prefer candidate-only to avoid accidental extra args.
    return (candidate,)


def _can_bind(signature: inspect.Signature, *args: object) -> bool:
    try:
        signature.bind(*args)
    except TypeError:
        return False
    return True


def _resolve_hook_callable(target: object, *, names: Sequence[str]) -> Callable[..., object] | None:
    if isinstance(target, Mapping):
        for name in names:
            candidate = target.get(name)
            if callable(candidate):
                return cast("Callable[..., object]", candidate)
        return None

    for name in names:
        candidate = getattr(target, name, None)
        if callable(candidate):
            return cast("Callable[..., object]", candidate)
    return None


def _extract_conflict_details(source: object) -> tuple[str, ...]:
    if isinstance(source, Mapping):
        for key in ("conflicts", "conflict_details", "conflict_files", "files"):
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return _normalize_identifier_tuple(value)
            if isinstance(value, str):
                return _normalize_identifier_tuple((value,))

    for attr in ("conflicts", "conflict_details", "conflict_files", "files"):
        value = getattr(source, attr, None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return _normalize_identifier_tuple(value)
        if isinstance(value, str):
            return _normalize_identifier_tuple((value,))

    return ()


def _coerce_risk_tier(value: object) -> RiskTier:
    if isinstance(value, RiskTier):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return RiskTier(normalized)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in RiskTier)
            raise ValueError(f"unsupported risk tier {value!r}; allowed: {allowed}") from exc
    raise ValueError(f"risk tier must be RiskTier or string, got {type(value).__name__}")


def _normalize_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _normalize_identifier_tuple(values: Sequence[object]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _expect_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _expect_str_sequence(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    return _normalize_identifier_tuple(value)


def _expect_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


def _mapping_truth(mapping: Mapping[str, object], *, keys: Sequence[str], default: bool) -> bool:
    for key in keys:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"pass", "passed", "true", "ok", "success", "yes"}:
                return True
            if normalized in {"fail", "failed", "false", "error", "no"}:
                return False
        return bool(value)
    return default


def _mapping_optional_text(mapping: Mapping[str, object], *, keys: Sequence[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return str(value)
    return None


def _coerce_mapping_to_json_object(value: Mapping[object, object]) -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        out[key] = _coerce_json_value(item)
    return out


def _coerce_json_value(value: object, *, depth: int = 0) -> JSONValue:
    if depth > 8:
        return str(value)

    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _coerce_json_value(item, depth=depth + 1) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_json_value(item, depth=depth + 1) for item in value]

    return str(value)


__all__ = [
    "MergeQueue",
    "MergeResult",
    "MergeQueueStateError",
    "MergeStatus",
    "QueueCandidate",
]
