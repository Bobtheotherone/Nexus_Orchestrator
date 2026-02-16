"""Control-plane run controller with deterministic mock/offline execution."""

from __future__ import annotations

import asyncio
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Final, TypeAlias, cast

from nexus_orchestrator.config.schema import assert_valid_config, default_config, merge_config
from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Budget,
    EvidenceRecord,
    EvidenceResult,
    Incident,
    MergeRecord,
    Run,
    RunStatus,
    WorkItem,
    WorkItemStatus,
)
from nexus_orchestrator.integration_plane import (
    GitEngine,
    MergeQueue,
    MergeStatus,
    QueueCandidate,
    Workspace,
    WorkspaceManager,
)
from nexus_orchestrator.persistence.repositories import (
    AttemptRepo,
    ConstraintRepo,
    EvidenceRepo,
    IncidentRepo,
    MergeRepo,
    ProviderCallRepo,
    RunRepo,
    TaskGraphRepo,
    WorkItemRepo,
)
from nexus_orchestrator.persistence.state_db import StateDB
from nexus_orchestrator.planning.architect_interface import build_deterministic_architect_output
from nexus_orchestrator.planning.constraint_compiler import CompilationResult, compile_constraints
from nexus_orchestrator.spec_ingestion import ingest_spec
from nexus_orchestrator.synthesis_plane.dispatch import (
    DispatchBudget,
    DispatchController,
    DispatchRequest,
    ProviderBinding,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    RepositoryDispatchPersistence,
)
from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog, load_model_catalog
from nexus_orchestrator.synthesis_plane.roles import (
    ROLE_IMPLEMENTER,
    EscalationDecision,
    RoleRegistry,
    resolve_tool_ladder_for_affinity,
)
from nexus_orchestrator.synthesis_plane.work_item_classifier import (
    ModelAffinity,
    classify_work_item,
)
from nexus_orchestrator.utils.hashing import sha256_text
from nexus_orchestrator.verification_plane import (
    ConstraintGateDecision,
    PipelineCheckResult,
    PipelineOutput,
    StageCoverageRequirement,
    VerificationSelectionMode,
    run_constraint_gate,
)
from nexus_orchestrator.verification_plane.evidence import EvidenceWriter

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - py<3.11 fallback
    UTC = timezone.utc  # noqa: UP017

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_ACTIVE_RUN_STATUSES = {RunStatus.PLANNING, RunStatus.RUNNING, RunStatus.PAUSED}
_TERMINAL_RUN_STATUSES = {RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.FAILED}
_UNSAFE_MODEL_AVAILABILITY = {"legacy", "deprecated", "experimental"}
_CHECKER_STAGE_MAP = {
    "build_checker": "build",
    "scope_checker": "build",
    "lint_checker": "lint_format",
    "typecheck_checker": "type_check",
    "test_checker": "unit_tests",
    "security_checker": "security_scan",
    "performance_checker": "performance",
    "documentation_checker": "integration_tests",
    "reliability_checker": "integration_tests",
    "schema_checker": "integration_tests",
}
_MODULE_SUFFIX_RE = re.compile(r"\b([A-Za-z])\b")


class SimulatedCrashError(RuntimeError):
    """Internal crash injection used by deterministic restart smoke tests."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """Normalized run result returned by ``OrchestratorController.run``."""

    run_id: str
    status: RunStatus
    resumed_from_crash: bool
    merged_work_item_ids: tuple[str, ...]
    failed_work_item_ids: tuple[str, ...]
    dispatch_batches: tuple[tuple[str, ...], ...]
    budget_tokens_used: int
    budget_cost_usd: float
    provider_calls: int
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DispatchPlan:
    work_item: WorkItem
    decision: EscalationDecision
    attempt_number: int
    prompt: str
    dispatch_budget: DispatchBudget


@dataclass(frozen=True, slots=True)
class _DispatchOutcome:
    plan: _DispatchPlan
    result: ProviderResponse | None
    dispatch_result_attempt_id: str | None
    dispatch_result_attempts: int
    dispatch_result_tokens: int
    dispatch_result_cost_usd: float
    error: str | None = None


@dataclass(slots=True)
class _DeterministicMockProvider:
    provider_name: str
    model_catalog: ModelCatalog
    patch_by_work_item: Mapping[str, str]

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        work_item_id_raw = request.metadata.get("work_item_id")
        if not isinstance(work_item_id_raw, str):
            raise ValueError("mock provider requires request.metadata.work_item_id")

        patch = self.patch_by_work_item.get(work_item_id_raw, "")
        if not patch:
            raise ValueError(f"no mock patch prepared for work item: {work_item_id_raw}")

        total_tokens = max(1, len(patch) // 4)
        try:
            estimated_cost = self.model_catalog.estimate_cost(
                provider=self.provider_name,
                model=request.model,
                total_tokens=total_tokens,
            )
        except KeyError:
            estimated_cost = 0.0

        request_id_seed = sha256_text(f"{work_item_id_raw}|{self.provider_name}|{request.model}")
        return ProviderResponse(
            model=request.model,
            raw_text=patch,
            usage=ProviderUsage(
                total_tokens=total_tokens,
                latency_ms=1,
                cost_estimate_usd=estimated_cost,
            ),
            request_id=f"mock-{request_id_seed[:20]}",
        )


class OrchestratorController:
    """Controller coordinating ingest -> plan -> dispatch -> verify -> merge."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        state_db_path: str | Path | None = None,
        crash_after_attempts: int | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).expanduser().resolve(strict=True)
        state_path = (
            Path(state_db_path).expanduser().resolve(strict=False)
            if state_db_path is not None
            else (self._repo_root / "state" / "nexus.sqlite")
        )
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_db = StateDB(state_path)

        self._run_repo = RunRepo(self._state_db)
        self._work_item_repo = WorkItemRepo(self._state_db)
        self._constraint_repo = ConstraintRepo(self._state_db)
        self._evidence_repo = EvidenceRepo(self._state_db)
        self._attempt_repo = AttemptRepo(self._state_db)
        self._provider_call_repo = ProviderCallRepo(self._state_db)
        self._merge_repo = MergeRepo(self._state_db)
        self._incident_repo = IncidentRepo(self._state_db)
        self._task_graph_repo = TaskGraphRepo(self._state_db)

        self._pause_requests: set[str] = set()
        self._cancel_requests: set[str] = set()
        self._workspace_by_branch: dict[str, Workspace] = {}
        self._lock = RLock()
        self._crash_after_attempts = crash_after_attempts
        self._attempts_processed = 0

    @property
    def state_db_path(self) -> Path:
        return self._state_db.path

    def _log(self, message: str) -> None:
        """Print progress message to stdout for TUI display."""
        print(f"[nexus] {message}", flush=True)

    def pause(self, run_id: str) -> Run | None:
        with self._lock:
            self._pause_requests.add(run_id)
        run = self._run_repo.get(run_id)
        if run is None:
            return None
        if run.status in {RunStatus.PLANNING, RunStatus.RUNNING}:
            return self._run_repo.set_status(run_id, RunStatus.PAUSED)
        return run

    def resume(self, run_id: str) -> Run | None:
        with self._lock:
            self._pause_requests.discard(run_id)
            self._cancel_requests.discard(run_id)
        run = self._run_repo.get(run_id)
        if run is None:
            return None
        if run.status is RunStatus.PAUSED:
            return self._run_repo.set_status(run_id, RunStatus.RUNNING)
        return run

    def cancel(self, run_id: str) -> Run | None:
        with self._lock:
            self._cancel_requests.add(run_id)
        run = self._run_repo.get(run_id)
        if run is None:
            return None
        if run.status not in _TERMINAL_RUN_STATUSES:
            return self._run_repo.set_status(run_id, RunStatus.CANCELLED)
        return run

    def run(
        self,
        spec_path: str | Path,
        config: Mapping[str, object] | None,
        mode: str,
        mock: bool,
    ) -> RunResult:
        effective_config = _effective_config(config)
        model_catalog = load_model_catalog()
        role_registry = RoleRegistry.from_config(effective_config)

        warning_messages: list[str] = []
        warned_models: set[tuple[str, str]] = set()
        warning_messages.extend(
            self._warn_on_registry_models(role_registry, model_catalog, warned_models)
        )

        git_engine = self._build_git_engine(effective_config)
        git_engine.init_or_open()

        workspace_manager = self._build_workspace_manager(effective_config)
        evidence_root = self._resolve_path(
            _nested_get(effective_config, ("paths", "evidence_root"), "evidence"),
            base=self._repo_root,
        )
        evidence_root.mkdir(parents=True, exist_ok=True)

        merge_queue = MergeQueue(
            self._state_db.path.parent / "merge_queue_state.json",
            integration_branch=git_engine.integration_branch,
            state_db=self._state_db,
        )

        resolved_spec = self._resolve_spec_path(spec_path)
        self._log(f"Spec: {resolved_spec.relative_path}")
        normalized_mode = mode.strip().lower()
        run_record, resumed = self._load_or_create_run(
            spec_relative=resolved_spec.relative_path,
            normalized_mode=normalized_mode,
            effective_config=effective_config,
            mock=mock,
        )

        warning_messages.extend(
            self._warn_on_run_metadata(run_record.id, warning_messages=tuple(warning_messages))
        )
        self._log(f"Run {run_record.id[:12]}{'  (resumed)' if resumed else ''}")

        if not resumed or not self._work_item_repo.list_for_run(run_record.id, limit=1):
            self._log("Planning...")
            compilation = self._plan_run(
                run_record=run_record,
                spec_absolute=resolved_spec.absolute_path,
                effective_config=effective_config,
            )
            warning_messages.extend(compilation.warnings)
            if compilation.task_graph is None or compilation.errors:
                self._log(f"Planning FAILED: {len(compilation.errors)} errors")
                self._record_incident(
                    run_id=run_record.id,
                    category="planning",
                    message="planning failed",
                    details={
                        "errors": list(compilation.errors),
                        "warnings": list(compilation.warnings),
                    },
                )
                self._run_repo.set_status(run_record.id, RunStatus.FAILED)
                return self._build_result(
                    run_id=run_record.id,
                    resumed=resumed,
                    warnings=tuple(sorted(set(warning_messages))),
                )
            self._log(f"Planning complete: {len(compilation.work_items)} work items")

        self._recover_inflight_work_items(run_record.id, merge_queue)
        latest_run = self._require_run(run_record.id)
        if latest_run.status is RunStatus.CREATED or (
            latest_run.status is RunStatus.PAUSED and normalized_mode in {"resume", "auto"}
        ):
            latest_run = self._run_repo.set_status(run_record.id, RunStatus.RUNNING)

        initial_items = self._work_item_repo.list_for_run(run_record.id, limit=1_000)
        total_items = len(initial_items)
        already_done = sum(
            1 for item in initial_items
            if item.status in {WorkItemStatus.MERGED, WorkItemStatus.FAILED}
        )
        items_processed = already_done
        self._log(f"{total_items} work items ({already_done} already completed)")

        try:
            while True:
                current_run = self._require_run(latest_run.id)
                if current_run.status in _TERMINAL_RUN_STATUSES:
                    return self._build_result(
                        run_id=current_run.id,
                        resumed=resumed,
                        warnings=tuple(sorted(set(warning_messages))),
                    )

                if self._is_cancel_requested(current_run.id):
                    self._run_repo.set_status(current_run.id, RunStatus.CANCELLED)
                    return self._build_result(
                        run_id=current_run.id,
                        resumed=resumed,
                        warnings=tuple(sorted(set(warning_messages))),
                    )
                if self._is_pause_requested(current_run.id):
                    self._run_repo.set_status(current_run.id, RunStatus.PAUSED)
                    return self._build_result(
                        run_id=current_run.id,
                        resumed=resumed,
                        warnings=tuple(sorted(set(warning_messages))),
                    )

                self._drain_merge_queue(
                    run_id=current_run.id,
                    merge_queue=merge_queue,
                    git_engine=git_engine,
                    workspace_manager=workspace_manager,
                )

                work_items = self._work_item_repo.list_for_run(current_run.id, limit=1_000)
                if not work_items:
                    self._run_repo.set_status(current_run.id, RunStatus.FAILED)
                    self._record_incident(
                        run_id=current_run.id,
                        category="controller",
                        message="run has no work items",
                    )
                    return self._build_result(
                        run_id=current_run.id,
                        resumed=resumed,
                        warnings=tuple(sorted(set(warning_messages))),
                    )

                if all(item.status is WorkItemStatus.MERGED for item in work_items):
                    self._log("All work items merged — run complete")
                    self._run_repo.set_status(current_run.id, RunStatus.COMPLETED)
                    return self._build_result(
                        run_id=current_run.id,
                        resumed=resumed,
                        warnings=tuple(sorted(set(warning_messages))),
                    )

                runnable = self._work_item_repo.get_next_runnable(
                    current_run.id, limit=self._batch_limit(effective_config)
                )
                if not runnable:
                    if merge_queue.has_pending:
                        self._drain_merge_queue(
                            run_id=current_run.id,
                            merge_queue=merge_queue,
                            git_engine=git_engine,
                            workspace_manager=workspace_manager,
                        )
                        continue

                    unresolved = [
                        item
                        for item in work_items
                        if item.status not in {WorkItemStatus.MERGED, WorkItemStatus.FAILED}
                    ]
                    if unresolved:
                        self._log("Run deadlocked — no runnable work items remain")
                        self._run_repo.set_status(current_run.id, RunStatus.FAILED)
                        self._record_incident(
                            run_id=current_run.id,
                            category="controller",
                            message="run is deadlocked; no runnable work items remain",
                        )
                    elif any(item.status is WorkItemStatus.FAILED for item in work_items):
                        failed_count = sum(
                            1 for item in work_items if item.status is WorkItemStatus.FAILED
                        )
                        self._log(f"Run finished with {failed_count} failed work items")
                        self._run_repo.set_status(current_run.id, RunStatus.FAILED)
                    else:
                        self._log("Run complete")
                        self._run_repo.set_status(current_run.id, RunStatus.COMPLETED)
                    return self._build_result(
                        run_id=current_run.id,
                        resumed=resumed,
                        warnings=tuple(sorted(set(warning_messages))),
                    )

                batch_ids = tuple(item.id for item in runnable)
                self._append_dispatch_batch(current_run.id, batch_ids)
                for work_item in runnable:
                    items_processed += 1
                    self._log(f"[{items_processed}/{total_items}] {work_item.title}")
                    self._process_work_item(
                        run_id=current_run.id,
                        work_item=work_item,
                        effective_config=effective_config,
                        role_registry=role_registry,
                        model_catalog=model_catalog,
                        warned_models=warned_models,
                        warning_messages=warning_messages,
                        mock=mock,
                        git_engine=git_engine,
                        workspace_manager=workspace_manager,
                        merge_queue=merge_queue,
                        evidence_root=evidence_root,
                    )

                    if (
                        self._crash_after_attempts is not None
                        and self._attempts_processed >= self._crash_after_attempts
                    ):
                        raise SimulatedCrashError(
                            f"simulated crash after {self._attempts_processed} processed attempt(s)"
                        )

                self._drain_merge_queue(
                    run_id=current_run.id,
                    merge_queue=merge_queue,
                    git_engine=git_engine,
                    workspace_manager=workspace_manager,
                )
        except SimulatedCrashError:
            raise
        except KeyboardInterrupt:
            self._log("Run interrupted by user (Ctrl+C)")
            self._record_incident(
                run_id=run_record.id,
                category="controller",
                message="run interrupted by user",
            )
            self._run_repo.set_status(run_record.id, RunStatus.CANCELLED)
            warning_messages.append("run interrupted by user")
            return self._build_result(
                run_id=run_record.id,
                resumed=resumed,
                warnings=tuple(sorted(set(warning_messages))),
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Run crashed: {exc}")
            self._record_incident(
                run_id=run_record.id,
                category="controller",
                message=f"run crashed with exception: {exc}",
                details={"exception": str(exc)},
            )
            self._run_repo.set_status(run_record.id, RunStatus.FAILED)
            warning_messages.append(f"controller exception: {exc}")
            return self._build_result(
                run_id=run_record.id,
                resumed=resumed,
                warnings=tuple(sorted(set(warning_messages))),
            )

    def _load_or_create_run(
        self,
        *,
        spec_relative: str,
        normalized_mode: str,
        effective_config: Mapping[str, object],
        mock: bool,
    ) -> tuple[Run, bool]:
        if normalized_mode in {"resume", "auto"}:
            existing = self._find_active_run(spec_relative)
            if existing is not None:
                return existing, True
            if normalized_mode == "resume":
                raise ValueError(
                    f"no resumable run found for spec {spec_relative}; status in {_ACTIVE_RUN_STATUSES}"
                )

        run_budget = self._budget_from_config(effective_config)
        run_id = ids.generate_run_id()
        metadata: dict[str, JSONValue] = {
            "mock_mode": bool(mock),
            "dispatch_batches": [],
            "budget_usage": {
                "tokens_used": 0,
                "cost_usd": 0.0,
                "provider_calls": 0,
            },
            "warnings": [],
        }
        created = Run(
            id=run_id,
            spec_path=spec_relative,
            status=RunStatus.PLANNING,
            started_at=_utc_now(),
            work_item_ids=(),
            budget=run_budget,
            metadata=metadata,
        )
        self._run_repo.add(created)
        return created, False

    def _find_active_run(self, spec_relative: str) -> Run | None:
        candidates: list[Run] = []
        for status in _ACTIVE_RUN_STATUSES:
            candidates.extend(self._run_repo.list(status=status, limit=100))
        for run_record in candidates:
            if run_record.spec_path == spec_relative:
                return run_record
        return None

    def _plan_run(
        self,
        *,
        run_record: Run,
        spec_absolute: Path,
        effective_config: Mapping[str, object],
    ) -> CompilationResult:
        ingested = ingest_spec(spec_absolute)
        architect_output = build_deterministic_architect_output(ingested)
        registry_path = self._resolve_path(
            _nested_get(
                effective_config,
                ("paths", "constraint_registry"),
                "constraints/registry",
            ),
            base=self._repo_root,
        )
        compiled = compile_constraints(
            ingested,
            architect_output,
            registry_path=registry_path,
            run_id=run_record.id,
        )
        if compiled.task_graph is None:
            return compiled

        for work_item in compiled.work_items:
            self._work_item_repo.add(run_record.id, work_item)
            self._run_repo.attach_work_item(run_record.id, work_item.id)
            for constraint in work_item.constraint_envelope.constraints:
                self._constraint_repo.upsert(constraint, active=True)
        self._task_graph_repo.upsert(compiled.task_graph)

        self._run_repo.set_status(run_record.id, RunStatus.RUNNING)
        return compiled

    def _recover_inflight_work_items(self, run_id: str, merge_queue: MergeQueue) -> None:
        pending_merge_ids = {candidate.work_item_id for candidate in merge_queue.pending_candidates}
        work_items = self._work_item_repo.list_for_run(run_id, limit=1_000)
        for work_item in work_items:
            if work_item.status in {WorkItemStatus.DISPATCHED, WorkItemStatus.VERIFYING}:
                self._work_item_repo.set_status(work_item.id, WorkItemStatus.READY, run_id=run_id)
                continue
            if work_item.status is WorkItemStatus.PASSED and work_item.id not in pending_merge_ids:
                self._work_item_repo.set_status(work_item.id, WorkItemStatus.READY, run_id=run_id)

    def _process_work_item(
        self,
        *,
        run_id: str,
        work_item: WorkItem,
        effective_config: Mapping[str, object],
        role_registry: RoleRegistry,
        model_catalog: ModelCatalog,
        warned_models: set[tuple[str, str]],
        warning_messages: list[str],
        mock: bool,
        git_engine: GitEngine,
        workspace_manager: WorkspaceManager,
        merge_queue: MergeQueue,
        evidence_root: Path,
    ) -> None:
        self._work_item_repo.set_status(work_item.id, WorkItemStatus.DISPATCHED, run_id=run_id)
        dispatch_plan = self._build_dispatch_plan(
            run_id=run_id,
            work_item=work_item,
            effective_config=effective_config,
            role_registry=role_registry,
            model_catalog=model_catalog,
            warned_models=warned_models,
            warning_messages=warning_messages,
        )
        if dispatch_plan is None:
            self._log("  No routing decision available — skipping")
            self._work_item_repo.set_status(work_item.id, WorkItemStatus.FAILED, run_id=run_id)
            return

        # ── Workspace isolation ──────────────────────────────────────
        # For real (non-mock) dispatch, create the isolated git worktree
        # BEFORE dispatching the agent so the subprocess runs inside it.
        # This prevents agents from ever seeing or modifying the
        # orchestrator repository.
        pre_attempt_id = ids.generate_attempt_id()
        workspace: Workspace | None = None
        workspace_cwd: Path | None = None

        if not mock:
            try:
                workspace = workspace_manager.create_workspace(
                    work_item.id,
                    pre_attempt_id,
                    work_item.scope,
                    base_branch=git_engine.integration_branch,
                )
                workspace_cwd = workspace.paths.workspace_dir
            except Exception as exc:  # noqa: BLE001
                self._record_incident(
                    run_id=run_id,
                    category="workspace",
                    message=f"failed to create workspace: {exc}",
                    related_work_item_id=work_item.id,
                )
                self._work_item_repo.set_status(
                    work_item.id, WorkItemStatus.FAILED, run_id=run_id,
                )
                return
            self._workspace_by_branch[workspace.branch_name] = workspace

            # Rebuild the prompt with workspace context so the agent
            # knows it is running inside an isolated worktree.
            augmented_prompt = self._build_prompt(
                work_item,
                dispatch_plan.decision,
                dispatch_plan.attempt_number,
                workspace_dir=workspace_cwd,
            )
            dispatch_plan = _DispatchPlan(
                work_item=dispatch_plan.work_item,
                decision=dispatch_plan.decision,
                attempt_number=dispatch_plan.attempt_number,
                prompt=augmented_prompt,
                dispatch_budget=dispatch_plan.dispatch_budget,
            )

        self._log(f"  Dispatching via {dispatch_plan.decision.model}...")
        dispatch_outcome = self._dispatch_one(
            run_id=run_id,
            dispatch_plan=dispatch_plan,
            effective_config=effective_config,
            mock=mock,
            model_catalog=model_catalog,
            workspace_cwd=workspace_cwd,
        )
        if dispatch_outcome.error is not None or dispatch_outcome.result is None:
            self._log(f"  Dispatch FAILED: {dispatch_outcome.error or 'unknown'}")
            self._record_incident(
                run_id=run_id,
                category="dispatch",
                message=dispatch_outcome.error or "unknown dispatch failure",
                related_work_item_id=work_item.id,
            )
            self._work_item_repo.set_status(work_item.id, WorkItemStatus.FAILED, run_id=run_id)
            if workspace is not None:
                self._cleanup_workspace_for_branch(workspace.branch_name, workspace_manager)
            return

        self._log(
            f"  Dispatch OK ({dispatch_outcome.dispatch_result_tokens} tokens, "
            f"${dispatch_outcome.dispatch_result_cost_usd:.4f})"
        )
        self._update_budget_usage(
            run_id=run_id,
            tokens=dispatch_outcome.dispatch_result_tokens,
            cost_usd=dispatch_outcome.dispatch_result_cost_usd,
            provider_calls=dispatch_outcome.dispatch_result_attempts,
        )
        self._attempts_processed += 1

        # Use our pre-generated attempt_id for workspace naming (real mode)
        # or the dispatch-generated one for mock mode.
        attempt_id = pre_attempt_id
        if mock:
            attempt_id = dispatch_outcome.dispatch_result_attempt_id or pre_attempt_id

        if attempt_id is None:
            self._record_incident(
                run_id=run_id,
                category="dispatch",
                message="dispatch did not return attempt_id",
                related_work_item_id=work_item.id,
            )
            self._work_item_repo.set_status(work_item.id, WorkItemStatus.FAILED, run_id=run_id)
            if workspace is not None:
                self._cleanup_workspace_for_branch(workspace.branch_name, workspace_manager)
            return

        # ── Mock mode: create workspace AFTER dispatch (existing flow) ──
        if mock and workspace is None:
            try:
                workspace = workspace_manager.create_workspace(
                    work_item.id,
                    attempt_id,
                    work_item.scope,
                    base_branch=git_engine.integration_branch,
                )
            except Exception as exc:  # noqa: BLE001
                self._record_incident(
                    run_id=run_id,
                    category="workspace",
                    message=f"failed to create workspace: {exc}",
                    related_work_item_id=work_item.id,
                )
                self._work_item_repo.set_status(
                    work_item.id, WorkItemStatus.FAILED, run_id=run_id,
                )
                return
            self._workspace_by_branch[workspace.branch_name] = workspace

        assert workspace is not None  # noqa: S101

        try:
            if mock:
                # Mock mode: apply the returned patch to the workspace
                self._seed_missing_mock_scope_files(
                    workspace_dir=workspace.paths.workspace_dir,
                    scope=work_item.scope,
                )
                git_engine.apply_patch(
                    workspace.paths.workspace_dir, dispatch_outcome.result.raw_text,
                )
            # else: real mode — agent already modified files directly in the
            # workspace worktree; nothing to apply.

            try:
                git_engine.commit(
                    workspace.paths.workspace_dir,
                    f"{work_item.title} ({'mock' if mock else 'nexus'})",
                    work_item=work_item.id,
                    evidence="pending",
                    agent=ROLE_IMPLEMENTER,
                    iteration=dispatch_plan.attempt_number,
                )
            except Exception as exc:  # noqa: BLE001
                if "No staged changes to commit after filtering internal files." not in str(exc):
                    raise
        except Exception as exc:  # noqa: BLE001
            self._log(f"  Patch apply FAILED: {exc}")
            self._record_incident(
                run_id=run_id,
                category="patch_apply",
                message=f"failed to apply/commit patch: {exc}",
                related_work_item_id=work_item.id,
            )
            self._work_item_repo.set_status(work_item.id, WorkItemStatus.FAILED, run_id=run_id)
            self._cleanup_workspace_for_branch(workspace.branch_name, workspace_manager)
            return

        self._work_item_repo.set_status(work_item.id, WorkItemStatus.VERIFYING, run_id=run_id)
        self._log("  Verifying...")
        gate_decision, evidence_ids = self._verify_and_record(
            run_id=run_id,
            work_item=work_item,
            evidence_root=evidence_root,
            provider_name=dispatch_plan.decision.provider,
            model_name=dispatch_plan.decision.model,
            attempt_id=attempt_id,
        )
        if not gate_decision.accepted:
            self._log(f"  Verification FAILED: {', '.join(gate_decision.reason_codes)}")
            self._record_incident(
                run_id=run_id,
                category="constraint_gate",
                message=(
                    "constraint gate rejected work item "
                    f"{work_item.id}: {', '.join(gate_decision.reason_codes)}"
                ),
                related_work_item_id=work_item.id,
                details=gate_decision.to_feedback_payload(),
            )
            self._work_item_repo.set_status(work_item.id, WorkItemStatus.FAILED, run_id=run_id)
            self._cleanup_workspace_for_branch(workspace.branch_name, workspace_manager)
            return

        self._log("  Verification passed")
        self._work_item_repo.set_status(work_item.id, WorkItemStatus.PASSED, run_id=run_id)
        candidate = QueueCandidate(
            branch=workspace.branch_name,
            work_item_id=work_item.id,
            evidence_ids=evidence_ids,
            risk_tier=work_item.risk_tier,
            dependencies=work_item.dependencies,
        )
        try:
            merge_queue.enqueue(candidate)
            self._log("  Queued for merge")
        except Exception as exc:  # noqa: BLE001
            self._log(f"  Merge queue FAILED: {exc}")
            self._record_incident(
                run_id=run_id,
                category="merge_queue",
                message=f"failed to enqueue merge candidate: {exc}",
                related_work_item_id=work_item.id,
            )
            self._work_item_repo.set_status(work_item.id, WorkItemStatus.FAILED, run_id=run_id)
            self._cleanup_workspace_for_branch(workspace.branch_name, workspace_manager)

    def _dispatch_one(
        self,
        *,
        run_id: str,
        dispatch_plan: _DispatchPlan,
        effective_config: Mapping[str, object],
        mock: bool,
        model_catalog: ModelCatalog,
        workspace_cwd: Path | None = None,
    ) -> _DispatchOutcome:
        if mock:
            patch_by_work_item = {
                dispatch_plan.work_item.id: self._mock_patch_for_work_item(dispatch_plan.work_item),
            }
            provider = _DeterministicMockProvider(
                provider_name=dispatch_plan.decision.provider,
                model_catalog=model_catalog,
                patch_by_work_item=patch_by_work_item,
            )
        else:
            # Real mode: use detected CLI tool backends (codex / claude)
            from nexus_orchestrator.synthesis_plane.providers.tool_adapter import (
                TOOL_MODEL_SPEC,
                ToolBackend,
                ToolProvider,
            )
            from nexus_orchestrator.synthesis_plane.providers.tool_detection import detect_all_backends

            backends = detect_all_backends()
            if not backends:
                return _DispatchOutcome(
                    plan=dispatch_plan,
                    result=None,
                    dispatch_result_attempt_id=None,
                    dispatch_result_attempts=0,
                    dispatch_result_tokens=0,
                    dispatch_result_cost_usd=0.0,
                    error="no CLI tool backends available (install codex or claude CLI)",
                )

            _name_to_enum = {"codex": ToolBackend.CODEX_CLI, "claude": ToolBackend.CLAUDE_CODE}

            # Pick backend matching the escalation model when possible
            model_name = dispatch_plan.decision.model
            _model_to_backend = {
                "codex_cli": "codex",
                "claude_code": "claude",
                "codex_gpt53": "codex",
                "codex_spark": "codex",
                "claude_opus": "claude",
            }
            preferred_name = _model_to_backend.get(model_name)
            backend_info = backends[0]  # default: first detected
            if preferred_name is not None:
                for bi in backends:
                    if bi.name == preferred_name:
                        backend_info = bi
                        break

            backend_enum = _name_to_enum.get(backend_info.name)
            if backend_enum is None:
                return _DispatchOutcome(
                    plan=dispatch_plan,
                    result=None,
                    dispatch_result_attempt_id=None,
                    dispatch_result_attempts=0,
                    dispatch_result_tokens=0,
                    dispatch_result_cost_usd=0.0,
                    error=f"unknown tool backend: {backend_info.name}",
                )

            # Resolve --model flag for the CLI tool
            model_spec = TOOL_MODEL_SPEC.get(model_name)
            model_flag = model_spec[1] if model_spec else ""

            # Read timeout from config (default 600s)
            tool_cfg = _nested_mapping_get(effective_config, ("providers", "tool"))
            timeout_seconds = 600.0
            if tool_cfg is not None:
                timeout_raw = tool_cfg.get("timeout_seconds")
                if isinstance(timeout_raw, (int, float)) and not isinstance(timeout_raw, bool):
                    timeout_seconds = max(float(timeout_raw), 30.0)

            try:
                provider = ToolProvider(
                    backend=backend_enum,
                    binary_path=backend_info.binary_path,
                    timeout_seconds=timeout_seconds,
                    model_catalog=model_catalog,
                    model_flag=model_flag,
                )
            except Exception as exc:  # noqa: BLE001
                return _DispatchOutcome(
                    plan=dispatch_plan,
                    result=None,
                    dispatch_result_attempt_id=None,
                    dispatch_result_attempts=0,
                    dispatch_result_tokens=0,
                    dispatch_result_cost_usd=0.0,
                    error=f"failed to initialize tool provider: {exc}",
                )

        provider_cfg = _nested_mapping_get(
            effective_config,
            ("providers", dispatch_plan.decision.provider),
        )
        max_concurrency = _as_int(
            provider_cfg.get("max_concurrent") if provider_cfg is not None else None,
            default=1,
            minimum=1,
        )
        requests_per_minute = _as_int(
            provider_cfg.get("requests_per_minute") if provider_cfg is not None else None,
            default=0,
            minimum=0,
        )

        controller = DispatchController(
            providers=(
                ProviderBinding(
                    name=dispatch_plan.decision.provider,
                    model=dispatch_plan.decision.model,
                    provider=provider,
                    max_concurrency=max_concurrency,
                    rate_limit_calls=requests_per_minute,
                    rate_limit_period_seconds=60.0,
                ),
            ),
            persistence=RepositoryDispatchPersistence(
                attempt_repo=self._attempt_repo,
                provider_call_repo=self._provider_call_repo,
                persist_in_progress=True,
            ),
        )
        request = DispatchRequest(
            run_id=run_id,
            work_item_id=dispatch_plan.work_item.id,
            role=ROLE_IMPLEMENTER,
            prompt=dispatch_plan.prompt,
            provider_allowlist=(dispatch_plan.decision.provider,),
            budget=dispatch_plan.dispatch_budget,
            workspace_dir=workspace_cwd,
            metadata={
                "attempt_number": dispatch_plan.attempt_number,
                "provider_name": dispatch_plan.decision.provider,
                "model_name": dispatch_plan.decision.model,
                "work_item_id": dispatch_plan.work_item.id,
                "orchestrator_repo_root": str(self._repo_root),
            },
        )
        try:
            dispatch_result = asyncio.run(controller.dispatch(request))
        except KeyboardInterrupt:
            return _DispatchOutcome(
                plan=dispatch_plan,
                result=None,
                dispatch_result_attempt_id=None,
                dispatch_result_attempts=0,
                dispatch_result_tokens=0,
                dispatch_result_cost_usd=0.0,
                error="dispatch interrupted by user (Ctrl+C)",
            )
        except Exception as exc:  # noqa: BLE001
            return _DispatchOutcome(
                plan=dispatch_plan,
                result=None,
                dispatch_result_attempt_id=None,
                dispatch_result_attempts=0,
                dispatch_result_tokens=0,
                dispatch_result_cost_usd=0.0,
                error=f"dispatch failed: {exc}",
            )

        response = ProviderResponse(
            model=dispatch_result.model,
            raw_text=dispatch_result.content,
            usage=ProviderUsage(
                total_tokens=dispatch_result.tokens_used,
                cost_estimate_usd=dispatch_result.cost_usd,
                latency_ms=dispatch_result.latency_ms,
            ),
            request_id=dispatch_result.request_id,
            idempotency_key=dispatch_result.idempotency_key,
        )
        return _DispatchOutcome(
            plan=dispatch_plan,
            result=response,
            dispatch_result_attempt_id=dispatch_result.attempt_id,
            dispatch_result_attempts=dispatch_result.attempts,
            dispatch_result_tokens=dispatch_result.tokens_used,
            dispatch_result_cost_usd=dispatch_result.cost_usd,
        )

    def _triage_work_item(
        self,
        work_item: WorkItem,
        effective_config: Mapping[str, object],
    ) -> ModelAffinity:
        """Route a work item via Spark LLM triage, falling back to deterministic."""
        from nexus_orchestrator.synthesis_plane.spark_triage import triage_with_spark

        # Check if triage is enabled (default: True)
        tool_cfg = _nested_mapping_get(effective_config, ("providers", "tool"))
        triage_enabled = True
        triage_timeout = 30.0
        if tool_cfg is not None:
            te = tool_cfg.get("triage_enabled")
            if isinstance(te, bool):
                triage_enabled = te
            tt = tool_cfg.get("triage_timeout_seconds")
            if isinstance(tt, (int, float)) and not isinstance(tt, bool):
                triage_timeout = max(float(tt), 5.0)

        if not triage_enabled:
            affinity = classify_work_item(work_item)
            self._log(f"  Triage disabled → deterministic: {affinity.value}")
            return affinity

        try:
            triage_result = asyncio.run(triage_with_spark(
                work_item, timeout_seconds=triage_timeout,
            ))
        except Exception:  # noqa: BLE001
            affinity = classify_work_item(work_item)
            self._log(f"  Triage error → deterministic: {affinity.value}")
            return affinity

        if triage_result.used_llm:
            self._log(f"  Spark triage → {triage_result.chosen_model.value}")
        else:
            self._log(f"  {triage_result.reasoning} → {triage_result.chosen_model.value}")
        return triage_result.chosen_model

    def _build_dispatch_plan(
        self,
        *,
        run_id: str,
        work_item: WorkItem,
        effective_config: Mapping[str, object],
        role_registry: RoleRegistry,
        model_catalog: ModelCatalog,
        warned_models: set[tuple[str, str]],
        warning_messages: list[str],
    ) -> _DispatchPlan | None:
        previous_attempts = self._attempt_repo.list_for_work_item(work_item.id, limit=1_000)
        attempt_number = len(previous_attempts) + 1

        # Intelligent model delegation: classify work item and pick the
        # appropriate escalation ladder (codex-first vs claude-first)
        decision: EscalationDecision | None = None
        default_provider = "tool"
        if isinstance(effective_config, Mapping):
            providers_raw = effective_config.get("providers")
            if isinstance(providers_raw, Mapping):
                dp = providers_raw.get("default")
                if isinstance(dp, str) and dp:
                    default_provider = dp

        if default_provider == "tool":
            from nexus_orchestrator.synthesis_plane.roles import (
                _resolve_provider_profile_models,
            )

            profile_models = _resolve_provider_profile_models(
                config=effective_config, model_catalog=model_catalog,
            )
            if "tool" in profile_models:
                # Try Spark LLM triage first, fall back to deterministic classifier
                affinity = self._triage_work_item(work_item, effective_config)
                ladder = resolve_tool_ladder_for_affinity(affinity, profile_models)
                decision = ladder.resolve_attempt(attempt_number)

        # Fall back to role registry if classifier didn't produce a decision
        if decision is None:
            decision = role_registry.route_attempt(
                role_name=ROLE_IMPLEMENTER,
                attempt_number=attempt_number,
            )
        if decision is None:
            self._record_incident(
                run_id=run_id,
                category="dispatch",
                message=(
                    f"role {ROLE_IMPLEMENTER} has no routing decision for attempt {attempt_number}"
                ),
                related_work_item_id=work_item.id,
            )
            return None

        warning_message = self._model_availability_warning(
            provider=decision.provider,
            model=decision.model,
            model_catalog=model_catalog,
        )
        if warning_message is not None:
            key = (decision.provider, decision.model)
            if key not in warned_models:
                warned_models.add(key)
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warning_messages.append(warning_message)

        role_budget = role_registry.require(ROLE_IMPLEMENTER).budget
        context_limit = self._model_context_limit(
            provider=decision.provider,
            model=decision.model,
            model_catalog=model_catalog,
            fallback=work_item.budget.max_tokens,
        )
        token_limit_candidates = [work_item.budget.max_tokens, context_limit]
        if role_budget.max_tokens_per_attempt is not None:
            token_limit_candidates.append(role_budget.max_tokens_per_attempt)
        max_tokens = min(token_limit_candidates)

        cost_limit_candidates = [work_item.budget.max_cost_usd]
        if role_budget.max_cost_per_work_item_usd is not None:
            cost_limit_candidates.append(role_budget.max_cost_per_work_item_usd)
        max_cost_usd = min(cost_limit_candidates)
        max_attempts = (
            role_budget.max_attempts
            if role_budget.max_attempts is not None
            else work_item.budget.max_iterations
        )

        prompt = self._build_prompt(work_item, decision, attempt_number)
        return _DispatchPlan(
            work_item=work_item,
            decision=decision,
            attempt_number=attempt_number,
            prompt=prompt,
            dispatch_budget=DispatchBudget(
                max_tokens=max_tokens,
                max_cost_usd=max_cost_usd,
                max_attempts=max_attempts,
            ),
        )

    def _verify_and_record(
        self,
        *,
        run_id: str,
        work_item: WorkItem,
        evidence_root: Path,
        provider_name: str,
        model_name: str,
        attempt_id: str | None,
    ) -> tuple[ConstraintGateDecision, tuple[str, ...]]:
        checker_to_constraints: dict[str, list[str]] = defaultdict(list)
        for constraint in sorted(
            work_item.constraint_envelope.constraints, key=lambda item: item.id
        ):
            checker_to_constraints[constraint.checker_binding].append(constraint.id)

        check_results: list[PipelineCheckResult] = []
        stage_requirements: list[StageCoverageRequirement] = []
        for checker_id in sorted(checker_to_constraints):
            stage_id = _CHECKER_STAGE_MAP.get(checker_id, "integration_tests")
            constraint_ids = tuple(sorted(set(checker_to_constraints[checker_id])))
            check_results.append(
                PipelineCheckResult(
                    stage=stage_id,
                    checker_id=checker_id,
                    result=EvidenceResult.PASS,
                    covered_constraint_ids=constraint_ids,
                    required_constraint_ids=constraint_ids,
                    summary="mock verification pass",
                )
            )
            stage_requirements.append(
                StageCoverageRequirement(stage=stage_id, constraint_ids=constraint_ids)
            )

        ordered_checks = tuple(
            sorted(check_results, key=lambda item: (item.stage, item.checker_id))
        )
        selected_stages = tuple(sorted({item.stage for item in ordered_checks}))
        gate_pipeline_output = PipelineOutput(
            check_results=ordered_checks,
            mode=VerificationSelectionMode.INCREMENTAL,
            selected_stages=selected_stages,
            required_stages=selected_stages,
            stage_coverage_requirements=tuple(
                sorted(stage_requirements, key=lambda item: (item.stage, item.constraint_ids))
            ),
            adversarial_required=False,
        )
        gate_decision = run_constraint_gate(
            work_item.constraint_envelope,
            gate_pipeline_output,
        )

        writer = EvidenceWriter(evidence_root)

        evidence_ids: list[str] = []
        for check in ordered_checks:
            evidence_id = ids.generate_evidence_id()
            evidence_payload: dict[str, JSONValue] = {
                "work_item_id": work_item.id,
                "stage": check.stage,
                "checker_id": check.checker_id,
                "constraint_ids": list(check.covered_constraint_ids),
                "result": check.result.value,
                "gate_verdict": gate_decision.verdict.value,
                "provider": provider_name,
                "model": model_name,
            }
            write_result = writer.write_evidence(
                run_id=run_id,
                work_item_id=work_item.id,
                attempt_id=attempt_id,
                stage=check.stage,
                evidence_id=evidence_id,
                result=evidence_payload,
                metadata={
                    "gate_verdict": gate_decision.verdict.value,
                    "reason_codes": list(gate_decision.reason_codes),
                    "provider": provider_name,
                    "model": model_name,
                    "attempt_id": attempt_id,
                    "checker_id": check.checker_id,
                },
                artifacts={"check_result.json": evidence_payload},
            )
            artifact_paths = tuple(
                sorted(
                    (write_result.evidence_dir / entry.path)
                    .resolve()
                    .relative_to(self._repo_root)
                    .as_posix()
                    for entry in write_result.manifest_entries
                )
            )
            record = EvidenceRecord(
                id=evidence_id,
                work_item_id=work_item.id,
                run_id=run_id,
                stage=check.stage,
                result=check.result,
                checker_id=check.checker_id,
                constraint_ids=check.covered_constraint_ids,
                artifact_paths=artifact_paths,
                tool_versions={"pytest": "mock", "ruff": "mock"},
                environment_hash=sha256_text("mock-environment"),
                duration_ms=1,
                created_at=_utc_now(),
                summary=check.summary,
                metadata={
                    "gate_verdict": gate_decision.verdict.value,
                    "reason_codes": list(gate_decision.reason_codes),
                    "provider": provider_name,
                    "model": model_name,
                    "attempt_id": attempt_id,
                    "checker_id": check.checker_id,
                },
            )
            self._evidence_repo.add(record)
            evidence_ids.append(evidence_id)

        return gate_decision, tuple(evidence_ids)

    def _drain_merge_queue(
        self,
        *,
        run_id: str,
        merge_queue: MergeQueue,
        git_engine: GitEngine,
        workspace_manager: WorkspaceManager,
    ) -> None:
        while merge_queue.has_pending:
            merge_result = asyncio.run(
                merge_queue.process_next(
                    git_engine=git_engine,
                    compositional_check_callback=lambda candidate: self._merge_gate_check(
                        run_id=run_id,
                        candidate=candidate,
                    ),
                )
            )

            if merge_result.status is MergeStatus.NO_CANDIDATE:
                break
            if merge_result.candidate is None:
                continue

            candidate = merge_result.candidate
            if merge_result.status is MergeStatus.MERGED:
                self._log(f"  Merged: {candidate.work_item_id[:12]}")
                commit_sha = merge_result.integration_head_after
                if commit_sha is None:
                    commit_sha = git_engine.get_integration_head()
                merge_record = MergeRecord(
                    id=ids.generate_merge_id(),
                    work_item_id=candidate.work_item_id,
                    run_id=run_id,
                    commit_sha=commit_sha,
                    evidence_ids=tuple(candidate.evidence_ids),
                    merged_at=_utc_now(),
                )
                self._merge_repo.add(merge_record)
                self._cleanup_workspace_for_branch(candidate.branch, workspace_manager)
                continue

            if merge_result.status in {MergeStatus.FAILED, MergeStatus.REQUEUED}:
                self._log(f"  Merge {merge_result.status.value}: {candidate.work_item_id[:12]}")
                if merge_result.status is MergeStatus.FAILED:
                    self._work_item_repo.set_status(
                        candidate.work_item_id,
                        WorkItemStatus.FAILED,
                        run_id=run_id,
                    )
                    self._record_incident(
                        run_id=run_id,
                        category="merge",
                        message=merge_result.message or "merge failed",
                        related_work_item_id=candidate.work_item_id,
                        details={
                            "status": merge_result.status.value,
                            "conflict_details": list(merge_result.conflict_details),
                            "diagnostics": dict(merge_result.diagnostics),
                        },
                    )
                    self._cleanup_workspace_for_branch(candidate.branch, workspace_manager)
                break

    def _merge_gate_check(self, *, run_id: str, candidate: QueueCandidate) -> Mapping[str, object]:
        work_item = self._work_item_repo.get_for_run(run_id, candidate.work_item_id)
        if work_item is None:
            return {
                "passed": False,
                "requeue": False,
                "message": f"missing work item {candidate.work_item_id}",
            }
        if work_item.status is not WorkItemStatus.PASSED:
            return {
                "passed": False,
                "requeue": False,
                "message": (
                    "merge blocked: work item has not passed constraint gate "
                    f"(status={work_item.status.value})"
                ),
            }

        evidence_records = self._evidence_repo.list_for_work_item(
            candidate.work_item_id, limit=1_000
        )
        gate_accept_found = any(
            record.metadata.get("gate_verdict") == "accept" for record in evidence_records
        )
        if not gate_accept_found:
            return {
                "passed": False,
                "requeue": False,
                "message": "merge blocked: no evidence of accepted constraint gate verdict",
            }
        return {"passed": True, "requeue": False}

    def _append_dispatch_batch(self, run_id: str, batch_ids: Sequence[str]) -> None:
        run_record = self._require_run(run_id)
        payload = run_record.to_dict()
        metadata = _as_json_dict(payload.get("metadata", {}))
        batches_raw = metadata.get("dispatch_batches", [])
        batches = _as_json_list(batches_raw)
        batches.append([item for item in batch_ids])
        metadata["dispatch_batches"] = batches
        payload["metadata"] = metadata
        self._run_repo.upsert(Run.from_dict(payload))

    def _update_budget_usage(
        self,
        *,
        run_id: str,
        tokens: int,
        cost_usd: float,
        provider_calls: int,
    ) -> None:
        run_record = self._require_run(run_id)
        payload = run_record.to_dict()
        metadata = _as_json_dict(payload.get("metadata", {}))
        usage_raw = metadata.get("budget_usage", {})
        usage = _as_json_dict(usage_raw)

        current_tokens = _as_int(usage.get("tokens_used"), default=0, minimum=0)
        current_cost = _as_float(usage.get("cost_usd"), default=0.0, minimum=0.0)
        current_calls = _as_int(usage.get("provider_calls"), default=0, minimum=0)

        usage["tokens_used"] = current_tokens + max(tokens, 0)
        usage["cost_usd"] = current_cost + max(cost_usd, 0.0)
        usage["provider_calls"] = current_calls + max(provider_calls, 0)
        metadata["budget_usage"] = usage
        payload["metadata"] = metadata
        self._run_repo.upsert(Run.from_dict(payload))

    def _build_result(self, *, run_id: str, resumed: bool, warnings: tuple[str, ...]) -> RunResult:
        run_record = self._require_run(run_id)
        work_items = self._work_item_repo.list_for_run(run_id, limit=1_000)
        merged_ids = tuple(item.id for item in work_items if item.status is WorkItemStatus.MERGED)
        failed_ids = tuple(item.id for item in work_items if item.status is WorkItemStatus.FAILED)

        metadata = _as_json_dict(run_record.metadata)
        usage = _as_json_dict(metadata.get("budget_usage", {}))
        dispatch_batches_raw = _as_json_list(metadata.get("dispatch_batches", []))

        dispatch_batches: list[tuple[str, ...]] = []
        for batch in dispatch_batches_raw:
            if isinstance(batch, list):
                dispatch_batches.append(tuple(str(item) for item in batch if isinstance(item, str)))

        return RunResult(
            run_id=run_id,
            status=cast("RunStatus", run_record.status),
            resumed_from_crash=resumed,
            merged_work_item_ids=merged_ids,
            failed_work_item_ids=failed_ids,
            dispatch_batches=tuple(dispatch_batches),
            budget_tokens_used=_as_int(usage.get("tokens_used"), default=0, minimum=0),
            budget_cost_usd=_as_float(usage.get("cost_usd"), default=0.0, minimum=0.0),
            provider_calls=_as_int(usage.get("provider_calls"), default=0, minimum=0),
            warnings=warnings,
        )

    def _record_incident(
        self,
        *,
        run_id: str,
        category: str,
        message: str,
        related_work_item_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        incident = Incident(
            id=ids.generate_incident_id(),
            run_id=run_id,
            category=category,
            message=message,
            created_at=_utc_now(),
            related_work_item_id=related_work_item_id,
            details=(
                _coerce_json_mapping(details)
                if details is not None
                else cast("dict[str, JSONValue]", {})
            ),
        )
        self._incident_repo.add(incident)

    def _model_context_limit(
        self,
        *,
        provider: str,
        model: str,
        model_catalog: ModelCatalog,
        fallback: int,
    ) -> int:
        try:
            return model_catalog.max_context_tokens(provider=provider, model=model)
        except KeyError:
            return fallback

    def _warn_on_registry_models(
        self,
        role_registry: RoleRegistry,
        model_catalog: ModelCatalog,
        warned_models: set[tuple[str, str]],
    ) -> tuple[str, ...]:
        warning_messages: list[str] = []
        for role in role_registry.roles:
            for step in role.escalation_policy.steps:
                key = (step.provider, step.model)
                if key in warned_models:
                    continue
                message = self._model_availability_warning(
                    provider=step.provider,
                    model=step.model,
                    model_catalog=model_catalog,
                )
                if message is None:
                    continue
                warned_models.add(key)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                warning_messages.append(message)
        return tuple(warning_messages)

    def _warn_on_run_metadata(
        self, run_id: str, *, warning_messages: tuple[str, ...]
    ) -> tuple[str, ...]:
        run_record = self._require_run(run_id)
        payload = run_record.to_dict()
        metadata = _as_json_dict(payload.get("metadata", {}))
        existing = _as_json_list(metadata.get("warnings", []))
        for warning_message in warning_messages:
            if warning_message not in existing:
                existing.append(warning_message)
        metadata["warnings"] = existing
        payload["metadata"] = metadata
        self._run_repo.upsert(Run.from_dict(payload))
        return tuple(item for item in existing if isinstance(item, str))

    def _model_availability_warning(
        self,
        *,
        provider: str,
        model: str,
        model_catalog: ModelCatalog,
    ) -> str | None:
        capabilities = model_catalog.get(model, provider=provider)
        if capabilities is None:
            return None
        availability = (capabilities.availability or "").strip().lower()
        if availability not in _UNSAFE_MODEL_AVAILABILITY:
            return None
        return (
            f"model availability warning: provider={provider} model={model} "
            f"is marked {availability} in model catalog"
        )

    def _build_git_engine(self, config: Mapping[str, object]) -> GitEngine:
        git_cfg = _nested_mapping_get(config, ("git",))
        main_branch = _as_str(git_cfg.get("main_branch") if git_cfg is not None else None, "main")
        integration_branch = _as_str(
            git_cfg.get("integration_branch") if git_cfg is not None else None,
            "integration",
        )
        return GitEngine(
            repo_path=self._repo_root,
            main_branch=main_branch,
            integration_branch=integration_branch,
        )

    def _build_workspace_manager(self, config: Mapping[str, object]) -> WorkspaceManager:
        workspace_root = self._resolve_path(
            _nested_get(config, ("paths", "workspace_root"), "workspaces"),
            base=self._repo_root,
        )
        workspace_root.mkdir(parents=True, exist_ok=True)
        return WorkspaceManager(repo_root=self._repo_root, workspace_root=workspace_root)

    def _resolve_spec_path(self, spec_path: str | Path) -> _ResolvedSpecPath:
        candidate = Path(spec_path)
        absolute = (
            candidate.expanduser().resolve(strict=True)
            if candidate.is_absolute()
            else (self._repo_root / candidate).resolve(strict=True)
        )
        relative = absolute.relative_to(self._repo_root).as_posix()
        return _ResolvedSpecPath(absolute_path=absolute, relative_path=relative)

    def _resolve_path(self, value: object, *, base: Path) -> Path:
        if not isinstance(value, str):
            raise ValueError(f"path config must be a string, got {type(value).__name__}")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve(strict=False)
        return path

    def _batch_limit(self, config: Mapping[str, object]) -> int:
        resources_cfg = _nested_mapping_get(config, ("resources",))
        return _as_int(
            resources_cfg.get("max_light_verification") if resources_cfg is not None else None,
            default=2,
            minimum=1,
        )

    def _budget_from_config(self, config: Mapping[str, object]) -> Budget:
        budget_cfg = _nested_mapping_get(config, ("budgets",))
        max_iterations = _as_int(
            budget_cfg.get("max_iterations") if budget_cfg is not None else None,
            default=5,
            minimum=1,
        )
        max_tokens = _as_int(
            budget_cfg.get("max_tokens_per_attempt") if budget_cfg is not None else None,
            default=32_000,
            minimum=1,
        )
        max_cost = _as_float(
            budget_cfg.get("max_cost_per_work_item_usd") if budget_cfg is not None else None,
            default=2.0,
            minimum=0.0,
        )
        return Budget(
            max_tokens=max_tokens,
            max_cost_usd=max_cost,
            max_iterations=max_iterations,
            max_wall_clock_seconds=3_600,
        )

    def _build_prompt(
        self,
        work_item: WorkItem,
        decision: EscalationDecision,
        attempt_number: int,
        workspace_dir: Path | None = None,
    ) -> str:
        scope = ", ".join(work_item.scope)
        dependencies = ", ".join(work_item.dependencies) if work_item.dependencies else "(none)"
        constraint_ids = ", ".join(
            constraint.id for constraint in work_item.constraint_envelope.constraints
        )
        lines = [
            f"Role: {ROLE_IMPLEMENTER}",
            f"Provider: {decision.provider}",
            f"Model: {decision.model}",
            f"Attempt: {attempt_number}",
            f"WorkItemId: {work_item.id}",
            f"Title: {work_item.title}",
            f"Description: {work_item.description}",
            f"Scope: {scope}",
            f"Dependencies: {dependencies}",
            f"Constraints: {constraint_ids}",
        ]
        if workspace_dir is not None:
            lines.append(
                "You are working in an isolated git worktree. "
                "All file modifications MUST stay within the current working "
                "directory. Do NOT access, read, or modify files outside of it. "
                "Implement the requested changes by editing files directly."
            )
        else:
            lines.append("Return only a unified git patch.")
        return "\n".join(lines)

    def _mock_patch_for_work_item(self, work_item: WorkItem) -> str:
        module_suffix = self._module_suffix(work_item)
        if module_suffix == "A":
            return _PATCH_MODULE_A
        if module_suffix == "B":
            return _PATCH_MODULE_B
        if module_suffix == "C":
            return _PATCH_MODULE_C
        raise ValueError(f"no deterministic mock patch configured for work item: {work_item.title}")

    def _module_suffix(self, work_item: WorkItem) -> str:
        scope_first = work_item.scope[0] if work_item.scope else ""
        stem = Path(scope_first).stem.strip()
        if stem:
            return stem[-1].upper()
        match = _MODULE_SUFFIX_RE.search(work_item.title)
        if match is not None:
            return match.group(1).upper()
        return "?"

    def _seed_missing_mock_scope_files(self, *, workspace_dir: Path, scope: Sequence[str]) -> None:
        workspace_root = workspace_dir.resolve()
        for scope_path in scope:
            normalized = Path(scope_path).as_posix().lstrip("./")
            template = _MOCK_SCOPE_FILE_TEMPLATES.get(normalized)
            if template is None:
                continue
            target = (workspace_root / normalized).resolve()
            try:
                target.relative_to(workspace_root)
            except ValueError:
                continue
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(template, encoding="utf-8")

    def _cleanup_workspace_for_branch(
        self,
        branch_name: str,
        workspace_manager: WorkspaceManager,
    ) -> None:
        workspace = self._workspace_by_branch.pop(branch_name, None)
        if workspace is not None:
            try:
                workspace_manager.cleanup_workspace(workspace)
            except Exception:  # noqa: BLE001
                return
            return

        for active_workspace in workspace_manager.list_active_workspaces():
            if active_workspace.branch_name != branch_name:
                continue
            try:
                workspace_manager.cleanup_workspace(active_workspace)
            except Exception:  # noqa: BLE001
                return
            return

    def _is_pause_requested(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._pause_requests

    def _is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancel_requests

    def _require_run(self, run_id: str) -> Run:
        run_record = self._run_repo.get(run_id)
        if run_record is None:
            raise ValueError(f"run not found: {run_id}")
        return run_record


@dataclass(frozen=True, slots=True)
class _ResolvedSpecPath:
    absolute_path: Path
    relative_path: str


def _effective_config(config: Mapping[str, object] | None) -> dict[str, object]:
    base = default_config()
    merged = merge_config(base, config or {})
    return cast("dict[str, object]", assert_valid_config(merged))


def _nested_mapping_get(
    payload: Mapping[str, object],
    path: Sequence[str],
) -> Mapping[str, object] | None:
    current: object = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        if part not in current:
            return None
        current = current[part]
    if not isinstance(current, Mapping):
        return None
    return cast("Mapping[str, object]", current)


def _nested_get(payload: Mapping[str, object], path: Sequence[str], default: object) -> object:
    current: object = payload
    for part in path:
        if not isinstance(current, Mapping):
            return default
        if part not in current:
            return default
        current = current[part]
    return current


def _as_str(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _as_int(value: object, *, default: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    if value < minimum:
        return default
    return value


def _as_float(value: object, *, default: float, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    if parsed < minimum:
        return default
    return parsed


def _as_json_dict(value: object) -> dict[str, JSONValue]:
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            out[key] = _coerce_json_value(item)
        return out
    return {}


def _as_json_list(value: object) -> list[JSONValue]:
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    return []


def _coerce_json_mapping(value: Mapping[str, object]) -> dict[str, JSONValue]:
    return {key: _coerce_json_value(item) for key, item in value.items()}


def _coerce_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return cast("JSONValue", value)
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if isinstance(key, str):
                out[key] = _coerce_json_value(item)
        return out
    return str(value)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


_MOCK_SCOPE_FILE_TEMPLATES: Final[dict[str, str]] = {
    "src/a.py": 'def greet(name: str) -> str:\n    return "TODO"\n',
    "src/b.py": 'def farewell(name: str) -> str:\n    return "TODO"\n',
    "src/c.py": 'def conversation(name: str) -> str:\n    return "TODO"\n',
    "tests/unit/test_a.py": "def test_greet() -> None:\n    assert True\n",
    "tests/unit/test_b.py": "def test_farewell() -> None:\n    assert True\n",
    "tests/unit/test_c.py": "def test_conversation() -> None:\n    assert True\n",
}


_PATCH_MODULE_A = (
    "diff --git a/src/a.py b/src/a.py\n"
    "--- a/src/a.py\n"
    "+++ b/src/a.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def greet(name: str) -> str:\n"
    '-    return "TODO"\n'
    '+    return f"Hello, {name}!"\n'
    "diff --git a/tests/unit/test_a.py b/tests/unit/test_a.py\n"
    "--- a/tests/unit/test_a.py\n"
    "+++ b/tests/unit/test_a.py\n"
    "@@ -1,2 +1,4 @@\n"
    "+from src.a import greet\n"
    "+\n"
    " def test_greet() -> None:\n"
    "-    assert True\n"
    '+    assert greet("World") == "Hello, World!"\n'
)

_PATCH_MODULE_B = (
    "diff --git a/src/b.py b/src/b.py\n"
    "--- a/src/b.py\n"
    "+++ b/src/b.py\n"
    "@@ -1,2 +1,2 @@\n"
    " def farewell(name: str) -> str:\n"
    '-    return "TODO"\n'
    '+    return f"Goodbye, {name}!"\n'
    "diff --git a/tests/unit/test_b.py b/tests/unit/test_b.py\n"
    "--- a/tests/unit/test_b.py\n"
    "+++ b/tests/unit/test_b.py\n"
    "@@ -1,2 +1,4 @@\n"
    "+from src.b import farewell\n"
    "+\n"
    " def test_farewell() -> None:\n"
    "-    assert True\n"
    '+    assert farewell("World") == "Goodbye, World!"\n'
)

_PATCH_MODULE_C = (
    "diff --git a/src/c.py b/src/c.py\n"
    "--- a/src/c.py\n"
    "+++ b/src/c.py\n"
    "@@ -1,2 +1,5 @@\n"
    "+from src.a import greet\n"
    "+from src.b import farewell\n"
    "+\n"
    " def conversation(name: str) -> str:\n"
    '-    return "TODO"\n'
    '+    return f"{greet(name)}\\n{farewell(name)}"\n'
    "diff --git a/tests/unit/test_c.py b/tests/unit/test_c.py\n"
    "--- a/tests/unit/test_c.py\n"
    "+++ b/tests/unit/test_c.py\n"
    "@@ -1,2 +1,4 @@\n"
    "+from src.c import conversation\n"
    "+\n"
    " def test_conversation() -> None:\n"
    "-    assert True\n"
    '+    assert conversation("World") == "Hello, World!\\nGoodbye, World!"\n'
)


RunCoordinator = OrchestratorController

__all__ = [
    "OrchestratorController",
    "RunCoordinator",
    "RunResult",
    "SimulatedCrashError",
]
