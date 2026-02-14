"""Command-line interface router for nexus-orchestrator."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from nexus_orchestrator.config import (
    ConfigLoadError,
    ConfigValidationError,
    assert_valid_config,
    load_config,
)
from nexus_orchestrator.control_plane import OrchestratorController
from nexus_orchestrator.domain.models import Run, RunStatus, WorkItem
from nexus_orchestrator.knowledge_plane.evidence_ledger import EvidenceLedger
from nexus_orchestrator.persistence import (
    AttemptRepo,
    EvidenceRepo,
    IncidentRepo,
    MergeRepo,
    RunRepo,
    StateDB,
    TaskGraphRepo,
    WorkItemRepo,
)
from nexus_orchestrator.planning import build_deterministic_architect_output, compile_constraints
from nexus_orchestrator.spec_ingestion import SpecIngestError, ingest_spec
from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog, load_model_catalog
from nexus_orchestrator.synthesis_plane.providers.base import ProviderError
from nexus_orchestrator.synthesis_plane.roles import RoleRegistry

DEFAULT_SPEC_PATH: Final[str] = "samples/specs/minimal_design_doc.md"
DEFAULT_STATE_DB_PATH: Final[str] = "state/nexus.sqlite"
UNSAFE_MODEL_AVAILABILITY: Final[frozenset[str]] = frozenset(
    {"deprecated", "experimental", "legacy"}
)


@dataclass(frozen=True, slots=True)
class CLIError(RuntimeError):
    """Typed CLI failure with an explicit process exit code."""

    message: str
    exit_code: int = 1

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class RoutingStepSnapshot:
    stage_index: int
    attempts: int
    provider: str
    model: str
    availability: str


@dataclass(frozen=True, slots=True)
class RoleRoutingSnapshot:
    role: str
    enabled: bool
    steps: tuple[RoutingStepSnapshot, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse command router for all supported CLI workflows."""

    parser = argparse.ArgumentParser(
        prog="nexus",
        description="nexus-orchestrator command interface",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo-root",
        default=".",
        help="Repository root directory (default: current working directory).",
    )
    common.add_argument(
        "--config",
        dest="config_path",
        default=None,
        help="Path to orchestrator TOML config (default: ./orchestrator.toml if present).",
    )
    common.add_argument(
        "--profile",
        default=None,
        help="Optional config profile overlay name.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan", parents=[common], help="Compile a deterministic plan"
    )
    plan_parser.add_argument("spec_path", help="Path to the spec/design document")
    plan_parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output")
    plan_parser.set_defaults(handler=_cmd_plan)

    run_parser = subparsers.add_parser("run", parents=[common], help="Execute orchestration")
    run_parser.add_argument(
        "spec_path",
        nargs="?",
        default=DEFAULT_SPEC_PATH,
        help=f"Spec path (default: {DEFAULT_SPEC_PATH})",
    )
    run_parser.add_argument(
        "--mock", action="store_true", help="Use deterministic mock provider flow"
    )
    run_parser.add_argument(
        "--mode",
        choices=("auto", "resume", "fresh"),
        default="auto",
        help="Run startup mode (default: auto)",
    )
    run_parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output")
    run_parser.set_defaults(handler=_cmd_run)

    status_parser = subparsers.add_parser(
        "status", parents=[common], help="Show latest run status and routing info"
    )
    status_parser.add_argument("--run-id", default=None, help="Inspect a specific run ID instead")
    status_parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output")
    status_parser.set_defaults(handler=_cmd_status)

    inspect_parser = subparsers.add_parser(
        "inspect",
        parents=[common],
        help="Inspect a run or work item; defaults to latest run",
    )
    inspect_parser.add_argument("target", nargs="?", default=None, help="Run ID or work-item ID")
    inspect_parser.add_argument(
        "--json", action="store_true", help="Emit deterministic JSON output"
    )
    inspect_parser.set_defaults(handler=_cmd_inspect)

    export_parser = subparsers.add_parser(
        "export", parents=[common], help="Export deterministic audit bundle for a run"
    )
    export_parser.add_argument(
        "run_id", nargs="?", default=None, help="Run ID (default: latest run)"
    )
    export_parser.add_argument("--output", default=None, help="Optional output ZIP path")
    export_parser.add_argument(
        "--key-log",
        action="append",
        dest="key_logs",
        default=None,
        help="Additional key log path to include (repeatable)",
    )
    export_parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output")
    export_parser.set_defaults(handler=_cmd_export)

    clean_parser = subparsers.add_parser(
        "clean", parents=[common], help="Remove ephemeral state/evidence/workspaces artifacts"
    )
    clean_parser.add_argument(
        "--dry-run", action="store_true", help="Show deletions without mutating"
    )
    clean_parser.add_argument("--json", action="store_true", help="Emit deterministic JSON output")
    clean_parser.set_defaults(handler=_cmd_clean)

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Parse argv, route to a command handler, and return process exit code."""

    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(namespace, "handler", None)
    if not callable(handler):
        parser.print_help(sys.stderr)
        return 2

    try:
        result = handler(namespace)
    except CLIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    return int(result)


def main(argv: Sequence[str] | None = None) -> int:
    """Compatibility wrapper for main-module wiring."""

    return run_cli(argv)


def cli_entrypoint() -> None:
    """Console-script entrypoint."""

    raise SystemExit(run_cli())


def _cmd_plan(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args)
    config = _load_effective_config(args)
    spec_path = _resolve_spec_path(
        _require_str(getattr(args, "spec_path", None), "spec_path"), repo_root
    )
    registry_path = _path_from_config(config, ("paths", "constraint_registry"), repo_root)

    try:
        spec_map = ingest_spec(spec_path)
        architect_output = build_deterministic_architect_output(spec_map)
        compilation = compile_constraints(
            spec_map,
            architect_output,
            registry_path=registry_path,
        )
    except SpecIngestError as exc:
        raise CLIError(str(exc), exit_code=2) from exc
    except ValueError as exc:
        raise CLIError(f"planning failed: {exc}", exit_code=2) from exc

    task_graph_payload: dict[str, object] | None = None
    if compilation.task_graph is not None:
        task_graph_payload = {
            "run_id": compilation.task_graph.run_id,
            "work_item_ids": [item.id for item in compilation.task_graph.work_items],
            "edge_count": len(compilation.task_graph.edges),
            "critical_path": list(compilation.task_graph.critical_path),
        }

    payload: dict[str, object] = {
        "command": "plan",
        "spec_path": _display_path(spec_path, repo_root),
        "constraint_registry": _display_path(registry_path, repo_root),
        "requirements": len(spec_map.requirements),
        "interfaces": len(spec_map.interfaces),
        "work_items": [
            {
                "id": item.id,
                "title": item.title,
                "risk_tier": item.risk_tier.value,
                "dependencies": list(item.dependencies),
                "scope": list(item.scope),
                "constraint_count": len(item.constraint_envelope.constraints),
            }
            for item in compilation.work_items
        ],
        "task_graph": task_graph_payload,
        "warnings": list(compilation.warnings),
        "errors": list(compilation.errors),
    }

    lines = [
        f"Planned spec: {_display_path(spec_path, repo_root)}",
        f"Requirements: {len(spec_map.requirements)}",
        f"Interfaces: {len(spec_map.interfaces)}",
        f"Work items: {len(compilation.work_items)}",
    ]
    if task_graph_payload is not None:
        lines.append(f"Task graph edges: {task_graph_payload['edge_count']}")
    if compilation.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in compilation.warnings)
    if compilation.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in compilation.errors)

    _emit(payload, lines, json_mode=_flag(args, "json"))
    return 1 if compilation.errors else 0


def _cmd_run(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args)
    config = _load_effective_config(args)
    spec_arg = _require_str(getattr(args, "spec_path", None), "spec_path")
    spec_path = _resolve_spec_path(spec_arg, repo_root)
    state_db_path = _state_db_path(config, repo_root)

    mode_raw = _require_str(getattr(args, "mode", None), "mode")
    mode = "run" if mode_raw == "fresh" else mode_raw
    mock = _flag(args, "mock")

    controller = OrchestratorController(
        repo_root=repo_root,
        state_db_path=state_db_path,
    )

    try:
        result = controller.run(spec_path=spec_path, config=config, mode=mode, mock=mock)
    except FileNotFoundError as exc:
        raise CLIError(f"spec not found: {exc}", exit_code=2) from exc
    except ValueError as exc:
        raise CLIError(str(exc), exit_code=2) from exc
    except ProviderError as exc:
        raise CLIError(str(exc), exit_code=3) from exc
    except ModuleNotFoundError as exc:
        if exc.name in {"openai", "anthropic"}:
            raise CLIError(
                f"provider dependency missing: {exc.name}; install SDK or use --mock",
                exit_code=3,
            ) from exc
        raise

    state_db = StateDB(controller.state_db_path)
    run_repo = RunRepo(state_db)
    work_item_repo = WorkItemRepo(state_db)
    run_record = run_repo.get(result.run_id)
    if run_record is None:
        raise CLIError(f"run not found after execution: {result.run_id}", exit_code=4)

    work_items = work_item_repo.list_for_run(run_record.id, limit=1_000)
    work_status_counts = _work_item_status_counts(work_items)

    payload: dict[str, object] = {
        "command": "run",
        "run_id": result.run_id,
        "status": result.status.value,
        "spec_path": run_record.spec_path,
        "mode": mode_raw,
        "mock": mock,
        "resumed_from_crash": result.resumed_from_crash,
        "merged_work_item_ids": list(result.merged_work_item_ids),
        "failed_work_item_ids": list(result.failed_work_item_ids),
        "dispatch_batches": [list(batch) for batch in result.dispatch_batches],
        "budget_usage": {
            "tokens_used": result.budget_tokens_used,
            "cost_usd": result.budget_cost_usd,
            "provider_calls": result.provider_calls,
        },
        "work_item_counts": work_status_counts,
        "warnings": list(result.warnings),
        "state_db": _display_path(controller.state_db_path, repo_root),
    }

    lines = [
        f"Run ID: {result.run_id}",
        f"Status: {result.status.value}",
        f"Spec: {run_record.spec_path}",
        f"Mock mode: {str(mock).lower()}",
        f"Budget usage: tokens={result.budget_tokens_used} cost_usd={result.budget_cost_usd:.6f} "
        f"provider_calls={result.provider_calls}",
        f"Work items: {work_status_counts}",
    ]
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in result.warnings)

    _emit(payload, lines, json_mode=_flag(args, "json"))
    return 0 if result.status is RunStatus.COMPLETED else 1


def _cmd_status(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args)
    config = _load_effective_config(args)
    state_db_path = _state_db_path(config, repo_root)

    state_db = StateDB(state_db_path)
    run_repo = RunRepo(state_db)
    work_item_repo = WorkItemRepo(state_db)
    incident_repo = IncidentRepo(state_db)
    merge_repo = MergeRepo(state_db)

    run_arg = _optional_str(getattr(args, "run_id", None))
    run_record = _resolve_status_run(run_repo, run_arg)

    routing_ladder, model_warnings = _build_routing_ladder(config)
    ladder_payload = _routing_payload(routing_ladder)

    if run_record is None:
        payload: dict[str, object] = {
            "command": "status",
            "run": None,
            "state_db": _display_path(state_db_path, repo_root),
            "routing_ladder": ladder_payload,
            "model_catalog_warnings": list(model_warnings),
        }
        lines = [f"No runs found in {_display_path(state_db_path, repo_root)}"]
        lines.extend(_format_routing_lines(routing_ladder))
        if model_warnings:
            lines.append("Model availability warnings:")
            lines.extend(f"- {warning}" for warning in model_warnings)
        _emit(payload, lines, json_mode=_flag(args, "json"))
        return 0

    summary = _summarize_run(
        run_record,
        work_items=work_item_repo.list_for_run(run_record.id, limit=1_000),
        incidents=incident_repo.list_for_run(run_record.id, limit=1_000),
        merges=merge_repo.list_for_run(run_record.id, limit=1_000),
    )

    payload = {
        "command": "status",
        "run": summary,
        "state_db": _display_path(state_db_path, repo_root),
        "routing_ladder": ladder_payload,
        "model_catalog_warnings": list(model_warnings),
    }
    budget_usage = _budget_usage_from_metadata(run_record.metadata)

    lines = [
        f"Latest run: {run_record.id}",
        f"Status: {_run_status_text(run_record.status)}",
        f"Spec: {run_record.spec_path}",
        f"Started: {run_record.started_at.isoformat()}",
        (
            f"Finished: {run_record.finished_at.isoformat()}"
            if run_record.finished_at is not None
            else "Finished: (in progress)"
        ),
        f"Work items: {summary['work_item_counts']}",
        (
            "Budget usage: "
            f"tokens={budget_usage['tokens_used']} "
            f"cost_usd={budget_usage['cost_usd']:.6f} "
            f"provider_calls={budget_usage['provider_calls']}"
        ),
    ]
    lines.extend(_format_routing_lines(routing_ladder))
    if model_warnings:
        lines.append("Model availability warnings:")
        lines.extend(f"- {warning}" for warning in model_warnings)

    _emit(payload, lines, json_mode=_flag(args, "json"))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args)
    config = _load_effective_config(args)
    state_db_path = _state_db_path(config, repo_root)

    state_db = StateDB(state_db_path)
    run_repo = RunRepo(state_db)
    work_item_repo = WorkItemRepo(state_db)
    task_graph_repo = TaskGraphRepo(state_db)
    attempt_repo = AttemptRepo(state_db)
    evidence_repo = EvidenceRepo(state_db)
    incident_repo = IncidentRepo(state_db)
    merge_repo = MergeRepo(state_db)

    routing_ladder, model_warnings = _build_routing_ladder(config)
    ladder_payload = _routing_payload(routing_ladder)

    target = _optional_str(getattr(args, "target", None))
    if target is None:
        latest = _latest_run(run_repo)
        if latest is None:
            raise CLIError("inspect failed: no runs found", exit_code=1)
        target = latest.id

    run_record = _safe_get_run(run_repo, target)
    if run_record is not None:
        work_items = work_item_repo.list_for_run(run_record.id, limit=1_000)
        incidents = incident_repo.list_for_run(run_record.id, limit=1_000)
        merges = merge_repo.list_for_run(run_record.id, limit=1_000)
        task_graph = task_graph_repo.get(run_record.id)

        work_items_payload = [
            {
                "id": item.id,
                "title": item.title,
                "status": item.status.value,
                "risk_tier": item.risk_tier.value,
                "dependencies": list(item.dependencies),
                "scope": list(item.scope),
                "constraint_count": len(item.constraint_envelope.constraints),
                "evidence_count": len(item.evidence_ids),
                "commit_sha": item.commit_sha,
            }
            for item in work_items
        ]

        incidents_payload = [
            {
                "id": incident.id,
                "category": incident.category,
                "message": incident.message,
                "related_work_item_id": incident.related_work_item_id,
                "created_at": incident.created_at.isoformat(),
            }
            for incident in sorted(incidents, key=lambda item: (item.created_at, item.id))
        ]
        merges_payload = [
            {
                "id": merge.id,
                "work_item_id": merge.work_item_id,
                "commit_sha": merge.commit_sha,
                "merged_at": merge.merged_at.isoformat(),
            }
            for merge in sorted(merges, key=lambda item: (item.merged_at, item.id))
        ]

        payload: dict[str, object] = {
            "command": "inspect",
            "target_type": "run",
            "run": _summarize_run(
                run_record,
                work_items=work_items,
                incidents=incidents,
                merges=merges,
            ),
            "work_items": work_items_payload,
            "task_graph": (
                None
                if task_graph is None
                else {
                    "critical_path": list(task_graph.critical_path),
                    "edge_count": len(task_graph.edges),
                    "edges": [list(edge) for edge in task_graph.edges],
                }
            ),
            "incidents": incidents_payload,
            "merges": merges_payload,
            "routing_ladder": ladder_payload,
            "model_catalog_warnings": list(model_warnings),
        }

        lines = [
            f"Inspect run: {run_record.id}",
            f"Status: {_run_status_text(run_record.status)}",
            f"Spec: {run_record.spec_path}",
            f"Work items: {len(work_items)}",
            f"Incidents: {len(incidents)}",
            f"Merges: {len(merges)}",
        ]
        lines.extend(_format_routing_lines(routing_ladder))
        if model_warnings:
            lines.append("Model availability warnings:")
            lines.extend(f"- {warning}" for warning in model_warnings)

        _emit(payload, lines, json_mode=_flag(args, "json"))
        return 0

    work_item = _safe_get_work_item(work_item_repo, target)
    if work_item is None:
        raise CLIError(f"inspect failed: target not found: {target}", exit_code=2)

    run_id = _run_id_for_work_item(state_db, work_item.id)
    if run_id is None:
        raise CLIError(
            f"inspect failed: missing run link for work item {work_item.id}",
            exit_code=4,
        )

    run_record = run_repo.get(run_id)
    if run_record is None:
        raise CLIError(
            f"inspect failed: run not found for work item {work_item.id}: {run_id}",
            exit_code=4,
        )

    attempts = sorted(
        attempt_repo.list_for_work_item(work_item.id, limit=1_000),
        key=lambda item: (item.iteration, item.id),
    )
    evidence = sorted(
        evidence_repo.list_for_work_item(work_item.id, limit=1_000),
        key=lambda item: (item.created_at, item.id),
    )
    incidents = [
        incident
        for incident in incident_repo.list_for_run(run_id, limit=1_000)
        if incident.related_work_item_id == work_item.id
    ]
    incidents.sort(key=lambda item: (item.created_at, item.id))
    merges = [
        merge
        for merge in merge_repo.list_for_run(run_id, limit=1_000)
        if merge.work_item_id == work_item.id
    ]
    merges.sort(key=lambda item: (item.merged_at, item.id))

    payload = {
        "command": "inspect",
        "target_type": "work_item",
        "run": {
            "id": run_record.id,
            "status": _run_status_text(run_record.status),
            "spec_path": run_record.spec_path,
        },
        "work_item": {
            "id": work_item.id,
            "title": work_item.title,
            "description": work_item.description,
            "status": work_item.status.value,
            "risk_tier": work_item.risk_tier.value,
            "scope": list(work_item.scope),
            "dependencies": list(work_item.dependencies),
            "constraint_ids": list(work_item.constraint_ids),
            "evidence_ids": list(work_item.evidence_ids),
            "commit_sha": work_item.commit_sha,
            "created_at": work_item.created_at.isoformat(),
            "updated_at": work_item.updated_at.isoformat(),
        },
        "attempts": [
            {
                "id": attempt.id,
                "iteration": attempt.iteration,
                "provider": attempt.provider,
                "model": attempt.model,
                "result": attempt.result.value,
                "cost_usd": attempt.cost_usd,
                "tokens_used": attempt.tokens_used,
                "created_at": attempt.created_at.isoformat(),
            }
            for attempt in attempts
        ],
        "evidence": [
            {
                "id": record.id,
                "stage": record.stage,
                "result": record.result.value,
                "checker_id": record.checker_id,
                "created_at": record.created_at.isoformat(),
                "artifact_paths": list(record.artifact_paths),
            }
            for record in evidence
        ],
        "incidents": [
            {
                "id": incident.id,
                "category": incident.category,
                "message": incident.message,
                "created_at": incident.created_at.isoformat(),
            }
            for incident in incidents
        ],
        "merges": [
            {
                "id": merge.id,
                "commit_sha": merge.commit_sha,
                "merged_at": merge.merged_at.isoformat(),
            }
            for merge in merges
        ],
        "routing_ladder": ladder_payload,
        "model_catalog_warnings": list(model_warnings),
    }

    lines = [
        f"Inspect work item: {work_item.id}",
        f"Run ID: {run_id}",
        f"Status: {work_item.status.value}",
        f"Attempts: {len(attempts)}",
        f"Evidence: {len(evidence)}",
        f"Incidents: {len(incidents)}",
        f"Merges: {len(merges)}",
    ]
    lines.extend(_format_routing_lines(routing_ladder))
    if model_warnings:
        lines.append("Model availability warnings:")
        lines.extend(f"- {warning}" for warning in model_warnings)

    _emit(payload, lines, json_mode=_flag(args, "json"))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args)
    config = _load_effective_config(args)
    state_db_path = _state_db_path(config, repo_root)
    evidence_root = _path_from_config(config, ("paths", "evidence_root"), repo_root)

    state_db = StateDB(state_db_path)
    run_repo = RunRepo(state_db)

    run_id_raw = _optional_str(getattr(args, "run_id", None))
    run_id = run_id_raw
    if run_id is None:
        latest = _latest_run(run_repo)
        if latest is None:
            raise CLIError("export failed: no runs found", exit_code=1)
        run_id = latest.id

    output_raw = _optional_str(getattr(args, "output", None))
    output_path = None if output_raw is None else _resolve_optional_path(output_raw, repo_root)

    key_logs = _string_sequence(getattr(args, "key_logs", None))

    ledger = EvidenceLedger(state_db, evidence_root=evidence_root, repo_root=repo_root)
    try:
        bundle = ledger.export_audit_bundle(
            run_id=run_id,
            output_path=output_path,
            key_log_paths=key_logs,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise CLIError(str(exc), exit_code=2) from exc

    payload: dict[str, object] = {
        "command": "export",
        "run_id": bundle.run_id,
        "bundle_path": _display_path(bundle.bundle_path, repo_root),
        "member_count": len(bundle.member_names),
        "member_names": list(bundle.member_names),
        "evidence_ids": list(bundle.evidence_ids),
    }

    lines = [
        f"Exported run: {bundle.run_id}",
        f"Bundle path: {_display_path(bundle.bundle_path, repo_root)}",
        f"Members: {len(bundle.member_names)}",
        f"Evidence archives: {len(bundle.evidence_ids)}",
    ]

    _emit(payload, lines, json_mode=_flag(args, "json"))
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    repo_root = _repo_root(args)
    config = _load_effective_config(args)
    dry_run = _flag(args, "dry_run")

    workspace_root = _path_from_config(config, ("paths", "workspace_root"), repo_root)
    evidence_root = _path_from_config(config, ("paths", "evidence_root"), repo_root)
    state_db_path = _state_db_path(config, repo_root)
    artifacts_root = (repo_root / "artifacts").resolve()

    targets: list[Path] = [workspace_root, evidence_root, artifacts_root]
    state_dir = state_db_path.parent.resolve()
    if _is_within(state_dir, repo_root):
        targets.append(state_dir)
    else:
        targets.extend(
            [
                state_db_path,
                Path(f"{state_db_path}-wal"),
                Path(f"{state_db_path}-shm"),
            ]
        )

    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for candidate in targets:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_targets.append(resolved)

    unique_targets.sort(key=lambda item: (len(item.parts), item.as_posix()), reverse=True)

    removed: list[str] = []
    missing: list[str] = []
    errors: list[str] = []

    for target in unique_targets:
        rendered = _display_path(target, repo_root)
        if not target.exists() and not target.is_symlink():
            missing.append(rendered)
            continue
        if _is_protected_delete_target(target, repo_root):
            errors.append(f"refused to delete protected path: {rendered}")
            continue

        if dry_run:
            removed.append(rendered)
            continue

        try:
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            removed.append(rendered)
        except OSError as exc:
            errors.append(f"{rendered}: {exc}")

    payload: dict[str, object] = {
        "command": "clean",
        "dry_run": dry_run,
        "removed": removed,
        "missing": missing,
        "errors": errors,
    }

    lines = [
        f"Clean dry_run: {str(dry_run).lower()}",
        f"Removed: {len(removed)}",
        f"Missing: {len(missing)}",
        f"Errors: {len(errors)}",
    ]
    if errors:
        lines.append("Error details:")
        lines.extend(f"- {item}" for item in errors)

    _emit(payload, lines, json_mode=_flag(args, "json"))
    return 1 if errors else 0


def _repo_root(args: argparse.Namespace) -> Path:
    raw = _require_str(getattr(args, "repo_root", None), "repo_root")
    candidate = Path(raw).expanduser().resolve()
    if not candidate.exists() or not candidate.is_dir():
        raise CLIError(f"repo root is not a directory: {candidate}", exit_code=2)
    return candidate


def _load_effective_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = _optional_str(getattr(args, "config_path", None))
    profile = _optional_str(getattr(args, "profile", None))

    try:
        loaded = load_config(config_path, profile=profile)
        validated = assert_valid_config(loaded, active_profile=profile)
    except (ConfigLoadError, ConfigValidationError) as exc:
        raise CLIError(str(exc), exit_code=2) from exc

    return {key: value for key, value in validated.items()}


def _resolve_spec_path(spec_arg: str, repo_root: Path) -> Path:
    candidate = Path(spec_arg).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise CLIError(f"spec path not found: {resolved}", exit_code=2)
    return resolved


def _resolve_optional_path(path_arg: str, repo_root: Path) -> Path:
    candidate = Path(path_arg).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _state_db_path(config: Mapping[str, object], repo_root: Path) -> Path:
    return _path_from_config(
        config, ("paths", "state_db"), repo_root, default=DEFAULT_STATE_DB_PATH
    )


def _path_from_config(
    config: Mapping[str, object],
    path: Sequence[str],
    repo_root: Path,
    *,
    default: str | None = None,
) -> Path:
    current: object = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            current = None
            break
        current = current[part]

    value: str
    if isinstance(current, str) and current.strip():
        value = current.strip()
    elif default is not None:
        value = default
    else:
        joined = ".".join(path)
        raise CLIError(f"missing config path: {joined}", exit_code=2)

    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _build_routing_ladder(
    config: Mapping[str, object],
) -> tuple[tuple[RoleRoutingSnapshot, ...], tuple[str, ...]]:
    registry = RoleRegistry.from_config(config)
    catalog = load_model_catalog()

    snapshots: list[RoleRoutingSnapshot] = []
    warning_messages: list[str] = []
    seen_warnings: set[tuple[str, str, str]] = set()

    for role in registry.roles:
        role_steps: list[RoutingStepSnapshot] = []
        for index, step in enumerate(role.escalation_policy.steps, start=1):
            availability = _model_availability(catalog, provider=step.provider, model=step.model)
            role_steps.append(
                RoutingStepSnapshot(
                    stage_index=index,
                    attempts=step.attempts,
                    provider=step.provider,
                    model=step.model,
                    availability=availability,
                )
            )
            if availability in UNSAFE_MODEL_AVAILABILITY:
                warning_key = (step.provider, step.model, availability)
                if warning_key not in seen_warnings:
                    seen_warnings.add(warning_key)
                    warning_messages.append(
                        "model availability warning: "
                        f"provider={step.provider} model={step.model} "
                        f"is marked {availability} in model catalog"
                    )

        snapshots.append(
            RoleRoutingSnapshot(
                role=role.name,
                enabled=role.enabled,
                steps=tuple(role_steps),
            )
        )

    return tuple(snapshots), tuple(warning_messages)


def _model_availability(catalog: ModelCatalog, *, provider: str, model: str) -> str:
    capabilities = catalog.get(model, provider=provider)
    if capabilities is None:
        return "unknown"
    raw = (capabilities.availability or "").strip().lower()
    return raw or "unknown"


def _routing_payload(snapshots: Sequence[RoleRoutingSnapshot]) -> list[dict[str, object]]:
    return [
        {
            "role": snapshot.role,
            "enabled": snapshot.enabled,
            "steps": [
                {
                    "stage_index": step.stage_index,
                    "attempts": step.attempts,
                    "provider": step.provider,
                    "model": step.model,
                    "availability": step.availability,
                }
                for step in snapshot.steps
            ],
        }
        for snapshot in snapshots
    ]


def _format_routing_lines(snapshots: Sequence[RoleRoutingSnapshot]) -> list[str]:
    lines = ["Routing ladder:"]
    for snapshot in snapshots:
        segments = [f"{step.provider}/{step.model} x{step.attempts}" for step in snapshot.steps]
        ladder = " -> ".join(segments)
        state = "enabled" if snapshot.enabled else "disabled"
        lines.append(f"- {snapshot.role} [{state}]: {ladder}")
    return lines


def _resolve_status_run(run_repo: RunRepo, run_id: str | None) -> Run | None:
    if run_id is None:
        return _latest_run(run_repo)
    run = _safe_get_run(run_repo, run_id)
    if run is None:
        raise CLIError(f"run not found: {run_id}", exit_code=2)
    return run


def _latest_run(run_repo: RunRepo) -> Run | None:
    runs = run_repo.list(limit=1)
    if not runs:
        return None
    return runs[0]


def _safe_get_run(run_repo: RunRepo, run_id: str) -> Run | None:
    try:
        return run_repo.get(run_id)
    except ValueError:
        return None


def _safe_get_work_item(work_item_repo: WorkItemRepo, work_item_id: str) -> WorkItem | None:
    try:
        return work_item_repo.get(work_item_id)
    except ValueError:
        return None


def _run_id_for_work_item(state_db: StateDB, work_item_id: str) -> str | None:
    row = state_db.query_one(
        "SELECT run_id FROM work_items WHERE id = ?",
        (work_item_id,),
    )
    if row is None:
        return None
    value = row.get("run_id")
    if not isinstance(value, str):
        return None
    return value


def _summarize_run(
    run_record: Run,
    *,
    work_items: Sequence[WorkItem],
    incidents: Sequence[object],
    merges: Sequence[object],
) -> dict[str, object]:
    budget_usage = _budget_usage_from_metadata(run_record.metadata)
    warnings = _warning_messages_from_metadata(run_record.metadata)

    return {
        "id": run_record.id,
        "spec_path": run_record.spec_path,
        "status": _run_status_text(run_record.status),
        "started_at": run_record.started_at.isoformat(),
        "finished_at": (
            None if run_record.finished_at is None else run_record.finished_at.isoformat()
        ),
        "work_item_count": len(work_items),
        "work_item_counts": _work_item_status_counts(work_items),
        "incident_count": len(incidents),
        "merge_count": len(merges),
        "budget_usage": budget_usage,
        "warnings": list(warnings),
    }


def _work_item_status_counts(work_items: Sequence[WorkItem]) -> dict[str, int]:
    counts = Counter(item.status.value for item in work_items)
    return {key: counts[key] for key in sorted(counts)}


def _budget_usage_from_metadata(metadata: Mapping[str, object]) -> dict[str, int | float]:
    usage_raw = metadata.get("budget_usage")
    if not isinstance(usage_raw, Mapping):
        return {"tokens_used": 0, "cost_usd": 0.0, "provider_calls": 0}

    tokens = usage_raw.get("tokens_used")
    cost = usage_raw.get("cost_usd")
    calls = usage_raw.get("provider_calls")

    return {
        "tokens_used": _non_negative_int(tokens),
        "cost_usd": _non_negative_float(cost),
        "provider_calls": _non_negative_int(calls),
    }


def _run_status_text(status: RunStatus | str) -> str:
    if isinstance(status, RunStatus):
        return status.value
    return status


def _warning_messages_from_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
    warnings_raw = metadata.get("warnings")
    if not isinstance(warnings_raw, list):
        return ()
    warnings = sorted({item for item in warnings_raw if isinstance(item, str)})
    return tuple(warnings)


def _is_protected_delete_target(path: Path, repo_root: Path) -> bool:
    resolved = path.resolve()
    root_path = Path(resolved.anchor)
    if resolved == root_path:
        return True
    if resolved == repo_root:
        return True
    return resolved == Path.home().resolve()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    if _is_within(resolved, repo_root):
        return resolved.relative_to(repo_root).as_posix()
    return resolved.as_posix()


def _emit(payload: Mapping[str, object], lines: Sequence[str], *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return
    print("\n".join(lines))


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CLIError(f"invalid {name}: expected string", exit_code=2)
    cleaned = value.strip()
    if not cleaned:
        raise CLIError(f"invalid {name}: value cannot be empty", exit_code=2)
    return cleaned


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CLIError("invalid optional string argument", exit_code=2)
    cleaned = value.strip()
    return cleaned or None


def _flag(args: argparse.Namespace, name: str) -> bool:
    value = getattr(args, name, False)
    return bool(value)


def _string_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        cleaned = value.strip()
        return () if not cleaned else (cleaned,)
    if not isinstance(value, Sequence):
        raise CLIError("invalid sequence argument", exit_code=2)

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise CLIError("invalid sequence argument", exit_code=2)
        cleaned = item.strip()
        if cleaned:
            parsed.append(cleaned)
    return tuple(parsed)


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _non_negative_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if parsed >= 0.0 else 0.0


__all__ = [
    "build_parser",
    "cli_entrypoint",
    "main",
    "run_cli",
]


if __name__ == "__main__":
    raise SystemExit(run_cli())
