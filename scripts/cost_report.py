"""
nexus-orchestrator — deterministic provider cost reporting.

Purpose
- Generate cost usage reports from StateDB provider-call records.
- Attribute usage to run/work item/provider/model/phase with stable output ordering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if TYPE_CHECKING:
    from nexus_orchestrator.domain.models import Attempt, Run, WorkItem
    from nexus_orchestrator.persistence.repositories import (
        AttemptRepo,
        ProviderCallRecord,
        ProviderCallRepo,
        RunRepo,
        WorkItemRepo,
    )

JSONScalar = str | int | float | bool | None
JSONOutput = JSONScalar | list["JSONOutput"] | dict[str, "JSONOutput"]

_PAGE_SIZE = 500


def _ensure_src_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


@dataclass(slots=True)
class _Aggregate:
    run_id: str
    work_item_id: str
    phase: str
    provider: str
    model: str
    call_count: int = 0
    attempt_ids: set[str] = field(default_factory=set)
    tokens_total: int = 0
    cost_usd_total: float = 0.0
    latency_ms_total: int = 0
    first_call_at: datetime | None = None
    last_call_at: datetime | None = None

    def add(self, *, attempt_id: str, record: ProviderCallRecord) -> None:
        self.call_count += 1
        self.attempt_ids.add(attempt_id)
        self.tokens_total += int(record.tokens)
        self.cost_usd_total += float(record.cost_usd)
        self.latency_ms_total += int(record.latency_ms)

        created_at = _ensure_utc(record.created_at)
        if self.first_call_at is None or created_at < self.first_call_at:
            self.first_call_at = created_at
        if self.last_call_at is None or created_at > self.last_call_at:
            self.last_call_at = created_at


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic provider cost reports from StateDB.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("state") / "nexus.sqlite",
        help="State DB path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run ID scope.",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Optional lower bound timestamp (ISO-8601 or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Optional upper bound timestamp (ISO-8601 or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Repeatable provider filter (exact match).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Repeatable model filter (exact match).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args(argv)


def _parse_time_boundary(value: str, *, is_end: bool) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp boundary must not be empty")

    if len(text) == 10:
        try:
            day = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"invalid date boundary {value!r}") from exc
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        if is_end:
            return start + timedelta(days=1) - timedelta(microseconds=1)
        return start

    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid datetime boundary {value!r}") from exc
    return _ensure_utc(parsed)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalize_filter(values: Sequence[str]) -> tuple[str, ...]:
    normalized = {item.strip() for item in values if item.strip()}
    return tuple(sorted(normalized))


def _list_runs(run_repo: RunRepo, *, run_id: str | None) -> tuple[Run, ...]:
    if run_id is not None:
        run = run_repo.get(run_id)
        if run is None:
            raise ValueError(f"run_id not found: {run_id}")
        return (run,)

    runs: list[Run] = []
    offset = 0
    while True:
        batch = run_repo.list(limit=_PAGE_SIZE, offset=offset)
        if not batch:
            break
        runs.extend(batch)
        offset += len(batch)
        if len(batch) < _PAGE_SIZE:
            break

    runs.sort(key=lambda item: (_ensure_utc(item.started_at), item.id))
    return tuple(runs)


def _list_work_items(work_item_repo: WorkItemRepo, *, run_id: str) -> tuple[WorkItem, ...]:
    items: list[WorkItem] = []
    offset = 0
    while True:
        batch = work_item_repo.list_for_run(run_id, limit=_PAGE_SIZE, offset=offset)
        if not batch:
            break
        items.extend(batch)
        offset += len(batch)
        if len(batch) < _PAGE_SIZE:
            break

    items.sort(key=lambda item: (_ensure_utc(item.created_at), item.id))
    return tuple(items)


def _list_attempts_for_work_item(
    attempt_repo: AttemptRepo,
    *,
    work_item_id: str,
) -> tuple[Attempt, ...]:
    attempts: list[Attempt] = []
    offset = 0
    while True:
        attempt_batch = attempt_repo.list_for_work_item(work_item_id, limit=_PAGE_SIZE, offset=offset)
        if not attempt_batch:
            break
        attempts.extend(attempt_batch)
        offset += len(attempt_batch)
        if len(attempt_batch) < _PAGE_SIZE:
            break

    attempts.sort(key=lambda item: (_ensure_utc(item.created_at), item.id))
    return tuple(attempts)


def _list_provider_calls_for_attempt(
    provider_repo: ProviderCallRepo,
    *,
    attempt_id: str,
) -> tuple[ProviderCallRecord, ...]:
    provider_calls: list[ProviderCallRecord] = []
    offset = 0
    while True:
        call_batch = provider_repo.list_for_attempt(attempt_id, limit=_PAGE_SIZE, offset=offset)
        if not call_batch:
            break
        provider_calls.extend(call_batch)
        offset += len(call_batch)
        if len(call_batch) < _PAGE_SIZE:
            break

    provider_calls.sort(key=lambda item: (_ensure_utc(item.created_at), item.id))
    return tuple(provider_calls)


def _phase_for_call(record: ProviderCallRecord, *, attempt_role: str) -> str:
    for key in ("phase", "stage", "step"):
        value = record.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return attempt_role.strip() if attempt_role.strip() else "unknown"


def _to_iso8601z(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _ensure_utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _to_json_output(value: object) -> JSONOutput:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, datetime):
        return _to_iso8601z(value)
    if isinstance(value, Mapping):
        out: dict[str, JSONOutput] = {}
        for key in sorted(value.keys(), key=lambda item: str(item)):
            out[str(key)] = _to_json_output(value[key])
        return out
    if isinstance(value, (list, tuple)):
        return [_to_json_output(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _emit_json(payload: Mapping[str, object]) -> None:
    print(
        json.dumps(
            _to_json_output(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


def _emit_text(payload: Mapping[str, object]) -> None:
    print(f"db_path: {payload['db_path']}")
    print(f"run_id: {payload['run_id']}")
    print(f"rows: {payload['row_count']}")
    print(
        "totals: "
        + f"calls={payload['total_calls']} "
        + f"attempts={payload['total_attempts']} "
        + f"tokens={payload['total_tokens']} "
        + f"cost_usd={payload['total_cost_usd']}"
    )

    rows_obj = payload.get("rows")
    rows = rows_obj if isinstance(rows_obj, list) else []
    if not rows:
        return

    print("report_rows:")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        print(
            "  - "
            + f"run={row.get('run_id')} "
            + f"wi={row.get('work_item_id')} "
            + f"phase={row.get('phase')} "
            + f"provider={row.get('provider')} "
            + f"model={row.get('model')} "
            + f"calls={row.get('call_count')} "
            + f"tokens={row.get('tokens_total')} "
            + f"cost_usd={row.get('cost_usd_total')}"
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "run_id",
        "work_item_id",
        "phase",
        "provider",
        "model",
        "call_count",
        "attempt_count",
        "tokens_total",
        "cost_usd_total",
        "latency_ms_total",
        "latency_ms_avg",
        "first_call_at",
        "last_call_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = args.db.expanduser().resolve()
    csv_path = args.csv.expanduser().resolve() if args.csv is not None else None

    _ensure_src_path()
    from nexus_orchestrator.persistence.repositories import (
        AttemptRepo,
        ProviderCallRepo,
        RunRepo,
        WorkItemRepo,
    )
    from nexus_orchestrator.persistence.state_db import StateDB

    try:
        since = _parse_time_boundary(args.since, is_end=False) if args.since is not None else None
        until = _parse_time_boundary(args.until, is_end=True) if args.until is not None else None
        if since is not None and until is not None and since > until:
            raise ValueError("--since must be <= --until")

        provider_filter = set(_normalize_filter(args.provider))
        model_filter = set(_normalize_filter(args.model))

        db = StateDB(db_path)
        run_repo = RunRepo(db)
        work_item_repo = WorkItemRepo(db)
        attempt_repo = AttemptRepo(db)
        provider_repo = ProviderCallRepo(db)

        aggregates: dict[tuple[str, str, str, str, str], _Aggregate] = {}

        for run in _list_runs(run_repo, run_id=args.run_id):
            for work_item in _list_work_items(work_item_repo, run_id=run.id):
                for attempt in _list_attempts_for_work_item(attempt_repo, work_item_id=work_item.id):
                    for call in _list_provider_calls_for_attempt(provider_repo, attempt_id=attempt.id):
                        created_at = _ensure_utc(call.created_at)
                        if since is not None and created_at < since:
                            continue
                        if until is not None and created_at > until:
                            continue

                        provider = call.provider
                        model = call.model if call.model is not None else attempt.model
                        phase = _phase_for_call(call, attempt_role=attempt.role)

                        if provider_filter and provider not in provider_filter:
                            continue
                        if model_filter and model not in model_filter:
                            continue

                        key = (run.id, work_item.id, phase, provider, model)
                        aggregate = aggregates.get(key)
                        if aggregate is None:
                            aggregate = _Aggregate(
                                run_id=run.id,
                                work_item_id=work_item.id,
                                phase=phase,
                                provider=provider,
                                model=model,
                            )
                            aggregates[key] = aggregate
                        aggregate.add(attempt_id=attempt.id, record=call)

        ordered_aggregates = [aggregates[key] for key in sorted(aggregates.keys())]

        row_payloads: list[dict[str, object]] = []
        for item in ordered_aggregates:
            latency_avg = 0.0
            if item.call_count > 0:
                latency_avg = item.latency_ms_total / item.call_count

            row_payloads.append(
                {
                    "run_id": item.run_id,
                    "work_item_id": item.work_item_id,
                    "phase": item.phase,
                    "provider": item.provider,
                    "model": item.model,
                    "call_count": item.call_count,
                    "attempt_count": len(item.attempt_ids),
                    "tokens_total": item.tokens_total,
                    "cost_usd_total": round(item.cost_usd_total, 12),
                    "latency_ms_total": item.latency_ms_total,
                    "latency_ms_avg": round(latency_avg, 6),
                    "first_call_at": _to_iso8601z(item.first_call_at),
                    "last_call_at": _to_iso8601z(item.last_call_at),
                }
            )

        total_calls = sum(item.call_count for item in ordered_aggregates)
        total_attempts = sum(len(item.attempt_ids) for item in ordered_aggregates)
        total_tokens = sum(item.tokens_total for item in ordered_aggregates)
        total_cost = round(sum(item.cost_usd_total for item in ordered_aggregates), 12)

        payload: dict[str, object] = {
            "db_path": db_path.as_posix(),
            "run_id": args.run_id,
            "since": _to_iso8601z(since),
            "until": _to_iso8601z(until),
            "provider_filter": sorted(provider_filter),
            "model_filter": sorted(model_filter),
            "row_count": len(row_payloads),
            "total_calls": total_calls,
            "total_attempts": total_attempts,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "rows": row_payloads,
        }

        if csv_path is not None:
            _write_csv(csv_path, row_payloads)
            payload["csv_path"] = csv_path.as_posix()

        if args.json:
            _emit_json(payload)
        else:
            _emit_text(payload)

        return 0
    except Exception as exc:  # noqa: BLE001
        error_payload = {
            "db_path": db_path.as_posix(),
            "run_id": args.run_id,
            "error": str(exc),
        }
        if args.json:
            _emit_json(error_payload)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
