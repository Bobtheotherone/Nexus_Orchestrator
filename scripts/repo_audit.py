"""
nexus-orchestrator — repository audit CLI wrapper

File: scripts/repo_audit.py
Last updated: 2026-02-13

Purpose
- Provide a stable, WSL-friendly entrypoint for repository blueprint generation and validation.
- Provide a canonical offline context-audit command for deterministic assembly inspection.

Expected CLI usage
- python scripts/repo_audit.py --summary --validate
- python scripts/repo_audit.py --print-phase-map
- python scripts/repo_audit.py --json
- python scripts/repo_audit.py --write-artifacts --validate --fail-on-warn

Functional requirements
- Must run without installation by resolving `src/` on `sys.path`.
- Must delegate to `nexus_orchestrator.repo_blueprint` without changing semantics.
- Must expose context-assembly audit output without requiring ad-hoc user scripts.

Non-functional requirements
- Keep wrapper minimal and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


if TYPE_CHECKING:
    from collections.abc import Sequence

    from nexus_orchestrator.synthesis_plane.context_assembler import ContextDoc


def _ensure_src_path() -> None:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))


def _load_main():
    _ensure_src_path()
    try:
        from nexus_orchestrator.repo_blueprint import main as loaded_main

        return loaded_main
    except ModuleNotFoundError:
        from nexus_orchestrator.repo_blueprint import main as loaded_main

        return loaded_main


def _hash_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fixed_randbytes(byte: int):
    seed = bytes([byte % 256])

    def provider(size: int) -> bytes:
        return seed * size

    return provider


def _build_audit_work_item(*, scope: tuple[str, ...]):
    from nexus_orchestrator.domain import ids
    from nexus_orchestrator.domain.models import (
        Budget,
        Constraint,
        ConstraintEnvelope,
        ConstraintSeverity,
        ConstraintSource,
        RiskTier,
        SandboxPolicy,
        WorkItem,
        WorkItemStatus,
    )

    work_item_id = ids.generate_work_item_id(
        timestamp_ms=1_700_000_000_000,
        randbytes=_fixed_randbytes(17),
    )
    created_at = datetime(2026, 2, 13, 0, 0, 0, tzinfo=UTC)
    constraint = Constraint(
        id="CON-SEC-9001",
        severity=ConstraintSeverity.MUST,
        category="security",
        description="Never leak secrets in audit output",
        checker_binding="security_checker",
        parameters={"mode": "strict"},
        requirement_links=("REQ-0001",),
        source=ConstraintSource.MANUAL,
        created_at=created_at,
    )
    envelope = ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=(constraint,),
        inherited_constraint_ids=(),
        compiled_at=created_at,
    )
    return WorkItem(
        id=work_item_id,
        title="Repository context audit",
        description="Build deterministic context pack for offline auditability checks",
        scope=scope,
        constraint_envelope=envelope,
        dependencies=(),
        status=WorkItemStatus.READY,
        risk_tier=RiskTier.MEDIUM,
        budget=Budget(
            max_tokens=2_000,
            max_cost_usd=0.0,
            max_iterations=1,
            max_wall_clock_seconds=60,
        ),
        sandbox_policy=SandboxPolicy(
            allow_network=False,
            allow_privileged_tools=False,
            allowed_tools=("python3", "rg"),
            read_only_paths=("src", "tests", "docs", "scripts", "constraints"),
            write_paths=("state",),
        ),
        requirement_links=("REQ-0001",),
        constraint_ids=(constraint.id,),
        evidence_ids=(),
        expected_artifacts=("artifacts/context_audit.json",),
        created_at=created_at,
        updated_at=created_at,
    )


class _AuditRetriever:
    def __init__(self, *, max_docs: int) -> None:
        self.max_docs = max(0, max_docs)

    def retrieve(
        self,
        *,
        work_item: object,
        index: object,
        token_budget: int,
        changed_paths: Sequence[str],
        preferred_contract_paths: Sequence[str],
    ) -> tuple[ContextDoc, ...]:
        del work_item, token_budget, changed_paths, preferred_contract_paths
        from nexus_orchestrator.knowledge_plane.indexer import RepositoryIndexer
        from nexus_orchestrator.synthesis_plane.context_assembler import ContextDoc
        from nexus_orchestrator.utils.hashing import sha256_text

        if not isinstance(index, RepositoryIndexer):
            return ()

        docs: list[ContextDoc] = []
        for path, record in sorted(index.files.items(), key=lambda item: item[0]):
            if len(docs) >= self.max_docs:
                break
            summary = json.dumps(
                {
                    "path": path,
                    "language": record.language,
                    "imports": list(record.imports[:16]),
                    "symbols": list(record.symbols[:16]),
                    "modules": list(record.modules[:8]),
                    "parse_error": record.parse_error,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            docs.append(
                ContextDoc(
                    name=Path(path).name,
                    path=path,
                    doc_type="dependency",
                    content=summary,
                    content_hash=sha256_text(summary),
                    why_included="audit: deterministic index summary",
                    metadata={
                        "bytes_total": len(summary.encode("utf-8")),
                        "truncated": record.truncated,
                    },
                )
            )
        return tuple(docs)


def _run_audit_context(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/repo_audit.py audit-context",
        description="Run deterministic context-assembly audit using RepositoryIndexer directly.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=10, help="Max audit-manifest rows to print."
    )
    parser.add_argument(
        "--max-retrieved-docs",
        type=int,
        default=12,
        help="Max optional docs synthesized from the repository index.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Repeatable scope prefix/path. Defaults to src/tests/docs/scripts/constraints.",
    )
    args = parser.parse_args(argv)

    _ensure_src_path()
    from nexus_orchestrator.knowledge_plane.indexer import RepositoryIndexer
    from nexus_orchestrator.synthesis_plane.context_assembler import ContextAssembler

    default_scope = ("src", "tests", "docs", "scripts", "constraints")
    scope = tuple(args.scope) if args.scope else default_scope

    indexer = RepositoryIndexer(repo_root=REPO_ROOT)
    assembler = ContextAssembler(
        repo_root=REPO_ROOT,
        indexer=indexer,
        retriever=_AuditRetriever(max_docs=max(0, args.max_retrieved_docs)),
    )
    work_item = _build_audit_work_item(scope=scope)

    first = assembler.assemble(
        work_item=work_item, role="auditor", index_refresh_key="audit-context"
    )
    second = assembler.assemble(
        work_item=work_item, role="auditor", index_refresh_key="audit-context"
    )

    first_rows = first.audit_manifest()
    second_rows = second.audit_manifest()
    first_manifest_digest = _hash_json(first_rows)
    second_manifest_digest = _hash_json(second_rows)

    language_counts = Counter(record.language for record in indexer.files.values())
    dependency_edges = sum(len(record.imports) for record in indexer.files.values())

    output = {
        "repo_root": REPO_ROOT.as_posix(),
        "index_stats": {
            "file_count": len(indexer.files),
            "language_adapter_counts": {
                language: count
                for language, count in sorted(language_counts.items(), key=lambda item: item[0])
            },
            "dependency_edges": dependency_edges,
        },
        "determinism": {
            "first": {"prompt_hash": first.prompt_hash, "manifest_digest": first_manifest_digest},
            "second": {
                "prompt_hash": second.prompt_hash,
                "manifest_digest": second_manifest_digest,
            },
            "matches": (
                first.prompt_hash == second.prompt_hash
                and first_manifest_digest == second_manifest_digest
            ),
        },
        "audit_manifest_rows": first_rows[: max(0, args.max_rows)],
    }
    print(json.dumps(output, sort_keys=True, indent=2, ensure_ascii=True))
    return 0 if output["determinism"]["matches"] else 2


def _print_wrapper_help() -> int:
    print("Usage:")
    print("  python scripts/repo_audit.py [repo_blueprint options]")
    print("  python scripts/repo_audit.py audit-context [--max-rows N]")
    print("")
    print("Examples:")
    print("  python scripts/repo_audit.py --summary --validate")
    print("  python scripts/repo_audit.py audit-context --max-rows 10")
    print("")
    print("Tip: run `python scripts/repo_audit.py audit-context --help` for audit options.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return _load_main()(args)
    if args[0] in {"-h", "--help"}:
        return _print_wrapper_help()
    if args[0] == "audit-context":
        return _run_audit_context(args[1:])
    return _load_main()(args)


if __name__ == "__main__":
    raise SystemExit(main())
