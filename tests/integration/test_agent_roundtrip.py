"""
nexus-orchestrator — test skeleton

File: tests/integration/test_agent_roundtrip.py
Last updated: 2026-02-11

Purpose
- Validate end-to-end agent roundtrip with mocked provider adapters.

What this test file should cover
- Prompt assembly → provider call → patch application → verification invocation.
- Budget and retry loops.
- Evidence ledger entry creation.

Functional requirements
- No real provider calls; mock network.

Non-functional requirements
- Deterministic; stable fixtures.
"""

from __future__ import annotations

import asyncio
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Budget,
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    RiskTier,
    Run,
    RunStatus,
    SandboxPolicy,
    WorkItem,
    WorkItemStatus,
)
from nexus_orchestrator.knowledge_plane.indexer import RepositoryIndexer
from nexus_orchestrator.knowledge_plane.retrieval import (
    RetrievalCandidate,
    retrieve_context_docs,
)
from nexus_orchestrator.persistence.repositories import (
    AttemptRepo,
    ProviderCallRepo,
    RunRepo,
    WorkItemRepo,
)
from nexus_orchestrator.persistence.state_db import StateDB
from nexus_orchestrator.synthesis_plane.context_assembler import ContextAssembler, ContextDoc
from nexus_orchestrator.synthesis_plane.dispatch import (
    DispatchController,
    DispatchRequest,
    ProviderBinding,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

try:
    from datetime import UTC
except ImportError:
    UTC = timezone.utc  # noqa: UP017


def _randbytes(seed: int) -> Callable[[int], bytes]:
    byte = (seed % 251) + 1

    def provider(size: int) -> bytes:
        return bytes([byte]) * size

    return provider


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@dataclass(slots=True)
class _IndexerAdapter:
    indexer: RepositoryIndexer

    def build(self, *, changed_paths: Sequence[str] | None = None) -> object:
        if changed_paths is None:
            self.indexer.reindex()
        else:
            self.indexer.update_paths(changed_paths)
        return self.indexer


@dataclass(slots=True)
class _RetrieverAdapter:
    repo_root: Path

    def retrieve(
        self,
        *,
        work_item: WorkItem,
        index: object,
        token_budget: int,
        changed_paths: Sequence[str],
        preferred_contract_paths: Sequence[str],
    ) -> object:
        del changed_paths, preferred_contract_paths
        if not isinstance(index, RepositoryIndexer):
            raise TypeError("expected RepositoryIndexer index snapshot")
        scope_set = set(work_item.scope)
        scope_stems = {Path(item).stem for item in work_item.scope}
        candidates: list[RetrievalCandidate] = []
        for relative_path, _record in sorted(index.files.items(), key=lambda item: item[0]):
            local_path = self.repo_root / relative_path
            content = local_path.read_text(encoding="utf-8")
            is_contract = "contract" in relative_path or relative_path.endswith(".md")
            is_direct_dependency = relative_path in scope_set
            similarity = (
                1.0 if Path(relative_path).stem in scope_stems and not is_direct_dependency else 0.0
            )
            candidates.append(
                RetrievalCandidate(
                    path=relative_path,
                    content=content,
                    is_contract=is_contract,
                    is_direct_dependency=is_direct_dependency,
                    similarity_score=similarity,
                    recency_score=0,
                )
            )

        bundle = retrieve_context_docs(candidates, max_tokens=token_budget)
        return tuple(
            ContextDoc(
                name=Path(doc.path).name,
                path=doc.path,
                doc_type=doc.tier.name.lower(),
                content=doc.content,
                content_hash=doc.content_sha256,
                why_included=doc.inclusion_rationale,
                metadata={"source_sha256": doc.source_sha256},
            )
            for doc in bundle.docs
        )


@dataclass(slots=True)
class _PatchProvider:
    patch: str
    calls: list[ProviderRequest] = field(default_factory=list)

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        return ProviderResponse(
            content=self.patch,
            usage=ProviderUsage(tokens=42, cost_usd=0.01, latency_ms=2),
            model=request.model,
            request_id="req-roundtrip-1",
        )


@dataclass(slots=True)
class _RecordingVerifier:
    calls: list[dict[str, str]] = field(default_factory=list)

    def verify(self, *, work_item_id: str, patched_file: Path, prompt_hash: str) -> dict[str, str]:
        self.calls.append(
            {
                "work_item_id": work_item_id,
                "patched_file": patched_file.as_posix(),
                "prompt_hash": prompt_hash,
            }
        )
        return {"status": "pass"}


def _make_work_item(now: datetime, *, work_item_id: str) -> WorkItem:
    constraint = Constraint(
        id="CON-SEC-0001",
        severity=ConstraintSeverity.MUST,
        category="security",
        description="No secret leakage",
        checker_binding="security_checker",
        parameters={},
        requirement_links=("REQ-0001",),
        source=ConstraintSource.MANUAL,
        created_at=now,
    )
    envelope = ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=(constraint,),
        inherited_constraint_ids=(),
        compiled_at=now,
    )
    return WorkItem(
        id=work_item_id,
        title="Update greeting function",
        description="Return personalized greeting",
        scope=("src/app.py",),
        constraint_envelope=envelope,
        dependencies=(),
        status=WorkItemStatus.READY,
        risk_tier=RiskTier.MEDIUM,
        budget=Budget(
            max_tokens=2_000,
            max_cost_usd=3.0,
            max_iterations=3,
            max_wall_clock_seconds=300,
        ),
        sandbox_policy=SandboxPolicy(
            allow_network=False,
            allow_privileged_tools=False,
            allowed_tools=("git",),
            read_only_paths=("docs",),
            write_paths=("src", "tests"),
        ),
        requirement_links=("REQ-0001",),
        constraint_ids=("CON-SEC-0001",),
        evidence_ids=(),
        expected_artifacts=("artifacts/patch.diff",),
        created_at=now,
        updated_at=now,
    )


def test_agent_roundtrip_assemble_dispatch_patch_verify_with_persistence(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _write(
        repo_root / "src" / "app.py",
        'def greet(name: str) -> str:\n    return "hello"\n',
    )
    _write(
        repo_root / "docs" / "contracts" / "app_contract.md",
        "# Contract\nFunction greet returns a greeting string.\n",
    )

    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)

    indexer = _IndexerAdapter(indexer=RepositoryIndexer(repo_root=repo_root))
    retriever = _RetrieverAdapter(repo_root=repo_root)
    assembler = ContextAssembler(
        repo_root=repo_root,
        indexer=indexer,
        retriever=retriever,
        interface_contracts={"src.app": "greet(name: str) -> str"},
    )

    now = datetime(2026, 2, 12, 15, 0, 0, tzinfo=UTC)
    run_id = ids.generate_run_id(timestamp_ms=1_800_000_000_000, randbytes=_randbytes(1))
    work_item_id = ids.generate_work_item_id(
        timestamp_ms=1_800_000_000_001,
        randbytes=_randbytes(2),
    )
    work_item = _make_work_item(now, work_item_id=work_item_id)

    state_db = StateDB(repo_root / "state.db")
    run_repo = RunRepo(state_db)
    work_item_repo = WorkItemRepo(state_db)
    attempt_repo = AttemptRepo(state_db)
    provider_call_repo = ProviderCallRepo(state_db)
    run_repo.add(
        Run(
            id=run_id,
            spec_path="docs/spec.md",
            status=RunStatus.RUNNING,
            started_at=now,
            finished_at=None,
            work_item_ids=(work_item.id,),
            budget=work_item.budget,
            risk_tier=RiskTier.MEDIUM,
            metadata={},
        )
    )
    work_item_repo.add(run_id, work_item)

    context_pack = assembler.assemble(work_item=work_item, role="implementer")
    patch = (
        "diff --git a/src/app.py b/src/app.py\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def greet(name: str) -> str:\n"
        '-    return "hello"\n'
        '+    return f"hello {name}"\n'
        "\n"
    )
    provider = _PatchProvider(patch=patch)
    controller = DispatchController(
        [
            ProviderBinding(
                name="mock-provider",
                model="mock-model",
                provider=provider,
            )
        ],
        attempt_repo=attempt_repo,
        provider_call_repo=provider_call_repo,
    )

    dispatch_output = asyncio.run(
        controller.dispatch(
            DispatchRequest(
                run_id=run_id,
                work_item_id=work_item.id,
                role="implementer",
                prompt=context_pack.prompt,
            )
        )
    )

    subprocess.run(
        ["git", "apply", "-"],
        cwd=repo_root,
        input=dispatch_output.content,
        text=True,
        check=True,
    )

    patched_text = (repo_root / "src" / "app.py").read_text(encoding="utf-8")
    assert 'return f"hello {name}"' in patched_text

    verifier = _RecordingVerifier()
    verify_result = verifier.verify(
        work_item_id=work_item.id,
        patched_file=repo_root / "src" / "app.py",
        prompt_hash=context_pack.prompt_hash,
    )
    assert verify_result["status"] == "pass"
    assert verifier.calls and verifier.calls[0]["prompt_hash"] == context_pack.prompt_hash

    attempts = attempt_repo.list_for_work_item(work_item.id)
    assert len(attempts) == 1
    assert attempts[0].prompt_hash == context_pack.prompt_hash

    assert dispatch_output.attempt_id is not None
    provider_calls = provider_call_repo.list_for_attempt(dispatch_output.attempt_id)
    assert len(provider_calls) == 1
    assert provider_calls[0].provider == "mock-provider"

    transcripts = controller.transcripts(work_item_id=work_item.id)
    assert len(transcripts) == 1
    assert transcripts[0].idempotency_key == dispatch_output.idempotency_key
