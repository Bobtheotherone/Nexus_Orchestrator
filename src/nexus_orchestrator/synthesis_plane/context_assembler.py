"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/context_assembler.py
Last updated: 2026-02-11

Purpose
- Builds the context package for an agent attempt: contracts, relevant code slices, constraints, and failure history.

What should be included in this file
- Deterministic selection rules: dependencies first, then similarity, then recency.
- Context size budgeting and truncation strategy with rationale.
- Content hygiene filters: exclude untrusted content that looks like prompt injection.
- Ability to include structured summaries instead of raw files.

Functional requirements
- Must guarantee inclusion of the work item's contract + scope + constraint envelope.
- Must support incremental updates as files change in integration branch.

Non-functional requirements
- Must be fast and cacheable; avoid re-indexing whole repo for each attempt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from nexus_orchestrator.security.redaction import redact_text
from nexus_orchestrator.utils.hashing import sha256_text

if TYPE_CHECKING:
    from nexus_orchestrator.domain.models import WorkItem
    from nexus_orchestrator.spec_ingestion.spec_map import (
        InterfaceContract,
    )
    from nexus_orchestrator.spec_ingestion.spec_map import (
        SpecMap as IngestedSpecMap,
    )


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_SUSPICIOUS_INSTRUCTION_RE = re.compile(
    r"(?is)\b(ignore\s+previous|system\s+prompt|developer\s+message|"
    r"exfiltrat|reveal\s+secrets?|prompt\s+injection|do\s+not\s+obey)\b"
)
_SIGNATURE_RE = re.compile(
    r"^\s*(?:async\s+def|def|class|interface|export\s+function|export\s+class|"
    r"public\s+class|func|type|struct)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


class IndexerProtocol(Protocol):
    """Protocol for indexers used by ``ContextAssembler``."""

    def build(self, *, changed_paths: Sequence[str] | None = None) -> object:
        """Build or incrementally refresh the index."""


class RetrieverProtocol(Protocol):
    """Protocol for retrievers used by ``ContextAssembler``."""

    def retrieve(
        self,
        *,
        work_item: WorkItem,
        index: object,
        token_budget: int,
        changed_paths: Sequence[str],
        preferred_contract_paths: Sequence[str],
    ) -> object:
        """Return retrieval output containing context documents."""


@dataclass(frozen=True, slots=True)
class ContextDoc:
    """Single document chunk included in an agent context pack."""

    name: str
    path: str
    doc_type: str
    content: str
    content_hash: str
    why_included: str
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TruncationRecord:
    """Deterministic truncation manifest entry."""

    path: str
    content_hash: str
    included_byte_range: str
    reason: str


@dataclass(frozen=True, slots=True)
class ContextManifestEntry:
    """Audit row describing one included context document."""

    order: int
    path: str
    doc_type: str
    content_hash: str
    why_included: str
    token_estimate: int
    bytes_included: int
    bytes_total: int


@dataclass(frozen=True, slots=True)
class ContextPack:
    """Final deterministic context package dispatched to a provider."""

    role: str
    prompt: str
    prompt_hash: str
    docs: tuple[ContextDoc, ...]
    manifest: tuple[ContextManifestEntry, ...]
    truncation_manifest: tuple[TruncationRecord, ...]
    truncation_rationale: str | None
    metadata: dict[str, JSONValue]

    @property
    def token_estimate(self) -> int:
        """Return total estimated tokens from manifest rows."""

        return sum(item.token_estimate for item in self.manifest)

    def audit_manifest(self) -> list[dict[str, JSONValue]]:
        """Return a stable, JSON-serializable, secret-safe manifest view."""

        truncation_reasons: dict[str, tuple[str, ...]] = {}
        reasons_by_path: dict[str, set[str]] = {}
        for record in self.truncation_manifest:
            reasons_by_path.setdefault(record.path, set()).add(record.reason)
        for path, reasons in sorted(reasons_by_path.items(), key=lambda item: item[0]):
            truncation_reasons[path] = tuple(sorted(reasons))

        docs_by_order = {index: doc for index, doc in enumerate(self.docs)}
        rows: list[dict[str, JSONValue]] = []

        for entry in sorted(self.manifest, key=lambda item: item.order):
            doc = docs_by_order.get(entry.order)
            name = doc.name if doc is not None else PurePosixPath(entry.path).name
            metadata_raw: dict[str, JSONValue] = doc.metadata if doc is not None else {}
            reason_raw = entry.why_included if doc is None else doc.why_included

            path_reasons = truncation_reasons.get(entry.path, ())
            truncation_reason: str | None = "|".join(path_reasons) if path_reasons else None

            rows.append(
                {
                    "order": entry.order,
                    "doc_type": entry.doc_type,
                    "path": entry.path,
                    "name": name,
                    "content_hash": entry.content_hash,
                    "token_estimate": entry.token_estimate,
                    "bytes_included": entry.bytes_included,
                    "bytes_total": entry.bytes_total,
                    "inclusion_reason": redact_text(reason_raw),
                    "truncation_reason": truncation_reason,
                    "metadata": _sanitize_audit_metadata(metadata_raw),
                }
            )

        return rows


@dataclass(frozen=True, slots=True)
class ContextAssemblerConfig:
    """Configuration for deterministic context assembly."""

    max_context_tokens: int = 8_000
    summary_threshold_bytes: int = 8_192
    min_truncation_tokens: int = 32
    enforce_scope_filter: bool = True
    exclude_untrusted_suspicious: bool = True
    redact_untrusted_content: bool = True

    def __post_init__(self) -> None:
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        if self.summary_threshold_bytes <= 0:
            raise ValueError("summary_threshold_bytes must be > 0")
        if self.min_truncation_tokens <= 0:
            raise ValueError("min_truncation_tokens must be > 0")


class ContextAssembler:
    """Deterministic context package builder for synthesis dispatch."""

    def __init__(
        self,
        *,
        repo_root: Path | str,
        indexer: IndexerProtocol,
        retriever: RetrieverProtocol,
        config: ContextAssemblerConfig | None = None,
        token_estimator: Callable[[str], int] | None = None,
        spec_map: IngestedSpecMap | None = None,
        interface_contracts: Mapping[str, str] | None = None,
        prompt_renderer: Callable[[str, WorkItem, Sequence[ContextDoc]], str] | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._indexer = indexer
        self._retriever = retriever
        self._config = config or ContextAssemblerConfig()
        self._token_estimator = token_estimator or _default_token_estimator
        self._spec_map = spec_map
        self._interface_contracts = dict(interface_contracts or {})
        self._prompt_renderer = prompt_renderer or _render_prompt_fallback

        self._cached_index: object | None = None
        self._cached_refresh_key: str | None = None
        self._index_refresh_count = 0

    @property
    def index_refresh_count(self) -> int:
        """Expose how many times index build/refresh was executed."""

        return self._index_refresh_count

    def assemble(
        self,
        *,
        work_item: WorkItem,
        role: str,
        changed_paths: Sequence[str] = (),
        index_refresh_key: str | None = None,
        token_budget: int | None = None,
    ) -> ContextPack:
        """
        Assemble a deterministic context pack for one work item.

        The pack always includes work-item contract, scope, constraint envelope,
        budget metadata, and relevant interface contracts.
        """

        effective_budget = self._resolve_budget(work_item=work_item, token_budget=token_budget)
        index_snapshot = self._resolve_index(
            changed_paths=changed_paths,
            refresh_key=index_refresh_key,
        )

        mandatory_docs = self._build_mandatory_docs(work_item=work_item)
        contract_docs = self._build_interface_contract_docs(work_item=work_item)
        always_docs = mandatory_docs + contract_docs

        reserved_tokens = sum(self._estimate_doc_tokens(doc) for doc in always_docs)
        retrieval_budget = max(1, effective_budget - reserved_tokens)

        retrieved_docs = self._coerce_retrieval_docs(
            self._retriever.retrieve(
                work_item=work_item,
                index=index_snapshot,
                token_budget=retrieval_budget,
                changed_paths=tuple(changed_paths),
                preferred_contract_paths=tuple(doc.path for doc in contract_docs),
            )
        )
        filtered_docs = self._filter_optional_docs(work_item=work_item, docs=retrieved_docs)
        normalized_optional = tuple(
            self._normalize_doc(doc, allow_summary=True) for doc in filtered_docs
        )

        all_docs, truncation_manifest, rationale = self._budget_docs(
            mandatory_docs=tuple(always_docs),
            optional_docs=normalized_optional,
            token_budget=effective_budget,
        )
        prompt = self._render_prompt(role=role, work_item=work_item, docs=all_docs)
        prompt_hash = sha256_text(prompt)

        manifest = tuple(
            ContextManifestEntry(
                order=index,
                path=doc.path,
                doc_type=doc.doc_type,
                content_hash=doc.content_hash,
                why_included=doc.why_included,
                token_estimate=self._estimate_doc_tokens(doc),
                bytes_included=len(doc.content.encode("utf-8")),
                bytes_total=_metadata_bytes_total(doc),
            )
            for index, doc in enumerate(all_docs)
        )

        return ContextPack(
            role=role,
            prompt=prompt,
            prompt_hash=prompt_hash,
            docs=all_docs,
            manifest=manifest,
            truncation_manifest=truncation_manifest,
            truncation_rationale=rationale,
            metadata={
                "work_item_id": work_item.id,
                "token_budget": effective_budget,
                "token_estimate": sum(item.token_estimate for item in manifest),
                "refresh_key": index_refresh_key,
                "changed_paths": list(changed_paths),
            },
        )

    def _resolve_budget(self, *, work_item: WorkItem, token_budget: int | None) -> int:
        max_from_work_item = work_item.budget.max_tokens
        if token_budget is None:
            return min(max_from_work_item, self._config.max_context_tokens)
        if token_budget <= 0:
            raise ValueError("token_budget must be > 0")
        return min(token_budget, max_from_work_item, self._config.max_context_tokens)

    def _resolve_index(self, *, changed_paths: Sequence[str], refresh_key: str | None) -> object:
        if self._cached_index is None:
            self._cached_index = self._build_or_refresh_index(changed_paths=None)
            self._cached_refresh_key = refresh_key
            self._index_refresh_count += 1
            return self._cached_index

        if refresh_key != self._cached_refresh_key:
            self._cached_index = self._build_or_refresh_index(changed_paths=None)
            self._cached_refresh_key = refresh_key
            self._index_refresh_count += 1
            return self._cached_index

        if changed_paths:
            self._cached_index = self._build_or_refresh_index(changed_paths=tuple(changed_paths))
            self._index_refresh_count += 1
            return self._cached_index

        return self._cached_index

    def _build_or_refresh_index(self, *, changed_paths: Sequence[str] | None) -> object:
        build = getattr(self._indexer, "build", None)
        if callable(build):
            normalized = tuple(changed_paths) if changed_paths is not None else None
            return build(changed_paths=normalized)

        reindex = getattr(self._indexer, "reindex", None)
        update_paths = getattr(self._indexer, "update_paths", None)

        if changed_paths:
            if callable(update_paths):
                update_paths(tuple(changed_paths))
                return self._indexer
            if callable(reindex):
                reindex()
                return self._indexer
            raise TypeError(
                "indexer must expose build(changed_paths=...) or update_paths/reindex methods"
            )

        if callable(reindex):
            reindex()
            return self._indexer

        raise TypeError(
            "indexer must expose build(changed_paths=...) or update_paths/reindex methods"
        )

    def _build_mandatory_docs(self, *, work_item: WorkItem) -> tuple[ContextDoc, ...]:
        constraint_lines = [
            (
                f"- {constraint.id} | severity={constraint.severity.value} "
                f"| category={constraint.category} | checker={constraint.checker_binding} "
                f"| {constraint.description}"
            )
            for constraint in sorted(work_item.constraint_envelope.constraints, key=lambda c: c.id)
        ]

        contract_content = "\n".join(
            (
                f"work_item_id: {work_item.id}",
                f"title: {work_item.title}",
                f"goal: {work_item.description}",
                f"risk_tier: {work_item.risk_tier.value}",
                f"requirement_links: {', '.join(work_item.requirement_links) or '(none)'}",
                f"dependencies: {', '.join(work_item.dependencies) or '(none)'}",
            )
        )
        scope_content = "\n".join(work_item.scope)
        envelope_content = "\n".join(
            (
                f"work_item_id: {work_item.constraint_envelope.work_item_id}",
                f"compiled_at: {work_item.constraint_envelope.compiled_at.isoformat()}",
                "constraints:",
                *constraint_lines,
            )
        )
        budget_content = "\n".join(
            (
                f"max_tokens: {work_item.budget.max_tokens}",
                f"max_cost_usd: {work_item.budget.max_cost_usd}",
                f"max_iterations: {work_item.budget.max_iterations}",
                f"max_wall_clock_seconds: {work_item.budget.max_wall_clock_seconds}",
            )
        )

        return (
            _build_doc(
                name="work-item-contract",
                path="_meta/work_item_contract.txt",
                doc_type="work_item_contract",
                content=contract_content,
                why="mandatory: work item contract/goal",
            ),
            _build_doc(
                name="work-item-scope",
                path="_meta/work_item_scope.txt",
                doc_type="work_item_scope",
                content=scope_content,
                why="mandatory: declared ownership scope",
            ),
            _build_doc(
                name="constraint-envelope",
                path="_meta/constraint_envelope.txt",
                doc_type="constraint_envelope",
                content=envelope_content,
                why="mandatory: constraint envelope for auditability",
            ),
            _build_doc(
                name="budget-limits",
                path="_meta/budget_limits.txt",
                doc_type="budget",
                content=budget_content,
                why="mandatory: budget guardrails",
            ),
        )

    def _build_interface_contract_docs(self, *, work_item: WorkItem) -> tuple[ContextDoc, ...]:
        docs: list[ContextDoc] = []
        relevant_modules = _collect_relevant_modules(work_item=work_item)

        for module_name in sorted(self._interface_contracts):
            if not _module_relevant(module_name=module_name, relevant_modules=relevant_modules):
                continue
            content = self._interface_contracts[module_name]
            docs.append(
                _build_doc(
                    name=f"interface-{module_name}",
                    path=f"_contracts/{module_name}.txt",
                    doc_type="interface_contract",
                    content=content,
                    why=f"mandatory: relevant interface contract {module_name}",
                )
            )

        if self._spec_map is not None:
            for contract in sorted(self._spec_map.interfaces, key=lambda item: item.module_name):
                if not _module_relevant(
                    module_name=contract.module_name,
                    relevant_modules=relevant_modules,
                ):
                    continue
                docs.append(self._doc_from_ingested_contract(contract))

        deduped: dict[str, ContextDoc] = {}
        for doc in docs:
            deduped.setdefault(doc.path, doc)
        return tuple(deduped[path] for path in sorted(deduped))

    def _doc_from_ingested_contract(self, contract: InterfaceContract) -> ContextDoc:
        content = "\n".join(
            (
                f"module: {contract.module_name}",
                f"summary: {contract.summary or '(none)'}",
                f"dependencies: {', '.join(contract.dependencies) or '(none)'}",
                f"exposed_symbols: {', '.join(contract.exposed_symbols) or '(none)'}",
                f"requirement_links: {', '.join(contract.requirement_links) or '(none)'}",
            )
        )
        return _build_doc(
            name=f"spec-interface-{contract.module_name}",
            path=contract.source.path,
            doc_type="interface_contract",
            content=content,
            why=f"mandatory: relevant interface contract {contract.module_name}",
        )

    def _coerce_retrieval_docs(self, payload: object) -> tuple[ContextDoc, ...]:
        docs_raw: object = payload
        if hasattr(payload, "docs"):
            docs_raw = payload.docs
        if not isinstance(docs_raw, Sequence) or isinstance(docs_raw, (str, bytes)):
            raise TypeError("retriever output must be a sequence of docs or expose .docs")

        docs: list[ContextDoc] = []
        for item in docs_raw:
            docs.append(_coerce_context_doc(item))
        return tuple(docs)

    def _filter_optional_docs(
        self, *, work_item: WorkItem, docs: Sequence[ContextDoc]
    ) -> tuple[ContextDoc, ...]:
        filtered: list[ContextDoc] = []
        for doc in docs:
            if self._config.enforce_scope_filter and not _optional_doc_in_scope(
                work_item, doc.path
            ):
                continue
            if not _safe_relative_path(doc.path):
                continue
            filtered.append(doc)
        filtered.sort(key=lambda item: (item.doc_type, item.path, item.content_hash))
        return tuple(filtered)

    def _normalize_doc(self, doc: ContextDoc, *, allow_summary: bool) -> ContextDoc:
        trusted = _is_trusted_path(doc.path)
        suspicious = bool(_SUSPICIOUS_INSTRUCTION_RE.search(doc.content))

        if suspicious and not trusted and self._config.exclude_untrusted_suspicious:
            return ContextDoc(
                name=doc.name,
                path=doc.path,
                doc_type=doc.doc_type,
                content="",
                content_hash=doc.content_hash,
                why_included=f"{doc.why_included}; excluded_by_hygiene=true",
                metadata={**doc.metadata, "excluded_by_hygiene": True},
            )

        content = doc.content
        if suspicious and not trusted and self._config.redact_untrusted_content:
            content = redact_text(content)

        bytes_total = len(content.encode("utf-8"))
        if allow_summary and bytes_total > self._config.summary_threshold_bytes:
            content = _deterministic_summary(content=content, path=doc.path)
            why = f"{doc.why_included}; summary_mode=deterministic"
        else:
            why = doc.why_included

        return ContextDoc(
            name=doc.name,
            path=doc.path,
            doc_type=doc.doc_type,
            content=content,
            content_hash=sha256_text(content),
            why_included=why,
            metadata={**doc.metadata, "bytes_total": bytes_total, "trusted": trusted},
        )

    def _estimate_doc_tokens(self, doc: ContextDoc) -> int:
        return self._token_estimator(doc.content)

    def _budget_docs(
        self,
        *,
        mandatory_docs: tuple[ContextDoc, ...],
        optional_docs: tuple[ContextDoc, ...],
        token_budget: int,
    ) -> tuple[tuple[ContextDoc, ...], tuple[TruncationRecord, ...], str | None]:
        truncations: list[TruncationRecord] = []
        included: list[ContextDoc] = []
        remaining = token_budget

        mandatory_cost = sum(self._estimate_doc_tokens(doc) for doc in mandatory_docs)
        if mandatory_cost > token_budget:
            for index, doc in enumerate(mandatory_docs):
                docs_left = len(mandatory_docs) - index
                allocation = max(1, remaining // docs_left)
                truncated_doc, truncation = _truncate_doc_to_tokens(
                    doc,
                    max_tokens=allocation,
                )
                included.append(truncated_doc)
                remaining -= self._estimate_doc_tokens(truncated_doc)
                if truncation is not None:
                    truncations.append(truncation)
            mandatory_rationale = (
                "mandatory context exceeded budget; mandatory docs were truncated deterministically"
            )
            return tuple(included), tuple(truncations), mandatory_rationale

        for doc in mandatory_docs:
            included.append(doc)
            remaining -= self._estimate_doc_tokens(doc)

        for doc in optional_docs:
            if doc.metadata.get("excluded_by_hygiene") is True:
                continue
            if remaining <= 0:
                break
            cost = self._estimate_doc_tokens(doc)
            if cost <= remaining:
                included.append(doc)
                remaining -= cost
                continue

            if remaining < self._config.min_truncation_tokens:
                break

            truncated_doc, truncation = _truncate_doc_to_tokens(doc, max_tokens=remaining)
            included.append(truncated_doc)
            remaining -= self._estimate_doc_tokens(truncated_doc)
            if truncation is not None:
                truncations.append(truncation)
            break

        included_sorted = tuple(
            sorted(
                included,
                key=lambda item: (
                    _doc_sort_group(item.doc_type),
                    item.path,
                    item.content_hash,
                ),
            )
        )
        rationale: str | None = (
            "token budget reached; optional docs truncated/excluded deterministically"
            if truncations or len(included_sorted) < len(mandatory_docs) + len(optional_docs)
            else None
        )
        return included_sorted, tuple(truncations), rationale

    def _render_prompt(self, *, role: str, work_item: WorkItem, docs: Sequence[ContextDoc]) -> str:
        try:
            prompt = self._prompt_renderer(role, work_item, docs)
        except Exception:
            prompt = _render_prompt_fallback(role, work_item, docs)
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = _render_prompt_fallback(role, work_item, docs)
        return prompt


def _build_doc(*, name: str, path: str, doc_type: str, content: str, why: str) -> ContextDoc:
    return ContextDoc(
        name=name,
        path=path,
        doc_type=doc_type,
        content=content,
        content_hash=sha256_text(content),
        why_included=why,
        metadata={"bytes_total": len(content.encode("utf-8"))},
    )


def _collect_relevant_modules(*, work_item: WorkItem) -> frozenset[str]:
    tokens: set[str] = set()
    for scope_path in work_item.scope:
        pure = PurePosixPath(scope_path)
        if pure.suffix == ".py":
            module = ".".join(pure.with_suffix("").parts)
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            tokens.add(module)
        tokens.add(pure.stem)
    tokens.update(work_item.requirement_links)
    return frozenset(token for token in tokens if token)


def _module_relevant(*, module_name: str, relevant_modules: frozenset[str]) -> bool:
    normalized = module_name.strip()
    if not normalized:
        return False
    if normalized in relevant_modules:
        return True
    for token in relevant_modules:
        if "." in token:
            if normalized.startswith(f"{token}.") or token.startswith(f"{normalized}."):
                return True
            continue
        if normalized.rsplit(".", 1)[-1] == token:
            return True
    return False


def _coerce_context_doc(value: object) -> ContextDoc:
    if isinstance(value, ContextDoc):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        name = _coerce_non_empty_str(data.get("name"), "doc.name")
        path = _coerce_non_empty_str(data.get("path"), "doc.path")
        doc_type = _coerce_non_empty_str(
            data.get("doc_type", data.get("type", "context")),
            "doc.doc_type",
        )
        content = _coerce_str(data.get("content", ""), "doc.content")
        content_hash_raw = data.get("content_hash")
        content_hash = (
            _coerce_non_empty_str(content_hash_raw, "doc.content_hash")
            if content_hash_raw is not None
            else sha256_text(content)
        )
        why = _coerce_non_empty_str(
            data.get("why_included", data.get("reason", "retrieved context")),
            "doc.why_included",
        )
        metadata_raw = data.get("metadata", {})
        metadata: dict[str, JSONValue]
        if isinstance(metadata_raw, Mapping):
            metadata = {str(key): _coerce_json_value(item) for key, item in metadata_raw.items()}
        else:
            metadata = {}
        return ContextDoc(
            name=name,
            path=path,
            doc_type=doc_type,
            content=content,
            content_hash=content_hash,
            why_included=why,
            metadata=metadata,
        )
    raise TypeError(f"context doc must be ContextDoc or mapping, got {type(value).__name__}")


def _coerce_non_empty_str(value: object, path: str) -> str:
    text = _coerce_str(value, path).strip()
    if not text:
        raise ValueError(f"{path} must not be empty")
    return text


def _coerce_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string, got {type(value).__name__}")
    return value


def _coerce_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _coerce_json_value(v) for k, v in value.items()}
    return str(value)


def _default_token_estimator(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _metadata_bytes_total(doc: ContextDoc) -> int:
    value = doc.metadata.get("bytes_total")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return len(doc.content.encode("utf-8"))


def _sanitize_audit_metadata(metadata: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    sanitized: dict[str, JSONValue] = {}
    for key in sorted(metadata):
        sanitized[key] = _sanitize_audit_value(metadata[key])
    return sanitized


def _sanitize_audit_value(value: JSONValue) -> JSONValue:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_sanitize_audit_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sanitize_audit_value(item) for key, item in sorted(value.items())}
    return str(value)


def _optional_doc_in_scope(work_item: WorkItem, path: str) -> bool:
    if path.startswith("_meta/") or path.startswith("_contracts/"):
        return True
    normalized_path = path.strip("/")
    for scope in work_item.scope:
        normalized_scope = scope.strip("/")
        if normalized_path == normalized_scope:
            return True
        if normalized_path.startswith(f"{normalized_scope}/"):
            return True
        scope_parent = str(PurePosixPath(normalized_scope).parent)
        if scope_parent and scope_parent != "." and normalized_path.startswith(f"{scope_parent}/"):
            return True
    return False


def _safe_relative_path(path: str) -> bool:
    if not path:
        return False
    pure = PurePosixPath(path)
    if pure.is_absolute():
        return False
    return all(part not in {"", ".", ".."} for part in pure.parts)


def _is_trusted_path(path: str) -> bool:
    lowered = path.lower()
    trusted_prefixes = (
        "docs/schemas/",
        "docs/prompts/templates/",
        "constraints/registry/",
        "_meta/",
        "_contracts/",
    )
    return lowered.startswith(trusted_prefixes)


def _doc_sort_group(doc_type: str) -> int:
    if doc_type in {"work_item_contract", "work_item_scope", "constraint_envelope", "budget"}:
        return 0
    if doc_type == "interface_contract":
        return 1
    if doc_type in {"dependency", "direct_dependency"}:
        return 2
    if doc_type in {"similar", "similar_module"}:
        return 3
    if doc_type in {"recent", "recent_change"}:
        return 4
    return 5


def _truncate_doc_to_tokens(
    doc: ContextDoc, *, max_tokens: int
) -> tuple[ContextDoc, TruncationRecord | None]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    max_chars = max_tokens * 4
    encoded = doc.content.encode("utf-8")
    if len(encoded) <= max_chars:
        return doc, None

    truncated_bytes = encoded[:max_chars]
    truncated_content = truncated_bytes.decode("utf-8", errors="ignore")
    updated = ContextDoc(
        name=doc.name,
        path=doc.path,
        doc_type=doc.doc_type,
        content=truncated_content,
        content_hash=sha256_text(truncated_content),
        why_included=f"{doc.why_included}; truncated=true",
        metadata={**doc.metadata, "bytes_total": len(encoded)},
    )
    truncation = TruncationRecord(
        path=doc.path,
        content_hash=doc.content_hash,
        included_byte_range=f"0:{len(truncated_content.encode('utf-8'))}",
        reason="token_budget_limit",
    )
    return updated, truncation


def _deterministic_summary(*, content: str, path: str) -> str:
    lines = content.splitlines()
    header = []
    for line in lines[:20]:
        stripped = line.strip()
        if stripped:
            header.append(stripped)
        if len(header) >= 8:
            break

    imports: list[str] = []
    signatures: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith(("import ", "from ", "#include ", "use ")):
            imports.append(stripped)
        signature_match = _SIGNATURE_RE.match(raw)
        if signature_match:
            signatures.append(signature_match.group(0).strip())

    unique_imports = sorted(set(imports))[:40]
    unique_signatures = sorted(set(signatures))[:60]

    payload: dict[str, Any] = {
        "path": path,
        "summary_type": "deterministic_structured",
        "header": header,
        "imports": unique_imports,
        "public_api_signatures": unique_signatures,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _render_prompt_fallback(role: str, work_item: WorkItem, docs: Sequence[ContextDoc]) -> str:
    sections: list[str] = []
    sections.append(f"ROLE: {role}")
    sections.append(f"WORK_ITEM_ID: {work_item.id}")
    sections.append(f"TITLE: {work_item.title}")
    sections.append(f"GOAL: {work_item.description}")
    sections.append("CONTEXT_BEGIN")
    for doc in docs:
        sections.append(
            "\n".join(
                (
                    f"[DOC path={doc.path} type={doc.doc_type} hash={doc.content_hash}]",
                    f"why: {doc.why_included}",
                    doc.content,
                    "[/DOC]",
                )
            )
        )
    sections.append("CONTEXT_END")
    return "\n".join(sections)


__all__ = [
    "ContextAssembler",
    "ContextAssemblerConfig",
    "ContextDoc",
    "ContextManifestEntry",
    "ContextPack",
    "IndexerProtocol",
    "RetrieverProtocol",
    "TruncationRecord",
]
