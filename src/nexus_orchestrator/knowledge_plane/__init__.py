"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/knowledge_plane/__init__.py
Last updated: 2026-02-11

Purpose
- Knowledge plane: indexing, retrieval, constraint registry, evidence ledger access, personalization memory, and failure pattern mining.

What should be included in this file
- Interfaces to query codebase knowledge and historical runs.

Functional requirements
- Must serve context assembler and scheduler decisions.

Non-functional requirements
- Must be incremental and lightweight on a single machine.
"""

from nexus_orchestrator.knowledge_plane.indexer import (
    INDEX_SCHEMA_VERSION,
    AdapterParseError,
    FileAnalysis,
    GenericTextAdapter,
    IndexedFile,
    IndexerLoadError,
    IndexExcludes,
    LanguageAdapter,
    PythonAstAdapter,
    RepositoryIndexer,
)
from nexus_orchestrator.knowledge_plane.retrieval import (
    ContextDoc,
    RetrievalBundle,
    RetrievalCandidate,
    RetrievalTier,
    TokenEstimator,
    TruncationManifestEntry,
    classify_candidate_tier,
    estimate_tokens,
    rank_candidates,
    retrieve_context_docs,
)

__all__ = [
    "AdapterParseError",
    "ContextDoc",
    "FileAnalysis",
    "GenericTextAdapter",
    "INDEX_SCHEMA_VERSION",
    "IndexedFile",
    "IndexExcludes",
    "IndexerLoadError",
    "LanguageAdapter",
    "PythonAstAdapter",
    "RepositoryIndexer",
    "RetrievalBundle",
    "RetrievalCandidate",
    "RetrievalTier",
    "TokenEstimator",
    "TruncationManifestEntry",
    "classify_candidate_tier",
    "estimate_tokens",
    "rank_candidates",
    "retrieve_context_docs",
]
