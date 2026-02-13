"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/integration_plane/__init__.py
Last updated: 2026-02-11

Purpose
- Integration plane: Git operations, workspace management, merge queue, conflict resolution, rollback.

What should be included in this file
- Public interfaces for creating branches, applying patches, and merging with evidence metadata.

Functional requirements
- Must serialize merges and enforce scope/ownership.

Non-functional requirements
- Must be robust to conflicts and partial failures.
"""

from nexus_orchestrator.integration_plane.conflict_resolution import (
    ConflictAuditEntry,
    ConflictClassification,
    ConflictInput,
    ConflictResolutionResult,
    ConflictResolver,
    ResolutionStatus,
    TrivialConflictProof,
)
from nexus_orchestrator.integration_plane.git_engine import (
    BranchResult,
    ChangedFileEntry,
    CommandResult,
    CommitResult,
    DryRunMergeResult,
    GitCommandError,
    GitEngine,
    GitEngineError,
    ProtectedBranchError,
    RebaseResult,
    RepoInitResult,
    RevertResult,
    SanitizationError,
    ScopeCheckResult,
    WorktreeResult,
)
from nexus_orchestrator.integration_plane.git_engine import (
    MergeResult as GitMergeResult,
)
from nexus_orchestrator.integration_plane.merge_queue import (
    MergeQueue,
    MergeQueueStateError,
    MergeStatus,
    QueueCandidate,
)
from nexus_orchestrator.integration_plane.merge_queue import (
    MergeResult as QueueMergeResult,
)
from nexus_orchestrator.integration_plane.workspace_manager import (
    Workspace,
    WorkspaceManager,
    WorkspacePaths,
)

__all__ = [
    "BranchResult",
    "ChangedFileEntry",
    "CommandResult",
    "CommitResult",
    "ConflictAuditEntry",
    "ConflictClassification",
    "ConflictInput",
    "ConflictResolutionResult",
    "ConflictResolver",
    "DryRunMergeResult",
    "GitCommandError",
    "GitEngine",
    "GitEngineError",
    "GitMergeResult",
    "MergeQueue",
    "MergeQueueStateError",
    "MergeStatus",
    "ProtectedBranchError",
    "QueueCandidate",
    "QueueMergeResult",
    "RebaseResult",
    "RepoInitResult",
    "ResolutionStatus",
    "RevertResult",
    "SanitizationError",
    "ScopeCheckResult",
    "TrivialConflictProof",
    "Workspace",
    "WorkspaceManager",
    "WorkspacePaths",
    "WorktreeResult",
]
