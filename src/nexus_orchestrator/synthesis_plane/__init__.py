"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/__init__.py
Last updated: 2026-02-11

Purpose
- Synthesis plane: LLM agent runtime, role system, prompt/context assembly, provider routing, and tool requests.

What should be included in this file
- Top-level interfaces for dispatching an agent attempt.

Functional requirements
- Must be provider-agnostic through adapters.

Non-functional requirements
- Must enforce strict prompt/context hygiene to mitigate injection.
"""

from nexus_orchestrator.synthesis_plane.context_assembler import (
    ContextAssembler,
    ContextAssemblerConfig,
    ContextDoc,
    ContextManifestEntry,
    ContextPack,
    IndexerProtocol,
    RetrieverProtocol,
    TruncationRecord,
)
from nexus_orchestrator.synthesis_plane.dispatch import (
    AttemptRepoLike,
    BudgetExceededError,
    DeterministicRateLimiter,
    DispatchBudget,
    DispatchController,
    DispatchError,
    DispatchFailedError,
    DispatchRequest,
    DispatchResult,
    ProviderBinding,
    ProviderCallError,
    ProviderCallRepoLike,
    ProviderProtocol,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    TranscriptEntry,
)

__all__ = [
    "AttemptRepoLike",
    "BudgetExceededError",
    "ContextAssembler",
    "ContextAssemblerConfig",
    "ContextDoc",
    "ContextManifestEntry",
    "ContextPack",
    "DeterministicRateLimiter",
    "DispatchBudget",
    "DispatchController",
    "DispatchError",
    "DispatchFailedError",
    "DispatchRequest",
    "DispatchResult",
    "IndexerProtocol",
    "ProviderBinding",
    "ProviderCallError",
    "ProviderCallRepoLike",
    "ProviderProtocol",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderUsage",
    "RetrieverProtocol",
    "TranscriptEntry",
    "TruncationRecord",
]
