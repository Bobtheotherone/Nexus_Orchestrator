"""
nexus-orchestrator — provider adapters and shared provider API

File: src/nexus_orchestrator/synthesis_plane/providers/__init__.py
Last updated: 2026-02-13

Purpose
- Provider adapters (OpenAI/Codex, Anthropic/Claude, local model future).

What should be included in this file
- Abstract provider interface and concrete adapters.

Functional requirements
- Must normalize responses (text, tool calls, costs) into a common format.

Non-functional requirements
- Must never log secrets or raw API keys.
"""

from nexus_orchestrator.synthesis_plane.providers.anthropic_adapter import (
    AnthropicAdapter,
    AnthropicProvider,
)
from nexus_orchestrator.synthesis_plane.providers.base import (
    BackoffConfig,
    BaseProvider,
    BudgetSnapshot,
    ContextDoc,
    ContextDocument,
    JSONValue,
    ProviderAuthenticationError,
    ProviderAuthError,
    ProviderContextLengthError,
    ProviderError,
    ProviderFactory,
    ProviderInvalidRequestError,
    ProviderProtocol,
    ProviderRateLimitError,
    ProviderRegistry,
    ProviderRequest,
    ProviderResponse,
    ProviderResponseError,
    ProviderServiceError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUsage,
    RedactedTranscript,
    ToolCall,
    ToolDefinition,
    compute_backoff_delay,
    derive_idempotency_key,
    instantiate_provider_error,
    is_retryable_error,
    parse_tool_arguments,
    provider_error_subclasses,
    redact_transcript,
    run_with_retries,
)
from nexus_orchestrator.synthesis_plane.providers.openai_adapter import (
    OpenAIAdapter,
    OpenAIProvider,
)
from nexus_orchestrator.synthesis_plane.providers.tool_adapter import (
    ToolBackend,
    ToolProvider,
)
from nexus_orchestrator.synthesis_plane.providers.tool_detection import (
    ToolBackendInfo,
    detect_all_backends,
    detect_claude_code_cli,
    detect_codex_cli,
)

__all__ = [
    "AnthropicAdapter",
    "AnthropicProvider",
    "BackoffConfig",
    "BaseProvider",
    "BudgetSnapshot",
    "ContextDoc",
    "ContextDocument",
    "JSONValue",
    "OpenAIAdapter",
    "OpenAIProvider",
    "ProviderAuthenticationError",
    "ProviderAuthError",
    "ProviderContextLengthError",
    "ProviderError",
    "ProviderFactory",
    "ProviderInvalidRequestError",
    "ProviderProtocol",
    "ProviderRateLimitError",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderResponseError",
    "ProviderServiceError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "ProviderUsage",
    "RedactedTranscript",
    "ToolBackend",
    "ToolBackendInfo",
    "ToolCall",
    "ToolDefinition",
    "ToolProvider",
    "compute_backoff_delay",
    "detect_all_backends",
    "detect_claude_code_cli",
    "detect_codex_cli",
    "derive_idempotency_key",
    "instantiate_provider_error",
    "is_retryable_error",
    "parse_tool_arguments",
    "provider_error_subclasses",
    "redact_transcript",
    "run_with_retries",
]
