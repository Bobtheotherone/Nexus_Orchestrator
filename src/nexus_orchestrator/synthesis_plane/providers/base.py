"""
nexus-orchestrator — provider base models and shared utilities

File: src/nexus_orchestrator/synthesis_plane/providers/base.py
Last updated: 2026-02-13

Purpose
- Abstract provider interface and common request/response models for LLM calls.

What should be included in this file
- Request fields: model, role, prompt, context docs, tool permissions, budget limits.
- Response fields: content, structured outputs, token usage, cost, latency, errors.
- Error taxonomy and retryability classification.

Functional requirements
- Must support idempotent retries with consistent attribution.

Non-functional requirements
- Must make it easy to add new providers without touching core logic.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import random as random_module
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from nexus_orchestrator.security.redaction import redact_text
from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog, load_model_catalog

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

SleepFn: TypeAlias = Callable[[float], Awaitable[None]]
RandomFn: TypeAlias = Callable[[], float]


def _validate_non_empty_str(value: str, field_name: str, *, strip: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip() if strip else value
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


def _validate_optional_str(value: str | None, field_name: str, *, strip: bool = True) -> str | None:
    if value is None:
        return None
    return _validate_non_empty_str(value, field_name, strip=strip)


def _coerce_json_value(value: object, *, path: str) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            out[key] = _coerce_json_value(item, path=f"{path}.{key}")
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_json_value(item, path=f"{path}[]") for item in value]
    raise TypeError(f"{path} must be JSON-serializable")


def _coerce_json_mapping(mapping: Mapping[str, object], *, path: str) -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings")
        out[key] = _coerce_json_value(value, path=f"{path}.{key}")
    return out


def _canonical_json(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ContextDocument:
    """Named context document passed to provider calls."""

    name: str
    content: str
    path: str | None = None
    doc_type: str = "untrusted"
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_non_empty_str(self.name, "ContextDocument.name"))
        if not isinstance(self.content, str):
            raise TypeError("ContextDocument.content must be a string")
        object.__setattr__(
            self,
            "path",
            _validate_optional_str(self.path, "ContextDocument.path")
            if self.path is not None
            else None,
        )
        object.__setattr__(
            self,
            "doc_type",
            _validate_non_empty_str(self.doc_type, "ContextDocument.doc_type"),
        )
        object.__setattr__(
            self,
            "metadata",
            _coerce_json_mapping(dict(self.metadata), path="ContextDocument.metadata"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "name": self.name,
            "content": self.content,
            "doc_type": self.doc_type,
            "metadata": dict(self.metadata),
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Tool contract exposed to providers that support function/tool calling."""

    name: str
    description: str | None = None
    json_schema: Mapping[str, JSONValue] = field(default_factory=dict)
    strict: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_non_empty_str(self.name, "ToolDefinition.name"))
        object.__setattr__(
            self,
            "description",
            _validate_optional_str(self.description, "ToolDefinition.description", strip=False),
        )
        object.__setattr__(
            self,
            "json_schema",
            _coerce_json_mapping(dict(self.json_schema), path="ToolDefinition.json_schema"),
        )
        object.__setattr__(self, "strict", bool(self.strict))

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "name": self.name,
            "json_schema": dict(self.json_schema),
            "strict": self.strict,
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class StructuredOutputDefinition:
    """Structured output contract exposed to providers."""

    name: str
    json_schema: Mapping[str, JSONValue] = field(default_factory=dict)
    strict: bool = True
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validate_non_empty_str(self.name, "StructuredOutputDefinition.name"),
        )
        object.__setattr__(
            self,
            "json_schema",
            _coerce_json_mapping(
                dict(self.json_schema),
                path="StructuredOutputDefinition.json_schema",
            ),
        )
        object.__setattr__(self, "strict", bool(self.strict))
        object.__setattr__(
            self,
            "description",
            _validate_optional_str(
                self.description,
                "StructuredOutputDefinition.description",
                strip=False,
            ),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "name": self.name,
            "json_schema": dict(self.json_schema),
            "strict": self.strict,
        }
        if self.description is not None:
            payload["description"] = self.description
        return payload


@dataclass(frozen=True, slots=True)
class RemoteMCPToolConfig:
    """Remote MCP server configuration for provider-hosted tool execution."""

    server_label: str
    server_url: str
    allowed_tools: tuple[str, ...] = ()
    require_approval: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    authorization_token_env: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "server_label",
            _validate_non_empty_str(self.server_label, "RemoteMCPToolConfig.server_label"),
        )
        object.__setattr__(
            self,
            "server_url",
            _validate_non_empty_str(self.server_url, "RemoteMCPToolConfig.server_url"),
        )

        normalized_tools: list[str] = []
        for index, tool_name in enumerate(self.allowed_tools):
            normalized_tools.append(
                _validate_non_empty_str(
                    tool_name,
                    f"RemoteMCPToolConfig.allowed_tools[{index}]",
                )
            )
        object.__setattr__(self, "allowed_tools", tuple(normalized_tools))

        object.__setattr__(
            self,
            "require_approval",
            _validate_optional_str(
                self.require_approval,
                "RemoteMCPToolConfig.require_approval",
            ),
        )

        normalized_headers: dict[str, str] = {}
        for key, value in self.headers.items():
            if not isinstance(key, str):
                raise TypeError("RemoteMCPToolConfig.headers keys must be strings")
            if not isinstance(value, str):
                raise TypeError("RemoteMCPToolConfig.headers values must be strings")
            normalized_key = _validate_non_empty_str(
                key,
                "RemoteMCPToolConfig.headers key",
                strip=False,
            )
            normalized_headers[normalized_key] = value
        object.__setattr__(self, "headers", normalized_headers)

        object.__setattr__(
            self,
            "authorization_token_env",
            _validate_optional_str(
                self.authorization_token_env,
                "RemoteMCPToolConfig.authorization_token_env",
            ),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "server_label": self.server_label,
            "server_url": self.server_url,
            "allowed_tools": list(self.allowed_tools),
            "headers": dict(self.headers),
        }
        if self.require_approval is not None:
            payload["require_approval"] = self.require_approval
        if self.authorization_token_env is not None:
            payload["authorization_token_env"] = self.authorization_token_env
        return payload


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Normalized tool call emitted by providers."""

    call_id: str
    name: str
    arguments: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "call_id", _validate_non_empty_str(self.call_id, "ToolCall.call_id")
        )
        object.__setattr__(self, "name", _validate_non_empty_str(self.name, "ToolCall.name"))
        object.__setattr__(
            self,
            "arguments",
            _coerce_json_mapping(dict(self.arguments), path="ToolCall.arguments"),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Budget envelope attached to a provider request."""

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("BudgetSnapshot.max_input_tokens must be > 0")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("BudgetSnapshot.max_output_tokens must be > 0")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("BudgetSnapshot.max_total_tokens must be > 0")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("BudgetSnapshot.max_cost_usd must be >= 0")

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {}
        if self.max_input_tokens is not None:
            payload["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens
        if self.max_total_tokens is not None:
            payload["max_total_tokens"] = self.max_total_tokens
        if self.max_cost_usd is not None:
            payload["max_cost_usd"] = self.max_cost_usd
        return payload


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    """Token/cost accounting for a provider response."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int | None = None
    cost_estimate_usd: float | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.cost_estimate_usd is not None and self.cost_estimate_usd < 0:
            raise ValueError("cost_estimate_usd must be >= 0")

    @property
    def cost_usd(self) -> float | None:
        """Backward-compatible alias for prior field name."""

        return self.cost_estimate_usd

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if self.cost_estimate_usd is not None:
            payload["cost_estimate_usd"] = self.cost_estimate_usd
        return payload


@dataclass(slots=True)
class ProviderRequest:
    """Provider-agnostic request payload."""

    model: str
    role_id: str
    system_prompt: str
    user_prompt: str
    temperature: float | None
    max_tokens: int | None
    context_docs: tuple[ContextDocument, ...]
    tool_permissions: tuple[ToolDefinition, ...]
    requested_tools: tuple[str, ...]
    reasoning_effort: str | None
    tool_choice: str | None
    remote_mcp_tools: tuple[RemoteMCPToolConfig, ...]
    structured_output: StructuredOutputDefinition | None
    idempotency_key: str | None
    budget: BudgetSnapshot | None
    metadata: Mapping[str, JSONValue]

    def __init__(
        self,
        *,
        model: str,
        role_id: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context_docs: Sequence[ContextDocument] = (),
        tool_permissions: Sequence[ToolDefinition] = (),
        requested_tools: Sequence[str] = (),
        reasoning_effort: str | None = None,
        tool_choice: str | None = None,
        remote_mcp_tools: Sequence[RemoteMCPToolConfig] = (),
        structured_output: StructuredOutputDefinition | None = None,
        idempotency_key: str | None = None,
        budget: BudgetSnapshot | None = None,
        metadata: Mapping[str, object] | None = None,
        # Backward-compatible names.
        role: str | None = None,
        prompt: str | None = None,
        context_documents: Sequence[ContextDocument] | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        allow_tool_calls: bool | None = None,
        max_output_tokens: int | None = None,
        structured_output_schema: Mapping[str, object] | None = None,
        structured_output_name: str | None = None,
        structured_output_strict: bool = True,
    ) -> None:
        self.model = _validate_non_empty_str(model, "ProviderRequest.model")

        resolved_role = role_id if role_id is not None else role
        if resolved_role is None:
            raise ValueError("ProviderRequest.role_id is required")
        self.role_id = _validate_non_empty_str(resolved_role, "ProviderRequest.role_id")

        resolved_user_prompt = user_prompt if user_prompt is not None else prompt
        if resolved_user_prompt is None:
            raise ValueError("ProviderRequest.user_prompt is required")

        self.system_prompt = system_prompt if system_prompt is not None else ""
        if not isinstance(self.system_prompt, str):
            raise TypeError("ProviderRequest.system_prompt must be a string")
        if not isinstance(resolved_user_prompt, str):
            raise TypeError("ProviderRequest.user_prompt must be a string")
        self.user_prompt = resolved_user_prompt

        self.temperature = temperature
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            raise ValueError("ProviderRequest.temperature must be between 0.0 and 2.0")

        resolved_max_tokens = max_tokens if max_tokens is not None else max_output_tokens
        self.max_tokens = resolved_max_tokens
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("ProviderRequest.max_tokens must be > 0")

        docs = context_documents if context_documents is not None else context_docs
        self.context_docs = tuple(docs)
        self.tool_permissions = tuple(tools if tools is not None else tool_permissions)

        if allow_tool_calls is False and self.tool_permissions:
            self.tool_permissions = ()

        normalized_requested_tools: list[str] = []
        for index, item in enumerate(requested_tools):
            if not isinstance(item, str):
                raise TypeError(f"ProviderRequest.requested_tools[{index}] must be a string")
            normalized_requested_tools.append(
                _validate_non_empty_str(item, f"ProviderRequest.requested_tools[{index}]")
            )
        self.requested_tools = tuple(normalized_requested_tools)
        self.reasoning_effort = _validate_optional_str(
            reasoning_effort,
            "ProviderRequest.reasoning_effort",
        )
        self.tool_choice = _validate_optional_str(tool_choice, "ProviderRequest.tool_choice")

        normalized_remote_mcp: list[RemoteMCPToolConfig] = []
        for index, server in enumerate(remote_mcp_tools):
            if not isinstance(server, RemoteMCPToolConfig):
                raise TypeError(
                    f"ProviderRequest.remote_mcp_tools[{index}] must be RemoteMCPToolConfig"
                )
            normalized_remote_mcp.append(server)
        self.remote_mcp_tools = tuple(normalized_remote_mcp)

        resolved_structured_output = structured_output
        if resolved_structured_output is not None and structured_output_schema is not None:
            raise ValueError(
                "Provide either structured_output or structured_output_schema, not both"
            )
        if resolved_structured_output is None and structured_output_schema is not None:
            resolved_name = (
                structured_output_name
                if structured_output_name is not None
                else f"{self.role_id}_output"
            )
            resolved_structured_output = StructuredOutputDefinition(
                name=resolved_name,
                json_schema=_coerce_json_mapping(
                    dict(structured_output_schema),
                    path="ProviderRequest.structured_output_schema",
                ),
                strict=structured_output_strict,
            )
        if resolved_structured_output is not None and not isinstance(
            resolved_structured_output, StructuredOutputDefinition
        ):
            raise TypeError("ProviderRequest.structured_output must be StructuredOutputDefinition")
        self.structured_output = resolved_structured_output

        self.idempotency_key = _validate_optional_str(
            idempotency_key,
            "ProviderRequest.idempotency_key",
        )

        if budget is not None and not isinstance(budget, BudgetSnapshot):
            raise TypeError("ProviderRequest.budget must be BudgetSnapshot")
        self.budget = budget

        metadata_payload = metadata if metadata is not None else {}
        self.metadata = _coerce_json_mapping(
            dict(metadata_payload), path="ProviderRequest.metadata"
        )

    @property
    def role(self) -> str:
        """Backward-compatible alias."""

        return self.role_id

    @property
    def prompt(self) -> str:
        """Backward-compatible alias combining system and user prompts."""

        if self.system_prompt:
            return f"{self.system_prompt}\n\n{self.user_prompt}"
        return self.user_prompt

    @property
    def context_documents(self) -> tuple[ContextDocument, ...]:
        """Backward-compatible alias."""

        return self.context_docs

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        """Backward-compatible alias."""

        return self.tool_permissions

    @property
    def allow_tool_calls(self) -> bool:
        """Backward-compatible alias."""

        return bool(self.tool_permissions)

    @property
    def max_output_tokens(self) -> int | None:
        """Backward-compatible alias."""

        return self.max_tokens

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "model": self.model,
            "role_id": self.role_id,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "context_docs": [doc.to_dict() for doc in self.context_docs],
            "tool_permissions": [tool.to_dict() for tool in self.tool_permissions],
            "requested_tools": list(self.requested_tools),
            "remote_mcp_tools": [tool.to_dict() for tool in self.remote_mcp_tools],
            "metadata": dict(self.metadata),
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice
        if self.structured_output is not None:
            payload["structured_output"] = self.structured_output.to_dict()
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.idempotency_key is not None:
            payload["idempotency_key"] = self.idempotency_key
        if self.budget is not None:
            payload["budget"] = self.budget.to_dict()
        return payload


@dataclass(slots=True)
class ProviderResponse:
    """Provider-agnostic normalized response payload."""

    model: str
    raw_text: str
    usage: ProviderUsage
    structured_output: JSONValue | None
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str | None
    request_id: str | None
    idempotency_key: str | None
    metadata: Mapping[str, JSONValue]

    def __init__(
        self,
        *,
        model: str,
        raw_text: str | None = None,
        content: str | None = None,
        usage: ProviderUsage | None = None,
        structured_output: object | None = None,
        tool_calls: Sequence[ToolCall] = (),
        finish_reason: str | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.model = _validate_non_empty_str(model, "ProviderResponse.model")

        resolved_text = (
            raw_text if raw_text is not None else (content if content is not None else "")
        )
        if not isinstance(resolved_text, str):
            raise TypeError("ProviderResponse.raw_text must be a string")
        self.raw_text = resolved_text

        self.usage = usage if usage is not None else ProviderUsage()
        if not isinstance(self.usage, ProviderUsage):
            raise TypeError("ProviderResponse.usage must be ProviderUsage")

        self.structured_output = (
            None
            if structured_output is None
            else _coerce_json_value(structured_output, path="ProviderResponse.structured_output")
        )

        self.tool_calls = tuple(tool_calls)
        self.finish_reason = _validate_optional_str(finish_reason, "ProviderResponse.finish_reason")
        self.request_id = _validate_optional_str(request_id, "ProviderResponse.request_id")
        self.idempotency_key = _validate_optional_str(
            idempotency_key,
            "ProviderResponse.idempotency_key",
        )

        metadata_payload = metadata if metadata is not None else {}
        self.metadata = _coerce_json_mapping(
            dict(metadata_payload), path="ProviderResponse.metadata"
        )

        if not self.raw_text and not self.tool_calls:
            raise ValueError("ProviderResponse must include raw_text or at least one tool call")

    @property
    def content(self) -> str:
        """Backward-compatible alias."""

        return self.raw_text

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "model": self.model,
            "raw_text": self.raw_text,
            "usage": self.usage.to_dict(),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "metadata": dict(self.metadata),
        }
        if self.structured_output is not None:
            payload["structured_output"] = self.structured_output
        if self.finish_reason is not None:
            payload["finish_reason"] = self.finish_reason
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        if self.idempotency_key is not None:
            payload["idempotency_key"] = self.idempotency_key
        return payload


class BaseProvider(abc.ABC):
    """Provider-agnostic abstract adapter API for synthesis-plane dispatch."""

    provider_name: str = "provider"

    def __init__(self, *, model_catalog: ModelCatalog | None = None) -> None:
        self._model_catalog = model_catalog if model_catalog is not None else load_model_catalog()

    @abc.abstractmethod
    async def send(self, request: ProviderRequest) -> ProviderResponse:
        """Send one provider request and return a normalized response."""

    def supports_tool_calling(self, model: str) -> bool:
        normalized = _validate_non_empty_str(model, "model")
        try:
            return self._model_catalog.supports_tool_calling(
                provider=self.provider_name,
                model=normalized,
            )
        except KeyError:
            return True

    def max_context_tokens(self, model: str) -> int:
        normalized = _validate_non_empty_str(model, "model")
        try:
            return self._model_catalog.max_context_tokens(
                provider=self.provider_name,
                model=normalized,
            )
        except KeyError:
            return 128_000

    def estimate_cost(self, tokens: int, model: str) -> float:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        normalized = _validate_non_empty_str(model, "model")
        try:
            return self._model_catalog.estimate_cost(
                provider=self.provider_name,
                model=normalized,
                total_tokens=tokens,
            )
        except KeyError:
            return (tokens / 1000.0) * 0.010


@runtime_checkable
class ProviderProtocol(Protocol):
    """Protocol implemented by concrete provider adapters."""

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        """Send one provider request and return normalized response."""


ProviderFactory: TypeAlias = Callable[[], ProviderProtocol]


class ProviderError(RuntimeError):
    """Base normalized provider error with deterministic machine-readable fields."""

    def __init__(
        self,
        *,
        provider: str,
        code: str,
        detail: str,
        retryable: bool,
        http_status: int | None = None,
        provider_code: str | None = None,
    ) -> None:
        self.provider = _validate_non_empty_str(provider, "provider")
        self.code = _validate_non_empty_str(code, "code")
        self.detail = _normalize_detail(detail)
        self.retryable = bool(retryable)
        self.http_status = http_status
        self.provider_code = _validate_optional_str(provider_code, "provider_code")

        parts = [
            f"provider={self.provider}",
            f"code={self.code}",
            f"retryable={str(self.retryable).lower()}",
        ]
        if self.http_status is not None:
            parts.append(f"http_status={self.http_status}")
        if self.provider_code is not None:
            parts.append(f"provider_code={self.provider_code}")
        parts.append(f"detail={self.detail}")
        super().__init__(" ".join(parts))


class ProviderUnavailableError(ProviderError):
    """Raised when provider runtime/SDK is unavailable."""

    def __init__(self, detail: str, *, provider: str = "provider") -> None:
        super().__init__(provider=provider, code="unavailable", detail=detail, retryable=False)


class ProviderAuthenticationError(ProviderError):
    """Authentication/authorization failures."""

    def __init__(
        self,
        detail: str,
        *,
        provider: str = "provider",
        http_status: int | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            code="auth",
            detail=detail,
            retryable=False,
            http_status=http_status,
        )


class ProviderInvalidRequestError(ProviderError):
    """Request payload invalid for provider API."""

    def __init__(
        self,
        detail: str,
        *,
        provider: str = "provider",
        http_status: int | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            code="invalid_request",
            detail=detail,
            retryable=False,
            http_status=http_status,
        )


class ProviderContextLengthError(ProviderError):
    """Request exceeds provider context constraints."""

    def __init__(
        self,
        detail: str,
        *,
        provider: str = "provider",
        http_status: int | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            code="context_length",
            detail=detail,
            retryable=False,
            http_status=http_status,
        )


class ProviderRateLimitError(ProviderError):
    """Provider rate-limit responses (retryable)."""

    def __init__(
        self,
        detail: str,
        *,
        provider: str = "provider",
        http_status: int | None = 429,
    ) -> None:
        super().__init__(
            provider=provider,
            code="rate_limit",
            detail=detail,
            retryable=True,
            http_status=http_status,
        )


class ProviderTimeoutError(ProviderError):
    """Provider timeout failures (retryable)."""

    def __init__(self, detail: str, *, provider: str = "provider") -> None:
        super().__init__(provider=provider, code="timeout", detail=detail, retryable=True)


class ProviderServiceError(ProviderError):
    """Provider API/service failures."""

    def __init__(
        self,
        detail: str,
        *,
        provider: str = "provider",
        retryable: bool = True,
        http_status: int | None = None,
        provider_code: str | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            code="service",
            detail=detail,
            retryable=retryable,
            http_status=http_status,
            provider_code=provider_code,
        )


class ProviderResponseError(ProviderError):
    """Raised when provider response normalization fails."""

    def __init__(self, detail: str, *, provider: str = "provider") -> None:
        super().__init__(provider=provider, code="response_invalid", detail=detail, retryable=False)


ProviderAuthError = ProviderAuthenticationError


def is_retryable_error(error: BaseException) -> bool:
    """Return retryability classification for normalized provider errors."""

    return isinstance(error, ProviderError) and error.retryable


def provider_error_subclasses() -> tuple[type[ProviderError], ...]:
    """Return all ProviderError subclasses in deterministic order."""

    discovered: set[type[ProviderError]] = set()
    pending = list(ProviderError.__subclasses__())
    while pending:
        current = pending.pop()
        discovered.add(current)
        pending.extend(current.__subclasses__())
    scoped = tuple(
        error_type
        for error_type in discovered
        if error_type.__module__.startswith("nexus_orchestrator.synthesis_plane.providers")
    )
    return tuple(
        sorted(
            scoped,
            key=lambda error_type: (error_type.__module__, error_type.__qualname__),
        )
    )


def instantiate_provider_error(
    error_type: type[ProviderError],
    *,
    detail: str,
    provider: str = "audit",
) -> ProviderError:
    """Instantiate a ProviderError subclass using the uniform audit contract."""

    if not isinstance(error_type, type) or not issubclass(error_type, ProviderError):
        raise TypeError("error_type must be a ProviderError subclass")
    if error_type is ProviderError:
        raise TypeError("error_type must be a concrete ProviderError subclass")
    constructor = cast("_ProviderErrorAuditCtor", error_type)
    return constructor(detail, provider=provider)


class _ProviderErrorAuditCtor(Protocol):
    def __call__(self, detail: str, *, provider: str = "provider") -> ProviderError: ...


@dataclass(frozen=True, slots=True)
class BackoffConfig:
    """Bounded exponential backoff policy."""

    max_retries: int = 2
    initial_delay_seconds: float = 0.25
    multiplier: float = 2.0
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be >= 0")
        if self.initial_delay_seconds > self.max_delay_seconds:
            raise ValueError("initial_delay_seconds must be <= max_delay_seconds")
        if not (0.0 <= self.jitter_ratio <= 1.0):
            raise ValueError("jitter_ratio must be between 0.0 and 1.0")


def compute_backoff_delay(
    *,
    retry_number: int,
    config: BackoffConfig,
    random_fn: RandomFn = random_module.random,
) -> float:
    """Return bounded exponential backoff delay for retry attempt N (1-based)."""

    if retry_number <= 0:
        raise ValueError("retry_number must be > 0")

    base_delay = config.initial_delay_seconds * (config.multiplier ** (retry_number - 1))
    bounded_delay = min(base_delay, config.max_delay_seconds)

    if config.jitter_ratio == 0.0:
        return bounded_delay

    random_value = random_fn()
    if not (0.0 <= random_value <= 1.0):
        raise ValueError("random_fn must return values in [0.0, 1.0]")

    max_jitter = bounded_delay * config.jitter_ratio
    jitter = ((random_value * 2.0) - 1.0) * max_jitter
    return max(0.0, min(config.max_delay_seconds, bounded_delay + jitter))


_CallbackT = TypeVar("_CallbackT")
RetryCallback: TypeAlias = Callable[[int, ProviderError, float], None]


async def run_with_retries(
    operation: Callable[[], Awaitable[_CallbackT]],
    *,
    map_exception: Callable[[Exception], ProviderError],
    backoff: BackoffConfig,
    sleep: SleepFn = asyncio.sleep,
    random_fn: RandomFn = random_module.random,
    on_retry: RetryCallback | None = None,
) -> _CallbackT:
    """Run an async operation with bounded retries based on ProviderError retryability."""

    retry_count = 0
    while True:
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001
            mapped = exc if isinstance(exc, ProviderError) else map_exception(exc)
            if not isinstance(mapped, ProviderError):
                raise TypeError("map_exception must return ProviderError") from exc

            if not mapped.retryable or retry_count >= backoff.max_retries:
                if mapped is exc:
                    raise
                raise mapped from exc

            retry_count += 1
            delay_seconds = compute_backoff_delay(
                retry_number=retry_count,
                config=backoff,
                random_fn=random_fn,
            )
            if on_retry is not None:
                on_retry(retry_count, mapped, delay_seconds)
            await sleep(delay_seconds)


@dataclass(frozen=True, slots=True)
class RedactedTranscript:
    """Safe transcript snapshot for logs or persistence."""

    provider: str
    model: str
    idempotency_key: str
    prompt: str
    response: str | None
    error: str | None


def redact_transcript(
    *,
    provider: str,
    model: str,
    idempotency_key: str,
    prompt: str,
    response: str | None,
    error: str | None,
) -> RedactedTranscript:
    """Return deterministic redacted transcript fields for logging/audit."""

    return RedactedTranscript(
        provider=_validate_non_empty_str(provider, "provider"),
        model=_validate_non_empty_str(model, "model"),
        idempotency_key=_validate_non_empty_str(idempotency_key, "idempotency_key"),
        prompt=redact_text(_validate_non_empty_str(prompt, "prompt", strip=False)),
        response=redact_text(response) if response is not None else None,
        error=redact_text(error) if error is not None else None,
    )


class ProviderRegistry:
    """Registry for provider adapter factories."""

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory, *, overwrite: bool = False) -> None:
        normalized = _validate_non_empty_str(name, "name").lower()
        if normalized in self._factories and not overwrite:
            raise ValueError(f"provider already registered: {normalized}")
        self._factories[normalized] = factory

    def unregister(self, name: str) -> None:
        normalized = _validate_non_empty_str(name, "name").lower()
        self._factories.pop(normalized, None)

    def is_registered(self, name: str) -> bool:
        normalized = _validate_non_empty_str(name, "name").lower()
        return normalized in self._factories

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories.keys()))

    def names(self) -> tuple[str, ...]:
        """Backward-compatible alias."""

        return self.list()

    def get(self, name: str) -> ProviderProtocol:
        normalized = _validate_non_empty_str(name, "name").lower()
        factory = self._factories.get(normalized)
        if factory is None:
            raise ProviderUnavailableError(
                provider=normalized,
                detail="provider is not registered",
            )
        adapter = factory()
        if not isinstance(adapter, ProviderProtocol):
            raise TypeError(f"provider factory returned invalid adapter for {normalized}")
        return adapter

    def create(self, name: str) -> ProviderProtocol:
        """Backward-compatible alias."""

        return self.get(name)


def derive_idempotency_key(request: ProviderRequest, *, provider: str) -> str:
    """Return deterministic idempotency key for provider requests."""

    if request.idempotency_key is not None:
        return request.idempotency_key

    payload: dict[str, JSONValue] = {
        "provider": provider,
        "request": request.to_dict(),
    }
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_tool_arguments(
    arguments: object,
    *,
    provider: str,
    tool_name: str,
) -> dict[str, JSONValue]:
    """Normalize provider tool-argument payload to a JSON object."""

    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return _coerce_json_mapping(arguments, path="arguments")
    if isinstance(arguments, str):
        candidate = arguments.strip()
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                provider=provider,
                detail=f"invalid tool arguments for {tool_name}: non-JSON string",
            ) from exc
        if not isinstance(parsed, dict):
            raise ProviderResponseError(
                provider=provider,
                detail=f"invalid tool arguments for {tool_name}: expected JSON object",
            )
        return _coerce_json_mapping(parsed, path="arguments")

    raise ProviderResponseError(
        provider=provider,
        detail=(
            f"invalid tool arguments for {tool_name}: unsupported type {type(arguments).__name__}"
        ),
    )


def _normalize_detail(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unknown error"
    return " ".join(text.split())


# Backward-compatible alias.
ContextDoc = ContextDocument


__all__ = [
    "BackoffConfig",
    "BaseProvider",
    "BudgetSnapshot",
    "ContextDoc",
    "ContextDocument",
    "JSONValue",
    "ProviderAuthError",
    "ProviderAuthenticationError",
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
    "RandomFn",
    "RemoteMCPToolConfig",
    "RedactedTranscript",
    "SleepFn",
    "StructuredOutputDefinition",
    "ToolCall",
    "ToolDefinition",
    "compute_backoff_delay",
    "derive_idempotency_key",
    "instantiate_provider_error",
    "is_retryable_error",
    "parse_tool_arguments",
    "provider_error_subclasses",
    "redact_transcript",
    "run_with_retries",
]
