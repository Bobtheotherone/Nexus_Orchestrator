"""
nexus-orchestrator — Anthropic provider adapter

File: src/nexus_orchestrator/synthesis_plane/providers/anthropic_adapter.py
Last updated: 2026-02-13

Purpose
- Anthropic provider adapter (Claude-class).

What should be included in this file
- API client wrapper, retries/backoff, token accounting mapping.
- Model routing names and capability mapping.

Functional requirements
- Must support long-context prompts for Architect role.

Non-functional requirements
- Must be configurable and safe; no secrets in logs.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import random as random_module
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, cast

from nexus_orchestrator.synthesis_plane.providers.base import (
    BackoffConfig,
    BaseProvider,
    JSONValue,
    ProviderAuthenticationError,
    ProviderContextLengthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderRequest,
    ProviderResponse,
    ProviderResponseError,
    ProviderServiceError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUsage,
    RandomFn,
    RemoteMCPToolConfig,
    SleepFn,
    ToolCall,
    ToolDefinition,
    derive_idempotency_key,
    parse_tool_arguments,
    run_with_retries,
)

if TYPE_CHECKING:
    from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog


class _AnthropicMessagesAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _AnthropicClient(Protocol):
    messages: _AnthropicMessagesAPI


class AnthropicProvider(BaseProvider):
    """Anthropic messages adapter with optional SDK dependency and injected client support."""

    provider_name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        client: _AnthropicClient | None = None,
        backoff: BackoffConfig | None = None,
        sleep: SleepFn = asyncio.sleep,
        random_fn: RandomFn = random_module.random,
        model_catalog: ModelCatalog | None = None,
    ) -> None:
        super().__init__(model_catalog=model_catalog)
        self.model = _validate_non_empty_str(model, "model")
        self._api_key = _validate_optional_str(api_key, "api_key")
        self._api_key_env = _validate_optional_str(api_key_env, "api_key_env")
        self._base_url = _validate_optional_str(base_url, "base_url")

        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        self._timeout_seconds = timeout_seconds

        self._client = client
        self._backoff = backoff if backoff is not None else BackoffConfig()
        self._sleep = sleep
        self._random_fn = random_fn

    @property
    def name(self) -> str:
        """Backward-compatible alias."""

        return self.provider_name

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        request_payload = _normalize_request(request, default_model=self.model)
        idempotency_key = derive_idempotency_key(request_payload, provider=self.provider_name)
        payload = self._build_payload(request_payload, idempotency_key=idempotency_key)

        async def operation() -> ProviderResponse:
            client = self._ensure_client()
            started = time.perf_counter()
            raw_response, idempotency_header_sent = await self._create_response(
                client=client,
                payload=payload,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            return self._normalize_response(
                raw_response,
                request=request_payload,
                idempotency_key=idempotency_key,
                latency_ms=latency_ms,
                idempotency_header_sent=idempotency_header_sent,
            )

        return await run_with_retries(
            operation,
            map_exception=self._map_exception,
            backoff=self._backoff,
            sleep=self._sleep,
            random_fn=self._random_fn,
        )

    def supports_tool_calling(self, model: str) -> bool:
        return super().supports_tool_calling(model)

    def max_context_tokens(self, model: str) -> int:
        return super().max_context_tokens(model)

    def estimate_cost(self, tokens: int, model: str) -> float:
        return super().estimate_cost(tokens, model)

    def _ensure_client(self) -> _AnthropicClient:
        if self._client is not None:
            return self._client
        self._client = self._create_default_client()
        return self._client

    def _create_default_client(self) -> _AnthropicClient:
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:
            raise ProviderUnavailableError(
                provider=self.provider_name,
                detail="anthropic SDK is not installed",
            ) from exc

        async_anthropic = getattr(anthropic_module, "AsyncAnthropic", None)
        if async_anthropic is None:
            raise ProviderUnavailableError(
                provider=self.provider_name,
                detail="anthropic SDK does not expose AsyncAnthropic",
            )

        api_key = self._resolve_api_key()

        init_kwargs: dict[str, object] = {"api_key": api_key}
        if self._base_url is not None:
            init_kwargs["base_url"] = self._base_url
        if self._timeout_seconds is not None:
            init_kwargs["timeout"] = self._timeout_seconds

        client = async_anthropic(**init_kwargs)
        if not hasattr(client, "messages"):
            raise ProviderUnavailableError(
                provider=self.provider_name,
                detail="anthropic client missing messages API",
            )

        return cast("_AnthropicClient", client)

    def _resolve_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key

        if self._api_key_env is not None:
            configured = os.getenv(self._api_key_env)
            if configured is None or not configured.strip():
                raise ProviderAuthenticationError(
                    provider=self.provider_name,
                    detail=f"missing Anthropic API key in configured env var {self._api_key_env}",
                    http_status=401,
                )
            return configured

        fallback_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("NEXUS_ANTHROPIC_API_KEY")
        if fallback_key is None or not fallback_key.strip():
            raise ProviderAuthenticationError(
                provider=self.provider_name,
                detail=(
                    "missing Anthropic API key; set ANTHROPIC_API_KEY or NEXUS_ANTHROPIC_API_KEY"
                ),
                http_status=401,
            )
        return fallback_key

    async def _create_response(
        self,
        *,
        client: _AnthropicClient,
        payload: dict[str, object],
    ) -> tuple[object, bool]:
        try:
            raw = await client.messages.create(**payload)
            return raw, True
        except TypeError as exc:
            message = str(exc).lower()
            if "extra_headers" not in message:
                raise
            payload_without_headers = dict(payload)
            payload_without_headers.pop("extra_headers", None)
            raw = await client.messages.create(**payload_without_headers)
            return raw, False

    def _build_payload(
        self, request: ProviderRequest, *, idempotency_key: str
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": "user", "content": _build_input_text(request)}],
            "extra_headers": {"Idempotency-Key": idempotency_key},
            "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.reasoning_effort is not None:
            payload["thinking"] = {"type": "enabled", "effort": request.reasoning_effort}

        tools: list[dict[str, object]] = []
        if request.tool_permissions:
            tools.extend(_tool_definition_payload(tool) for tool in request.tool_permissions)
        if request.structured_output is not None:
            tools.append(
                _structured_output_tool_payload(
                    name=request.structured_output.name,
                    json_schema=request.structured_output.json_schema,
                    description=request.structured_output.description,
                )
            )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.structured_output.name,
                    "schema": dict(request.structured_output.json_schema),
                    "strict": bool(request.structured_output.strict),
                },
            }
        if tools:
            payload["tools"] = tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        elif request.requested_tools:
            payload["tool_choice"] = {
                "type": "allowed_tools",
                "allowed_tools": list(request.requested_tools),
            }
        elif request.structured_output is not None:
            payload["tool_choice"] = {"type": "tool", "name": request.structured_output.name}

        if request.remote_mcp_tools:
            payload["mcp_servers"] = [
                _remote_mcp_server_payload(server) for server in request.remote_mcp_tools
            ]

        return payload

    def _normalize_response(
        self,
        raw_response: object,
        *,
        request: ProviderRequest,
        idempotency_key: str,
        latency_ms: int,
        idempotency_header_sent: bool,
    ) -> ProviderResponse:
        raw_text, tool_calls, structured_output = _extract_anthropic_output(
            raw_response,
            structured_output_tool_name=(
                request.structured_output.name if request.structured_output is not None else None
            ),
        )
        model_value = _read_str(raw_response, "model") or request.model
        request_id = _read_str(raw_response, "id") or _read_str(raw_response, "request_id")

        usage = _normalize_usage(raw_response)
        total_tokens = usage.total_tokens
        if total_tokens == 0:
            total_tokens = usage.input_tokens + usage.output_tokens
        estimated_cost = _estimate_cost_from_catalog(
            provider=self.provider_name,
            model_catalog=self._model_catalog,
            model=model_value,
            total_tokens=total_tokens,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            fallback=self.estimate_cost,
        )
        if usage.total_tokens == 0:
            usage = ProviderUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_estimate_usd=estimated_cost,
            )
        else:
            usage = ProviderUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=latency_ms,
                cost_estimate_usd=estimated_cost,
            )

        finish_reason = _read_str(raw_response, "stop_reason")

        metadata: dict[str, JSONValue] = {
            "idempotency_header_sent": idempotency_header_sent,
            "requested_tools": list(request.requested_tools),
        }
        stop_sequence = _read_str(raw_response, "stop_sequence")
        if stop_sequence is not None:
            metadata["stop_sequence"] = stop_sequence

        return ProviderResponse(
            model=model_value,
            raw_text=raw_text,
            usage=usage,
            structured_output=structured_output,
            tool_calls=tool_calls,
            request_id=request_id,
            idempotency_key=idempotency_key,
            finish_reason=finish_reason,
            metadata=metadata,
        )

    def _map_exception(self, exc: Exception) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc

        status_code = _read_status_code(exc)
        class_name = exc.__class__.__name__.lower()
        detail = _exception_detail(exc)
        detail_lower = detail.lower()

        if status_code in {401, 403} or "auth" in class_name or "permission" in class_name:
            return ProviderAuthenticationError(
                provider=self.provider_name,
                detail=detail,
                http_status=status_code,
            )

        if status_code == 429 or "ratelimit" in class_name:
            return ProviderRateLimitError(
                provider=self.provider_name,
                detail=detail,
                http_status=status_code,
            )

        if isinstance(exc, asyncio.TimeoutError) or "timeout" in class_name:
            return ProviderTimeoutError(provider=self.provider_name, detail=detail)

        if (
            status_code in {400, 413, 422}
            and "context" in detail_lower
            and "length" in detail_lower
        ) or "contextlength" in class_name:
            return ProviderContextLengthError(
                provider=self.provider_name,
                detail=detail,
                http_status=status_code,
            )

        if status_code is not None and status_code in {400, 404, 409, 422}:
            return ProviderInvalidRequestError(
                provider=self.provider_name,
                detail=detail,
                http_status=status_code,
            )

        if "badrequest" in class_name or "invalidrequest" in class_name:
            return ProviderInvalidRequestError(provider=self.provider_name, detail=detail)

        if status_code is not None and status_code >= 500:
            return ProviderServiceError(
                provider=self.provider_name,
                detail=detail,
                retryable=True,
                http_status=status_code,
            )

        if "connection" in class_name or "apierror" in class_name or "overloaded" in class_name:
            return ProviderServiceError(provider=self.provider_name, detail=detail, retryable=True)

        return ProviderServiceError(provider=self.provider_name, detail=detail, retryable=True)


AnthropicAdapter = AnthropicProvider


def _normalize_request(request: ProviderRequest, *, default_model: str) -> ProviderRequest:
    if request.model == default_model:
        return request
    return ProviderRequest(
        model=request.model,
        role_id=request.role_id,
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        context_docs=request.context_docs,
        tool_permissions=request.tool_permissions,
        requested_tools=request.requested_tools,
        reasoning_effort=request.reasoning_effort,
        tool_choice=request.tool_choice,
        remote_mcp_tools=request.remote_mcp_tools,
        structured_output=request.structured_output,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        idempotency_key=request.idempotency_key,
        budget=request.budget,
        metadata=request.metadata,
    )


def _build_input_text(request: ProviderRequest) -> str:
    blocks: list[str] = []
    if request.system_prompt:
        blocks.append(f"System:\n{request.system_prompt}")
    blocks.append(f"Role: {request.role_id}")
    blocks.append(f"User:\n{request.user_prompt}")
    if request.context_docs:
        blocks.append("Context:")
        for document in request.context_docs:
            label = document.path or document.name
            blocks.append(f"[{label}]\n{document.content}")
    return "\n\n".join(blocks)


def _tool_definition_payload(tool: ToolDefinition) -> dict[str, object]:
    out: dict[str, object] = {
        "name": tool.name,
        "input_schema": dict(tool.json_schema),
    }
    if tool.description is not None:
        out["description"] = tool.description
    return out


def _structured_output_tool_payload(
    *,
    name: str,
    json_schema: Mapping[str, JSONValue],
    description: str | None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "name": name,
        "input_schema": dict(json_schema),
    }
    if description is not None:
        out["description"] = description
    return out


def _remote_mcp_server_payload(server: RemoteMCPToolConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "server_name": server.server_label,
        "server_url": server.server_url,
    }
    if server.allowed_tools:
        payload["allowed_tools"] = list(server.allowed_tools)
    if server.require_approval is not None:
        payload["require_approval"] = server.require_approval

    headers = dict(server.headers)
    if server.authorization_token_env is not None:
        token_value = os.getenv(server.authorization_token_env)
        if token_value is not None and token_value.strip():
            headers["Authorization"] = f"Bearer {token_value}"
    if headers:
        payload["headers"] = headers
    return payload


def _extract_anthropic_output(
    raw_response: object,
    *,
    structured_output_tool_name: str | None = None,
) -> tuple[str, tuple[ToolCall, ...], JSONValue | None]:
    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    structured_output = _extract_anthropic_structured_output(raw_response)

    for item in _read_sequence(raw_response, "content"):
        item_type = (_read_str(item, "type") or "").lower()
        if item_type == "text":
            text_value = _read_str(item, "text")
            if text_value:
                text_chunks.append(text_value)
            if structured_output is None and _looks_like_json_document(text_value or ""):
                structured_output = _coerce_structured_candidate(text_value)
        elif item_type in {"json", "output_json"}:
            parsed = _coerce_structured_candidate(
                _read_value(item, "json", default=_read_value(item, "value"))
            )
            if parsed is not None:
                structured_output = parsed
        elif item_type == "tool_use":
            name = _read_str(item, "name")
            if name is None:
                raise ProviderResponseError(
                    provider="anthropic", detail="tool_use block missing name"
                )
            call_id = _read_str(item, "id") or f"anthropic-tool-call-{len(tool_calls) + 1}"
            arguments = parse_tool_arguments(
                _read_value(item, "input"), provider="anthropic", tool_name=name
            )
            if structured_output_tool_name is not None and name == structured_output_tool_name:
                structured_output = arguments
            else:
                tool_calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments))

    combined_text = "\n".join(chunk for chunk in text_chunks if chunk.strip())
    if not combined_text and not tool_calls and structured_output is None:
        raise ProviderResponseError(
            provider="anthropic",
            detail="response does not contain text or tool calls",
        )

    return combined_text, tuple(tool_calls), structured_output


def _extract_anthropic_structured_output(raw_response: object) -> JSONValue | None:
    direct_candidates = (
        _read_value(raw_response, "output_parsed"),
        _read_value(raw_response, "parsed"),
        _read_value(raw_response, "structured_output"),
    )
    for candidate in direct_candidates:
        parsed = _coerce_structured_candidate(candidate)
        if parsed is not None:
            return parsed
    return None


def _coerce_structured_candidate(candidate: object) -> JSONValue | None:
    if candidate is None:
        return None
    if isinstance(candidate, bool):
        return candidate
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, float):
        return candidate
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if not stripped:
            return None
        if _looks_like_json_document(stripped):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
            return _coerce_structured_candidate(parsed)
        return stripped
    if isinstance(candidate, Mapping):
        out: dict[str, JSONValue] = {}
        for key, value in candidate.items():
            if not isinstance(key, str):
                return None
            normalized = _coerce_structured_candidate(value)
            if normalized is None and value is not None:
                return None
            out[key] = normalized
        return out
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        out_list: list[JSONValue] = []
        for item in candidate:
            normalized = _coerce_structured_candidate(item)
            if normalized is None and item is not None:
                return None
            out_list.append(normalized)
        return out_list
    return None


def _looks_like_json_document(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    return stripped.startswith("{") or stripped.startswith("[")


def _normalize_usage(raw_response: object) -> ProviderUsage:
    usage_payload = _read_value(raw_response, "usage")
    if usage_payload is None:
        return ProviderUsage()

    input_tokens = _read_int(usage_payload, "input_tokens") or 0
    output_tokens = _read_int(usage_payload, "output_tokens") or 0
    total_tokens = _read_int(usage_payload, "total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _estimate_cost_from_catalog(
    *,
    provider: str,
    model_catalog: ModelCatalog,
    model: str,
    total_tokens: int,
    input_tokens: int,
    output_tokens: int,
    fallback: Callable[[int, str], float],
) -> float:
    try:
        return model_catalog.estimate_cost(
            provider=provider,
            model=model,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except KeyError:
        return float(fallback(total_tokens, model))


def _exception_detail(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return " ".join(text.split())
    return exc.__class__.__name__


def _read_status_code(exc: BaseException) -> int | None:
    for key in ("status_code", "status", "http_status"):
        value = getattr(exc, key, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        nested = getattr(response, "status_code", None)
        if isinstance(nested, int):
            return nested
    return None


def _read_value(value: object, key: str, *, default: object | None = None) -> object | None:
    if isinstance(value, Mapping):
        return cast("object | None", value.get(key, default))
    return cast("object | None", getattr(value, key, default))


def _read_sequence(value: object, key: str) -> tuple[object, ...]:
    candidate = _read_value(value, key)
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
        return tuple(candidate)
    return ()


def _read_str(value: object, key: str) -> str | None:
    candidate = _read_value(value, key)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    return None


def _read_int(value: object, key: str) -> int | None:
    candidate = _read_value(value, key)
    if isinstance(candidate, int):
        return candidate
    return None


def _validate_non_empty_str(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _validate_optional_str(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _validate_non_empty_str(value, name)


__all__ = ["AnthropicAdapter", "AnthropicProvider"]
