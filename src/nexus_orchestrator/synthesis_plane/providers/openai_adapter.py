"""
nexus-orchestrator — OpenAI provider adapter

File: src/nexus_orchestrator/synthesis_plane/providers/openai_adapter.py
Last updated: 2026-02-13

Purpose
- OpenAI provider adapter (Codex-class and/or GPT-class).

What should be included in this file
- API client wrapper, rate limit/backoff, streaming handling (if used).
- Model routing names and capabilities mapping.
- Cost estimation and accounting integration.

Functional requirements
- Must support tool-calling if used by the agent runtime.

Non-functional requirements
- Must be configurable and safe; do not hardcode endpoints/keys.
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


class _OpenAIResponsesAPI(Protocol):
    async def create(self, **kwargs: object) -> object: ...


class _OpenAIClient(Protocol):
    responses: _OpenAIResponsesAPI


class OpenAIProvider(BaseProvider):
    """OpenAI responses adapter with optional SDK dependency and injected client support."""

    provider_name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        timeout_seconds: float | None = None,
        client: _OpenAIClient | None = None,
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
        self._organization = _validate_optional_str(organization, "organization")

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

    def _ensure_client(self) -> _OpenAIClient:
        if self._client is not None:
            return self._client
        self._client = self._create_default_client()
        return self._client

    def _create_default_client(self) -> _OpenAIClient:
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            raise ProviderUnavailableError(
                provider=self.provider_name,
                detail="openai SDK is not installed",
            ) from exc

        async_openai = getattr(openai_module, "AsyncOpenAI", None)
        if async_openai is None:
            raise ProviderUnavailableError(
                provider=self.provider_name,
                detail="openai SDK does not expose AsyncOpenAI",
            )

        api_key = self._resolve_api_key()

        init_kwargs: dict[str, object] = {"api_key": api_key}
        if self._base_url is not None:
            init_kwargs["base_url"] = self._base_url
        if self._organization is not None:
            init_kwargs["organization"] = self._organization
        if self._timeout_seconds is not None:
            init_kwargs["timeout"] = self._timeout_seconds

        client = async_openai(**init_kwargs)
        if not hasattr(client, "responses"):
            raise ProviderUnavailableError(
                provider=self.provider_name,
                detail="openai client missing responses API",
            )
        return cast("_OpenAIClient", client)

    def _resolve_api_key(self) -> str:
        if self._api_key is not None:
            return self._api_key

        if self._api_key_env is not None:
            configured = os.getenv(self._api_key_env)
            if configured is None or not configured.strip():
                raise ProviderAuthenticationError(
                    provider=self.provider_name,
                    detail=f"missing OpenAI API key in configured env var {self._api_key_env}",
                    http_status=401,
                )
            return configured

        fallback_key = os.getenv("OPENAI_API_KEY") or os.getenv("NEXUS_OPENAI_API_KEY")
        if fallback_key is None or not fallback_key.strip():
            raise ProviderAuthenticationError(
                provider=self.provider_name,
                detail="missing OpenAI API key; set OPENAI_API_KEY or NEXUS_OPENAI_API_KEY",
                http_status=401,
            )
        return fallback_key

    async def _create_response(
        self,
        *,
        client: _OpenAIClient,
        payload: dict[str, object],
    ) -> tuple[object, bool]:
        try:
            raw = await client.responses.create(**payload)
            return raw, True
        except TypeError as exc:
            message = str(exc).lower()
            if "extra_headers" not in message:
                raise
            payload_without_headers = dict(payload)
            payload_without_headers.pop("extra_headers", None)
            raw = await client.responses.create(**payload_without_headers)
            return raw, False

    def _build_payload(
        self, request: ProviderRequest, *, idempotency_key: str
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": request.model,
            "input": _build_input_text(request),
            "extra_headers": {"Idempotency-Key": idempotency_key},
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_output_tokens"] = request.max_tokens
        if request.reasoning_effort is not None:
            payload["reasoning"] = {"effort": request.reasoning_effort}

        tools: list[dict[str, object]] = []
        if request.tool_permissions:
            tools.extend(_tool_definition_payload(tool) for tool in request.tool_permissions)
        if request.remote_mcp_tools:
            tools.extend(_remote_mcp_tool_payload(tool) for tool in request.remote_mcp_tools)
        if tools:
            payload["tools"] = tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice
        elif request.requested_tools:
            payload["tool_choice"] = {
                "type": "allowed_tools",
                "allowed_tools": list(request.requested_tools),
            }

        if request.structured_output is not None:
            format_payload: dict[str, object] = {
                "type": "json_schema",
                "name": request.structured_output.name,
                "schema": dict(request.structured_output.json_schema),
                "strict": bool(request.structured_output.strict),
            }
            if request.structured_output.description is not None:
                format_payload["description"] = request.structured_output.description
            payload["text"] = {"format": format_payload}

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
        raw_text, tool_calls, structured_output = _extract_openai_output(raw_response)
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

        finish_reason: str | None = None
        choices = _read_sequence(raw_response, "choices")
        if choices:
            finish_reason = _read_str(choices[0], "finish_reason")

        metadata: dict[str, JSONValue] = {
            "idempotency_header_sent": idempotency_header_sent,
            "requested_tools": list(request.requested_tools),
        }

        if finish_reason is not None:
            metadata["finish_reason"] = finish_reason

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

        if "connection" in class_name or "apierror" in class_name or "server" in class_name:
            return ProviderServiceError(provider=self.provider_name, detail=detail, retryable=True)

        return ProviderServiceError(provider=self.provider_name, detail=detail, retryable=True)


OpenAIAdapter = OpenAIProvider


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
        "type": "function",
        "name": tool.name,
        "parameters": dict(tool.json_schema),
        "strict": bool(tool.strict),
    }
    if tool.description is not None:
        out["description"] = tool.description
    return out


def _remote_mcp_tool_payload(tool: RemoteMCPToolConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "mcp",
        "server_label": tool.server_label,
        "server_url": tool.server_url,
    }
    if tool.allowed_tools:
        payload["allowed_tools"] = list(tool.allowed_tools)
    if tool.require_approval is not None:
        payload["require_approval"] = tool.require_approval

    headers = dict(tool.headers)
    if tool.authorization_token_env is not None:
        token_value = os.getenv(tool.authorization_token_env)
        if token_value is not None and token_value.strip():
            headers["Authorization"] = f"Bearer {token_value}"
    if headers:
        payload["headers"] = headers
    return payload


def _extract_openai_output(
    raw_response: object,
) -> tuple[str, tuple[ToolCall, ...], JSONValue | None]:
    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    structured_output = _extract_openai_structured_output(raw_response)

    direct_output = _read_str(raw_response, "output_text")
    if direct_output:
        text_chunks.append(direct_output)

    output_items = _read_sequence(raw_response, "output")
    for item in output_items:
        item_type = (_read_str(item, "type") or "").lower()

        if item_type == "message":
            for content_part in _read_sequence(item, "content"):
                content_type = (_read_str(content_part, "type") or "").lower()
                if content_type in {"output_text", "text"}:
                    text_value = _read_str(content_part, "text") or _read_str(content_part, "value")
                    if text_value:
                        text_chunks.append(text_value)
                elif content_type in {"output_json", "json"}:
                    parsed = _coerce_structured_candidate(
                        _read_value(
                            content_part, "json", default=_read_value(content_part, "value")
                        )
                    )
                    if parsed is not None:
                        structured_output = parsed
                elif content_type in {"function_call", "tool_call"}:
                    tool_calls.append(
                        _normalize_openai_tool_call(content_part, len(tool_calls) + 1)
                    )

        elif item_type in {"function_call", "tool_call"}:
            tool_calls.append(_normalize_openai_tool_call(item, len(tool_calls) + 1))
        elif item_type in {"output_json", "json"}:
            parsed = _coerce_structured_candidate(
                _read_value(item, "json", default=_read_value(item, "value"))
            )
            if parsed is not None:
                structured_output = parsed

    choices = _read_sequence(raw_response, "choices")
    for choice in choices:
        message = _read_value(choice, "message")
        message_content = _read_str(message, "content")
        if message_content:
            text_chunks.append(message_content)
        parsed_choice = _coerce_structured_candidate(
            _read_value(message, "parsed", default=_read_value(choice, "parsed"))
        )
        if parsed_choice is not None:
            structured_output = parsed_choice

        for raw_tool_call in _read_sequence(message, "tool_calls"):
            tool_calls.append(_normalize_openai_tool_call(raw_tool_call, len(tool_calls) + 1))

    combined_text = "\n".join(chunk for chunk in text_chunks if chunk.strip())
    if structured_output is None and _looks_like_json_document(combined_text):
        parsed_text = _coerce_structured_candidate(combined_text)
        if parsed_text is not None:
            structured_output = parsed_text

    if not combined_text and not tool_calls and structured_output is None:
        raise ProviderResponseError(
            provider="openai",
            detail="response does not contain text or tool calls",
        )

    return combined_text, tuple(tool_calls), structured_output


def _extract_openai_structured_output(raw_response: object) -> JSONValue | None:
    direct_candidates = (
        _read_value(raw_response, "output_parsed"),
        _read_value(raw_response, "parsed"),
        _read_value(raw_response, "structured_output"),
    )
    for candidate in direct_candidates:
        parsed = _coerce_structured_candidate(candidate)
        if parsed is not None:
            return parsed

    for output_item in _read_sequence(raw_response, "output"):
        parsed = _coerce_structured_candidate(
            _read_value(
                output_item,
                "parsed",
                default=_read_value(output_item, "json", default=_read_value(output_item, "value")),
            )
        )
        if parsed is not None:
            return parsed

        for content_part in _read_sequence(output_item, "content"):
            parsed = _coerce_structured_candidate(
                _read_value(
                    content_part,
                    "parsed",
                    default=_read_value(
                        content_part,
                        "json",
                        default=_read_value(content_part, "value"),
                    ),
                )
            )
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


def _normalize_openai_tool_call(raw_call: object, index: int) -> ToolCall:
    function_payload = _read_value(raw_call, "function")

    name = _read_str(raw_call, "name")
    arguments_source: object = _read_value(raw_call, "arguments")

    if function_payload is not None:
        name = _read_str(function_payload, "name") or name
        arguments_source = _read_value(function_payload, "arguments", default=arguments_source)

    if name is None:
        raise ProviderResponseError(provider="openai", detail="tool call missing function name")

    call_id = (
        _read_str(raw_call, "call_id") or _read_str(raw_call, "id") or f"openai-tool-call-{index}"
    )

    arguments = parse_tool_arguments(arguments_source, provider="openai", tool_name=name)

    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _normalize_usage(raw_response: object) -> ProviderUsage:
    usage_payload = _read_value(raw_response, "usage")
    if usage_payload is None:
        return ProviderUsage()

    input_tokens = _read_int(usage_payload, "input_tokens")
    if input_tokens is None:
        input_tokens = _read_int(usage_payload, "prompt_tokens") or 0

    output_tokens = _read_int(usage_payload, "output_tokens")
    if output_tokens is None:
        output_tokens = _read_int(usage_payload, "completion_tokens") or 0

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


__all__ = ["OpenAIAdapter", "OpenAIProvider"]
