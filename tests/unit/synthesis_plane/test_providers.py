"""
Unit tests for synthesis-plane provider abstractions and adapters.

Coverage:
- Shared provider models, deterministic error taxonomy, and provider registry.
- Deterministic bounded backoff behavior.
- OpenAI/Anthropic adapter normalization and retry classification.
- Idempotency key stability across retries.
- Transcript redaction safety.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pytest

from nexus_orchestrator.synthesis_plane.providers import (
    AnthropicAdapter,
    BackoffConfig,
    BudgetSnapshot,
    ContextDocument,
    OpenAIAdapter,
    ProviderAuthError,
    ProviderContextLengthError,
    ProviderRequest,
    ProviderUnavailableError,
    ProviderUsage,
    ToolDefinition,
    compute_backoff_delay,
    redact_transcript,
)
from nexus_orchestrator.synthesis_plane.providers.base import (
    ProviderError,
    ProviderRegistry,
    instantiate_provider_error,
    provider_error_subclasses,
)


@dataclass(slots=True)
class _ScriptedOpenAIResponses:
    outcomes: deque[object | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    async def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if not self.outcomes:
            raise RuntimeError("scripted openai outcomes exhausted")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass(slots=True)
class _FakeOpenAIClient:
    responses: _ScriptedOpenAIResponses


@dataclass(slots=True)
class _ScriptedAnthropicMessages:
    outcomes: deque[object | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    async def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if not self.outcomes:
            raise RuntimeError("scripted anthropic outcomes exhausted")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass(slots=True)
class _FakeAnthropicClient:
    messages: _ScriptedAnthropicMessages


@dataclass(slots=True)
class _StubProvider:
    name: str = "stub"

    async def send(self, request: ProviderRequest) -> object:
        _ = request
        return ProviderUsage()


@dataclass(slots=True)
class _SleepRecorder:
    calls: list[float] = field(default_factory=list)

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class APIConnectionError(Exception):
    pass


class ContextLengthError(Exception):
    status_code = 413


def _request(
    *,
    model: str,
    prompt: str,
    idempotency_key: str | None = None,
) -> ProviderRequest:
    return ProviderRequest(
        model=model,
        role="implementer",
        prompt=prompt,
        allow_tool_calls=True,
        tools=(
            ToolDefinition(
                name="lookup_contract",
                description="Lookup contract details",
                json_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            ),
        ),
        idempotency_key=idempotency_key,
    )


def test_provider_request_contract_fields_and_aliases() -> None:
    request = ProviderRequest(
        model="gpt-4.1-mini",
        role_id="implementer",
        system_prompt="system-instructions",
        user_prompt="build feature x",
        max_tokens=512,
        context_docs=(ContextDocument(name="constraints", content="must pass"),),
        requested_tools=("ruff", "pytest"),
        budget=BudgetSnapshot(max_total_tokens=4_096, max_cost_usd=1.25),
        metadata={"work_item_id": "WI-42"},
    )

    assert request.role == "implementer"
    assert request.prompt == "system-instructions\n\nbuild feature x"
    assert request.context_documents[0].name == "constraints"
    assert request.max_output_tokens == 512
    assert request.requested_tools == ("ruff", "pytest")
    assert request.budget is not None
    assert request.budget.max_total_tokens == 4_096
    assert request.metadata["work_item_id"] == "WI-42"


def test_compute_backoff_delay_is_bounded_and_deterministic() -> None:
    random_values = iter((0.0, 1.0, 0.5))

    def random_fn() -> float:
        return next(random_values)

    config = BackoffConfig(
        max_retries=4,
        initial_delay_seconds=1.0,
        multiplier=3.0,
        max_delay_seconds=5.0,
        jitter_ratio=0.5,
    )

    delay_1 = compute_backoff_delay(retry_number=1, config=config, random_fn=random_fn)
    delay_2 = compute_backoff_delay(retry_number=2, config=config, random_fn=random_fn)
    delay_3 = compute_backoff_delay(retry_number=3, config=config, random_fn=random_fn)

    assert delay_1 == pytest.approx(0.5)
    assert delay_2 == pytest.approx(4.5)
    assert delay_3 == pytest.approx(5.0)


def test_provider_registry_register_create_and_missing_error() -> None:
    registry = ProviderRegistry()
    registry.register("stub", lambda: _StubProvider())

    assert registry.names() == ("stub",)
    assert registry.is_registered("stub") is True

    provider = registry.create("stub")
    assert provider.name == "stub"

    with pytest.raises(ValueError, match="provider already registered: stub"):
        registry.register("stub", lambda: _StubProvider())

    with pytest.raises(
        ProviderUnavailableError,
        match=(
            r"provider=missing code=unavailable retryable=false detail=provider is not registered"
        ),
    ):
        registry.create("missing")


def test_provider_error_subclasses_enumeration_is_deterministic() -> None:
    expected_names = (
        "ProviderAuthenticationError",
        "ProviderContextLengthError",
        "ProviderInvalidRequestError",
        "ProviderRateLimitError",
        "ProviderResponseError",
        "ProviderServiceError",
        "ProviderTimeoutError",
        "ProviderUnavailableError",
    )

    first = provider_error_subclasses()
    second = provider_error_subclasses()

    assert tuple(error_type.__name__ for error_type in first) == expected_names
    assert first == second


def test_provider_error_subclasses_are_uniformly_constructible_for_audit() -> None:
    detail = "audit-detail"
    provider = "audit-provider"
    error_types = provider_error_subclasses()

    first_instances = tuple(
        instantiate_provider_error(error_type, detail=detail, provider=provider)
        for error_type in error_types
    )
    second_instances = tuple(
        instantiate_provider_error(error_type, detail=detail, provider=provider)
        for error_type in error_types
    )

    assert tuple(type(error) for error in first_instances) == error_types
    assert tuple(str(error) for error in first_instances) == tuple(
        str(error) for error in second_instances
    )

    for error in first_instances:
        assert isinstance(error.retryable, bool)
        assert isinstance(error.code, str)
        assert error.code
        assert str(error).startswith(
            f"provider={provider} code={error.code} retryable={str(error.retryable).lower()}"
        )
        assert str(error).endswith(f"detail={detail}")

    for error_type in error_types:
        positional = error_type(detail)
        assert positional.provider == "provider"

    with pytest.raises(TypeError, match="concrete ProviderError subclass"):
        instantiate_provider_error(ProviderError, detail=detail, provider=provider)


def test_redact_transcript_redacts_prompt_and_response() -> None:
    fake_openai_key = "sk-" + "1234567890ABCDEFGHijkl"
    fake_anthropic_key = "sk-ant-" + "1234567890_ABCDEFGHijkl"
    fake_bearer_token = "SUPER" + "SECRETTOKEN"
    transcript = redact_transcript(
        provider="openai",
        model="gpt-4.1-mini",
        idempotency_key="idem-1",
        prompt=f"OPENAI_API_KEY={fake_openai_key} should be redacted",
        response=f"authorization: bearer {fake_bearer_token} should be redacted",
        error=f"anthropic key {fake_anthropic_key} should be redacted",
    )

    assert fake_openai_key not in transcript.prompt
    assert fake_bearer_token not in (transcript.response or "")
    assert fake_anthropic_key not in (transcript.error or "")
    assert transcript.idempotency_key == "idem-1"


@pytest.mark.unit
async def test_openai_adapter_normalizes_response_content_tools_and_usage() -> None:
    fake_api = _ScriptedOpenAIResponses(
        outcomes=deque(
            [
                {
                    "id": "resp-1",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Plan complete."},
                            ],
                        },
                        {
                            "type": "function_call",
                            "call_id": "call-1",
                            "name": "lookup_contract",
                            "arguments": '{"id": "REQ-001"}',
                        },
                    ],
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "total_tokens": 18,
                    },
                }
            ]
        )
    )
    adapter = OpenAIAdapter(model="gpt-4.1-mini", client=_FakeOpenAIClient(responses=fake_api))

    response = await adapter.send(_request(model="gpt-4.1-mini", prompt="Do the thing"))

    assert response.model == "gpt-4.1-mini"
    assert response.content == "Plan complete."
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 18
    assert response.request_id == "resp-1"
    assert response.idempotency_key is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].call_id == "call-1"
    assert response.tool_calls[0].name == "lookup_contract"
    assert response.tool_calls[0].arguments == {"id": "REQ-001"}

    assert len(fake_api.calls) == 1
    payload = fake_api.calls[0]
    assert payload["model"] == "gpt-4.1-mini"
    assert payload["extra_headers"] == {"Idempotency-Key": response.idempotency_key}


@pytest.mark.unit
async def test_openai_adapter_supports_tool_only_responses() -> None:
    fake_api = _ScriptedOpenAIResponses(
        outcomes=deque(
            [
                {
                    "id": "resp-tool-only",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call-only",
                            "name": "lookup_contract",
                            "arguments": '{"id": "REQ-TOOL"}',
                        }
                    ],
                }
            ]
        )
    )
    adapter = OpenAIAdapter(model="gpt-4.1-mini", client=_FakeOpenAIClient(responses=fake_api))

    response = await adapter.send(_request(model="gpt-4.1-mini", prompt="tool only"))

    assert response.content == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].call_id == "call-only"
    assert response.tool_calls[0].arguments == {"id": "REQ-TOOL"}


@pytest.mark.unit
async def test_openai_adapter_retries_rate_limit_with_stable_idempotency_key() -> None:
    fake_api = _ScriptedOpenAIResponses(
        outcomes=deque(
            [
                RateLimitError("too many requests"),
                {
                    "id": "resp-2",
                    "model": "gpt-4.1-mini",
                    "output_text": "Recovered",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                },
            ]
        )
    )

    sleep_recorder = _SleepRecorder()
    adapter = OpenAIAdapter(
        model="gpt-4.1-mini",
        client=_FakeOpenAIClient(responses=fake_api),
        backoff=BackoffConfig(
            max_retries=2,
            initial_delay_seconds=0.2,
            multiplier=2.0,
            max_delay_seconds=2.0,
        ),
        sleep=sleep_recorder,
    )

    response = await adapter.send(_request(model="gpt-4.1-mini", prompt="Retry please"))

    assert response.content == "Recovered"
    assert len(fake_api.calls) == 2
    assert sleep_recorder.calls == [0.2]

    first_key = fake_api.calls[0]["extra_headers"]
    second_key = fake_api.calls[1]["extra_headers"]
    assert first_key == second_key


@pytest.mark.unit
async def test_openai_adapter_auth_errors_are_not_retried() -> None:
    fake_api = _ScriptedOpenAIResponses(outcomes=deque([AuthenticationError("invalid api key")]))
    sleep_recorder = _SleepRecorder()

    adapter = OpenAIAdapter(
        model="gpt-4.1-mini",
        client=_FakeOpenAIClient(responses=fake_api),
        backoff=BackoffConfig(max_retries=4, initial_delay_seconds=0.1),
        sleep=sleep_recorder,
    )

    with pytest.raises(
        ProviderAuthError,
        match=r"provider=openai code=auth retryable=false .*detail=invalid api key",
    ):
        await adapter.send(_request(model="gpt-4.1-mini", prompt="Auth failure"))

    assert len(fake_api.calls) == 1
    assert sleep_recorder.calls == []


@pytest.mark.unit
async def test_openai_adapter_context_length_errors_are_not_retried() -> None:
    fake_api = _ScriptedOpenAIResponses(
        outcomes=deque([ContextLengthError("context length exceeded")])
    )
    sleep_recorder = _SleepRecorder()
    adapter = OpenAIAdapter(
        model="gpt-4.1-mini",
        client=_FakeOpenAIClient(responses=fake_api),
        backoff=BackoffConfig(max_retries=4, initial_delay_seconds=0.1),
        sleep=sleep_recorder,
    )

    with pytest.raises(
        ProviderContextLengthError,
        match=r"provider=openai code=context_length retryable=false",
    ):
        await adapter.send(_request(model="gpt-4.1-mini", prompt="too much context"))

    assert len(fake_api.calls) == 1
    assert sleep_recorder.calls == []


@pytest.mark.unit
async def test_anthropic_adapter_normalizes_text_tool_use_and_usage() -> None:
    fake_api = _ScriptedAnthropicMessages(
        outcomes=deque(
            [
                {
                    "id": "msg-1",
                    "model": "claude-3-7-sonnet",
                    "content": [
                        {"type": "text", "text": "Executing lookup"},
                        {
                            "type": "tool_use",
                            "id": "toolu-1",
                            "name": "lookup_contract",
                            "input": {"id": "REQ-777"},
                        },
                    ],
                    "usage": {"input_tokens": 14, "output_tokens": 9},
                    "stop_reason": "end_turn",
                }
            ]
        )
    )

    adapter = AnthropicAdapter(
        model="claude-3-7-sonnet", client=_FakeAnthropicClient(messages=fake_api)
    )
    response = await adapter.send(_request(model="claude-3-7-sonnet", prompt="Use tool"))

    assert response.content == "Executing lookup"
    assert response.model == "claude-3-7-sonnet"
    assert response.request_id == "msg-1"
    assert response.finish_reason == "end_turn"
    assert response.usage.input_tokens == 14
    assert response.usage.output_tokens == 9
    assert response.usage.total_tokens == 23
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].call_id == "toolu-1"
    assert response.tool_calls[0].name == "lookup_contract"
    assert response.tool_calls[0].arguments == {"id": "REQ-777"}


@pytest.mark.unit
async def test_anthropic_adapter_retries_connection_failures_and_reuses_idempotency_key() -> None:
    fake_api = _ScriptedAnthropicMessages(
        outcomes=deque(
            [
                APIConnectionError("socket reset"),
                {
                    "id": "msg-2",
                    "model": "claude-3-7-sonnet",
                    "content": [{"type": "text", "text": "Recovered"}],
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                },
            ]
        )
    )
    sleep_recorder = _SleepRecorder()

    adapter = AnthropicAdapter(
        model="claude-3-7-sonnet",
        client=_FakeAnthropicClient(messages=fake_api),
        backoff=BackoffConfig(max_retries=2, initial_delay_seconds=0.3, multiplier=2.0),
        sleep=sleep_recorder,
    )

    response = await adapter.send(_request(model="claude-3-7-sonnet", prompt="Retry me"))

    assert response.content == "Recovered"
    assert len(fake_api.calls) == 2
    assert sleep_recorder.calls == [0.3]
    assert fake_api.calls[0]["extra_headers"] == fake_api.calls[1]["extra_headers"]


@pytest.mark.unit
async def test_sdk_missing_raises_provider_unavailable_only_when_send_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real_import_module = importlib.import_module

    def blocking_import(name: str, package: str | None = None) -> object:
        if name in {"openai", "anthropic"}:
            raise ImportError(f"blocked: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocking_import)

    openai_adapter = OpenAIAdapter(model="gpt-4.1-mini")
    anthropic_adapter = AnthropicAdapter(model="claude-3-7-sonnet")

    # Constructor should not require vendor SDKs.
    assert openai_adapter.model == "gpt-4.1-mini"
    assert anthropic_adapter.model == "claude-3-7-sonnet"

    with pytest.raises(
        ProviderUnavailableError,
        match=r"provider=openai code=unavailable retryable=false detail=openai SDK is not installed",
    ):
        await openai_adapter.send(_request(model="gpt-4.1-mini", prompt="x"))

    with pytest.raises(
        ProviderUnavailableError,
        match=r"provider=anthropic code=unavailable retryable=false detail=anthropic SDK is not installed",
    ):
        await anthropic_adapter.send(_request(model="claude-3-7-sonnet", prompt="x"))


@pytest.mark.unit
async def test_explicit_idempotency_key_is_preserved_in_request_headers() -> None:
    fake_api = _ScriptedOpenAIResponses(
        outcomes=deque(
            [
                {
                    "id": "resp-x",
                    "model": "gpt-4.1-mini",
                    "output_text": "done",
                }
            ]
        )
    )

    adapter = OpenAIAdapter(model="gpt-4.1-mini", client=_FakeOpenAIClient(responses=fake_api))
    explicit_idempotency = "idem-" + "explicit-123"
    response = await adapter.send(
        _request(
            model="gpt-4.1-mini",
            prompt="Use explicit idempotency",
            idempotency_key=explicit_idempotency,
        )
    )

    assert fake_api.calls[0]["extra_headers"] == {"Idempotency-Key": explicit_idempotency}
    assert response.idempotency_key == explicit_idempotency
