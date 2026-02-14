"""Unit tests for tool provider adapter.

Tests verify subprocess-based CLI execution with mocked asyncio subprocess
calls. No real codex/claude binaries are required.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from nexus_orchestrator.synthesis_plane.providers.base import (
    ProviderAuthenticationError,
    ProviderRequest,
    ProviderResponseError,
    ProviderServiceError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from nexus_orchestrator.synthesis_plane.providers.tool_adapter import (
    ToolBackend,
    ToolProvider,
    _assemble_prompt,
    _parse_claude_json_output,
)


def _make_request(
    *,
    user_prompt: str = "Write hello world",
    system_prompt: str = "You are a helpful assistant.",
    model: str = "codex_cli",
) -> ProviderRequest:
    return ProviderRequest(
        model=model,
        role_id="implementer",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _mock_process(
    stdout: str = "Hello world!",
    stderr: str = "",
    returncode: int = 0,
) -> AsyncMock:
    """Create a mock asyncio subprocess process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode("utf-8"), stderr.encode("utf-8")))
    proc.returncode = returncode
    proc.kill = AsyncMock()
    proc.wait = AsyncMock()
    return proc


class TestToolProviderInit:
    """Tests for ToolProvider construction."""

    def test_binary_not_found_raises_unavailable(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(ProviderUnavailableError, match="not found on PATH"),
        ):
            ToolProvider(backend=ToolBackend.CODEX_CLI)

    def test_explicit_binary_path(self) -> None:
        provider = ToolProvider(
            backend=ToolBackend.CODEX_CLI,
            binary_path="/usr/local/bin/codex",
        )
        assert provider._binary_path == "/usr/local/bin/codex"

    def test_auto_detect_binary(self) -> None:
        with patch("shutil.which", return_value="/usr/bin/codex"):
            provider = ToolProvider(backend=ToolBackend.CODEX_CLI)
            assert provider._binary_path == "/usr/bin/codex"


class TestToolProviderSend:
    """Tests for ToolProvider.send()."""

    @pytest.mark.asyncio
    async def test_codex_success(self) -> None:
        proc = _mock_process(stdout="Hello world!")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            response = await provider.send(_make_request())
            assert response.raw_text == "Hello world!"
            assert response.model == "codex_cli"
            assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_claude_json_success(self) -> None:
        json_output = json.dumps({"result": "Generated code here"})
        proc = _mock_process(stdout=json_output)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CLAUDE_CODE,
                binary_path="/usr/bin/claude",
            )
            response = await provider.send(_make_request(model="claude_code"))
            assert response.raw_text == "Generated code here"
            assert response.model == "claude_code"

    @pytest.mark.asyncio
    async def test_usage_fields_zero(self) -> None:
        proc = _mock_process(stdout="output")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            response = await provider.send(_make_request())
            assert response.usage.input_tokens == 0
            assert response.usage.output_tokens == 0
            assert response.usage.total_tokens == 0
            assert response.usage.cost_estimate_usd == 0.0

    @pytest.mark.asyncio
    async def test_latency_measured(self) -> None:
        proc = _mock_process(stdout="output")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            response = await provider.send(_make_request())
            assert response.usage.latency_ms is not None
            assert response.usage.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self) -> None:
        proc = _mock_process()
        proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
                timeout_seconds=1.0,
            )
            with pytest.raises(ProviderTimeoutError, match="timed out"):
                await provider.send(_make_request())

    @pytest.mark.asyncio
    async def test_auth_error_in_stderr(self) -> None:
        proc = _mock_process(stdout="", stderr="Error: not logged in", returncode=1)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            with pytest.raises(ProviderAuthenticationError, match="log in"):
                await provider.send(_make_request())

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_service_error(self) -> None:
        proc = _mock_process(stdout="", stderr="Some error", returncode=1)
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            with pytest.raises(ProviderServiceError, match="exited with code 1"):
                await provider.send(_make_request())

    @pytest.mark.asyncio
    async def test_empty_stdout_raises_response_error(self) -> None:
        proc = _mock_process(stdout="", stderr="")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            with pytest.raises(ProviderResponseError, match="empty output"):
                await provider.send(_make_request())

    @pytest.mark.asyncio
    async def test_idempotency_key_preserved(self) -> None:
        proc = _mock_process(stdout="output")
        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            request = ProviderRequest(
                model="codex_cli",
                role_id="implementer",
                user_prompt="test",
                idempotency_key="test-key-123",
            )
            response = await provider.send(request)
            assert response.idempotency_key == "test-key-123"


class TestAssemblePrompt:
    """Tests for _assemble_prompt()."""

    def test_includes_system_prompt(self) -> None:
        request = _make_request(
            system_prompt="You are an expert coder.",
            user_prompt="Write a function.",
        )
        text = _assemble_prompt(request)
        assert "[System]" in text
        assert "You are an expert coder." in text
        assert "[Task]" in text
        assert "Write a function." in text

    def test_empty_system_prompt_omitted(self) -> None:
        request = _make_request(system_prompt="", user_prompt="Do something.")
        text = _assemble_prompt(request)
        assert "[System]" not in text
        assert "[Task]" in text
        assert "Do something." in text

    def test_includes_context_docs(self) -> None:
        from nexus_orchestrator.synthesis_plane.providers.base import ContextDocument

        request = ProviderRequest(
            model="codex_cli",
            role_id="implementer",
            system_prompt="sys",
            user_prompt="task",
            context_docs=[
                ContextDocument(name="file.py", content="def hello(): pass", path="src/file.py"),
            ],
        )
        text = _assemble_prompt(request)
        assert "[Context]" in text
        assert "--- file.py ---" in text
        assert "src/file.py" in text
        assert "def hello(): pass" in text


class TestParseClaudeJsonOutput:
    """Tests for _parse_claude_json_output()."""

    def test_result_field(self) -> None:
        stdout = json.dumps({"result": "hello"})
        assert _parse_claude_json_output(stdout) == "hello"

    def test_content_field(self) -> None:
        stdout = json.dumps({"content": "hello"})
        assert _parse_claude_json_output(stdout) == "hello"

    def test_result_list_of_text_blocks(self) -> None:
        stdout = json.dumps(
            {
                "result": [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ]
            }
        )
        assert _parse_claude_json_output(stdout) == "part1\npart2"

    def test_invalid_json_returns_raw(self) -> None:
        stdout = "not json at all"
        assert _parse_claude_json_output(stdout) == "not json at all"

    def test_empty_string(self) -> None:
        assert _parse_claude_json_output("") == ""

    def test_fallback_to_raw_stdout(self) -> None:
        stdout = json.dumps({"unknown_key": 42})
        assert _parse_claude_json_output(stdout) == stdout.strip()
