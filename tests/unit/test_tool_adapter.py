"""Unit tests for tool provider adapter.

Tests verify subprocess-based CLI execution with mocked asyncio subprocess
calls. No real codex/claude binaries are required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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
    TOOL_MODEL_SPEC,
    ToolBackend,
    ToolProvider,
    _assemble_prompt,
    _format_tool_code,
    _parse_claude_json_fallback,
    _parse_claude_stream_output,
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
    """Create a mock asyncio subprocess process.

    The new _execute_cli reads stdout via .read() or .readline() and stderr
    via .readline(), then awaits proc.wait(). We mock these stream interfaces.
    """
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")

    # Mock stdout stream — supports both read(n) and readline()
    mock_stdout = AsyncMock()
    _stdout_read_done = {"done": False}

    async def _stdout_read(n: int = -1) -> bytes:
        if _stdout_read_done["done"]:
            return b""
        _stdout_read_done["done"] = True
        return stdout_bytes

    _stdout_lines = (stdout_bytes.split(b"\n") if stdout_bytes else [])
    _stdout_line_iter = iter([line + b"\n" for line in _stdout_lines if line] + [b""])

    async def _stdout_readline() -> bytes:
        return next(_stdout_line_iter, b"")

    mock_stdout.read = _stdout_read
    mock_stdout.readline = _stdout_readline

    # Mock stderr stream — readline interface
    _stderr_lines = (stderr_bytes.split(b"\n") if stderr_bytes else [])
    _stderr_line_iter = iter([line + b"\n" for line in _stderr_lines if line] + [b""])
    mock_stderr = AsyncMock()

    async def _stderr_readline() -> bytes:
        return next(_stderr_line_iter, b"")

    mock_stderr.readline = _stderr_readline

    # Mock stdin (for _feed_stdin)
    mock_stdin = AsyncMock()
    mock_stdin.write = AsyncMock()
    mock_stdin.drain = AsyncMock(return_value=None)
    mock_stdin.close = AsyncMock()

    proc = AsyncMock()
    proc.stdout = mock_stdout
    proc.stderr = mock_stderr
    proc.stdin = mock_stdin
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
    async def test_claude_stream_json_success(self) -> None:
        # stream-json format: JSONL with a result event at the end
        stream_output = (
            json.dumps({"type": "system", "subtype": "init"}) + "\n"
            + json.dumps({"type": "result", "subtype": "success", "result": "Generated code here"}) + "\n"
        )
        proc = _mock_process(stdout=stream_output)
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

        # Make stdout read hang so the real timeout fires
        async def _hang(*args: object, **kwargs: object) -> bytes:
            await asyncio.sleep(999999)
            return b""  # unreachable

        proc.stdout.read = _hang
        proc.stdout.readline = _hang

        with patch("asyncio.create_subprocess_exec", return_value=proc):
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
                timeout_seconds=0.01,
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


class TestParseClaudeStreamOutput:
    """Tests for _parse_claude_stream_output() (stream-json JSONL format)."""

    def test_result_event_string(self) -> None:
        stdout = json.dumps({"type": "result", "subtype": "success", "result": "hello"})
        assert _parse_claude_stream_output(stdout) == "hello"

    def test_result_event_with_content_blocks(self) -> None:
        stdout = json.dumps({
            "type": "result",
            "subtype": "success",
            "result": [
                {"type": "text", "text": "part1"},
                {"type": "text", "text": "part2"},
            ],
        })
        assert _parse_claude_stream_output(stdout) == "part1\npart2"

    def test_multiline_jsonl(self) -> None:
        lines = [
            json.dumps({"type": "system", "subtype": "init"}),
            json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "thinking"}]}}),
            json.dumps({"type": "result", "subtype": "success", "result": "final answer"}),
        ]
        stdout = "\n".join(lines)
        assert _parse_claude_stream_output(stdout) == "final answer"

    def test_empty_string_fallback(self) -> None:
        assert _parse_claude_stream_output("") == ""

    def test_no_result_event_falls_back(self) -> None:
        stdout = json.dumps({"type": "system", "subtype": "init"})
        # Falls back to legacy parser which returns raw stdout
        result = _parse_claude_stream_output(stdout)
        assert result == stdout.strip()


class TestParseClaudeJsonFallback:
    """Tests for _parse_claude_json_fallback() (legacy single-JSON format)."""

    def test_result_field(self) -> None:
        stdout = json.dumps({"result": "hello"})
        assert _parse_claude_json_fallback(stdout) == "hello"

    def test_content_field(self) -> None:
        stdout = json.dumps({"content": "hello"})
        assert _parse_claude_json_fallback(stdout) == "hello"

    def test_result_list_of_text_blocks(self) -> None:
        stdout = json.dumps(
            {
                "result": [
                    {"type": "text", "text": "part1"},
                    {"type": "text", "text": "part2"},
                ]
            }
        )
        assert _parse_claude_json_fallback(stdout) == "part1\npart2"

    def test_invalid_json_returns_raw(self) -> None:
        stdout = "not json at all"
        assert _parse_claude_json_fallback(stdout) == "not json at all"

    def test_empty_string(self) -> None:
        assert _parse_claude_json_fallback("") == ""

    def test_fallback_to_raw_stdout(self) -> None:
        stdout = json.dumps({"unknown_key": 42})
        assert _parse_claude_json_fallback(stdout) == stdout.strip()


class TestWorkspaceIsolation:
    """Tests for workspace directory isolation in ToolProvider."""

    @pytest.mark.asyncio
    async def test_workspace_dir_passed_as_cwd(self, tmp_path: Path) -> None:
        """When workspace_dir is set, it must be passed as cwd to the subprocess."""
        proc = _mock_process(stdout="output")
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            request = ProviderRequest(
                model="codex_cli",
                role_id="implementer",
                user_prompt="test",
                workspace_dir=tmp_path,
            )
            await provider.send(request)
            mock_exec.assert_called_once()
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs.get("cwd") == str(tmp_path)

    @pytest.mark.asyncio
    async def test_no_workspace_dir_passes_none_cwd(self) -> None:
        """Without workspace_dir, cwd should be None (inherit parent cwd)."""
        proc = _mock_process(stdout="output")
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            await provider.send(_make_request())
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs.get("cwd") is None

    @pytest.mark.asyncio
    async def test_invalid_workspace_dir_raises_error(self, tmp_path: Path) -> None:
        """A workspace_dir that does not exist must raise ProviderServiceError."""
        provider = ToolProvider(
            backend=ToolBackend.CODEX_CLI,
            binary_path="/usr/bin/codex",
        )
        request = ProviderRequest(
            model="codex_cli",
            role_id="implementer",
            user_prompt="test",
            workspace_dir=tmp_path / "nonexistent",
        )
        with pytest.raises(ProviderServiceError, match="not a directory"):
            await provider.send(request)

    @pytest.mark.asyncio
    async def test_orchestrator_repo_root_rejected(self, tmp_path: Path) -> None:
        """workspace_dir that equals orchestrator_repo_root must be rejected."""
        provider = ToolProvider(
            backend=ToolBackend.CODEX_CLI,
            binary_path="/usr/bin/codex",
        )
        request = ProviderRequest(
            model="codex_cli",
            role_id="implementer",
            user_prompt="test",
            workspace_dir=tmp_path,
            metadata={"orchestrator_repo_root": str(tmp_path)},
        )
        with pytest.raises(ProviderServiceError, match="must not be the orchestrator"):
            await provider.send(request)

    @pytest.mark.asyncio
    async def test_workspace_dir_different_from_repo_root_allowed(
        self, tmp_path: Path,
    ) -> None:
        """workspace_dir that is NOT the orchestrator root should be allowed."""
        workspace = tmp_path / "workspaces" / "item1"
        workspace.mkdir(parents=True)
        proc = _mock_process(stdout="output")
        with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
            provider = ToolProvider(
                backend=ToolBackend.CODEX_CLI,
                binary_path="/usr/bin/codex",
            )
            request = ProviderRequest(
                model="codex_cli",
                role_id="implementer",
                user_prompt="test",
                workspace_dir=workspace,
                metadata={"orchestrator_repo_root": str(tmp_path)},
            )
            await provider.send(request)
            call_kwargs = mock_exec.call_args
            assert call_kwargs.kwargs.get("cwd") == str(workspace)


class TestModelFlag:
    """Tests for --model flag in CLI commands."""

    def test_codex_command_with_model_flag(self) -> None:
        provider = ToolProvider(
            backend=ToolBackend.CODEX_CLI,
            binary_path="/usr/bin/codex",
            model_flag="gpt-5.3",
        )
        cmd = provider._build_command("hello")
        assert "--model" in cmd
        assert "gpt-5.3" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "gpt-5.3"

    def test_claude_command_with_model_flag(self) -> None:
        provider = ToolProvider(
            backend=ToolBackend.CLAUDE_CODE,
            binary_path="/usr/bin/claude",
            model_flag="claude-opus-4-6",
        )
        cmd = provider._build_command("hello")
        assert "--model" in cmd
        assert "claude-opus-4-6" in cmd

    def test_codex_command_without_model_flag(self) -> None:
        provider = ToolProvider(
            backend=ToolBackend.CODEX_CLI,
            binary_path="/usr/bin/codex",
        )
        cmd = provider._build_command("hello")
        assert "--model" not in cmd

    def test_claude_command_without_model_flag(self) -> None:
        provider = ToolProvider(
            backend=ToolBackend.CLAUDE_CODE,
            binary_path="/usr/bin/claude",
        )
        cmd = provider._build_command("hello")
        assert "--model" not in cmd

    def test_codex_stdin_command_with_model_flag(self) -> None:
        provider = ToolProvider(
            backend=ToolBackend.CODEX_CLI,
            binary_path="/usr/bin/codex",
            model_flag="gpt-5.3-spark",
        )
        cmd = provider._build_command_stdin()
        assert "--model" in cmd
        assert "gpt-5.3-spark" in cmd
        # "-" should be last (stdin marker)
        assert cmd[-1] == "-"

    def test_tool_model_spec_mapping(self) -> None:
        """Verify TOOL_MODEL_SPEC has entries for all expected models."""
        assert "codex_gpt53" in TOOL_MODEL_SPEC
        assert "codex_spark" in TOOL_MODEL_SPEC
        assert "claude_opus" in TOOL_MODEL_SPEC
        assert "codex_cli" in TOOL_MODEL_SPEC
        assert "claude_code" in TOOL_MODEL_SPEC

        # New models have non-empty flags
        assert TOOL_MODEL_SPEC["codex_gpt53"][1] == "gpt-5.3"
        assert TOOL_MODEL_SPEC["codex_spark"][1] == "gpt-5.3-spark"
        assert TOOL_MODEL_SPEC["claude_opus"][1] == "claude-opus-4-6"

        # Legacy models have empty flags
        assert TOOL_MODEL_SPEC["codex_cli"][1] == ""
        assert TOOL_MODEL_SPEC["claude_code"][1] == ""


class TestFormatToolCode:
    """Tests for _format_tool_code() code extraction."""

    def test_write_tool(self) -> None:
        lines = _format_tool_code("Write", {
            "file_path": "src/main.py",
            "content": "def hello():\n    print('hi')\n",
        })
        assert len(lines) > 0
        assert any("Write: src/main.py" in line for line in lines)
        assert any("| def hello():" in line for line in lines)
        assert any("--- end ---" in line for line in lines)

    def test_edit_tool_with_old_and_new(self) -> None:
        lines = _format_tool_code("Edit", {
            "file_path": "src/foo.py",
            "old_string": "old_code()",
            "new_string": "new_code()",
        })
        assert any("Edit: src/foo.py" in line for line in lines)
        assert any("- old_code()" in line for line in lines)
        assert any(">>>" in line for line in lines)
        assert any("+ new_code()" in line for line in lines)

    def test_bash_tool(self) -> None:
        lines = _format_tool_code("Bash", {
            "command": "pytest tests/ -v",
        })
        assert any("Bash" in line for line in lines)
        assert any("$ pytest tests/ -v" in line for line in lines)

    def test_unknown_tool_returns_empty(self) -> None:
        lines = _format_tool_code("Grep", {"pattern": "foo"})
        assert lines == []

    def test_write_multiline_content(self) -> None:
        content = "\n".join([f"line {i}" for i in range(20)])
        lines = _format_tool_code("Write", {
            "file_path": "test.py",
            "content": content,
        })
        # All 20 lines should appear (no truncation)
        code_lines = [l for l in lines if l.startswith("  | ")]
        assert len(code_lines) == 20
