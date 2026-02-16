"""Tool provider adapter — delegates to local CLI tools (codex, claude).

File: src/nexus_orchestrator/synthesis_plane/providers/tool_adapter.py

Purpose
- Provider adapter that executes prompts via locally-installed CLI tools.
- Supports OpenAI Codex CLI and Anthropic Claude Code CLI.
- Uses non-interactive subprocess mode (no PTY required).

Security
- Never extracts or logs auth tokens, cookies, or API keys.
- CLI tools manage their own authentication via browser-based login.
- Prompts and responses are passed via subprocess stdout/stderr only.
"""

from __future__ import annotations

import asyncio
import enum
import json
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

from nexus_orchestrator.synthesis_plane.providers.base import (
    BaseProvider,
    ProviderAuthenticationError,
    ProviderRequest,
    ProviderResponse,
    ProviderServiceError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderUsage,
)

if TYPE_CHECKING:
    from nexus_orchestrator.synthesis_plane.model_catalog import ModelCatalog


class ToolBackend(enum.Enum):
    """Supported CLI tool backends."""

    CODEX_CLI = "codex_cli"
    CLAUDE_CODE = "claude_code"


# Map backend enum to binary name
TOOL_BINARY_NAMES: Final[dict[ToolBackend, str]] = {
    ToolBackend.CODEX_CLI: "codex",
    ToolBackend.CLAUDE_CODE: "claude",
}

# Auth-related keywords in stderr that indicate login is needed
_AUTH_KEYWORDS: Final[tuple[str, ...]] = (
    "auth",
    "login",
    "unauthorized",
    "not logged in",
    "sign in",
    "authentication",
)

# Maximum prompt size before switching from CLI arg to stdin pipe
_STDIN_THRESHOLD: Final[int] = 100_000

# Maps catalog model name → (ToolBackend, CLI --model flag value)
# Empty string means no --model flag (legacy default behaviour)
TOOL_MODEL_SPEC: Final[dict[str, tuple[ToolBackend, str]]] = {
    "codex_gpt53": (ToolBackend.CODEX_CLI, "gpt-5.3-codex"),
    "codex_spark": (ToolBackend.CODEX_CLI, "gpt-5.3-codex-spark"),
    "claude_opus": (ToolBackend.CLAUDE_CODE, "claude-opus-4-6"),
    "codex_cli": (ToolBackend.CODEX_CLI, ""),
    "claude_code": (ToolBackend.CLAUDE_CODE, ""),
}


class ToolProvider(BaseProvider):
    """Provider that delegates to local CLI tools (codex/claude) via subprocess.

    This provider does NOT use API keys. The CLI tools handle their own
    authentication via browser-based login flows managed by those tools.
    Billing is governed by the user's existing subscription to those services.
    """

    provider_name = "tool"

    def __init__(
        self,
        *,
        backend: ToolBackend,
        binary_path: str | None = None,
        timeout_seconds: float = 600.0,
        model_catalog: ModelCatalog | None = None,
        model_flag: str = "",
        catalog_model: str = "",
    ) -> None:
        super().__init__(model_catalog=model_catalog)
        self._backend = backend
        self._timeout_seconds = timeout_seconds
        self._model_flag = model_flag
        self._catalog_model = catalog_model

        if binary_path is not None:
            self._binary_path = binary_path
        else:
            binary_name = TOOL_BINARY_NAMES[backend]
            resolved = shutil.which(binary_name)
            if resolved is None:
                raise ProviderUnavailableError(
                    f"{binary_name} CLI not found on PATH. "
                    f"Install it or ensure it is in your PATH.",
                    provider="tool",
                )
            self._binary_path = resolved

    async def send(self, request: ProviderRequest) -> ProviderResponse:
        """Execute a prompt via the CLI tool subprocess."""
        prompt_text = _assemble_prompt(request)
        workspace_cwd = self._resolve_workspace_dir(request)
        start = time.perf_counter()

        try:
            stdout, stderr, returncode = await self._execute_cli(
                prompt_text, cwd=workspace_cwd,
            )
        except TimeoutError as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            raise ProviderTimeoutError(
                f"{self._backend.value} timed out after {self._timeout_seconds}s "
                f"(elapsed {elapsed_ms}ms)",
                provider="tool",
            ) from exc

        elapsed_ms = int((time.perf_counter() - start) * 1000)

        if returncode != 0:
            self._handle_error(stderr, returncode)

        # Parse response text
        response_text = self._parse_output(stdout)
        if not response_text.strip():
            raise ProviderServiceError(
                f"{self._backend.value} returned empty output "
                f"(stdout={len(stdout)} bytes, returncode={returncode})",
                provider="tool",
                retryable=True,
            )

        return ProviderResponse(
            model=self._backend.value,
            raw_text=response_text,
            usage=ProviderUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=elapsed_ms,
                cost_estimate_usd=0.0,
            ),
            finish_reason="stop",
            idempotency_key=request.idempotency_key,
        )

    async def _execute_cli(
        self, prompt_text: str, *, cwd: str | None = None,
    ) -> tuple[str, str, int]:
        """Run the CLI tool and capture output, streaming stderr for visibility."""
        cmd = self._build_command(prompt_text)
        stdin_data: bytes | None = None

        # For large prompts, pipe via stdin instead of CLI arg
        if len(prompt_text) > _STDIN_THRESHOLD:
            cmd = self._build_command_stdin()
            stdin_data = prompt_text.encode("utf-8")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        model_tag = _model_tag_for(self._catalog_model, self._backend)
        palette = _palette_for_model(self._catalog_model)

        async def _feed_stdin() -> None:
            if stdin_data is not None and proc.stdin is not None:
                proc.stdin.write(stdin_data)
                await proc.stdin.drain()
                proc.stdin.close()

        async def _read_stdout() -> None:
            assert proc.stdout is not None  # noqa: S101
            if self._backend == ToolBackend.CLAUDE_CODE:
                # Stream-json: read line-by-line and echo events for TUI
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    stdout_chunks.append(line)
                    _echo_stream_event(line, model_tag, palette)
            else:
                # Codex: read in chunks (raw text result)
                while True:
                    chunk = await proc.stdout.read(8192)
                    if not chunk:
                        break
                    stdout_chunks.append(chunk)

        async def _read_stderr() -> None:
            assert proc.stderr is not None  # noqa: S101
            # Throttle stderr echo to prevent output flooding.
            # Codex CLI writes full file diffs to stderr; without
            # throttling this generates thousands of transcript lines.
            _echoed = 0
            _max_echo = 200  # max lines echoed per invocation
            _suppressed = 0
            _quiet_mode = False
            # Patterns worth echoing even in quiet mode
            _important_re = re.compile(
                r"(Agent complete|error|Error|ERROR|tokens used|"
                r"---\s*(Write|Edit|Bash).*---|---\s*end\s*---)",
            )
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                stderr_chunks.append(line)
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                if not _quiet_mode:
                    _echoed += 1
                    if _echoed > _max_echo:
                        _quiet_mode = True
                        _emit(
                            f"  [{model_tag}] ... (verbose output suppressed, "
                            f"showing summaries only)",
                            palette.text,
                        )
                    else:
                        _emit(f"  [{model_tag}] {text}", palette.text)
                else:
                    # In quiet mode, only echo important lines
                    if _important_re.search(text):
                        _emit(f"  [{model_tag}] {text}", palette.text)
                    else:
                        _suppressed += 1
            if _suppressed > 0:
                _emit(
                    f"  [{model_tag}] ({_suppressed} verbose lines suppressed)",
                    palette.text,
                )

        try:
            await asyncio.wait_for(
                asyncio.gather(_feed_stdin(), _read_stdout(), _read_stderr()),
                timeout=self._timeout_seconds,
            )
            await proc.wait()
        except (TimeoutError, asyncio.TimeoutError):
            # asyncio.TimeoutError != builtins.TimeoutError on Python <3.11
            proc.kill()
            await proc.wait()
            raise TimeoutError(
                f"{self._backend.value} timed out after {self._timeout_seconds}s"
            )

        return (
            b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            proc.returncode or 0,
        )

    def _build_command(self, prompt_text: str) -> list[str]:
        """Build the subprocess command with prompt as CLI argument."""
        if self._backend == ToolBackend.CODEX_CLI:
            cmd = [self._binary_path, "exec", "--full-auto"]
            if self._model_flag:
                cmd.extend(["-m", self._model_flag])
            cmd.append(prompt_text)
            return cmd
        # Claude Code: print mode with streaming JSON for live progress
        cmd = [self._binary_path, "-p", prompt_text, "--verbose", "--output-format", "stream-json"]
        if self._model_flag:
            cmd.extend(["--model", self._model_flag])
        return cmd

    def _build_command_stdin(self) -> list[str]:
        """Build command that reads prompt from stdin."""
        if self._backend == ToolBackend.CODEX_CLI:
            cmd = [self._binary_path, "exec", "--full-auto"]
            if self._model_flag:
                cmd.extend(["-m", self._model_flag])
            cmd.append("-")
            return cmd
        cmd = [self._binary_path, "-p", "-", "--verbose", "--output-format", "stream-json"]
        if self._model_flag:
            cmd.extend(["--model", self._model_flag])
        return cmd

    def _parse_output(self, stdout: str) -> str:
        """Parse tool output into response text."""
        if self._backend == ToolBackend.CLAUDE_CODE:
            return _parse_claude_stream_output(stdout)
        # Codex: stdout is the raw text response
        return stdout.strip()

    def _resolve_workspace_dir(self, request: ProviderRequest) -> str | None:
        """Validate and resolve workspace directory for subprocess cwd.

        Safety invariant: agents must NEVER run inside the orchestrator
        repository itself.  The orchestrator repo root is communicated
        via ``request.metadata["orchestrator_repo_root"]``.
        """
        workspace_dir = request.workspace_dir
        if workspace_dir is None:
            return None
        resolved = Path(workspace_dir).resolve(strict=False)
        if not resolved.is_dir():
            raise ProviderServiceError(
                f"workspace_dir does not exist or is not a directory: {resolved}",
                provider="tool",
                retryable=False,
            )
        # Safety: reject workspace_dir that is the orchestrator repo itself
        orchestrator_root_raw = request.metadata.get("orchestrator_repo_root")
        if isinstance(orchestrator_root_raw, str) and orchestrator_root_raw:
            orchestrator_root = Path(orchestrator_root_raw).resolve(strict=False)
            if resolved == orchestrator_root:
                raise ProviderServiceError(
                    "workspace_dir must not be the orchestrator repository root — "
                    "agents must work in an isolated workspace, never the orchestrator repo",
                    provider="tool",
                    retryable=False,
                )
        return str(resolved)

    def _handle_error(self, stderr: str, returncode: int) -> None:
        """Map subprocess error to appropriate ProviderError."""
        stderr_lower = stderr.lower()

        if any(kw in stderr_lower for kw in _AUTH_KEYWORDS):
            binary_name = TOOL_BINARY_NAMES[self._backend]
            raise ProviderAuthenticationError(
                f"Not logged in. Please run `{binary_name}` directly to log in, then retry.",
                provider="tool",
            )

        raise ProviderServiceError(
            f"{self._backend.value} exited with code {returncode}: "
            f"{stderr.strip()[:200] if stderr.strip() else '(no stderr)'}",
            provider="tool",
        )


def _assemble_prompt(request: ProviderRequest) -> str:
    """Assemble a single text prompt from the provider request fields."""
    parts: list[str] = []

    if request.system_prompt:
        parts.append(f"[System]\n{request.system_prompt}\n")

    if request.context_docs:
        parts.append("[Context]")
        for doc in request.context_docs:
            header = f"--- {doc.name} ---"
            if doc.path:
                header += f" ({doc.path})"
            parts.append(header)
            parts.append(doc.content)
        parts.append("")

    parts.append(f"[Task]\n{request.user_prompt}")

    return "\n".join(parts)


def _echo_stream_event(raw_line: bytes, tag: str, palette: _ModelPalette) -> None:
    """Parse a stream-json line and echo relevant content for TUI visibility.

    Shows full agent output including code from tool_use blocks with
    model-specific ANSI coloring (green=GPT, orange=Anthropic, blue=system).
    """
    text = raw_line.decode("utf-8", errors="replace").strip()
    if not text:
        return
    try:
        event = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return

    event_type = event.get("type")

    if event_type == "assistant":
        message = event.get("message", {})
        if not isinstance(message, dict):
            return
        content = message.get("content", [])
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                block_text = block.get("text", "")
                if block_text:
                    for line in block_text.splitlines():
                        line = line.strip()
                        if line:
                            _emit(f"  [{tag}] {line}", palette.text)
            elif block_type == "tool_use":
                tool_name = block.get("name", "?")
                tool_input = block.get("input", {})
                summary = _summarize_tool_use(tool_name, tool_input)
                _emit(f"  [{tag}] {summary}", palette.tool_header)
                # Show full code content from tool_use blocks
                for code_line in _format_tool_code(tool_name, tool_input):
                    _emit(f"  [{tag}] {code_line}", palette.code)

    elif event_type == "result":
        subtype = event.get("subtype", "done")
        cost = event.get("cost_usd")
        dur = event.get("duration_ms")
        parts = [f"Agent complete ({subtype})"]
        if isinstance(dur, (int, float)):
            parts.append(f"{dur / 1000:.1f}s")
        if isinstance(cost, (int, float)):
            parts.append(f"${cost:.4f}")
        _emit(f"  [{tag}] {' | '.join(parts)}", palette.result)


# --- ANSI 24-bit true color codes ---
_CLR_RESET: Final[str] = "\033[0m"


class _ModelPalette(NamedTuple):
    """ANSI color palette for a model family."""

    tool_header: str
    file_path: str
    code: str
    text: str
    result: str


# Green palette — GPT models (codex_gpt53, codex_spark)
_PALETTE_GREEN: Final[_ModelPalette] = _ModelPalette(
    tool_header="\033[1;38;2;78;201;144m",   # #4ec990 bold
    file_path="\033[38;2;126;232;181m",       # #7ee8b5
    code="\033[38;2;200;230;208m",            # #c8e6d0
    text="\033[38;2;125;163;138m",            # #7da38a
    result="\033[1;38;2;78;201;144m",         # #4ec990 bold
)

# Orange palette — Anthropic models (claude_opus)
_PALETTE_ORANGE: Final[_ModelPalette] = _ModelPalette(
    tool_header="\033[1;38;2;245;166;35m",    # #f5a623 bold
    file_path="\033[38;2;255;200;112m",       # #ffc870
    code="\033[38;2;230;216;200m",            # #e6d8c8
    text="\033[38;2;163;137;127m",            # #a3897f
    result="\033[1;38;2;245;166;35m",         # #f5a623 bold
)

# Blue palette — system/nexus and legacy fallback
_PALETTE_BLUE: Final[_ModelPalette] = _ModelPalette(
    tool_header="\033[1;38;2;63;169;245m",    # #3fa9f5 bold
    file_path="\033[38;2;114;199;255m",       # #72c7ff
    code="\033[38;2;200;205;216m",            # #c8cdd8
    text="\033[38;2;127;138;163m",            # #7f8aa3
    result="\033[38;2;78;201;144m",           # #4ec990
)

# Model-specific tag names for TUI display
_MODEL_TAG_MAP: Final[dict[str, str]] = {
    "codex_gpt53": "gpt53",
    "codex_spark": "spark",
    "claude_opus": "opus",
}


def _model_tag_for(catalog_model: str, backend: ToolBackend) -> str:
    """Return a short display tag for the given catalog model."""
    tag = _MODEL_TAG_MAP.get(catalog_model)
    if tag:
        return tag
    return "codex" if backend == ToolBackend.CODEX_CLI else "claude"


def _palette_for_model(catalog_model: str) -> _ModelPalette:
    """Return the color palette for the given catalog model name."""
    if catalog_model in ("codex_gpt53", "codex_spark"):
        return _PALETTE_GREEN
    if catalog_model in ("claude_opus",):
        return _PALETTE_ORANGE
    return _PALETTE_BLUE


def _emit(line: str, color: str = "") -> None:
    """Print a line with optional ANSI color."""
    if color:
        print(f"{color}{line}{_CLR_RESET}", flush=True)
    else:
        print(line, flush=True)


# Maximum code lines shown per tool block before truncation.
# Keeps transcript readable and prevents output flooding.
_MAX_TOOL_CODE_LINES: Final[int] = 30


def _format_tool_code(name: str, inputs: object) -> list[str]:
    """Extract and format code content from a tool_use block.

    Truncates after _MAX_TOOL_CODE_LINES to prevent flooding.
    """
    if not isinstance(inputs, dict):
        return []
    lines: list[str] = []
    limit = _MAX_TOOL_CODE_LINES

    if name in ("Write", "write") and "content" in inputs:
        file_path = inputs.get("file_path", "?")
        content_lines = str(inputs["content"]).splitlines()
        lines.append(f"  --- Write: {file_path} ({len(content_lines)} lines) ---")
        for code_line in content_lines[:limit]:
            lines.append(f"  | {code_line}")
        if len(content_lines) > limit:
            lines.append(f"  ... ({len(content_lines) - limit} more lines)")
        lines.append("  --- end ---")

    elif name in ("Edit", "edit") and "new_string" in inputs:
        file_path = inputs.get("file_path", "?")
        lines.append(f"  --- Edit: {file_path} ---")
        if "old_string" in inputs:
            old_lines = str(inputs["old_string"]).splitlines()
            for code_line in old_lines[:limit]:
                lines.append(f"  - {code_line}")
            if len(old_lines) > limit:
                lines.append(f"  ... ({len(old_lines) - limit} more lines)")
            lines.append("  >>>")
        new_lines = str(inputs["new_string"]).splitlines()
        for code_line in new_lines[:limit]:
            lines.append(f"  + {code_line}")
        if len(new_lines) > limit:
            lines.append(f"  ... ({len(new_lines) - limit} more lines)")
        lines.append("  --- end ---")

    elif name in ("Bash", "bash") and "command" in inputs:
        cmd_lines = str(inputs["command"]).splitlines()
        lines.append("  --- Bash ---")
        for code_line in cmd_lines[:limit]:
            lines.append(f"  $ {code_line}")
        if len(cmd_lines) > limit:
            lines.append(f"  ... ({len(cmd_lines) - limit} more lines)")
        lines.append("  --- end ---")

    return lines


def _summarize_tool_use(name: str, inputs: object) -> str:
    """Build a short summary of a tool_use block for TUI display."""
    if not isinstance(inputs, dict):
        return name
    if "file_path" in inputs:
        return f"{name}: {inputs['file_path']}"
    if "command" in inputs:
        cmd = str(inputs["command"]).splitlines()[0][:120]
        return f"{name}: {cmd}"
    if "pattern" in inputs:
        return f"{name}: {inputs['pattern']}"
    if "query" in inputs:
        return f"{name}: {inputs['query']}"
    return name


def _parse_claude_stream_output(stdout: str) -> str:
    """Parse Claude Code stream-json output to extract the final result.

    Stream-json format is JSONL.  We try three strategies in order:

    1. Find a ``type="result"`` event and extract its ``result`` field.
    2. Collect all text blocks from ``type="assistant"`` events.
    3. Fall back to the legacy single-JSON parser.

    This robustness is important because the CLI may crash, timeout,
    or omit the final result event while still having produced useful
    assistant output earlier in the stream.
    """
    lines = stdout.splitlines()

    # Strategy 1 — look for a result event (scan from end)
    for raw_line in reversed(lines):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            continue
        if event.get("type") != "result":
            continue

        result = event.get("result")
        if isinstance(result, str) and result.strip():
            return result
        if isinstance(result, list):
            texts = _extract_text_blocks(result)
            if texts:
                return "\n".join(texts)

    # Strategy 2 — collect all text from assistant events
    all_texts: list[str] = []
    for raw_line in lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, ValueError):
            continue
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            all_texts.extend(_extract_text_blocks(content))
    if all_texts:
        return "\n".join(all_texts)

    # Strategy 3 — legacy single-JSON parser
    return _parse_claude_json_fallback(stdout)


def _extract_text_blocks(content: list[object]) -> list[str]:
    """Extract text from a list of content blocks."""
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text", "")
            if isinstance(t, str) and t.strip():
                texts.append(t)
    return texts


def _parse_claude_json_fallback(stdout: str) -> str:
    """Legacy parser for Claude Code --output-format json."""
    stdout = stdout.strip()
    if not stdout:
        return ""

    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return stdout

    if isinstance(data, dict):
        for key in ("result", "content", "text", "response"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        result_value = data.get("result")
        if isinstance(result_value, list):
            texts: list[str] = []
            for block in result_value:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                return "\n".join(texts)

    return stdout


__all__ = [
    "TOOL_BINARY_NAMES",
    "TOOL_MODEL_SPEC",
    "ToolBackend",
    "ToolProvider",
    "_palette_for_model",
]
