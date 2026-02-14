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
import shutil
import time
from typing import TYPE_CHECKING, Final

from nexus_orchestrator.synthesis_plane.providers.base import (
    BaseProvider,
    ProviderAuthenticationError,
    ProviderRequest,
    ProviderResponse,
    ProviderResponseError,
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
        timeout_seconds: float = 120.0,
        model_catalog: ModelCatalog | None = None,
    ) -> None:
        super().__init__(model_catalog=model_catalog)
        self._backend = backend
        self._timeout_seconds = timeout_seconds

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
        start = time.perf_counter()

        try:
            stdout, stderr, returncode = await self._execute_cli(prompt_text)
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
            raise ProviderResponseError(
                f"{self._backend.value} returned empty output",
                provider="tool",
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

    async def _execute_cli(self, prompt_text: str) -> tuple[str, str, int]:
        """Run the CLI tool and capture output."""
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
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        return (
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            proc.returncode or 0,
        )

    def _build_command(self, prompt_text: str) -> list[str]:
        """Build the subprocess command with prompt as CLI argument."""
        if self._backend == ToolBackend.CODEX_CLI:
            return [self._binary_path, "-q", prompt_text]
        # Claude Code: print mode with JSON output
        return [self._binary_path, "-p", prompt_text, "--output-format", "json"]

    def _build_command_stdin(self) -> list[str]:
        """Build command that reads prompt from stdin."""
        if self._backend == ToolBackend.CODEX_CLI:
            return [self._binary_path, "-q"]
        return [self._binary_path, "-p", "-", "--output-format", "json"]

    def _parse_output(self, stdout: str) -> str:
        """Parse tool output into response text."""
        if self._backend == ToolBackend.CLAUDE_CODE:
            return _parse_claude_json_output(stdout)
        # Codex: stdout is the raw text response
        return stdout.strip()

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


def _parse_claude_json_output(stdout: str) -> str:
    """Parse Claude Code JSON output to extract the result text.

    Claude Code with --output-format json returns a JSON object.
    We extract the text content from the result field.
    Falls back to raw stdout if JSON parsing fails.
    """
    stdout = stdout.strip()
    if not stdout:
        return ""

    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON — return raw text
        return stdout

    # Claude Code JSON format: {"result": "..."} or {"content": "..."}
    if isinstance(data, dict):
        for key in ("result", "content", "text", "response"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        # If it's a list of content blocks, concatenate text blocks
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

    # Fallback: return raw stdout
    return stdout


__all__ = [
    "TOOL_BINARY_NAMES",
    "ToolBackend",
    "ToolProvider",
]
