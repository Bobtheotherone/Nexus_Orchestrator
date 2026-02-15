"""Design document generator — thin service over the architect-tier model.

File: src/nexus_orchestrator/ui/tui/services/design_doc.py

NO Textual imports. Uses the auth strategy to resolve the best available
provider (LOCAL_CLI tool backends preferred, API_KEY as fallback).
Generates structured engineering design documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from nexus_orchestrator.auth.strategy import AuthMode, BackendAuthStatus, resolve_auth
from nexus_orchestrator.synthesis_plane.providers.base import (
    ProviderProtocol,
    ProviderRequest,
    ProviderResponse,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-opus-4-6"
_DEFAULT_MAX_OUTPUT_TOKENS = 16_384

DESIGN_DOC_SYSTEM_PROMPT = """\
You are a senior software architect. Given the user's prompt, generate a \
structured engineering design document in Markdown format.

The document MUST use the following structure:

# <Concise Title>

## Problem Statement
Clearly define the problem or need this design addresses.

## Context & Background
Relevant context, prior art, and constraints that inform the design.

## Proposed Solution
High-level description of the approach.

## Implementation Details
Concrete implementation steps, algorithms, data flows, and component interactions.

## API / Interface Design
Public interfaces, endpoints, function signatures, or CLI commands (if applicable).

## Data Model Changes
Schema changes, new tables/fields, migration notes (if applicable).

## Security Considerations
Threat surface, authentication, authorization, input validation, secrets handling.

## Testing Strategy
Unit tests, integration tests, edge cases, and acceptance criteria.

## Rollout Plan
Phased rollout, feature flags, backward compatibility, and rollback strategy.

## Open Questions
Unresolved decisions or areas needing further investigation.

Write clearly and concisely. Use code blocks where helpful. \
Do NOT include preamble or meta-commentary — output only the design document."""


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesignDocResult:
    """Result of a design document generation."""

    title: str
    content: str
    model: str
    tokens_used: int
    cost_usd: float
    file_path: str | None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NoProviderAvailableError(RuntimeError):
    """Raised when no provider backend is available for generation."""

    def __init__(self, statuses: list[BackendAuthStatus] | None = None) -> None:
        self.statuses = statuses or []
        lines = [
            "No provider available for design doc generation.",
            "",
            "To use this feature, set up at least one of:",
            "  1. Claude Code CLI  — run `claude` to log in (no API key needed)",
            "  2. Codex CLI        — run `codex` to log in (no API key needed)",
            "  3. Anthropic API    — set ANTHROPIC_API_KEY env var",
            "",
            "Tip: Install a CLI tool for zero-key operation.",
        ]
        if statuses:
            lines.append("")
            lines.append("Detected backends:")
            for s in statuses:
                status = "available" if s.available else "unavailable"
                extra = ""
                if s.remediation:
                    extra = f" — {s.remediation}"
                lines.append(f"  {s.name}: {status}{extra}")
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DesignDocGenerator:
    """Design document generator using the best available provider.

    Resolution order (default, LOCAL_CLI preferred):
      1. Claude Code CLI (logged in) — no API key needed
      2. Codex CLI (logged in) — no API key needed
      3. Anthropic API (SDK + key set)
      4. OpenAI API (SDK + key set)

    If a provider is injected via constructor, it is used directly.
    """

    def __init__(
        self,
        *,
        provider: ProviderProtocol | None = None,
        model: str = _DEFAULT_MODEL,
        output_dir: Path | str | None = None,
        max_output_tokens: int = _DEFAULT_MAX_OUTPUT_TOKENS,
        prefer: AuthMode = AuthMode.LOCAL_CLI,
    ) -> None:
        self._provider = provider
        self._model = model
        self._output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self._max_output_tokens = max_output_tokens
        self._prefer = prefer

    async def generate(self, prompt: str) -> DesignDocResult:
        """Generate a design doc from the user prompt."""
        provider = self._ensure_provider()

        request = ProviderRequest(
            model=self._model,
            role_id="architect",
            system_prompt=DESIGN_DOC_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=self._max_output_tokens,
            temperature=0.4,
            reasoning_effort="high",
        )

        response: ProviderResponse = await provider.send(request)

        content = response.raw_text
        title = self._extract_title(content)
        tokens_used = response.usage.total_tokens
        cost_usd = response.usage.cost_estimate_usd or 0.0

        # Save to disk
        file_path: str | None = None
        try:
            out_path = self._resolve_output_path(title)
            self._save_to_disk(content, out_path)
            file_path = str(out_path)
        except OSError:
            pass

        return DesignDocResult(
            title=title,
            content=content,
            model=response.model,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            file_path=file_path,
        )

    def _ensure_provider(self) -> ProviderProtocol:
        if self._provider is not None:
            return self._provider

        auth = resolve_auth(prefer=self._prefer)
        if auth is None:
            from nexus_orchestrator.auth.strategy import detect_all_auth

            raise NoProviderAvailableError(detect_all_auth())

        self._provider = _create_provider_from_auth(auth, model=self._model)
        return self._provider

    def _extract_title(self, content: str) -> str:
        """Extract the H1 title from the generated markdown."""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                return stripped[2:].strip()
        return "Untitled Design Document"

    def _resolve_output_path(self, title: str) -> Path:
        """Determine where to save the .md file."""
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
        date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
        filename = f"{date_prefix}_{slug}.md"
        output_dir = self._output_dir / "designs"
        output_dir.mkdir(parents=True, exist_ok=True)
        candidate = output_dir / filename
        counter = 1
        while candidate.exists():
            candidate = output_dir / f"{date_prefix}_{slug}_{counter}.md"
            counter += 1
        return candidate

    def _save_to_disk(self, content: str, path: Path) -> None:
        """Write content to disk."""
        path.write_text(content + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Provider factory from auth status
# ---------------------------------------------------------------------------


def _create_provider_from_auth(
    auth: BackendAuthStatus,
    *,
    model: str,
) -> ProviderProtocol:
    """Create a provider instance from a resolved BackendAuthStatus."""
    if auth.auth_mode == AuthMode.LOCAL_CLI:
        from nexus_orchestrator.synthesis_plane.providers.tool_adapter import (
            ToolBackend,
            ToolProvider,
        )

        backend_map = {
            "claude": ToolBackend.CLAUDE_CODE,
            "codex": ToolBackend.CODEX_CLI,
        }
        backend = backend_map.get(auth.name)
        if backend is None:
            raise ValueError(f"Unknown CLI backend: {auth.name}")
        return ToolProvider(
            backend=backend,
            binary_path=auth.binary_path,
            timeout_seconds=300.0,  # Design docs may take a while
        )

    if auth.auth_mode == AuthMode.API_KEY:
        if auth.name == "anthropic":
            from nexus_orchestrator.synthesis_plane.providers.anthropic_adapter import (
                AnthropicProvider,
            )

            return AnthropicProvider(model=model)

        if auth.name == "openai":
            from nexus_orchestrator.synthesis_plane.providers.openai_adapter import (
                OpenAIProvider,
            )

            return OpenAIProvider(model=model)

    raise ValueError(f"Cannot create provider for backend: {auth.name} ({auth.auth_mode})")


__all__ = ["DesignDocGenerator", "DesignDocResult", "NoProviderAvailableError"]
