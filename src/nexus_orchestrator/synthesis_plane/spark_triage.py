"""Spark-based LLM triage for intelligent 3-model routing.

File: src/nexus_orchestrator/synthesis_plane/spark_triage.py

Purpose
- Calls GPT 5.3 Spark via codex CLI with a lightweight classification prompt.
- Parses the response to determine which of the 3 models should handle a task.
- Falls back to the deterministic classifier on any failure.

Security
- No secrets or API keys — uses the codex CLI's own authentication.
- Prompts contain only task metadata (title, description, scope, risk tier).
"""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from nexus_orchestrator.domain.models import WorkItem

from nexus_orchestrator.synthesis_plane.work_item_classifier import (
    ModelAffinity,
    classify_work_item,
)

# Default timeout for triage subprocess (seconds)
_DEFAULT_TRIAGE_TIMEOUT: Final[float] = 30.0

# Maximum scope entries to include in triage prompt
_MAX_SCOPE_ENTRIES: Final[int] = 15

_TRIAGE_PROMPT: Final[str] = """\
You are a task routing agent. Analyze the following software engineering task \
and decide which AI model should handle it.

Available models:
- GPT53: Best for complex reasoning, architecture design, multi-step planning, \
config and infrastructure tasks, design documents
- SPARK: Best for simple/quick tasks, small fixes, formatting, documentation \
updates, trivial single-file changes
- OPUS: Best for deep coding logic, large refactors, writing tests, \
parser/serializer implementation, multi-file code changes

Task title: {title}
Task description: {description}
Files in scope: {scope}
Risk tier: {risk_tier}
Number of constraints: {constraint_count}

Respond with EXACTLY one line in this format:
ROUTE: <GPT53|SPARK|OPUS>"""

_ROUTE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"ROUTE:\s*(GPT53|SPARK|OPUS)", re.IGNORECASE,
)

_AFFINITY_MAP: Final[dict[str, ModelAffinity]] = {
    "GPT53": ModelAffinity.GPT53,
    "SPARK": ModelAffinity.SPARK,
    "OPUS": ModelAffinity.OPUS,
}


@dataclass(frozen=True, slots=True)
class TriageResult:
    """Result of Spark triage classification."""

    chosen_model: ModelAffinity
    reasoning: str
    used_llm: bool  # False if fell back to deterministic classifier


async def triage_with_spark(
    work_item: WorkItem,
    *,
    codex_binary_path: str | None = None,
    timeout_seconds: float = _DEFAULT_TRIAGE_TIMEOUT,
    model_flag: str = "gpt-5.3-spark",
) -> TriageResult:
    """Call Spark to classify the work item. Falls back to deterministic on failure.

    This uses a direct subprocess call (not ToolProvider) to keep the triage
    lightweight and avoid circular dependencies with the dispatch system.
    """
    # Resolve codex binary
    binary = codex_binary_path
    if binary is None:
        binary = shutil.which("codex")
    if binary is None:
        return _deterministic_fallback(work_item, reason="codex CLI not found on PATH")

    # Build triage prompt
    scope_str = ", ".join(work_item.scope[:_MAX_SCOPE_ENTRIES])
    if len(work_item.scope) > _MAX_SCOPE_ENTRIES:
        scope_str += f" (+{len(work_item.scope) - _MAX_SCOPE_ENTRIES} more)"

    prompt = _TRIAGE_PROMPT.format(
        title=work_item.title,
        description=work_item.description[:500],
        scope=scope_str or "(no files specified)",
        risk_tier=getattr(work_item.risk_tier, "value", str(work_item.risk_tier)),
        constraint_count=len(work_item.constraint_envelope.constraints),
    )

    # Build command: codex exec --full-auto --model gpt-5.3-spark "prompt"
    cmd = [binary, "exec", "--full-auto"]
    if model_flag:
        cmd.extend(["--model", model_flag])
    cmd.append(prompt)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, _ = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except (TimeoutError, asyncio.TimeoutError):
        return _deterministic_fallback(work_item, reason="triage timed out")
    except OSError as exc:
        return _deterministic_fallback(work_item, reason=f"subprocess error: {exc}")

    if proc.returncode != 0:
        return _deterministic_fallback(
            work_item, reason=f"codex exited with code {proc.returncode}",
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    if not stdout:
        return _deterministic_fallback(work_item, reason="empty triage response")

    # Parse ROUTE: <MODEL> from response
    match = _ROUTE_PATTERN.search(stdout)
    if match is None:
        return _deterministic_fallback(
            work_item, reason=f"unparseable triage response: {stdout[:200]}",
        )

    model_key = match.group(1).upper()
    affinity = _AFFINITY_MAP.get(model_key)
    if affinity is None:
        return _deterministic_fallback(work_item, reason=f"unknown model: {model_key}")

    return TriageResult(
        chosen_model=affinity,
        reasoning=f"Spark triage → {model_key}",
        used_llm=True,
    )


def _deterministic_fallback(
    work_item: WorkItem, *, reason: str,
) -> TriageResult:
    """Fall back to the deterministic classifier."""
    affinity = classify_work_item(work_item)
    return TriageResult(
        chosen_model=affinity,
        reasoning=f"deterministic fallback ({reason})",
        used_llm=False,
    )


__all__ = [
    "TriageResult",
    "triage_with_spark",
]
