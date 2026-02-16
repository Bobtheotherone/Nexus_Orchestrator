"""Work item classifier for intelligent model delegation.

File: src/nexus_orchestrator/synthesis_plane/work_item_classifier.py

Purpose
- Analyze a WorkItem and determine which model should be primary.
- Decision tree: first check if it's a reasoning/architecture task (→ GPT53),
  then check if it's trivial/small (→ SPARK), otherwise default to OPUS for coding.
- Pure-function classifier: deterministic, same input = same output.

Security
- No network calls, no secret access. Operates on in-memory WorkItem data only.
"""

from __future__ import annotations

import enum
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexus_orchestrator.domain.models import WorkItem


class ModelAffinity(enum.Enum):
    """Which model should be primary for a work item.

    Three-way routing:
    - GPT53: complex reasoning, architecture, multi-step planning (codex -m gpt-5.3-codex)
    - SPARK: quick/simple tasks, small fixes, docs (codex -m gpt-5.3-codex-spark)
    - OPUS: deep coding, large refactors, tests (claude --model claude-opus-4-6)
    """

    GPT53 = "gpt53"
    SPARK = "spark"
    OPUS = "opus"

    # Backward-compatible aliases (map to the closest new value)
    CODEX_FIRST = "gpt53"
    CLAUDE_FIRST = "opus"


# --- Scope file patterns ---
_CODE_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.py$", re.IGNORECASE),
    re.compile(r"\.ts$", re.IGNORECASE),
    re.compile(r"\.tsx$", re.IGNORECASE),
    re.compile(r"\.js$", re.IGNORECASE),
    re.compile(r"\.jsx$", re.IGNORECASE),
    re.compile(r"\.rs$", re.IGNORECASE),
    re.compile(r"\.go$", re.IGNORECASE),
)

_CONFIG_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.md$", re.IGNORECASE),
    re.compile(r"\.toml$", re.IGNORECASE),
    re.compile(r"\.yaml$", re.IGNORECASE),
    re.compile(r"\.yml$", re.IGNORECASE),
    re.compile(r"\.json$", re.IGNORECASE),
    re.compile(r"\.cfg$", re.IGNORECASE),
    re.compile(r"\.ini$", re.IGNORECASE),
    re.compile(r"Dockerfile$", re.IGNORECASE),
    re.compile(r"\.dockerfile$", re.IGNORECASE),
)

# --- Title/description keywords ---
# GPT53 keywords: tasks that need high-level reasoning/architecture
_GPT53_KEYWORDS: frozenset[str] = frozenset({
    "design", "architect", "analyze", "review", "plan", "strategy",
    "document", "spec", "schema", "config", "configure", "setup",
    "deploy", "infrastructure", "ci", "cd", "pipeline", "workflow",
    "migrate", "upgrade", "deprecate", "audit", "assess",
})

# SPARK keywords: trivial/small tasks
_SPARK_KEYWORDS: frozenset[str] = frozenset({
    "rename", "typo", "comment", "formatting", "lint", "cleanup",
    "trivial", "minor", "simple", "small", "tweak", "bump",
    "changelog", "readme", "license", "todo",
})

# OPUS keywords: heavy implementation/coding tasks
_OPUS_KEYWORDS: frozenset[str] = frozenset({
    "implement", "fix", "refactor", "bug", "test", "write",
    "code", "function", "method", "class", "module", "endpoint",
    "api", "handler", "parser", "serialize", "deserialize",
    "validate", "parse", "compile", "generate", "build",
})


def classify_work_item(work_item: WorkItem) -> ModelAffinity:
    """Classify a work item to determine primary model affinity.

    Uses a decision-tree approach instead of a single score:

    1. If task is architecture/reasoning-heavy → GPT53
       (config-only scope, OR strong architecture keywords, OR HIGH/CRITICAL risk
        with no code scope)
    2. If task is trivial/small → SPARK
       (single-file scope, spark keywords, low complexity, short description)
    3. Otherwise → OPUS  (default for implementation/coding tasks)

    This ensures coding tasks with many constraints still go to OPUS (the best
    coder), while architecture/design tasks go to GPT53 (the best reasoner).
    """
    scope = work_item.scope
    title = work_item.title
    description = work_item.description
    constraints = work_item.constraint_envelope.constraints

    # Classify scope
    code_files = 0
    config_files = 0
    for path in scope:
        if any(p.search(path) for p in _CODE_SCOPE_PATTERNS):
            code_files += 1
        elif any(p.search(path) for p in _CONFIG_SCOPE_PATTERNS):
            config_files += 1

    # Extract keywords from title + description
    text = f"{title} {description}".lower()
    words = set(re.findall(r"[a-z]+", text))
    gpt53_hits = len(words & _GPT53_KEYWORDS)
    spark_hits = len(words & _SPARK_KEYWORDS)
    opus_hits = len(words & _OPUS_KEYWORDS)

    # Get risk tier
    from nexus_orchestrator.domain.models import RiskTier
    is_high_risk = (
        isinstance(work_item.risk_tier, RiskTier)
        and work_item.risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL)
    )

    # --- Decision tree ---

    # Rule 1: Trivial/small → SPARK  (check first — quick tasks are cheap)
    # Single file scope with simple/spark keywords
    if len(scope) <= 1 and spark_hits > 0:
        return ModelAffinity.SPARK
    # Very short description, single file, few constraints, no strong signals
    if (len(scope) <= 1 and len(description.split()) < 30
            and len(constraints) <= 2 and gpt53_hits <= 1 and opus_hits == 0):
        return ModelAffinity.SPARK
    # Config-only, few constraints, no architecture keywords
    if config_files > 0 and code_files == 0 and gpt53_hits == 0 and len(constraints) <= 2:
        return ModelAffinity.SPARK

    # Rule 2: Architecture/reasoning → GPT53
    # Strong architecture keyword dominance (2+ hits, more than coding keywords)
    if gpt53_hits >= 2 and gpt53_hits > opus_hits + 1:
        return ModelAffinity.GPT53
    # Config-only scope with multiple architecture keywords
    if config_files > 0 and code_files == 0 and gpt53_hits >= 2:
        return ModelAffinity.GPT53
    # High/critical risk with NO code files (pure planning/design)
    if is_high_risk and code_files == 0:
        return ModelAffinity.GPT53

    # Rule 3: Default → OPUS (heavy coding with large memory)
    # Any task with code files, implementation keywords, or substantial scope
    return ModelAffinity.OPUS


__all__ = [
    "ModelAffinity",
    "classify_work_item",
]
