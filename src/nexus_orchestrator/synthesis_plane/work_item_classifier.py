"""Work item classifier for intelligent model delegation.

File: src/nexus_orchestrator/synthesis_plane/work_item_classifier.py

Purpose
- Analyze a WorkItem and determine which model should be primary (codex vs claude).
- Codex is better at reasoning/architecture tasks; Claude is better at coding logic.
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


# --- Signal weights ---
_WEIGHT_SCOPE = 3
_WEIGHT_KEYWORDS = 2
_WEIGHT_CONSTRAINTS = 1
_WEIGHT_RISK = 1

# --- Scope file patterns ---
_CLAUDE_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"test[s_]?.*\.py$", re.IGNORECASE),
    re.compile(r"\.py$", re.IGNORECASE),
    re.compile(r"\.ts$", re.IGNORECASE),
    re.compile(r"\.tsx$", re.IGNORECASE),
    re.compile(r"\.js$", re.IGNORECASE),
    re.compile(r"\.jsx$", re.IGNORECASE),
    re.compile(r"\.rs$", re.IGNORECASE),
    re.compile(r"\.go$", re.IGNORECASE),
)

_CODEX_SCOPE_PATTERNS: tuple[re.Pattern[str], ...] = (
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
_CLAUDE_KEYWORDS: frozenset[str] = frozenset({
    "implement", "fix", "refactor", "bug", "test", "write",
    "code", "function", "method", "class", "module", "endpoint",
    "api", "handler", "parser", "serialize", "deserialize",
    "validate", "parse", "compile", "generate", "build",
})

_CODEX_KEYWORDS: frozenset[str] = frozenset({
    "design", "architect", "analyze", "review", "plan", "strategy",
    "document", "spec", "schema", "config", "configure", "setup",
    "deploy", "infrastructure", "ci", "cd", "pipeline", "workflow",
    "migrate", "upgrade", "deprecate", "audit", "assess",
})

# --- Constraint thresholds ---
_HIGH_CONSTRAINT_THRESHOLD = 4


def classify_work_item(work_item: WorkItem) -> ModelAffinity:
    """Classify a work item to determine primary model affinity.

    Scores 4 signals with weighted contributions:
    - Scope file patterns (weight 3): source code → OPUS, config/docs → GPT53
    - Title/desc keywords (weight 2): implementation → OPUS, architecture → GPT53
    - Constraint count (weight 1): few → OPUS, many → GPT53
    - Risk tier (weight 1): LOW → OPUS, HIGH/CRITICAL → GPT53

    Three-way routing with score bands:
    - score < -2  → GPT53 (complex reasoning/architecture)
    - score > 2   → OPUS  (deep coding, large context)
    - otherwise   → SPARK (quick/simple tasks)
    """
    # Positive score = OPUS (deep coding), negative = GPT53 (reasoning)
    score = 0

    # Signal 1: Scope file patterns
    score += _score_scope(work_item.scope) * _WEIGHT_SCOPE

    # Signal 2: Title/description keywords
    score += _score_keywords(work_item.title, work_item.description) * _WEIGHT_KEYWORDS

    # Signal 3: Constraint count
    score += _score_constraints(work_item.constraint_envelope.constraints) * _WEIGHT_CONSTRAINTS

    # Signal 4: Risk tier
    score += _score_risk(work_item.risk_tier) * _WEIGHT_RISK

    # Three-way routing
    if score < -2:
        return ModelAffinity.GPT53
    if score > 2:
        return ModelAffinity.OPUS
    return ModelAffinity.SPARK


def _score_scope(scope: tuple[str, ...]) -> int:
    """Score scope files: +1 for Claude patterns, -1 for Codex patterns."""
    claude_count = 0
    codex_count = 0

    for path in scope:
        for pattern in _CLAUDE_SCOPE_PATTERNS:
            if pattern.search(path):
                claude_count += 1
                break
        else:
            for pattern in _CODEX_SCOPE_PATTERNS:
                if pattern.search(path):
                    codex_count += 1
                    break

    if claude_count > codex_count:
        return 1
    if codex_count > claude_count:
        return -1
    return 0


def _score_keywords(title: str, description: str) -> int:
    """Score title/description keywords: +1 for Claude, -1 for Codex."""
    text = f"{title} {description}".lower()
    words = set(re.findall(r"[a-z]+", text))

    claude_hits = len(words & _CLAUDE_KEYWORDS)
    codex_hits = len(words & _CODEX_KEYWORDS)

    if claude_hits > codex_hits:
        return 1
    if codex_hits > claude_hits:
        return -1
    return 0


def _score_constraints(constraints: tuple[object, ...]) -> int:
    """Score constraint count: few (0-1) → +1 Claude, many (4+) → -1 Codex."""
    count = len(constraints)
    if count <= 1:
        return 1
    if count >= _HIGH_CONSTRAINT_THRESHOLD:
        return -1
    return 0


def _score_risk(risk_tier: object) -> int:
    """Score risk tier: LOW → +1 Claude, HIGH/CRITICAL → -1 Codex."""
    # Import here to avoid circular import at module level
    from nexus_orchestrator.domain.models import RiskTier

    if isinstance(risk_tier, RiskTier):
        if risk_tier == RiskTier.LOW:
            return 1
        if risk_tier in (RiskTier.HIGH, RiskTier.CRITICAL):
            return -1
    return 0


__all__ = [
    "ModelAffinity",
    "classify_work_item",
]
