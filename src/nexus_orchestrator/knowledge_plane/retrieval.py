"""
nexus-orchestrator — deterministic context retrieval for context assembly

File: src/nexus_orchestrator/knowledge_plane/retrieval.py
Last updated: 2026-02-13

Purpose
- Retrieval API for context assembly: dependency-based, keyword-based, and optional semantic
  retrieval.

What should be included in this file
- Ranking policy: contracts first, then direct deps, then similar modules, then recent changes.
- Safety filters: exclude suspicious content that resembles prompt injection.

Functional requirements
- Must support token-budgeted context bundles.

Non-functional requirements
- Must be deterministic given the same index snapshot.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


class TokenEstimateFn(Protocol):
    """Callable protocol for token estimation hooks."""

    def __call__(self, text: str) -> int: ...


class RetrievalTier(IntEnum):
    """Deterministic retrieval priority tiers."""

    CONTRACTS = 0
    DIRECT_DEPENDENCIES = 1
    SIMILAR_MODULES = 2
    RECENT_CHANGES = 3


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One index candidate for retrieval."""

    path: str
    content: str
    is_contract: bool = False
    is_direct_dependency: bool = False
    similarity_score: float = 0.0
    recency_score: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_relative_path(self.path, field_name="path"))
        if not isinstance(self.content, str):
            raise TypeError("RetrievalCandidate.content must be a string")
        if not isinstance(self.is_contract, bool):
            raise TypeError("RetrievalCandidate.is_contract must be a boolean")
        if not isinstance(self.is_direct_dependency, bool):
            raise TypeError("RetrievalCandidate.is_direct_dependency must be a boolean")

        if isinstance(self.similarity_score, bool) or not isinstance(
            self.similarity_score, (int, float)
        ):
            raise TypeError("RetrievalCandidate.similarity_score must be a finite number")
        similarity = float(self.similarity_score)
        if not math.isfinite(similarity) or similarity < 0:
            raise ValueError("RetrievalCandidate.similarity_score must be finite and >= 0")
        object.__setattr__(self, "similarity_score", similarity)

        if isinstance(self.recency_score, bool) or not isinstance(self.recency_score, int):
            raise TypeError("RetrievalCandidate.recency_score must be an integer")
        if self.recency_score < 0:
            raise ValueError("RetrievalCandidate.recency_score must be >= 0")


@dataclass(frozen=True, slots=True)
class ContextDoc:
    """Context payload emitted by retrieval for provider requests."""

    path: str
    tier: RetrievalTier
    rank: int
    estimated_tokens: int
    source_sha256: str
    content_sha256: str
    inclusion_rationale: str
    content: str

    @property
    def name(self) -> str:
        """Compatibility alias for downstream context pack builders."""

        return PurePosixPath(self.path).name

    @property
    def doc_type(self) -> str:
        """Compatibility alias for downstream context pack builders."""

        return self.tier.name.lower()

    @property
    def content_hash(self) -> str:
        """Compatibility alias for downstream context pack builders."""

        return self.content_sha256

    @property
    def why_included(self) -> str:
        """Compatibility alias for downstream context pack builders."""

        return self.inclusion_rationale

    @property
    def metadata(self) -> dict[str, int | str]:
        """Compatibility metadata map used by context assemblers/providers."""

        return {
            "rank": self.rank,
            "estimated_tokens": self.estimated_tokens,
            "source_sha256": self.source_sha256,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True, slots=True)
class TruncationManifestEntry:
    """Deterministic token-budget accounting for one included context doc."""

    path: str
    source_sha256: str
    included_sha256: str
    original_tokens: int
    included_tokens: int
    omitted_tokens: int
    was_truncated: bool
    reason: str


@dataclass(frozen=True, slots=True)
class RetrievalBundle:
    """Token-budgeted retrieval output."""

    docs: tuple[ContextDoc, ...]
    truncation_manifest: tuple[TruncationManifestEntry, ...]
    token_budget: int
    used_tokens: int
    hygiene_excluded_paths: tuple[str, ...]

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.token_budget - self.used_tokens)


@dataclass(frozen=True, slots=True)
class TokenEstimator:
    """Deterministic token estimator with optional model/provider override."""

    estimate_fn: TokenEstimateFn | None = None

    def estimate(self, text: str) -> int:
        if self.estimate_fn is None:
            return estimate_tokens(text)
        estimated = self.estimate_fn(text)
        if isinstance(estimated, bool) or not isinstance(estimated, int):
            raise TypeError("token estimator override must return an integer")
        if estimated < 0:
            raise ValueError("token estimator override must return >= 0")
        return estimated


@dataclass(frozen=True, slots=True)
class _HygieneOutcome:
    content: str
    suspicious_line_numbers: tuple[int, ...]
    excluded: bool


_SANITIZED_LINE_MARKER: Final[str] = "[FILTERED_SUSPICIOUS_CONTENT]"
_HYGIENE_EXCLUDE_RATIO: Final[float] = 0.5
_HYGIENE_EXCLUDE_MIN_SUSPICIOUS_LINES: Final[int] = 3
_SUSPICIOUS_LINE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(
        r"(?i)\bignore\b.{0,48}\b(previous|prior|above)\b.{0,24}"
        r"\b(instruction|prompt|rule)s?\b"
    ),
    re.compile(r"(?i)\b(system|developer)\s+prompt\b"),
    re.compile(r"(?i)\byou are (chatgpt|an ai assistant|a large language model)\b"),
    re.compile(r"(?i)\b(exfiltrate|leak|steal)\b.{0,24}\b(secret|credential|token|password)s?\b"),
    re.compile(r"(?i)\b(run|execute)\b.{0,24}\b(shell|terminal|bash|command)s?\b"),
)


def classify_candidate_tier(candidate: RetrievalCandidate) -> RetrievalTier:
    """Classify one candidate using strict policy precedence."""

    if candidate.is_contract:
        return RetrievalTier.CONTRACTS
    if candidate.is_direct_dependency:
        return RetrievalTier.DIRECT_DEPENDENCIES
    if candidate.similarity_score > 0:
        return RetrievalTier.SIMILAR_MODULES
    return RetrievalTier.RECENT_CHANGES


def estimate_tokens(text: str) -> int:
    """
    Deterministic token estimator used for budgeting.

    The estimator is intentionally simple and stable:
    - 1 token per 4 UTF-8 codepoints (rounded up)
    - +1 token per newline
    """

    if not isinstance(text, str):
        raise TypeError("estimate_tokens expects a string")
    if not text:
        return 0

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    char_tokens = (len(normalized) + 3) // 4
    newline_tokens = normalized.count("\n")
    return char_tokens + newline_tokens


def rank_candidates(candidates: Sequence[RetrievalCandidate]) -> tuple[RetrievalCandidate, ...]:
    """Return a deterministic retrieval order with path-level deduplication."""

    best_by_path: dict[str, tuple[tuple[int, float, int, str, str], RetrievalCandidate]] = {}

    for candidate in candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise TypeError("rank_candidates expects RetrievalCandidate items")
        sort_key = _candidate_sort_key(candidate)
        current = best_by_path.get(candidate.path)
        if current is None or sort_key < current[0]:
            best_by_path[candidate.path] = (sort_key, candidate)

    ordered = sorted(best_by_path.values(), key=lambda item: item[0])
    return tuple(candidate for _, candidate in ordered)


def retrieve_context_docs(
    candidates: Sequence[RetrievalCandidate],
    *,
    max_tokens: int,
    token_estimator: TokenEstimator | TokenEstimateFn | None = None,
) -> RetrievalBundle:
    """Build a deterministic token-budgeted context bundle."""

    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer")
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    estimator = _resolve_token_estimator(token_estimator)

    ranked_candidates = rank_candidates(candidates)

    docs: list[ContextDoc] = []
    manifest: list[TruncationManifestEntry] = []
    hygiene_excluded_paths: list[str] = []
    excluded_seen: set[str] = set()
    used_tokens = 0

    for candidate in ranked_candidates:
        remaining = max_tokens - used_tokens
        if remaining <= 0:
            break

        hygiene = _apply_hygiene(candidate.content)
        if hygiene.excluded:
            if candidate.path not in excluded_seen:
                hygiene_excluded_paths.append(candidate.path)
                excluded_seen.add(candidate.path)
            continue

        if not hygiene.content:
            continue

        source_sha = _sha256_hex(candidate.content)
        original_tokens = estimator(hygiene.content)

        included_content, included_tokens, was_truncated = _truncate_to_token_budget(
            hygiene.content,
            max_tokens=remaining,
            token_estimator=estimator,
        )
        if included_tokens <= 0:
            continue

        included_sha = _sha256_hex(included_content)
        tier = classify_candidate_tier(candidate)
        rationale = _build_inclusion_rationale(
            candidate,
            tier=tier,
            hygiene=hygiene,
            was_truncated=was_truncated,
        )

        docs.append(
            ContextDoc(
                path=candidate.path,
                tier=tier,
                rank=len(docs) + 1,
                estimated_tokens=included_tokens,
                source_sha256=source_sha,
                content_sha256=included_sha,
                inclusion_rationale=rationale,
                content=included_content,
            )
        )
        manifest.append(
            TruncationManifestEntry(
                path=candidate.path,
                source_sha256=source_sha,
                included_sha256=included_sha,
                original_tokens=original_tokens,
                included_tokens=included_tokens,
                omitted_tokens=max(0, original_tokens - included_tokens),
                was_truncated=was_truncated,
                reason="token_budget_cap" if was_truncated else "within_budget",
            )
        )
        used_tokens += included_tokens

    return RetrievalBundle(
        docs=tuple(docs),
        truncation_manifest=tuple(manifest),
        token_budget=max_tokens,
        used_tokens=used_tokens,
        hygiene_excluded_paths=tuple(hygiene_excluded_paths),
    )


def _candidate_sort_key(candidate: RetrievalCandidate) -> tuple[int, float, int, str, str]:
    tier = classify_candidate_tier(candidate)
    similarity_key = -candidate.similarity_score if tier is RetrievalTier.SIMILAR_MODULES else 0.0
    recency_key = -candidate.recency_score if tier is RetrievalTier.RECENT_CHANGES else 0
    return (
        int(tier),
        similarity_key,
        recency_key,
        candidate.path,
        _sha256_hex(candidate.content),
    )


def _apply_hygiene(text: str) -> _HygieneOutcome:
    if not text:
        return _HygieneOutcome(content=text, suspicious_line_numbers=(), excluded=False)

    lines = text.splitlines(keepends=True)
    suspicious_line_numbers: list[int] = []
    sanitized_lines = list(lines)

    for line_number, line in enumerate(lines, start=1):
        if _looks_suspicious(line):
            suspicious_line_numbers.append(line_number)
            sanitized_lines[line_number - 1] = _sanitize_line(line)

    if not suspicious_line_numbers:
        return _HygieneOutcome(content=text, suspicious_line_numbers=(), excluded=False)

    non_empty_lines = sum(1 for line in lines if line.strip())
    suspicious_count = len(suspicious_line_numbers)
    exclude_for_density = suspicious_count >= _HYGIENE_EXCLUDE_MIN_SUSPICIOUS_LINES
    if non_empty_lines > 0:
        exclude_for_density = exclude_for_density or (
            (suspicious_count / non_empty_lines) >= _HYGIENE_EXCLUDE_RATIO
        )

    if exclude_for_density:
        return _HygieneOutcome(
            content="",
            suspicious_line_numbers=tuple(suspicious_line_numbers),
            excluded=True,
        )

    return _HygieneOutcome(
        content="".join(sanitized_lines),
        suspicious_line_numbers=tuple(suspicious_line_numbers),
        excluded=False,
    )


def _looks_suspicious(line: str) -> bool:
    return any(pattern.search(line) for pattern in _SUSPICIOUS_LINE_PATTERNS)


def _sanitize_line(line: str) -> str:
    if line.endswith("\r\n"):
        return f"{_SANITIZED_LINE_MARKER}\r\n"
    if line.endswith("\n"):
        return f"{_SANITIZED_LINE_MARKER}\n"
    if line.endswith("\r"):
        return f"{_SANITIZED_LINE_MARKER}\r"
    return _SANITIZED_LINE_MARKER


def _truncate_to_token_budget(
    text: str,
    *,
    max_tokens: int,
    token_estimator: TokenEstimateFn,
) -> tuple[str, int, bool]:
    if max_tokens <= 0:
        return "", 0, bool(text)

    full_tokens = token_estimator(text)
    if full_tokens <= max_tokens:
        return text, full_tokens, False

    low = 0
    high = len(text)
    best = 0

    # Deterministic binary search for the longest prefix that fits the budget.
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        tokens = token_estimator(candidate)
        if tokens <= max_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    included = text[:best]
    return included, token_estimator(included), True


def _resolve_token_estimator(
    token_estimator: TokenEstimator | TokenEstimateFn | None,
) -> TokenEstimateFn:
    if token_estimator is None:
        return estimate_tokens
    if isinstance(token_estimator, TokenEstimator):
        return token_estimator.estimate
    return TokenEstimator(estimate_fn=token_estimator).estimate


def _build_inclusion_rationale(
    candidate: RetrievalCandidate,
    *,
    tier: RetrievalTier,
    hygiene: _HygieneOutcome,
    was_truncated: bool,
) -> str:
    if tier is RetrievalTier.CONTRACTS:
        parts = ["Tier=contracts (highest priority)."]
    elif tier is RetrievalTier.DIRECT_DEPENDENCIES:
        parts = ["Tier=direct_dependencies."]
    elif tier is RetrievalTier.SIMILAR_MODULES:
        parts = [f"Tier=similar_modules (similarity={candidate.similarity_score:.3f})."]
    else:
        parts = [f"Tier=recent_changes (recency={candidate.recency_score})."]

    if hygiene.suspicious_line_numbers and not hygiene.excluded:
        parts.append(f"Sanitized {len(hygiene.suspicious_line_numbers)} suspicious line(s).")
    if was_truncated:
        parts.append("Truncated to fit token budget.")

    return " ".join(parts)


def _normalize_relative_path(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"RetrievalCandidate.{field_name} must be a string")

    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise ValueError(f"RetrievalCandidate.{field_name} must not be empty")

    pure = PurePosixPath(normalized)
    if pure.is_absolute():
        raise ValueError(f"RetrievalCandidate.{field_name} must be a relative path")
    if any(part == ".." for part in pure.parts):
        raise ValueError(f"RetrievalCandidate.{field_name} must not contain traversal")

    canonical = pure.as_posix()
    if canonical in {"", "."}:
        raise ValueError(f"RetrievalCandidate.{field_name} must point to a file")
    return canonical


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "ContextDoc",
    "RetrievalBundle",
    "RetrievalCandidate",
    "RetrievalTier",
    "TokenEstimator",
    "TruncationManifestEntry",
    "classify_candidate_tier",
    "estimate_tokens",
    "rank_candidates",
    "retrieve_context_docs",
]
