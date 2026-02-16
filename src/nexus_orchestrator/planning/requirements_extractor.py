"""
Multi-stage requirements extraction pipeline.

Accepts arbitrary markdown design documents and extracts requirements using
a cascading strategy:

1. Explicit section parsing (deterministic) — looks for known headings
2. Heuristic extraction (deterministic) — scans for RFC-2119 keywords
3. LLM semantic extractor (preferred fallback) — calls the repo's LLM backend
4. Graceful fallback — infers minimal requirements from doc summary
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

RequirementType = Literal[
    "functional", "nonfunctional", "constraint", "assumption", "open_question"
]


@dataclass(frozen=True, slots=True)
class ExtractedRequirement:
    """A single requirement extracted from a design document."""

    id: str
    text: str
    type: RequirementType = "functional"
    priority: str | None = None
    acceptance_criteria: tuple[str, ...] = ()
    source: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Outcome of the multi-stage extraction pipeline."""

    requirements: tuple[ExtractedRequirement, ...]
    confidence: float  # 0–1
    warnings: tuple[str, ...] = ()
    used_strategy: str = "explicit_sections"


# ---------------------------------------------------------------------------
# LLM backend protocol (for dependency injection / mocking)
# ---------------------------------------------------------------------------


class LLMExtractor(Protocol):
    """Callable that takes markdown text and returns JSON requirements."""

    def __call__(self, markdown: str, file_path: str) -> list[dict[str, object]]:
        ...


# ---------------------------------------------------------------------------
# Stage 1: Explicit section parsing
# ---------------------------------------------------------------------------

_REQUIREMENT_HEADINGS: set[str] = {
    "requirements",
    "functional requirements",
    "non-functional requirements",
    "non functional requirements",
    "nonfunctional requirements",
    "goals",
    "objectives",
    "scope",
    "acceptance criteria",
    "constraints",
    "user stories",
    "features",
    "capabilities",
    "deliverables",
    "success criteria",
    "key outcomes",
    "key requirements",
    "system requirements",
    "design goals",
    "design constraints",
    "use cases",
}

_ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_UNDERLINE_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_EXPLICIT_REQ_ID_RE = re.compile(
    r"^(?P<id>[A-Za-z][A-Za-z0-9_-]*\d{3,})\s*:\s*(?P<text>.+)$"
)
# Numbered section heading like "4) System overview" or "7.2 Search orchestration"
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]\s+)(.+)$"
)


def _parse_sections(lines: list[str]) -> dict[str, list[tuple[int, str]]]:
    """Extract content grouped by heading, returning {normalized_heading: [(line_no, text)]}."""
    sections: dict[str, list[tuple[int, str]]] = {}
    current_heading = ""
    in_fence = False

    for idx, raw_line in enumerate(lines, start=1):
        # Track fenced code blocks
        if _FENCE_RE.match(raw_line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # ATX heading
        m = _ATX_HEADING_RE.match(raw_line)
        if m:
            heading_text = m.group(2).strip()
            normalized = _normalize_heading(heading_text)
            if normalized in _REQUIREMENT_HEADINGS:
                current_heading = normalized
                if current_heading not in sections:
                    sections[current_heading] = []
            else:
                current_heading = ""
            continue

        # Setext heading (check next line)
        # (handled by checking if current line becomes heading via underline on next)
        # We handle this by looking ahead, but for simplicity we rely on ATX + numbered

        # Numbered section heading as plain text (e.g., "3) Constraints and compute assumptions")
        nm = _NUMBERED_HEADING_RE.match(raw_line.strip())
        if nm and not _LIST_ITEM_RE.match(raw_line):
            heading_text = nm.group(1).strip()
            normalized = _normalize_heading(heading_text)
            if normalized in _REQUIREMENT_HEADINGS:
                current_heading = normalized
                if current_heading not in sections:
                    sections[current_heading] = []
                continue

        if current_heading and raw_line.strip():
            sections[current_heading].append((idx, raw_line.strip()))

    return sections


def _normalize_heading(text: str) -> str:
    """Normalize heading text for matching against known requirement headings."""
    # Strip markdown formatting, numbering, and extra whitespace
    cleaned = re.sub(r"^\d+(?:\.\d+)*[.)]\s*", "", text)  # Remove leading numbers
    cleaned = re.sub(r"[*_`#]+", "", cleaned)  # Remove markdown formatting
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    # Remove trailing words like "section"
    cleaned = re.sub(r"\s+section$", "", cleaned)
    return cleaned


def _extract_from_sections(
    sections: dict[str, list[tuple[int, str]]], file_path: str
) -> list[ExtractedRequirement]:
    """Extract requirements from identified sections."""
    requirements: list[ExtractedRequirement] = []
    seen_ids: set[str] = set()

    for heading, items in sections.items():
        req_type = _heading_to_type(heading)
        for line_no, text in items:
            # Try to extract list items
            list_match = _LIST_ITEM_RE.match(text)
            item_text = list_match.group(1).strip() if list_match else text

            if not item_text or len(item_text) < 5:
                continue

            # Check for explicit IDs (e.g., "REQ-001: ...")
            id_match = _EXPLICIT_REQ_ID_RE.match(item_text)
            if id_match:
                req_id = id_match.group("id")
                req_text = id_match.group("text").strip()
            else:
                req_id = _generate_id(item_text, req_type, len(requirements) + 1)
                req_text = item_text

            if req_id in seen_ids:
                continue
            seen_ids.add(req_id)

            requirements.append(
                ExtractedRequirement(
                    id=req_id,
                    text=req_text,
                    type=req_type,
                    source={"file": file_path, "line": line_no, "heading": heading},
                )
            )

    return requirements


def _heading_to_type(heading: str) -> RequirementType:
    """Map a heading to a requirement type."""
    if heading in {
        "non-functional requirements",
        "non functional requirements",
        "nonfunctional requirements",
    }:
        return "nonfunctional"
    if heading in {"constraints", "design constraints"}:
        return "constraint"
    if heading in {"acceptance criteria", "success criteria"}:
        return "functional"
    return "functional"


def _generate_id(text: str, req_type: RequirementType, seq: int) -> str:
    """Generate a stable ID from text content."""
    prefix_map: dict[RequirementType, str] = {
        "functional": "FREQ",
        "nonfunctional": "NFREQ",
        "constraint": "CON",
        "assumption": "ASSM",
        "open_question": "OQ",
    }
    prefix = prefix_map.get(req_type, "REQ")
    # Use hash of text for stability across runs
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:6].upper()
    return f"{prefix}-{seq:04d}-{text_hash}"


# ---------------------------------------------------------------------------
# Stage 2: Heuristic extraction (RFC-2119 keyword scanning)
# ---------------------------------------------------------------------------

_RFC2119_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:must|shall|will|is required to|needs to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:should|recommended|ought to)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:supports?|ensures?|provides?|enables?|handles?|manages?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:capable of|responsible for|designed to|intended to)\b",
        re.IGNORECASE,
    ),
]

_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*$"),
    re.compile(r"^[-=*]{3,}"),  # horizontal rules
    re.compile(r"^\s*```"),  # code fences
    re.compile(r"^\s*<!--"),  # HTML comments
    re.compile(r"^\s*\|"),  # table rows
]


def _is_noise_line(line: str) -> bool:
    return any(p.match(line) for p in _NOISE_PATTERNS)


def _extract_heuristic(
    lines: list[str], file_path: str, existing_ids: set[str]
) -> list[ExtractedRequirement]:
    """Scan entire doc for requirement-like sentences using RFC-2119 keywords."""
    requirements: list[ExtractedRequirement] = []
    in_fence = False
    seq = len(existing_ids) + 1

    for idx, raw_line in enumerate(lines, start=1):
        if _FENCE_RE.match(raw_line):
            in_fence = not in_fence
            continue
        if in_fence or _is_noise_line(raw_line):
            continue

        stripped = raw_line.strip()
        if len(stripped) < 15:
            continue

        # Skip headings
        if _ATX_HEADING_RE.match(raw_line):
            continue

        # Check for RFC-2119 keywords
        match_strength = sum(1 for p in _RFC2119_PATTERNS if p.search(stripped))
        if match_strength == 0:
            continue

        # Extract the meaningful text
        list_match = _LIST_ITEM_RE.match(raw_line)
        text = list_match.group(1).strip() if list_match else stripped

        # Determine type heuristically
        req_type: RequirementType = "functional"
        lower = text.lower()
        if any(
            kw in lower
            for kw in (
                "performance",
                "latency",
                "throughput",
                "scalab",
                "reliab",
                "availab",
                "security",
                "encrypt",
            )
        ):
            req_type = "nonfunctional"
        elif any(kw in lower for kw in ("constraint", "limit", "restrict", "bound")):
            req_type = "constraint"

        # Check for explicit ID already in text
        id_match = _EXPLICIT_REQ_ID_RE.match(text)
        if id_match:
            req_id = id_match.group("id")
            text = id_match.group("text").strip()
        else:
            req_id = _generate_id(text, req_type, seq)

        if req_id in existing_ids:
            continue

        existing_ids.add(req_id)
        seq += 1

        # Assign priority based on keyword strength
        priority = "high" if match_strength >= 2 else "medium"

        requirements.append(
            ExtractedRequirement(
                id=req_id,
                text=text,
                type=req_type,
                priority=priority,
                source={"file": file_path, "line": idx, "strategy": "heuristic"},
            )
        )

    return requirements


# ---------------------------------------------------------------------------
# Stage 3: LLM semantic extractor
# ---------------------------------------------------------------------------

LLM_EXTRACTION_PROMPT = """\
You are a requirements engineering expert. Analyze the following design document \
and extract ALL requirements, constraints, assumptions, and open questions.

For each requirement, output a JSON object with these fields:
- "id": a short unique identifier (e.g., "REQ-001", "NFR-001", "CON-001")
- "text": the requirement statement (clear, actionable)
- "type": one of "functional", "nonfunctional", "constraint", "assumption", "open_question"
- "priority": one of "high", "medium", "low" (best effort)
- "acceptance_criteria": list of testable criteria (best effort, can be empty)

Return a JSON array of these objects. Output ONLY valid JSON, no markdown fences.

---
DOCUMENT:
{document}
"""


def _parse_llm_response(raw: str) -> list[dict[str, object]]:
    """Parse and validate LLM JSON response, handling common formatting issues."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()

    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        if isinstance(parsed, dict) and "requirements" in parsed:
            parsed = parsed["requirements"]
        else:
            raise ValueError("LLM response must be a JSON array")

    validated: list[dict[str, object]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if "text" not in item and "statement" in item:
            item["text"] = item["statement"]
        if "text" not in item:
            continue
        validated.append(item)

    return validated


def _llm_results_to_requirements(
    raw_items: list[dict[str, object]], file_path: str
) -> list[ExtractedRequirement]:
    """Convert validated LLM output into ExtractedRequirement objects."""
    requirements: list[ExtractedRequirement] = []
    seen_ids: set[str] = set()

    for idx, item in enumerate(raw_items, start=1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        req_id = str(item.get("id", "")).strip()
        if not req_id or req_id in seen_ids:
            req_type_raw = str(item.get("type", "functional"))
            req_type = _validate_type(req_type_raw)
            req_id = _generate_id(text, req_type, idx)

        if req_id in seen_ids:
            continue
        seen_ids.add(req_id)

        req_type_raw = str(item.get("type", "functional"))
        req_type = _validate_type(req_type_raw)

        priority = str(item.get("priority", "")) or None

        ac_raw = item.get("acceptance_criteria", [])
        if isinstance(ac_raw, list):
            acceptance = tuple(str(c).strip() for c in ac_raw if str(c).strip())
        else:
            acceptance = ()

        requirements.append(
            ExtractedRequirement(
                id=req_id,
                text=text,
                type=req_type,
                priority=priority,
                acceptance_criteria=acceptance,
                source={"file": file_path, "strategy": "llm"},
            )
        )

    return requirements


def _validate_type(raw: str) -> RequirementType:
    """Validate and normalize requirement type."""
    valid: set[RequirementType] = {
        "functional",
        "nonfunctional",
        "constraint",
        "assumption",
        "open_question",
    }
    normalized = raw.strip().lower().replace("-", "").replace(" ", "")
    # Map common variations
    mapping: dict[str, RequirementType] = {
        "functional": "functional",
        "nonfunctional": "nonfunctional",
        "nfr": "nonfunctional",
        "constraint": "constraint",
        "assumption": "assumption",
        "openquestion": "open_question",
        "open_question": "open_question",
    }
    result = mapping.get(normalized)
    if result is not None:
        return result
    if normalized in valid:
        return normalized  # type: ignore[return-value]
    return "functional"


# ---------------------------------------------------------------------------
# Stage 4: Graceful fallback (deterministic)
# ---------------------------------------------------------------------------


def _fallback_from_summary(
    lines: list[str], file_path: str
) -> list[ExtractedRequirement]:
    """Last-resort extraction: infer high-level requirements from document structure."""
    requirements: list[ExtractedRequirement] = []
    headings: list[str] = []

    # Collect all headings to understand document structure
    for _idx, line in enumerate(lines, start=1):
        m = _ATX_HEADING_RE.match(line)
        if m:
            headings.append(m.group(2).strip())

    # If we have headings, create investigation requirements per major section
    if headings:
        for i, heading in enumerate(headings[:10], start=1):
            req_id = f"INFERRED-{i:04d}"
            requirements.append(
                ExtractedRequirement(
                    id=req_id,
                    text=f"Implement capabilities described in: {heading}",
                    type="functional",
                    priority="medium",
                    source={"file": file_path, "strategy": "fallback_summary"},
                )
            )
    else:
        # Extract first meaningful paragraph as a single investigation item
        first_paragraph: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped and first_paragraph:
                break
            if stripped and not _ATX_HEADING_RE.match(line):
                first_paragraph.append(stripped)

        summary = " ".join(first_paragraph)[:200] if first_paragraph else "Design document"
        requirements.append(
            ExtractedRequirement(
                id="INFERRED-0001",
                text=f"Investigate and implement: {summary}",
                type="functional",
                priority="medium",
                source={"file": file_path, "strategy": "fallback_summary"},
            )
        )

    return requirements


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

# Minimum number of requirements before we escalate to the next strategy
_MIN_CONFIDENCE_THRESHOLD = 0.5
_MIN_REQUIREMENTS_FOR_EXPLICIT = 3


def extract_requirements(
    file_path: Path,
    *,
    llm_extractor: LLMExtractor | None = None,
    min_requirements: int = _MIN_REQUIREMENTS_FOR_EXPLICIT,
    confidence_threshold: float = _MIN_CONFIDENCE_THRESHOLD,
) -> ExtractionResult:
    """
    Multi-stage requirements extraction pipeline.

    Cascades through strategies until sufficient requirements are found:
    1. Explicit section parsing
    2. Heuristic RFC-2119 keyword extraction
    3. LLM semantic extraction (if backend available)
    4. Graceful fallback from document structure

    Parameters
    ----------
    file_path:
        Path to the markdown design document.
    llm_extractor:
        Optional callable for LLM-based extraction. When None, stage 3 is skipped.
    min_requirements:
        Minimum number of requirements before the pipeline considers a stage successful.
    confidence_threshold:
        Minimum confidence (0–1) before escalating to the next strategy.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    path_str = str(file_path)
    warnings: list[str] = []
    all_requirements: list[ExtractedRequirement] = []

    # --- Stage 1: Explicit section parsing ---
    sections = _parse_sections(lines)
    if sections:
        stage1_reqs = _extract_from_sections(sections, path_str)
        all_requirements.extend(stage1_reqs)

    if len(all_requirements) >= min_requirements:
        confidence = min(1.0, len(all_requirements) / (min_requirements * 2))
        return ExtractionResult(
            requirements=tuple(all_requirements),
            confidence=max(confidence, 0.7),
            warnings=tuple(warnings),
            used_strategy="explicit_sections",
        )

    # --- Stage 2: Heuristic extraction ---
    existing_ids = {r.id for r in all_requirements}
    stage2_reqs = _extract_heuristic(lines, path_str, existing_ids)
    all_requirements.extend(stage2_reqs)

    if len(all_requirements) >= min_requirements:
        confidence = min(1.0, len(all_requirements) / (min_requirements * 2))
        strategy = (
            "explicit_sections+heuristics"
            if any(r.source.get("strategy") != "heuristic" for r in all_requirements)
            else "heuristics"
        )
        return ExtractionResult(
            requirements=tuple(all_requirements),
            confidence=max(confidence, 0.5),
            warnings=tuple(warnings),
            used_strategy=strategy,
        )

    # --- Stage 3: LLM semantic extraction ---
    if llm_extractor is not None:
        try:
            raw_items = llm_extractor(text, path_str)
            stage3_reqs = _llm_results_to_requirements(raw_items, path_str)
            # Deduplicate against existing
            existing_ids = {r.id for r in all_requirements}
            for req in stage3_reqs:
                if req.id not in existing_ids:
                    all_requirements.append(req)
                    existing_ids.add(req.id)

            if all_requirements:
                confidence = min(1.0, len(all_requirements) / (min_requirements * 2))
                return ExtractionResult(
                    requirements=tuple(all_requirements),
                    confidence=max(confidence, 0.6),
                    warnings=tuple(warnings),
                    used_strategy="llm",
                )
        except Exception as exc:
            warnings.append(
                f"LLM extraction failed ({type(exc).__name__}: {exc}); "
                f"falling back to document structure analysis"
            )

    elif len(all_requirements) < min_requirements:
        warnings.append(
            "LLM extractor unavailable; using document structure analysis for "
            "requirement inference. Results may be less precise."
        )

    # --- Stage 4: Graceful fallback ---
    if len(all_requirements) < min_requirements:
        existing_ids = {r.id for r in all_requirements}
        fallback_reqs = _fallback_from_summary(lines, path_str)
        for req in fallback_reqs:
            if req.id not in existing_ids:
                all_requirements.append(req)
                existing_ids.add(req.id)

    if not all_requirements:
        # Absolute last resort: single investigation item
        all_requirements.append(
            ExtractedRequirement(
                id="INFERRED-0001",
                text="Investigate and clarify requirements from design document",
                type="open_question",
                priority="high",
                source={"file": path_str, "strategy": "fallback_empty"},
            )
        )
        warnings.append(
            "No requirements could be extracted; created investigation placeholder"
        )

    strategy_parts: list[str] = []
    strategies_used = {
        r.source.get("strategy", "explicit") for r in all_requirements if isinstance(r.source, dict)
    }
    if strategies_used - {"heuristic", "llm", "fallback_summary", "fallback_empty"}:
        strategy_parts.append("explicit_sections")
    if "heuristic" in strategies_used:
        strategy_parts.append("heuristics")
    if "llm" in strategies_used:
        strategy_parts.append("llm")
    if strategies_used & {"fallback_summary", "fallback_empty"}:
        strategy_parts.append("fallback")

    return ExtractionResult(
        requirements=tuple(all_requirements),
        confidence=0.3 if "fallback" in strategy_parts else 0.5,
        warnings=tuple(warnings),
        used_strategy="+".join(strategy_parts) if strategy_parts else "fallback",
    )


__all__ = [
    "ExtractedRequirement",
    "ExtractionResult",
    "LLMExtractor",
    "extract_requirements",
]
