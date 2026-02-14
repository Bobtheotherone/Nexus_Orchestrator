"""
nexus-orchestrator — repository blueprint engine

File: src/nexus_orchestrator/repo_blueprint.py
Last updated: 2026-02-11

Purpose
- Build a deterministic, machine-readable repository blueprint used as a quality contract.

What should be included in this file
- Tracked-file discovery with git-first and deterministic filesystem fallback.
- Header extraction/parser utilities for Python, Markdown/HTML comments, and comment-prefixed files.
- Phase/plane inference, cross-reference indexing, placeholder scanning, and validation checks.
- CLI helpers to print summaries, emit JSON, and write generated docs artifacts.

Functional requirements
- Must run offline and produce deterministic output for unchanged repository state.
- Must expose validation checks that are hard to cheat across independent sources.
- Must support WSL-friendly execution (`python scripts/repo_audit.py ...`).

Non-functional requirements
- Keep runtime low for local iteration (<20s typical on this repository).
- Never require provider SDKs or network access.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PathMap = dict[str, Any]

KNOWN_TOP_LEVEL_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "constraints/",
    "docker/",
    "docs/",
    "profiles/",
    "samples/",
    "scripts/",
    "src/",
    "tests/",
    "tools/",
)

ROOT_FILES_EXCLUDED: set[str] = {
    ".python-version",
}

KNOWN_ROOT_REFERENCES: set[str] = {
    ".editorconfig",
    ".env.example",
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "design_document.md",
    "orchestrator.toml",
    "pyproject.toml",
}

EPHEMERAL_DIR_SEGMENTS: set[str] = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "evidence",
    "state",
    "workspaces",
}

CATEGORY_ORDER: tuple[str, ...] = (
    "src",
    "tests",
    "docs",
    "constraints",
    "tools",
    "scripts",
    "workflows",
    "samples",
    "profiles",
    "docker",
    "root",
    "other",
)

SECTION_ALIASES: tuple[tuple[str, str], ...] = (
    ("purpose", "file_purpose"),
    ("what should be included in this file", "must_include_items"),
    ("what should be included in this workflow", "must_include_items"),
    ("what should be included", "must_include_items"),
    ("what this test file should cover", "must_include_items"),
    ("expected cli usage", "must_include_items"),
    ("key interfaces / contracts to define here", "must_include_items"),
    ("functional requirements", "functional_requirements"),
    ("non-functional requirements", "nonfunctional_requirements"),
    ("failure modes / edge cases to handle", "invariants"),
    ("testing guidance", "invariants"),
    ("invariants across all phases", "invariants"),
    ("invariants", "invariants"),
    ("behavior", "functional_requirements"),
)

PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "skeleton_header",
        re.compile(r"\b(module|test|documentation|script)\s+skeleton\b", re.IGNORECASE),
    ),
    ("todo_fixme", re.compile(r"(?m)^\s*(?:#|//|\*+)\s*(?:TODO|FIXME)\b")),
    ("not_implemented", re.compile(r"raise\s+NotImplementedError")),
)

PATH_TOKEN_PATTERN = re.compile(
    r"(?:\.github/workflows/|constraints/|docker/|docs/|profiles/|samples/|scripts/|src/|"
    r"tests/|tools/|state/|evidence/|workspaces/)[A-Za-z0-9_./-]*"
    r"|(?:CHANGELOG\.md|CONTRIBUTING\.md|LICENSE|Makefile|README\.md|SECURITY\.md|"
    r"design_document\.md|orchestrator\.toml|pyproject\.toml)"
)

DOC_REFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"`([^`]+)`"),
    re.compile(r"\[[^\]]+\]\(([^)]+)\)"),
)

PLANNED_MARKERS: tuple[str, ...] = (
    "future",
    "planned",
    "placeholder",
    "todo",
    "tbd",
)

PLANE_MAP: dict[str, str] = {
    "control_plane": "control",
    "synthesis_plane": "synthesis",
    "verification_plane": "verification",
    "integration_plane": "integration",
    "knowledge_plane": "knowledge",
    "sandbox": "sandbox",
    "domain": "domain",
    "utils": "utils",
    "ui": "ui",
}


@dataclass(frozen=True)
class HeaderExtraction:
    """Structured header extraction result."""

    header_present: bool
    header_style: str | None
    raw_header: str
    clean_header: str
    declared_sections: tuple[str, ...]
    file_purpose: tuple[str, ...]
    must_include_items: tuple[str, ...]
    functional_requirements: tuple[str, ...]
    nonfunctional_requirements: tuple[str, ...]
    invariants: tuple[str, ...]
    referenced_modules_or_files: tuple[str, ...]
    is_skeleton: bool


def repo_root_from_file() -> Path:
    """Return repository root based on this module location."""

    return Path(__file__).resolve().parents[2]


def _run_command(repo_root: Path, args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        args,
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _normalize_repo_path(token: str) -> str:
    normalized = token.replace("\\", "/")
    while True:
        previous = normalized
        normalized = normalized.strip()
        normalized = normalized.strip("`'\"")
        normalized = normalized.removeprefix("./")
        normalized = normalized.rstrip(".,:;")
        if normalized == previous:
            return normalized


def _is_relevant_path(path: str) -> bool:
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    if "/" not in normalized:
        return normalized not in ROOT_FILES_EXCLUDED
    if any(segment in EPHEMERAL_DIR_SEGMENTS for segment in Path(normalized).parts):
        return False
    return normalized.startswith(KNOWN_TOP_LEVEL_PREFIXES)


def _category_for_path(path: str) -> str:
    if path.startswith("src/"):
        return "src"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("docs/"):
        return "docs"
    if path.startswith("constraints/"):
        return "constraints"
    if path.startswith("tools/"):
        return "tools"
    if path.startswith("scripts/"):
        return "scripts"
    if path.startswith(".github/workflows/"):
        return "workflows"
    if path.startswith("samples/"):
        return "samples"
    if path.startswith("profiles/"):
        return "profiles"
    if path.startswith("docker/"):
        return "docker"
    if "/" not in path:
        return "root"
    return "other"


def _bucketize_paths(paths: list[str]) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = {category: [] for category in CATEGORY_ORDER}
    for path in paths:
        by_category[_category_for_path(path)].append(path)
    for category in CATEGORY_ORDER:
        by_category[category] = sorted(by_category[category])
    return by_category


def _filesystem_fallback_paths(repo_root: Path) -> list[str]:
    paths: list[str] = []
    for file_path in sorted(repo_root.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(repo_root).as_posix()
        if _is_relevant_path(rel_path):
            paths.append(rel_path)
    return paths


def discover_tracked_files(repo_root: Path) -> tuple[list[str], str]:
    """Discover tracked files using git, with deterministic filesystem fallback."""

    code, stdout, _stderr = _run_command(repo_root, ["git", "ls-files"])
    if code == 0:
        git_paths = [
            _normalize_repo_path(line)
            for line in stdout.splitlines()
            if _is_relevant_path(line.strip())
        ]
        if git_paths:
            return sorted(set(git_paths)), "git_ls_files"
    return _filesystem_fallback_paths(repo_root), "filesystem_fallback"


def _extract_python_docstring(lines: list[str], start_index: int) -> tuple[str, int] | None:
    line = lines[start_index].lstrip()
    if not (line.startswith('"""') or line.startswith("'''")):
        return None
    quote = line[:3]
    raw_lines: list[str] = [lines[start_index]]
    stripped = lines[start_index].strip()
    if stripped.count(quote) >= 2 and len(stripped) > len(quote) * 2:
        return "\n".join(raw_lines), start_index + 1
    for idx in range(start_index + 1, len(lines)):
        raw_lines.append(lines[idx])
        if quote in lines[idx]:
            return "\n".join(raw_lines), idx + 1
    return None


def _extract_prefixed_block(
    lines: list[str],
    start_index: int,
    prefix: str,
    style: str,
) -> tuple[str, str, int] | None:
    if not lines[start_index].lstrip().startswith(prefix):
        return None
    raw_lines: list[str] = []
    clean_lines: list[str] = []
    idx = start_index
    while idx < len(lines):
        stripped = lines[idx].lstrip()
        if not stripped.startswith(prefix):
            break
        raw_lines.append(lines[idx])
        clean_lines.append(stripped[len(prefix) :].lstrip())
        idx += 1
    return "\n".join(raw_lines), "\n".join(clean_lines), idx


def _extract_html_comment(lines: list[str], start_index: int) -> tuple[str, int] | None:
    if not lines[start_index].lstrip().startswith("<!--"):
        return None
    raw_lines: list[str] = [lines[start_index]]
    if "-->" in lines[start_index]:
        return "\n".join(raw_lines), start_index + 1
    for idx in range(start_index + 1, len(lines)):
        raw_lines.append(lines[idx])
        if "-->" in lines[idx]:
            return "\n".join(raw_lines), idx + 1
    return None


def _strip_markup(style: str, raw_header: str) -> str:
    if style == "python_docstring":
        if raw_header.startswith('"""') and raw_header.endswith('"""'):
            return raw_header[3:-3].strip("\n")
        if raw_header.startswith("'''") and raw_header.endswith("'''"):
            return raw_header[3:-3].strip("\n")
        return raw_header.strip()
    if style == "html_comment":
        text = raw_header.strip()
        if text.startswith("<!--"):
            text = text[4:]
        if text.endswith("-->"):
            text = text[:-3]
        return text.strip("\n")
    return raw_header.strip("\n")


def _map_section_to_field(section: str) -> str | None:
    normalized = section.strip().rstrip(":").lower()
    for alias, field in SECTION_ALIASES:
        if normalized == alias or normalized.startswith(alias):
            return field
    return None


def _extract_sectioned_fields(
    clean_header: str,
) -> tuple[dict[str, list[str]], set[str], list[str]]:
    fields: dict[str, list[str]] = {
        "file_purpose": [],
        "must_include_items": [],
        "functional_requirements": [],
        "nonfunctional_requirements": [],
        "invariants": [],
    }
    declared_sections: set[str] = set()
    preamble: list[str] = []
    current_field: str | None = None
    for line in clean_header.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        mapped_field = _map_section_to_field(stripped)
        if mapped_field is not None:
            current_field = mapped_field
            declared_sections.add(mapped_field)
            continue
        normalized = stripped
        if normalized.startswith("-") or normalized.startswith("*"):
            normalized = normalized[1:].strip()
        if not normalized:
            continue
        if current_field is None:
            preamble.append(normalized)
            continue
        fields[current_field].append(normalized)
    if not fields["file_purpose"] and preamble:
        fields["file_purpose"] = [preamble[0]]
    return fields, declared_sections, preamble


def _extract_header_references(text: str) -> list[str]:
    refs: set[str] = set()
    for token in PATH_TOKEN_PATTERN.findall(text):
        normalized = _normalize_repo_path(token)
        if normalized:
            refs.add(normalized)
    return sorted(refs)


def extract_header_metadata_from_text(path: str, text: str) -> HeaderExtraction:
    """Extract structured header metadata from raw file content."""

    normalized_text = text.lstrip("\ufeff")
    lines = normalized_text.splitlines()
    if not lines:
        return HeaderExtraction(
            header_present=False,
            header_style=None,
            raw_header="",
            clean_header="",
            declared_sections=tuple(),
            file_purpose=tuple(),
            must_include_items=tuple(),
            functional_requirements=tuple(),
            nonfunctional_requirements=tuple(),
            invariants=tuple(),
            referenced_modules_or_files=tuple(),
            is_skeleton=False,
        )
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return HeaderExtraction(
            header_present=False,
            header_style=None,
            raw_header="",
            clean_header="",
            declared_sections=tuple(),
            file_purpose=tuple(),
            must_include_items=tuple(),
            functional_requirements=tuple(),
            nonfunctional_requirements=tuple(),
            invariants=tuple(),
            referenced_modules_or_files=tuple(),
            is_skeleton=False,
        )
    suffix = Path(path).suffix.lower()
    filename = Path(path).name
    raw_header = ""
    clean_header = ""
    style: str | None = None
    end_index = index
    if suffix == ".py":
        docstring_result = _extract_python_docstring(lines, index)
        if docstring_result is not None:
            raw_header, end_index = docstring_result
            style = "python_docstring"
            clean_header = _strip_markup(style, raw_header)
    if style is None and (suffix == ".md" or filename.lower().endswith(".md")):
        html_result = _extract_html_comment(lines, index)
        if html_result is not None:
            raw_header, end_index = html_result
            style = "html_comment"
            clean_header = _strip_markup(style, raw_header)
    if style is None:
        prefix = ""
        style = None
        if suffix in {".toml", ".yaml", ".yml"} or filename in {"Makefile"}:
            prefix = "#"
            style = "hash_comment"
        elif suffix in {".json", ".jsonc"}:
            prefix = "//"
            style = "slash_comment"
        if style is not None:
            prefixed = _extract_prefixed_block(lines, index, prefix, style)
            if prefixed is not None:
                raw_header, clean_header, end_index = prefixed
            else:
                style = None
    if not raw_header:
        return HeaderExtraction(
            header_present=False,
            header_style=None,
            raw_header="",
            clean_header="",
            declared_sections=tuple(),
            file_purpose=tuple(),
            must_include_items=tuple(),
            functional_requirements=tuple(),
            nonfunctional_requirements=tuple(),
            invariants=tuple(),
            referenced_modules_or_files=tuple(),
            is_skeleton=False,
        )
    section_fields, declared_sections, _preamble = _extract_sectioned_fields(clean_header)
    body_lines = lines[end_index:] if end_index <= len(lines) else []
    body_text = "\n".join(body_lines).strip()
    lower_header = clean_header.lower()
    is_skeleton = bool(
        re.search(r"\bskeleton\b", lower_header)
        or ("placeholder" in lower_header and not body_text)
    )
    return HeaderExtraction(
        header_present=True,
        header_style=style,
        raw_header=raw_header,
        clean_header=clean_header,
        declared_sections=tuple(sorted(declared_sections)),
        file_purpose=tuple(section_fields["file_purpose"]),
        must_include_items=tuple(section_fields["must_include_items"]),
        functional_requirements=tuple(section_fields["functional_requirements"]),
        nonfunctional_requirements=tuple(section_fields["nonfunctional_requirements"]),
        invariants=tuple(section_fields["invariants"]),
        referenced_modules_or_files=tuple(_extract_header_references(clean_header)),
        is_skeleton=is_skeleton,
    )


def extract_header_metadata(repo_root: Path, rel_path: str) -> HeaderExtraction:
    file_path = repo_root / rel_path
    text = file_path.read_text(encoding="utf-8")
    return extract_header_metadata_from_text(rel_path, text)


def _header_record_to_dict(record: HeaderExtraction) -> dict[str, Any]:
    raw_hash = (
        hashlib.sha256(record.raw_header.encode("utf-8")).hexdigest() if record.raw_header else None
    )
    return {
        "header_present": record.header_present,
        "header_style": record.header_style,
        "raw_header_sha256": raw_hash,
        "declared_sections": list(record.declared_sections),
        "file_purpose": list(record.file_purpose),
        "must_include_items": list(record.must_include_items),
        "functional_requirements": list(record.functional_requirements),
        "nonfunctional_requirements": list(record.nonfunctional_requirements),
        "invariants": list(record.invariants),
        "referenced_modules_or_files": list(record.referenced_modules_or_files),
        "is_skeleton": record.is_skeleton,
    }


def _is_normative_doc(path: str) -> bool:
    return path in {
        "docs/BUILD_ORDER.md",
        "docs/FILE_MAP.md",
        "docs/quality/STYLE_AND_LINT.md",
        "docs/threat_model.md",
    } or path.startswith(("docs/architecture/", "docs/runbooks/", "docs/schemas/"))


def _extract_doc_references(
    repo_root: Path, doc_paths: list[str], tracked_set: set[str]
) -> list[PathMap]:
    references: dict[tuple[str, str], PathMap] = {}
    for doc_path in doc_paths:
        content = (repo_root / doc_path).read_text(encoding="utf-8")
        lines = content.splitlines()
        for line in lines:
            marker_present = any(marker in line.lower() for marker in PLANNED_MARKERS)
            for pattern in DOC_REFERENCE_PATTERNS:
                for match in pattern.findall(line):
                    token = match if isinstance(match, str) else match[0]
                    normalized = _resolve_reference_target(token, tracked_set)
                    if normalized is None:
                        continue
                    exists = _path_exists(repo_root, normalized, tracked_set)
                    key = (doc_path, normalized)
                    references[key] = {
                        "source": doc_path,
                        "target": normalized,
                        "exists": exists,
                        "planned_marker": marker_present,
                    }
    return [references[key] for key in sorted(references)]


def _resolve_reference_target(token: str, tracked_set: set[str]) -> str | None:
    normalized = _normalize_repo_path(token)
    if not normalized or " " in normalized or "..." in normalized:
        return None
    if normalized in tracked_set:
        return normalized
    if "/" not in normalized:
        if normalized in KNOWN_ROOT_REFERENCES:
            return normalized
        directory_prefix = f"{normalized}/"
        if any(path.startswith(directory_prefix) for path in tracked_set):
            return directory_prefix
        basename_matches = sorted(path for path in tracked_set if Path(path).name == normalized)
        if len(basename_matches) == 1:
            return basename_matches[0]
        return None
    if _is_relevant_path(normalized):
        return normalized
    return None


def _path_exists(repo_root: Path, rel_path: str, tracked_set: set[str]) -> bool:
    if rel_path in tracked_set:
        return True
    if rel_path.endswith("/"):
        return any(path.startswith(rel_path) for path in tracked_set)
    if "*" in rel_path or "?" in rel_path:
        return bool(list(repo_root.glob(rel_path)))
    return (repo_root / rel_path).exists()


def _parse_build_order(repo_root: Path, tracked_set: set[str]) -> dict[str, Any]:
    build_order_path = repo_root / "docs/BUILD_ORDER.md"
    if not build_order_path.exists():
        return {
            "ordered_phases": [],
            "phases": [],
            "dag_edges": [],
            "acyclic": True,
        }
    text = build_order_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    phase_pattern = re.compile(r"^##\s+(Phase\s+(\d+)\s+—\s+.+)$")
    file_pattern = re.compile(r"`([^`]+)`")
    phases: list[PathMap] = []
    current_phase: PathMap | None = None
    for line in lines:
        phase_match = phase_pattern.match(line.strip())
        if phase_match:
            if current_phase is not None:
                current_phase["files"] = sorted(set(current_phase["files"]))
                current_phase["notes"] = " ".join(current_phase["notes"]).strip()
                phases.append(current_phase)
            name = phase_match.group(1).strip()
            number = int(phase_match.group(2))
            current_phase = {
                "name": name,
                "number": number,
                "files": [],
                "notes": [],
                "depends_on_numbers": set(),
            }
            continue
        if current_phase is None:
            continue
        stripped = line.strip()
        if re.match(r"^(?:\d+\.\s+|- |\* )", stripped):
            for token in file_pattern.findall(line):
                resolved = _resolve_reference_target(token, tracked_set)
                if resolved is None:
                    continue
                if resolved.endswith("/") or "/" not in resolved:
                    continue
                current_phase["files"].append(resolved)
        if stripped.startswith("**Goal:**") or stripped.startswith("**Dependencies:**"):
            current_phase["notes"].append(stripped)
        if stripped.startswith("**Acceptance test:**"):
            current_phase["notes"].append(stripped)
        if "Dependencies:" in stripped or "**Dependencies:**" in stripped:
            for dep_number in re.findall(r"Phase\s+(\d+)", stripped):
                current_phase["depends_on_numbers"].add(int(dep_number))
    if current_phase is not None:
        current_phase["files"] = sorted(set(current_phase["files"]))
        current_phase["notes"] = " ".join(current_phase["notes"]).strip()
        phases.append(current_phase)
    number_to_name = {phase["number"]: phase["name"] for phase in phases}
    for phase in phases:
        depends_on = sorted(
            number_to_name[number]
            for number in phase["depends_on_numbers"]
            if number in number_to_name and number < phase["number"]
        )
        phase["depends_on"] = depends_on
        del phase["depends_on_numbers"]
    ordered_phases = [phase["name"] for phase in sorted(phases, key=lambda item: item["number"])]
    dag_edges: list[tuple[str, str]] = []
    for phase in sorted(phases, key=lambda item: item["number"]):
        for dependency in phase["depends_on"]:
            dag_edges.append((dependency, phase["name"]))
    if len(ordered_phases) > 1:
        for prev_phase, next_phase in zip(ordered_phases[:-1], ordered_phases[1:], strict=False):
            edge = (prev_phase, next_phase)
            if edge not in dag_edges:
                dag_edges.append(edge)
    acyclic = _is_acyclic(ordered_phases, dag_edges)
    serializable_phases = [
        {
            "name": phase["name"],
            "number": phase["number"],
            "files": phase["files"],
            "notes": phase["notes"],
            "depends_on": phase["depends_on"],
        }
        for phase in sorted(phases, key=lambda item: item["number"])
    ]
    return {
        "ordered_phases": ordered_phases,
        "phases": serializable_phases,
        "dag_edges": [[left, right] for left, right in sorted(dag_edges)],
        "acyclic": acyclic,
    }


def _is_acyclic(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    indegree: dict[str, int] = {node: 0 for node in nodes}
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        if source not in indegree or target not in indegree:
            continue
        adjacency[source].append(target)
        indegree[target] += 1
    queue: deque[str] = deque(sorted([node for node, degree in indegree.items() if degree == 0]))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in sorted(adjacency[node]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return visited == len(nodes)


def _classify_plane(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3 and parts[0] == "src" and parts[1] == "nexus_orchestrator":
        top = parts[2]
        if top in PLANE_MAP:
            return PLANE_MAP[top]
    return "control"


def _extract_python_dependencies(repo_root: Path, src_path: str) -> list[str]:
    file_path = repo_root / src_path
    text = file_path.read_text(encoding="utf-8")
    try:
        module = ast.parse(text)
    except SyntaxError:
        return []
    deps: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name.startswith("nexus_orchestrator."):
                    rel = "src/" + module_name.replace(".", "/") + ".py"
                    deps.add(rel)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("nexus_orchestrator.")
        ):
            rel = "src/" + node.module.replace(".", "/") + ".py"
            deps.add(rel)
    return sorted(dep for dep in deps if dep != src_path)


def _build_phase_lookup(phase_order: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for phase in phase_order["phases"]:
        phase_name = phase["name"]
        for path in phase["files"]:
            lookup[path] = phase_name
    return lookup


def _build_tests_covering_lookup(phase_order: dict[str, Any]) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = defaultdict(list)
    for phase in phase_order["phases"]:
        src_files = [path for path in phase["files"] if path.startswith("src/")]
        test_files = [path for path in phase["files"] if path.startswith("tests/")]
        for src_file in src_files:
            lookup[src_file].extend(test_files)
    for src_file, tests in lookup.items():
        lookup[src_file] = sorted(set(tests))
    return dict(lookup)


def scan_placeholders_in_paths(
    repo_root: Path,
    paths: list[str],
    protected_paths: set[str],
) -> dict[str, list[PathMap]]:
    warnings: list[PathMap] = []
    errors: list[PathMap] = []
    for path in paths:
        text = (repo_root / path).read_text(encoding="utf-8")
        matches = sorted(
            {
                pattern_name
                for pattern_name, pattern in PLACEHOLDER_PATTERNS
                if pattern.search(text) is not None
            }
        )
        if not matches:
            continue
        finding = {
            "path": path,
            "matches": matches,
            "severity": "error" if path in protected_paths else "warning",
        }
        if finding["severity"] == "error":
            errors.append(finding)
        else:
            warnings.append(finding)
    return {
        "warnings": sorted(warnings, key=lambda item: item["path"]),
        "errors": sorted(errors, key=lambda item: item["path"]),
    }


def _build_risk_map(blueprint: dict[str, Any]) -> dict[str, list[str]]:
    core: set[str] = set()
    security_sensitive: set[str] = set()
    correctness_critical: set[str] = set()
    for path in blueprint["tracked_files"]["all"]:
        if path.startswith("src/nexus_orchestrator/control_plane/"):
            core.add(path)
        if path.startswith("src/nexus_orchestrator/planning/"):
            core.add(path)
        if path in {
            "src/nexus_orchestrator/main.py",
            "src/nexus_orchestrator/verification_plane/constraint_gate.py",
            "src/nexus_orchestrator/verification_plane/pipeline.py",
            "scripts/repo_audit.py",
        }:
            core.add(path)
        if "/security/" in path or "/sandbox/" in path:
            security_sensitive.add(path)
        if path.startswith("constraints/registry/") or path.startswith("tools/registry"):
            security_sensitive.add(path)
        if path.startswith("src/nexus_orchestrator/domain/"):
            correctness_critical.add(path)
        if path.startswith("src/nexus_orchestrator/verification_plane/"):
            correctness_critical.add(path)
        if path.startswith("src/nexus_orchestrator/integration_plane/"):
            correctness_critical.add(path)
        if path.startswith("src/nexus_orchestrator/config/"):
            correctness_critical.add(path)
        if path.startswith("constraints/registry/"):
            correctness_critical.add(path)
    return {
        "core": sorted(core),
        "security_sensitive": sorted(security_sensitive),
        "correctness_critical": sorted(correctness_critical),
    }


def build_repo_blueprint(repo_root: Path) -> dict[str, Any]:
    tracked_files, source_strategy = discover_tracked_files(repo_root)
    tracked_set = set(tracked_files)
    by_category = _bucketize_paths(tracked_files)
    header_records: dict[str, dict[str, Any]] = {}
    observed_styles: set[str] = set()
    observed_sections: set[str] = set()
    suspected_skeleton_files: list[str] = []
    for path in tracked_files:
        if _category_for_path(path) not in {
            "src",
            "tests",
            "docs",
            "constraints",
            "tools",
            "scripts",
            "workflows",
            "root",
        }:
            continue
        metadata = extract_header_metadata(repo_root, path)
        record = _header_record_to_dict(metadata)
        header_records[path] = record
        if metadata.header_style:
            observed_styles.add(metadata.header_style)
        observed_sections.update(record["declared_sections"])
        if metadata.is_skeleton:
            suspected_skeleton_files.append(path)
    normative_docs = sorted(path for path in by_category["docs"] if _is_normative_doc(path))
    docs_mentions = _extract_doc_references(repo_root, normative_docs, tracked_set)
    header_mentions: list[PathMap] = []
    for path, record in sorted(header_records.items()):
        for target in record["referenced_modules_or_files"]:
            header_mentions.append(
                {
                    "source": path,
                    "target": target,
                    "exists": _path_exists(repo_root, target, tracked_set),
                }
            )
    phase_order = _parse_build_order(repo_root, tracked_set)
    phase_lookup = _build_phase_lookup(phase_order)
    tests_lookup = _build_tests_covering_lookup(phase_order)
    src_planes: dict[str, PathMap] = {}
    src_module_index: list[PathMap] = []
    for src_path in by_category["src"]:
        plane = _classify_plane(src_path)
        phase = phase_lookup.get(src_path, "Unassigned")
        import_deps = (
            _extract_python_dependencies(repo_root, src_path) if src_path.endswith(".py") else []
        )
        header_deps = [
            dep
            for dep in header_records[src_path]["referenced_modules_or_files"]
            if dep.startswith("src/")
            and dep != src_path
            and _path_exists(repo_root, dep, tracked_set)
        ]
        dependencies = sorted(set(import_deps + header_deps))
        tests_covering = tests_lookup.get(src_path, [])
        src_planes[src_path] = {"plane": plane, "phase": phase}
        src_module_index.append(
            {
                "path": src_path,
                "plane": plane,
                "phase": phase,
                "dependencies": dependencies,
                "tests_covering": tests_covering,
            }
        )
    protected_paths = {
        "src/nexus_orchestrator/repo_blueprint.py",
        "scripts/repo_audit.py",
    }
    placeholder_scan = scan_placeholders_in_paths(repo_root, by_category["src"], protected_paths)
    blueprint: dict[str, Any] = {
        "metadata": {
            "schema_version": "1",
            "source_strategy": source_strategy,
            "repo_root": str(repo_root),
        },
        "tracked_files": {
            "all": tracked_files,
            "by_category": by_category,
        },
        "headers": {
            "format_observed": {
                "styles": sorted(observed_styles),
                "section_fields": sorted(observed_sections),
            },
            "records": header_records,
            "suspected_skeleton_files": sorted(set(suspected_skeleton_files)),
        },
        "phase_order": phase_order,
        "src_planes": src_planes,
        "src_module_index": sorted(src_module_index, key=lambda item: item["path"]),
        "cross_reference_index": {
            "normative_docs": normative_docs,
            "docs_mentions": docs_mentions,
            "header_mentions": sorted(
                header_mentions, key=lambda item: (item["source"], item["target"])
            ),
        },
        "risk_map": {},
        "placeholder_scan": placeholder_scan,
    }
    blueprint["risk_map"] = _build_risk_map(blueprint)
    return blueprint


def validate_repo_blueprint(blueprint: dict[str, Any], repo_root: Path) -> dict[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    tracked_files = blueprint["tracked_files"]["all"]
    if len(tracked_files) != len(set(tracked_files)):
        errors.append("tracked_files contains duplicate paths")
    for path in tracked_files:
        normalized = _normalize_repo_path(path)
        if path != normalized:
            errors.append(f"path normalization mismatch: {path!r} -> {normalized!r}")
        if ".." in Path(path).parts:
            errors.append(f"path traversal segment not allowed: {path}")
    discovered_files, strategy = discover_tracked_files(repo_root)
    if tracked_files != discovered_files:
        errors.append(
            "tracked_files mismatch with discovery source "
            f"({strategy}); regenerate blueprint artifacts"
        )
    src_and_tests = (
        blueprint["tracked_files"]["by_category"]["src"]
        + blueprint["tracked_files"]["by_category"]["tests"]
    )
    records: dict[str, dict[str, Any]] = blueprint["headers"]["records"]
    for path in src_and_tests:
        record = records.get(path)
        if record is None:
            errors.append(f"missing header record: {path}")
            continue
        if not record["header_present"]:
            errors.append(f"missing required top header: {path}")
            continue
        first = extract_header_metadata(repo_root, path)
        second = extract_header_metadata(repo_root, path)
        first_hash = hashlib.sha256(first.raw_header.encode("utf-8")).hexdigest()
        second_hash = hashlib.sha256(second.raw_header.encode("utf-8")).hexdigest()
        if first_hash != second_hash:
            errors.append(f"unstable header hash across repeated extraction: {path}")
        for field in (
            "file_purpose",
            "must_include_items",
            "functional_requirements",
            "nonfunctional_requirements",
            "invariants",
        ):
            if field in record["declared_sections"] and not record[field]:
                errors.append(f"declared section {field} is empty in {path}")
    for mention in blueprint["cross_reference_index"]["docs_mentions"]:
        if not mention["exists"] and not mention["planned_marker"]:
            errors.append(
                "missing path referenced by normative docs without planned marker: "
                f"{mention['source']} -> {mention['target']}"
            )
    for mention in blueprint["cross_reference_index"]["header_mentions"]:
        if not mention["exists"]:
            warnings.append(
                f"header reference target missing: {mention['source']} -> {mention['target']}"
            )
    phase_order = blueprint["phase_order"]
    if not phase_order["acyclic"]:
        errors.append("phase order graph contains a cycle")
    phase_file_counts: Counter[str] = Counter()
    tracked_set = set(tracked_files)
    for phase in phase_order["phases"]:
        for path in phase["files"]:
            phase_file_counts[path] += 1
            if path not in tracked_set:
                errors.append(f"phase references missing file: {phase['name']} -> {path}")
    duplicate_phase_assignments = [path for path, count in phase_file_counts.items() if count > 1]
    if duplicate_phase_assignments:
        errors.append(
            "files assigned to multiple phases: " + ", ".join(sorted(duplicate_phase_assignments))
        )
    placeholder_scan = blueprint["placeholder_scan"]
    for item in placeholder_scan["errors"]:
        errors.append(
            f"placeholder pattern in protected path {item['path']}: " + ", ".join(item["matches"])
        )
    for item in placeholder_scan["warnings"]:
        warnings.append(
            f"placeholder pattern in skeleton path {item['path']}: " + ", ".join(item["matches"])
        )
    return {"errors": errors, "warnings": warnings}


def blueprint_to_json(blueprint: dict[str, Any]) -> str:
    return json.dumps(blueprint, sort_keys=True, indent=2) + "\n"


def _phase_map_lines(blueprint: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for phase in blueprint["phase_order"]["phases"]:
        lines.append(f"- {phase['name']}: {len(phase['files'])} files")
    return lines


def _summary_lines(blueprint: dict[str, Any], validation: dict[str, list[str]]) -> list[str]:
    tracked_count = len(blueprint["tracked_files"]["all"])
    src_count = len(blueprint["tracked_files"]["by_category"]["src"])
    test_count = len(blueprint["tracked_files"]["by_category"]["tests"])
    docs_count = len(blueprint["tracked_files"]["by_category"]["docs"])
    lines = [
        "Repo Blueprint Summary",
        "----------------------",
        f"Source strategy: {blueprint['metadata']['source_strategy']}",
        f"Tracked files: {tracked_count}",
        f"src={src_count} tests={test_count} docs={docs_count}",
        (f"Skeleton files detected: {len(blueprint['headers']['suspected_skeleton_files'])}"),
        (
            "Validation findings: "
            f"{len(validation['errors'])} errors, {len(validation['warnings'])} warnings"
        ),
    ]
    return lines


def render_blueprint_markdown(blueprint: dict[str, Any]) -> str:
    lines: list[str] = [
        "<!--",
        "nexus-orchestrator — repository blueprint report",
        "",
        "File: docs/REPO_BLUEPRINT.md",
        "Last updated: 2026-02-11",
        "",
        "Purpose",
        "- Human-readable map of repository planes, phases, dependencies, and quality risks.",
        "",
        "What should be included in this file",
        "- Compact architecture overview.",
        "- Source module table: path, plane, phase, dependencies, tests covering it.",
        "- Missing/skeleton areas with exact file pointers.",
        "",
        "Functional requirements",
        "- Must be derived from deterministic audit extraction, not hand-maintained prose.",
        "",
        "Non-functional requirements",
        "- Keep this document concise enough for small context windows.",
        "-->",
        "",
        "# Repo Blueprint",
        "",
        "## How The Repo Fits Together",
        "",
        (
            "NEXUS is organized into plane-oriented Python packages under "
            "`src/nexus_orchestrator/`, with implementation sequencing defined in "
            "`docs/BUILD_ORDER.md` and folder-level intent defined in `docs/FILE_MAP.md`."
        ),
        (
            "This report is generated from repository state plus headers and doc references, "
            "then validated against deterministic invariants."
        ),
        "",
        "## Source Map",
        "",
        "| Path | Plane | Phase | Dependencies | Tests Covering |",
        "|---|---|---|---|---|",
    ]
    for module in blueprint["src_module_index"]:
        deps = ", ".join(module["dependencies"]) if module["dependencies"] else "-"
        tests = ", ".join(module["tests_covering"]) if module["tests_covering"] else "-"
        lines.append(
            f"| `{module['path']}` | `{module['plane']}` | `{module['phase']}` | "
            f"`{deps}` | `{tests}` |"
        )
    lines.extend(
        [
            "",
            "## Missing / Skeleton Areas",
            "",
            "The following files are still identified as skeleton placeholders:",
            "",
        ]
    )
    for path in blueprint["headers"]["suspected_skeleton_files"]:
        if path.startswith("src/"):
            lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Risk Map",
            "",
            f"- Core modules: {len(blueprint['risk_map']['core'])}",
            f"- Security-sensitive modules: {len(blueprint['risk_map']['security_sensitive'])}",
            f"- Correctness-critical modules: {len(blueprint['risk_map']['correctness_critical'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_blueprint_artifacts(repo_root: Path, blueprint: dict[str, Any]) -> tuple[Path, Path]:
    generated_dir = repo_root / "docs" / "_generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    json_path = generated_dir / "repo_blueprint.json"
    markdown_path = repo_root / "docs" / "REPO_BLUEPRINT.md"
    json_path.write_text(blueprint_to_json(blueprint), encoding="utf-8")
    markdown_path.write_text(render_blueprint_markdown(blueprint), encoding="utf-8")
    return json_path, markdown_path


def _print_lines(lines: list[str]) -> None:
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and validate NEXUS repository blueprint."
    )
    parser.add_argument("--summary", action="store_true", help="Print compact blueprint summary.")
    parser.add_argument(
        "--print-phase-map", action="store_true", help="Print phase ordering overview."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run consistency validation checks."
    )
    parser.add_argument("--json", action="store_true", help="Print blueprint JSON to stdout.")
    parser.add_argument(
        "--fail-on-warn", action="store_true", help="Treat validation warnings as failures."
    )
    parser.add_argument(
        "--write-artifacts", action="store_true", help="Write docs artifacts to disk."
    )
    parser.add_argument("--repo-root", default=None, help="Override repository root path.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_file()
    blueprint = build_repo_blueprint(repo_root)
    validation = validate_repo_blueprint(blueprint, repo_root)

    no_flags_selected = not any(
        [args.summary, args.print_phase_map, args.validate, args.json, args.write_artifacts]
    )
    if args.write_artifacts:
        json_path, markdown_path = write_blueprint_artifacts(repo_root, blueprint)
        print(f"Wrote {json_path.relative_to(repo_root)}")
        print(f"Wrote {markdown_path.relative_to(repo_root)}")
    if args.summary or no_flags_selected:
        _print_lines(_summary_lines(blueprint, validation))
    if args.print_phase_map:
        print("Phase Map")
        print("---------")
        _print_lines(_phase_map_lines(blueprint))
    if args.validate or no_flags_selected:
        print("Validation")
        print("----------")
        if validation["errors"]:
            for error in validation["errors"]:
                print(f"ERROR: {error}")
        if validation["warnings"]:
            for warning in validation["warnings"]:
                print(f"WARN: {warning}")
        if not validation["errors"] and not validation["warnings"]:
            print("OK: no findings")
    if args.json:
        output = blueprint_to_json(blueprint)
        sys.stdout.write(output)

    enforce_failure_status = args.validate or no_flags_selected or args.fail_on_warn
    has_errors = bool(validation["errors"])
    has_warnings = bool(validation["warnings"])
    if enforce_failure_status and has_errors:
        return 1
    if enforce_failure_status and args.fail_on_warn and has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
