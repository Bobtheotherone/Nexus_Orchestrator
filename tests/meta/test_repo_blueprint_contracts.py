"""
nexus-orchestrator — repository blueprint contract tests

File: tests/meta/test_repo_blueprint_contracts.py
Last updated: 2026-02-11

Purpose
- Enforce hard-to-cheat invariants for tracked files, headers, docs references, phases, and audit determinism.

What this test file should cover
- Git-vs-filesystem tracked file invariants.
- Header extraction integrity and parser robustness for adversarial inputs.
- Normative docs cross-reference and phase ordering consistency.
- Placeholder scanning behavior and strictness for protected implementation files.
- Deterministic JSON audit output across repeated runs.
- Property-based safety checks for parser stability and normalization idempotence.

Functional requirements
- All tests must run offline and deterministically.
- No external provider SDKs or keys required.

Non-functional requirements
- Keep runtime low for local iteration.
"""

from __future__ import annotations

import hashlib
import json
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"


def _load_repo_blueprint_module() -> object:
    try:
        import nexus_orchestrator.repo_blueprint as module

        return module
    except ModuleNotFoundError:
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))
        import nexus_orchestrator.repo_blueprint as module

        return module


_repo_blueprint_module = _load_repo_blueprint_module()
_filesystem_fallback_paths = _repo_blueprint_module._filesystem_fallback_paths
_normalize_repo_path = _repo_blueprint_module._normalize_repo_path
build_repo_blueprint = _repo_blueprint_module.build_repo_blueprint
discover_tracked_files = _repo_blueprint_module.discover_tracked_files
extract_header_metadata = _repo_blueprint_module.extract_header_metadata
extract_header_metadata_from_text = _repo_blueprint_module.extract_header_metadata_from_text
scan_placeholders_in_paths = _repo_blueprint_module.scan_placeholders_in_paths
validate_repo_blueprint = _repo_blueprint_module.validate_repo_blueprint


def test_git_or_fallback_discovery_matches_blueprint_and_paths_are_normalized() -> None:
    blueprint = build_repo_blueprint(REPO_ROOT)
    discovered_paths, strategy = discover_tracked_files(REPO_ROOT)
    assert blueprint["tracked_files"]["all"] == discovered_paths
    assert len(discovered_paths) == len(set(discovered_paths))
    for rel_path in discovered_paths:
        assert rel_path == _normalize_repo_path(rel_path)
        assert not rel_path.startswith("./")
        assert ".." not in Path(rel_path).parts
    filesystem_paths = _filesystem_fallback_paths(REPO_ROOT)
    assert set(discovered_paths).issubset(set(filesystem_paths))
    if strategy == "filesystem_fallback":
        assert discovered_paths == filesystem_paths


def test_headers_exist_and_are_stable_for_all_src_and_test_files() -> None:
    blueprint = build_repo_blueprint(REPO_ROOT)
    header_records = blueprint["headers"]["records"]
    src_and_tests = (
        blueprint["tracked_files"]["by_category"]["src"]
        + blueprint["tracked_files"]["by_category"]["tests"]
    )
    for rel_path in src_and_tests:
        assert rel_path in header_records
        record = header_records[rel_path]
        assert record["header_present"], rel_path
        first = extract_header_metadata(REPO_ROOT, rel_path)
        second = extract_header_metadata(REPO_ROOT, rel_path)
        first_hash = hashlib.sha256(first.raw_header.encode("utf-8")).hexdigest()
        second_hash = hashlib.sha256(second.raw_header.encode("utf-8")).hexdigest()
        assert first_hash == second_hash
        for field_name in (
            "file_purpose",
            "must_include_items",
            "functional_requirements",
            "nonfunctional_requirements",
            "invariants",
        ):
            if field_name in record["declared_sections"]:
                assert record[field_name], (rel_path, field_name)


@pytest.mark.parametrize(
    ("path", "content", "style"),
    (
        (
            "sample.py",
            '\ufeff#!/usr/bin/env python3\n\n"""\nPurpose\n- parser should survive BOM and shebang.\n"""\n',
            "python_docstring",
        ),
        (
            "sample.md",
            "\n<!--\nPurpose\n- parser should handle html comments.\n-->\n# title\n",
            "html_comment",
        ),
        (
            "sample.toml",
            "# Purpose\n# - parser should handle hash comments.\n[section]\nkey = 'value'\n",
            "hash_comment",
        ),
        (
            "sample.jsonc",
            "// Purpose\n// - parser should handle slash comments.\n{}\n",
            "slash_comment",
        ),
    ),
)
def test_header_parser_handles_adversarial_encodings_and_prefixes(
    path: str,
    content: str,
    style: str,
) -> None:
    metadata = extract_header_metadata_from_text(path, content)
    assert metadata.header_present
    assert metadata.header_style == style
    assert metadata.file_purpose


def test_normative_doc_references_exist_or_are_explicitly_planned() -> None:
    blueprint = build_repo_blueprint(REPO_ROOT)
    for mention in blueprint["cross_reference_index"]["docs_mentions"]:
        assert mention["exists"] or mention["planned_marker"], mention


def test_phase_files_exist_are_unique_and_phase_graph_is_acyclic() -> None:
    blueprint = build_repo_blueprint(REPO_ROOT)
    phase_order = blueprint["phase_order"]
    if not phase_order["phases"]:
        pytest.skip("BUILD_ORDER phase map is not present.")
    tracked_paths = set(blueprint["tracked_files"]["all"])
    counts: Counter[str] = Counter()
    for phase in phase_order["phases"]:
        for rel_path in phase["files"]:
            counts[rel_path] += 1
            assert rel_path in tracked_paths, (phase["name"], rel_path)
    duplicates = sorted(path for path, count in counts.items() if count > 1)
    assert not duplicates
    assert phase_order["acyclic"]


def test_placeholder_scanner_enforces_protected_paths() -> None:
    blueprint = build_repo_blueprint(REPO_ROOT)
    assert blueprint["placeholder_scan"]["errors"] == []


def test_placeholder_scanner_flags_known_placeholder_pattern(tmp_path: Path) -> None:
    temp_file = tmp_path / "placeholder.py"
    temp_file.write_text(
        '"""\nmodule skeleton\n"""\n# TODO: replace me\n',
        encoding="utf-8",
    )
    scan = scan_placeholders_in_paths(tmp_path, ["placeholder.py"], {"placeholder.py"})
    assert scan["errors"]
    matches = set(scan["errors"][0]["matches"])
    assert {"skeleton_header", "todo_fixme"}.issubset(matches)


def test_audit_json_output_is_byte_stable() -> None:
    cmd = [sys.executable, "scripts/repo_audit.py", "--json"]
    first = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    second = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert first.stdout == second.stdout
    parsed = json.loads(first.stdout)
    assert parsed["metadata"]["source_strategy"] in {"git_ls_files", "filesystem_fallback"}


def test_validate_repo_blueprint_has_no_errors() -> None:
    blueprint = build_repo_blueprint(REPO_ROOT)
    validation = validate_repo_blueprint(blueprint, REPO_ROOT)
    assert validation["errors"] == []


HEADER_TEXT_ALPHABET = string.ascii_letters + string.digits + " \n\t:;.,_-/`'\"#!*"
PATH_TEXT_ALPHABET = string.ascii_letters + string.digits + "._-/\\`'\" "


@settings(max_examples=80, deadline=None)
@given(
    content=st.text(alphabet=HEADER_TEXT_ALPHABET, max_size=500),
    suffix=st.sampled_from([".py", ".md", ".toml", ".jsonc"]),
)
def test_property_header_parser_never_crashes(content: str, suffix: str) -> None:
    path = f"generated{suffix}"
    metadata = extract_header_metadata_from_text(path, content)
    assert isinstance(metadata.header_present, bool)
    assert metadata.header_style in {
        None,
        "python_docstring",
        "html_comment",
        "hash_comment",
        "slash_comment",
    }
    assert isinstance(metadata.referenced_modules_or_files, tuple)


@settings(max_examples=120, deadline=None)
@given(value=st.text(alphabet=PATH_TEXT_ALPHABET, max_size=120))
def test_property_normalize_repo_path_is_idempotent(value: str) -> None:
    once = _normalize_repo_path(value)
    twice = _normalize_repo_path(once)
    assert once == twice
