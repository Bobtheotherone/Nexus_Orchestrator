"""
nexus-orchestrator — unit tests for certainty-aware placeholder audit

File: tests/unit/quality/test_placeholder_audit.py
Last updated: 2026-02-13

Purpose
- Verify ERROR vs WARNING classification, determinism, and output formatting for
  semantic placeholder scanning.

What this test file should cover
- High-certainty placeholder violations that must fail.
- Ambiguous placeholder-like patterns that warn only when explicitly enabled.
- Deterministic sorting and stable output payload shape.
- CLI argument wiring and exit code semantics.
- Deterministic fuzz/property coverage for opt-in string-marker warnings.

Functional requirements
- Offline only.

Non-functional requirements
- Deterministic and hard to bypass.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

from nexus_orchestrator.quality import placeholder_audit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pytest

try:
    from hypothesis import given, seed, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback path
    HYPOTHESIS_AVAILABLE = False


def _write(root: Path, rel_path: str, text: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scan(
    root: Path,
    *,
    roots: tuple[str, ...] = ("src",),
    exclude: tuple[str, ...] = (),
    show_context: int = 2,
    include_docstrings: bool = False,
    include_string_literals: bool = False,
    scan_tests_as_blocking: bool = False,
    severity_by_path: (Mapping[str, str] | Sequence[str] | Sequence[tuple[str, str]] | None) = None,
    warn_on_string_markers: bool = False,
    warn_on_audit_tool_self_references: bool = False,
    self_reference_allowlist: tuple[str, ...] = placeholder_audit.DEFAULT_SELF_REFERENCE_ALLOWLIST,
) -> placeholder_audit.AuditResult:
    return placeholder_audit.run_placeholder_audit(
        repo_root=root,
        roots=roots,
        exclude=exclude,
        show_context=show_context,
        include_docstrings=include_docstrings,
        include_string_literals=include_string_literals,
        scan_tests_as_blocking=scan_tests_as_blocking,
        severity_by_path=severity_by_path,
        warn_on_string_markers=warn_on_string_markers,
        warn_on_audit_tool_self_references=warn_on_audit_tool_self_references,
        self_reference_allowlist=self_reference_allowlist,
    )


def _kinds(result: placeholder_audit.AuditResult, severity: str) -> set[str]:
    return {item.kind for item in result.findings if item.severity == severity}


def test_certain_error_classification_for_executable_placeholders(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/errors.py",
        (
            "# TODO: replace implementation\n"
            "def implement_me() -> None:\n"
            "    raise NotImplementedError('missing')\n\n"
            "def pass_stub() -> None:\n"
            "    pass\n\n"
            "def done() -> None:\n"
            "    return None\n"
        ),
    )

    result = _scan(tmp_path)

    assert result.error_count >= 3
    assert "todo_fixme_comment" in _kinds(result, "ERROR")
    assert "raise_not_implemented_error" in _kinds(result, "ERROR")
    assert "pass_only_function_body" in _kinds(result, "ERROR")


def test_ambiguous_protocol_and_type_checking_ellipsis_are_warnings(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/typing_cases.py",
        (
            "from __future__ import annotations\n"
            "from abc import abstractmethod\n"
            "from typing import TYPE_CHECKING, Protocol, overload\n\n"
            "class Contract(Protocol):\n"
            "    def run(self) -> None: ...\n\n"
            "class AbstractBase:\n"
            "    @abstractmethod\n"
            "    def build(self) -> None: ...\n\n"
            "@overload\n"
            "def decode(value: int) -> int: ...\n\n"
            "@overload\n"
            "def decode(value: str) -> str: ...\n\n"
            "def decode(value: int | str) -> int | str:\n"
            "    return value\n\n"
            "if TYPE_CHECKING:\n"
            "    class TypeOnly(Protocol):\n"
            "        def check(self) -> None: ...\n"
        ),
    )

    result = _scan(tmp_path)

    assert result.error_count == 0
    warning_kinds = _kinds(result, "WARNING")
    assert "ellipsis_only_function_body" in warning_kinds


def test_string_literals_are_suppressed_by_default(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/ambiguous.py",
        (
            "import re\n\n"
            "PATTERN = r'TODO|FIXME|NotImplementedError'\n"
            "MESSAGE = 'NotImplementedError appears in content, not code flow'\n"
            "COMPILED = re.compile(r'raise\\s+NotImplementedError')\n"
        ),
    )

    result = _scan(tmp_path)

    assert result.error_count == 0
    assert result.warning_count == 0
    assert "placeholder_string_literal" not in _kinds(result, "WARNING")


def test_todo_comment_in_tests_is_warning_by_default(tmp_path: Path) -> None:
    _write(tmp_path, "tests/unit/test_warning_case.py", "# TODO: fixture follow-up\n")

    result = _scan(tmp_path, roots=("tests",), exclude=())

    assert result.error_count == 0
    assert result.warning_count == 1
    assert _kinds(result, "ERROR") == set()
    assert _kinds(result, "WARNING") == {"todo_fixme_comment"}
    assert result.findings[0].path == "tests/unit/test_warning_case.py"


def test_todo_comment_in_tests_can_be_blocking_by_policy(tmp_path: Path) -> None:
    _write(tmp_path, "tests/unit/test_warning_case.py", "# TODO: fixture follow-up\n")

    result = _scan(tmp_path, roots=("tests",), exclude=(), scan_tests_as_blocking=True)

    assert result.error_count == 1
    assert result.warning_count == 0
    assert _kinds(result, "ERROR") == {"todo_fixme_comment"}


def test_fixture_strings_in_tests_are_not_blocking_by_default(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "tests/unit/test_fixture_payloads.py",
        (
            "import re\n\n"
            "FIXTURE_TEXT = 'TODO/FIXME/NotImplementedError fixture payload sample'\n"
            "PATTERN = r'TODO|FIXME|NotImplementedError'\n"
            "MATCHER = re.compile(r'raise\\s+NotImplementedError')\n"
        ),
    )

    result = _scan(tmp_path, roots=("tests",), exclude=())

    assert result.error_count == 0
    assert result.warning_count == 0
    assert result.findings == ()


def test_docstring_markers_are_opt_in(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/docstrings.py",
        (
            '"""TODO: module docs marker"""\n\n'
            "def build() -> None:\n"
            '    """FIXME: function docs marker"""\n'
            "    return None\n"
        ),
    )

    default_result = _scan(tmp_path, include_string_literals=True)
    opt_in_result = _scan(tmp_path, include_string_literals=True, include_docstrings=True)

    assert default_result.error_count == 0
    assert default_result.warning_count == 0
    assert not default_result.findings

    assert opt_in_result.error_count == 0
    assert opt_in_result.warning_count == 2
    assert _kinds(opt_in_result, "WARNING") == {"placeholder_string_literal"}


def test_path_severity_override_can_demote_src_comment(tmp_path: Path) -> None:
    _write(tmp_path, "src/example.py", "# TODO: demote to warning\n")

    result = _scan(tmp_path, severity_by_path=("src/**=WARNING", "**=WARNING"))

    assert result.error_count == 0
    assert result.warning_count == 1
    assert _kinds(result, "WARNING") == {"todo_fixme_comment"}


def test_warn_on_string_markers_emits_warning_with_context(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/strings.py",
        (
            "def render() -> str:\n"
            "    marker = 'TODO appears in a user-facing payload'\n"
            "    return marker\n"
        ),
    )

    result = _scan(tmp_path, warn_on_string_markers=True, show_context=1)
    string_findings = [
        item for item in result.findings if item.kind == "placeholder_string_literal"
    ]

    assert result.error_count == 0
    assert len(string_findings) == 1

    finding = string_findings[0]
    assert finding.severity == "WARNING"
    assert finding.path == "src/strings.py"
    assert finding.context_lines
    assert any(context.line == finding.line for context in finding.context_lines)
    assert "TODO" in finding.snippet


def test_self_reference_allowlist_suppresses_and_flag_enables_deterministically(
    tmp_path: Path,
) -> None:
    ordinary_path = "src/project/strings.py"
    _write(tmp_path, ordinary_path, "MESSAGE = 'TODO marker in normal project file'\n")
    for rel_path in placeholder_audit.DEFAULT_SELF_REFERENCE_ALLOWLIST:
        _write(tmp_path, rel_path, "PATTERN = 'TODO marker used by audit tooling'\n")

    suppressed = _scan(tmp_path, warn_on_string_markers=True)
    suppressed_string_paths = {
        item.path for item in suppressed.findings if item.kind == "placeholder_string_literal"
    }

    assert suppressed.error_count == 0
    assert ordinary_path in suppressed_string_paths
    for rel_path in placeholder_audit.DEFAULT_SELF_REFERENCE_ALLOWLIST:
        assert rel_path not in suppressed_string_paths

    first = _scan(
        tmp_path,
        warn_on_string_markers=True,
        warn_on_audit_tool_self_references=True,
    )
    second = _scan(
        tmp_path,
        warn_on_string_markers=True,
        warn_on_audit_tool_self_references=True,
    )

    first_string_findings = [
        item for item in first.findings if item.kind == "placeholder_string_literal"
    ]
    second_string_findings = [
        item for item in second.findings if item.kind == "placeholder_string_literal"
    ]
    first_paths = {item.path for item in first_string_findings}
    second_paths = {item.path for item in second_string_findings}

    expected_paths = {
        ordinary_path,
        *placeholder_audit.DEFAULT_SELF_REFERENCE_ALLOWLIST,
    }
    assert first_paths == expected_paths
    assert second_paths == expected_paths
    assert all(item.severity == "WARNING" for item in first_string_findings)
    assert [item.to_dict() for item in first.findings] == [
        item.to_dict() for item in second.findings
    ]
    assert placeholder_audit.format_text(first) == placeholder_audit.format_text(second)


def test_tests_meta_is_excluded_by_default_and_included_when_requested(tmp_path: Path) -> None:
    _write(tmp_path, "docs/guide.md", "# TODO: docs token\n")
    _write(tmp_path, "tests/meta/fixture.py", "# TODO: meta fixture token\n")

    default_result = _scan(
        tmp_path,
        roots=("docs", "tests"),
        exclude=placeholder_audit.DEFAULT_EXCLUDE,
    )
    assert any(item.path == "docs/guide.md" for item in default_result.findings)
    assert not any(item.path.startswith("tests/meta/") for item in default_result.findings)

    included_result = _scan(tmp_path, roots=("docs", "tests"), exclude=())
    assert any(item.path.startswith("tests/meta/") for item in included_result.findings)


def test_two_runs_produce_identical_findings_and_text_output(tmp_path: Path) -> None:
    _write(tmp_path, "src/z_last.py", "# TODO: z\n")
    _write(tmp_path, "src/a_first.py", "PATTERN = 'TODO marker'\n")
    _write(tmp_path, "src/m_mid.py", "def x() -> None:\n    raise NotImplementedError\n")
    _write(tmp_path, "src/b_mid.py", "def y() -> None:\n    pass\n")

    first = _scan(tmp_path, warn_on_string_markers=True)
    second = _scan(tmp_path, warn_on_string_markers=True)

    first_keys = [item.sort_key() for item in first.findings]
    second_keys = [item.sort_key() for item in second.findings]

    assert [item.to_dict() for item in first.findings] == [
        item.to_dict() for item in second.findings
    ]
    assert first_keys == second_keys
    assert first_keys == sorted(first_keys)
    assert placeholder_audit.format_text(first) == placeholder_audit.format_text(second)
    assert placeholder_audit.format_json(first) == placeholder_audit.format_json(second)


def test_text_and_json_output_include_required_audit_fields(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/output_case.py",
        (
            "# TODO: keep for audit\n"
            "def build() -> None:\n"
            "    return None\n\n"
            "MARKER = 'TODO in string'\n"
        ),
    )

    result = _scan(tmp_path, show_context=2, warn_on_string_markers=True)
    text_output = placeholder_audit.format_text(result)

    assert "ERRORS (fail build)" in text_output
    assert "WARNINGS (review)" in text_output
    assert re.search(r"src/output_case.py:\d+:\d+ \[[^\]]+\]", text_output)
    assert "ast: class=" in text_output
    assert ">" in text_output

    payload = json.loads(placeholder_audit.format_json(result))
    assert isinstance(payload["findings"], list)
    assert payload["summary"]["error_count"] >= 1
    assert payload["summary"]["warning_count"] >= 1

    first_finding = payload["findings"][0]
    assert first_finding["severity"] in {"error", "warning"}
    assert {"kind", "path", "line", "col", "context_lines", "ast_context"}.issubset(first_finding)

    ast_context = first_finding["ast_context"]
    assert {"class", "function", "decorators", "in_type_checking", "in_protocol"}.issubset(
        ast_context
    )


def test_cli_wires_string_marker_and_self_reference_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning_root = tmp_path / "warning_repo"
    rel_path = "src/tooling/string_markers.py"
    _write(warning_root, rel_path, "PATTERN = 'TODO in string'\n")

    monkeypatch.chdir(warning_root)
    assert placeholder_audit.main(["--roots", "src", "--format", "text"]) == 0
    assert (
        placeholder_audit.main(
            [
                "--roots",
                "src",
                "--format",
                "text",
                "--warn-on-string-markers",
                "--self-reference-allowlist",
                rel_path,
                "--fail-on-warn",
            ]
        )
        == 0
    )
    assert (
        placeholder_audit.main(
            [
                "--roots",
                "src",
                "--format",
                "text",
                "--warn-on-string-markers",
                "--warn-on-audit-tool-self-references",
                "--self-reference-allowlist",
                rel_path,
                "--fail-on-warn",
            ]
        )
        == 1
    )

    error_root = tmp_path / "error_repo"
    _write(error_root, "src/error_only.py", "# TODO: fail\n")
    monkeypatch.chdir(error_root)
    assert placeholder_audit.main(["--roots", "src", "--format", "json"]) == 1


if HYPOTHESIS_AVAILABLE:

    @settings(max_examples=30, derandomize=True, deadline=None)
    @seed(20260213)
    @given(keyword=st.sampled_from(("TODO", "FIXME")))
    def test_property_string_keywords_default_suppressed_then_warn_when_enabled(
        keyword: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write(
                root,
                "src/fuzz.py",
                (f"def render() -> str:\n    return 'prefix {keyword} suffix'\n"),
            )

            default_result = _scan(root)
            warned_result = _scan(root, warn_on_string_markers=True)
        assert default_result.error_count == 0
        assert not any(
            item.kind == "placeholder_string_literal" for item in default_result.findings
        )

        string_findings = [
            item for item in warned_result.findings if item.kind == "placeholder_string_literal"
        ]
        assert warned_result.error_count == 0
        assert len(string_findings) == 1
        assert string_findings[0].severity == "WARNING"

else:

    def test_seeded_string_keywords_default_suppressed_then_warn_when_enabled(
        tmp_path: Path,
    ) -> None:
        rng = Random(20260213)
        keywords = ("TODO", "FIXME")

        for _ in range(40):
            keyword = keywords[rng.randrange(0, len(keywords))]
            _write(
                tmp_path,
                "src/fuzz.py",
                (f"def render() -> str:\n    return 'prefix {keyword} suffix'\n"),
            )

            default_result = _scan(tmp_path)
            warned_result = _scan(tmp_path, warn_on_string_markers=True)
            assert default_result.error_count == 0
            assert not any(
                item.kind == "placeholder_string_literal" for item in default_result.findings
            )

            string_findings = [
                item for item in warned_result.findings if item.kind == "placeholder_string_literal"
            ]
            assert warned_result.error_count == 0
            assert len(string_findings) == 1
            assert string_findings[0].severity == "WARNING"
