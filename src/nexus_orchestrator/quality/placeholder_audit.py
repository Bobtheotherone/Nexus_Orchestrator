"""
nexus-orchestrator — certainty-aware placeholder audit utility

File: src/nexus_orchestrator/quality/placeholder_audit.py
Last updated: 2026-02-13

Purpose
- Provide deterministic semantic placeholder scanning for CI/local audits with
  explicit certainty classes (ERROR vs WARNING).

What should be included in this file
- Token + AST analysis that avoids false positives from string data.
- Deterministic finding ordering and stable serialization.
- Text/JSON formatters with audit-friendly context and AST scope metadata.
- CLI entrypoint with deterministic exit codes.

Functional requirements
- Exit code `0` when no ERROR findings exist.
- Exit code `1` when ERROR findings exist (or warnings with `--fail-on-warn`).
- Exit code `2` for internal scanner/runtime failures.

Non-functional requirements
- Offline only.
- Do not emit secret-bearing data beyond local source snippets under review.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import json
import os
import re
import sys
import tokenize
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_ROOTS: tuple[str, ...] = ("src", "scripts")
DEFAULT_EXCLUDE: tuple[str, ...] = ("tests/meta",)
DEFAULT_SEVERITY_BY_PATH: tuple[tuple[str, str], ...] = (
    ("src/**", "ERROR"),
    ("scripts/**", "ERROR"),
    ("tests/**", "WARNING"),
    ("docs/**", "WARNING"),
    ("**", "WARNING"),
)
DEFAULT_SELF_REFERENCE_ALLOWLIST: tuple[str, ...] = (
    "src/nexus_orchestrator/quality/placeholder_audit.py",
    "src/nexus_orchestrator/repo_blueprint.py",
)

_ALWAYS_IGNORED_DIRS = frozenset(
    {
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        ".cache",
    }
)

_TODO_FIXME_RE = re.compile(r"\b(?:TODO|FIXME)\b", re.IGNORECASE)
_STRING_PLACEHOLDER_RE = re.compile(r"\b(?:TODO|FIXME)\b", re.IGNORECASE)
_NON_PY_COMMENT_PREFIXES: dict[str, tuple[str, ...]] = {
    ".cfg": ("#", ";", "//"),
    ".conf": ("#", ";", "//"),
    ".env": ("#", ";"),
    ".ini": ("#", ";"),
    ".jsonc": ("//", "/*", "*"),
    ".md": ("<!--", "#", "-", "*"),
    ".rst": ("..",),
    ".sh": ("#",),
    ".toml": ("#",),
    ".txt": ("#", ";"),
    ".yaml": ("#",),
    ".yml": ("#",),
}

_SEVERITY_ORDER = {"ERROR": 0, "WARNING": 1}
_SEVERITY_PATTERN_DELIMITERS: tuple[str, ...] = ("=", ":")

SeverityRule = tuple[str, str]
SeverityByPathInput = Mapping[str, str] | Sequence[str] | Sequence[SeverityRule]


@dataclass(frozen=True, slots=True)
class AstContext:
    class_name: str | None
    function_name: str | None
    decorators: tuple[str, ...]
    in_type_checking: bool
    in_protocol: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "class": self.class_name,
            "function": self.function_name,
            "decorators": list(self.decorators),
            "in_type_checking": self.in_type_checking,
            "in_protocol": self.in_protocol,
        }


@dataclass(frozen=True, slots=True)
class ContextLine:
    line: int
    text: str

    def to_dict(self) -> dict[str, object]:
        return {"line": self.line, "text": self.text}


@dataclass(frozen=True, slots=True)
class Finding:
    severity: str
    kind: str
    path: str
    line: int
    col: int
    snippet: str
    reason: str
    context_lines: tuple[ContextLine, ...]
    ast_context: AstContext

    def sort_key(self) -> tuple[int, str, int, int, str]:
        return (
            _SEVERITY_ORDER.get(self.severity, 99),
            self.path,
            self.line,
            self.col,
            self.kind,
        )

    def dedupe_key(self) -> tuple[str, str, str, int, int, str, str]:
        return (
            self.severity,
            self.kind,
            self.path,
            self.line,
            self.col,
            self.reason,
            self.snippet,
        )

    def to_dict(self) -> dict[str, object]:
        context_payload = [item.to_dict() for item in self.context_lines]
        return {
            "severity": self.severity.lower(),
            "kind": self.kind,
            "path": self.path,
            "line": self.line,
            "col": self.col,
            "snippet": self.snippet,
            "reason": self.reason,
            "context_lines": context_payload,
            "context": context_payload,
            "ast_context": self.ast_context.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AuditResult:
    findings: tuple[Finding, ...]
    scanned_files: tuple[str, ...]
    roots: tuple[str, ...]
    exclude: tuple[str, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for item in self.findings if item.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.findings if item.severity == "WARNING")

    def summary(self) -> dict[str, object]:
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "total_findings": len(self.findings),
            "scanned_files": len(self.scanned_files),
            "roots": list(self.roots),
            "exclude": list(self.exclude),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "findings": [item.to_dict() for item in self.findings],
            "scanned_files": list(self.scanned_files),
        }


@dataclass(frozen=True, slots=True)
class _TraversalState:
    class_name: str | None = None
    function_name: str | None = None
    decorators: tuple[str, ...] = ()
    in_type_checking: bool = False
    in_protocol: bool = False

    def to_public(self) -> AstContext:
        return AstContext(
            class_name=self.class_name,
            function_name=self.function_name,
            decorators=self.decorators,
            in_type_checking=self.in_type_checking,
            in_protocol=self.in_protocol,
        )


@dataclass(frozen=True, slots=True)
class _LineContextRecord:
    depth: int
    state: _TraversalState


class _AstAnalyzer:
    def __init__(
        self,
        *,
        rel_path: str,
        lines: Sequence[str],
        context_radius: int,
        severity_by_path: Sequence[tuple[str, str]],
    ) -> None:
        self._rel_path = rel_path
        self._lines = lines
        self._context_radius = context_radius
        self._severity_by_path = severity_by_path
        self._protocol_class_names: set[str] = set()
        self.findings: list[Finding] = []
        self._line_context: dict[int, _LineContextRecord] = {}

    def analyze(self, module: ast.Module) -> tuple[list[Finding], dict[int, AstContext]]:
        base_state = _TraversalState()
        for line_no in range(1, len(self._lines) + 1):
            self._line_context[line_no] = _LineContextRecord(depth=0, state=base_state)

        self._check_bare_ellipsis_body(module.body, base_state)
        self._visit_stmt_list(module.body, base_state, depth=1)

        mapped = {
            line_no: record.state.to_public()
            for line_no, record in sorted(self._line_context.items())
        }
        return self.findings, mapped

    def _visit_stmt_list(
        self,
        statements: Sequence[ast.stmt],
        state: _TraversalState,
        *,
        depth: int,
    ) -> None:
        for statement in statements:
            self._visit_stmt(statement, state, depth=depth)

    def _visit_stmt(self, statement: ast.stmt, state: _TraversalState, *, depth: int) -> None:
        self._mark_node(statement, state, depth)

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(statement, state, depth=depth)
            return

        if isinstance(statement, ast.ClassDef):
            self._visit_class(statement, state, depth=depth)
            return

        if isinstance(statement, ast.If):
            self._visit_if(statement, state, depth=depth)
            return

        if isinstance(statement, ast.Try):
            self._visit_try(statement, state, depth=depth)
            return

        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            self._check_bare_ellipsis_body(statement.body, state)
            self._check_bare_ellipsis_body(statement.orelse, state)
            self._visit_stmt_list(statement.body, state, depth=depth + 1)
            self._visit_stmt_list(statement.orelse, state, depth=depth + 1)
            return

        if isinstance(statement, (ast.With, ast.AsyncWith)):
            self._check_bare_ellipsis_body(statement.body, state)
            self._visit_stmt_list(statement.body, state, depth=depth + 1)
            return

        if isinstance(statement, ast.Match):
            for case in statement.cases:
                self._check_bare_ellipsis_body(case.body, state)
                self._visit_stmt_list(case.body, state, depth=depth + 1)
            return

        if isinstance(statement, ast.Raise):
            self._check_not_implemented_raise(statement, state)

    def _visit_function(
        self,
        statement: ast.FunctionDef | ast.AsyncFunctionDef,
        state: _TraversalState,
        *,
        depth: int,
    ) -> None:
        decorators = _decorator_names(statement.decorator_list)
        is_abstract = state.in_protocol or _has_terminal_decorator(decorators, "abstractmethod")
        is_overload = _has_terminal_decorator(decorators, "overload")
        function_state = _TraversalState(
            class_name=state.class_name,
            function_name=statement.name,
            decorators=decorators,
            in_type_checking=state.in_type_checking,
            in_protocol=state.in_protocol,
        )

        self._mark_node(statement, function_state, depth + 1)

        body = _strip_optional_docstring(statement.body)
        if len(body) == 1:
            only_statement = body[0]
            if isinstance(only_statement, ast.Pass):
                if not is_abstract and not function_state.in_protocol:
                    self._append_finding(
                        severity="ERROR",
                        kind="pass_only_function_body",
                        line=only_statement.lineno,
                        col=only_statement.col_offset + 1,
                        reason=(
                            "Function/method body is only `pass` in non-abstract, "
                            "non-Protocol code."
                        ),
                        state=function_state,
                    )
            elif _is_bare_ellipsis_statement(only_statement):
                if (
                    function_state.in_type_checking
                    or function_state.in_protocol
                    or is_abstract
                    or is_overload
                ):
                    self._append_finding(
                        severity="WARNING",
                        kind="ellipsis_only_function_body",
                        line=only_statement.lineno,
                        col=only_statement.col_offset + 1,
                        reason=(
                            "Function/method body is only `...` in abstract/Protocol/"
                            "overload/TYPE_CHECKING context."
                        ),
                        state=function_state,
                    )
                else:
                    self._append_finding(
                        severity="ERROR",
                        kind="ellipsis_only_function_body",
                        line=only_statement.lineno,
                        col=only_statement.col_offset + 1,
                        reason=(
                            "Function/method body is only `...` in executable non-abstract code."
                        ),
                        state=function_state,
                    )

        self._visit_stmt_list(statement.body, function_state, depth=depth + 1)

    def _visit_class(self, statement: ast.ClassDef, state: _TraversalState, *, depth: int) -> None:
        decorators = _decorator_names(statement.decorator_list)
        is_protocol = state.in_protocol or _class_declares_protocol(
            statement,
            known_protocols=self._protocol_class_names,
        )
        if is_protocol:
            self._protocol_class_names.add(statement.name)

        class_state = _TraversalState(
            class_name=statement.name,
            function_name=state.function_name,
            decorators=decorators,
            in_type_checking=state.in_type_checking,
            in_protocol=is_protocol,
        )

        self._mark_node(statement, class_state, depth + 1)

        body = _strip_optional_docstring(statement.body)
        if len(body) == 1:
            only_statement = body[0]
            if isinstance(only_statement, ast.Pass):
                if _is_exception_subclass(statement) or _is_marker_class(statement):
                    self._append_finding(
                        severity="WARNING",
                        kind="pass_only_empty_class",
                        line=only_statement.lineno,
                        col=only_statement.col_offset + 1,
                        reason="Empty Exception/marker class body uses `pass`.",
                        state=class_state,
                    )
            elif _is_bare_ellipsis_statement(only_statement):
                if class_state.in_type_checking or class_state.in_protocol:
                    self._append_finding(
                        severity="WARNING",
                        kind="ellipsis_only_class_body",
                        line=only_statement.lineno,
                        col=only_statement.col_offset + 1,
                        reason="Class body is only `...` inside Protocol/TYPE_CHECKING context.",
                        state=class_state,
                    )
                else:
                    self._append_finding(
                        severity="ERROR",
                        kind="ellipsis_only_class_body",
                        line=only_statement.lineno,
                        col=only_statement.col_offset + 1,
                        reason="Class body is only `...` in executable context.",
                        state=class_state,
                    )

        self._visit_stmt_list(statement.body, class_state, depth=depth + 1)

    def _visit_if(self, statement: ast.If, state: _TraversalState, *, depth: int) -> None:
        in_type_checking = _is_type_checking_test(statement.test)
        body_state = state

        if in_type_checking:
            body_state = _TraversalState(
                class_name=state.class_name,
                function_name=state.function_name,
                decorators=state.decorators,
                in_type_checking=True,
                in_protocol=state.in_protocol,
            )
            self._mark_statement_block(statement.body, body_state, depth + 1)

        self._check_bare_ellipsis_body(statement.body, body_state)
        self._check_bare_ellipsis_body(statement.orelse, state)

        self._visit_stmt_list(statement.body, body_state, depth=depth + 1)
        self._visit_stmt_list(statement.orelse, state, depth=depth + 1)

    def _visit_try(self, statement: ast.Try, state: _TraversalState, *, depth: int) -> None:
        self._check_bare_ellipsis_body(statement.body, state)
        self._check_bare_ellipsis_body(statement.orelse, state)
        self._check_bare_ellipsis_body(statement.finalbody, state)

        self._visit_stmt_list(statement.body, state, depth=depth + 1)
        self._visit_stmt_list(statement.orelse, state, depth=depth + 1)
        self._visit_stmt_list(statement.finalbody, state, depth=depth + 1)

        for handler in statement.handlers:
            self._mark_node(handler, state, depth + 1)
            self._check_bare_ellipsis_body(handler.body, state)
            self._visit_stmt_list(handler.body, state, depth=depth + 2)

    def _check_not_implemented_raise(self, statement: ast.Raise, state: _TraversalState) -> None:
        if state.in_type_checking:
            return
        if _is_not_implemented_error_expr(statement.exc):
            self._append_finding(
                severity="ERROR",
                kind="raise_not_implemented_error",
                line=statement.lineno,
                col=statement.col_offset + 1,
                reason="Executable `raise NotImplementedError` detected.",
                state=state,
            )

    def _check_bare_ellipsis_body(
        self, statements: Sequence[ast.stmt], state: _TraversalState
    ) -> None:
        body = _strip_optional_docstring(statements)
        if len(body) != 1:
            return
        only_statement = body[0]
        if not _is_bare_ellipsis_statement(only_statement):
            return

        if state.in_type_checking or state.in_protocol:
            self._append_finding(
                severity="WARNING",
                kind="ellipsis_only_body",
                line=only_statement.lineno,
                col=only_statement.col_offset + 1,
                reason="Bare `...` body found in Protocol/TYPE_CHECKING context.",
                state=state,
            )
        else:
            self._append_finding(
                severity="ERROR",
                kind="ellipsis_only_body",
                line=only_statement.lineno,
                col=only_statement.col_offset + 1,
                reason="Bare `...` body found in executable context.",
                state=state,
            )

    def _append_finding(
        self,
        *,
        severity: str,
        kind: str,
        line: int,
        col: int,
        reason: str,
        state: _TraversalState,
    ) -> None:
        effective_severity = _apply_path_severity_policy(
            rel_path=self._rel_path,
            base_severity=severity,
            severity_by_path=self._severity_by_path,
        )
        self.findings.append(
            _make_finding(
                severity=effective_severity,
                kind=kind,
                path=self._rel_path,
                line=line,
                col=col,
                reason=reason,
                lines=self._lines,
                context_radius=self._context_radius,
                state=state.to_public(),
            )
        )

    def _mark_node(self, node: ast.AST, state: _TraversalState, depth: int) -> None:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", start)
        if start is None or end is None:
            return

        for line_no in range(start, end + 1):
            existing = self._line_context.get(line_no)
            if existing is None or depth >= existing.depth:
                self._line_context[line_no] = _LineContextRecord(depth=depth, state=state)

    def _mark_statement_block(
        self,
        statements: Sequence[ast.stmt],
        state: _TraversalState,
        depth: int,
    ) -> None:
        if not statements:
            return

        start = getattr(statements[0], "lineno", None)
        end = getattr(statements[-1], "end_lineno", getattr(statements[-1], "lineno", None))
        if start is None or end is None:
            return

        for line_no in range(start, end + 1):
            existing = self._line_context.get(line_no)
            if existing is None or depth >= existing.depth:
                self._line_context[line_no] = _LineContextRecord(depth=depth, state=state)


def run_placeholder_audit(
    *,
    repo_root: Path | None = None,
    roots: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    show_context: int = 2,
    include_docstrings: bool = False,
    include_string_literals: bool = False,
    scan_tests_as_blocking: bool = False,
    severity_by_path: SeverityByPathInput | None = None,
    warn_on_string_markers: bool = False,
    warn_on_audit_tool_self_references: bool = False,
    self_reference_allowlist: tuple[str, ...] = DEFAULT_SELF_REFERENCE_ALLOWLIST,
) -> AuditResult:
    root = (repo_root or Path.cwd()).resolve()
    normalized_roots = _normalize_inputs(DEFAULT_ROOTS if roots is None else roots)
    normalized_exclude = _normalize_inputs(DEFAULT_EXCLUDE if exclude is None else exclude)
    normalized_self_reference_allowlist = _normalize_inputs(self_reference_allowlist)
    normalized_severity_by_path = _normalize_severity_by_path(
        DEFAULT_SEVERITY_BY_PATH if severity_by_path is None else severity_by_path,
        scan_tests_as_blocking=scan_tests_as_blocking,
    )
    context_radius = max(show_context, 0)
    include_string_literals_effective = include_string_literals or warn_on_string_markers

    files = _collect_files(root, roots=normalized_roots, exclude=normalized_exclude)

    findings: list[Finding] = []
    scanned: list[str] = []
    for rel_path in files:
        absolute_path = root / rel_path
        if not absolute_path.is_file():
            continue

        text = absolute_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        scanned.append(rel_path)

        if absolute_path.suffix == ".py":
            findings.extend(
                _scan_python_file(
                    rel_path=rel_path,
                    text=text,
                    lines=lines,
                    context_radius=context_radius,
                    include_docstrings=include_docstrings,
                    include_string_literals=include_string_literals_effective,
                    severity_by_path=normalized_severity_by_path,
                    warn_on_audit_tool_self_references=warn_on_audit_tool_self_references,
                    self_reference_allowlist=normalized_self_reference_allowlist,
                )
            )
            continue

        findings.extend(
            _scan_non_python_file(
                rel_path=rel_path,
                lines=lines,
                context_radius=context_radius,
                severity_by_path=normalized_severity_by_path,
            )
        )

    deduped: dict[tuple[str, str, str, int, int, str, str], Finding] = {}
    for finding in findings:
        deduped[finding.dedupe_key()] = finding

    ordered_findings = tuple(sorted(deduped.values(), key=lambda item: item.sort_key()))
    return AuditResult(
        findings=ordered_findings,
        scanned_files=tuple(sorted(scanned)),
        roots=normalized_roots,
        exclude=normalized_exclude,
    )


def format_text(result: AuditResult) -> str:
    errors = [item for item in result.findings if item.severity == "ERROR"]
    warnings = [item for item in result.findings if item.severity == "WARNING"]

    output_lines: list[str] = []
    output_lines.extend(_format_section("ERRORS (fail build)", errors))
    output_lines.append("")
    output_lines.extend(_format_section("WARNINGS (review)", warnings))
    output_lines.append("")
    summary = result.summary()
    output_lines.append(
        "Summary: "
        f"errors={summary['error_count']} "
        f"warnings={summary['warning_count']} "
        f"total={summary['total_findings']} "
        f"scanned_files={summary['scanned_files']}"
    )

    return "\n".join(output_lines).rstrip() + "\n"


def format_json(result: AuditResult) -> str:
    return json.dumps(result.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        repo_root = Path(args.repo_root).resolve()
        result = run_placeholder_audit(
            repo_root=repo_root,
            roots=args.roots,
            exclude=args.exclude,
            show_context=args.show_context,
            include_docstrings=args.include_docstrings,
            include_string_literals=args.include_string_literals,
            scan_tests_as_blocking=args.scan_tests_as_blocking,
            severity_by_path=args.severity_by_path,
            warn_on_string_markers=args.warn_on_string_markers,
            warn_on_audit_tool_self_references=args.warn_on_audit_tool_self_references,
            self_reference_allowlist=tuple(args.self_reference_allowlist),
        )

        if args.output_format == "json":
            sys.stdout.write(format_json(result))
        else:
            sys.stdout.write(format_text(result))

        should_fail = result.error_count > 0 or (args.fail_on_warn and result.warning_count > 0)
        return 1 if should_fail else 0
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        sys.stderr.write(f"placeholder audit crashed: {exc}\n")
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit placeholder signals in source trees.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository/workspace root to scan (default: current working directory).",
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=list(DEFAULT_ROOTS),
        help="Root paths to scan (default: src scripts).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=list(DEFAULT_EXCLUDE),
        help="Relative paths to exclude (default: tests/meta).",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return non-zero when warnings exist.",
    )
    parser.add_argument(
        "--show-context",
        type=int,
        default=2,
        help="Context lines before/after each finding.",
    )
    parser.add_argument(
        "--include-string-literals",
        action="store_true",
        help="Include TODO/FIXME markers found in string literals (non-blocking by default).",
    )
    parser.add_argument(
        "--include-docstrings",
        action="store_true",
        help=(
            "When string-literal scanning is enabled, include docstring literals in marker checks."
        ),
    )
    parser.add_argument(
        "--scan-tests-as-blocking",
        action="store_true",
        help="Do not demote tests/** blocking findings to WARNING in path-severity policy.",
    )
    parser.add_argument(
        "--severity-by-path",
        action="append",
        default=None,
        metavar="GLOB=SEVERITY",
        help=(
            "Path-severity rule (repeatable), e.g. src/**=ERROR tests/**=WARNING. "
            "Order is first-match and defaults to src/scripts ERROR, tests/docs WARNING."
        ),
    )
    parser.add_argument(
        "--warn-on-string-markers",
        action="store_true",
        help=(
            "Compatibility alias for --include-string-literals; emits non-blocking string-marker"
            " findings."
        ),
    )
    parser.add_argument(
        "--warn-on-audit-tool-self-references",
        action="store_true",
        help=(
            "When string marker warnings are enabled, include allowlisted audit-tool "
            "self-reference files."
        ),
    )
    parser.add_argument(
        "--self-reference-allowlist",
        nargs="+",
        default=list(DEFAULT_SELF_REFERENCE_ALLOWLIST),
        help=(
            "Relative file paths treated as audit-tool self-references for string markers "
            "(default: placeholder audit + repo blueprint files)."
        ),
    )
    return parser


def _scan_python_file(
    *,
    rel_path: str,
    text: str,
    lines: Sequence[str],
    context_radius: int,
    include_docstrings: bool,
    include_string_literals: bool,
    severity_by_path: Sequence[tuple[str, str]],
    warn_on_audit_tool_self_references: bool,
    self_reference_allowlist: tuple[str, ...],
) -> list[Finding]:
    default_state = AstContext(
        class_name=None,
        function_name=None,
        decorators=(),
        in_type_checking=False,
        in_protocol=False,
    )
    line_context: dict[int, AstContext] = {
        line_no: default_state for line_no in range(1, len(lines) + 1)
    }

    ast_findings: list[Finding] = []
    docstring_starts: set[tuple[int, int]] = set()
    try:
        module = ast.parse(text)
    except SyntaxError:
        module = None

    if module is not None:
        analyzer = _AstAnalyzer(
            rel_path=rel_path,
            lines=lines,
            context_radius=context_radius,
            severity_by_path=severity_by_path,
        )
        ast_findings, line_context = analyzer.analyze(module)
        docstring_starts = _collect_docstring_starts(module)

    token_findings = _scan_tokens(
        rel_path=rel_path,
        text=text,
        lines=lines,
        context_radius=context_radius,
        line_context=line_context,
        include_docstrings=include_docstrings,
        include_string_literals=include_string_literals,
        severity_by_path=severity_by_path,
        warn_on_audit_tool_self_references=warn_on_audit_tool_self_references,
        self_reference_allowlist=self_reference_allowlist,
        docstring_starts=docstring_starts,
    )
    return ast_findings + token_findings


def _scan_tokens(
    *,
    rel_path: str,
    text: str,
    lines: Sequence[str],
    context_radius: int,
    line_context: dict[int, AstContext],
    include_docstrings: bool,
    include_string_literals: bool,
    severity_by_path: Sequence[tuple[str, str]],
    warn_on_audit_tool_self_references: bool,
    self_reference_allowlist: tuple[str, ...],
    docstring_starts: set[tuple[int, int]],
) -> list[Finding]:
    findings: list[Finding] = []

    try:
        stream = io.StringIO(text)
        tokens = tokenize.generate_tokens(stream.readline)
        for token in tokens:
            line, col = token.start
            if token.type == tokenize.COMMENT:
                if _TODO_FIXME_RE.search(token.string) is None:
                    continue
                severity = _apply_path_severity_policy(
                    rel_path=rel_path,
                    base_severity="ERROR",
                    severity_by_path=severity_by_path,
                )
                reason = (
                    "TODO/FIXME comment in blocking path."
                    if severity == "ERROR"
                    else "TODO/FIXME comment in non-blocking path."
                )
                findings.append(
                    _make_finding(
                        severity=severity,
                        kind="todo_fixme_comment",
                        path=rel_path,
                        line=line,
                        col=col + 1,
                        reason=reason,
                        lines=lines,
                        context_radius=context_radius,
                        state=_context_for_line(line_context, line),
                    )
                )
                continue

            if token.type == tokenize.STRING:
                if not include_string_literals:
                    continue
                if _is_docstring_token(token.start, docstring_starts) and not include_docstrings:
                    continue
                if _STRING_PLACEHOLDER_RE.search(token.string) is None:
                    continue
                if not warn_on_audit_tool_self_references and rel_path in self_reference_allowlist:
                    continue
                findings.append(
                    _make_finding(
                        severity=_apply_path_severity_policy(
                            rel_path=rel_path,
                            base_severity="WARNING",
                            severity_by_path=severity_by_path,
                        ),
                        kind="placeholder_string_literal",
                        path=rel_path,
                        line=line,
                        col=col + 1,
                        reason="String literal contains TODO/FIXME marker.",
                        lines=lines,
                        context_radius=context_radius,
                        state=_context_for_line(line_context, line),
                    )
                )
    except (SyntaxError, tokenize.TokenError):
        return findings

    return findings


def _scan_non_python_file(
    *,
    rel_path: str,
    lines: Sequence[str],
    context_radius: int,
    severity_by_path: Sequence[SeverityRule],
) -> list[Finding]:
    prefixes = _non_python_comment_prefixes(rel_path)
    if not prefixes:
        return []

    findings: list[Finding] = []
    for line_no, line_text in enumerate(lines, start=1):
        if _TODO_FIXME_RE.search(line_text) is None:
            continue
        if not _non_python_line_matches_comment_marker(line_text=line_text, prefixes=prefixes):
            continue

        findings.append(
            _make_finding(
                severity=_apply_path_severity_policy(
                    rel_path=rel_path,
                    base_severity="WARNING",
                    severity_by_path=severity_by_path,
                ),
                kind="todo_fixme_non_python_comment",
                path=rel_path,
                line=line_no,
                col=1,
                reason="TODO/FIXME comment-like marker in non-Python file.",
                lines=lines,
                context_radius=context_radius,
                state=AstContext(
                    class_name=None,
                    function_name=None,
                    decorators=(),
                    in_type_checking=False,
                    in_protocol=False,
                ),
            )
        )

    return findings


def _collect_files(repo_root: Path, *, roots: Sequence[str], exclude: Sequence[str]) -> list[str]:
    discovered: set[str] = set()

    for root in roots:
        base = repo_root if root == "" else (repo_root / root)
        if not base.exists():
            continue

        if base.is_file():
            rel = _relative_posix(base, repo_root)
            if rel is None or _should_skip_path(rel, exclude):
                continue
            discovered.add(rel)
            continue

        for dirpath, dirnames, filenames in os.walk(base):
            current_dir = Path(dirpath)
            rel_dir = _relative_posix(current_dir, repo_root)
            if rel_dir is None:
                continue

            kept_dirs: list[str] = []
            for dirname in sorted(dirnames):
                candidate = f"{rel_dir}/{dirname}" if rel_dir else dirname
                if _should_skip_directory(dirname, candidate, exclude):
                    continue
                kept_dirs.append(dirname)
            dirnames[:] = kept_dirs

            for filename in sorted(filenames):
                file_path = current_dir / filename
                rel_file = _relative_posix(file_path, repo_root)
                if rel_file is None or _should_skip_path(rel_file, exclude):
                    continue
                discovered.add(rel_file)

    return sorted(discovered)


def _should_skip_directory(dirname: str, rel_dir: str, exclude: Sequence[str]) -> bool:
    lowered = dirname.lower()
    if dirname in _ALWAYS_IGNORED_DIRS:
        return True
    if lowered.endswith("cache") or lowered.endswith("_cache") or lowered == "caches":
        return True
    return _is_excluded(rel_dir, exclude)


def _should_skip_path(rel_path: str, exclude: Sequence[str]) -> bool:
    if _is_excluded(rel_path, exclude):
        return True

    parts = PurePosixPath(rel_path).parts
    for part in parts[:-1]:
        lowered = part.lower()
        if part in _ALWAYS_IGNORED_DIRS:
            return True
        if lowered.endswith("cache") or lowered.endswith("_cache") or lowered == "caches":
            return True

    return False


def _is_excluded(rel_path: str, exclude: Sequence[str]) -> bool:
    for excluded in exclude:
        if excluded == "":
            continue
        if rel_path == excluded or rel_path.startswith(f"{excluded}/"):
            return True
    return False


def _relative_posix(path: Path, repo_root: Path) -> str | None:
    try:
        relative = path.resolve(strict=False).relative_to(repo_root)
    except ValueError:
        return None
    return relative.as_posix()


def _normalize_inputs(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        candidate = _normalize_path(value)
        if candidate in normalized:
            continue
        normalized.append(candidate)
    return tuple(normalized)


def _normalize_path(value: str) -> str:
    raw = value.strip().replace("\\", "/")
    if raw in {"", "."}:
        return ""

    parts = [part for part in PurePosixPath(raw).parts if part not in {"", "."}]
    normalized = PurePosixPath(*parts).as_posix() if parts else ""
    if normalized.startswith("/"):
        normalized = normalized.lstrip("/")
    return normalized


def _format_section(title: str, findings: Sequence[Finding]) -> list[str]:
    output_lines = [title]
    if not findings:
        output_lines.append("(none)")
        return output_lines

    for finding in findings:
        output_lines.append(
            f"{finding.path}:{finding.line}:{finding.col} [{finding.kind}] {finding.reason}"
        )
        output_lines.append(f"  snippet: {finding.snippet}")
        output_lines.append(f"  ast: {_format_ast_context(finding.ast_context)}")
        output_lines.append("  context:")
        for context_line in finding.context_lines:
            marker = ">" if context_line.line == finding.line else " "
            output_lines.append(f"    {marker} {context_line.line:>5} | {context_line.text}")
    return output_lines


def _format_ast_context(state: AstContext) -> str:
    decorators = ",".join(state.decorators) if state.decorators else "-"
    class_name = state.class_name or "-"
    function_name = state.function_name or "-"
    return (
        f"class={class_name} function={function_name} decorators={decorators} "
        f"in_type_checking={state.in_type_checking} in_protocol={state.in_protocol}"
    )


def _make_finding(
    *,
    severity: str,
    kind: str,
    path: str,
    line: int,
    col: int,
    reason: str,
    lines: Sequence[str],
    context_radius: int,
    state: AstContext,
) -> Finding:
    snippet = _line_text(lines, line)
    context_lines = _build_context(lines, line, context_radius)
    return Finding(
        severity=severity,
        kind=kind,
        path=path,
        line=max(line, 1),
        col=max(col, 1),
        snippet=snippet,
        reason=reason,
        context_lines=context_lines,
        ast_context=state,
    )


def _build_context(lines: Sequence[str], line: int, context_radius: int) -> tuple[ContextLine, ...]:
    if not lines:
        return tuple()

    target = min(max(line, 1), len(lines))
    start = max(1, target - context_radius)
    end = min(len(lines), target + context_radius)
    return tuple(ContextLine(line=index, text=lines[index - 1]) for index in range(start, end + 1))


def _line_text(lines: Sequence[str], line: int) -> str:
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return ""


def _context_for_line(line_context: dict[int, AstContext], line: int) -> AstContext:
    return line_context.get(
        line,
        AstContext(
            class_name=None,
            function_name=None,
            decorators=(),
            in_type_checking=False,
            in_protocol=False,
        ),
    )


def _strip_optional_docstring(statements: Sequence[ast.stmt]) -> Sequence[ast.stmt]:
    if not statements:
        return statements

    if _leading_docstring_expression(statements) is not None:
        return statements[1:]
    return statements


def _is_bare_ellipsis_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr):
        return False
    value = statement.value
    if isinstance(value, ast.Constant):
        return value.value is Ellipsis
    return False


def _is_not_implemented_error_expr(expression: ast.expr | None) -> bool:
    if expression is None:
        return False
    if isinstance(expression, ast.Name):
        return expression.id == "NotImplementedError"
    if isinstance(expression, ast.Call):
        return _is_not_implemented_error_expr(expression.func)
    if isinstance(expression, ast.Attribute):
        return expression.attr == "NotImplementedError"
    return False


def _decorator_names(decorators: Sequence[ast.expr]) -> tuple[str, ...]:
    return tuple(_expr_name(decorator) for decorator in decorators)


def _has_terminal_decorator(decorators: Sequence[str], target: str) -> bool:
    return any(decorator.split(".")[-1] == target for decorator in decorators)


def _class_declares_protocol(statement: ast.ClassDef, *, known_protocols: set[str]) -> bool:
    for base in statement.bases:
        base_name = _expr_name(base)
        terminal = base_name.split(".")[-1]
        if terminal == "Protocol":
            return True
        if terminal in known_protocols:
            return True
    return False


def _is_exception_subclass(statement: ast.ClassDef) -> bool:
    for base in statement.bases:
        terminal = _expr_name(base).split(".")[-1]
        if terminal in {"Exception", "BaseException"}:
            return True
        if terminal.endswith("Error"):
            return True
    return False


def _is_marker_class(statement: ast.ClassDef) -> bool:
    if statement.decorator_list:
        return False

    if not statement.bases:
        return True

    base_terminals = {_expr_name(base).split(".")[-1] for base in statement.bases}
    return base_terminals.issubset({"object"})


def _expr_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        prefix = _expr_name(expression.value)
        return f"{prefix}.{expression.attr}" if prefix else expression.attr
    if isinstance(expression, ast.Call):
        return _expr_name(expression.func)

    try:
        return ast.unparse(expression)
    except Exception:  # pragma: no cover - defensive fallback
        return expression.__class__.__name__


def _is_type_checking_test(expression: ast.expr) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id == "TYPE_CHECKING"

    if isinstance(expression, ast.Attribute):
        return expression.attr == "TYPE_CHECKING"

    if isinstance(expression, ast.BoolOp):
        return any(_is_type_checking_test(value) for value in expression.values)

    if isinstance(expression, ast.UnaryOp):
        return _is_type_checking_test(expression.operand)

    if isinstance(expression, ast.Compare):
        return _is_type_checking_test(expression.left) or any(
            _is_type_checking_test(comparator) for comparator in expression.comparators
        )

    return False


def _is_docs_or_tests_path(rel_path: str) -> bool:
    parts = PurePosixPath(rel_path).parts
    if not parts:
        return False
    return parts[0] in {"docs", "tests"}


def _non_python_comment_prefixes(rel_path: str) -> tuple[str, ...]:
    suffix = PurePosixPath(rel_path).suffix.lower()
    if suffix in _NON_PY_COMMENT_PREFIXES:
        return _NON_PY_COMMENT_PREFIXES[suffix]
    if _is_docs_or_tests_path(rel_path):
        return ("#", ";", "//", "<!--", "-", "*", "..")
    return ()


def _non_python_line_matches_comment_marker(*, line_text: str, prefixes: Sequence[str]) -> bool:
    stripped = line_text.lstrip()
    if not stripped:
        return False
    return any(stripped.startswith(prefix) for prefix in prefixes)


def _collect_docstring_starts(module: ast.Module) -> set[tuple[int, int]]:
    starts: set[tuple[int, int]] = set()
    for node in ast.walk(module):
        statements: Sequence[ast.stmt] | None = None
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            statements = node.body

        if statements is None:
            continue

        docstring_expr = _leading_docstring_expression(statements)
        if docstring_expr is None:
            continue
        starts.add((docstring_expr.lineno, docstring_expr.col_offset))
    return starts


def _leading_docstring_expression(statements: Sequence[ast.stmt]) -> ast.Expr | None:
    if not statements:
        return None
    first = statements[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def _is_docstring_token(
    token_start: tuple[int, int], docstring_starts: set[tuple[int, int]]
) -> bool:
    return token_start in docstring_starts


def _normalize_severity_by_path(
    raw_rules: SeverityByPathInput,
    *,
    scan_tests_as_blocking: bool,
) -> tuple[SeverityRule, ...]:
    normalized_rules: list[SeverityRule] = []

    if isinstance(raw_rules, Mapping):
        for pattern_raw, severity_raw in raw_rules.items():
            normalized_rules.append(
                (_normalize_glob_pattern(pattern_raw), _normalize_severity(severity_raw))
            )
    else:
        for rule in raw_rules:
            if isinstance(rule, str):
                normalized_rules.append(_parse_severity_rule_spec(rule))
                continue

            if len(rule) != 2:
                raise ValueError(
                    "severity_by_path tuple entries must have exactly two items: (glob, severity)"
                )
            pattern_raw, severity_raw = rule
            if not isinstance(pattern_raw, str) or not isinstance(severity_raw, str):
                raise ValueError("severity_by_path tuple entries must be (str, str)")
            normalized_rules.append(
                (_normalize_glob_pattern(pattern_raw), _normalize_severity(severity_raw))
            )

    if scan_tests_as_blocking:
        normalized_rules.insert(0, ("tests/**", "ERROR"))

    if not normalized_rules:
        normalized_rules = [*DEFAULT_SEVERITY_BY_PATH]
    elif not any(pattern == "**" for pattern, _ in normalized_rules):
        normalized_rules.append(("**", "WARNING"))

    return tuple(normalized_rules)


def _parse_severity_rule_spec(spec: str) -> SeverityRule:
    raw = spec.strip()
    if not raw:
        raise ValueError("severity_by_path entries cannot be empty")

    for delimiter in _SEVERITY_PATTERN_DELIMITERS:
        if delimiter not in raw:
            continue
        pattern_raw, severity_raw = raw.split(delimiter, 1)
        return _normalize_glob_pattern(pattern_raw), _normalize_severity(severity_raw)

    parts = raw.split()
    if len(parts) == 2:
        pattern_raw, severity_raw = parts
        return _normalize_glob_pattern(pattern_raw), _normalize_severity(severity_raw)

    raise ValueError("severity_by_path entries must look like 'glob=SEVERITY' or 'glob SEVERITY'")


def _normalize_glob_pattern(value: str) -> str:
    normalized = _normalize_path(value)
    if normalized in {"", "."}:
        return "**"
    return normalized


def _normalize_severity(value: str) -> str:
    severity = value.strip().upper()
    if severity not in _SEVERITY_ORDER:
        raise ValueError(f"unsupported severity '{value}'. Expected one of: ERROR, WARNING")
    return severity


def _severity_for_path(rel_path: str, severity_by_path: Sequence[SeverityRule]) -> str:
    normalized_path = _normalize_path(rel_path)
    for pattern, severity in severity_by_path:
        if _path_matches_glob(normalized_path, pattern):
            return severity
    return "WARNING"


def _path_matches_glob(rel_path: str, pattern: str) -> bool:
    if pattern == "**":
        return True

    if not _contains_glob_magic(pattern):
        if rel_path == pattern:
            return True
        return rel_path.startswith(f"{pattern}/")

    return fnmatch.fnmatchcase(rel_path, pattern)


def _contains_glob_magic(pattern: str) -> bool:
    return any(symbol in pattern for symbol in ("*", "?", "["))


def _apply_path_severity_policy(
    *, rel_path: str, base_severity: str, severity_by_path: Sequence[SeverityRule]
) -> str:
    normalized_base = base_severity.strip().upper()
    if normalized_base not in _SEVERITY_ORDER:
        return normalized_base

    path_severity = _severity_for_path(rel_path, severity_by_path)
    if _SEVERITY_ORDER[path_severity] > _SEVERITY_ORDER[normalized_base]:
        return path_severity
    return normalized_base


__all__ = [
    "AuditResult",
    "AstContext",
    "ContextLine",
    "DEFAULT_EXCLUDE",
    "DEFAULT_ROOTS",
    "DEFAULT_SEVERITY_BY_PATH",
    "DEFAULT_SELF_REFERENCE_ALLOWLIST",
    "Finding",
    "format_json",
    "format_text",
    "main",
    "run_placeholder_audit",
]


if __name__ == "__main__":
    raise SystemExit(main())
