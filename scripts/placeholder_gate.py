"""
nexus-orchestrator — canonical placeholder gate wrapper

File: scripts/placeholder_gate.py
Last updated: 2026-02-13

Purpose
- Provide one canonical WSL-safe CLI for placeholder gating in local/CI runs.

What should be included in this file
- Deterministic command-line wrapper around `nexus_orchestrator.quality.placeholder_audit`.
- Stable text/json output and strict gate exit semantics.

Functional requirements
- Exit code `0` when no blocking findings exist.
- Exit code `1` when blocking findings exist (or warnings with `--fail-on-warn`).
- Exit code `2` for wrapper/runtime failures.

Non-functional requirements
- Offline only.
- No dependency on ripgrep/PCRE2.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
DEFAULT_PATHS: tuple[str, ...] = ("src", "tests")
DEFAULT_EXCLUDE: tuple[str, ...] = ("tests/meta",)


class _AuditResultProtocol(Protocol):
    findings: tuple[object, ...]
    scanned_files: tuple[str, ...]
    error_count: int
    warning_count: int


class _AuditFindingProtocol(Protocol):
    severity: str
    kind: str
    path: str
    line: int
    col: int
    snippet: str
    reason: str


class _PlaceholderAuditModule(Protocol):
    DEFAULT_SELF_REFERENCE_ALLOWLIST: tuple[str, ...]

    def run_placeholder_audit(self, **kwargs: object) -> _AuditResultProtocol: ...

    def format_json(self, result: _AuditResultProtocol) -> str: ...


def _load_module() -> _PlaceholderAuditModule:
    try:
        from nexus_orchestrator.quality import placeholder_audit as loaded_module

        return cast("_PlaceholderAuditModule", loaded_module)
    except ModuleNotFoundError:
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))
        from nexus_orchestrator.quality import placeholder_audit as loaded_module

        return cast("_PlaceholderAuditModule", loaded_module)


def _build_parser(default_self_reference_allowlist: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical placeholder gate (WSL-safe, deterministic, no PCRE2 dependency)."
    )
    parser.add_argument(
        "--root",
        "--repo-root",
        dest="repo_root",
        default=".",
        help="Repository/workspace root to scan (default: current working directory).",
    )
    parser.add_argument(
        "--paths",
        "--roots",
        dest="paths",
        nargs="+",
        default=list(DEFAULT_PATHS),
        help="Root paths to scan (default: src tests).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
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
        "--show-context",
        type=int,
        default=1,
        help="Context lines before/after each finding for text output internals.",
    )
    parser.add_argument(
        "--include-string-literals",
        action="store_true",
        help="Include TODO/FIXME string-literal markers as findings.",
    )
    parser.add_argument(
        "--include-docstrings",
        action="store_true",
        help="Include docstring markers when string-literal scanning is enabled.",
    )
    parser.add_argument(
        "--scan-tests-as-blocking",
        action="store_true",
        help="Do not demote tests/** blocking findings to warnings.",
    )
    parser.add_argument(
        "--severity-by-path",
        action="append",
        default=None,
        metavar="GLOB=SEVERITY",
        help="Path severity override, repeatable (e.g. src/**=ERROR tests/**=WARNING).",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Return exit code 1 when warnings exist.",
    )
    parser.add_argument(
        "--warn-on-audit-tool-self-references",
        action="store_true",
        help="Include allowlisted audit-tool self-reference files in string marker scanning.",
    )
    parser.add_argument(
        "--self-reference-allowlist",
        nargs="+",
        default=list(default_self_reference_allowlist),
        help="Relative paths allowlisted for audit-tool self-reference string marker checks.",
    )
    return parser


def _format_compact_text(
    *,
    result: _AuditResultProtocol,
    roots: Sequence[str],
    exclude: Sequence[str],
) -> str:
    root_text = ",".join(roots) if roots else "."
    exclude_text = ",".join(exclude) if exclude else "-"
    lines: list[str] = [
        (
            "placeholder-gate "
            f"roots={root_text} "
            f"exclude={exclude_text} "
            f"scanned_files={len(result.scanned_files)} "
            f"blocking_errors={result.error_count} "
            f"warnings={result.warning_count}"
        )
    ]

    for raw_finding in result.findings:
        finding = cast("_AuditFindingProtocol", raw_finding)
        snippet = finding.snippet.strip() if finding.snippet.strip() else "<blank>"
        lines.append(
            f"{finding.severity} {finding.path}:{finding.line}:{finding.col} "
            f"[{finding.kind}] {finding.reason} :: {snippet}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    module = _load_module()
    parser = _build_parser(module.DEFAULT_SELF_REFERENCE_ALLOWLIST)
    args = parser.parse_args(argv)

    try:
        run_result = module.run_placeholder_audit(
            repo_root=Path(cast("str", args.repo_root)).resolve(),
            roots=tuple(cast("list[str]", args.paths)),
            exclude=tuple(cast("list[str]", args.exclude)),
            show_context=max(int(args.show_context), 0),
            include_docstrings=bool(args.include_docstrings),
            include_string_literals=bool(args.include_string_literals),
            scan_tests_as_blocking=bool(args.scan_tests_as_blocking),
            severity_by_path=cast("list[str] | None", args.severity_by_path),
            warn_on_audit_tool_self_references=bool(args.warn_on_audit_tool_self_references),
            self_reference_allowlist=tuple(cast("list[str]", args.self_reference_allowlist)),
        )

        if args.output_format == "json":
            sys.stdout.write(module.format_json(run_result))
        else:
            sys.stdout.write(
                _format_compact_text(
                    result=run_result,
                    roots=tuple(cast("list[str]", args.paths)),
                    exclude=tuple(cast("list[str]", args.exclude)),
                )
            )

        has_blocking_findings = run_result.error_count > 0
        has_warning_failures = bool(args.fail_on_warn) and run_result.warning_count > 0
        return 1 if has_blocking_findings or has_warning_failures else 0
    except Exception as exc:  # pragma: no cover - defensive wrapper boundary
        sys.stderr.write(f"placeholder-gate crashed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
