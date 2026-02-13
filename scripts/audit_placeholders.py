"""
nexus-orchestrator — placeholder audit CLI wrapper

File: scripts/audit_placeholders.py
Last updated: 2026-02-13

Purpose
- Provide a stable, no-install wrapper for the placeholder audit utility.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _load_module():
    try:
        from nexus_orchestrator.quality import placeholder_audit as loaded_module

        return loaded_module
    except ModuleNotFoundError:
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))
        from nexus_orchestrator.quality import placeholder_audit as loaded_module

        return loaded_module


def _build_parser(module) -> argparse.ArgumentParser:
    parser = module._build_parser()
    if "--warn-on-string-markers" not in parser._option_string_actions:
        parser.add_argument(
            "--warn-on-string-markers",
            action="store_true",
            help="Warn on placeholder-like TODO/FIXME/NotImplementedError string markers.",
        )
    if "--warn-on-audit-tool-self-references" not in parser._option_string_actions:
        parser.add_argument(
            "--warn-on-audit-tool-self-references",
            action="store_true",
            help="Warn when the audit tool references its own marker patterns.",
        )
    return parser


def _build_run_kwargs(module, args: argparse.Namespace) -> dict[str, object]:
    run_kwargs: dict[str, object] = {
        "repo_root": Path(args.repo_root).resolve(),
        "roots": args.roots,
        "exclude": args.exclude,
        "show_context": args.show_context,
    }
    run_signature = inspect.signature(module.run_placeholder_audit)
    if "warn_on_string_markers" in run_signature.parameters:
        run_kwargs["warn_on_string_markers"] = args.warn_on_string_markers
    if "warn_on_audit_tool_self_references" in run_signature.parameters:
        run_kwargs["warn_on_audit_tool_self_references"] = args.warn_on_audit_tool_self_references
    if "self_reference_allowlist" in run_signature.parameters and hasattr(
        args, "self_reference_allowlist"
    ):
        run_kwargs["self_reference_allowlist"] = tuple(args.self_reference_allowlist)
    return run_kwargs


def main(argv: Sequence[str] | None = None) -> int:
    module = _load_module()
    parser = _build_parser(module)
    args = parser.parse_args(argv)

    try:
        result = module.run_placeholder_audit(**_build_run_kwargs(module, args))
        if args.output_format == "json":
            sys.stdout.write(module.format_json(result))
        else:
            sys.stdout.write(module.format_text(result))

        should_fail = result.error_count > 0 or (args.fail_on_warn and result.warning_count > 0)
        return 1 if should_fail else 0
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        sys.stderr.write(f"placeholder audit crashed: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
