"""Executable CLI entrypoint for ``nexus_orchestrator``."""

from __future__ import annotations

import sys
import traceback
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExitCode(IntEnum):
    """Deterministic process exit-code contract."""

    SUCCESS = 0
    VERIFICATION_REJECTED = 1
    CONFIG_ERROR = 2
    PROVIDER_ERROR = 3
    INTERNAL_ERROR = 4


def cli_entrypoint(argv: Sequence[str] | None = None) -> int:
    """Entrypoint used by ``python -m nexus_orchestrator`` and script shims."""

    try:
        from nexus_orchestrator.ui.cli import run_cli

        return _normalize_exit_code(run_cli(argv))
    except SystemExit as exc:  # pragma: no cover - defensive in case argparse bubbles out.
        return _normalize_exit_code(exc.code)
    except BaseException as exc:  # noqa: BLE001 - CLI boundary normalization.
        exit_code = _route_exception(exc)
        _emit_failure(exc, exit_code)
        return int(exit_code)


def _normalize_exit_code(raw_code: object) -> int:
    if isinstance(raw_code, int) and raw_code in {0, 1, 2, 3, 4}:
        return raw_code
    if raw_code is None:
        return int(ExitCode.SUCCESS)
    if isinstance(raw_code, str) and raw_code.strip():
        _write_stderr(raw_code.strip())
    return int(ExitCode.INTERNAL_ERROR)


def _route_exception(exc: BaseException) -> ExitCode:
    config_error_types = _load_config_error_types()
    provider_error_type = _load_provider_error_type()

    for item in _iter_exception_chain(exc):
        if isinstance(item, config_error_types):
            return ExitCode.CONFIG_ERROR
        if isinstance(item, (FileNotFoundError, NotADirectoryError, PermissionError, ValueError)):
            return ExitCode.CONFIG_ERROR
        if provider_error_type is not None and isinstance(item, provider_error_type):
            return ExitCode.PROVIDER_ERROR
        if _is_provider_sdk_missing_error(item):
            return ExitCode.PROVIDER_ERROR
    return ExitCode.INTERNAL_ERROR


def _load_config_error_types() -> tuple[type[BaseException], ...]:
    try:
        from nexus_orchestrator.config.loader import ConfigLoadError
        from nexus_orchestrator.config.schema import ConfigValidationError
    except Exception:  # pragma: no cover - defensive import fallback.
        return ()
    return (ConfigLoadError, ConfigValidationError)


def _load_provider_error_type() -> type[BaseException] | None:
    try:
        from nexus_orchestrator.synthesis_plane.providers.base import ProviderError
    except Exception:  # pragma: no cover - defensive import fallback.
        return None
    return ProviderError


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    seen: set[int] = set()
    items: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None:
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        items.append(current)
        if current.__cause__ is not None:
            current = current.__cause__
            continue
        if current.__context__ is not None and not current.__suppress_context__:
            current = current.__context__
            continue
        break
    return items


def _is_provider_sdk_missing_error(exc: BaseException) -> bool:
    if not isinstance(exc, ModuleNotFoundError):
        return False
    return exc.name in {"openai", "anthropic"}


def _emit_failure(exc: BaseException, exit_code: ExitCode) -> None:
    if exit_code is ExitCode.INTERNAL_ERROR:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        return
    _write_stderr(str(exc).strip() or exc.__class__.__name__)


def _write_stderr(message: str) -> None:
    sys.stderr.write(message.rstrip("\n") + "\n")


__all__ = ["ExitCode", "cli_entrypoint"]
