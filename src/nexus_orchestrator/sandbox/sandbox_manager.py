"""Sandbox execution manager with safe-by-default local backend semantics."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from nexus_orchestrator.sandbox.network_policy import (
    NetworkDecision,
    NetworkPolicy,
    NetworkPolicyMode,
)
from nexus_orchestrator.utils.fs import is_within

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class SandboxError(RuntimeError):
    """Base error for sandbox manager failures."""


class UnsupportedSandboxBackendError(SandboxError):
    """Raised when requesting an unsupported backend."""


class SandboxPolicyError(SandboxError):
    """Raised when execution violates sandbox policy."""


class SandboxBackend(str, Enum):
    """Backend names accepted by :class:`SandboxManager`."""

    NONE = "none"
    DOCKER = "docker"
    PODMAN = "podman"


@dataclass(frozen=True, slots=True)
class SandboxCommandResult:
    """Normalized command result emitted by sandbox execution backends."""

    backend: SandboxBackend
    command: tuple[str, ...]
    cwd: Path
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: float
    network_decision: NetworkDecision | None = None

    @property
    def succeeded(self) -> bool:
        return not self.timed_out and self.returncode == 0


class SandboxManager:
    """Execute commands under a unified sandbox API.

    Current implementation supports the ``none`` backend (local subprocess execution)
    while enforcing working-directory containment and policy defaults.
    """

    def __init__(
        self,
        workspace_root: Path | str,
        *,
        backend: SandboxBackend | str = SandboxBackend.NONE,
        network_policy: NetworkPolicy | None = None,
        default_timeout_seconds: float = 300.0,
        env_overrides: Mapping[str, str] | None = None,
        inherit_host_env: bool = False,
    ) -> None:
        root = Path(workspace_root).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"{root!s} is not a directory")
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be > 0")

        self._workspace_root = root
        self._backend = _coerce_backend(backend)
        self._network_policy = network_policy or NetworkPolicy(mode=NetworkPolicyMode.DENY)
        self._default_timeout_seconds = float(default_timeout_seconds)
        self._env_overrides = dict(env_overrides or {})
        self._inherit_host_env = bool(inherit_host_env)

    @property
    def backend(self) -> SandboxBackend:
        return self._backend

    @property
    def workspace_root(self) -> Path:
        return self._workspace_root

    @property
    def network_policy(self) -> NetworkPolicy:
        return self._network_policy

    def execute(
        self,
        command: Sequence[str],
        *,
        cwd: Path | str,
        timeout_seconds: float | None = None,
        stdin_text: str | None = None,
        env: Mapping[str, str] | None = None,
        network_target: str | None = None,
    ) -> SandboxCommandResult:
        parsed_command = _normalize_command(command)
        resolved_cwd = self._resolve_cwd(cwd)
        effective_timeout = (
            self._default_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
        )
        if effective_timeout <= 0:
            raise ValueError("timeout_seconds must be > 0")

        if self._backend is not SandboxBackend.NONE:
            raise UnsupportedSandboxBackendError(
                f"backend {self._backend.value!r} is not implemented yet"
            )

        network_decision: NetworkDecision | None = None
        if network_target is not None:
            network_decision = self._network_policy.enforce(network_target)

        run_env = self._build_environment(env)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                list(parsed_command),
                cwd=resolved_cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=run_env,
                input=stdin_text,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            return SandboxCommandResult(
                backend=self._backend,
                command=parsed_command,
                cwd=resolved_cwd,
                returncode=None,
                stdout=_coerce_timeout_stream(exc.stdout),
                stderr=_coerce_timeout_stream(exc.stderr),
                timed_out=True,
                duration_ms=duration_ms,
                network_decision=network_decision,
            )

        duration_ms = (time.perf_counter() - started) * 1000.0
        return SandboxCommandResult(
            backend=self._backend,
            command=parsed_command,
            cwd=resolved_cwd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=False,
            duration_ms=duration_ms,
            network_decision=network_decision,
        )

    def check_network_access(
        self,
        target: str,
        *,
        port: int | None = None,
        scheme: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> NetworkDecision:
        return self._network_policy.enforce(target, port=port, scheme=scheme, context=context)

    def _resolve_cwd(self, cwd: Path | str) -> Path:
        path = Path(cwd).resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(f"{path!s} is not a directory")
        if not is_within(path, self._workspace_root):
            raise SandboxPolicyError(
                f"working directory {path!s} is outside sandbox workspace {self._workspace_root!s}"
            )
        return path

    def _build_environment(self, env: Mapping[str, str] | None) -> dict[str, str]:
        if self._inherit_host_env:
            merged = dict(os.environ)
        else:
            merged = {}
            host_path = os.environ.get("PATH")
            if host_path:
                merged["PATH"] = host_path
        merged.update(self._env_overrides)
        if env is not None:
            merged.update(env)
        return merged


def _coerce_backend(value: SandboxBackend | str) -> SandboxBackend:
    if isinstance(value, SandboxBackend):
        return value
    if not isinstance(value, str):
        raise ValueError("backend must be a string or SandboxBackend")
    normalized = value.strip().lower()
    try:
        return SandboxBackend(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SandboxBackend)
        raise ValueError(f"unsupported backend {value!r}; expected one of: {allowed}") from exc


def _normalize_command(command: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(command, (list, tuple)):
        raise ValueError("command must be a sequence of strings")
    normalized = tuple(item.strip() for item in command if item.strip())
    if not normalized:
        raise ValueError("command must not be empty")
    return normalized


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = [
    "SandboxBackend",
    "SandboxCommandResult",
    "SandboxError",
    "SandboxManager",
    "SandboxPolicyError",
    "UnsupportedSandboxBackendError",
]
