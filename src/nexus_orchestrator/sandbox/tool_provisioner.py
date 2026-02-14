"""Tool provisioning with registry pinning, checksums, and persistence audit records."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]

from nexus_orchestrator.persistence.repositories import ToolInstallRecord
from nexus_orchestrator.utils.fs import temp_directory
from nexus_orchestrator.utils.hashing import sha256_file, sha256_text

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python <3.11 compatibility.
    UTC = timezone.utc  # noqa: UP017

JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_HEX_DIGITS = frozenset("0123456789abcdef")


class ToolProvisioningError(RuntimeError):
    """Base error for tool provisioning failures."""


class ToolRegistryError(ValueError):
    """Raised when tool registry payloads are invalid."""


class ToolInstallCommandError(ToolProvisioningError):
    """Raised when a command-driven install strategy fails."""

    def __init__(
        self,
        *,
        command: Sequence[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = stderr.strip() or stdout.strip()
        message = f"command failed ({returncode}): {' '.join(command)}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ToolRegistryEntry:
    """Validated registry entry loaded from ``tools/registry.toml``."""

    name: str
    version: str
    source: str
    risk: str
    package: str | None = None
    repo: str | None = None
    asset: str | None = None
    download_url: str | None = None
    checksum: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_tool_name(self.name, "ToolRegistryEntry.name"))
        object.__setattr__(
            self,
            "version",
            _normalize_text(self.version, "ToolRegistryEntry.version", max_len=128),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_text(self.source, "ToolRegistryEntry.source", max_len=64).lower(),
        )
        object.__setattr__(
            self,
            "risk",
            _normalize_text(self.risk, "ToolRegistryEntry.risk", max_len=64).lower(),
        )
        if self.package is not None:
            object.__setattr__(
                self,
                "package",
                _normalize_text(self.package, "ToolRegistryEntry.package", max_len=128),
            )
        if self.repo is not None:
            object.__setattr__(
                self,
                "repo",
                _normalize_text(self.repo, "ToolRegistryEntry.repo", max_len=256),
            )
        if self.asset is not None:
            object.__setattr__(
                self,
                "asset",
                _normalize_text(self.asset, "ToolRegistryEntry.asset", max_len=256),
            )
        if self.download_url is not None:
            object.__setattr__(
                self,
                "download_url",
                _normalize_text(self.download_url, "ToolRegistryEntry.download_url", max_len=1024),
            )
        if self.checksum is not None:
            object.__setattr__(
                self,
                "checksum",
                _normalize_checksum(self.checksum, field_name="ToolRegistryEntry.checksum"),
            )
        normalized_metadata = {
            _normalize_text(key, "ToolRegistryEntry.metadata key", max_len=128): _normalize_text(
                value,
                "ToolRegistryEntry.metadata value",
                max_len=2048,
            )
            for key, value in self.metadata.items()
        }
        object.__setattr__(self, "metadata", normalized_metadata)


@dataclass(frozen=True, slots=True)
class CommandExecutionResult:
    """Normalized subprocess execution result for install strategies."""

    command: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class DownloadedReleaseArtifact:
    """Result returned by a GitHub release downloader implementation."""

    path: Path
    source_url: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        if self.checksum is not None:
            object.__setattr__(
                self,
                "checksum",
                _normalize_checksum(self.checksum, field_name="DownloadedReleaseArtifact.checksum"),
            )


@dataclass(frozen=True, slots=True)
class ToolProvisioningResult:
    """Detailed provisioning result that includes persisted audit record."""

    record: ToolInstallRecord
    registry_entry: ToolRegistryEntry
    artifact_path: Path | None
    checksum: str


class ToolInstallRepository(Protocol):
    """Minimal repository contract used by :class:`ToolProvisioner`."""

    def save(self, record: ToolInstallRecord) -> ToolInstallRecord: ...


class CommandRunner(Protocol):
    """Injectable command runner used for deterministic/offline testing."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecutionResult: ...


class GithubReleaseDownloader(Protocol):
    """Injectable GitHub release downloader interface."""

    def download(
        self,
        *,
        entry: ToolRegistryEntry,
        destination_dir: Path,
    ) -> DownloadedReleaseArtifact: ...


class SubprocessCommandRunner:
    """Default command runner backed by ``subprocess.run``."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecutionResult:
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolProvisioningError(
                f"command timed out after {timeout_seconds} seconds: {' '.join(command)}"
            ) from exc

        return CommandExecutionResult(
            command=tuple(command),
            cwd=cwd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


@dataclass(frozen=True, slots=True)
class _InstallArtifact:
    source: str
    path: Path | None
    checksum: str | None
    metadata: dict[str, str] = field(default_factory=dict)


class ToolProvisioner:
    """Provision approved tools from a pinned registry and persist audit records."""

    def __init__(
        self,
        *,
        install_repo: ToolInstallRepository,
        registry_path: Path | str = "tools/registry.toml",
        command_runner: CommandRunner | None = None,
        github_downloader: GithubReleaseDownloader | None = None,
        install_root: Path | str = ".cache/tool-installs",
        python_executable: str = sys.executable,
        now_provider: Callable[[], datetime] | None = None,
        default_installed_by: str = "orchestrator",
    ) -> None:
        self._install_repo = install_repo
        self._registry_path = Path(registry_path)
        self._command_runner = command_runner or SubprocessCommandRunner()
        self._github_downloader = github_downloader
        self._install_root = Path(install_root)
        self._install_root.mkdir(parents=True, exist_ok=True)
        self._python_executable = _normalize_text(
            python_executable, "python_executable", max_len=1024
        )
        self._now = now_provider or _utc_now
        self._default_installed_by = _normalize_text(
            default_installed_by, "default_installed_by", max_len=128
        )
        self._registry = load_tool_registry(self._registry_path)

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    @property
    def registry(self) -> dict[str, ToolRegistryEntry]:
        return dict(self._registry)

    def reload_registry(self) -> None:
        self._registry = load_tool_registry(self._registry_path)

    def get_registry_entry(self, tool_name: str) -> ToolRegistryEntry | None:
        return self._registry.get(_normalize_tool_name(tool_name, "tool_name"))

    def provision(
        self,
        tool_name: str,
        *,
        version: str | None = None,
        timeout_seconds: float = 300.0,
        installed_by: str | None = None,
        notes: str | None = None,
    ) -> ToolProvisioningResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")

        entry = self._resolve_entry(tool_name, version=version)
        artifact = self._install(entry, timeout_seconds=timeout_seconds)
        checksum = self._resolve_checksum(entry, artifact)
        installed_at = _coerce_datetime_utc(self._now())
        record_id = _build_record_id(entry, installed_at, checksum)

        metadata: dict[str, JSONValue] = {
            "source": entry.source,
            "risk": entry.risk,
            "strategy": artifact.source,
        }
        if artifact.path is not None:
            metadata["artifact_path"] = str(artifact.path)
        metadata.update(entry.metadata)
        metadata.update(artifact.metadata)

        record = ToolInstallRecord(
            id=record_id,
            tool=entry.name,
            version=entry.version,
            checksum=checksum,
            approved=True,
            installed_at=installed_at,
            installed_by=(
                _normalize_text(installed_by, "installed_by", max_len=128)
                if installed_by is not None
                else self._default_installed_by
            ),
            notes=(
                _normalize_text(notes, "notes", max_len=2048)
                if notes is not None
                else f"installed via {entry.source}"
            ),
            metadata=metadata,
        )
        persisted = self._install_repo.save(record)
        return ToolProvisioningResult(
            record=persisted,
            registry_entry=entry,
            artifact_path=artifact.path,
            checksum=checksum,
        )

    def _resolve_entry(self, tool_name: str, *, version: str | None) -> ToolRegistryEntry:
        normalized_name = _normalize_tool_name(tool_name, "tool_name")
        entry = self._registry.get(normalized_name)
        if entry is None:
            raise ToolProvisioningError(f"tool {tool_name!r} is not present in the registry")
        if version is not None:
            normalized_version = _normalize_text(version, "version", max_len=128)
            if normalized_version != entry.version:
                raise ToolProvisioningError(
                    f"requested version {normalized_version!r} does not match pinned "
                    f"version {entry.version!r} for {entry.name!r}"
                )
        return entry

    def _install(self, entry: ToolRegistryEntry, *, timeout_seconds: float) -> _InstallArtifact:
        if entry.source == "pypi":
            return self._install_from_pypi(entry, timeout_seconds=timeout_seconds)
        if entry.source == "github-release":
            return self._install_from_github_release(entry)
        raise ToolProvisioningError(
            f"unsupported install source {entry.source!r} for tool {entry.name!r}"
        )

    def _install_from_pypi(
        self, entry: ToolRegistryEntry, *, timeout_seconds: float
    ) -> _InstallArtifact:
        package = entry.package or entry.name
        requirement = f"{package}=={entry.version}"
        with temp_directory(prefix=f"tool-{entry.name}-") as stage_dir:
            download_command = [
                self._python_executable,
                "-m",
                "pip",
                "download",
                "--disable-pip-version-check",
                "--no-deps",
                "--dest",
                str(stage_dir),
                requirement,
            ]
            self._run_command(download_command, cwd=stage_dir, timeout_seconds=timeout_seconds)

            artifact_path = _select_downloaded_artifact(stage_dir)
            if artifact_path is None:
                install_command = [
                    self._python_executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--no-input",
                    "--no-deps",
                    requirement,
                ]
                self._run_command(install_command, cwd=stage_dir, timeout_seconds=timeout_seconds)
                fallback_checksum = sha256_text(f"{entry.name}:{entry.version}:{entry.source}")
                return _InstallArtifact(
                    source="pypi",
                    path=None,
                    checksum=fallback_checksum,
                    metadata={"requirement": requirement, "checksum_source": "descriptor"},
                )

            checksum = sha256_file(artifact_path)
            persisted_artifact = self._persist_artifact(entry, artifact_path)
            install_command = [
                self._python_executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--no-deps",
                str(persisted_artifact),
            ]
            self._run_command(install_command, cwd=stage_dir, timeout_seconds=timeout_seconds)
            return _InstallArtifact(
                source="pypi",
                path=persisted_artifact,
                checksum=checksum,
                metadata={
                    "requirement": requirement,
                    "checksum_source": "artifact",
                },
            )

    def _install_from_github_release(self, entry: ToolRegistryEntry) -> _InstallArtifact:
        if self._github_downloader is None:
            raise ToolProvisioningError(
                "github-release install requested but no github_downloader was configured"
            )

        with temp_directory(prefix=f"tool-{entry.name}-") as stage_dir:
            downloaded = self._github_downloader.download(entry=entry, destination_dir=stage_dir)
            source_path = downloaded.path.resolve(strict=True)
            if not source_path.is_file():
                raise ToolProvisioningError(f"downloaded artifact is not a file: {source_path!s}")
            checksum = downloaded.checksum or sha256_file(source_path)
            checksum = _normalize_checksum(
                checksum,
                field_name=f"{entry.name}.github_release_checksum",
            )
            persisted_artifact = self._persist_artifact(entry, source_path)
            metadata: dict[str, str] = {
                "checksum_source": "artifact",
            }
            if downloaded.source_url is not None:
                metadata["source_url"] = downloaded.source_url
            return _InstallArtifact(
                source="github-release",
                path=persisted_artifact,
                checksum=checksum,
                metadata=metadata,
            )

    def _persist_artifact(self, entry: ToolRegistryEntry, source_path: Path) -> Path:
        target_dir = self._install_root / entry.name / entry.version
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source_path.name
        shutil.copy2(source_path, target_path)
        return target_path

    def _run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecutionResult:
        result = self._command_runner.run(command, cwd=cwd, timeout_seconds=timeout_seconds)
        if result.returncode != 0:
            raise ToolInstallCommandError(
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    def _resolve_checksum(self, entry: ToolRegistryEntry, artifact: _InstallArtifact) -> str:
        if artifact.checksum is None:
            checksum = sha256_text(f"{entry.name}:{entry.version}:{entry.source}")
        else:
            checksum = artifact.checksum
        checksum = _normalize_checksum(checksum, field_name=f"{entry.name}.resolved_checksum")
        if entry.checksum is not None and checksum != entry.checksum:
            raise ToolProvisioningError(
                f"checksum mismatch for {entry.name!r}: expected {entry.checksum}, got {checksum}"
            )
        return checksum


def load_tool_registry(path: Path | str) -> dict[str, ToolRegistryEntry]:
    """Load and validate ``tools/registry.toml``."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise ToolRegistryError(f"tool registry file not found: {registry_path!s}")
    try:
        with registry_path.open("rb") as handle:
            payload = tomllib.load(handle)
    except OSError as exc:
        raise ToolRegistryError(f"unable to read tool registry {registry_path!s}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ToolRegistryError(f"invalid TOML in {registry_path!s}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolRegistryError("tool registry root must be a table")
    raw_tools = payload.get("tool")
    if not isinstance(raw_tools, dict):
        raise ToolRegistryError("tool registry must include [tool.<name>] entries")

    entries: dict[str, ToolRegistryEntry] = {}
    for raw_name, raw_entry in sorted(raw_tools.items(), key=lambda item: str(item[0])):
        try:
            if not isinstance(raw_name, str):
                raise ToolRegistryError("tool registry names must be strings")
            if not isinstance(raw_entry, dict):
                raise ToolRegistryError(f"entry for {raw_name!r} must be a table")

            name = _normalize_tool_name(raw_name, f"tool.{raw_name}")
            raw_version = raw_entry.get("version")
            if not isinstance(raw_version, str):
                raise ToolRegistryError(f"tool {raw_name!r} must define a string version")
            source = raw_entry.get("source", "pypi")
            risk = raw_entry.get("risk", "unknown")
            if not isinstance(source, str):
                raise ToolRegistryError(f"tool {raw_name!r} source must be a string")
            if not isinstance(risk, str):
                raise ToolRegistryError(f"tool {raw_name!r} risk must be a string")

            raw_checksum = raw_entry.get("checksum", raw_entry.get("sha256"))
            checksum: str | None
            if raw_checksum is None:
                checksum = None
            elif isinstance(raw_checksum, str):
                checksum = raw_checksum
            else:
                raise ToolRegistryError(f"tool {raw_name!r} checksum must be a string")

            package = _optional_str(raw_entry.get("package"), field_name=f"tool.{raw_name}.package")
            repo = _optional_str(raw_entry.get("repo"), field_name=f"tool.{raw_name}.repo")
            asset = _optional_str(raw_entry.get("asset"), field_name=f"tool.{raw_name}.asset")
            download_url = _optional_str(
                raw_entry.get("download_url", raw_entry.get("url")),
                field_name=f"tool.{raw_name}.download_url",
            )

            metadata = {
                _normalize_text(
                    str(key),
                    f"tool.{raw_name}.metadata key",
                    max_len=128,
                ): _normalize_text(
                    str(value),
                    f"tool.{raw_name}.metadata value",
                    max_len=2048,
                )
                for key, value in raw_entry.items()
                if key
                not in {
                    "version",
                    "source",
                    "risk",
                    "checksum",
                    "sha256",
                    "package",
                    "repo",
                    "asset",
                    "download_url",
                    "url",
                }
            }

            entry = ToolRegistryEntry(
                name=name,
                version=raw_version,
                source=source,
                risk=risk,
                package=package,
                repo=repo,
                asset=asset,
                download_url=download_url,
                checksum=checksum,
                metadata=metadata,
            )
        except ValueError as exc:
            raise ToolRegistryError(f"invalid tool entry {raw_name!r}: {exc}") from exc
        if entry.source not in {"pypi", "github-release"}:
            raise ToolRegistryError(f"tool {raw_name!r} has unsupported source {entry.source!r}")
        if entry.name in entries:
            raise ToolRegistryError(f"duplicate tool entry {entry.name!r}")
        entries[entry.name] = entry

    return entries


def _optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolRegistryError(f"{field_name} must be a string when provided")
    return value


def _normalize_text(value: str, field_name: str, *, max_len: int = 2048) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > max_len:
        raise ValueError(f"{field_name} must be <= {max_len} characters")
    if "\x00" in normalized:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return normalized


def _normalize_tool_name(value: str, field_name: str) -> str:
    normalized = _normalize_text(value, field_name, max_len=128).lower().replace("_", "-")
    if any(char.isspace() for char in normalized):
        raise ValueError(f"{field_name} must not contain whitespace")
    return normalized


def _normalize_checksum(value: str, *, field_name: str) -> str:
    normalized = _normalize_text(value, field_name, max_len=128).lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.split(":", 1)[1]
    if len(normalized) != 64 or any(char not in _HEX_DIGITS for char in normalized):
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return normalized


def _select_downloaded_artifact(directory: Path) -> Path | None:
    candidates = sorted(path for path in directory.iterdir() if path.is_file())
    if not candidates:
        return None
    return candidates[0]


def _build_record_id(entry: ToolRegistryEntry, installed_at: datetime, checksum: str) -> str:
    digest_seed = (
        f"{entry.name}:{entry.version}:{entry.source}:{checksum}:{installed_at.isoformat()}"
    )
    digest = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()[:12]
    version_token = entry.version.replace(".", "-")
    return f"tool-{entry.name}-{version_token}-{digest}"


def _coerce_datetime_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


__all__ = [
    "CommandExecutionResult",
    "CommandRunner",
    "DownloadedReleaseArtifact",
    "GithubReleaseDownloader",
    "SubprocessCommandRunner",
    "ToolInstallCommandError",
    "ToolInstallRepository",
    "ToolProvisioner",
    "ToolProvisioningError",
    "ToolProvisioningResult",
    "ToolRegistryEntry",
    "ToolRegistryError",
    "load_tool_registry",
]
