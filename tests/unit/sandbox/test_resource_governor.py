"""Unit tests for sandbox resource governance and provisioning primitives."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.sandbox.network_policy import (
    NetworkAccessEvent,
    NetworkDecision,
    NetworkPolicy,
    NetworkPolicyMode,
    NetworkPolicyViolation,
)
from nexus_orchestrator.sandbox.resource_governor import (
    BackpressureLevel,
    ResourceGovernor,
    ResourceGovernorConfig,
    ResourceMetricsSnapshot,
)
from nexus_orchestrator.sandbox.sandbox_manager import SandboxManager, SandboxPolicyError
from nexus_orchestrator.sandbox.tool_provisioner import (
    CommandExecutionResult,
    DownloadedReleaseArtifact,
    ToolProvisioner,
    ToolProvisioningError,
    ToolRegistryEntry,
)
from nexus_orchestrator.utils.hashing import sha256_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nexus_orchestrator.persistence.repositories import ToolInstallRecord

try:
    from datetime import UTC
except ImportError:  # pragma: no cover - Python <3.11 compatibility.
    UTC = timezone.utc  # noqa: UP017


def _snapshot(
    *,
    cpu_percent: float = 20.0,
    memory_total_gib: int = 16,
    memory_available_gib: int = 12,
    disk_total_gib: int = 512,
    disk_free_gib: int = 220,
) -> ResourceMetricsSnapshot:
    gib = 1024 * 1024 * 1024
    return ResourceMetricsSnapshot(
        captured_at=datetime(2026, 2, 1, 12, 0, 0, tzinfo=UTC),
        cpu_percent=cpu_percent,
        memory_total_bytes=memory_total_gib * gib,
        memory_available_bytes=memory_available_gib * gib,
        disk_total_bytes=disk_total_gib * gib,
        disk_free_bytes=disk_free_gib * gib,
        disk_read_bytes=1_000_000,
        disk_write_bytes=2_000_000,
        swap_used_bytes=0,
    )


def test_resource_governor_orders_degradation_and_recovers() -> None:
    config = ResourceGovernorConfig(
        default_verification_concurrency=8,
        default_dispatch_concurrency=6,
        elevated_memory_available_bytes=8 * 1024 * 1024 * 1024,
        critical_memory_available_bytes=6 * 1024 * 1024 * 1024,
    )
    governor = ResourceGovernor(config=config)

    elevated = _snapshot(memory_available_gib=7)
    elevated_decision = governor.evaluate(elevated)

    assert elevated_decision.level is BackpressureLevel.ELEVATED
    assert elevated_decision.actions == (
        "disable_speculative_execution",
        "reduce_verification_concurrency",
    )
    assert not elevated_decision.limits.speculative_execution_enabled
    assert (
        elevated_decision.limits.verification_concurrency < config.default_verification_concurrency
    )
    assert elevated_decision.limits.dispatch_concurrency == config.default_dispatch_concurrency

    critical = _snapshot(memory_available_gib=5)
    critical_decision = governor.evaluate(critical)

    assert critical_decision.level is BackpressureLevel.CRITICAL
    assert critical_decision.actions == (
        "throttle_dispatch_concurrency",
        "tighten_verification_concurrency",
    )
    assert critical_decision.limits.dispatch_concurrency < config.default_dispatch_concurrency
    assert critical_decision.limits.verification_concurrency < (
        elevated_decision.limits.verification_concurrency
    )

    normal = _snapshot(memory_available_gib=12)
    normal_decision = governor.evaluate(normal)

    assert normal_decision.level is BackpressureLevel.NORMAL
    assert normal_decision.actions == (
        "restore_dispatch_concurrency",
        "restore_verification_concurrency",
        "enable_speculative_execution",
    )
    assert normal_decision.limits.speculative_execution_enabled
    assert (
        normal_decision.limits.verification_concurrency == config.default_verification_concurrency
    )
    assert normal_decision.limits.dispatch_concurrency == config.default_dispatch_concurrency


def test_resource_governor_disk_minimum_triggers_critical_backpressure() -> None:
    config = ResourceGovernorConfig(
        elevated_disk_free_bytes=120 * 1024 * 1024 * 1024,
        critical_disk_free_bytes=100 * 1024 * 1024 * 1024,
    )
    governor = ResourceGovernor(config=config)

    decision = governor.evaluate(_snapshot(disk_free_gib=80))

    assert decision.level is BackpressureLevel.CRITICAL
    assert "throttle_dispatch_concurrency" in decision.actions


def test_network_policy_allowlist_mode_enforces_and_logs() -> None:
    decisions: list[NetworkDecision] = []
    policy = NetworkPolicy(
        mode=NetworkPolicyMode.ALLOWLIST,
        allowlist=("pypi.org", "*.pythonhosted.org", "10.0.0.0/8"),
        decision_logger=decisions.append,
    )

    allowed = policy.evaluate("https://files.pythonhosted.org/simple")
    denied = policy.evaluate("https://example.com")

    assert allowed.allowed
    assert allowed.matched_rule == "*.pythonhosted.org"
    assert not denied.allowed
    assert denied.matched_rule is None
    assert len(decisions) == 2

    with pytest.raises(NetworkPolicyViolation):
        policy.enforce("https://example.com")


def test_network_policy_logged_permissive_allows_and_records_access() -> None:
    decisions: list[NetworkDecision] = []
    events: list[NetworkAccessEvent] = []
    policy = NetworkPolicy(
        mode=NetworkPolicyMode.LOGGED_PERMISSIVE,
        allowlist=("pypi.org",),
        decision_logger=decisions.append,
        access_logger=events.append,
    )

    decision = policy.enforce("https://unknown.invalid/path")
    event = policy.log_access(
        decision,
        bytes_sent=128,
        bytes_received=2048,
        duration_ms=12.5,
    )

    assert decision.allowed
    assert decision.reason == "network access allowed in logged_permissive mode"
    assert event.bytes_received == 2048
    assert len(decisions) == 1
    assert len(events) == 1


def test_sandbox_manager_none_backend_executes_and_captures_output(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)
    manager = SandboxManager(workspace_root=workspace_root, backend="none")

    result = manager.execute(
        [
            sys.executable,
            "-c",
            "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr)",
        ],
        cwd=workspace_root,
        timeout_seconds=2.0,
    )

    assert result.returncode == 0
    assert not result.timed_out
    assert "stdout-ok" in result.stdout
    assert "stderr-ok" in result.stderr


def test_sandbox_manager_rejects_working_dir_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True)
    manager = SandboxManager(workspace_root=workspace_root, backend="none")

    with pytest.raises(SandboxPolicyError):
        manager.execute([sys.executable, "-c", "print('x')"], cwd=outside)


def test_sandbox_manager_timeout_is_captured(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True)
    manager = SandboxManager(workspace_root=workspace_root, backend="none")

    result = manager.execute(
        [sys.executable, "-c", "import time; time.sleep(1.0)"],
        cwd=workspace_root,
        timeout_seconds=0.1,
    )

    assert result.timed_out
    assert result.returncode is None


@dataclass
class _InMemoryToolInstallRepo:
    saved: list[ToolInstallRecord] = field(default_factory=list)

    def save(self, record: ToolInstallRecord) -> ToolInstallRecord:
        self.saved.append(record)
        return record


@dataclass
class _FakeCommandRunner:
    commands: list[tuple[str, ...]] = field(default_factory=list)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
    ) -> CommandExecutionResult:
        del timeout_seconds
        command_tuple = tuple(command)
        self.commands.append(command_tuple)

        if "download" in command_tuple:
            dest_idx = command_tuple.index("--dest")
            dest_dir = Path(command_tuple[dest_idx + 1])
            requirement = command_tuple[-1]
            package, version = requirement.split("==", maxsplit=1)
            wheel_path = dest_dir / f"{package}-{version}-py3-none-any.whl"
            wheel_path.write_bytes(f"{package}:{version}".encode())
            return CommandExecutionResult(
                command=command_tuple,
                cwd=cwd,
                returncode=0,
                stdout="downloaded",
                stderr="",
            )

        if "install" in command_tuple:
            return CommandExecutionResult(
                command=command_tuple,
                cwd=cwd,
                returncode=0,
                stdout="installed",
                stderr="",
            )

        return CommandExecutionResult(
            command=command_tuple,
            cwd=cwd,
            returncode=0,
            stdout="",
            stderr="",
        )


@dataclass
class _FakeGithubDownloader:
    payload: bytes = b"release-asset-binary"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def download(
        self,
        *,
        entry: ToolRegistryEntry,
        destination_dir: Path,
    ) -> DownloadedReleaseArtifact:
        tool_name = entry.name
        version = entry.version
        self.calls.append((tool_name, version))
        path = destination_dir / f"{tool_name}-{version}.tar.gz"
        path.write_bytes(self.payload)
        return DownloadedReleaseArtifact(path=path, source_url="https://example.invalid/release")


def _fixed_now() -> datetime:
    return datetime(2026, 2, 1, 12, 30, 0, tzinfo=UTC)


def test_tool_provisioner_pypi_strategy_persists_checksum_and_record(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        "\n".join(
            [
                "[tool.pytest]",
                'version = "9.0.2"',
                'source = "pypi"',
                'risk = "low"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    repo = _InMemoryToolInstallRepo()
    runner = _FakeCommandRunner()
    provisioner = ToolProvisioner(
        install_repo=repo,
        registry_path=registry_path,
        command_runner=runner,
        install_root=tmp_path / "installed",
        now_provider=_fixed_now,
    )

    result = provisioner.provision("pytest")

    assert result.record.tool == "pytest"
    assert result.record.version == "9.0.2"
    assert result.record.metadata["source"] == "pypi"
    assert result.artifact_path is not None
    assert result.checksum == sha256_file(result.artifact_path)
    assert len(repo.saved) == 1
    assert len(runner.commands) == 2
    assert "download" in runner.commands[0]
    assert "install" in runner.commands[1]


def test_tool_provisioner_github_release_strategy_is_offline_testable(tmp_path: Path) -> None:
    payload = b"github-release-bytes"
    checksum = hashlib.sha256(payload).hexdigest()
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        "\n".join(
            [
                "[tool.gitleaks]",
                'version = "8.24.2"',
                'source = "github-release"',
                'risk = "medium"',
                f'checksum = "{checksum}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    repo = _InMemoryToolInstallRepo()
    downloader = _FakeGithubDownloader(payload=payload)
    provisioner = ToolProvisioner(
        install_repo=repo,
        registry_path=registry_path,
        github_downloader=downloader,
        install_root=tmp_path / "installed",
        now_provider=_fixed_now,
    )

    result = provisioner.provision("gitleaks")

    assert result.record.tool == "gitleaks"
    assert result.checksum == checksum
    assert result.artifact_path is not None
    assert result.record.metadata["strategy"] == "github-release"
    assert len(downloader.calls) == 1


def test_tool_provisioner_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.toml"
    registry_path.write_text(
        "\n".join(
            [
                "[tool.gitleaks]",
                'version = "8.24.2"',
                'source = "github-release"',
                'risk = "medium"',
                'checksum = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    provisioner = ToolProvisioner(
        install_repo=_InMemoryToolInstallRepo(),
        registry_path=registry_path,
        github_downloader=_FakeGithubDownloader(payload=b"different"),
        install_root=tmp_path / "installed",
        now_provider=_fixed_now,
    )

    with pytest.raises(ToolProvisioningError, match="checksum mismatch"):
        provisioner.provision("gitleaks")
