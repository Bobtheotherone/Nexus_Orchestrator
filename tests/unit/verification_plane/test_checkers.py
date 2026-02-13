"""
nexus-orchestrator — test skeleton

File: tests/unit/verification_plane/test_checkers.py
Last updated: 2026-02-11

Purpose
- Validate checker interface contracts and result normalization.

What this test file should cover
- All checkers produce CheckResult with required fields.
- Timeout behavior is handled uniformly.
- Tool version capture is recorded.

Functional requirements
- No real tool execution; use fakes.

Non-functional requirements
- Deterministic.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Final

import pytest

from nexus_orchestrator.security.redaction import REDACTED_VALUE
from nexus_orchestrator.verification_plane.checkers import (
    BuildChecker,
    CheckerContext,
    CheckStatus,
    CommandExecutor,
    CommandResult,
    CommandSpec,
    DocumentationChecker,
    LintChecker,
    PerformanceChecker,
    ReliabilityChecker,
    SchemaChecker,
    ScopeChecker,
    SecurityChecker,
    TypecheckChecker,
    Violation,
    normalize_artifact_paths,
    normalize_violations,
)
from nexus_orchestrator.verification_plane.checkers import (
    TestChecker as NexusTestChecker,
)
from nexus_orchestrator.verification_plane.evidence import (
    EvidenceWriter,
    export_audit_bundle,
    verify_integrity,
)


@dataclass(frozen=True, slots=True)
class FakeOutcome:
    exit_code: int | None = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 5
    timed_out: bool = False
    error: str | None = None


class FakeExecutor(CommandExecutor):
    """Deterministic fake executor with virtual-time accounting."""

    def __init__(
        self, responses: dict[tuple[str, ...], FakeOutcome | Exception] | None = None
    ) -> None:
        self.responses = responses or {}
        self.calls: list[CommandSpec] = []
        self.virtual_time_ms = 0

    async def run(self, spec: CommandSpec) -> CommandResult:
        self.calls.append(spec)

        key = tuple(spec.argv)
        outcome = self.responses.get(key)
        if isinstance(outcome, Exception):
            raise outcome

        if outcome is None:
            if spec.argv and spec.argv[-1] == "--version":
                outcome = FakeOutcome(stdout=f"{spec.argv[0]} 1.0.0")
            else:
                outcome = FakeOutcome()

        self.virtual_time_ms += outcome.duration_ms

        return CommandResult(
            argv=tuple(spec.argv),
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            duration_ms=outcome.duration_ms,
            timed_out=outcome.timed_out,
            error=outcome.error,
        )


class TimeoutScopeProvider:
    async def get_changed_paths(
        self,
        *,
        context: CheckerContext,
        timeout_seconds: float,
    ) -> tuple[object, ...]:
        _ = context
        _ = timeout_seconds
        raise TimeoutError("simulated timeout")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    _write_text(
        workspace / "src" / "mod.py",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
    )
    _write_text(
        workspace / "src" / "reliable_client.py",
        (
            "import requests\n\n"
            "def fetch(url: str) -> object:\n"
            "    retries = 3\n"
            "    return requests.get(url, timeout=5)\n"
        ),
    )
    _write_text(workspace / "docs" / "guide.md", "# Guide\n")

    _write_text(
        workspace / "constraints" / "registry" / "000_base_constraints.yaml",
        (
            "- id: CON-ARC-0001\n"
            "  severity: must\n"
            "  category: structural\n"
            "  description: scope check\n"
            "  checker: scope_checker\n"
            "  parameters: {}\n"
            "  requirement_links: []\n"
            "  source: manual\n"
        ),
    )

    repo_orchestrator = Path("orchestrator.toml")
    _write_text(workspace / "orchestrator.toml", repo_orchestrator.read_text(encoding="utf-8"))
    _write_text(workspace / "state" / "ledger.json", "{}\n")

    return workspace


@pytest.mark.asyncio
async def test_checkers_produce_required_fields_and_tool_versions(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)
    executor = FakeExecutor()

    cases: tuple[tuple[str, object, dict[str, object], str], ...] = (
        (
            "build",
            BuildChecker(),
            {"command": ["python", "-m", "compileall", "src"], "constraint_ids": ["CON-COR-0001"]},
            "python",
        ),
        (
            "lint",
            LintChecker(),
            {"check_type": "lint", "target": "src", "constraint_ids": ["CON-STY-0001"]},
            "ruff",
        ),
        (
            "typecheck",
            TypecheckChecker(),
            {"mode": "strict", "target": "src", "constraint_ids": ["CON-STY-0003"]},
            "mypy",
        ),
        (
            "test",
            NexusTestChecker(),
            {"test_type": "unit", "constraint_ids": ["CON-COR-0002"]},
            "pytest",
        ),
        (
            "security",
            SecurityChecker(),
            {
                "scan_type": "secret_patterns",
                "patterns": ["UNLIKELY_TEST_TOKEN_VALUE"],
                "scan_paths": ["src"],
                "constraint_ids": ["CON-SEC-0001"],
            },
            "security_checker",
        ),
        (
            "scope",
            ScopeChecker(),
            {
                "changed_files": [{"path": "src/mod.py", "kind": "modified"}],
                "explicit_files": ["src/mod.py"],
                "constraint_ids": ["CON-ARC-0001"],
            },
            "scope_checker",
        ),
        (
            "schema",
            SchemaChecker(),
            {
                "target_kind": "constraint_registry",
                "registry_path": "constraints/registry",
                "constraint_ids": ["CON-REG-0001"],
            },
            "schema_checker",
        ),
        (
            "documentation",
            DocumentationChecker(),
            {
                "changed_files": ["src/mod.py", "docs/guide.md"],
                "constraint_ids": ["CON-DOC-0001"],
            },
            "documentation_checker",
        ),
        (
            "reliability",
            ReliabilityChecker(),
            {
                "include_globs": ["src/**/*.py"],
                "constraint_ids": ["CON-REL-0001"],
            },
            "reliability_checker",
        ),
        (
            "performance",
            PerformanceChecker(),
            {"command": ["pytest", "tests/smoke", "-q"], "constraint_ids": ["CON-PERF-0001"]},
            "pytest",
        ),
    )

    required_fields: Final[set[str]] = {
        "status",
        "violations",
        "covered_constraint_ids",
        "tool_versions",
        "artifact_paths",
        "logs_path",
        "duration_ms",
        "metadata",
        "checker_id",
        "stage",
    }

    for _, checker, config, expected_tool in cases:
        context = CheckerContext(
            workspace_path=str(workspace),
            config=config,
            command_executor=executor,
        )
        result = await checker.check(context)  # type: ignore[attr-defined]

        payload = result.to_dict()
        assert required_fields.issubset(payload.keys())
        assert expected_tool in result.tool_versions
        assert tuple(sorted(result.artifact_paths)) == result.artifact_paths
        assert (
            tuple(sorted(result.violations, key=lambda item: item.sort_key())) == result.violations
        )


@pytest.mark.asyncio
async def test_timeout_behavior_is_uniform_and_deterministic(tmp_path: Path) -> None:
    workspace = _seed_workspace(tmp_path)

    command_timeouts: tuple[tuple[object, tuple[str, ...], str], ...] = (
        (BuildChecker(), ("python", "-m", "compileall", "src"), "build_checker.timeout"),
        (LintChecker(), ("ruff", "check", "src"), "lint_checker.timeout"),
        (TypecheckChecker(), ("mypy", "src"), "typecheck_checker.timeout"),
        (NexusTestChecker(), ("pytest", "tests/unit", "-q"), "test_checker.timeout"),
        (PerformanceChecker(), ("pytest", "tests/smoke", "-q"), "performance_checker.timeout"),
    )

    for checker, command, expected_code in command_timeouts:
        executor = FakeExecutor(
            responses={
                command: FakeOutcome(
                    exit_code=None,
                    timed_out=True,
                    error="command timed out",
                    duration_ms=9,
                )
            }
        )
        context = CheckerContext(
            workspace_path=str(workspace),
            config={"command": list(command), "constraint_ids": ["CON-TIM-0001"]},
            command_executor=executor,
        )
        result = await checker.check(context)  # type: ignore[attr-defined]

        assert result.status is CheckStatus.TIMEOUT
        assert any(item.code == expected_code for item in result.violations)

    security_executor = FakeExecutor(
        responses={
            ("pip-audit", "-f", "json"): FakeOutcome(
                exit_code=None,
                timed_out=True,
                error="dependency audit timed out",
            )
        }
    )
    security_result = await SecurityChecker().check(
        CheckerContext(
            workspace_path=str(workspace),
            config={
                "scan_type": "dependency_audit",
                "dependency_audit_command": ["pip-audit", "-f", "json"],
                "dependency_audit_required": True,
            },
            command_executor=security_executor,
        )
    )
    assert security_result.status is CheckStatus.TIMEOUT
    assert any(
        item.code == "security.dependency_audit.timeout" for item in security_result.violations
    )

    scope_result = await ScopeChecker(change_provider=TimeoutScopeProvider()).check(
        CheckerContext(
            workspace_path=str(workspace),
            config={"explicit_files": ["src/mod.py"]},
            command_executor=FakeExecutor(),
        )
    )
    assert scope_result.status is CheckStatus.TIMEOUT
    assert any(item.code == "scope.timeout" for item in scope_result.violations)

    schema_result = await SchemaChecker().check(
        CheckerContext(
            workspace_path=str(workspace),
            config={"force_timeout": True, "target_kind": "constraint_registry"},
            command_executor=FakeExecutor(),
        )
    )
    documentation_result = await DocumentationChecker().check(
        CheckerContext(
            workspace_path=str(workspace),
            config={"force_timeout": True, "changed_files": ["docs/guide.md"]},
            command_executor=FakeExecutor(),
        )
    )
    reliability_result = await ReliabilityChecker().check(
        CheckerContext(
            workspace_path=str(workspace),
            config={"force_timeout": True},
            command_executor=FakeExecutor(),
        )
    )

    assert schema_result.status is CheckStatus.TIMEOUT
    assert documentation_result.status is CheckStatus.TIMEOUT
    assert reliability_result.status is CheckStatus.TIMEOUT


def test_violation_normalization_is_stable_under_deterministic_fuzz() -> None:
    rng = Random(20260212)

    for _ in range(50):
        raw: list[Violation] = []
        for _inner in range(12):
            raw.append(
                Violation(
                    constraint_id=f"CON-TST-{rng.randint(1, 9):04d}",
                    code=f"code_{rng.randint(0, 7)}",
                    message=f"msg_{rng.randint(0, 99)}",
                    path=f"src/file_{rng.randint(0, 5)}.py",
                    line=rng.randint(1, 300),
                    column=rng.randint(1, 120),
                )
            )

        shuffled_a = list(raw)
        shuffled_b = list(raw)
        rng.shuffle(shuffled_a)
        rng.shuffle(shuffled_b)

        normalized_a = normalize_violations(shuffled_a)
        normalized_b = normalize_violations(shuffled_b)

        assert normalized_a == normalized_b
        assert tuple(sorted(normalized_a, key=lambda item: item.sort_key())) == normalized_a


def test_artifact_path_normalization_is_stable_under_deterministic_fuzz() -> None:
    rng = Random(7)
    for _ in range(50):
        base = [f"artifacts/r{rng.randint(0, 20)}.json" for _inner in range(20)]
        shuffled_a = list(base)
        shuffled_b = list(base)
        rng.shuffle(shuffled_a)
        rng.shuffle(shuffled_b)

        normalized_a = normalize_artifact_paths(shuffled_a)
        normalized_b = normalize_artifact_paths(shuffled_b)

        assert normalized_a == normalized_b
        assert normalized_a == tuple(sorted(set(normalized_a)))


def test_evidence_writer_integrity_and_audit_bundle(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    writer = EvidenceWriter(root)

    source = tmp_path / "secret.txt"
    source.write_text("API_KEY=sk-THISSHOULDBEREDACTED123456", encoding="utf-8")

    result = writer.write_evidence(
        run_id="run-001",
        work_item_id="wi-001",
        stage="security",
        evidence_id="evi-001",
        metadata={"note": "token=sk-THISSHOULDBEREDACTED123456"},
        logs={"security.log": "password=super-secret"},
        artifacts={"report.json": {"secret": "sk-THISSHOULDBEREDACTED123456"}},
        copied_artifacts={"copied.txt": source},
    )

    metadata_text = result.metadata_path.read_text(encoding="utf-8")
    assert "THISSHOULDBEREDACTED" not in metadata_text
    assert REDACTED_VALUE in metadata_text

    initial_integrity = verify_integrity(result.evidence_dir)
    assert initial_integrity.is_valid

    tamper_target = result.evidence_dir / "logs" / "security.log"
    tamper_target.write_text("tampered", encoding="utf-8")
    tampered_integrity = verify_integrity(result.evidence_dir)
    assert not tampered_integrity.is_valid
    assert "logs/security.log" in tampered_integrity.hash_mismatches

    result_missing = writer.write_evidence(
        run_id="run-001",
        work_item_id="wi-001",
        stage="security",
        evidence_id="evi-002",
        artifacts={"payload.json": {"ok": True}},
    )
    missing_target = result_missing.evidence_dir / "artifacts" / "payload.json"
    missing_target.unlink()
    missing_integrity = verify_integrity(result_missing.evidence_dir)
    assert not missing_integrity.is_valid
    assert "artifacts/payload.json" in missing_integrity.missing_paths

    result_bundle = writer.write_evidence(
        run_id="run-001",
        work_item_id="wi-001",
        stage="security",
        evidence_id="evi-003",
        artifacts={"payload.json": {"ok": True}},
    )
    bundle_path = export_audit_bundle(
        result_bundle.evidence_dir, output_path=tmp_path / "bundle.zip"
    )
    assert bundle_path.exists()

    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = sorted(archive.namelist())

    prefix = result_bundle.evidence_dir.name
    assert f"{prefix}/manifest.json" in names
    assert f"{prefix}/metadata.json" in names

    with pytest.raises(FileExistsError):
        writer.write_evidence(
            run_id="run-001",
            work_item_id="wi-001",
            stage="security",
            evidence_id="evi-003",
            artifacts={"payload.json": {"ok": True}},
        )


def test_evidence_manifest_entries_are_deterministic(tmp_path: Path) -> None:
    writer = EvidenceWriter(tmp_path / "evidence")

    first = writer.write_evidence(
        run_id="run-002",
        work_item_id="wi-002",
        stage="build",
        evidence_id="evi-a",
        artifacts={"b.txt": "2", "a.txt": "1"},
    )
    second = writer.write_evidence(
        run_id="run-002",
        work_item_id="wi-002",
        stage="build",
        evidence_id="evi-b",
        artifacts={"a.txt": "1", "b.txt": "2"},
    )

    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    first_entries = first_manifest["entries"]
    second_entries = second_manifest["entries"]

    assert [entry["path"] for entry in first_entries] == sorted(
        entry["path"] for entry in first_entries
    )
    assert [entry["path"] for entry in second_entries] == sorted(
        entry["path"] for entry in second_entries
    )

    first_artifacts = [
        entry for entry in first_entries if str(entry["path"]).startswith("artifacts/")
    ]
    second_artifacts = [
        entry for entry in second_entries if str(entry["path"]).startswith("artifacts/")
    ]
    assert first_artifacts == second_artifacts
