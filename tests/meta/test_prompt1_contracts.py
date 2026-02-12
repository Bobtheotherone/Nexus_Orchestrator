"""
nexus-orchestrator — Prompt 1 meta contract tests

File: tests/meta/test_prompt1_contracts.py
Last updated: 2026-02-12

Purpose
- Enforce deterministic Prompt 1 repository contracts for tooling, workflows, constraints, and scoped placeholders.

What this test file should cover
- META-1 pyproject contract (dev deps, extras, and tool config).
- META-2 CI workflow contract.
- META-3 security workflow contract.
- META-4 constraints registry coherence.
- META-5 placeholder enforcement scoped to explicit Phase 0 ownership.
- META-6 deterministic `scripts/repo_audit.py --json` bytes when present.

Functional requirements
- Tests must run offline and quickly.
- Contracts should be hard to satisfy with commented-out or cosmetic-only changes.

Non-functional requirements
- Keep assertions deterministic and explicit.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback for local runs.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"

REQUIRED_DEV_PACKAGES = (
    "ruff",
    "mypy",
    "pytest",
    "hypothesis",
    "pyyaml",
    "pytest-asyncio",
    "pytest-cov",
    "pip-audit",
)

REQUIRED_TOOL_REGISTRY_ENTRIES = (
    "ruff",
    "mypy",
    "pytest",
    "hypothesis",
    "pip-audit",
    "gitleaks",
)

BANNED_PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "todo_fixme",
        re.compile(r"(?m)^\s*(?:#|//|\*+)\s*(?:TODO|FIXME)\b"),
    ),
    ("raise_not_implemented", re.compile(r"raise\s+NotImplementedError")),
    ("pass_placeholder", re.compile(r"pass\s*#\s*placeholder", re.IGNORECASE)),
    ("angle_placeholder", re.compile(r"<\s*placeholder\s*>", re.IGNORECASE)),
)

REQUIRED_CONSTRAINT_FIELDS = (
    "id",
    "severity",
    "category",
    "description",
    "checker",
    "parameters",
    "requirement_links",
    "source",
)

NON_EMPTY_STRING_FIELDS = ("id", "severity", "category", "description", "checker", "source")

CONSTRAINT_ID_PATTERN = re.compile(r"^CON-[A-Z]{3,6}-\d{4}$")
BUILD_ORDER_PHASE_HEADER = re.compile(r"^##\s+Phase\s+(\d+)\s+—\s+.+$")


def _load_repo_blueprint_module() -> object:
    try:
        import nexus_orchestrator.repo_blueprint as module

        return module
    except ModuleNotFoundError:
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))
        import nexus_orchestrator.repo_blueprint as module

        return module


scan_placeholders_in_paths = _load_repo_blueprint_module().scan_placeholders_in_paths


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement_entry(entry: str) -> tuple[str, str]:
    normalized = entry.split(";", maxsplit=1)[0].strip()
    match = re.match(r"^([A-Za-z0-9_.-]+)\s*(.*)$", normalized)
    assert match is not None, f"Invalid dependency entry: {entry!r}"
    package_name = _canonical_name(match.group(1))
    specifier = match.group(2).strip()
    return package_name, specifier


def _is_pinned_or_tightly_constrained(specifier: str) -> bool:
    if not specifier:
        return False
    if specifier.startswith("=="):
        return bool(re.fullmatch(r"==\d+\.\d+\.\d+", specifier))
    clauses = [clause.strip() for clause in specifier.split(",") if clause.strip()]
    has_lower = any(clause.startswith((">", ">=")) for clause in clauses)
    has_upper = any(clause.startswith(("<", "<=")) for clause in clauses)
    return has_lower and has_upper


def _load_pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _dependency_map(entries: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        package_name, specifier = _parse_requirement_entry(entry)
        result[package_name] = specifier
    return result


def _load_workflow(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict), f"Workflow did not parse as mapping: {path}"
    return loaded


def _workflow_on_config(workflow: dict[str, Any]) -> dict[str, Any]:
    on_cfg = workflow.get("on")
    if isinstance(on_cfg, dict):
        return on_cfg
    # YAML 1.1 loaders can coerce "on" to boolean true in unquoted mappings.
    bool_key_cfg = workflow.get(True)
    if isinstance(bool_key_cfg, dict):
        return bool_key_cfg
    return {}


def _iter_workflow_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        return []
    steps: list[dict[str, Any]] = []
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        raw_steps = job.get("steps")
        if not isinstance(raw_steps, list):
            continue
        for step in raw_steps:
            if isinstance(step, dict):
                steps.append(step)
    return steps


def _iter_shell_commands(run_script: str) -> list[str]:
    commands: list[str] = []
    buffer = ""
    for raw_line in run_script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        buffer = f"{buffer} {line}".strip() if buffer else line
        if buffer.endswith("\\"):
            buffer = buffer[:-1].rstrip()
            continue
        for segment in re.split(r"\s*(?:&&|;)\s*", buffer):
            command = segment.strip()
            if command:
                commands.append(command)
        buffer = ""
    if buffer:
        commands.append(buffer)
    return commands


def _workflow_commands(workflow: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    for step in _iter_workflow_steps(workflow):
        run_script = step.get("run")
        if isinstance(run_script, str):
            commands.extend(_iter_shell_commands(run_script))
    return commands


def _workflow_uses(workflow: dict[str, Any]) -> list[str]:
    uses_entries: list[str] = []
    for step in _iter_workflow_steps(workflow):
        uses = step.get("uses")
        if isinstance(uses, str):
            uses_entries.append(uses.strip())
    return uses_entries


def _load_tools_registry() -> dict[str, Any]:
    return tomllib.loads((REPO_ROOT / "tools/registry.toml").read_text(encoding="utf-8"))


def _makefile_commands() -> str:
    return (REPO_ROOT / "Makefile").read_text(encoding="utf-8")


def _invokes_tool(command: str, tool_name: str) -> bool:
    pattern = re.compile(rf"^(?:python(?:3)?\s+-m\s+)?{re.escape(tool_name)}\b")
    return pattern.search(command) is not None


def _phase_files_from_build_order(phase_number: int) -> list[str]:
    lines = (REPO_ROOT / "docs/BUILD_ORDER.md").read_text(encoding="utf-8").splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        match = BUILD_ORDER_PHASE_HEADER.match(line.strip())
        if match and int(match.group(1)) == phase_number:
            start_index = index + 1
            break
    assert start_index is not None, f"Phase {phase_number} section not found in BUILD_ORDER."
    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if lines[index].startswith("## "):
            end_index = index
            break
    file_pattern = re.compile(r"`([^`]+)`")
    files: list[str] = []
    for line in lines[start_index:end_index]:
        stripped = line.strip()
        if re.match(r"^(?:\d+\.\s+|- |\* )", stripped):
            files.extend(match.strip() for match in file_pattern.findall(line))
    return sorted(set(files))


def _load_constraint_records() -> list[tuple[str, dict[str, Any]]]:
    registry_dir = REPO_ROOT / "constraints/registry"
    registry_files = sorted(registry_dir.glob("*.yaml"))
    assert registry_files, "No constraints/registry/*.yaml files found."
    records: list[tuple[str, dict[str, Any]]] = []
    for registry_file in registry_files:
        loaded = yaml.safe_load(registry_file.read_text(encoding="utf-8"))
        assert isinstance(loaded, list), f"{registry_file} must contain a YAML list of constraints."
        for entry in loaded:
            assert isinstance(entry, dict), f"{registry_file} has non-mapping constraint entry."
            records.append((registry_file.relative_to(REPO_ROOT).as_posix(), entry))
    assert records, "Constraint registry files are empty."
    return records


def test_meta_1_pyproject_contract() -> None:
    pyproject = _load_pyproject()
    project = pyproject.get("project")
    assert isinstance(project, dict), "pyproject [project] table is missing."
    optional_deps = project.get("optional-dependencies")
    assert isinstance(optional_deps, dict), "[project.optional-dependencies] is missing."
    assert "providers" in optional_deps, "providers extra must exist."
    providers_entries = optional_deps.get("providers")
    assert isinstance(providers_entries, list), "providers extra must be a dependency list."
    dev_entries = optional_deps.get("dev")
    assert isinstance(dev_entries, list), "dev extra must be a dependency list."
    dev_dep_map = _dependency_map(dev_entries)
    for package_name in REQUIRED_DEV_PACKAGES:
        assert package_name in dev_dep_map, f"dev dependency missing: {package_name}"
        specifier = dev_dep_map[package_name]
        assert _is_pinned_or_tightly_constrained(specifier), (
            "Dependency must be pinned (==x.y.z) or tightly constrained (>=x,<y).",
            package_name,
            specifier,
        )
    runtime_entries = project.get("dependencies")
    assert isinstance(runtime_entries, list), "[project.dependencies] must be a list."
    runtime_names = {
        _parse_requirement_entry(entry)[0] for entry in runtime_entries if isinstance(entry, str)
    }
    provider_names = {
        _parse_requirement_entry(entry)[0]
        for entry in providers_entries
        if isinstance(entry, str) and entry.strip()
    }
    assert provider_names.isdisjoint(set(dev_dep_map)), (
        "Provider dependencies must not be part of the dev extra baseline install."
    )
    assert runtime_names.isdisjoint(provider_names), (
        "Provider dependencies must stay optional and not appear in base runtime deps."
    )
    tool = pyproject.get("tool")
    assert isinstance(tool, dict), "[tool] table is missing."
    assert isinstance(tool.get("ruff"), dict), "ruff config must exist under [tool.ruff]."
    assert isinstance(tool.get("mypy"), dict), "mypy config must exist under [tool.mypy]."
    pytest_cfg = tool.get("pytest")
    assert isinstance(pytest_cfg, dict), "pytest config must exist under [tool.pytest]."
    assert isinstance(pytest_cfg.get("ini_options"), dict), (
        "pytest ini options must exist under [tool.pytest.ini_options]."
    )


def test_meta_1b_tool_registry_contract() -> None:
    registry = _load_tools_registry()
    tool_table = registry.get("tool")
    assert isinstance(tool_table, dict), "tools/registry.toml must define [tool.*] entries."
    for tool_name in REQUIRED_TOOL_REGISTRY_ENTRIES:
        assert tool_name in tool_table, f"Missing tool registry entry: {tool_name}"
        entry = tool_table[tool_name]
        assert isinstance(entry, dict), f"[tool.{tool_name}] must be a table."
        version = entry.get("version")
        source = entry.get("source")
        risk = entry.get("risk")
        assert isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version), (
            f"[tool.{tool_name}] version must be pinned to x.y.z."
        )
        assert isinstance(source, str) and source.strip(), (
            f"[tool.{tool_name}] source must be a non-empty string."
        )
        assert risk in {"low", "medium", "high"}, (
            f"[tool.{tool_name}] risk must be one of low/medium/high."
        )


def test_meta_2_ci_workflow_contract() -> None:
    workflow = _load_workflow(REPO_ROOT / ".github/workflows/ci.yml")
    on_cfg = _workflow_on_config(workflow)
    assert "push" in on_cfg, "CI workflow must trigger on push."
    assert "pull_request" in on_cfg, "CI workflow must trigger on pull_request."
    steps = _iter_workflow_steps(workflow)
    assert steps, "CI workflow has no executable steps."
    setup_steps = [
        step
        for step in steps
        if isinstance(step.get("uses"), str)
        and str(step["uses"]).strip().startswith("actions/setup-python@")
    ]
    assert setup_steps, "CI workflow must use actions/setup-python."
    has_python_311 = any(
        isinstance(step.get("with"), dict)
        and str(step["with"].get("python-version", "")).strip().strip("'\"") == "3.11"
        for step in setup_steps
    )
    assert has_python_311, "CI must pin setup-python to version 3.11."
    commands = _workflow_commands(workflow)
    assert any(
        re.search(r"^(?:python(?:3)?\s+-m\s+pip|pip(?:3)?)\s+install\b.*\.\[dev\]", command)
        for command in commands
    ), "CI must install `.[dev]`."
    assert any(
        _invokes_tool(command, "ruff") and re.search(r"\bruff\s+check\b", command)
        for command in commands
    ), "CI must run `ruff check`."
    assert any(
        _invokes_tool(command, "ruff")
        and re.search(r"\bruff\s+format\b", command)
        and "--check" in command
        for command in commands
    ), "CI must run `ruff format --check`."
    assert any(_invokes_tool(command, "mypy") for command in commands), "CI must run mypy."
    assert any(
        _invokes_tool(command, "pytest") and "tests/meta" in command and "tests/unit" in command
        for command in commands
    ), "CI must run pytest including tests/meta and tests/unit."


def test_meta_2b_ci_local_gate_parity() -> None:
    workflow = _load_workflow(REPO_ROOT / ".github/workflows/ci.yml")
    ci_commands = "\n".join(_workflow_commands(workflow))
    makefile_text = _makefile_commands()
    required_fragments = (
        "ruff check src/ tests/",
        "ruff format --check src/ tests/",
        "mypy src/nexus_orchestrator/",
        "pytest tests/meta tests/unit tests/integration tests/smoke -v",
    )
    for fragment in required_fragments:
        assert fragment in ci_commands, f"CI missing gate command fragment: {fragment}"
        assert fragment in makefile_text, f"Makefile missing gate command fragment: {fragment}"


def test_meta_3_security_workflow_contract() -> None:
    workflow = _load_workflow(REPO_ROOT / ".github/workflows/security.yml")
    on_cfg = _workflow_on_config(workflow)
    assert on_cfg, "Security workflow must define `on` triggers."
    push_cfg = on_cfg.get("push")
    assert isinstance(push_cfg, dict), "Security workflow must include push trigger."
    branches = push_cfg.get("branches")
    if isinstance(branches, str):
        push_branches = [branches.strip()]
    elif isinstance(branches, list):
        push_branches = [str(branch).strip() for branch in branches]
    else:
        push_branches = []
    assert "main" in push_branches, "Security workflow push trigger must include main."
    schedule_cfg = on_cfg.get("schedule")
    assert isinstance(schedule_cfg, list) and schedule_cfg, (
        "Security workflow must include a non-empty schedule trigger."
    )
    commands = _workflow_commands(workflow)
    uses_entries = _workflow_uses(workflow)
    pip_audit_present = any("pip-audit" in command.lower() for command in commands) or any(
        "pip-audit" in uses.lower() for uses in uses_entries
    )
    assert pip_audit_present, "Security workflow must include a pip-audit step."
    strict_pip_audit_commands = [
        command
        for command in commands
        if _invokes_tool(command, "pip_audit") and "--strict" in command
    ]
    assert strict_pip_audit_commands, "Security workflow pip_audit command must include --strict."
    for command in strict_pip_audit_commands:
        assert "--skip-editable" in command, (
            "Security workflow pip_audit commands using --strict must also use --skip-editable."
        )
    secret_scan_tokens = ("gitleaks", "trufflehog", "detect-secrets", "secretlint", "ggshield")
    secret_scan_present = any(
        any(token in command.lower() for token in secret_scan_tokens) for command in commands
    ) or any(any(token in uses.lower() for token in secret_scan_tokens) for uses in uses_entries)
    assert secret_scan_present, "Security workflow must include a secret scanning step."


def test_meta_4_constraints_registry_coherence() -> None:
    records = _load_constraint_records()
    ids: list[str] = []
    for source_path, record in records:
        for field_name in REQUIRED_CONSTRAINT_FIELDS:
            assert field_name in record, f"{source_path} missing required field {field_name!r}"
        for field_name in NON_EMPTY_STRING_FIELDS:
            value = record.get(field_name)
            assert isinstance(value, str) and value.strip(), (
                f"{source_path} field {field_name!r} must be non-empty string."
            )
        assert isinstance(record["parameters"], dict), (
            f"{source_path} field 'parameters' must be a mapping."
        )
        assert isinstance(record["requirement_links"], list), (
            f"{source_path} field 'requirement_links' must be a list."
        )
        constraint_id = str(record["id"])
        assert CONSTRAINT_ID_PATTERN.fullmatch(constraint_id), (
            f"{source_path} has invalid constraint ID format: {constraint_id!r}"
        )
        ids.append(constraint_id)
    assert len(ids) == len(set(ids)), "Constraint IDs must be unique across registry files."
    id_set = set(ids)
    assert {"CON-STY-0001", "CON-STY-0002"}.issubset(id_set), (
        "Registry must include mandatory style constraints CON-STY-0001 and CON-STY-0002."
    )
    style_doc = (REPO_ROOT / "docs/quality/STYLE_AND_LINT.md").read_text(encoding="utf-8")
    if re.search(r"\bCON-STY-0003\b", style_doc):
        assert "CON-STY-0003" in id_set, (
            "STYLE_AND_LINT references CON-STY-0003, so it must exist in registry."
        )
    for source_path, record in records:
        parameters = record["parameters"]
        schema_path_value = parameters.get("schema_path")
        if schema_path_value is None:
            continue
        assert isinstance(schema_path_value, str) and schema_path_value.strip(), (
            f"{source_path} has empty schema_path."
        )
        schema_path = Path(schema_path_value.strip())
        assert not schema_path.is_absolute(), f"{source_path} schema_path must be relative."
        assert ".." not in schema_path.parts, f"{source_path} schema_path must not traverse upward."
        assert (REPO_ROOT / schema_path).exists(), (
            f"{source_path} schema_path does not exist: {schema_path.as_posix()!r}"
        )


def test_meta_4b_constraint_and_tool_registry_coherence() -> None:
    records = _load_constraint_records()
    registry = _load_tools_registry()
    tool_table = registry.get("tool")
    assert isinstance(tool_table, dict), "tools/registry.toml must define [tool.*] entries."
    style_ids_to_checker = {record["id"]: record["checker"] for _src, record in records}
    assert style_ids_to_checker.get("CON-STY-0001") == "lint_checker"
    assert style_ids_to_checker.get("CON-STY-0002") == "lint_checker"
    assert style_ids_to_checker.get("CON-STY-0003") == "typecheck_checker"
    assert "ruff" in tool_table, "Lint constraints require ruff in tool registry."
    assert "mypy" in tool_table, "Typecheck constraints require mypy in tool registry."
    dependency_audit_constraints = [
        record
        for _src, record in records
        if record.get("checker") == "security_checker"
        and isinstance(record.get("parameters"), dict)
        and record["parameters"].get("scan_type") == "dependency_audit"
    ]
    if dependency_audit_constraints:
        assert "pip-audit" in tool_table, (
            "Dependency audit constraints require pip-audit in tool registry."
        )


def test_meta_5_placeholder_enforcement_scoped_to_phase0_owned_files() -> None:
    phase0_files = _phase_files_from_build_order(0)
    assert phase0_files, "Phase 0 must explicitly list owned files in BUILD_ORDER."
    for relative_path in phase0_files:
        assert (REPO_ROOT / relative_path).exists(), f"Phase 0 listed path missing: {relative_path}"
    file_map_text = (REPO_ROOT / "docs/FILE_MAP.md").read_text(encoding="utf-8")
    assert "Phase 0" in file_map_text, "FILE_MAP must formalize Phase 0 ownership."
    for relative_path in phase0_files:
        assert f"`{relative_path}`" in file_map_text, (
            f"FILE_MAP must include Phase 0 owned path {relative_path!r}."
        )
    repo_blueprint_scan = scan_placeholders_in_paths(REPO_ROOT, phase0_files, set(phase0_files))
    assert repo_blueprint_scan["warnings"] == []
    assert repo_blueprint_scan["errors"] == []
    for relative_path in phase0_files:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        for pattern_name, pattern in BANNED_PLACEHOLDER_PATTERNS:
            assert pattern.search(text) is None, (
                f"Phase 0 file contains banned placeholder pattern {pattern_name!r}: "
                f"{relative_path}"
            )


@settings(max_examples=24, deadline=None)
@given(
    prefix=st.sampled_from(["#", "//", "*", "***"]),
    marker=st.sampled_from(["TODO", "FIXME"]),
    spacing=st.text(alphabet=" \t", min_size=0, max_size=4),
    trailer=st.text(
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 _-.",
        max_size=24,
    ),
)
def test_meta_5_property_placeholder_pattern_detection_and_scope(
    prefix: str,
    marker: str,
    spacing: str,
    trailer: str,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        owned_path = temp_root / "owned.py"
        suffix = f" {trailer}" if trailer else ""
        line = f"{prefix}{spacing}{marker}{suffix}"
        owned_path.write_text(f"{line}\npass\n", encoding="utf-8")
        scan = scan_placeholders_in_paths(temp_root, ["owned.py"], {"owned.py"})
        assert scan["warnings"] == []
        assert len(scan["errors"]) == 1
        assert scan["errors"][0]["path"] == "owned.py"
        assert "todo_fixme" in set(scan["errors"][0]["matches"])


def test_meta_6_repo_audit_json_output_is_byte_stable_when_available() -> None:
    script_path = REPO_ROOT / "scripts/repo_audit.py"
    if not script_path.exists():
        pytest.skip("scripts/repo_audit.py is not present in this repository.")
    command = [sys.executable, str(script_path), "--json"]
    first = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    second = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    assert first.stdout == second.stdout
    parsed = json.loads(first.stdout.decode("utf-8"))
    assert isinstance(parsed, dict)
    metadata = parsed.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("source_strategy") in {"git_ls_files", "filesystem_fallback"}
