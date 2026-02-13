"""
nexus-orchestrator — test skeleton

File: tests/unit/synthesis_plane/test_tools.py
Last updated: 2026-02-13

Purpose
- Validate synthesis-plane tool request policy decisions and audit payloads.

What this test file should cover
- Allowlist auto-approval with pinned versions.
- Denylist hard deny behavior.
- Unknown tool review-required behavior.
- Deterministic and JSON-serializable audit output.

Functional requirements
- No provider calls.

Non-functional requirements
- Deterministic.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.synthesis_plane.tools import ToolRequest, ToolRequestHandler

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ModuleNotFoundError:
    HYPOTHESIS_AVAILABLE = False
else:
    HYPOTHESIS_AVAILABLE = True

if TYPE_CHECKING:
    from nexus_orchestrator.synthesis_plane.tools import JSONValue


def _request(tool: str, *, version: str | None = None) -> ToolRequest:
    return ToolRequest(
        tool=tool,
        version=version,
        purpose="Run project checks",
        scope="workspace tooling",
        provenance="unit-test",
        expected_benefit="faster deterministic feedback",
        requested_by_role="toolsmith",
        metadata={"work_item_id": "WI-001"},
    )


def _write_registry(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


@dataclass(slots=True)
class _RecordingAuditSink:
    rows: list[dict[str, JSONValue]] = field(default_factory=list)

    def record_tool_decision(self, decision: dict[str, JSONValue]) -> None:
        self.rows.append(decision)


def test_allowlisted_tool_is_auto_approved_with_pinned_version(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.2.3"
        source = "pypi"
        risk = "low"
        """,
    )
    handler = ToolRequestHandler(registry_path=registry)

    result = handler.handle_request(_request("ruff"))

    assert result.decision == "approved"
    assert result.status == "approved"
    assert result.resolved_version == "1.2.3"
    assert result.pinned_version == "1.2.3"
    assert result.policy_source == "allowlist"
    assert result.review_required is False


def test_allowlisted_version_mismatch_requires_review(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.2.3"
        source = "pypi"
        risk = "low"
        """,
    )
    handler = ToolRequestHandler(registry_path=registry)

    result = handler.handle_request(_request("ruff", version="9.9.9"))

    assert result.decision == "review_required"
    assert result.resolved_version == "1.2.3"
    assert result.policy_source == "allowlist_version_mismatch"
    assert result.review_required is True


def test_denylist_always_hard_denies_even_if_allowlisted(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.2.3"
        source = "pypi"
        risk = "low"

        [denylist]
        tools = ["ruff"]
        """,
    )
    handler = ToolRequestHandler(registry_path=registry)

    result = handler.handle_request(_request("ruff"))

    assert result.decision == "denied"
    assert result.policy_source == "denylist"
    assert result.review_required is False


def test_unknown_tool_is_marked_review_required(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.2.3"
        source = "pypi"
        risk = "low"
        """,
    )
    handler = ToolRequestHandler(registry_path=registry)

    result = handler.handle_request(_request("unknown-tool"))

    assert result.decision == "review_required"
    assert result.policy_source == "unlisted_tool"
    assert result.review_required is True


def test_decisions_are_deterministic_and_json_serializable(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.pytest]
        version = "9.0.2"
        source = "pypi"
        risk = "low"
        """,
    )
    handler = ToolRequestHandler(registry_path=registry)
    request = _request("pytest")

    first = handler.handle_request(request)
    second = handler.handle_request(request)

    assert first.to_dict() == second.to_dict()
    assert first.audit_id == second.audit_id
    assert json.dumps(first.to_dict(), sort_keys=True)
    assert first.to_json() == second.to_json()


def test_handler_uses_stable_pinning_snapshot(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.0.0"
        source = "pypi"
        risk = "low"
        """,
    )
    handler = ToolRequestHandler(registry_path=registry)

    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "2.0.0"
        source = "pypi"
        risk = "low"
        """,
    )

    result_with_loaded_snapshot = handler.handle_request(_request("ruff"))
    result_with_reloaded_handler = ToolRequestHandler(registry_path=registry).handle_request(
        _request("ruff")
    )

    assert result_with_loaded_snapshot.resolved_version == "1.0.0"
    assert result_with_reloaded_handler.resolved_version == "2.0.0"


def test_audit_sink_receives_decision_payload(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.2.3"
        source = "pypi"
        risk = "low"
        """,
    )
    sink = _RecordingAuditSink()
    handler = ToolRequestHandler(registry_path=registry, audit_sink=sink)

    result = handler.handle_request(_request("ruff"))

    assert len(sink.rows) == 1
    assert sink.rows[0] == result.to_dict()
    assert json.dumps(sink.rows[0], sort_keys=True)


def test_registry_rejects_allowlist_entries_without_pinned_version(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        source = "pypi"
        risk = "low"
        """,
    )

    with pytest.raises(ValueError, match="missing pinned version"):
        ToolRequestHandler(registry_path=registry)


def test_handler_strictness_dial_is_configurable_and_serialized(tmp_path: Path) -> None:
    registry = tmp_path / "registry.toml"
    _write_registry(
        registry,
        """
        [tool.ruff]
        version = "1.2.3"
        source = "pypi"
        risk = "low"
        """,
    )
    handler = ToolRequestHandler(registry_path=registry, strictness="permissive")

    assert handler.strictness == "permissive"
    assert handler.policy_snapshot()["strictness"] == "permissive"

    with pytest.raises(ValueError, match="strictness must be either"):
        ToolRequestHandler(registry_path=registry, strictness="invalid")


def test_request_metadata_must_be_json_serializable() -> None:
    with pytest.raises(ValueError, match="unsupported type"):
        ToolRequest(
            tool="ruff",
            purpose="Run checks",
            scope="workspace tooling",
            provenance="unit-test",
            expected_benefit="faster feedback",
            requested_by_role="toolsmith",
            metadata={"bad": object()},
        )


if HYPOTHESIS_AVAILABLE:

    @given(order=st.permutations(("ruff", "pytest", "mypy")))
    @settings(max_examples=20, deadline=None)
    def test_registry_entry_order_does_not_change_decision_determinism(
        order: tuple[str, str, str],
    ) -> None:
        entries = {
            "ruff": ('[tool.ruff]\nversion = "1.2.3"\nsource = "pypi"\nrisk = "low"\n',),
            "pytest": ('[tool.pytest]\nversion = "9.0.2"\nsource = "pypi"\nrisk = "low"\n',),
            "mypy": ('[tool.mypy]\nversion = "1.19.1"\nsource = "pypi"\nrisk = "low"\n',),
        }
        chunks = [entries[name][0] for name in order]
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = Path(temp_dir) / "registry.toml"
            _write_registry(registry, "\n".join(chunks))

            handler = ToolRequestHandler(registry_path=registry)
            decision = handler.handle_request(_request("pytest"))

            assert decision.decision == "approved"
            assert decision.resolved_version == "9.0.2"
            assert decision.audit_id == handler.handle_request(_request("pytest")).audit_id
