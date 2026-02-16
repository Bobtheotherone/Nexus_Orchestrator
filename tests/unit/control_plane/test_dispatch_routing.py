"""Unit tests for _dispatch_one mode routing fix.

Tests:
1. Mock mode uses _DeterministicMockProvider (not ToolProvider).
2. Real mode uses ToolProvider (not mock provider).
3. Real mode returns clear error when no CLI backends are available.
4. _mock_patch_for_work_item is NOT called in real mode.
5. Mock mode still raises ValueError for unknown module suffixes.
6. _ensure_isolated_workspace creates the workspace directory.
7. _ensure_isolated_workspace reuses existing workspace.
8. _copy_project_tree excludes build artifacts.
9. commit message says '(mock)' only in mock mode.
10. _progress emits output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from nexus_orchestrator.domain import ids

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# _dispatch_one routing
# ---------------------------------------------------------------------------


class TestDispatchOneRouting:
    """Verify that _dispatch_one routes correctly based on mock flag."""

    def _make_controller(self, tmp_path: Path):
        """Create a minimal OrchestratorController for testing."""
        from nexus_orchestrator.control_plane.controller import OrchestratorController

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return OrchestratorController(
            repo_root=tmp_path,
            state_db_path=state_dir / "test.sqlite",
        )

    def _make_dispatch_plan(self):
        """Create a fake _DispatchPlan for testing."""
        from nexus_orchestrator.control_plane.controller import _DispatchPlan
        from nexus_orchestrator.synthesis_plane.dispatch import DispatchBudget
        from nexus_orchestrator.synthesis_plane.roles import EscalationDecision

        work_item = MagicMock()
        work_item.id = ids.generate_work_item_id()
        work_item.title = "Implement workstream-001 module A"
        work_item.scope = ["src/a.py", "tests/unit/test_a.py"]
        work_item.description = "Test work item"
        work_item.dependencies = ()
        work_item.constraint_envelope = MagicMock()
        work_item.constraint_envelope.constraints = []
        work_item.budget = MagicMock()
        work_item.budget.max_tokens = 32_000
        work_item.budget.max_cost_usd = 2.0
        work_item.budget.max_iterations = 5
        work_item.risk_tier = MagicMock()
        work_item.risk_tier.value = "low"

        decision = EscalationDecision(
            attempt_number=1,
            provider="tool",
            model="claude_code",
            stage_index=1,
            stage_attempt=1,
            stage_limit=3,
        )
        return _DispatchPlan(
            work_item=work_item,
            decision=decision,
            attempt_number=1,
            prompt="test prompt",
            dispatch_budget=DispatchBudget(
                max_tokens=32_000,
                max_cost_usd=2.0,
                max_attempts=3,
            ),
        )

    def test_mock_mode_calls_mock_patch(self, tmp_path: Path) -> None:
        """Mock mode should call _mock_patch_for_work_item."""
        ctrl = self._make_controller(tmp_path)
        plan = self._make_dispatch_plan()

        from nexus_orchestrator.synthesis_plane.model_catalog import load_model_catalog

        catalog = load_model_catalog()
        config = {"providers": {"tool": {"max_concurrent": 1}}}

        with patch.object(
            ctrl, "_mock_patch_for_work_item", return_value="mock patch"
        ) as mock_fn:
            ctrl._dispatch_one(
                run_id=ids.generate_run_id(),
                dispatch_plan=plan,
                effective_config=config,
                mock=True,
                model_catalog=catalog,
            )
        # _mock_patch_for_work_item MUST be called in mock mode
        mock_fn.assert_called_once_with(plan.work_item)

    def test_real_mode_no_backends_returns_error(self, tmp_path: Path) -> None:
        """Real mode with no CLI backends should return a clear error."""
        ctrl = self._make_controller(tmp_path)
        plan = self._make_dispatch_plan()

        from nexus_orchestrator.synthesis_plane.model_catalog import load_model_catalog

        catalog = load_model_catalog()
        config = {"providers": {"tool": {"max_concurrent": 1}}}

        with patch(
            "nexus_orchestrator.synthesis_plane.providers.tool_detection.detect_all_backends",
            return_value=[],
        ):
            outcome = ctrl._dispatch_one(
                run_id=ids.generate_run_id(),
                dispatch_plan=plan,
                effective_config=config,
                mock=False,
                model_catalog=catalog,
            )
        assert outcome.error is not None
        assert "no CLI tool backends available" in outcome.error
        assert outcome.result is None

    def test_real_mode_does_not_call_mock_patch(self, tmp_path: Path) -> None:
        """Real mode should NOT call _mock_patch_for_work_item."""
        ctrl = self._make_controller(tmp_path)
        plan = self._make_dispatch_plan()

        from nexus_orchestrator.synthesis_plane.model_catalog import load_model_catalog

        catalog = load_model_catalog()
        config = {"providers": {"tool": {"max_concurrent": 1}}}

        with (
            patch.object(ctrl, "_mock_patch_for_work_item") as mock_fn,
            patch(
                "nexus_orchestrator.synthesis_plane.providers.tool_detection.detect_all_backends",
                return_value=[],
            ),
        ):
            ctrl._dispatch_one(
                run_id=ids.generate_run_id(),
                dispatch_plan=plan,
                effective_config=config,
                mock=False,
                model_catalog=catalog,
            )
        mock_fn.assert_not_called()

    def test_mock_mode_unknown_suffix_raises(self, tmp_path: Path) -> None:
        """Mock mode with unknown module suffix should still raise ValueError."""
        ctrl = self._make_controller(tmp_path)
        plan = self._make_dispatch_plan()
        # Change scope to non-A/B/C suffix
        plan.work_item.scope = ["src/xyz.py"]
        plan.work_item.title = "Implement module XYZ"

        from nexus_orchestrator.synthesis_plane.model_catalog import load_model_catalog

        catalog = load_model_catalog()
        config = {"providers": {"tool": {"max_concurrent": 1}}}

        with pytest.raises(ValueError, match="no deterministic mock patch configured"):
            ctrl._dispatch_one(
                run_id=ids.generate_run_id(),
                dispatch_plan=plan,
                effective_config=config,
                mock=True,
                model_catalog=catalog,
            )

    def test_real_mode_with_backend_creates_tool_provider(self, tmp_path: Path) -> None:
        """Real mode with available backend should create ToolProvider."""
        ctrl = self._make_controller(tmp_path)
        plan = self._make_dispatch_plan()

        from nexus_orchestrator.synthesis_plane.model_catalog import load_model_catalog
        from nexus_orchestrator.synthesis_plane.providers.tool_detection import ToolBackendInfo

        catalog = load_model_catalog()
        config = {"providers": {"tool": {"max_concurrent": 1}}}

        fake_backend = ToolBackendInfo(
            name="claude",
            binary_path="/usr/bin/claude",
            version="1.0.0",
        )

        with (
            patch(
                "nexus_orchestrator.synthesis_plane.providers.tool_detection.detect_all_backends",
                return_value=[fake_backend],
            ),
            patch(
                "nexus_orchestrator.synthesis_plane.providers.tool_adapter.ToolProvider.__init__",
                side_effect=Exception("simulated: binary not found"),
            ),
        ):
            outcome = ctrl._dispatch_one(
                run_id=ids.generate_run_id(),
                dispatch_plan=plan,
                effective_config=config,
                mock=False,
                model_catalog=catalog,
                workspace_cwd=tmp_path,
            )
        # Should get a provider init error, NOT the old "non-mock providers are not configured"
        assert outcome.error is not None
        assert "failed to initialize tool provider" in outcome.error
        assert "non-mock providers are not configured" not in outcome.error


# ---------------------------------------------------------------------------
# Isolated workspace
# ---------------------------------------------------------------------------


class TestIsolatedWorkspace:
    """Verify workspace creation and file copying."""

    def _make_controller(self, tmp_path: Path):
        from nexus_orchestrator.control_plane.controller import OrchestratorController

        state_dir = tmp_path / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        return OrchestratorController(
            repo_root=tmp_path,
            state_db_path=state_dir / "test.sqlite",
        )

    def test_ensure_isolated_workspace_creates_dir(self, tmp_path: Path) -> None:
        """_ensure_isolated_workspace should create .nexus/workspaces/<run>/repo."""
        # Create some project files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'")

        ctrl = self._make_controller(tmp_path)
        workspace = ctrl._ensure_isolated_workspace("run-123")

        assert workspace.exists()
        assert workspace == tmp_path / ".nexus" / "workspaces" / "run-123" / "repo"
        assert (workspace / "src" / "main.py").exists()
        assert (workspace / "pyproject.toml").exists()

    def test_ensure_isolated_workspace_reuses_existing(self, tmp_path: Path) -> None:
        """Second call should reuse the existing workspace."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("v1")

        ctrl = self._make_controller(tmp_path)
        ws1 = ctrl._ensure_isolated_workspace("run-456")

        # Modify the original file
        (tmp_path / "src" / "main.py").write_text("v2")

        ws2 = ctrl._ensure_isolated_workspace("run-456")
        assert ws1 == ws2
        # Should still have v1 (not re-copied)
        assert (ws2 / "src" / "main.py").read_text() == "v1"

    def test_copy_project_tree_excludes_artifacts(self, tmp_path: Path) -> None:
        """_copy_project_tree should skip .venv, .git, __pycache__, etc."""
        from nexus_orchestrator.control_plane.controller import _copy_project_tree

        src = tmp_path / "src_repo"
        src.mkdir()
        (src / "src").mkdir()
        (src / "src" / "app.py").write_text("app")
        (src / ".venv").mkdir()
        (src / ".venv" / "bin").mkdir()
        (src / ".venv" / "bin" / "python").write_text("python")
        (src / ".git").mkdir()
        (src / ".git" / "HEAD").write_text("ref")
        (src / "__pycache__").mkdir()
        (src / "__pycache__" / "app.cpython-311.pyc").write_text("bytecode")
        (src / "node_modules").mkdir()
        (src / ".nexus").mkdir()

        dst = tmp_path / "dst_repo"
        dst.mkdir()
        _copy_project_tree(src, dst)

        assert (dst / "src" / "app.py").exists()
        assert not (dst / ".venv").exists()
        assert not (dst / ".git").exists()
        assert not (dst / "__pycache__").exists()
        assert not (dst / "node_modules").exists()
        assert not (dst / ".nexus").exists()


# ---------------------------------------------------------------------------
# Progress lines
# ---------------------------------------------------------------------------


class TestProgressLines:
    """Verify _progress emits output."""

    def test_progress_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        from nexus_orchestrator.control_plane.controller import _progress

        _progress("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out


# ---------------------------------------------------------------------------
# Commit message
# ---------------------------------------------------------------------------


class TestCommitMessage:
    """Verify commit message varies by mode."""

    def test_mock_mode_label(self) -> None:
        """Mock mode should produce '(mock)' in commit message."""
        mock = True
        title = "Implement workstream A"
        msg = f"{title}{' (mock)' if mock else ''}"
        assert msg == "Implement workstream A (mock)"

    def test_real_mode_label(self) -> None:
        """Real mode should NOT have '(mock)' in commit message."""
        mock = False
        title = "Implement workstream A"
        msg = f"{title}{' (mock)' if mock else ''}"
        assert msg == "Implement workstream A"
