"""Unit tests for spark_triage — LLM-based 3-model routing with fallback.

File: tests/unit/test_spark_triage.py

Covers:
- Successful parse of ROUTE: <MODEL> response
- Timeout fallback to deterministic classifier
- Invalid response fallback
- Binary not found fallback
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from nexus_orchestrator.domain import ids
from nexus_orchestrator.domain.models import (
    Constraint,
    ConstraintEnvelope,
    ConstraintSeverity,
    ConstraintSource,
    RiskTier,
    WorkItem,
)
from nexus_orchestrator.synthesis_plane.spark_triage import (
    TriageResult,
    triage_with_spark,
)
from nexus_orchestrator.synthesis_plane.work_item_classifier import ModelAffinity

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _randbytes(seed: int) -> callable:
    byte_value = (seed % 251) + 1

    def _provider(size: int) -> bytes:
        return bytes([byte_value]) * size

    return _provider


def _make_work_item(
    seed: int = 1,
    *,
    title: str = "Implement feature X",
    description: str = "Write the implementation",
    scope: tuple[str, ...] = ("src/feature.py",),
    risk_tier: RiskTier = RiskTier.MEDIUM,
) -> WorkItem:
    work_item_id = ids.generate_work_item_id(
        timestamp_ms=1_700_000_000_000 + seed,
        randbytes=_randbytes(seed),
    )
    constraint = Constraint(
        id=f"CON-TST-{(seed % 9000) + 1:04d}",
        severity=ConstraintSeverity.MUST,
        category="test",
        description="Test constraint",
        checker_binding="test",
        parameters={},
        requirement_links=("REQ-0001",),
        source=ConstraintSource.MANUAL,
        created_at=_NOW,
    )
    envelope = ConstraintEnvelope(
        work_item_id=work_item_id,
        constraints=(constraint,),
        inherited_constraint_ids=(),
        compiled_at=_NOW,
    )
    return WorkItem(
        id=work_item_id,
        title=title,
        description=description,
        scope=scope,
        constraint_envelope=envelope,
        dependencies=(),
        risk_tier=risk_tier,
        requirement_links=("REQ-0001",),
        constraint_ids=(constraint.id,),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _mock_process(stdout: str, returncode: int = 0) -> AsyncMock:
    """Create a mock process with given stdout and returncode."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout.encode(), b""))
    return proc


class TestTriageSuccessfulParse:
    """Test that valid ROUTE: <MODEL> responses are parsed correctly."""

    @pytest.mark.asyncio
    async def test_route_gpt53(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("ROUTE: GPT53")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.chosen_model == ModelAffinity.GPT53
        assert result.used_llm is True
        assert "GPT53" in result.reasoning

    @pytest.mark.asyncio
    async def test_route_spark(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("ROUTE: SPARK")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.chosen_model == ModelAffinity.SPARK
        assert result.used_llm is True

    @pytest.mark.asyncio
    async def test_route_opus(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("ROUTE: OPUS")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.chosen_model == ModelAffinity.OPUS
        assert result.used_llm is True

    @pytest.mark.asyncio
    async def test_route_case_insensitive(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("route: opus")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.chosen_model == ModelAffinity.OPUS
        assert result.used_llm is True

    @pytest.mark.asyncio
    async def test_route_with_surrounding_text(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("I think this task should be\nROUTE: GPT53\nbecause it needs reasoning")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.chosen_model == ModelAffinity.GPT53
        assert result.used_llm is True


class TestTriageFallback:
    """Test fallback to deterministic classifier on various failures."""

    @pytest.mark.asyncio
    async def test_binary_not_found(self) -> None:
        wi = _make_work_item()

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.shutil.which",
                    return_value=None):
            result = await triage_with_spark(wi)

        assert result.used_llm is False
        assert "codex CLI not found" in result.reasoning

    @pytest.mark.asyncio
    async def test_timeout_falls_back(self) -> None:
        wi = _make_work_item()

        async def slow_communicate():
            await asyncio.sleep(10)
            return (b"ROUTE: OPUS", b"")

        proc = AsyncMock()
        proc.communicate = slow_communicate

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(
                wi, codex_binary_path="/usr/bin/codex", timeout_seconds=0.01
            )

        assert result.used_llm is False
        assert "timed out" in result.reasoning

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_falls_back(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("", returncode=1)

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.used_llm is False
        assert "exited with code 1" in result.reasoning

    @pytest.mark.asyncio
    async def test_empty_response_falls_back(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.used_llm is False
        assert "empty" in result.reasoning

    @pytest.mark.asyncio
    async def test_unparseable_response_falls_back(self) -> None:
        wi = _make_work_item()
        proc = _mock_process("I don't know which model to use")

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    return_value=proc):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.used_llm is False
        assert "unparseable" in result.reasoning

    @pytest.mark.asyncio
    async def test_os_error_falls_back(self) -> None:
        wi = _make_work_item()

        with patch("nexus_orchestrator.synthesis_plane.spark_triage.asyncio.create_subprocess_exec",
                    side_effect=OSError("Permission denied")):
            result = await triage_with_spark(wi, codex_binary_path="/usr/bin/codex")

        assert result.used_llm is False
        assert "subprocess error" in result.reasoning


class TestTriageResult:
    """Test TriageResult dataclass."""

    def test_triage_result_fields(self) -> None:
        result = TriageResult(
            chosen_model=ModelAffinity.OPUS,
            reasoning="test reason",
            used_llm=True,
        )
        assert result.chosen_model == ModelAffinity.OPUS
        assert result.reasoning == "test reason"
        assert result.used_llm is True

    def test_triage_result_frozen(self) -> None:
        result = TriageResult(
            chosen_model=ModelAffinity.SPARK,
            reasoning="frozen test",
            used_llm=False,
        )
        with pytest.raises(AttributeError):
            result.chosen_model = ModelAffinity.OPUS  # type: ignore[misc]
