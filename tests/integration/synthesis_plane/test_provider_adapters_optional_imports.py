"""Integration tests for optional provider SDK imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_python(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_path = str(_repo_root() / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}:{existing}"

    return subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


@pytest.mark.integration
def test_provider_modules_import_when_vendor_sdks_are_unavailable() -> None:
    script = """
import builtins
import importlib

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "openai" or name.startswith("openai."):
        raise ImportError(f"blocked import: {name}")
    if name == "anthropic" or name.startswith("anthropic."):
        raise ImportError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

importlib.import_module("nexus_orchestrator.synthesis_plane.providers.openai_adapter")
importlib.import_module("nexus_orchestrator.synthesis_plane.providers.anthropic_adapter")
importlib.import_module("nexus_orchestrator.synthesis_plane.providers")
importlib.import_module("nexus_orchestrator.synthesis_plane")

print("imports-ok")
"""
    result = _run_python(script)

    assert result.returncode == 0, result.stderr
    assert "imports-ok" in result.stdout


@pytest.mark.integration
def test_provider_unavailable_errors_are_deferred_until_send_time() -> None:
    script = """
import asyncio
import builtins

original_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "openai" or name.startswith("openai."):
        raise ImportError(f"blocked import: {name}")
    if name == "anthropic" or name.startswith("anthropic."):
        raise ImportError(f"blocked import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

from nexus_orchestrator.synthesis_plane.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
    ProviderRequest,
    ProviderUnavailableError,
)

openai_adapter = OpenAIAdapter(model="gpt-4.1-mini")
anthropic_adapter = AnthropicAdapter(model="claude-3-7-sonnet")

async def main() -> None:
    try:
        await openai_adapter.send(
            ProviderRequest(model="gpt-4.1-mini", role="implementer", prompt="ping")
        )
        raise AssertionError("expected openai ProviderUnavailableError")
    except ProviderUnavailableError as exc:
        assert "provider=openai code=unavailable" in str(exc)

    try:
        await anthropic_adapter.send(
            ProviderRequest(model="claude-3-7-sonnet", role="architect", prompt="ping")
        )
        raise AssertionError("expected anthropic ProviderUnavailableError")
    except ProviderUnavailableError as exc:
        assert "provider=anthropic code=unavailable" in str(exc)

asyncio.run(main())
print("runtime-ok")
"""
    result = _run_python(script)

    assert result.returncode == 0, result.stderr
    assert "runtime-ok" in result.stdout
