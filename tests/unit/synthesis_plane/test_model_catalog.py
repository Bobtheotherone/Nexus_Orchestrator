"""Unit tests for synthesis-plane file-backed model catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.synthesis_plane.model_catalog import (
    ModelCatalog,
    load_model_catalog,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_bundled_catalog_exposes_metadata_and_default_profiles() -> None:
    catalog = load_model_catalog()

    assert catalog.version
    assert catalog.last_updated
    assert catalog.default_model_for_profile(provider="openai", capability_profile="code")
    assert catalog.default_model_for_profile(provider="anthropic", capability_profile="architect")


def test_catalog_resolves_aliases_and_estimates_cost() -> None:
    catalog = load_model_catalog()

    resolved = catalog.require("claude-sonnet-4-5-20250929", provider="anthropic")
    cost = catalog.estimate_cost(
        provider="anthropic",
        model=resolved.model,
        total_tokens=2_000,
        input_tokens=1_500,
        output_tokens=500,
    )

    assert resolved.model == "claude-sonnet-4-5"
    assert cost > 0


def test_catalog_rejects_default_profile_pointing_to_unknown_model(tmp_path: Path) -> None:
    bad_path = tmp_path / "model_catalog.json"
    bad_path.write_text(
        """
{
  "version": "test",
  "last_updated": "2026-02-13",
  "default_profiles": {
    "openai": {
      "code": "missing-model"
    }
  },
  "models": [
    {
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "max_context_tokens": 128000,
      "supports_tool_calling": true,
      "supports_structured_outputs": true,
      "reasoning_effort_allowed": ["low"],
      "cost": {
        "input_per_1k_usd": 0.001,
        "output_per_1k_usd": 0.002,
        "blended_per_1k_usd": 0.0015
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="default_profiles.openai.code references unknown model"):
        ModelCatalog.from_file(bad_path)
