"""Adversarial verification components and policy helpers."""

from __future__ import annotations

from nexus_orchestrator.domain.models import RiskTier
from nexus_orchestrator.verification_plane.adversarial.test_generator import (
    AdversarialTestGenerator,
    GeneratedAdversarialTarget,
)
from nexus_orchestrator.verification_plane.pipeline import DEFAULT_RISK_TIER_POLICY


def adversarial_required_for_risk_tier(risk_tier: RiskTier | str) -> bool:
    """Return whether adversarial stage is required for the provided risk tier."""

    tier = risk_tier if isinstance(risk_tier, RiskTier) else RiskTier(risk_tier.strip().lower())
    return DEFAULT_RISK_TIER_POLICY[tier].adversarial_required


def risk_tiers_requiring_adversarial() -> tuple[RiskTier, ...]:
    """Return risk tiers that require adversarial verification, in stable order."""

    return tuple(tier for tier in RiskTier if DEFAULT_RISK_TIER_POLICY[tier].adversarial_required)


__all__ = [
    "AdversarialTestGenerator",
    "GeneratedAdversarialTarget",
    "adversarial_required_for_risk_tier",
    "risk_tiers_requiring_adversarial",
]
