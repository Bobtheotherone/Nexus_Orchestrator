"""Stable constants shared across orchestrator planes."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final

# Git branch names.
DEFAULT_MAIN_BRANCH: Final[str] = "main"
DEFAULT_INTEGRATION_BRANCH: Final[str] = "integration"
DEFAULT_CONTRACT_BRANCH: Final[str] = "contract"
DEFAULT_WORK_BRANCH_PREFIX: Final[str] = "work"
DEFAULT_VERIFY_BRANCH_PREFIX: Final[str] = "verify"

# Schema versions for persisted contracts.
CONFIG_SCHEMA_VERSION: Final[int] = 1
CONSTRAINT_REGISTRY_SCHEMA_VERSION: Final[int] = 1
EVIDENCE_LEDGER_SCHEMA_VERSION: Final[int] = 1
STATE_DB_SCHEMA_VERSION: Final[int] = 1
SPEC_MAP_SCHEMA_VERSION: Final[int] = 1
TASK_GRAPH_SCHEMA_VERSION: Final[int] = 1

# Default runtime paths (relative to workspace root unless overridden by config).
STATE_DIR: Final[PurePosixPath] = PurePosixPath("state")
WORKSPACES_DIR: Final[PurePosixPath] = PurePosixPath("workspaces")
EVIDENCE_DIR: Final[PurePosixPath] = PurePosixPath("evidence")
ARTIFACTS_DIR: Final[PurePosixPath] = PurePosixPath("artifacts")
CACHE_DIR: Final[PurePosixPath] = PurePosixPath(".nexus/cache")

# Risk tiers and weights for deterministic sorting/escalation.
RISK_TIERS: Final[tuple[str, ...]] = ("low", "medium", "high", "critical")
RISK_TIER_WEIGHT: Final[dict[str, int]] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

__all__ = [
    "ARTIFACTS_DIR",
    "CACHE_DIR",
    "CONFIG_SCHEMA_VERSION",
    "CONSTRAINT_REGISTRY_SCHEMA_VERSION",
    "DEFAULT_CONTRACT_BRANCH",
    "DEFAULT_INTEGRATION_BRANCH",
    "DEFAULT_MAIN_BRANCH",
    "DEFAULT_VERIFY_BRANCH_PREFIX",
    "DEFAULT_WORK_BRANCH_PREFIX",
    "EVIDENCE_DIR",
    "EVIDENCE_LEDGER_SCHEMA_VERSION",
    "RISK_TIERS",
    "RISK_TIER_WEIGHT",
    "SPEC_MAP_SCHEMA_VERSION",
    "STATE_DB_SCHEMA_VERSION",
    "STATE_DIR",
    "TASK_GRAPH_SCHEMA_VERSION",
    "WORKSPACES_DIR",
]
