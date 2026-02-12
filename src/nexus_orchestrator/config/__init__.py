"""
nexus-orchestrator config package public API.

File: src/nexus_orchestrator/config/__init__.py
Last updated: 2026-02-12

Purpose
- Export config loading/validation entrypoints and public error types.

What should be included in this file
- Public schema constants and validation/report types.
- Loader APIs for effective runtime config and redacted dumps.
- No provider adapters or external runtime side effects.

Functional requirements
- Support loading from ``orchestrator.toml`` + ``NEXUS_`` env overrides.
- Fail fast with clear structured validation/load errors.

Non-functional requirements
- Keep import-time surface small and deterministic.
"""

from nexus_orchestrator.config.loader import (
    DEFAULT_CONFIG_FILE,
    ENV_PREFIX,
    ConfigLoadError,
    dump_effective_config,
    effective_config,
    load_config,
    load_config_file,
    normalize_paths,
)
from nexus_orchestrator.config.schema import (
    BUILTIN_PROFILE_NAMES,
    DEFAULT_CONFIG,
    PATH_FIELDS,
    ConfigSchemaVersion,
    ConfigValidationError,
    ConfigValidationIssue,
    ConfigValidationResult,
    OrchestratorConfig,
    ProfileOverlay,
    apply_profile_overlay,
    assert_valid_config,
    default_config,
    dump_redacted,
    merge_config,
    migration_guidance,
    redact_config,
    validate_config,
)

__all__ = [
    "BUILTIN_PROFILE_NAMES",
    "ConfigLoadError",
    "ConfigSchemaVersion",
    "ConfigValidationError",
    "ConfigValidationIssue",
    "ConfigValidationResult",
    "DEFAULT_CONFIG",
    "DEFAULT_CONFIG_FILE",
    "ENV_PREFIX",
    "OrchestratorConfig",
    "PATH_FIELDS",
    "ProfileOverlay",
    "apply_profile_overlay",
    "assert_valid_config",
    "default_config",
    "dump_effective_config",
    "dump_redacted",
    "effective_config",
    "load_config",
    "load_config_file",
    "merge_config",
    "migration_guidance",
    "normalize_paths",
    "redact_config",
    "validate_config",
]
