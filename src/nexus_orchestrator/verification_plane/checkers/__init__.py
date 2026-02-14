"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/__init__.py
Last updated: 2026-02-11

Purpose
- Checker plugins: each checker produces machine-checkable evidence for a constraint category.

What should be included in this file
- Checker base protocol and registration mechanism.
- Built-in checkers directory layout (lint/typecheck/test/security/perf).

Functional requirements
- Must support external checker plugins (ext/checkers).

Non-functional requirements
- Checker execution must be sandboxed.
"""

from nexus_orchestrator.verification_plane.adversarial.test_generator import (
    AdversarialTestGenerator,
)
from nexus_orchestrator.verification_plane.checkers.base import (
    DEFAULT_CHECKER_REGISTRY,
    BaseChecker,
    CheckerContext,
    CheckerFactory,
    CheckerRegistration,
    CheckerRegistry,
    CheckerSource,
    CheckResult,
    CheckStatus,
    CommandExecutor,
    CommandResult,
    CommandSpec,
    JSONScalar,
    JSONValue,
    LocalSubprocessExecutor,
    MetadataRedactor,
    RedactionHooks,
    TextRedactor,
    Violation,
    normalize_artifact_paths,
    normalize_command_lines,
    normalize_violations,
    register_builtin_checker,
    register_external_checker,
    register_external_plugins,
)
from nexus_orchestrator.verification_plane.checkers.build_checker import BuildChecker
from nexus_orchestrator.verification_plane.checkers.documentation_checker import (
    DocumentationChecker,
)
from nexus_orchestrator.verification_plane.checkers.lint_checker import LintChecker
from nexus_orchestrator.verification_plane.checkers.performance_checker import PerformanceChecker
from nexus_orchestrator.verification_plane.checkers.reliability_checker import ReliabilityChecker
from nexus_orchestrator.verification_plane.checkers.schema_checker import SchemaChecker
from nexus_orchestrator.verification_plane.checkers.scope_checker import ScopeChecker
from nexus_orchestrator.verification_plane.checkers.security_checker import SecurityChecker
from nexus_orchestrator.verification_plane.checkers.test_checker import TestChecker
from nexus_orchestrator.verification_plane.checkers.typecheck_checker import TypecheckChecker

__all__ = [
    "BaseChecker",
    "AdversarialTestGenerator",
    "BuildChecker",
    "CheckResult",
    "CheckStatus",
    "CheckerContext",
    "CheckerFactory",
    "CheckerRegistration",
    "CheckerRegistry",
    "CheckerSource",
    "CommandExecutor",
    "CommandResult",
    "CommandSpec",
    "DEFAULT_CHECKER_REGISTRY",
    "DocumentationChecker",
    "JSONScalar",
    "JSONValue",
    "LintChecker",
    "LocalSubprocessExecutor",
    "MetadataRedactor",
    "PerformanceChecker",
    "RedactionHooks",
    "ReliabilityChecker",
    "SchemaChecker",
    "ScopeChecker",
    "SecurityChecker",
    "TestChecker",
    "TextRedactor",
    "TypecheckChecker",
    "Violation",
    "normalize_command_lines",
    "normalize_artifact_paths",
    "normalize_violations",
    "register_builtin_checker",
    "register_external_checker",
    "register_external_plugins",
]
