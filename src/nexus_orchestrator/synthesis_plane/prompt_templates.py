"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/synthesis_plane/prompt_templates.py
Last updated: 2026-02-11

Purpose
- Loads and renders role prompt templates from docs/prompts/templates/ with strict placeholders.

What should be included in this file
- Template rendering rules and allowed variables.
- Prompt versioning and hashing (for evidence reproducibility).

Functional requirements
- Must render prompts deterministically for same inputs.

Non-functional requirements
- Must prevent template injection (escape untrusted fields).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, StrictUndefined, meta

from nexus_orchestrator.security.prompt_hygiene import (
    DEFAULT_POLICY_MODE,
    HygienePolicyMode,
    TrustLevel,
    sanitize_context,
)
from nexus_orchestrator.utils.hashing import sha256_text

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping


_ROLE_NAME_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.md)?$")
_LAST_UPDATED_RE = re.compile(r"(?im)^\s*Last updated:\s*(.+?)\s*$")


class PromptTemplateError(RuntimeError):
    """Base error for prompt template loading and rendering."""


class PromptTemplateNotFoundError(PromptTemplateError, FileNotFoundError):
    """Raised when a role template file does not exist."""


class PromptTemplateVariableError(PromptTemplateError, ValueError):
    """Raised for missing/extra variables or whitelist violations."""


@dataclass(frozen=True, slots=True)
class VariableOrigin:
    """Optional origin metadata used for trust classification per variable."""

    path: str | None = None
    doc_type: str | None = None
    trust: TrustLevel | str | None = None


@dataclass(frozen=True, slots=True)
class PromptTemplateMetadata:
    """Template identity + reproducibility metadata for rendered prompts."""

    role: str
    template_name: str
    template_version: str
    template_path: str
    template_path_relative: str
    template_hash: str
    declared_variables: tuple[str, ...]
    allowed_variables: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """Rendered prompt and deterministic hashes for evidence logging."""

    prompt: str
    prompt_hash: str
    template_metadata: PromptTemplateMetadata
    hygiene_mode: HygienePolicyMode
    dropped_variables: tuple[str, ...]
    variable_findings: tuple[tuple[str, int], ...]


class PromptTemplateEngine:
    """Deterministic role prompt template loader + renderer."""

    def __init__(
        self,
        *,
        template_root: Path | str | None = None,
        hygiene_mode: HygienePolicyMode | str = DEFAULT_POLICY_MODE,
    ) -> None:
        root = Path(template_root) if template_root is not None else _default_template_root()
        resolved_root = root.resolve()
        if not resolved_root.exists():
            raise PromptTemplateNotFoundError(f"template root does not exist: {resolved_root}")
        if not resolved_root.is_dir():
            raise NotADirectoryError(f"template root is not a directory: {resolved_root}")

        self._template_root = resolved_root
        self._hygiene_mode = _coerce_hygiene_mode(hygiene_mode)
        self._environment = Environment(
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
            newline_sequence="\n",
            keep_trailing_newline=True,
        )

    @property
    def template_root(self) -> Path:
        """Resolved template root path."""

        return self._template_root

    @property
    def hygiene_mode(self) -> HygienePolicyMode:
        """Hygiene policy mode used during rendering."""

        return self._hygiene_mode

    def render(
        self,
        role: str,
        *,
        variables: Mapping[str, object],
        allowed_variables: Collection[str],
        variable_origins: Mapping[str, VariableOrigin] | None = None,
        trusted_variables: Collection[str] = (),
    ) -> RenderedPrompt:
        """Render one role template with strict variable/whitelist checks."""

        template_name = _normalize_role_to_template_name(role)
        template_path = self._resolve_template_path(template_name)
        template_source = _normalize_newlines(template_path.read_text(encoding="utf-8"))

        template = self._environment.from_string(template_source)
        declared_variables = tuple(
            sorted(meta.find_undeclared_variables(self._environment.parse(template_source)))
        )

        allowed = _normalize_variable_names(allowed_variables, field_name="allowed_variables")
        allowed_set = set(allowed)

        unexpected_in_template = sorted(set(declared_variables) - allowed_set)
        if unexpected_in_template:
            raise PromptTemplateVariableError(
                "template uses variables not allowed by whitelist: "
                + ", ".join(unexpected_in_template)
            )

        variable_payload = _normalize_variable_mapping(variables)
        unexpected_inputs = sorted(set(variable_payload) - allowed_set)
        if unexpected_inputs:
            raise PromptTemplateVariableError(
                "unexpected variables were provided: " + ", ".join(unexpected_inputs)
            )

        missing_required = sorted(set(declared_variables) - set(variable_payload))
        if missing_required:
            raise PromptTemplateVariableError(
                "missing required template variables: " + ", ".join(missing_required)
            )

        trusted_set = set(
            _normalize_variable_names(trusted_variables, field_name="trusted_variables")
        )
        unknown_trusted = sorted(trusted_set - allowed_set)
        if unknown_trusted:
            raise PromptTemplateVariableError(
                "trusted_variables contains unknown variable(s): " + ", ".join(unknown_trusted)
            )

        origins = dict(variable_origins or {})
        unknown_origins = sorted(set(origins) - allowed_set)
        if unknown_origins:
            raise PromptTemplateVariableError(
                "variable_origins contains unknown variable(s): " + ", ".join(unknown_origins)
            )

        rendered_values: dict[str, str] = {}
        dropped_variables: list[str] = []
        findings_by_variable: list[tuple[str, int]] = []

        for key in sorted(variable_payload):
            origin = origins.get(key)
            trust_override: TrustLevel | str | None = None
            path: str | None = None
            doc_type: str | None = None

            if origin is not None:
                path = origin.path
                doc_type = origin.doc_type
                trust_override = origin.trust

            if key in trusted_set:
                trust_override = TrustLevel.TRUSTED

            sanitized = sanitize_context(
                _normalize_newlines(_serialize_variable_value(variable_payload[key])),
                path=path,
                doc_type=doc_type,
                trust=trust_override,
                mode=self._hygiene_mode,
            )
            rendered_values[key] = sanitized.sanitized_text
            findings_by_variable.append((key, len(sanitized.findings)))
            if sanitized.dropped:
                dropped_variables.append(key)

        rendered_prompt = _normalize_newlines(template.render(**rendered_values))
        template_hash = sha256_text(template_source, encoding="utf-8")
        prompt_hash = sha256_text(rendered_prompt, encoding="utf-8")
        template_version = _extract_template_version(template_source)

        metadata = PromptTemplateMetadata(
            role=role.strip(),
            template_name=template_name,
            template_version=template_version,
            template_path=template_path.as_posix(),
            template_path_relative=template_path.relative_to(self._template_root).as_posix(),
            template_hash=template_hash,
            declared_variables=declared_variables,
            allowed_variables=allowed,
        )
        return RenderedPrompt(
            prompt=rendered_prompt,
            prompt_hash=prompt_hash,
            template_metadata=metadata,
            hygiene_mode=self._hygiene_mode,
            dropped_variables=tuple(sorted(dropped_variables)),
            variable_findings=tuple(sorted(findings_by_variable, key=lambda item: item[0])),
        )

    def _resolve_template_path(self, template_name: str) -> Path:
        candidate = (self._template_root / template_name).resolve()
        try:
            candidate.relative_to(self._template_root)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise PromptTemplateError(
                f"template path escapes template root: {template_name!r}"
            ) from exc

        if not candidate.exists() or not candidate.is_file():
            raise PromptTemplateNotFoundError(
                f"template not found for role: {template_name!r} under {self._template_root}"
            )
        return candidate


def render_prompt_template(
    role: str,
    *,
    variables: Mapping[str, object],
    allowed_variables: Collection[str],
    template_root: Path | str | None = None,
    variable_origins: Mapping[str, VariableOrigin] | None = None,
    trusted_variables: Collection[str] = (),
    hygiene_mode: HygienePolicyMode | str = DEFAULT_POLICY_MODE,
) -> RenderedPrompt:
    """Convenience one-shot renderer."""

    engine = PromptTemplateEngine(template_root=template_root, hygiene_mode=hygiene_mode)
    return engine.render(
        role,
        variables=variables,
        allowed_variables=allowed_variables,
        variable_origins=variable_origins,
        trusted_variables=trusted_variables,
    )


def _default_template_root() -> Path:
    return Path(__file__).resolve().parents[3] / "docs" / "prompts" / "templates"


def _normalize_role_to_template_name(role: str) -> str:
    if not isinstance(role, str):
        raise TypeError("role must be a string")

    cleaned = role.strip()
    if not cleaned:
        raise ValueError("role must not be empty")
    if not _ROLE_NAME_RE.fullmatch(cleaned):
        raise ValueError(f"invalid role name: {role!r}")

    stem = cleaned[:-3] if cleaned.lower().endswith(".md") else cleaned
    return f"{stem.upper()}.md"


def _normalize_variable_mapping(variables: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in variables.items():
        if not isinstance(key, str):
            raise TypeError("variable names must be strings")
        cleaned = key.strip()
        if not cleaned:
            raise ValueError("variable names must not be empty")
        normalized[cleaned] = value
    return normalized


def _normalize_variable_names(
    names: Collection[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized: set[str] = set()
    for item in names:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} entries must be strings")
        cleaned = item.strip()
        if not cleaned:
            raise ValueError(f"{field_name} entries must not be empty")
        normalized.add(cleaned)
    return tuple(sorted(normalized))


def _serialize_variable_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return str(value)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _extract_template_version(template_source: str) -> str:
    match = _LAST_UPDATED_RE.search(template_source)
    if match is None:
        return "unversioned"
    return match.group(1).strip()


def _coerce_hygiene_mode(mode: HygienePolicyMode | str) -> HygienePolicyMode:
    if isinstance(mode, HygienePolicyMode):
        return mode
    if not isinstance(mode, str):
        raise TypeError("hygiene_mode must be a HygienePolicyMode or string")

    normalized = mode.strip().lower().replace("_", "-")
    for candidate in HygienePolicyMode:
        if normalized == candidate.value:
            return candidate

    valid = ", ".join(item.value for item in HygienePolicyMode)
    raise ValueError(f"unsupported hygiene mode: {mode!r}; expected one of: {valid}")


__all__ = [
    "PromptTemplateEngine",
    "PromptTemplateError",
    "PromptTemplateMetadata",
    "PromptTemplateNotFoundError",
    "PromptTemplateVariableError",
    "RenderedPrompt",
    "VariableOrigin",
    "render_prompt_template",
]
