"""
nexus-orchestrator — central model capability catalog.

File: src/nexus_orchestrator/synthesis_plane/model_catalog.py
Last updated: 2026-02-13

Purpose
- Load and expose the repository-shipped model catalog used by adapters and routing.

What should be included in this file
- File-backed loader for model capability/cost metadata.
- Deterministic lookup helpers by provider/model and capability profile.

Functional requirements
- Keep adapter capability/cost/context decisions data-driven via catalog entries.

Non-functional requirements
- Deterministic, offline-safe, and auditable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType


def _validate_non_empty_str(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    parsed = value.strip()
    if not parsed:
        raise ValueError(f"{field_name} cannot be empty")
    return parsed


def _validate_non_negative_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric")
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _coerce_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    return value


def _coerce_float(value: object, field_name: str) -> float:
    return _validate_non_negative_float(value, field_name)


def _normalize_key(value: str) -> str:
    return _validate_non_empty_str(value, "value").lower()


def _as_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    out: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings")
        out[key] = item
    return out


def _as_sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise TypeError(f"{field_name} must be an array")


@dataclass(frozen=True, slots=True)
class ModelCostEstimate:
    """Per-model deterministic cost estimate metadata."""

    input_per_1k_usd: float
    output_per_1k_usd: float
    blended_per_1k_usd: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_per_1k_usd",
            _validate_non_negative_float(
                self.input_per_1k_usd, "ModelCostEstimate.input_per_1k_usd"
            ),
        )
        object.__setattr__(
            self,
            "output_per_1k_usd",
            _validate_non_negative_float(
                self.output_per_1k_usd,
                "ModelCostEstimate.output_per_1k_usd",
            ),
        )
        object.__setattr__(
            self,
            "blended_per_1k_usd",
            _validate_non_negative_float(
                self.blended_per_1k_usd,
                "ModelCostEstimate.blended_per_1k_usd",
            ),
        )

    def estimate(
        self,
        *,
        total_tokens: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> float:
        if total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        if input_tokens is not None and input_tokens < 0:
            raise ValueError("input_tokens must be >= 0")
        if output_tokens is not None and output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")

        if input_tokens is not None and output_tokens is not None:
            in_cost = (input_tokens / 1000.0) * self.input_per_1k_usd
            out_cost = (output_tokens / 1000.0) * self.output_per_1k_usd
            return in_cost + out_cost

        return (total_tokens / 1000.0) * self.blended_per_1k_usd


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Capability record for one provider model."""

    provider: str
    model: str
    aliases: tuple[str, ...] = ()
    max_context_tokens: int = 0
    supports_tool_calling: bool = False
    supports_structured_outputs: bool = False
    reasoning_effort_allowed: tuple[str, ...] = ()
    cost: ModelCostEstimate = field(default_factory=lambda: ModelCostEstimate(0.0, 0.0, 0.0))
    availability: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _validate_non_empty_str(self.provider, "ModelCapabilities.provider").lower(),
        )
        object.__setattr__(
            self, "model", _validate_non_empty_str(self.model, "ModelCapabilities.model")
        )
        normalized_aliases: list[str] = []
        for index, alias in enumerate(self.aliases):
            parsed = _validate_non_empty_str(alias, f"ModelCapabilities.aliases[{index}]")
            normalized_aliases.append(parsed)
        if len({_normalize_key(alias) for alias in normalized_aliases}) != len(normalized_aliases):
            raise ValueError("ModelCapabilities.aliases must not contain duplicates")
        object.__setattr__(self, "aliases", tuple(normalized_aliases))

        if isinstance(self.max_context_tokens, bool) or not isinstance(
            self.max_context_tokens, int
        ):
            raise TypeError("ModelCapabilities.max_context_tokens must be an integer")
        if self.max_context_tokens <= 0:
            raise ValueError("ModelCapabilities.max_context_tokens must be > 0")

        object.__setattr__(self, "supports_tool_calling", bool(self.supports_tool_calling))
        object.__setattr__(
            self, "supports_structured_outputs", bool(self.supports_structured_outputs)
        )

        normalized_effort: list[str] = []
        for index, entry in enumerate(self.reasoning_effort_allowed):
            parsed = _validate_non_empty_str(
                entry,
                f"ModelCapabilities.reasoning_effort_allowed[{index}]",
            ).lower()
            normalized_effort.append(parsed)
        if len(set(normalized_effort)) != len(normalized_effort):
            raise ValueError(
                "ModelCapabilities.reasoning_effort_allowed must not contain duplicates"
            )
        object.__setattr__(self, "reasoning_effort_allowed", tuple(normalized_effort))

        if not isinstance(self.cost, ModelCostEstimate):
            raise TypeError("ModelCapabilities.cost must be ModelCostEstimate")

        availability = self.availability
        if availability is not None:
            availability = _validate_non_empty_str(availability, "ModelCapabilities.availability")
        object.__setattr__(self, "availability", availability)

        notes = self.notes
        if notes is not None:
            notes = _validate_non_empty_str(notes, "ModelCapabilities.notes")
        object.__setattr__(self, "notes", notes)

    @property
    def names(self) -> tuple[str, ...]:
        """Canonical model id plus aliases."""

        return (self.model, *self.aliases)


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """File-backed central model capability catalog."""

    version: str
    last_updated: str
    models: tuple[ModelCapabilities, ...]
    default_profiles: Mapping[str, Mapping[str, str]]
    _by_provider_and_model: Mapping[tuple[str, str], ModelCapabilities] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_model: Mapping[str, tuple[ModelCapabilities, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "version", _validate_non_empty_str(self.version, "ModelCatalog.version")
        )
        object.__setattr__(
            self,
            "last_updated",
            _validate_non_empty_str(self.last_updated, "ModelCatalog.last_updated"),
        )

        if not self.models:
            raise ValueError("ModelCatalog.models cannot be empty")

        by_provider_and_model: dict[tuple[str, str], ModelCapabilities] = {}
        by_model: dict[str, list[ModelCapabilities]] = {}
        for entry in self.models:
            if not isinstance(entry, ModelCapabilities):
                raise TypeError("ModelCatalog.models entries must be ModelCapabilities")
            for name in entry.names:
                key = (entry.provider, _normalize_key(name))
                if key in by_provider_and_model:
                    raise ValueError(
                        f"duplicate model catalog key for provider={entry.provider!r}, model={name!r}"
                    )
                by_provider_and_model[key] = entry

                unscoped_key = _normalize_key(name)
                by_model.setdefault(unscoped_key, []).append(entry)

        normalized_defaults: dict[str, Mapping[str, str]] = {}
        for provider_raw, profile_map_raw in self.default_profiles.items():
            provider = _normalize_key(provider_raw)
            profile_map = _as_mapping(profile_map_raw, f"default_profiles.{provider}")

            normalized_profile_map: dict[str, str] = {}
            for profile_raw, model_raw in profile_map.items():
                profile = _normalize_key(profile_raw)
                model = _validate_non_empty_str(
                    str(model_raw), f"default_profiles.{provider}.{profile}"
                )
                lookup_key = (provider, _normalize_key(model))
                if lookup_key not in by_provider_and_model:
                    raise ValueError(
                        f"default_profiles.{provider}.{profile} references unknown model {model!r}"
                    )
                normalized_profile_map[profile] = by_provider_and_model[lookup_key].model
            normalized_defaults[provider] = MappingProxyType(normalized_profile_map)

        object.__setattr__(
            self,
            "default_profiles",
            MappingProxyType(normalized_defaults),
        )
        object.__setattr__(
            self,
            "_by_provider_and_model",
            MappingProxyType(by_provider_and_model),
        )
        object.__setattr__(
            self,
            "_by_model",
            MappingProxyType({key: tuple(value) for key, value in by_model.items()}),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ModelCatalog:
        if "version" not in payload:
            raise ValueError("version is required")
        if "last_updated" not in payload:
            raise ValueError("last_updated is required")
        version = _validate_non_empty_str(str(payload["version"]), "version")
        last_updated = _validate_non_empty_str(str(payload["last_updated"]), "last_updated")

        models_raw = _as_sequence(payload.get("models"), "models")
        parsed_models: list[ModelCapabilities] = []
        for index, item in enumerate(models_raw):
            item_map = _as_mapping(item, f"models[{index}]")
            cost_map = _as_mapping(item_map.get("cost"), f"models[{index}].cost")
            aliases_raw = item_map.get("aliases", ())
            aliases = tuple(
                _validate_non_empty_str(str(value), f"models[{index}].aliases")
                for value in _as_sequence(aliases_raw, f"models[{index}].aliases")
            )
            reasoning_raw = item_map.get("reasoning_effort_allowed", ())
            reasoning = tuple(
                _validate_non_empty_str(
                    str(value),
                    f"models[{index}].reasoning_effort_allowed",
                )
                for value in _as_sequence(
                    reasoning_raw,
                    f"models[{index}].reasoning_effort_allowed",
                )
            )
            parsed_models.append(
                ModelCapabilities(
                    provider=_validate_non_empty_str(
                        str(item_map.get("provider")),
                        f"models[{index}].provider",
                    ),
                    model=_validate_non_empty_str(
                        str(item_map.get("model")),
                        f"models[{index}].model",
                    ),
                    aliases=aliases,
                    max_context_tokens=_coerce_int(
                        item_map.get("max_context_tokens", 0),
                        f"models[{index}].max_context_tokens",
                    ),
                    supports_tool_calling=bool(item_map.get("supports_tool_calling", False)),
                    supports_structured_outputs=bool(
                        item_map.get("supports_structured_outputs", False)
                    ),
                    reasoning_effort_allowed=reasoning,
                    cost=ModelCostEstimate(
                        input_per_1k_usd=_coerce_float(
                            cost_map.get("input_per_1k_usd", 0.0),
                            f"models[{index}].cost.input_per_1k_usd",
                        ),
                        output_per_1k_usd=_coerce_float(
                            cost_map.get("output_per_1k_usd", 0.0),
                            f"models[{index}].cost.output_per_1k_usd",
                        ),
                        blended_per_1k_usd=_coerce_float(
                            cost_map.get("blended_per_1k_usd", 0.0),
                            f"models[{index}].cost.blended_per_1k_usd",
                        ),
                    ),
                    availability=(
                        _validate_non_empty_str(
                            str(item_map["availability"]),
                            f"models[{index}].availability",
                        )
                        if "availability" in item_map and item_map["availability"] is not None
                        else None
                    ),
                    notes=(
                        _validate_non_empty_str(
                            str(item_map["notes"]),
                            f"models[{index}].notes",
                        )
                        if "notes" in item_map and item_map["notes"] is not None
                        else None
                    ),
                )
            )

        defaults_raw = _as_mapping(payload.get("default_profiles"), "default_profiles")
        default_profiles: dict[str, Mapping[str, str]] = {}
        for provider, profiles in defaults_raw.items():
            profile_map = _as_mapping(profiles, f"default_profiles.{provider}")
            default_profiles[provider] = {
                profile_name: _validate_non_empty_str(
                    str(model_name), f"default_profiles.{provider}.{profile_name}"
                )
                for profile_name, model_name in profile_map.items()
            }

        return cls(
            version=version,
            last_updated=last_updated,
            models=tuple(parsed_models),
            default_profiles=default_profiles,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ModelCatalog:
        candidate = Path(path).expanduser().resolve()
        try:
            raw_text = candidate.read_text(encoding="utf-8")
            payload = json.loads(_strip_leading_slash_comment_header(raw_text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model catalog JSON in {candidate}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"unable to read model catalog file {candidate}: {exc}") from exc

        payload_map = _as_mapping(payload, "catalog")
        return cls.from_mapping(payload_map)

    def get(
        self,
        model: str,
        *,
        provider: str | None = None,
    ) -> ModelCapabilities | None:
        model_key = _normalize_key(model)
        if provider is not None:
            provider_key = _normalize_key(provider)
            return self._by_provider_and_model.get((provider_key, model_key))

        matches = self._by_model.get(model_key, ())
        if not matches:
            return None
        if len(matches) > 1:
            providers = sorted({item.provider for item in matches})
            raise ValueError(
                f"model {model!r} exists under multiple providers; specify provider from {providers}"
            )
        return matches[0]

    def require(
        self,
        model: str,
        *,
        provider: str | None = None,
    ) -> ModelCapabilities:
        found = self.get(model, provider=provider)
        if found is None:
            if provider is None:
                raise KeyError(f"unknown model {model!r}")
            raise KeyError(f"unknown model {model!r} for provider {provider!r}")
        return found

    def default_model_for_profile(self, *, provider: str, capability_profile: str) -> str:
        provider_key = _normalize_key(provider)
        profile_key = _normalize_key(capability_profile)

        profile_map = self.default_profiles.get(provider_key)
        if profile_map is None:
            raise KeyError(f"unknown provider {provider!r} in model catalog defaults")
        model = profile_map.get(profile_key)
        if model is None:
            raise KeyError(
                f"provider {provider!r} does not define capability profile {capability_profile!r}"
            )
        return model

    def resolve_model_for_profile(
        self,
        *,
        provider: str,
        capability_profile: str,
        configured_model: str | None,
    ) -> str:
        if configured_model is not None:
            return self.require(configured_model, provider=provider).model
        return self.default_model_for_profile(
            provider=provider,
            capability_profile=capability_profile,
        )

    def max_context_tokens(self, *, provider: str, model: str) -> int:
        return self.require(model, provider=provider).max_context_tokens

    def supports_tool_calling(self, *, provider: str, model: str) -> bool:
        return self.require(model, provider=provider).supports_tool_calling

    def supports_structured_outputs(self, *, provider: str, model: str) -> bool:
        return self.require(model, provider=provider).supports_structured_outputs

    def reasoning_effort_allowed(self, *, provider: str, model: str) -> tuple[str, ...]:
        return self.require(model, provider=provider).reasoning_effort_allowed

    def estimate_cost(
        self,
        *,
        provider: str,
        model: str,
        total_tokens: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> float:
        capabilities = self.require(model, provider=provider)
        return capabilities.cost.estimate(
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _bundled_catalog_path() -> Path:
    return Path(__file__).resolve().with_name("model_catalog.json")


def _strip_leading_slash_comment_header(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    first_non_empty = 0
    while first_non_empty < len(lines) and not lines[first_non_empty].strip():
        first_non_empty += 1

    index = first_non_empty
    saw_comment = False
    while index < len(lines):
        stripped = lines[index].lstrip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("//"):
            saw_comment = True
            index += 1
            continue
        break

    if not saw_comment:
        return text
    return "\n".join(lines[index:])


@lru_cache(maxsize=8)
def load_model_catalog(path: str | Path | None = None) -> ModelCatalog:
    """Load model catalog from disk with deterministic caching."""

    resolved = _bundled_catalog_path() if path is None else Path(path).expanduser().resolve()
    return ModelCatalog.from_file(resolved)


__all__ = [
    "ModelCapabilities",
    "ModelCatalog",
    "ModelCostEstimate",
    "load_model_catalog",
]
