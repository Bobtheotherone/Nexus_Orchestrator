"""
nexus-orchestrator — evidence artifact writer and integrity utilities

File: src/nexus_orchestrator/verification_plane/evidence.py
Last updated: 2026-02-12

Purpose
- Write verification evidence artifacts using a stable append-only filesystem layout.
- Redact secrets before persistence and produce deterministic metadata/manifest JSON.
- Verify artifact integrity and export offline audit bundles.

Storage layout
- `<evidence_root>/<run_id>/<work_item_id>/<stage>/<evidence_id>/`
- `<evidence_root>/<run_id>/<work_item_id>/<attempt_id>/<stage>/<evidence_id>/` (optional attempt segment)
  - `result.json` (redacted normalized checker result payload)
  - `metadata.json` (redacted, deterministic key ordering)
  - `logs/` (redacted log text)
  - `artifacts/` (inline or copied artifacts; copied text is redacted)
  - `manifest.json` (deterministic SHA-256 manifest for every file except itself)

Functional requirements
- Append-only semantics: existing evidence directories are never mutated.
- Deterministic ordering for metadata keys and manifest entries.
- Offline-only implementation (no network access).
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias, TypeVar

from nexus_orchestrator.domain import ids as domain_ids
from nexus_orchestrator.security.redaction import RedactionConfig, redact_structure, redact_text
from nexus_orchestrator.utils.fs import atomic_write
from nexus_orchestrator.utils.hashing import sha256_file

PathLike: TypeAlias = str | os.PathLike[str]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

_METADATA_FILE_NAME: Final[str] = "metadata.json"
_RESULT_FILE_NAME: Final[str] = "result.json"
_MANIFEST_FILE_NAME: Final[str] = "manifest.json"
_LOGS_DIR: Final[PurePosixPath] = PurePosixPath("logs")
_ARTIFACTS_DIR: Final[PurePosixPath] = PurePosixPath("artifacts")
_MANIFEST_SCHEMA_VERSION: Final[int] = 1
_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_TEXT_FILE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".cfg",
        ".conf",
        ".csv",
        ".diff",
        ".env",
        ".ini",
        ".json",
        ".log",
        ".md",
        ".patch",
        ".py",
        ".rst",
        ".sh",
        ".sql",
        ".toml",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
)
_ZIP_FIXED_TIMESTAMP: Final[tuple[int, int, int, int, int, int]] = (1980, 1, 1, 0, 0, 0)

TValue = TypeVar("TValue")


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One deterministic manifest entry."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class EvidenceWriteResult:
    """Result of a successful append-only evidence write."""

    evidence_id: str
    evidence_dir: Path
    metadata_path: Path
    manifest_path: Path
    manifest_entries: tuple[ManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Deterministic integrity verification report for one evidence directory."""

    evidence_dir: Path
    manifest_path: Path
    missing_paths: tuple[str, ...]
    hash_mismatches: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_paths and not self.hash_mismatches


class EvidenceWriter:
    """Append-only evidence writer rooted at ``evidence_root``."""

    __slots__ = ("_evidence_root", "_redaction_config")

    def __init__(
        self,
        evidence_root: PathLike,
        *,
        redaction_config: RedactionConfig | None = None,
    ) -> None:
        root = Path(evidence_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._evidence_root = root
        self._redaction_config = redaction_config

    @property
    def evidence_root(self) -> Path:
        return self._evidence_root

    def evidence_dir_for(
        self,
        *,
        run_id: str,
        work_item_id: str,
        attempt_id: str | None = None,
        stage: str,
        evidence_id: str,
    ) -> Path:
        """Return canonical evidence directory path for explicit IDs."""

        run_segment = _normalize_path_segment(run_id, label="run_id")
        work_item_segment = _normalize_path_segment(work_item_id, label="work_item_id")
        attempt_segment = (
            _normalize_path_segment(attempt_id, label="attempt_id")
            if attempt_id is not None
            else None
        )
        stage_segment = _normalize_path_segment(stage, label="stage")
        evidence_segment = _normalize_path_segment(evidence_id, label="evidence_id")
        if attempt_segment is None:
            return (
                self._evidence_root
                / run_segment
                / work_item_segment
                / stage_segment
                / evidence_segment
            )
        return (
            self._evidence_root
            / run_segment
            / work_item_segment
            / attempt_segment
            / stage_segment
            / evidence_segment
        )

    def write_evidence(
        self,
        *,
        run_id: str,
        work_item_id: str,
        attempt_id: str | None = None,
        stage: str,
        evidence_id: str | None = None,
        result: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        logs: Mapping[str, str] | None = None,
        artifacts: Mapping[str, object] | None = None,
        copied_artifacts: Mapping[str, PathLike] | None = None,
    ) -> EvidenceWriteResult:
        """
        Persist one evidence directory in append-only mode.

        ``artifacts`` supports values of type:
        - ``str``: written as redacted UTF-8 text
        - ``bytes`` / ``bytearray`` / ``memoryview``: written as binary
        - ``os.PathLike``: source file copied into evidence (copied text is redacted)
        - ``Mapping`` / ``Sequence``: written as redacted deterministic JSON
        - any other value: stringified and redacted as text
        """

        resolved_evidence_id = (
            domain_ids.generate_evidence_id()
            if evidence_id is None
            else _normalize_path_segment(evidence_id, label="evidence_id")
        )

        evidence_dir = self.evidence_dir_for(
            run_id=run_id,
            work_item_id=work_item_id,
            attempt_id=attempt_id,
            stage=stage,
            evidence_id=resolved_evidence_id,
        )
        if evidence_dir.exists():
            raise FileExistsError(f"evidence directory already exists: {evidence_dir}")

        evidence_dir.parent.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=False, exist_ok=False)

        metadata_payload: dict[str, object] = {}
        if metadata is not None:
            metadata_payload.update(metadata)
        metadata_payload.setdefault("run_id", run_id)
        metadata_payload.setdefault("work_item_id", work_item_id)
        if attempt_id is not None:
            metadata_payload.setdefault("attempt_id", attempt_id)
        metadata_payload.setdefault("stage", stage)
        metadata_payload.setdefault("evidence_id", resolved_evidence_id)

        metadata_json = _redact_json_value(
            _to_json_value(metadata_payload),
            redaction_config=self._redaction_config,
        )
        if not isinstance(metadata_json, dict):
            raise TypeError("metadata payload must serialize to a JSON object")

        metadata_path = evidence_dir / _METADATA_FILE_NAME
        _write_json_file(metadata_path, metadata_json)

        written_rel_paths: set[str] = {_METADATA_FILE_NAME}

        if result is not None:
            result_json = _redact_json_value(
                _to_json_value(result),
                redaction_config=self._redaction_config,
            )
            result_path = evidence_dir / _RESULT_FILE_NAME
            _write_json_file(result_path, result_json)
            written_rel_paths.add(_RESULT_FILE_NAME)

        if logs is not None:
            for log_name, log_text in _sorted_mapping_items(logs, field_name="logs"):
                rel_path = _normalize_artifact_path(
                    log_name,
                    base_dir=_LOGS_DIR,
                    default_suffix=".log",
                )
                rel_path_text = rel_path.as_posix()
                _ensure_unique_rel_path(rel_path_text, written_rel_paths)
                redacted_log = redact_text(log_text, config=self._redaction_config)
                _write_text_file(evidence_dir / Path(*rel_path.parts), redacted_log)

        if artifacts is not None:
            for artifact_name, artifact_value in _sorted_mapping_items(
                artifacts, field_name="artifacts"
            ):
                rel_path = _normalize_artifact_path(artifact_name, base_dir=_ARTIFACTS_DIR)
                rel_path_text = rel_path.as_posix()
                _ensure_unique_rel_path(rel_path_text, written_rel_paths)
                destination = evidence_dir / Path(*rel_path.parts)
                self._write_artifact_value(destination=destination, value=artifact_value)

        if copied_artifacts is not None:
            for artifact_name, source_path in _sorted_mapping_items(
                copied_artifacts,
                field_name="copied_artifacts",
            ):
                rel_path = _normalize_artifact_path(artifact_name, base_dir=_ARTIFACTS_DIR)
                rel_path_text = rel_path.as_posix()
                _ensure_unique_rel_path(rel_path_text, written_rel_paths)
                destination = evidence_dir / Path(*rel_path.parts)
                _copy_artifact_with_redaction(
                    source=Path(source_path),
                    destination=destination,
                    redaction_config=self._redaction_config,
                )

        manifest_entries = _build_manifest_entries(evidence_dir)
        manifest_payload = {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "entries": [
                {
                    "path": entry.path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for entry in manifest_entries
            ],
        }
        manifest_json = _to_json_value(manifest_payload)
        if not isinstance(manifest_json, dict):
            raise TypeError("manifest payload must serialize to a JSON object")

        manifest_path = evidence_dir / _MANIFEST_FILE_NAME
        _write_json_file(manifest_path, manifest_json)

        return EvidenceWriteResult(
            evidence_id=resolved_evidence_id,
            evidence_dir=evidence_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            manifest_entries=manifest_entries,
        )

    def _write_artifact_value(self, *, destination: Path, value: object) -> None:
        if isinstance(value, (bytes, bytearray, memoryview)):
            _write_bytes_file(destination, bytes(value))
            return

        if isinstance(value, str):
            _write_text_file(destination, redact_text(value, config=self._redaction_config))
            return

        if isinstance(value, os.PathLike):
            _copy_artifact_with_redaction(
                source=Path(value),
                destination=destination,
                redaction_config=self._redaction_config,
            )
            return

        if isinstance(value, Mapping):
            payload = _redact_json_value(
                _to_json_value(value),
                redaction_config=self._redaction_config,
            )
            _write_json_file(destination, payload)
            return

        if isinstance(value, Sequence):
            payload = _redact_json_value(
                _to_json_value(list(value)),
                redaction_config=self._redaction_config,
            )
            _write_json_file(destination, payload)
            return

        rendered = redact_text(str(value), config=self._redaction_config)
        _write_text_file(destination, rendered)


def verify_integrity(evidence_dir: PathLike) -> IntegrityReport:
    """
    Verify ``manifest.json`` for one evidence directory.

    Reports missing files and SHA-256 mismatches in deterministic path order.
    """

    root = Path(evidence_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    manifest_path = root / _MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    entries = _load_manifest_entries(manifest_path)

    missing_paths: list[str] = []
    hash_mismatches: list[str] = []

    for entry in entries:
        target = root.joinpath(*PurePosixPath(entry.path).parts)

        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            missing_paths.append(entry.path)
            continue

        if not stat.S_ISREG(mode):
            hash_mismatches.append(entry.path)
            continue

        if target.stat().st_size != entry.size_bytes:
            hash_mismatches.append(entry.path)
            continue

        if sha256_file(target) != entry.sha256:
            hash_mismatches.append(entry.path)

    return IntegrityReport(
        evidence_dir=root,
        manifest_path=manifest_path,
        missing_paths=tuple(missing_paths),
        hash_mismatches=tuple(hash_mismatches),
    )


def export_audit_bundle(
    evidence_dir: PathLike,
    *,
    output_path: PathLike | None = None,
) -> Path:
    """
    Export a deterministic zip bundle containing the evidence directory and manifest.

    Returns the final bundle path.
    """

    root = Path(evidence_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    integrity = verify_integrity(root)
    if not integrity.is_valid:
        missing_csv = ", ".join(integrity.missing_paths) or "<none>"
        mismatch_csv = ", ".join(integrity.hash_mismatches) or "<none>"
        raise ValueError(
            "cannot export bundle for evidence that failed integrity checks: "
            f"missing=[{missing_csv}] mismatches=[{mismatch_csv}]"
        )

    resolved_output = (
        root.parent / f"{root.name}.audit.zip"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    if _is_relative_to(resolved_output, root):
        raise ValueError("output_path must not be inside the evidence directory")

    archive_members = [
        file_path
        for file_path in _iter_regular_files(root)
        if file_path.relative_to(root).as_posix()
    ]

    temp_output = resolved_output.with_name(f".{resolved_output.name}.tmp")
    if temp_output.exists():
        temp_output.unlink()

    try:
        with zipfile.ZipFile(
            temp_output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            allowZip64=True,
        ) as archive:
            for file_path in archive_members:
                rel_path = file_path.relative_to(root).as_posix()
                member_name = f"{root.name}/{rel_path}"

                zip_info = zipfile.ZipInfo(filename=member_name)
                zip_info.date_time = _ZIP_FIXED_TIMESTAMP
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zip_info.external_attr = (0o100644 & 0xFFFF) << 16
                zip_info.create_system = 3

                archive.writestr(zip_info, file_path.read_bytes())

        os.replace(temp_output, resolved_output)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    return resolved_output


def _build_manifest_entries(root: Path) -> tuple[ManifestEntry, ...]:
    entries: list[ManifestEntry] = []
    for file_path in _iter_regular_files(root):
        rel_path = file_path.relative_to(root).as_posix()
        if rel_path == _MANIFEST_FILE_NAME:
            continue
        entries.append(
            ManifestEntry(
                path=rel_path,
                sha256=sha256_file(file_path),
                size_bytes=file_path.stat().st_size,
            )
        )

    entries.sort(key=lambda item: item.path)
    return tuple(entries)


def _load_manifest_entries(manifest_path: Path) -> tuple[ManifestEntry, ...]:
    try:
        parsed: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {manifest_path}: {exc.msg}") from exc

    if not isinstance(parsed, Mapping):
        raise ValueError(f"manifest root must be a JSON object: {manifest_path}")

    entries_raw = parsed.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError(f"manifest.entries must be a JSON array: {manifest_path}")

    seen_paths: set[str] = set()
    entries: list[ManifestEntry] = []

    for index, candidate in enumerate(entries_raw):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"manifest.entries[{index}] must be an object")

        path_value = candidate.get("path")
        hash_value = candidate.get("sha256")
        size_value = candidate.get("size_bytes")

        if not isinstance(path_value, str):
            raise ValueError(f"manifest.entries[{index}].path must be a string")
        if not isinstance(hash_value, str):
            raise ValueError(f"manifest.entries[{index}].sha256 must be a string")
        if isinstance(size_value, bool) or not isinstance(size_value, int) or size_value < 0:
            raise ValueError(f"manifest.entries[{index}].size_bytes must be a non-negative integer")

        normalized_path = _normalize_relative_manifest_path(path_value)
        normalized_hash = hash_value.lower()
        if _SHA256_HEX_RE.fullmatch(normalized_hash) is None:
            raise ValueError(f"manifest.entries[{index}].sha256 must be a lowercase SHA-256 hex")

        if normalized_path in seen_paths:
            raise ValueError(f"manifest contains duplicate path: {normalized_path}")
        seen_paths.add(normalized_path)

        entries.append(
            ManifestEntry(
                path=normalized_path,
                sha256=normalized_hash,
                size_bytes=size_value,
            )
        )

    entries.sort(key=lambda item: item.path)
    return tuple(entries)


def _normalize_relative_manifest_path(path_value: str) -> str:
    if "\\" in path_value:
        raise ValueError(f"manifest path must use POSIX separators: {path_value!r}")
    posix = PurePosixPath(path_value)
    if not path_value or posix.is_absolute():
        raise ValueError(f"manifest path must be relative and non-empty: {path_value!r}")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"manifest path is not safe: {path_value!r}")
    return posix.as_posix()


def _iter_regular_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for current_dir, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
        dir_names.sort()
        file_names.sort()

        current_path = Path(current_dir)
        for file_name in file_names:
            candidate = current_path / file_name
            try:
                mode = candidate.lstat().st_mode
            except FileNotFoundError:
                continue

            if stat.S_ISREG(mode):
                files.append(candidate)

    files.sort(key=lambda item: item.relative_to(root).as_posix())
    return tuple(files)


def _normalize_path_segment(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"{label} must not contain path separators: {value!r}")
    if normalized in {".", ".."}:
        raise ValueError(f"{label} must not be '.' or '..'")
    return normalized


def _normalize_artifact_path(
    artifact_name: str,
    *,
    base_dir: PurePosixPath,
    default_suffix: str | None = None,
) -> PurePosixPath:
    if not isinstance(artifact_name, str):
        raise TypeError(f"artifact path must be a string, got {type(artifact_name).__name__}")

    normalized = artifact_name.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("artifact path cannot be empty")

    relative_path = PurePosixPath(normalized)
    if relative_path.is_absolute():
        raise ValueError(f"artifact path must be relative: {artifact_name!r}")
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ValueError(f"artifact path is not safe: {artifact_name!r}")

    if default_suffix is not None and relative_path.suffix == "":
        relative_path = relative_path.with_suffix(default_suffix)

    return base_dir.joinpath(*relative_path.parts)


def _ensure_unique_rel_path(path_value: str, seen_paths: set[str]) -> None:
    if path_value in seen_paths:
        raise ValueError(f"duplicate artifact path: {path_value}")
    seen_paths.add(path_value)


def _sorted_mapping_items(
    mapping: Mapping[str, TValue],
    *,
    field_name: str,
) -> tuple[tuple[str, TValue], ...]:
    items: list[tuple[str, TValue]] = []
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{field_name} keys must be strings, got {type(key).__name__}")
        items.append((key, value))
    items.sort(key=lambda item: item[0])
    return tuple(items)


def _copy_artifact_with_redaction(
    *,
    source: Path,
    destination: Path,
    redaction_config: RedactionConfig | None,
) -> None:
    resolved_source = source.expanduser().resolve(strict=True)
    if not resolved_source.is_file():
        raise FileNotFoundError(f"artifact source file not found: {resolved_source}")

    payload = resolved_source.read_bytes()
    decoded_text = _decode_text_if_applicable(payload, suffix=resolved_source.suffix.lower())
    if decoded_text is None:
        _write_bytes_file(destination, payload)
        return

    _write_text_file(destination, redact_text(decoded_text, config=redaction_config))


def _decode_text_if_applicable(payload: bytes, *, suffix: str) -> str | None:
    """Best-effort text detection for copied artifacts."""

    prefer_text = suffix in _TEXT_FILE_EXTENSIONS
    if not prefer_text and b"\x00" in payload:
        return None

    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _to_json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)

    if isinstance(value, PurePosixPath):
        return value.as_posix()

    if isinstance(value, Path):
        return value.as_posix()

    if isinstance(value, (bytes, bytearray, memoryview)):
        bytes_value = bytes(value)
        decoded = _decode_text_if_applicable(bytes_value, suffix="")
        if decoded is not None:
            return decoded
        return bytes_value.hex()

    if isinstance(value, Mapping):
        items: list[tuple[str, JSONValue]] = []
        keys = sorted(value.keys(), key=lambda item: str(item))
        for key in keys:
            key_text = str(key)
            items.append((key_text, _to_json_value(value[key])))
        return {key: item for key, item in items}

    if isinstance(value, Sequence):
        return [_to_json_value(item) for item in value]

    if isinstance(value, set):
        normalized = [_to_json_value(item) for item in value]
        normalized.sort(key=_stable_json_sort_key)
        return normalized

    return str(value)


def _redact_json_value(
    value: JSONValue,
    *,
    redaction_config: RedactionConfig | None,
) -> JSONValue:
    redacted = redact_structure(value, config=redaction_config)
    return _to_json_value(redacted)


def _stable_json_sort_key(value: JSONValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, text, encoding="utf-8")


def _write_bytes_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, payload)


def _write_json_file(
    path: Path, payload: Mapping[str, JSONValue] | Sequence[JSONValue] | JSONValue
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    atomic_write(path, f"{rendered}\n", encoding="utf-8")


def _is_relative_to(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "EvidenceWriteResult",
    "EvidenceWriter",
    "IntegrityReport",
    "ManifestEntry",
    "export_audit_bundle",
    "verify_integrity",
]
