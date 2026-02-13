"""
nexus-orchestrator — deterministic repository indexer

File: src/nexus_orchestrator/knowledge_plane/indexer.py
Last updated: 2026-02-13

Purpose
- Index the repository for retrieval: file catalog, symbol map, module map, and dependency graph hints.
- Support deterministic incremental updates and disk persistence for the knowledge plane.

Functional requirements
- Incremental add/modify/delete updates.
- Reverse dependency lookups and module resolution.
- Containment-safe reads from a configured repository root.

Non-functional requirements
- Deterministic output for the same repository snapshot.
- Disk-backed JSON persistence with schema version checks and atomic writes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import string
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Protocol

from nexus_orchestrator.utils.fs import atomic_write

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

PathLike = str | os.PathLike[str]

INDEX_SCHEMA_VERSION: Final[int] = 1
DEFAULT_MAX_FILE_BYTES: Final[int] = 1_000_000
DEFAULT_MAX_IMPORTS_PER_FILE: Final[int] = 256
DEFAULT_MAX_SYMBOLS_PER_FILE: Final[int] = 512
DEFAULT_MAX_MODULES_PER_FILE: Final[int] = 16

_DEFAULT_EXCLUDED_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        "state",
        "workspaces",
        "evidence",
    }
)
_DEFAULT_EXCLUDED_FILES: Final[frozenset[str]] = frozenset()
_DEFAULT_EXCLUDED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".class",
    }
)
_DEFAULT_EXCLUDED_GLOBS: Final[tuple[str, ...]] = (
    "*.egg-info/*",
    "*.min.js",
    "*.min.css",
)

_GENERIC_IMPORT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^\s*import\s+([A-Za-z_][\w.]*)", flags=re.MULTILINE),
    re.compile(r"^\s*from\s+([A-Za-z_][\w.]*)\s+import\b", flags=re.MULTILINE),
    re.compile(r"\bfrom\s+['\"]([A-Za-z0-9_./-]+)['\"]"),
    re.compile(r"\brequire\(\s*['\"]([A-Za-z0-9_./-]+)['\"]\s*\)"),
)
_GENERIC_SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_GENERIC_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "and",
        "class",
        "const",
        "def",
        "else",
        "false",
        "from",
        "function",
        "if",
        "import",
        "let",
        "none",
        "null",
        "or",
        "return",
        "self",
        "true",
        "var",
    }
)
_MODULE_PART_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEX_DIGITS: Final[frozenset[str]] = frozenset(string.hexdigits)


class IndexerLoadError(ValueError):
    """Raised when a persisted index payload is invalid or incompatible."""


class AdapterParseError(ValueError):
    """Raised when a language adapter cannot parse a source payload."""


class LanguageAdapter(Protocol):
    """Protocol for pluggable indexing adapters."""

    @property
    def name(self) -> str: ...

    @property
    def extensions(self) -> frozenset[str]: ...

    def analyze(
        self,
        *,
        source: str,
        relative_path: PurePosixPath,
        module_name: str | None,
    ) -> FileAnalysis: ...


@dataclass(frozen=True, slots=True)
class FileAnalysis:
    """Adapter output for one file."""

    language: str
    imports: tuple[str, ...]
    symbols: tuple[str, ...]
    modules: tuple[str, ...]
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class IndexedFile:
    """Stable file-level index record."""

    relative_path: str
    content_hash: str
    size_bytes: int
    truncated: bool
    language: str
    imports: tuple[str, ...]
    symbols: tuple[str, ...]
    modules: tuple[str, ...]
    parse_error: str | None = None


@dataclass(frozen=True, slots=True)
class IndexExcludes:
    """Centralized repository exclusion policy."""

    directories: frozenset[str] = _DEFAULT_EXCLUDED_DIRECTORIES
    file_names: frozenset[str] = _DEFAULT_EXCLUDED_FILES
    suffixes: frozenset[str] = _DEFAULT_EXCLUDED_SUFFIXES
    globs: tuple[str, ...] = _DEFAULT_EXCLUDED_GLOBS

    def should_exclude(self, relative_path: PurePosixPath, *, is_dir: bool) -> bool:
        """Return whether ``relative_path`` should be excluded from indexing."""

        parts = relative_path.parts
        if any(part in self.directories for part in parts[:-1]):
            return True

        leaf = relative_path.name
        if is_dir and leaf in self.directories:
            return True
        if not is_dir:
            if leaf in self.file_names:
                return True
            if relative_path.suffix.lower() in self.suffixes:
                return True

        candidate = relative_path.as_posix()
        return any(fnmatch(candidate, pattern) for pattern in self.globs)


@dataclass(frozen=True, slots=True)
class _FileReadSnapshot:
    content_hash: str
    source_text: str
    size_bytes: int
    truncated: bool
    binary: bool


@dataclass(frozen=True, slots=True)
class PythonAstAdapter:
    """AST-backed Python adapter for imports/symbols."""

    name: str = "python_ast"
    extensions: frozenset[str] = field(default_factory=lambda: frozenset({".py"}))
    max_imports: int = DEFAULT_MAX_IMPORTS_PER_FILE
    max_symbols: int = DEFAULT_MAX_SYMBOLS_PER_FILE

    def analyze(
        self,
        *,
        source: str,
        relative_path: PurePosixPath,
        module_name: str | None,
    ) -> FileAnalysis:
        is_package = relative_path.name == "__init__.py"
        try:
            tree = ast.parse(source, filename=relative_path.as_posix())
        except SyntaxError as exc:
            line = exc.lineno if exc.lineno is not None else 0
            col = exc.offset if exc.offset is not None else 0
            raise AdapterParseError(f"SyntaxError at line {line}, column {col}: {exc.msg}") from exc

        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name:
                        imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.update(
                    _imports_from_import_from(
                        node=node,
                        module_name=module_name,
                        is_package_module=is_package,
                    )
                )

        symbols: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    symbols.update(_collect_assignment_names(target))
                continue
            if isinstance(node, ast.AnnAssign):
                symbols.update(_collect_assignment_names(node.target))
                continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_name = alias.asname if alias.asname else alias.name.split(".")[0]
                    if imported_name:
                        symbols.add(imported_name)
                continue
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported_name = alias.asname if alias.asname else alias.name
                    if imported_name:
                        symbols.add(imported_name)

        modules = (module_name,) if module_name is not None else ()

        return FileAnalysis(
            language=self.name,
            imports=_stable_terms(imports, self.max_imports),
            symbols=_stable_terms(symbols, self.max_symbols),
            modules=_stable_terms(modules, DEFAULT_MAX_MODULES_PER_FILE),
            parse_error=None,
        )


@dataclass(frozen=True, slots=True)
class GenericTextAdapter:
    """Deterministic, language-agnostic adapter for text files."""

    name: str = "generic_text"
    extensions: frozenset[str] = field(default_factory=frozenset)
    max_imports: int = DEFAULT_MAX_IMPORTS_PER_FILE
    max_symbols: int = DEFAULT_MAX_SYMBOLS_PER_FILE

    def analyze(
        self,
        *,
        source: str,
        relative_path: PurePosixPath,
        module_name: str | None,
    ) -> FileAnalysis:
        del relative_path

        imports: set[str] = set()
        for pattern in _GENERIC_IMPORT_PATTERNS:
            for match in pattern.finditer(source):
                imports.add(match.group(1))

        symbols = {
            token
            for token in _GENERIC_SYMBOL_PATTERN.findall(source)
            if token.lower() not in _GENERIC_STOPWORDS
        }

        modules = (module_name,) if module_name is not None else ()

        return FileAnalysis(
            language=self.name,
            imports=_stable_terms(imports, self.max_imports),
            symbols=_stable_terms(symbols, self.max_symbols),
            modules=_stable_terms(modules, DEFAULT_MAX_MODULES_PER_FILE),
            parse_error=None,
        )


def _default_adapters() -> tuple[LanguageAdapter, ...]:
    return (PythonAstAdapter(),)


def _default_fallback_adapter() -> LanguageAdapter:
    return GenericTextAdapter()


@dataclass(slots=True)
class RepositoryIndexer:
    """
    Deterministic repository indexer with incremental updates and disk persistence.

    The index stores one record per file and derives secondary indexes for:
    - module -> files
    - symbol -> files
    - imported module -> importer files (reverse deps)
    """

    repo_root: Path
    excludes: IndexExcludes = field(default_factory=IndexExcludes)
    adapters: Sequence[LanguageAdapter] = field(default_factory=_default_adapters)
    fallback_adapter: LanguageAdapter = field(default_factory=_default_fallback_adapter)
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES

    _files: dict[str, IndexedFile] = field(init=False, default_factory=dict)
    _module_to_files: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict)
    _symbol_to_files: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict)
    _reverse_deps: dict[str, tuple[str, ...]] = field(init=False, default_factory=dict)
    _adapter_by_extension: dict[str, LanguageAdapter] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        resolved_root = self.repo_root.resolve(strict=True)
        if not resolved_root.is_dir():
            raise NotADirectoryError(f"{resolved_root!s} is not a directory")
        if self.max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be > 0")

        self.repo_root = resolved_root
        self._adapter_by_extension = _build_adapter_registry(self.adapters)

    @property
    def files(self) -> Mapping[str, IndexedFile]:
        """Return a deterministic copy of indexed files keyed by relative path."""

        return dict(sorted(self._files.items(), key=lambda item: item[0]))

    def build(self, *, changed_paths: Iterable[str | Path] | None = None) -> RepositoryIndexer:
        """
        Build or refresh the repository index and return ``self``.

        ``changed_paths=None`` performs a full deterministic rebuild. Otherwise,
        an incremental update is applied after deterministic normalization and
        containment-safe path validation.
        """

        if changed_paths is None:
            self.reindex()
            return self

        normalized = self._normalize_changed_paths(changed_paths, ignore_outside_repo=True)
        self._apply_updates(normalized)
        return self

    def reindex(self) -> None:
        """Rebuild the index from disk using the configured exclusion policy."""

        rebuilt: dict[str, IndexedFile] = {}
        for relative_path in self._iter_candidate_files():
            record = self._index_relative_file(relative_path)
            if record is not None:
                rebuilt[record.relative_path] = record

        self._files = dict(sorted(rebuilt.items(), key=lambda item: item[0]))
        self._rebuild_secondary_indexes()

    def update_paths(self, changed_paths: Iterable[PathLike]) -> None:
        """
        Incrementally apply add/modify/delete updates for selected paths.

        Any excluded path or missing file is removed from the index if present.
        """

        normalized = self._normalize_changed_paths(changed_paths)
        self._apply_updates(normalized)

    def remove_paths(self, paths: Iterable[PathLike]) -> None:
        """Remove paths from the index and rebuild derived maps."""

        normalized = self._normalize_changed_paths(paths)
        for relative_path in normalized:
            self._files.pop(relative_path.as_posix(), None)
        self._rebuild_secondary_indexes()

    def resolve_module(self, module_name: str) -> tuple[str, ...]:
        """Return files that define ``module_name``."""

        normalized = _normalize_non_empty(module_name)
        return self._module_to_files.get(normalized, ())

    def reverse_dependencies(self, module_name: str) -> tuple[str, ...]:
        """Return files that import ``module_name``."""

        normalized = _normalize_non_empty(module_name)
        return self._reverse_deps.get(normalized, ())

    def lookup_symbol(self, symbol: str) -> tuple[str, ...]:
        """Return files that define ``symbol``."""

        normalized = _normalize_non_empty(symbol)
        return self._symbol_to_files.get(normalized, ())

    def files_for_module(self, module_name: str) -> tuple[str, ...]:
        """Return module-defining files plus reverse-dependency files."""

        normalized = _normalize_non_empty(module_name)
        related = set(self.resolve_module(normalized))
        related.update(self.reverse_dependencies(normalized))
        return tuple(sorted(related))

    def relevant_files(
        self, *, module_name: str | None = None, symbol: str | None = None
    ) -> tuple[str, ...]:
        """
        Return files relevant to a module and/or symbol query.

        At least one query argument must be provided.
        """

        if module_name is None and symbol is None:
            raise ValueError("at least one of module_name or symbol must be provided")

        relevant: set[str] = set()
        if module_name is not None:
            normalized_module = _normalize_non_empty(module_name)
            relevant.update(self.resolve_module(normalized_module))
            relevant.update(self.reverse_dependencies(normalized_module))
        if symbol is not None:
            normalized_symbol = _normalize_non_empty(symbol)
            relevant.update(self.lookup_symbol(normalized_symbol))

        return tuple(sorted(relevant))

    def save(self, destination: PathLike) -> Path:
        """Atomically persist the current index under ``destination``."""

        target = self._resolve_storage_path(destination, require_exists=False)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "repo_root": self.repo_root.as_posix(),
            "files": [
                _serialize_indexed_file(record)
                for _, record in sorted(self._files.items(), key=lambda item: item[0])
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        atomic_write(target, encoded, encoding="utf-8")
        return target

    @classmethod
    def load(
        cls,
        repo_root: PathLike,
        source: PathLike,
        *,
        excludes: IndexExcludes | None = None,
        adapters: Sequence[LanguageAdapter] | None = None,
        fallback_adapter: LanguageAdapter | None = None,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> RepositoryIndexer:
        """Load a persisted index from disk with schema/version/root validation."""

        indexer = cls(
            repo_root=Path(repo_root),
            excludes=excludes if excludes is not None else IndexExcludes(),
            adapters=adapters if adapters is not None else _default_adapters(),
            fallback_adapter=(
                fallback_adapter if fallback_adapter is not None else _default_fallback_adapter()
            ),
            max_file_bytes=max_file_bytes,
        )

        source_path = indexer._resolve_storage_path(source, require_exists=True)
        raw = source_path.read_text(encoding="utf-8")
        payload = json.loads(raw)

        if not isinstance(payload, dict):
            raise IndexerLoadError("index payload must be a JSON object")

        schema_version = payload.get("schema_version")
        if schema_version != INDEX_SCHEMA_VERSION:
            raise IndexerLoadError(
                f"unsupported schema_version: {schema_version!r}; expected {INDEX_SCHEMA_VERSION}"
            )

        payload_root = payload.get("repo_root")
        if not isinstance(payload_root, str):
            raise IndexerLoadError("repo_root must be a string")
        if payload_root != indexer.repo_root.as_posix():
            raise IndexerLoadError(
                "repo_root mismatch between payload and requested repository root"
            )

        files_payload = payload.get("files")
        if not isinstance(files_payload, list):
            raise IndexerLoadError("files must be a JSON array")

        loaded_files: dict[str, IndexedFile] = {}
        for item in files_payload:
            record = _deserialize_indexed_file(item)
            relative = _parse_relative_posix_path(record.relative_path)
            if indexer.excludes.should_exclude(relative, is_dir=False):
                continue
            loaded_files[record.relative_path] = record

        indexer._files = dict(sorted(loaded_files.items(), key=lambda pair: pair[0]))
        indexer._rebuild_secondary_indexes()
        return indexer

    def _iter_candidate_files(self) -> list[PurePosixPath]:
        candidates: list[PurePosixPath] = []
        for current_dir, dir_names, file_names in os.walk(
            self.repo_root, topdown=True, followlinks=False
        ):
            current_path = Path(current_dir)

            kept_dirs: list[str] = []
            for directory in sorted(dir_names):
                dir_path = current_path / directory
                relative_dir = self._relative_path_from_local(dir_path)
                if relative_dir is None:
                    continue
                if self.excludes.should_exclude(relative_dir, is_dir=True):
                    continue
                kept_dirs.append(directory)
            dir_names[:] = kept_dirs

            for file_name in sorted(file_names):
                file_path = current_path / file_name
                relative_file = self._relative_path_from_local(file_path)
                if relative_file is None:
                    continue
                if self.excludes.should_exclude(relative_file, is_dir=False):
                    continue
                candidates.append(relative_file)

        candidates.sort(key=lambda path: path.as_posix())
        return candidates

    def _index_relative_file(self, relative_path: PurePosixPath) -> IndexedFile | None:
        local_path = self.repo_root.joinpath(*relative_path.parts)
        if not local_path.exists() or not local_path.is_file():
            return None
        if local_path.is_symlink():
            return None

        try:
            resolved = local_path.resolve(strict=True)
        except OSError:
            return None
        if not _is_relative_to(resolved, self.repo_root):
            return None

        snapshot = _read_file_snapshot(local_path, max_file_bytes=self.max_file_bytes)
        if snapshot.binary:
            return None

        module_name = _python_module_name(relative_path)
        adapter = self._adapter_for_path(relative_path)
        analysis: FileAnalysis
        parse_error = ""

        try:
            analysis = adapter.analyze(
                source=snapshot.source_text,
                relative_path=relative_path,
                module_name=module_name,
            )
        except AdapterParseError as exc:
            fallback = self.fallback_adapter.analyze(
                source=snapshot.source_text,
                relative_path=relative_path,
                module_name=module_name,
            )
            analysis = FileAnalysis(
                language=fallback.language,
                imports=fallback.imports,
                symbols=fallback.symbols,
                modules=fallback.modules,
                parse_error=str(exc),
            )
            parse_error = str(exc)

        if analysis.parse_error is not None and not parse_error:
            parse_error = analysis.parse_error
        if snapshot.truncated:
            suffix = f"truncated to first {self.max_file_bytes} bytes"
            parse_error = f"{parse_error}; {suffix}" if parse_error else suffix

        return IndexedFile(
            relative_path=relative_path.as_posix(),
            content_hash=snapshot.content_hash,
            size_bytes=snapshot.size_bytes,
            truncated=snapshot.truncated,
            language=analysis.language,
            imports=_stable_terms(analysis.imports, DEFAULT_MAX_IMPORTS_PER_FILE),
            symbols=_stable_terms(analysis.symbols, DEFAULT_MAX_SYMBOLS_PER_FILE),
            modules=_stable_terms(analysis.modules, DEFAULT_MAX_MODULES_PER_FILE),
            parse_error=parse_error or None,
        )

    def _adapter_for_path(self, relative_path: PurePosixPath) -> LanguageAdapter:
        extension = relative_path.suffix.lower()
        return self._adapter_by_extension.get(extension, self.fallback_adapter)

    def _normalize_changed_paths(
        self,
        paths: Iterable[PathLike],
        *,
        ignore_outside_repo: bool = False,
    ) -> tuple[PurePosixPath, ...]:
        normalized: set[PurePosixPath] = set()
        for path in paths:
            try:
                normalized.add(self._normalize_relative_path(path))
            except ValueError:
                if ignore_outside_repo:
                    continue
                raise
        return tuple(sorted(normalized, key=lambda value: value.as_posix()))

    def _apply_updates(self, normalized_paths: Iterable[PurePosixPath]) -> None:
        for relative_path in normalized_paths:
            key = relative_path.as_posix()
            if self.excludes.should_exclude(relative_path, is_dir=False):
                self._files.pop(key, None)
                continue

            record = self._index_relative_file(relative_path)
            if record is None:
                self._files.pop(key, None)
                continue

            prior = self._files.get(key)
            if prior is not None and prior == record:
                continue
            self._files[key] = record

        self._files = dict(sorted(self._files.items(), key=lambda item: item[0]))
        self._rebuild_secondary_indexes()

    def _normalize_relative_path(self, path: PathLike) -> PurePosixPath:
        candidate = Path(path)
        if candidate.is_absolute():
            resolved_candidate = candidate.resolve(strict=False)
            if not _is_relative_to(resolved_candidate, self.repo_root):
                raise ValueError(f"path outside repository root: {candidate!s}")
            relative = PurePosixPath(resolved_candidate.relative_to(self.repo_root).as_posix())
        else:
            relative = PurePosixPath(candidate.as_posix())

        _validate_safe_relative_path(relative)
        return relative

    def _relative_path_from_local(self, path: Path) -> PurePosixPath | None:
        try:
            relative = path.relative_to(self.repo_root)
        except ValueError:
            return None
        relative_posix = PurePosixPath(relative.as_posix())
        try:
            _validate_safe_relative_path(relative_posix)
        except ValueError:
            return None
        return relative_posix

    def _resolve_storage_path(self, path: PathLike, *, require_exists: bool) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate

        if require_exists:
            resolved = candidate.resolve(strict=True)
            if not _is_relative_to(resolved, self.repo_root):
                raise ValueError(f"storage path outside repository root: {candidate!s}")
            return resolved

        resolved_parent = candidate.parent.resolve(strict=True)
        if not _is_relative_to(resolved_parent, self.repo_root):
            raise ValueError(f"storage path outside repository root: {candidate!s}")
        return resolved_parent / candidate.name

    def _rebuild_secondary_indexes(self) -> None:
        module_map: dict[str, set[str]] = {}
        symbol_map: dict[str, set[str]] = {}
        reverse_map: dict[str, set[str]] = {}

        for path, record in sorted(self._files.items(), key=lambda item: item[0]):
            for module_name in record.modules:
                module_map.setdefault(module_name, set()).add(path)
            for symbol in record.symbols:
                symbol_map.setdefault(symbol, set()).add(path)
            for dependency in record.imports:
                reverse_map.setdefault(dependency, set()).add(path)

        self._module_to_files = {
            key: tuple(sorted(value))
            for key, value in sorted(module_map.items(), key=lambda item: item[0])
        }
        self._symbol_to_files = {
            key: tuple(sorted(value))
            for key, value in sorted(symbol_map.items(), key=lambda item: item[0])
        }
        self._reverse_deps = {
            key: tuple(sorted(value))
            for key, value in sorted(reverse_map.items(), key=lambda item: item[0])
        }


def _build_adapter_registry(adapters: Sequence[LanguageAdapter]) -> dict[str, LanguageAdapter]:
    registry: dict[str, LanguageAdapter] = {}
    for adapter in adapters:
        for extension in adapter.extensions:
            normalized = extension.lower().strip()
            if not normalized.startswith("."):
                raise ValueError(f"adapter extension must start with '.': {extension!r}")
            if normalized in registry:
                raise ValueError(f"duplicate adapter registration for extension {normalized!r}")
            registry[normalized] = adapter
    return registry


def _read_file_snapshot(path: Path, *, max_file_bytes: int) -> _FileReadSnapshot:
    digest = hashlib.sha256()
    collected = bytearray()
    size_bytes = 0

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size_bytes += len(chunk)
            if len(collected) < max_file_bytes:
                remaining = max_file_bytes - len(collected)
                collected.extend(chunk[:remaining])

    raw = bytes(collected)
    truncated = size_bytes > max_file_bytes
    binary = b"\x00" in raw
    return _FileReadSnapshot(
        content_hash=digest.hexdigest(),
        source_text=raw.decode("utf-8", errors="replace"),
        size_bytes=size_bytes,
        truncated=truncated,
        binary=binary,
    )


def _imports_from_import_from(
    *,
    node: ast.ImportFrom,
    module_name: str | None,
    is_package_module: bool,
) -> set[str]:
    imports: set[str] = set()

    resolved_base = _resolve_import_base(
        module_name=module_name,
        is_package_module=is_package_module,
        level=node.level,
        module=node.module,
    )

    if resolved_base is not None:
        imports.add(resolved_base)

    for alias in node.names:
        if alias.name == "*":
            continue
        if resolved_base is None:
            continue
        if resolved_base:
            imports.add(f"{resolved_base}.{alias.name}")
        else:
            imports.add(alias.name)

    return imports


def _resolve_import_base(
    *,
    module_name: str | None,
    is_package_module: bool,
    level: int,
    module: str | None,
) -> str | None:
    if level <= 0:
        return module

    if module_name is None:
        return module

    anchor_parts = module_name.split(".")
    if not is_package_module and anchor_parts:
        anchor_parts = anchor_parts[:-1]

    pops = level - 1
    if pops > len(anchor_parts):
        return None

    base_parts = anchor_parts[: len(anchor_parts) - pops]
    module_parts = module.split(".") if module else []
    resolved_parts = [*base_parts, *module_parts]
    if not resolved_parts:
        return ""
    return ".".join(resolved_parts)


def _collect_assignment_names(node: ast.expr) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(_collect_assignment_names(elt))
    return names


def _python_module_name(relative_path: PurePosixPath) -> str | None:
    if relative_path.suffix != ".py":
        return None

    parts = list(relative_path.parts)
    if not parts:
        return None

    module_parts = parts[:-1] if parts[-1] == "__init__.py" else [*parts[:-1], relative_path.stem]

    if not module_parts:
        return None
    if any(_MODULE_PART_PATTERN.fullmatch(part) is None for part in module_parts):
        return None
    return ".".join(module_parts)


def _stable_terms(values: Iterable[str], limit: int) -> tuple[str, ...]:
    if limit <= 0:
        return ()
    normalized = sorted({value for value in values if value})
    return tuple(normalized[:limit])


def _normalize_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_safe_relative_path(path: PurePosixPath) -> None:
    if path.is_absolute():
        raise ValueError(f"path must be relative: {path.as_posix()!r}")
    if not path.parts:
        raise ValueError("path must not be empty")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"path is not containment-safe: {path.as_posix()!r}")


def _parse_relative_posix_path(value: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    _validate_safe_relative_path(candidate)
    return candidate


def _serialize_indexed_file(record: IndexedFile) -> dict[str, object]:
    return {
        "relative_path": record.relative_path,
        "content_hash": record.content_hash,
        "size_bytes": record.size_bytes,
        "truncated": record.truncated,
        "language": record.language,
        "imports": list(record.imports),
        "symbols": list(record.symbols),
        "modules": list(record.modules),
        "parse_error": record.parse_error,
    }


def _deserialize_indexed_file(raw: object) -> IndexedFile:
    if not isinstance(raw, dict):
        raise IndexerLoadError("each file entry must be an object")

    relative_path = _expect_string(raw, "relative_path")
    content_hash = _expect_string(raw, "content_hash")
    if len(content_hash) != 64 or any(char not in _HEX_DIGITS for char in content_hash):
        raise IndexerLoadError("content_hash must be a 64-character SHA-256 hex digest")

    size_bytes = _expect_int(raw, "size_bytes")
    if size_bytes < 0:
        raise IndexerLoadError("size_bytes must be >= 0")

    truncated = _expect_bool(raw, "truncated")
    language = _expect_string(raw, "language")

    imports = _expect_string_list(raw, "imports")
    symbols = _expect_string_list(raw, "symbols")
    modules = _expect_string_list(raw, "modules")

    parse_error_value = raw.get("parse_error")
    if parse_error_value is not None and not isinstance(parse_error_value, str):
        raise IndexerLoadError("parse_error must be a string or null")

    _validate_safe_relative_path(PurePosixPath(relative_path))

    return IndexedFile(
        relative_path=relative_path,
        content_hash=content_hash,
        size_bytes=size_bytes,
        truncated=truncated,
        language=language,
        imports=_stable_terms(imports, DEFAULT_MAX_IMPORTS_PER_FILE),
        symbols=_stable_terms(symbols, DEFAULT_MAX_SYMBOLS_PER_FILE),
        modules=_stable_terms(modules, DEFAULT_MAX_MODULES_PER_FILE),
        parse_error=parse_error_value,
    )


def _expect_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise IndexerLoadError(f"{key} must be a string")
    return value


def _expect_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise IndexerLoadError(f"{key} must be an integer")
    return value


def _expect_bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise IndexerLoadError(f"{key} must be a boolean")
    return value


def _expect_string_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise IndexerLoadError(f"{key} must be a list")

    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise IndexerLoadError(f"{key} must contain only strings")
        strings.append(item)

    return tuple(strings)


__all__ = [
    "AdapterParseError",
    "FileAnalysis",
    "GenericTextAdapter",
    "INDEX_SCHEMA_VERSION",
    "IndexedFile",
    "IndexExcludes",
    "IndexerLoadError",
    "LanguageAdapter",
    "PythonAstAdapter",
    "RepositoryIndexer",
]
