"""
nexus-orchestrator — hashing utilities

File: src/nexus_orchestrator/utils/hashing.py
Last updated: 2026-02-12

Purpose
- Provide deterministic SHA-256 helpers for bytes, text, and files.
- Build and verify directory manifests for evidence integrity checks.

Functional requirements
- Manifest paths are relative POSIX strings with deterministic ordering.
- Verification reports missing and corrupted entries in deterministic order.

Non-functional requirements
- Standard library only; behavior is cross-platform deterministic.
"""

from __future__ import annotations

import hashlib
import os
import stat
import string
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

PathLike = str | os.PathLike[str]

_SHA256_HEX_LENGTH = 64
_FILE_READ_CHUNK_BYTES = 1024 * 1024
_HEX_DIGITS = set(string.hexdigits)

__all__ = [
    "create_manifest",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "verify_manifest",
]


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for raw bytes."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    """Return SHA-256 hex digest for text encoded with ``encoding``."""

    return sha256_bytes(text.encode(encoding))


def sha256_file(path: PathLike, *, chunk_size: int = _FILE_READ_CHUNK_BYTES) -> str:
    """Return SHA-256 hex digest for a file read in chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(directory: PathLike) -> dict[str, str]:
    """
    Build a deterministic file manifest for ``directory``.

    The returned mapping contains:
    - key: relative POSIX path (``a/b/file.txt``)
    - value: lowercase SHA-256 hex digest
    """

    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"{root!s} is not a directory")

    manifest: dict[str, str] = {}
    for current_dir, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
        dir_names.sort()
        file_names.sort()
        current = Path(current_dir)
        for file_name in file_names:
            file_path = current / file_name
            try:
                mode = file_path.lstat().st_mode
            except FileNotFoundError:
                # If a file disappears during traversal, skip it; callers can rerun for a stable snapshot.
                continue
            if not stat.S_ISREG(mode):
                continue
            rel_path = file_path.relative_to(root).as_posix()
            manifest[rel_path] = sha256_file(file_path)

    return dict(sorted(manifest.items(), key=lambda item: item[0]))


def verify_manifest(
    directory: PathLike,
    manifest: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """
    Verify files under ``directory`` against ``manifest``.

    Returns:
    - missing_paths: expected paths that are absent
    - corrupted_paths: present-but-mismatched paths (or non-regular files)
    """

    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"{root!s} is not a directory")

    missing_paths: list[str] = []
    corrupted_paths: list[str] = []

    for rel_path in sorted(manifest):
        expected_hash = manifest[rel_path]
        _validate_manifest_entry(rel_path, expected_hash)
        target = _manifest_path_to_local(root, rel_path)

        try:
            mode = target.lstat().st_mode
        except FileNotFoundError:
            missing_paths.append(rel_path)
            continue

        if not stat.S_ISREG(mode):
            corrupted_paths.append(rel_path)
            continue

        if sha256_file(target) != expected_hash.lower():
            corrupted_paths.append(rel_path)

    return missing_paths, corrupted_paths


def _validate_manifest_entry(relative_posix_path: str, expected_hash: str) -> None:
    if "\\" in relative_posix_path:
        raise ValueError(f"manifest path must use POSIX separators: {relative_posix_path!r}")
    if len(expected_hash) != _SHA256_HEX_LENGTH or not set(expected_hash).issubset(_HEX_DIGITS):
        raise ValueError(f"invalid SHA-256 hex digest for path {relative_posix_path!r}")


def _manifest_path_to_local(root: Path, relative_posix_path: str) -> Path:
    posix_path = PurePosixPath(relative_posix_path)
    if not relative_posix_path:
        raise ValueError("manifest path cannot be empty")
    if posix_path.is_absolute():
        raise ValueError(f"manifest path must be relative: {relative_posix_path!r}")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError(f"manifest path is not safe: {relative_posix_path!r}")
    return root.joinpath(*posix_path.parts)
