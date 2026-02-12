"""
nexus-orchestrator — filesystem utilities

File: src/nexus_orchestrator/utils/fs.py
Last updated: 2026-02-12

Purpose
- Provide safe, minimal filesystem helpers for atomic writes and guarded deletion.

Functional requirements
- Atomic writes use temp files in the destination directory and replace in a single step.
- Deletion refuses paths outside the configured workspace root.

Non-functional requirements
- Standard library only and cross-platform behavior where feasible.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

PathLike = str | os.PathLike[str]

__all__ = [
    "atomic_write",
    "is_within",
    "safe_delete",
    "temp_directory",
]


def atomic_write(path: PathLike, data: bytes | str, *, encoding: str = "utf-8") -> None:
    """
    Atomically write ``data`` to ``path``.

    The write strategy is:
    1. create temp file in the same directory,
    2. write + flush + fsync file data,
    3. replace target via ``os.replace``.
    """

    target = Path(path)
    target_parent = target.parent.resolve(strict=True)
    if not target_parent.is_dir():
        raise NotADirectoryError(f"{target_parent!s} is not a directory")

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target_parent),
    )
    temp_path = Path(temp_name)

    try:
        if isinstance(data, bytes):
            with os.fdopen(fd, "wb") as file_handle:
                file_handle.write(data)
                file_handle.flush()
                os.fsync(file_handle.fileno())
        else:
            with os.fdopen(fd, "w", encoding=encoding) as file_handle:
                file_handle.write(data)
                file_handle.flush()
                os.fsync(file_handle.fileno())

        os.replace(temp_path, target)
        _fsync_directory(target_parent)
    except Exception:
        with contextlib.suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise


def is_within(child: PathLike, parent: PathLike) -> bool:
    """Return ``True`` if resolved ``child`` is within resolved ``parent``."""

    try:
        resolved_parent = Path(parent).resolve(strict=True)
    except FileNotFoundError:
        return False
    if not resolved_parent.is_dir():
        return False

    try:
        resolved_child = Path(child).resolve(strict=True)
    except FileNotFoundError:
        return False

    return _is_relative_to(resolved_child, resolved_parent)


def safe_delete(path: PathLike, workspace_root: PathLike) -> None:
    """
    Delete ``path`` only if it is contained within ``workspace_root``.

    Symlinks are unlinked without traversing into their targets.
    """

    workspace = Path(workspace_root).resolve(strict=True)
    if not workspace.is_dir():
        raise NotADirectoryError(f"{workspace!s} is not a directory")

    target = Path(path)
    parent_resolved = target.parent.resolve(strict=True)
    candidate = parent_resolved / target.name
    if not _is_relative_to(candidate, workspace):
        raise ValueError(f"refusing to delete path outside workspace root: {target!s}")

    if target.is_symlink():
        target.unlink()
        return

    resolved_target = target.resolve(strict=True)
    if not _is_relative_to(resolved_target, workspace):
        raise ValueError(f"refusing to delete path outside workspace root: {target!s}")

    if target.is_dir():
        shutil.rmtree(target)
        return

    target.unlink()


@contextmanager
def temp_directory(prefix: str = "nexus-") -> Iterator[Path]:
    """Yield a temporary directory path and clean it up on exit."""

    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield Path(tmp)


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    """
    Best-effort directory fsync for metadata durability after ``os.replace``.

    Some platforms/filesystems do not support fsync on directories.
    """

    if os.name == "nt":
        return

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    try:
        dir_fd = os.open(path, flags)
    except OSError:
        return

    try:
        os.fsync(dir_fd)
    except OSError:
        return
    finally:
        os.close(dir_fd)
