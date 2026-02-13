"""Deterministic tests for the repository knowledge indexer."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from nexus_orchestrator.knowledge_plane.indexer import INDEX_SCHEMA_VERSION, RepositoryIndexer

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_python_syntax_error_falls_back_to_generic_text_adapter(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(
        tmp_path / "pkg" / "broken.py",
        "import os\nfrom pkg import util\n\ndef bad(:\n    return 1\n",
    )

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.reindex()

    broken = indexer.files["pkg/broken.py"]
    assert broken.language == "generic_text"
    assert broken.parse_error is not None
    assert "SyntaxError" in broken.parse_error
    assert "os" in broken.imports
    assert "pkg" in broken.imports
    assert indexer.resolve_module("pkg.broken") == ("pkg/broken.py",)


def test_import_and_symbol_extraction_and_module_lookup(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(
        tmp_path / "pkg" / "models.py",
        "class User:\n    pass\n",
    )
    _write(
        tmp_path / "pkg" / "service.py",
        "import os\n"
        "from pkg.models import User\n"
        "from .models import User as UserAlias\n\n"
        "CONFIG = 'x'\n\n"
        "def build_user() -> User:\n"
        "    return User()\n",
    )

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.reindex()

    service = indexer.files["pkg/service.py"]
    service_imports = set(service.imports)
    assert {"os", "pkg.models", "pkg.models.User"}.issubset(service_imports)
    assert {"CONFIG", "build_user", "UserAlias"}.issubset(set(service.symbols))

    assert indexer.resolve_module("pkg.service") == ("pkg/service.py",)
    assert "pkg/models.py" in indexer.lookup_symbol("User")
    assert "pkg/service.py" in indexer.reverse_dependencies("pkg.models")


def test_module_resolution_and_reverse_dependencies(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "a.py", "from pkg.b import Service\n")
    _write(tmp_path / "pkg" / "b.py", "class Service:\n    pass\n")
    _write(tmp_path / "consumer.py", "import pkg.a\n")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.reindex()

    assert indexer.resolve_module("pkg.a") == ("pkg/a.py",)
    assert indexer.reverse_dependencies("pkg.a") == ("consumer.py",)
    assert indexer.reverse_dependencies("pkg.b") == ("pkg/a.py",)
    assert indexer.files_for_module("pkg.a") == ("consumer.py", "pkg/a.py")


def test_incremental_update_add_modify_delete_keeps_indexes_consistent(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "base.py", "class Base:\n    pass\n")
    _write(tmp_path / "pkg" / "feature.py", "from pkg.base import Base\n")
    _write(tmp_path / "old_consumer.py", "import pkg.feature\n")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.reindex()

    _write(tmp_path / "pkg" / "base.py", "class Base2:\n    pass\n")
    _write(tmp_path / "pkg" / "new_consumer.py", "import pkg.base\n")
    (tmp_path / "old_consumer.py").unlink()

    indexer.update_paths(
        [
            tmp_path / "pkg" / "base.py",
            "pkg/new_consumer.py",
            tmp_path / "old_consumer.py",
        ]
    )

    assert "old_consumer.py" not in indexer.files
    assert indexer.resolve_module("pkg.new_consumer") == ("pkg/new_consumer.py",)

    base_entry = indexer.files["pkg/base.py"]
    assert "Base2" in base_entry.symbols
    assert "Base" not in base_entry.symbols

    assert indexer.reverse_dependencies("pkg.feature") == ()
    assert set(indexer.reverse_dependencies("pkg.base")) == {
        "pkg/feature.py",
        "pkg/new_consumer.py",
    }


def test_save_load_roundtrip_is_deterministic_and_query_equivalent(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "a.py", "from pkg.b import Service\n")
    _write(tmp_path / "pkg" / "b.py", "class Service:\n    pass\n")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.reindex()

    (tmp_path / ".index").mkdir(parents=True, exist_ok=True)
    saved_path = indexer.save(".index/index.json")
    saved_raw = saved_path.read_text(encoding="utf-8")

    loaded = RepositoryIndexer.load(repo_root=tmp_path, source=".index/index.json")
    saved_path_2 = loaded.save(".index/index-2.json")
    saved_raw_2 = saved_path_2.read_text(encoding="utf-8")

    assert loaded.files == indexer.files
    assert loaded.resolve_module("pkg.a") == indexer.resolve_module("pkg.a")
    assert loaded.reverse_dependencies("pkg.b") == indexer.reverse_dependencies("pkg.b")

    payload = json.loads(saved_raw)
    assert payload["schema_version"] == INDEX_SCHEMA_VERSION
    assert saved_raw == saved_raw_2


def test_incremental_update_rejects_paths_outside_repo_root(tmp_path: Path) -> None:
    _write(tmp_path / "in_repo.py", "x = 1\n")
    outside_path = tmp_path.parent / "outside.py"
    outside_path.write_text("y = 2\n", encoding="utf-8")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.reindex()

    with pytest.raises(ValueError, match="outside repository root"):
        indexer.update_paths([outside_path])


def test_build_none_performs_full_reindex_and_returns_self(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "service.py", "from pkg.models import User\n")
    _write(tmp_path / "pkg" / "models.py", "class User:\n    pass\n")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    built = indexer.build(changed_paths=None)

    assert built is indexer
    assert set(indexer.files) == {"pkg/__init__.py", "pkg/models.py", "pkg/service.py"}
    assert indexer.resolve_module("pkg.service") == ("pkg/service.py",)
    assert indexer.reverse_dependencies("pkg.models") == ("pkg/service.py",)


def test_build_incremental_update_accepts_path_and_str(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "base.py", "class Base:\n    pass\n")
    _write(tmp_path / "pkg" / "feature.py", "from pkg.base import Base\n")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.build(changed_paths=None)

    _write(tmp_path / "pkg" / "base.py", "class RenamedBase:\n    pass\n")
    _write(tmp_path / "pkg" / "new_consumer.py", "import pkg.base\n")

    built = indexer.build(
        changed_paths=(
            tmp_path / "pkg" / "base.py",
            "pkg/new_consumer.py",
        )
    )

    assert built is indexer
    assert indexer.resolve_module("pkg.new_consumer") == ("pkg/new_consumer.py",)
    base_record = indexer.files["pkg/base.py"]
    assert "RenamedBase" in base_record.symbols
    assert "Base" not in base_record.symbols
    assert set(indexer.reverse_dependencies("pkg.base")) == {
        "pkg/feature.py",
        "pkg/new_consumer.py",
    }


def test_build_incremental_ignores_paths_outside_repo_root(tmp_path: Path) -> None:
    _write(tmp_path / "in_repo.py", "x = 1\n")
    outside_path = tmp_path.parent / "outside.py"
    outside_path.write_text("y = 2\n", encoding="utf-8")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.build(changed_paths=None)

    _write(tmp_path / "in_repo.py", "x = 2\n")
    indexer.build(changed_paths=[outside_path, tmp_path / "in_repo.py"])

    assert "in_repo.py" in indexer.files
    assert (
        indexer.files["in_repo.py"].content_hash
        == RepositoryIndexer(repo_root=tmp_path)
        .build(changed_paths=None)
        .files["in_repo.py"]
        .content_hash
    )


def test_build_is_deterministic_for_unchanged_snapshot(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "__init__.py", "")
    _write(tmp_path / "pkg" / "a.py", "from pkg.b import Service\n")
    _write(tmp_path / "pkg" / "b.py", "class Service:\n    pass\n")

    indexer = RepositoryIndexer(repo_root=tmp_path)
    indexer.build(changed_paths=None)
    first_files = indexer.files
    first_module_map = dict(indexer._module_to_files)
    first_reverse_deps = dict(indexer._reverse_deps)

    indexer.build(changed_paths=None)

    assert indexer.files == first_files
    assert indexer._module_to_files == first_module_map
    assert indexer._reverse_deps == first_reverse_deps
