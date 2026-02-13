"""
nexus-orchestrator — verification plane evidence IO integration tests

File: tests/integration/test_verification_plane_evidence_io.py
Last updated: 2026-02-13

Purpose
- Validate append-only evidence layout, deterministic manifest hashing, and on-disk redaction.

What this test file should cover
- Stable directory structure and file set (including result/metadata/manifest artifacts).
- Manifest SHA-256 integrity checks over persisted artifacts.
- Redaction guarantees for logs/result/metadata payloads written to disk.

Functional requirements
- Offline deterministic execution against temporary local directories only.

Non-functional requirements
- Fast and repeatable for CI/WSL.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from nexus_orchestrator.security.redaction import REDACTED_VALUE
from nexus_orchestrator.utils.hashing import sha256_file
from nexus_orchestrator.verification_plane.evidence import (
    EvidenceWriter,
    export_audit_bundle,
    verify_integrity,
)


@pytest.mark.integration
def test_evidence_writer_layout_hashing_and_redaction(tmp_path: Path) -> None:
    writer = EvidenceWriter(tmp_path / "evidence")
    secret_text = "sk-EXAMPLESECRETVALUE1234567890"

    write_result = writer.write_evidence(
        run_id="run-100",
        work_item_id="wi-100",
        attempt_id="attempt-7",
        stage="build",
        evidence_id="evi-100",
        result={"status": "pass", "token": secret_text},
        metadata={"note": f"TOKEN={secret_text}"},
        logs={"build.log": f"PASSWORD={secret_text}"},
        artifacts={"summary.json": {"leak": secret_text, "ok": True}},
    )

    expected_dir = writer.evidence_dir_for(
        run_id="run-100",
        work_item_id="wi-100",
        attempt_id="attempt-7",
        stage="build",
        evidence_id="evi-100",
    )
    assert write_result.evidence_dir == expected_dir
    assert (expected_dir / "result.json").is_file()
    assert (expected_dir / "metadata.json").is_file()
    assert (expected_dir / "manifest.json").is_file()
    assert (expected_dir / "logs" / "build.log").is_file()
    assert (expected_dir / "artifacts" / "summary.json").is_file()

    for file_path in (
        expected_dir / "result.json",
        expected_dir / "metadata.json",
        expected_dir / "logs" / "build.log",
        expected_dir / "artifacts" / "summary.json",
    ):
        text = file_path.read_text(encoding="utf-8")
        assert secret_text not in text
        assert REDACTED_VALUE in text

    manifest_payload = json.loads((expected_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest_payload["entries"]
    paths = [entry["path"] for entry in entries]
    assert paths == sorted(paths)
    assert "result.json" in paths
    assert "metadata.json" in paths
    assert "logs/build.log" in paths
    assert "artifacts/summary.json" in paths

    for entry in entries:
        target = expected_dir / Path(*str(entry["path"]).split("/"))
        assert target.is_file()
        assert sha256_file(target) == entry["sha256"]
        assert target.stat().st_size == entry["size_bytes"]

    integrity = verify_integrity(expected_dir)
    assert integrity.is_valid


@pytest.mark.integration
def test_evidence_writer_append_only_and_audit_bundle_export(tmp_path: Path) -> None:
    writer = EvidenceWriter(tmp_path / "evidence")
    write_result = writer.write_evidence(
        run_id="run-200",
        work_item_id="wi-200",
        stage="test",
        evidence_id="evi-200",
        result={"status": "pass"},
        artifacts={"report.json": {"ok": True}},
    )

    with pytest.raises(FileExistsError):
        writer.write_evidence(
            run_id="run-200",
            work_item_id="wi-200",
            stage="test",
            evidence_id="evi-200",
            result={"status": "pass"},
        )

    bundle_path = export_audit_bundle(write_result.evidence_dir, output_path=tmp_path / "audit.zip")
    assert bundle_path.is_file()

    with zipfile.ZipFile(bundle_path, "r") as archive:
        names = sorted(archive.namelist())
    prefix = write_result.evidence_dir.name
    assert f"{prefix}/manifest.json" in names
    assert f"{prefix}/metadata.json" in names
    assert f"{prefix}/result.json" in names
