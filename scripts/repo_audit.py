"""
nexus-orchestrator — repository audit CLI wrapper

File: scripts/repo_audit.py
Last updated: 2026-02-11

Purpose
- Provide a stable, WSL-friendly entrypoint for repository blueprint generation and validation.

Expected CLI usage
- python scripts/repo_audit.py --summary --validate
- python scripts/repo_audit.py --print-phase-map
- python scripts/repo_audit.py --json
- python scripts/repo_audit.py --write-artifacts --validate --fail-on-warn

Functional requirements
- Must run without installation by resolving `src/` on `sys.path`.
- Must delegate to `nexus_orchestrator.repo_blueprint` without changing semantics.

Non-functional requirements
- Keep wrapper minimal and deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _load_main():
    try:
        from nexus_orchestrator.repo_blueprint import main as loaded_main

        return loaded_main
    except ModuleNotFoundError:
        if str(SRC_PATH) not in sys.path:
            sys.path.insert(0, str(SRC_PATH))
        from nexus_orchestrator.repo_blueprint import main as loaded_main

        return loaded_main


if __name__ == "__main__":
    raise SystemExit(_load_main()())
