"""Module entrypoint for ``python -m nexus_orchestrator``."""

from __future__ import annotations

from nexus_orchestrator.main import cli_entrypoint

if __name__ == "__main__":
    raise SystemExit(cli_entrypoint())
