"""
nexus-orchestrator — script skeleton

File: scripts/cost_report.py
Last updated: 2026-02-11

Purpose
- Generate an API cost report per run/work item/provider from state DB records.

Expected CLI usage
- python scripts/cost_report.py --db state/nexus.sqlite --since 2026-01-01

Functional requirements
- Must attribute costs to work items and phases.
- Must export as CSV/JSON.
- Must support filtering by provider/model.

Non-functional requirements
- Deterministic; no network required.
"""
