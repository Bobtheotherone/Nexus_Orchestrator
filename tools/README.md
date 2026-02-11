<!--
nexus-orchestrator — documentation skeleton

File: tools/README.md
Last updated: 2026-02-11

Purpose
- Defines the Tool Registry concept: what tools are allowed, how versions are pinned, and how provenance is recorded.

What should be included in this file
- Registry format (TOML/YAML) and fields: name, version, source, checksum, license, risk profile.
- Workflow: agent requests tool -> provisioner checks -> install -> evidence record.

Functional requirements
- Must be used by sandbox/tool_provisioner.py.

Non-functional requirements
- Default deny for unknown tools in strict mode; permissive mode still logs.

Suggested sections / outline
- Registry format
- Approval workflow
- Provenance
- Policies
-->

# Tool Registry

Approved tools for use in agent sandboxes.

## Registry Format
See `tools/registry.toml` for the allowlist.

## Workflow
1. Agent requests a tool (via `synthesis_plane/tools.py`)
2. Tool Provisioner checks `registry.toml` for approved version
3. If known + approved: install in sandbox
4. If unknown: run vulnerability scan, pin version, record in registry, install
5. All installs recorded in state DB for evidence/audit

## Policies
- **Strict mode:** Only pre-approved tools allowed
- **Permissive mode:** Unknown tools allowed after automated scan + logging
