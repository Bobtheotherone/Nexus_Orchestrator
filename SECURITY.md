<!--
nexus-orchestrator — documentation skeleton

File: SECURITY.md
Last updated: 2026-02-11

Purpose
- Security policy and threat model summary for the orchestrator itself (not the projects it builds).

What should be included in this file
- Threat model summary (prompt injection, supply chain, secret leakage, sandbox escapes, malicious diffs).
- Security boundaries: sandbox vs host, network egress policy, secrets handling.
- Vulnerability reporting policy (even if private-only).
- Secure development requirements: dependency scanning, pinned tools, reproducible builds.

Functional requirements
- Must document where secrets are allowed to exist (ideally nowhere in repo; use external secret manager).
- Must define minimum security checks in the constraint gate for this repo (secret scanning, SCA).

Non-functional requirements
- Must remain aligned with docs/threat_model.md and actual implementation.
- Must avoid giving actionable instructions that weaken security without explicit warnings.

Suggested sections / outline
- Security boundaries
- Threat model
- Secrets policy
- Supply chain policy
- Reporting
-->

# Security Policy

## Security Boundaries

- **Sandbox ↔ Host:** Agent tools run in containers with scoped filesystem mounts and controlled network.
- **Prompts ↔ Secrets:** Secrets are never inlined in prompts. Referenced by env var name only.
- **Agent ↔ Repository:** Agents cannot commit directly to integration or main. All changes go through the merge queue.

## Threat Model

See `docs/threat_model.md` for the full model. Key threats:
- Prompt injection via repository content
- Supply chain attacks through agent-installed dependencies
- Credential leakage into logs or evidence
- Sandbox escape

## Secrets Policy

- No secrets in source files, config files, prompts, or log output
- Secrets referenced via env var names (e.g., `NEXUS_OPENAI_API_KEY`)
- Secret scanning runs on every diff before merge
- Evidence artifacts are redacted before storage

## Reporting

This is a private tool. If you discover a vulnerability, fix it directly and record it as an incident in the state DB.
