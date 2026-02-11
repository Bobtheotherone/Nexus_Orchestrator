<!--
nexus-orchestrator — documentation skeleton

File: docs/threat_model.md
Last updated: 2026-02-11

Purpose
- Full threat model for the orchestrator and its sandboxes.

What should be included in this file
- Assets, actors, trust boundaries.
- Abuse cases: prompt injection via repo, malicious dependency installation, data exfiltration, sandbox escape, credential leakage, covert channels via logs.
- Mitigations: capability-based permissions, content hygiene, network egress controls, tool allowlists, provenance logs.

Functional requirements
- Must define enforceable controls (config flags + default policies) rather than vague guidance.

Non-functional requirements
- Must match the implemented sandbox and logging behavior; keep updated as controls change.

Suggested sections / outline
- Scope
- Assets
- Threats
- Mitigations
- Residual risk
- Testing the controls
-->

# Threat Model

## Assets
- Source code (target project and orchestrator itself)
- API credentials for LLM providers
- Evidence records and audit trail

## Threats and Mitigations

| Threat | Vector | Mitigation |
|---|---|---|
| Prompt injection | Malicious content in repo files included as agent context | Content hygiene in context assembler; delimiter strategy; trust tiers |
| Supply chain attack | Agent installs malicious dependency | Tool provisioner: version pinning, checksum verification, vulnerability scan |
| Credential leakage | Secrets appear in prompts, logs, or evidence | Secrets referenced by env var name only; redaction pipeline; secret scanning on diffs |
| Sandbox escape | Agent tool execution breaks out of container | Container isolation; scoped filesystem mounts; network egress controls |
| Exfiltration | Agent sends data out via tool calls or network | Network deny-by-default; logged egress; tool audit log |
| Scope violation | Agent modifies files outside its declared ownership | Git engine scope enforcement; blocked by default |

## Trust Boundaries

1. **Host ↔ Sandbox:** Containers with explicit mounts, no host filesystem access.
2. **Orchestrator ↔ Provider API:** HTTPS only. No secrets in prompts.
3. **Agent ↔ Repository:** All writes go through merge queue. No direct commits.
4. **Trusted content ↔ Untrusted content:** Templates/schemas are trusted. Generated code and user specs are untrusted.
