<!--
nexus-orchestrator — documentation skeleton

File: docker/README.md
Last updated: 2026-02-11

Purpose
- Documents container images used for sandboxes and tool execution.

What should be included in this file
- Base images strategy (minimal, pinned).
- How to build/update sandbox images.
- How to control network egress and mounts.

Functional requirements
- Must define at least one sandbox image that can run build/test tools in isolation.

Non-functional requirements
- Images must be reproducible and minimized; avoid unnecessary packages.

Suggested sections / outline
- Images
- Build
- Security controls
- Updating
-->

# Docker — Sandbox Container Images

## Purpose
Agents execute tools (build, test, lint, security scan) inside sandboxed containers.

## Images
- `agent-sandbox` — Base sandbox with common build tools. Extended by the Tool Provisioner on demand.

## Build
```bash
docker build -t nexus-agent-sandbox -f docker/agent-sandbox.Dockerfile .
```

## Security Controls
- Filesystem: workspace directory mounted RW, dependency caches mounted RO, host filesystem inaccessible
- Network: deny-by-default (configurable to allowlist or logged-permissive)
- Resources: CPU and memory limits set per sandbox instance
- No privileged mode, no host PID/network namespace
