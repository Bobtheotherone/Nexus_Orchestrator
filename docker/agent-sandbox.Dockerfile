# agent-sandbox.Dockerfile — isolated execution environment
#
# File: docker/agent-sandbox.Dockerfile
# Last updated: 2026-02-11
#
# Purpose
# - Define the container image used to run agent tools and verification steps in a sandbox.
#
# What should be included
# - Minimal base image with pinned OS packages.
# - Non-root user, least privilege defaults.
# - Tooling hooks for language runtimes (Python, Node, Rust) as needed.
#
# Security requirements
# - Keep attack surface small (no SSH server, no extra daemons).
# - Prefer explicit allowlists for network egress when policy requires.
#
# agent-sandbox.Dockerfile — base sandbox for agent tool execution
#
# Requirements:
# - Minimal base image (reduce attack surface)
# - Common build tools pre-installed (git, build-essential, curl)
# - Python 3.11+ and Node.js LTS for polyglot projects
# - Non-root user for execution
# - Tool Provisioner extends this at runtime with project-specific tools

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    python3.11 python3-pip python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /workspace

# Tool Provisioner will install additional tools at runtime via:
#   docker exec <container> pip install <package>
#   docker exec <container> npm install <package>
