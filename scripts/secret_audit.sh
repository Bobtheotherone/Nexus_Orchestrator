#!/usr/bin/env bash
# nexus-orchestrator — deterministic secret-audit helper
#
# File: scripts/secret_audit.sh
# Last updated: 2026-02-13
#
# Purpose
# - Scan repository source/config/docs for likely secret leaks while excluding known
#   test fixtures and cache directories to avoid false-positive panic during audits.
#
# Exit codes
# - 0: no candidate leaks found
# - 1: one or more candidate leaks found
# - 2: tool/runtime failure

set -euo pipefail

repo_root="${1:-.}"
cd "$repo_root"

if ! command -v rg >/dev/null 2>&1; then
  echo "secret-audit: ripgrep (rg) is required" >&2
  exit 2
fi

# Source-focused scan targets. Tests are intentionally excluded by default because
# redaction tests contain synthetic secret-like fixtures.
targets=(
  src
  docs
  scripts
  constraints
  profiles
  orchestrator.toml
  pyproject.toml
  .env.example
)

scan_targets=()
for path in "${targets[@]}"; do
  if [[ -e "$path" ]]; then
    scan_targets+=("$path")
  fi
done

if [[ "${#scan_targets[@]}" -eq 0 ]]; then
  echo "secret-audit: no scan targets found"
  exit 0
fi

# Conservative deterministic signatures for high-confidence leak indicators.
pattern='(sk-[A-Za-z0-9]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|authorization:[[:space:]]*bearer[[:space:]]+[A-Za-z0-9._~+/=-]{8,}|-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----)'

set +e
rg_output="$({
  rg -n -S --hidden \
    --glob '!.git/**' \
    --glob '!tests/**' \
    --glob '!.venv/**' \
    --glob '!.pytest_cache/**' \
    --glob '!.mypy_cache/**' \
    --glob '!.ruff_cache/**' \
    --glob '!.hypothesis/**' \
    --glob '!**/__pycache__/**' \
    "$pattern" "${scan_targets[@]}"
} 2>&1)"
rg_code=$?
set -e

if [[ "$rg_code" -eq 1 ]]; then
  echo "secret-audit: clean (no high-confidence leak patterns outside tests/caches)"
  exit 0
fi

if [[ "$rg_code" -eq 0 ]]; then
  echo "secret-audit: potential leaks detected (tests/** intentionally excluded):"
  printf '%s\n' "$rg_output"
  exit 1
fi

echo "secret-audit: scan failed"
printf '%s\n' "$rg_output" >&2
exit 2
