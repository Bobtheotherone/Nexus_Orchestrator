# Makefile — convenience commands for local development (optional)
#
# File: Makefile
# Last updated: 2026-02-12
#
# Purpose
# - Provide ergonomic shortcuts for the operator (you) and for agentic AIs running common tasks.
#
# What should be included
# - install, lint, typecheck, test (unit/integration/smoke), fmt, security-scan, clean, run, plan, status.
# - Targets should call underlying Python tooling, not contain complex logic.
#
# Non-functional requirements
# - Must be safe to run repeatedly (idempotent).
# - Must not require network access except where explicitly documented (e.g., vuln DB updates).
#
# Makefile — common dev tasks
# Run `make help` to see available targets.

PYTHON ?= python

.PHONY: help install lint typecheck test test-unit test-integration test-smoke security-scan placeholder-gate secret-audit audit run plan status clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (dev)
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run linter and formatter checks
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

typecheck: ## Run type checker
	$(PYTHON) -m mypy src/nexus_orchestrator/

test: ## Run all tests (meta + unit + integration + smoke)
	$(PYTHON) -m pytest tests/meta tests/unit tests/integration tests/smoke -v

test-unit: ## Run unit tests
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v

test-smoke: ## Run smoke tests (end-to-end with mocks)
	$(PYTHON) -m pytest tests/smoke/ -v

security-scan: ## Vulnerability scan of the current environment (best effort offline if cache exists)
	mkdir -p .security-reports
	$(PYTHON) -m pip freeze --all --exclude-editable > .security-reports/frozen-requirements.txt
	$(PYTHON) -m pip_audit --strict --no-deps --progress-spinner off --format json --output .security-reports/pip-audit.json -r .security-reports/frozen-requirements.txt

placeholder-gate: ## Run canonical placeholder gate (WSL-safe)
	$(PYTHON) scripts/placeholder_gate.py

secret-audit: ## Scan for likely secret leaks outside tests/caches (WSL-safe)
	bash scripts/secret_audit.sh

audit: lint typecheck test placeholder-gate secret-audit security-scan ## Run CI-style quality and security gates

plan: ## Run planning on sample spec
	$(PYTHON) -m nexus_orchestrator plan samples/specs/minimal_design_doc.md

run: ## Run orchestration with mock providers
	$(PYTHON) -m nexus_orchestrator run --mock

status: ## Show current run status
	$(PYTHON) -m nexus_orchestrator status

clean: ## Remove ephemeral state, evidence, and workspaces
	rm -rf state/ workspaces/ evidence/ artifacts/
	@echo "Cleaned ephemeral directories."
