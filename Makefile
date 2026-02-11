# Makefile — convenience commands for local development (optional)
#
# File: Makefile
# Last updated: 2026-02-11
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

.PHONY: help install lint typecheck test test-unit test-integration test-smoke run plan status clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (dev)
	pip install -e ".[dev,providers]" --break-system-packages

lint: ## Run linter and formatter checks
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck: ## Run type checker
	mypy src/nexus_orchestrator/

test: test-unit test-integration ## Run all tests

test-unit: ## Run unit tests
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-smoke: ## Run smoke tests (end-to-end with mocks)
	pytest tests/smoke/ -v

plan: ## Run planning on sample spec
	python -m nexus_orchestrator plan samples/specs/minimal_design_doc.md

run: ## Run orchestration with mock providers
	python -m nexus_orchestrator run --mock

status: ## Show current run status
	python -m nexus_orchestrator status

clean: ## Remove ephemeral state, evidence, and workspaces
	rm -rf state/ workspaces/ evidence/ artifacts/
	@echo "Cleaned ephemeral directories."
