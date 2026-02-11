<!--
nexus-orchestrator — documentation skeleton

File: docs/quality/STYLE_AND_LINT.md
Last updated: 2026-02-11

Purpose
- Defines formatting/linting/typecheck standards for the orchestrator repo itself.

What should be included in this file
- Chosen tools and their versions (once selected).
- Rules that reduce merge conflicts (format on save, import ordering).
- How style constraints are enforced by CI and local gate.

Functional requirements
- Must map to constraint records in constraints/registry/.

Non-functional requirements
- Minimize churn; prefer automated formatters.

Suggested sections / outline
- Tools
- Rules
- CI enforcement
- How to fix
-->

# Style and Lint Standards

## Tools (once installed)

| Tool | Purpose | Config |
|---|---|---|
| `ruff` | Linter + formatter (replaces flake8, isort, black) | `pyproject.toml [tool.ruff]` |
| `mypy` | Static type checker (strict mode) | `pyproject.toml [tool.mypy]` |
| `pytest` | Test runner | `pyproject.toml [tool.pytest]` |

## Rules to Minimize Merge Conflicts

- Format on save (ruff format). All agents produce identically formatted code.
- Deterministic import ordering (ruff I rules).
- No trailing whitespace (.editorconfig enforced).
- LF line endings only.

## Constraint Mapping

Style rules are enforced as `must`-severity constraints:
- `CON-STY-0001`: Code passes `ruff check` with zero errors
- `CON-STY-0002`: Code passes `ruff format --check`
- `CON-STY-0003`: Code passes `mypy --strict`
