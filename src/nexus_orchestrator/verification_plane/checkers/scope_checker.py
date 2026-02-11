"""
nexus-orchestrator — module skeleton

File: src/nexus_orchestrator/verification_plane/checkers/scope_checker.py
Last updated: 2026-02-11

Purpose
- Enforce work-item scope boundaries (file/path ownership) before a change is considered merge-eligible.

What should be included in this file
- A ScopeChecker implementing the BaseChecker interface and producing deterministic CheckResult artifacts.
- Logic to compute the *effective changed files* for a candidate patch (git diff) and compare against allowed scope.
- Support for scoped exceptions via an explicit, audited 'scope extension request' mechanism (never silent).
- Integration points with WorkspaceManager / CodeOwnershipMap if used.

Functional requirements
- Must fail if any modified/added/deleted file is outside the work item’s declared scope allowlist.
- Must support glob/path-prefix rules and explicit file lists.
- Must produce a machine-readable evidence artifact listing out-of-scope paths and the scope rule violated.
- Must support an explicit override flow that requires justification + operator approval, recorded in evidence ledger.

Non-functional requirements
- Deterministic: same inputs → same results.
- Fast: should run in < 1s for typical diffs; avoid expensive repo scans.
- Secure: treat symlinks/path traversal defensively when interpreting file lists.

Notes
- This checker is referenced by baseline constraint CON-ARC-0001 in constraints/registry/000_base_constraints.yaml.
"""
