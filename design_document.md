<!--
nexus-orchestrator — primary design document

File: design_document.md
Last updated: 2026-02-11

Purpose
- Canonical architecture spec for the orchestrator. This is the 'single design document' that other agents will ingest.

What should be included in this file
- The unified NEXUS design: goals/non-goals, planes/components, data model (Artifact/Constraint/Evidence/WorkItem), Git coordination protocol, verification pipeline, sandboxing/security, resource governance, observability, operating modes, and roadmap.
- Concrete definitions of 'perfect', 'constraint envelope', 'evidence ledger', 'merge queue', and 'constraint evolution'.
- A stable requirement ledger section with unique IDs (REQ-0001...) and acceptance criteria that will feed the Spec Ingestor.
- A glossary and cross-links to schemas in docs/schemas/.

Functional requirements
- Must be parseable into a structured Spec Map (requirements, NFRs, entities, interfaces, acceptance criteria).
- Must describe interfaces/contracts at a level sufficient to stabilize module boundaries early.
- Must explicitly define the constraint gate policy and the exception/override protocol.

Non-functional requirements
- Must remain the single source of truth; other docs elaborate but do not contradict.
- Must be updated via ADRs when major architectural choices change.

NOTE
- This repo skeleton ships with an abbreviated version of the agreed design text. Replace or expand this file by pasting the full agreed design document(s) from your prompt if desired. Keep the requirement IDs stable once established.
-->

# Design Document — Constraint-Driven Agentic LLM Orchestrator

## 1. Purpose and Scope

This document specifies the architecture for a personal, local-first orchestrator that takes a single design document and produces a high-quality, modular, production-grade codebase by coordinating hundreds of parallel LLM agents (Codex-class code generators and Claude-class reasoning agents) while preventing conflicts through Git-native isolation and merge discipline.

The orchestrator's core mechanism is Constraint-Based Program Synthesis (CBPS): every artifact an agent proposes — code, configs, docs, tests, migrations, CI definitions — must be accompanied by machine-checkable evidence that it satisfies an evolving set of constraints derived from the specification. "Perfect" is defined operationally: no merge is allowed without satisfying the full constraint set and providing sufficient evidence. There is no partial credit.

This is a personal tool for a single operator on a single workstation. Every design decision prioritizes directness, power, and minimal ceremony over distribution, multi-tenancy, or organizational governance.

---

## 2. Hard Constraints

### 2.1 Hardware Envelope

The orchestrator runs on one machine:

- GPU: NVIDIA RTX 5090, 24 GB VRAM
- RAM: 32 GB DDR5
- CPU: 16 cores at 2.9 GHz
- Disk: 1 TB SSD

This is a hard ceiling. The system must never exceed it, and must degrade gracefully as it approaches limits.

### 2.2 Implications for Architecture

Hundreds of "agents" are remote LLM API calls, not local processes. The parallelism is network-bound and cost-bound, not compute-bound. Local compute is reserved exclusively for: repository indexing and context retrieval, deterministic verification (compilation, testing, static analysis, security scanning), Git operations (branching, merging, conflict detection), orchestrator state management and scheduling, and optional lightweight local models (embeddings, small classifiers) only if they fit within the remaining budget after all verification workloads are accounted for.

The system must enforce strict concurrency limits for CPU-intensive and RAM-intensive verification jobs and must always preserve headroom to prevent thrashing.

---

## 3. Design Principles

**Constraints and Evidence Over Assurances.** If a claim cannot be verified by a machine-checkable process, it is not accepted. Agents must produce evidence, not promises. Every merged artifact carries a structured proof record.

**Maximum Agent Resourcefulness Within Sandboxed Boundaries.** Agents are treated as capable, creative problem-solvers. They can install tools, fetch documentation, restructure plans, build custom generators, try unconventional approaches, and do whatever it takes to satisfy their constraints. They cannot bypass security controls, disable checks, hide changes, introduce obfuscated logic, or access resources outside their sandbox. The constraint gate is the sole arbiter of correctness; the path to satisfaction is the agent's business.

**Parallel by Default, Deterministic by Merge.** Exploration and implementation are massively parallel. Integration is serialized through a merge queue with full constraint verification. This is how hundreds of agents work simultaneously without stepping on each other.

**Evolve the Constraint Set.** Every failure, flake, regression, or near-miss becomes a new constraint, checker, test, or architectural rule. The system gets stricter over time, never laxer. Classes of bugs that are caught once become impossible to reintroduce.

**Modularity as an Axiomatic Property.** Every module must be independently compilable, independently testable, and independently replaceable. This is not a best practice — it is a hard constraint enforced at decomposition time and verified at merge time. If two modules cannot be developed independently, they are the same module.

**Reproducibility and Auditability.** Every output is reproducible (inputs recorded, tools versioned, environments captured). Every decision is traceable (which agent, which constraints, which evidence, how many iterations, what feedback).

---

## 4. Core Data Model

### 4.1 Artifact

Any deliverable that can be committed or referenced: source code, configuration, documentation, tests, fixtures, build and CI definitions, benchmarks, schema migrations, and generated assets with provenance.

### 4.2 Constraint

A machine-checkable rule that must hold for an artifact to be accepted. Constraints are first-class, versioned entities maintained in a Constraint Registry alongside the code.

Constraint categories: structural constraints (module boundaries, dependency direction, naming, layering), behavioral constraints (input-output specifications, state machine transitions, protocol compliance), interface and contract constraints (type signatures, API surface, error conditions), performance constraints (latency bounds, memory budgets, algorithmic complexity limits), security constraints (no unsafe deserialization, no known-vulnerable dependencies, secrets handling), operational constraints (structured logging, error handling, timeout and retry policies), documentation constraints (public API docs, runnable examples, module READMEs), and style constraints (formatting, lint rules, idiom conformance).

Every constraint has a severity level: must (blocks merge), should (generates a warning, requires explicit override with justification), or may (advisory, tracked but not blocking).

### 4.3 Evidence

A structured record proving that constraints are satisfied. Evidence includes: test results with coverage metrics, lint and type-check outputs, build logs with tool versions and hashes, security scan reports, benchmark results against thresholds, and reproduction steps with environment capture. No evidence, no merge.

### 4.4 Work Item

The smallest independently-executable unit of work. Defined by: its scope (which files and modules it owns), its preconditions and dependencies on other work items, its constraint envelope (the full set of constraints it must satisfy), its expected artifacts and required evidence, its budget (maximum API tokens, maximum iterations, maximum wall-clock time), and its risk level (which determines whether adversarial review is required).

### 4.5 Task Graph

A directed acyclic graph of work items. Nodes are work items; edges are dependency and constraint-propagation relationships. The orchestrator schedules this graph to maximize parallelism while respecting dependency ordering, resource budgets, and merge safety.

---

## 5. System Architecture

### 5.1 Architectural Layers

The orchestrator is organized into five layers.

**Control Plane.** Task planning, scheduling, dispatch, budgets, state management, and the adaptive feedback loop. This is the brain of the system.

**Synthesis Plane.** The swarm of remote LLM agents generating patches, tests, docs, refactors, and tool integrations. All inference is remote; the local machine only assembles prompts and collects results.

**Verification Plane.** Deterministic local checkers that produce evidence: builds, tests, static analysis, type checks, security scans, benchmarks, and compositional integration checks. This is where local compute is spent.

**Integration Plane.** Git branch isolation, merge queue, conflict prevention, and change control. The single serialization point that keeps the repository coherent.

**Knowledge Plane.** Codebase index, constraint registry, evidence ledger, decision records, learned failure patterns, and the context retrieval engine that feeds agents the information they need.

### 5.2 Component Catalog

**Spec Ingestor.** Takes the input design document and extracts: requirements with unique identifiers, non-functional constraints (performance, security, reliability), domain entities and their relationships, integration points, and acceptance criteria. Produces a normalized Spec Map that all downstream components consume.

**Architect Agent.** The first agent invoked on any project. Receives the Spec Map and produces the initial decomposition: the module graph, interface contracts, constraint envelopes, dependency ordering, and an Architecture Decision Record (ADR) for every significant structural choice. Always a Claude instance at maximum context, because the decomposition task requires holistic understanding of the entire project. Also invoked during re-planning cycles when integration failures or constraint contradictions require structural changes.

**Constraint Compiler.** A deterministic, non-AI component that validates the decomposition. Checks that the task graph is a valid DAG, that every work item has a complete constraint envelope, that every interface is referenced by at least one producer and one consumer, that constraint envelopes are internally consistent, and that the decomposition covers the entire Spec Map with no gaps. Also propagates constraints through the dependency graph: if interface I guarantees non-null returns, every module consuming I inherits that guarantee, and the constraint gate verifies consistency.

**Constraint Registry.** The persistent, versioned store of all constraints. Each entry records: constraint ID, description, severity, checker type, associated requirements, creation source (spec-derived, failure-derived, or manually added), and history of failures and fixes. Constraints are reviewed and evolved like code.

**Planner and Scheduler.** Takes the validated task graph and produces a dispatch stream using a modified critical-path algorithm. Work items on the longest dependency chain are prioritized. Within a priority tier, items are ordered by estimated complexity so that the hardest problems start first and have the most time for iteration. The scheduler is adaptive: it re-plans when items complete faster or slower than expected, increases parallelism on low-conflict areas, throttles high-conflict zones, and escalates to stronger models for items with repeated failures.

**Dispatch Controller.** Manages the pool of concurrent API sessions to Codex and Claude. Handles rate limiting, exponential backoff, load balancing, and provider routing. Simple, well-constrained coding tasks go to Codex (faster, cheaper). Complex architectural reasoning, tasks requiring long context, tasks that have failed with Codex, and tasks requiring nuanced judgment go to Claude. The routing is adaptive: if a work item type consistently fails with one provider and succeeds with the other, the router updates its preferences. The controller also manages per-work-item token and cost budgets, detecting and terminating runaway loops.

**Context Assembler.** A critical component that builds the prompt for each agent. Agents need to see the interface contracts they must implement, the contracts of modules they depend on, relevant type definitions, architectural patterns established elsewhere in the codebase, and sometimes examples from already-completed modules. The Context Assembler selects this information dynamically within token limits using dependency analysis (include everything the work item directly depends on), semantic similarity from the codebase index (include modules solving analogous problems), and recency (include the latest versions of evolving interfaces). Poor context assembly produces poor agent output; this component is as important as the agents themselves.

**Agent Runtime and Role System.** Agents are not just "LLM calls" — they operate under role profiles that define their system prompt, available tools, budget, and success criteria. Core roles: Architect Agent (module boundaries, interfaces, ADRs), Implementer Agent (writes code within strict scope), Test Engineer Agent (unit, integration, and property-based tests), Reviewer Agent (adversarial review, edge cases, regression hunting), Security Agent (threat surface, secure defaults, dependency scrutiny), Performance Agent (benchmarks, profiling, hot-path analysis), Toolsmith Agent (integrates new tools, updates CI), Integrator Agent (resolves merge conflicts, refactors for coherence), Constraint Miner Agent (turns failures into new constraints), and Documentation Agent (API docs, guides, runnable examples). The role system is extensible; new roles can be defined as project needs evolve.

**Workspace Manager.** Creates ephemeral, isolated workspaces for each agent and work item: a fresh Git branch, scoped file boundaries, read-only dependency cache mounts. Enforces that agents cannot modify files outside their declared scope and cannot commit directly to the integration branch. Workspaces are garbage-collected aggressively after work items complete.

**Tool Provisioner.** Ensures agents have access to whatever tools they need. Maintains a Tool Registry of known tools with versions and security posture. When an agent requests a new tool, the provisioner pins it to a specific version, runs a vulnerability scan, and installs it into the agent's sandbox. All tool versions used to produce evidence are recorded for reproducibility. Agents are free to propose new tools — linters, analyzers, generators, profilers — and the provisioner gates them through automated checks before making them available.

**Verification Engine.** Executes the full constraint-checking pipeline for every submitted artifact. The pipeline stages are: compilation and build verification, static analysis and type checking, lint and formatting checks, unit test execution (both agent-provided tests and independently-generated adversarial tests), integration test execution, dependency and license audits, security scanning, and performance benchmarking where applicable. Produces standardized evidence artifacts for every stage. Runs incrementally where possible (module-affected tests, file-level dependency analysis) but requires periodic full verification runs to prevent blind spots.

**Adversarial Test Generator.** A separate agent instance that has access to the constraint envelope and interface contracts but not to the implementation. It generates tests designed to probe edge cases, boundary conditions, error paths, concurrency issues, and unusual input combinations. This separation — the implementer never writes its own verifier — is a deliberate quality assurance mechanism.

**Feedback Synthesizer.** When an agent's output fails the constraint gate, this component aggregates the failure information and distills it into structured, actionable feedback: the failing check, the expected behavior, the actual behavior, the relevant constraint, a diff showing the divergence, and any patterns from previous failures on similar work items. This is not "test failed, try again." This is a diagnostic package that accelerates the iteration cycle. The Feedback Synthesizer also feeds into the Constraint Miner: recurring failure patterns are candidates for new permanent constraints.

**Evidence Ledger.** The persistent store of all evidence records, linked to work items and merged commits. Provides traceability: which requirements were satisfied by which artifacts, which constraints were proven by which evidence, and full audit trails for every merge. Enables post-hoc questions like "why did this change merge?" and "what is our weakest area of coverage?"

**Resource Governor.** Monitors and enforces local resource budgets. Manages concurrency limits for CPU-intensive verification jobs, memory-aware scheduling to prevent OOM, GPU reservation for project-specific needs, disk I/O awareness, and token and cost budgets for remote API calls. Implements backpressure at every layer: when the verification pool is saturated, dispatch slows; when memory pressure rises, caches shrink and concurrency drops; when disk space falls low, garbage collection becomes aggressive and speculative execution is disabled.

**Integration Engine.** Manages the Git repository, the merge queue, and the serialized integration process. Handles branch creation, rebase, merge, conflict detection, rollback, and audit metadata. Every merge commit records which work item it satisfies, which agent produced it, which constraints governed it, and which evidence proves it.

---

## 6. Constraint-Based Program Synthesis

### 6.1 The Constraint Envelope

Every work item carries a constraint envelope: the complete set of constraints it must satisfy, including structural, behavioral, performance, interface, security, operational, documentation, and style constraints. An artifact is accepted if and only if it satisfies every constraint in its envelope and produces evidence for each. There is no partial credit.

Constraints propagate through the dependency graph. If module M depends on interface I, the guarantees and requirements of I become compositional constraints on M. This propagation is computed by the Constraint Compiler at planning time and included in each work item's envelope.

### 6.2 The Core Synthesis Loop

For each work item, the system executes: the implementer agent generates a candidate patch within scope, along with its own tests and rationale. The verification engine runs the full constraint-checking pipeline and the adversarial test generator produces independent tests. If constraints pass and evidence is sufficient, the work item proceeds to the merge queue. If constraints fail, the Feedback Synthesizer produces a structured diagnostic, and the agent is re-invoked with its previous attempt, the diagnostic, and any additional context. This iterates up to a configurable budget (default: five iterations). If the budget is exhausted, the work item escalates: first to the alternate provider, then to a higher-capability model, and finally to the operator with full diagnostic context.

This is counterexample-guided synthesis in engineering terms: the LLM proposes, the tools refute, the LLM repairs, until constraints are satisfied or the budget is spent.

### 6.3 Constraint Evolution — The "Never Again" System

Whenever a failure occurs — during synthesis iterations, after merge, or during integration hardening — the system generates one or more of: a new regression test, a new static analysis rule, a strengthened interface contract, a new architectural constraint, or an improved decomposition heuristic. The Constraint Miner Agent is responsible for analyzing failures and proposing new constraints, which are added to the Constraint Registry and applied to all future work items.

Every discovered class of bug becomes structurally harder to reintroduce. The constraint set grows monotonically in strictness. Over the lifecycle of a project, the system accumulates a body of project-specific knowledge about what goes wrong and encodes it into enforceable rules.

### 6.4 Constraint Conflict Resolution

Constraints can conflict (performance versus abstraction, security versus convenience). The system resolves conflicts through severity levels (must constraints override should constraints), explicit priority declarations from the Spec Map, and Architecture Decision Records that document the tradeoff and the rationale. When a genuine constraint contradiction is detected — constraints that cannot be simultaneously satisfied — the Constraint Compiler flags it and the Architect Agent is invoked to resolve it, either by relaxing a constraint, redefining a module boundary, or modifying an interface contract. All affected work items are revalidated after resolution.

### 6.5 Adversarial Verification

The constraint gate does not trust agents to test their own work honestly. In addition to agent-produced tests, the Adversarial Test Generator produces independent tests from the constraint envelope alone, without seeing the implementation. The Reviewer Agent performs adversarial code review, actively searching for edge cases, implicit assumptions, and failure modes. For high-risk work items (core modules, security-sensitive code, shared interfaces), both adversarial testing and adversarial review are mandatory.

---

## 7. Parallelism Model

### 7.1 Decomposition-Driven Parallelism

The degree of parallelism is determined by the decomposition. A project with 200 independent work items can have 200 agents working simultaneously, subject to API rate limits. The Architect Agent is therefore incentivized to produce the finest-grained decomposition that is still meaningful. Target work item size: 100 to 500 lines of output including tests. A 100,000-line project decomposes into roughly 200 to 1,000 work items, of which 40 to 60 percent can be parallelized in any given phase.

### 7.2 Pipeline Parallelism

Even with dependencies, parallelism is maintained through pipelining. While tier-2 work items are being implemented, the system simultaneously verifies tier-1 outputs, generates adversarial tests for tier-1, stabilizes tier-2 interface contracts, assembles context for tier-3 work items, and mines constraints from tier-1 failures. All five architectural layers are active simultaneously on different parts of the codebase.

### 7.3 Speculative Execution

For critical-path work items — those on the longest dependency chain where delays propagate to the entire project — the system dispatches the same work item to multiple agents simultaneously, potentially using different providers or different prompt strategies. The first output that passes the constraint gate is accepted; the rest are discarded. This trades API cost for latency on the critical path. The scheduler decides when speculative execution is warranted based on estimated criticality and the work item's failure history.

### 7.4 Hierarchical Coordination

To maintain coherence at scale, agents are organized hierarchically. The Architect Agent maintains global structural coherence. Domain-scope agents (designated per subsystem by the Architect) maintain local coherence within their subsystem. Worker agents implement individual work items. Reviewer, security, and performance agents operate cross-cutting, adversarially, and independently of the implementation hierarchy. This hierarchy is lightweight — it is a prompt-and-context structure, not a process bureaucracy.

### 7.5 Work Sharding Strategy

Sharding objectives: minimize overlapping files, stabilize interfaces before implementations, and make work items small and independently verifiable. Common shard types: interface definition shards (produce the contracts), implementation shards (produce the code), test shards (produce the test suites), documentation shards (produce the docs), and infrastructure shards (produce build, CI, and tooling configuration).

---

## 8. Git Coordination Protocol

### 8.1 Branch Architecture

The repository maintains: a **main** branch containing only fully-verified, fully-integrated, production-ready code. An **integration** branch that serves as the staging area where verified work items are merged and compositional checks are run. **Contract branches** (contract/module-name) containing stabilized interface contracts that are immutable during module implementation. **Work branches** (work/unit-id) created for each active work item, where agents commit their output. **Adversarial branches** (verify/unit-id) containing independently-generated test suites.

### 8.2 Non-Overlapping File Ownership

The decomposition guarantees that each source file is owned by exactly one work item. No two agents modify the same file simultaneously. This is enforced by the Constraint Compiler during decomposition validation and by a code ownership map in the repository. Shared code (utility libraries, type definitions, core configuration) is either part of a dedicated infrastructure work item completed early in the pipeline, or part of the immutable contract layer. For high-contention zones (core modules that multiple agents read from), the Workspace Manager provides read-only mounts and the Integration Engine uses optional path locks to prevent concurrent modifications.

### 8.3 Merge Queue Protocol

When a work item passes the constraint gate, the Integration Engine executes: rebase the work branch onto the current integration branch; run a fast compositional check (compile, integration tests, architectural constraint verification); if the compositional check passes, fast-forward merge into integration; update the task graph to mark the work item complete and unblock dependents. The merge queue is serialized — one merge at a time — ensuring that every compositional check runs against the true current state of integration.

### 8.4 Conflict Prevention and Resolution

Conflicts should be rare due to non-overlapping ownership. When they occur, the Integration Engine attempts automatic resolution for trivial cases (import ordering, formatting). Non-trivial conflicts are dispatched to the Integrator Agent, which has the context of both conflicting work items and their constraint envelopes. If the conflict implicates an interface contract, a re-planning cycle is triggered.

### 8.5 Rollback

If a compositional check fails after merge — a module that passed all individual constraints breaks something when combined with other modules — the Integration Engine reverts the merge. The Feedback Synthesizer analyzes the failure. The work item is re-dispatched with additional compositional context. If multiple recent merges are implicated, the system bisects to isolate the offending change.

---

## 9. Resource Management

### 9.1 CPU Allocation

The 16 cores are allocated with soft boundaries that the Resource Governor rebalances dynamically based on current bottleneck. Baseline allocation: 2 cores for the orchestrator (scheduler, dispatch controller, Git operations, monitoring), 2 cores for the Constraint Compiler and planning, 8 cores for the verification pool (compilation, test execution, static analysis — running up to 8 lightweight or up to 3 heavyweight verification jobs concurrently), and 4 cores for context assembly, codebase indexing, and prompt construction. The system maintains at least 15 percent CPU headroom at all times.

### 9.2 Memory Allocation

Of the 32 GB RAM: up to 4 GB for the orchestrator process and its data structures (constraint graph, schedule, agent state, dispatch queues), up to 8 GB for the verification pool (compilers, test runners, sandboxed execution), up to 8 GB for the Knowledge Plane (codebase index, semantic search, context cache, constraint registry), up to 6 GB for Git operations and workspace management, and approximately 6 GB held as headroom. The Resource Governor monitors memory pressure and reduces verification concurrency and cache sizes if usage exceeds 85 percent. The system must never swap.

### 9.3 Storage Management

Of the 1 TB SSD: up to 100 GB for the repository including full Git history, up to 100 GB for ephemeral verification sandboxes (aggressively garbage-collected), up to 50 GB for cached dependencies and tool installations, up to 50 GB for the codebase index and knowledge structures, and the remaining approximately 700 GB as working space. The storage manager runs garbage collection every 15 minutes, reclaiming sandbox and build artifacts no longer referenced by active work items. Evidence artifacts have retention policies: full records for the last N runs, summaries long-term, failure artifacts retained longer than success artifacts.

### 9.4 GPU Allocation

The 24 GB VRAM is reserved for project-specific needs. If the target project involves GPU code (CUDA, compute shaders, ML inference), the VRAM is used for compilation checks and benchmarking. If the project does not involve GPU compute, the VRAM is available for optional local models (embedding models for the Context Assembler, small classifier models for the scheduler) only if they fit within budget and only after all project-specific GPU needs are satisfied. The orchestrator itself never requires GPU.

### 9.5 API Cost Management

Every API call is logged with token count and estimated cost. The Resource Governor tracks cumulative cost per work item, per module, and per project. Cost ceilings can be set, and the system responds by disabling speculative execution, reducing iteration budgets, and preferring cheaper models for low-risk tasks. The default strategy: use fast, cheap models (Codex) for broad exploration and boilerplate; reserve expensive, strong models (Claude) for architectural decisions, debugging, and tasks that have already failed once.

### 9.6 Backpressure Protocol

Degradation is graceful and ordered. As resource pressure increases: first, speculative execution is disabled. Then verification concurrency is reduced. Then dispatch rate is throttled. Then cache sizes are reduced. The system always prefers slowing down to crashing. All throttling events are logged and surfaced on the dashboard.

---

## 10. Quality Assurance Architecture

### 10.1 Four Layers of Verification

Quality is assured through four independent layers, each catching different classes of defects.

Layer A: Agent Self-Verification. Agents test their own output before submission. They run their own tests, check types, verify interfaces against contracts. This catches obvious errors and reduces load on the constraint gate.

Layer B: The Constraint Gate. The full verification pipeline — compilation, static analysis, type checking, lint, unit tests, contract conformance, dependency audit. Every stage produces evidence. This catches all constraint violations.

Layer C: Adversarial Testing. Independently-generated tests by agents that see the constraint envelope but not the implementation. This catches subtle correctness issues, boundary conditions, error handling gaps, and concurrency problems the implementer did not consider.

Layer D: Compositional Verification. Full-build integration testing after merge into the integration branch. This catches issues arising from the combination of individually-correct modules: interface misunderstandings, protocol timing issues, shared state corruption, and emergent performance problems.

### 10.2 Quality Metrics

The system tracks: first-pass acceptance rate (target above 70 percent), average iterations to acceptance (target below 2.5), integration failure rate (target below 5 percent of merges), adversarial test discovery rate (adversarial tests should find issues in fewer than 15 percent of submissions that passed the agent's own tests), and constraint coverage (every requirement traces to at least one constraint, every constraint traces to at least one verification check). Deviation from targets triggers adaptive responses: increased context for agents, provider switching, re-decomposition of trouble areas, or increased speculative execution.

### 10.3 Definition of Done

A work item is merge-eligible only when: all must-severity constraints pass with evidence, all should-severity constraints either pass or carry an explicit override with justification, no files outside declared scope are modified, documentation is updated if public API surface changed, no new high-severity security findings exist, performance constraints (if applicable) are satisfied with benchmark evidence, and adversarial review is completed for high-risk items.

---

## 11. Operating Modes

### 11.1 Greenfield Mode

Repository starts empty. The orchestrator scaffolds the architecture: project structure, build system, CI pipeline, module layout, baseline constraints, shared type definitions, and foundational infrastructure. The Architect Agent produces the full decomposition from the Spec Map. Development proceeds through the standard parallel build pipeline.

### 11.2 Brownfield Mode

Repository already exists. The orchestrator indexes the existing code, infers module boundaries, identifies existing tests and their coverage, and builds the Constraint Registry incrementally from what the codebase already enforces. Work items focus on safe refactoring, adding constraint coverage to under-tested areas, and implementing new features within the discovered architecture. The Architect Agent proposes boundary adjustments and interface stabilization where the existing structure is ambiguous.

### 11.3 Hardening Mode

Primarily adds tests, security checks, performance budgets, and eliminates technical debt. Used before releases, after incidents, or when quality metrics fall below targets. Work items in this mode are heavily weighted toward the Test Engineer, Security, Performance, and Constraint Miner roles.

### 11.4 Exploration Mode

Generates multiple design candidates in parallel for an ambiguous requirement or a component where the best approach is unclear. Several agents independently implement different strategies for the same work item. The orchestrator evaluates all candidates against the constraint gate and selects the best (or presents the tradeoffs to the operator). Only the selected candidate merges.

---

## 12. Security and Sandboxing

### 12.1 Threat Model

The primary risks for this system are: prompt injection via repository content (a malicious file in the repo could influence agent behavior when included as context), supply chain attacks through agent-installed dependencies, credential leakage into prompts or logs, agents proposing unsafe operations under the banner of "resourcefulness," and exfiltration via tool calls or network egress.

### 12.2 Controls

Agent sandboxing: every agent's execution environment is containerized with filesystem mounts scoped to the workspace, controlled network egress, and no access to the orchestrator's process memory, the Git repository directly (all Git goes through the Integration Engine), or other agents' sandboxes.

Secrets handling: secrets are never placed in prompts or context. Secret scanning runs on every diff before merge. Logs and evidence artifacts are redacted. Real credentials are injected only in final deployment configuration, which is outside the orchestrator's scope. Mock credentials are used for all testing.

Dependency verification: when agents install packages, the Tool Provisioner checks against known vulnerability databases and pins versions. The constraint gate includes a dependency audit in its pipeline.

Audit logging: every tool invocation, every dependency addition, every constraint override, and every network request from an agent sandbox is recorded.

### 12.3 Agent Freedom Boundaries

Agents can: search for better approaches, propose and use new tools (after automated vetting), restructure plans, refactor aggressively within their module boundaries, try unconventional implementation strategies, and request scope extensions with justification. Agents cannot: access resources outside their sandbox, bypass or disable constraint checks, introduce obfuscated or intentionally unverifiable logic, modify files outside their declared scope without explicit approval, or commit directly to the integration branch.

---

## 13. Observability

### 13.1 Dashboard

The orchestrator exposes a real-time dashboard showing: the task graph with completion status for every work item, the dispatch queue and active agent sessions, the verification pipeline with pass/fail rates and current bottlenecks, the merge queue status, resource utilization (CPU, RAM, VRAM, disk, network), API cost accumulation per model and per subsystem, the critical path with estimated time to completion, and constraint coverage and quality metric trends.

### 13.2 Traceability

For any artifact in the final codebase, the audit trail answers: which agent produced it, which constraints governed it, which evidence proves it, how many iterations it took, what feedback was given, and which requirement it satisfies. The Evidence Ledger supports traceability queries: requirement to constraints to evidence to commits, constraint failure trends over time, and coverage gap identification.

### 13.3 Cost Transparency

Every API call is attributed to a work item and a project phase. The dashboard shows cost per work item, cost per module, cost per provider, and projected total cost. This enables informed decisions about when to enable speculative execution, when to escalate to expensive models, and when to tighten iteration budgets.

---

## 14. Project Lifecycle

### 14.1 Phase 1: Ingestion and Decomposition

The operator provides a design document. The Spec Ingestor extracts the Spec Map. The Architect Agent produces the decomposition. The Constraint Compiler validates it. The operator reviews and approves (or requests modifications). This is the only phase that requires operator involvement by default.

### 14.2 Phase 2: Foundation

The system builds the project foundation: repository initialization, build system, CI pipeline, shared types, utility libraries, and stabilization of all tier-1 interface contracts. A small number of agents work sequentially or with minimal parallelism, because the foundation must be solid before the parallel build begins.

### 14.3 Phase 3: Parallel Build

The bulk of the project. The scheduler dispatches work items at maximum parallelism. Completed items flow through the constraint gate and into the merge queue. The pipeline is fully active: dispatch, implementation, verification, and integration all run simultaneously on different parts of the codebase. This is where parallelism delivers its value.

### 14.4 Phase 4: Integration and Hardening

As the parallel build completes, the system shifts to integration-focused work. Full end-to-end tests. Performance benchmarks. Stress testing. The Constraint Miner runs on the full integrated codebase, generating new constraints from any issues discovered. The Architect Agent reviews holistically and flags architectural concerns.

### 14.5 Phase 5: Finalization

Documentation is verified complete. Build system is finalized. Release artifacts are produced. The Evidence Ledger and Constraint Registry are archived alongside the code as a formal record of what was specified, what was built, and how it was verified.

---

## 15. Failure Modes and Recovery

### 15.1 Agent Failure

Transient failures (API timeout, rate limit, service outage): retry with exponential backoff. Persistent failures: re-route to alternate provider. Budget exhaustion without constraint satisfaction: escalate to stronger model, then to operator with full diagnostic context.

### 15.2 Decomposition Failure

Systemic integration failures (rising merge failure rates, increasing iteration counts across multiple work items) indicate a flawed decomposition. The system detects this through quality metric monitoring and triggers a re-decomposition cycle: the Architect Agent is invoked with the current project state, the failure history, and the instruction to revise the decomposition for the affected area.

### 15.3 Specification Drift

If the implementation diverges from the spec over time — modules satisfy their individual constraints but the whole does not match the design document — the Constraint Compiler detects coverage gaps and the Architect Agent reconciles. This is caught during Phase 4 hardening if not earlier.

### 15.4 Resource Exhaustion

Graceful degradation per the backpressure protocol. The system never crashes due to resource exhaustion. It slows down, logs the throttling, and surfaces it on the dashboard.

### 15.5 Constraint Contradiction

When an agent reports that its constraint envelope is unsatisfiable, the Constraint Compiler performs consistency analysis. If confirmed, the Architect Agent resolves it by relaxing a constraint, redefining a boundary, or modifying a contract. All affected work items are revalidated.

### 15.6 Flaky Tests

Tests that pass and fail non-deterministically are a corruption vector — they erode trust in the constraint gate. Flaky tests are detected through repeated verification runs, quarantined immediately (removed from the blocking constraint set), and dispatched as high-priority work items to the Test Engineer Agent for stabilization. The Constraint Miner adds a new constraint: the specific flake pattern must include a reproduction environment capture.

---

## 16. Extensibility

### 16.1 Language and Framework Agnosticism

The orchestrator does not assume any particular programming language, framework, or stack. The constraint envelope format is abstract. The verification pipeline is pluggable: the appropriate compiler, type checker, test runner, linter, and static analyzer are selected and installed by the Tool Provisioner based on the target project's stack.

### 16.2 Pluggable Agent Providers

Adding a new provider (a new AI API, a local model, a human fallback) requires implementing a provider adapter that translates between the orchestrator's work item format and the provider's interface. The routing logic incorporates the new provider based on observed strengths. The architecture explicitly supports a future where local models running on the 5090 are good enough for some work item types — the provider adapter model makes this a configuration change, not an architectural change.

### 16.3 Reusable Constraint Libraries

Over time, the system accumulates reusable constraint sets for common patterns: REST API endpoints, database access layers, authentication modules, CLI tools, UI components. These accelerate future projects by providing pre-validated constraint envelopes the Architect Agent can reference during decomposition.

### 16.4 Custom Agent Roles

New roles can be defined as project needs evolve. A role is a system prompt, a tool set, a budget, and a success criterion. The operator can define project-specific roles (a "migration agent" for database work, a "compatibility agent" for cross-platform concerns) and the scheduler will dispatch to them based on work item tags.

---

## 17. Implementation Roadmap

### Phase 1 — Foundations

Spec Ingestor and Spec Map format. Constraint Registry and constraint envelope format. Task graph representation and dependency analysis. Workspace Manager with Git branch isolation. Basic Verification Engine (build, test, lint). Evidence Ledger with traceability. Integration Engine with serialized merge queue.

### Phase 2 — Parallel Swarm

Agent Runtime with role system and prompt assembly. Context Assembler with codebase indexing. Dispatch Controller with provider routing and rate limiting. Scheduler with critical-path algorithm and adaptive re-planning. Resource Governor enforcing hardware budgets. Feedback Synthesizer for structured iteration feedback.

### Phase 3 — Constraint Evolution and Adversarial Verification

Constraint Miner Agent and the "Never Again" pipeline. Adversarial Test Generator. Speculative execution for critical-path items. Constraint conflict detection and resolution. Exploration mode with parallel candidate evaluation.

### Phase 4 — Tooling Ecosystem and Hardening

Tool Provisioner with automated discovery and vetting. Security Agent integration and supply chain verification. Performance Agent with benchmark regression detection. Dashboard with real-time observability. Reusable constraint library infrastructure.

---

## 18. Appendix: Example Constraint Families

**Architecture.** No imports from higher layers into lower layers. All modules expose a single public API surface. No circular dependencies. Module boundaries align with the decomposition.

**Correctness.** All public functions have tests. All bug fixes include a regression test. All error paths are tested. No unchecked type coercions.

**Security.** No unsafe deserialization. No dependencies with known critical vulnerabilities. All external inputs are validated. No secrets in source or logs.

**Performance.** P95 latency under threshold for key operations. Memory usage under budget. No algorithms with worse-than-specified complexity.

**Reliability.** Every external call has a timeout and retry policy. Every subsystem emits structured logs. Circuit breakers on all external dependencies.

**Documentation.** Public API changes require updated docs with runnable examples. Every module has a README. Every significant decision has an ADR.

**Style.** Formatting matches project standard. Naming conventions enforced. No dead code. No commented-out code.

---

## 19. Summary

This orchestrator transforms a design document into a production-grade codebase through five mechanisms working in concert: constraint-based synthesis ensures correctness is defined formally and verified absolutely through machine-checkable evidence, not assurances. Maximum agent resourcefulness within sandboxed boundaries ensures AI agents bring their full capability to every problem without artificial limits. Fine-grained decomposition with immutable interface contracts enables hundreds of agents to work simultaneously without coordination overhead. Git-native serialized integration ensures the repository stays coherent despite massive parallelism. And continuous constraint evolution ensures that every failure makes the system permanently stricter, accumulating project-specific quality intelligence that grows monotonically over the project's lifetime.

The result is a personal tool that turns a design document and an API budget into a codebase — as modular, correct, and complete as current AI capabilities allow, verified at every step by evidence that no artifact can bypass.
