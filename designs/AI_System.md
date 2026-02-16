Design Document (v3)
24GB-VRAM Neuro-Symbolic Discovery Engine for Genuinely Novel Analytical Governing Equations
Maxwell-Class Structure Discovery with Field/Potential Discovery, Non-Cheating Closure, and Certification-Grade Deployment
1) Purpose and business outcome
1.1 Objective

Build an AI system that discovers compact, physically valid, certifiable symbolic theories (algebraic laws, ODEs, PDEs, closure relations, constitutive models, variational principles) that replace or dramatically simplify high-fidelity simulations, reducing hundreds of thousands (or millions) of DOF to a small set of state variables or a closed-form law while preserving accuracy in the regime that matters.

1.2 Business outcomes / ROI metrics (measured on target workflows)

Simulation speedup at fixed QoI tolerance (e.g., 100× faster for <2% QoI error).

Effective DOF reduction to meet tolerance (e.g., 500k DOF → 500–5k effective DOF or analytic predictor).

Engineering hours saved (mesh/tuning/reruns reduced; fewer param sweeps).

Certifiability and auditability: interpretable law, constraint compliance, traceability, reproducibility.

Deployment cost: drop-in solver/UMAT/closure plugin with stable Jacobians and monitoring.

Reliability contract: declared envelope, failure detectors, and fallback strategies.

1.3 Maxwell-class discovery intent (design goal)

The system is explicitly designed to search beyond fixed templates by enabling discovery of:

New state variables (closure coordinates / internal variables),

New operators (nonlocal, fractional, multiscale, memory operators),

New field definitions/potentials (discover the “right” variables where the law is simple),

New generative principles (Lagrangians / Hamiltonians / dissipation potentials) from which governing equations are derived,

New invariants and symmetry structures that unify regimes,

Anomaly-driven unification pressure (the “why” behind Maxwell-level structure).

Important realism note: This is not a promise that “a Maxwell-class law can be forced on demand.” It is a design that removes the usual architectural bottlenecks and introduces the missing discovery pressures that make such a discovery architecturally plausible when the right anomaly-rich data/regimes exist.

2) Scope: what “groundbreaking formula” means here

This system targets new symbolic structure, not “fit coefficients in a known template.” It must be capable of discovering:

New closures: symbolic operators replacing unresolved microphysics/turbulence/heterogeneity.

New constitutive theories: stress–strain–rate–history relations for complex materials, damage, plasticity, multiphase mixtures, etc., including thermodynamic admissibility.

New reduced governing PDEs: compact PDEs on minimal fields that reproduce QoIs from the full model.

New invariants / latent variables / internal variables: discovering the correct state representation that makes the law simple and predictive.

New field variables and potentials: discovering transformations/potentials that re-express the physics as a simpler unified structure (a hallmark of Maxwell-class discoveries).

New generative principles: variational / energy / dissipation formulations that imply conservation + stability.

New operator classes: nonlocal kernels, fractional operators, multiscale averaging, memory terms, and their symbolic compressions.

Novelty that is defensible: proven non-equivalence under allowed transforms + functional separation on adversarial regimes.

3) Constraints and compute assumptions
3.1 Hard constraint

Only 24GB VRAM available.

3.2 Design response

GPU is used only for guidance/proposals (quantized model / lightweight priors).

Heavy work—symbolic manipulation, canonicalization, equivalence checking, verification, solver calls, and certification—runs on CPU/RAM and existing simulation infrastructure.

3.3 Recommended hardware envelope

1× GPU with 24GB VRAM (RTX 4090 / RTX 6000 Ada 24GB class)

64–256GB RAM (symbolic trees, caches, datasets, solver snapshots, posterior ensembles)

Fast SSD (equation store, caches, simulation output)

Optional: CPU cluster for high-fidelity simulation calls (not required for VRAM feasibility)

3.4 Performance reality and throughput strategy

Hardware limits throughput; design maximizes “discovery per unit compute” via:

weak-form PDE identification (derivative-robust)

active learning (minimal new solves)

anomaly-driven targeting (don’t waste solves where baselines already work)

equivalence-class pruning (avoid re-evaluating same law in different guises)

staged discovery (structure → principle → equation → compression → deployment)

certificate ladder (cheap rejects first; expensive proofs only for survivors)

4) System overview (high-level architecture)
4.1 Core idea

A neuro-symbolic closed loop that discovers theory, not just regression:

represent candidate theories in a typed, physics-aware grammar,

propose candidate structures using a small GPU model + priors,

search globally over theory space with CPU-first synthesis,

verify rigorously (constraints, stability, well-posedness, limits),

actively acquire data where theories disagree,

compress into deployable reduced models,

audit novelty and provenance with equivalence-aware checks,

drive discovery with anomaly families (baseline failures) to force unification.

4.2 Modules (expanded, preserving v2 plus required fixes)

A. Data & Regime Manager: regimes, BC/ICs, nondimensionalization, QoIs, train/val splits by regime.
B. Representation Layer (Typed Theory DSL): grammar for fields/operators/tensors/invariants/latent variables/principles.
C. Hypothesis Generator (GPU-guided, optional): compact proposer that emits typed AST expansions.
D. Symbolic Search Engine (CPU): MCTS / evolutionary / Bayesian program synthesis over typed ASTs with equivalence pruning.
E. Structure-First Discovery Pipeline: symmetries → invariants → (field/state) → principles → equations → compression.
F. Physics-Aware Verifier + Certifier: hard constraints, stability/well-posedness, limits, thermodynamics, identifiability, uncertainty.
G. Weak-Form & Solver-Integrated Residual Engine: FEM-compatible integral residuals, adjoints, Jacobians, robust noise handling.
H. Active Data Selector (Cost-aware): simulation/experiment queries maximizing information gain per compute dollar and exposing regime failures.
I. Deployment Compiler: drop-in solver components, ROMs, closure plugins, Jacobians, monitors.
J. Novelty, Equivalence & Provenance: canonicalization, quotienting by symmetries/transforms, similarity + functional novelty checks, audit trail.
K. Competitor/Baseline Suite Manager: curated “best-known” models for head-to-head comparisons and novelty validation.

Added in v3 (fixes):
L. Field Discovery & Gauge/Redundancy Module: discover new fields/potentials/transformations (and gauge freedoms where applicable) with strict admissibility and equivalence handling.
M. Anomaly & Theory-Repair Loop: baseline failure clustering + targeted theory repair to create Maxwell-class unification pressure.
N. Certificate Ladder & Numerical Contract Manager: tiered certification (cheap→strong) and explicit solver-compatibility obligations.

5) Input/Output contracts
5.1 Inputs

High-fidelity data: simulations/experiments

𝐷
=
{
(
𝑥
,
𝑡
,
𝜃
)
↦
observed fields
,
QoIs
}
D={(x,t,θ)↦observed fields,QoIs}

Known constraints: symmetries, conservation, positivity, monotonicity, frame indifference, causality, dissipation, bounds, etc.

Target usage: operating ranges, acceptable errors, QoIs, deployment environment, solver interfaces.

Known limiting laws (optional but highly valuable): expected asymptotic behaviors in certain regimes.

Baseline model library: “best known” equations/closures/constitutive laws for the domain.

5.2 Outputs

One or more theories expressed as:

typed AST + LaTeX + executable code

optionally a principle (Lagrangian / Hamiltonian / dissipation potential) + derived PDEs

optionally a field transformation/potential representation (new variables where the theory is simplest)

Confidence/certification report:

residuals, stability/well-posedness checks, limit consistency, uncertainty bounds, identifiability, regime-of-validity map

certificate tier achieved + what remains unproven (if any)

Deployable artifacts:

ROM solver / closure plugin / UMAT / constitutive law module

Jacobians/tangents for implicit solvers

monitoring + out-of-regime detection logic + fallback policies

Novelty & provenance packet:

canonical form under equivalences (including field transforms)

similarity results, baseline comparisons, functional novelty tests

full lineage graph and reproducible discovery log

5.3 Value Contract (added in v3; required for “saves millions”)

For each target workflow, specify:

Baseline: solver + mesh/DOF + runtime + accuracy + engineering effort

Deliverable type:

closure plugin for coarse solver, or

constitutive UMAT, or

reduced PDE solver, or

QoI analytic predictor, or

hybrid ROM + monitors + fallback

KPI contract:

max QoI error, confidence bounds, failure probability (or allowed fallback rate)

max runtime, memory, integration constraints

Deployment boundary:

envelope of validity (dimensionless groups, BC/IC ranges, material params)

required monitoring signals and fallback triggers

This prevents the system from “discovering something cool” that doesn’t map to dollars.

6) Representation: making “not just coefficients” structurally inevitable
6.1 Strongly typed symbolic grammar (critical)

A grammar that can express:

Scalars/vectors/tensors (incl. symmetric tensors, 4th-order moduli)

Differential operators: grad/div/curl/Laplacian/material derivative

Integral/weak-form operators and test functions

Nonlocal operators: kernels, convolution, multiscale averages

Fractional operators: symbolic fractional Laplacian / memory kernels (where applicable)

History/memory: internal variable dynamics, hereditary integrals, state evolution

Piecewise regime logic with smooth switching (physically meaningful)

Potentials/principles: free energies, dissipation potentials, action integrals

Example type signatures:

grad: ScalarField → VectorField

div: VectorField → ScalarField

stress: (strain, strain_rate, history_vars, params) → SymTensorField

DissipationPotential: (state, state_rate, params) → Scalar

KernelOp: (Field, Kernel, params) → Field

6.2 Dimensional analysis + invariance baked in

Before search:

automatic nondimensionalization (Buckingham Pi)

enforce invariances: Galilean, rotational, frame indifference, isotropy/anisotropy as specified

represent laws in invariants where appropriate (principal invariants, objective rates, tensor bases)

This prevents “fancy coefficient fitting” by restricting the space to physically admissible structure.

6.3 Explicit support for discovering new state variables (closure coordinates / internal variables)

Representation allows proposing new internal variables 
𝑧
z with their own evolution:

𝑧
˙
=
𝑓
(
𝑧
,
invariants
,
∇
(
⋅
)
,
𝜃
)
z
˙
=f(z,invariants,∇(⋅),θ)

But in v3, this is not a loophole (see CNCC in 6.7).

6.4 Explicit support for discovering new operators

Allow operators that are not pre-enumerated as a finite list:

nonlocal kernels that later compress to local PDE corrections

fractional/memory operators

multiscale homogenization operators and asymptotic expansions

operator “macros” that the system can introduce, then later symbolically compress into standard operators + corrections

6.5 Discovery via generative principles (variational / energy / dissipation)

First-class hypothesis class:

free energy 
Ψ
(
⋅
)
Ψ(⋅)

dissipation potential 
Φ
(
⋅
)
Φ(⋅)

action functional 
𝑆
=
∫
𝐿
(
⋅
)
 
𝑑
𝑡
S=∫L(⋅)dt

Derive governing equations (Euler–Lagrange, Hamiltonian form, generalized standard materials):

conservation emerges structurally

stability and admissibility become easier to certify

Maxwell-style unification becomes possible because principles generalize across regimes

6.6 Field discovery, potentials, and gauge/redundancy (added in v3; Maxwell-critical)

Maxwell-class structure often appears after introducing the right fields (potentials) and recognizing redundancies (gauge freedoms). v3 adds a first-class capability:

6.6.1 Discoverable field transformations

Allow restricted, admissible transformations such as:

Potential introductions: e.g. 
𝐵
=
∇
×
𝐴
B=∇×A, 
𝐸
=
−
∇
𝜙
−
∂
𝑡
𝐴
E=−∇ϕ−∂
t
	​

A-style constructs (domain-general, not EM-specific)

Change-of-variables: 
𝑢
~
=
𝑇
(
𝑢
,
∇
𝑢
,
Π
,
𝜃
)
u
~
=T(u,∇u,Π,θ) in restricted classes (invertible or identifiable up to known gauge)

Coarse-graining functionals: 
𝑢
~
=
𝐺
[
𝑢
]
u
~
=G[u] where 
𝐺
G is constrained (e.g., averaging kernels with bounded support, multiscale projection operators)

6.6.2 Admissibility gates for field discovery

Any proposed field redefinition must satisfy:

Invertibility or defined equivalence: invertible in-envelope, or identifiable up to declared gauge freedom

Dimensional + invariance consistency

Complexity reduction: lowers MDL and improves out-of-regime generalization

Physical interpretability hooks: ties to measurable quantities or accepted constructs (e.g., potentials, invariants, coarse averages)

6.6.3 Transform-aware equivalence

Equivalence classes now include not only algebraic and weak-form identities, but also admissible field transforms (see Section 13).

6.7 Closure Non-Cheating Constraints (CNCC) for latent variables and operator macros (added in v3)

To prevent “latent z explains everything” and “kernel is just flexible coefficients,” any introduction of new latent variables or operator macros must pass hard constraints:

Observability / identifiability certificate

Show that the augmented state is identifiable from admissible data/sensors (up to a defined equivalence/gauge).

Reject theories where 
𝑧
z is unidentifiable or only identifiable via unrealistic measurements.

Minimality

Adding 
𝑧
z or an operator macro must reduce description length and improve extrapolation/generalization.

If it only improves fit in-sample, reject.

Physical anchoring
At least one of:

𝑧
z corresponds to a measurable microstructural descriptor, OR

𝑧
z is a constrained coarse-graining of observed fields, OR

𝑧
z connects to a conservation residual / defect field / anomaly family, OR

𝑧
z is a thermodynamic internal variable with boundedness + dissipation structure.

Compression or bounded approximation

Any macro operator (kernel/memory/fractional) must either:

symbolically compress into a local PDE correction in the target regime, OR

provide a certified bounded approximation error in the intended envelope/discretization.

These constraints make “novel state/operators” a true discovery path, not a cheat.

7) Hypothesis generation under 24GB VRAM
7.1 Small “theory proposer” model (GPU) — optional, not a hard dependency

A compact model (typically 1–7B params) fine-tuned to emit typed AST tokens:

4-bit/8-bit quantized so it fits in 24GB with KV-cache optimizations

emits incrementally (small context windows) to reduce VRAM pressure

used as a proposal prior, not the discovery engine

v3 requirement: the system must still function if the proposer is replaced with:

a probabilistic grammar prior,

a lightweight policy network,

or hand-built heuristics.

This avoids “we need a giant model to discover laws.”

7.2 Training/fine-tuning strategy (24GB-compatible)

parameter-efficient tuning (LoRA/QLoRA-style)

curriculum:

rediscover known laws,

discover hidden PDEs/closures from synthetic data,

“anti-template” tasks requiring latent variables/operators/principles,

“field discovery” tasks where only a change of variables exposes simplicity,

anomaly-driven repair tasks (see Section 8 and 11).

7.3 Proposal novelty pressure (added in v3)

To reduce “anchoring” on known forms, the proposer is guided by:

baseline-distance penalty in embedding space (avoid close matches to baseline theories)

novel-structure bonus for introducing:

admissible new fields/potentials,

new invariants,

macro operators that are compressible,

variational/dissipation principles

but only if the candidate passes cheap admissibility checks (dimensions, invariances, CNCC prechecks)

7.4 CPU-first orchestration: global symbolic synthesis

Use MCTS / evolutionary search / Bayesian program synthesis with:

neural policy prior: which expansions to try

neural value prior: predicted promise

novelty bonuses: avoid cycling known forms

regime coverage: candidates must explain multiple regimes, not just one slice

7.5 Search complexity control (prevents combinatorial death)

Mandatory mechanisms:

canonical hashing at every subtree

rewrite normalization during search (commutativity/associativity, tensor symmetries, invariant basis normalization)

memoized residual evaluation keyed by canonical form

partial evaluation (quick reject tests before expensive solver calls)

trust-region complexity growth: increase complexity only when it improves out-of-regime generalization

certificate-ladder gating: only pay for expensive checks on survivors (Section 9.8)

8) Structure-First Discovery Pipeline (prevents “term soup PDE regression”)

Instead of proposing random PDE term lists, v3 enforces staged discovery with explicit gates.

Stage 0 — Regime definition (same intent, now part of the Value Contract)

nondimensional groups

operating envelope

QoIs and constraints

baseline theories to beat

cost model for simulation calls

Stage 1 — Symmetry & invariant discovery

infer/validate invariances from data + domain constraints

discover conserved quantities or approximate invariants

identify objective rates and invariant tensor bases

Stage 2 — Field discovery & state discovery (expanded in v3)

propose minimal coarse variables + latent/internal variables that close dynamics

propose new fields/potentials/transformations that simplify laws

enforce CNCC: identifiability, minimality, anchoring

Stage 3 — Principle discovery (preferred path; now strongly gated)

search over 
Ψ
Ψ, 
Φ
Φ, 
𝐿
L

derive equations automatically

certify thermodynamic admissibility structurally

Stage 4 — Equation discovery (direct path when needed; gated)

search over conservative forms / weak forms / closure operators directly

remain equivalence-aware and constraint-first

Stage 5 — Operator compression

if nonlocal/memory operators exist, compress to local PDE + corrections in target regime

produce simplest deployable symbolic form

Stage gating rules (added in v3)

Default path is Stage 3 (principle-first) unless domain forbids.

Stage 4 is permitted only if:

Stage 3 fails to reach target generalization after defined compute budget, or

no admissible principle class exists for the target physics.

Complexity only increases if it improves out-of-regime performance and reduces anomaly residuals (Section 11).

9) Verification & certification: making discoveries real (and deployable)

The verifier is not “a scoring function”; it is a certification engine.

9.1 Hard constraints (instant reject)

dimensional consistency

required symmetries/invariances (frame indifference, isotropy/anisotropy)

conservation form where mandated (mass/momentum/energy)

positivity/bounds (e.g., density, viscosity, damage variables)

causal structure (no unphysical dependence)

CNCC prechecks for latent variables/operators (Section 6.7)

9.2 Thermodynamic consistency (constitutive/closure laws)

Enforce dissipation inequality structurally where relevant:

generalized standard materials via 
Ψ
Ψ and 
Φ
Φ

Clausius–Duhem compliance

monotone dissipation (
𝐷
≥
0
D≥0) as a structural guarantee, not a soft penalty

9.3 Well-posedness and PDE type certification (regime-aware)

For PDE candidates:

classify PDE type (hyperbolic/elliptic/parabolic or mixed) over regime

check stability conditions (energy estimates, monotonicity, spectral properties)

check solver-compatibility constraints (CFL bounds, stiffness, conditioning)

penalize or reject laws that create ill-posedness in target envelope

9.4 Asymptotic / limit consistency tests (unification pressure)

Systematically test whether discovered theory reduces to known limits:

small strain → linear elasticity (if expected)

low Reynolds → Stokes (if relevant)

dilute limit, quasi-static, isothermal, etc.

mesh refinement consistency where applicable

These tests force theories to unify regimes rather than overfit.

9.5 Soft scoring (multiobjective, subordinate to certification)
𝑆
𝑐
𝑜
𝑟
𝑒
(
𝐸
)
=
𝑤
1
𝐹
𝑖
𝑡
(
𝐸
)
+
𝑤
2
𝐺
𝑒
𝑛
(
𝐸
)
−
𝑤
3
𝑀
𝐷
𝐿
(
𝐸
)
−
𝑤
4
𝐼
𝑛
𝑠
𝑡
𝑎
𝑏
𝑖
𝑙
𝑖
𝑡
𝑦
(
𝐸
)
−
𝑤
5
𝑁
𝑜
𝑛
𝐼
𝑑
𝑒
𝑛
𝑡
(
𝐸
)
Score(E)=w
1
	​

Fit(E)+w
2
	​

Gen(E)−w
3
	​

MDL(E)−w
4
	​

Instability(E)−w
5
	​

NonIdent(E)

Where:

Fit: weak-form residual + QoI error

Gen: regime-split extrapolation + adversarial BC/IC tests

MDL: minimum description length (AST size, operator count, state dimension)

Instability: blow-ups, stiffness, poor conditioning

NonIdent: identifiability penalties (hard-gated for latent/operator additions)

9.6 Identifiability + equivalence awareness (prevents fake novelty)

detect non-identifiable parameterizations (sloppiness, flat directions)

quotient by equivalences:

variable transforms / scalings / nondimensionalization

integration by parts identities (weak form)

tensor index symmetries and basis equivalences

gauge freedoms / reparameterizations

admissible field transforms (added v3; see Section 13)

treat each equivalence class as one theory; keep simplest canonical representative

9.7 Uncertainty quantification: maintain a theory posterior

Don’t output “one equation”; maintain a posterior / ensemble:

Bayesian program search approximation or top-K theory ensemble

disagreement maps guide active learning

uncertainty bounds accompany deployment and monitoring

9.8 Certificate ladder (added in v3; scalable proof obligations)

To make certification tractable under hardware limits, every candidate progresses through tiers:

Tier 0 (Syntactic/Local):

dimensional + invariance checks

basic bounds/positivity

CNCC prechecks

cheap symbolic sanity checks

Tier 1 (Structural):

existence of a valid energy/dissipation structure (if in that hypothesis class)

conservative form validation where required

basic limit consistency

Tier 2 (Numerical/Empirical in declared envelope):

stable forward sims across regime grid

mesh/step sensitivity tests

perturbation robustness

conditioning + stiffness diagnostics

must satisfy the Numerical Contract (Section 9.9)

Tier 3 (Analytic/Semianalytic where feasible):

provable hyperbolicity/parabolicity in envelope

energy estimates / monotonicity proofs

contraction properties, spectral bounds

optional theorem prover hooks for invariance/consistency proofs

Deployment minimum: Tier 2 always; Tier 3 required for safety-critical workflows where demanded.

9.9 Numerical Contract (added in v3; solver-compatibility as a first-class constraint)

Any deployable theory must declare and satisfy:

differentiability class required by solver (C¹/C²) in the envelope

bounded derivatives in the envelope (avoid solver blow-ups)

availability of stable Jacobians/tangents (analytic or certified AD)

stable behavior under implicit integration and nonlinear solves

defined fallback behavior when monitors indicate extrapolation

This prevents “cool equations” that bankrupt the solver.

10) Weak-Form & Solver-Integrated Residual Engine (derivative-robust and FEM-native)

Default to weak form identification:

compute integral residuals with test functions

integration by parts reduces differentiation order

compatible with FEM outputs and irregular meshes

constraints evaluated in conservative/variational form

Capabilities:

multiple test function families (local, global, multiscale)

boundary term handling explicitly

adjoint/automatic differentiation hooks for efficient sensitivity

supports noisy measurements and sparse sensors (experiments)

10.1 Test-function coverage requirement (added in v3)

Weak-form discovery can be “fooled” if tests don’t excite discriminating modes. Require:

a coverage library spanning relevant scales/modes

adaptive enrichment: if top-K theories are indistinguishable, generate new test functions (or new BC/IC probes) that maximally separate them

explicit reporting of mode coverage in the certification report

11) Active learning loop (how it beats brute-force DOF) + anomaly-driven discovery pressure
11.1 Cost-aware active learning (expanded in v3)

When multiple theories survive certification and fit data:

select new simulation/experiment queries that maximally separate candidates per unit cost

Acquisition objectives:

expected information gain over theory posterior / cost

“where top-K theories disagree most on QoIs”

“BC/IC that makes theory A stable but B unstable”

“parameter sweep that violates an invariance or limit behavior”

“probe points that maximize identifiability of latent variables”

cheap probe tier first: coarse/low-fidelity solves to narrow candidates, then expensive solves for finalists

11.2 Anomaly & Theory-Repair Loop (added in v3; Maxwell-level pressure)

Maxwell-class laws arise from explaining and unifying structured failures of existing laws. v3 adds:

Baseline fitting + residual diagnostics

Fit best-known baselines from the baseline suite

Compute structured residuals in weak form and in QoIs

Identify where, when, and how baselines fail

Anomaly family clustering

Cluster failures into anomaly families (e.g., “high-shear instability,” “boundary-layer mismatch,” “multiphase coupling defect,” “memory effect signature”)

Each anomaly family has:

a regime signature (dimensionless groups, BC/IC patterns)

a failure mode (conservation defect, instability, QoI bias)

a target repair objective

Theory repair objectives

Force candidate theories to eliminate anomaly families with minimal added complexity

Prefer theories that unify multiple anomaly families under one principle/field transformation

Discovery pressure

Candidate scoring includes anomaly-family elimination and unification as first-class targets

Active learning prioritizes regimes that reproduce anomaly families

This creates the same kind of “pressure” that historically drives new governing structures.

12) How the discovered theory reduces DOF in practice

DOF savings come from:

Closure discovery: replace unresolved physics with symbolic closure → coarse meshes possible.

Homogenized constitutive law: eliminate microscale DOF by mapping coarse invariants → stresses/fluxes.

Reduced state variables: identify minimal Markovian state (incl. CNCC-compliant latent variables) → low-dimensional evolution.

Operator compression: macro operators collapse to compact local PDE + corrections in target regime.

Principle-based reduction: variational structure yields stable reduced integrators and extrapolation.

Field transformation/potentials: choose variables where dynamics are simpler and fewer fields are needed.

12.1 Online error estimation + fallback (added in v3; deployment-critical)

To ensure savings persist in production:

include a runtime monitor estimating closure-induced error / extrapolation risk

if risk triggers, fall back to:

safer baseline closure, or

locally refine mesh/DOF, or

call a higher-fidelity model selectively

13) Novelty assurance (defensible “never before known” workflow)

Novelty cannot be proven absolutely, but it can be made defensible through explicit obligations.

13.1 Equivalence-aware canonicalization (expanded in v3)

Canonicalize under:

algebraic rewrites (commutativity, associativity, factorization)

tensor index canonical forms and invariant basis normalization

weak-form equivalences (integration by parts)

admissible scalings and nondimensional transforms

gauge freedoms (where declared)

admissible field transforms/potentials (added v3)

Each equivalence class yields one canonical representative.

13.2 Similarity search vs corpora (syntactic novelty evidence)

embed AST/graph forms; nearest-neighbor search against:

public physics/engineering equation corpora

internal historical models

curated baseline libraries

report similarity scores + closest matches + transformations needed to map

13.3 Functional novelty obligations (the primary novelty layer)

A theory is considered “functionally novel” only if it meets:

Functional separation obligation

For each strong baseline, there exists at least one adversarial regime test where:

baseline fails,

candidate succeeds,

separation is robust (noise/mesh/BC perturbations)

Compression obligation

If candidate uses macro operators:

provide symbolic compression to local PDE form in target regime, OR

provide a certified bounded approximation error

Non-cheating latent obligation

If candidate uses latent variables:

provide identifiability + minimality + anchoring certificates (CNCC)

Transform-equivalence obligation

Attempt to map candidate to baseline via admissible field transforms.

If mapping succeeds within tolerance, the candidate is not claimed as a fundamentally new law (it may still be a valuable re-formulation, but must be labeled correctly).

13.4 Provenance and priority packet

full audit trail: data → constraints → anomaly families → search lineage → validations

reproducible run scripts and hashes

theory lineage graphs (which theories were rejected and why)

14) Validation suite and acceptance gates (engineering-grade)
14.1 Mandatory validation tests

recovery tests: rediscover known laws from synthetic data

regime splits: train on one set of BC/IC/params; validate on disjoint regimes

stress tests: perturbations, noise, mesh changes, domain scaling

conservation/dissipation checks: integrated quantities

stability tests: forward simulate reduced model; no blow-ups in envelope

downstream KPI: QoI error on real engineering targets

14.2 Maxwell-readiness “red team” tests (added in v3; anti-coefficient collapse)

Template trap test

Data generated by a law not representable without introducing a macro operator or field transform.

Fail if the system returns a flexible approximation disguised as novelty.

Latent-variable cheat test

Allow latent variables but withhold identifiability conditions.

Fail if it “discovers” a great-fit latent model without identifiability certificate.

Out-of-regime inversion test

Train on one regime; validate on another requiring correct field/state choice.

Pass only if field/state discovery generalizes.

Adversarial equivalence test

Provide syntactically different but equivalent theories (IBP, scaling, field transforms).

Pass only if equivalence quotienting merges them.

Solver integration test

Plug discovered closure into implicit solver with Jacobians.

Fail if it destabilizes or causes non-convergence in envelope.

14.3 Acceptance gates (expanded)

Gate 1: sanity—known law rediscovery + Tier 0/1 certificates + unit tests

Gate 2: value—beats curated best-known baselines on QoIs and runtime under the Value Contract

Gate 3: risk—Tier 2 certificate + out-of-regime failure characterization + monitors + fallback

Gate 4: deploy—solver integration, stable tangents/Jacobians, reproducibility artifacts, maintenance plan

15) Implementation details (24GB VRAM practical plan)
15.1 Core stack

PyTorch for proposer (quantized; optional)

SymPy + custom rewrite engine for canonicalization/equivalence

JAX/NumPy/Numba for fast residual evaluation

MCTS/evolutionary search in Python/C++ with multiprocessing

FEM/solver integration layer for weak-form residuals

Optional theorem prover hooks (Lean/Isabelle) for invariance proofs where feasible

15.2 VRAM management

4-bit/8-bit quantization for proposer

CPU offload for KV cache if needed

incremental AST emission, small contexts

batch candidate evaluation on CPU; GPU only for proposal logits

proposer can be replaced with non-neural priors

15.3 Throughput accelerators

equivalence-class hashing/memoization

staged certification (cheap rejects first; certificate ladder)

cached solver calls and surrogate prefilters

cost-aware active learning to minimize expensive simulations

anomaly-family targeting to focus compute where baselines fail

15.4 Field discovery implementation hooks (added in v3)

restricted transform DSL (invertible maps, potential introductions, constrained coarse-graining)

transform-equivalence search (attempt mapping candidate ↔ baseline)

gauge declaration and canonical gauge fixing (where applicable)

16) Deployment and monitoring (so it actually saves millions)

Deliver not only equations but a production-ready reduced model:

stable integrator settings and recommended timestepping

Jacobians/tangents for implicit solvers (UMAT/closure)

regime-of-validity contract and monitors:

detect extrapolation / out-of-distribution states

estimate closure-induced error online

trigger fallback to higher-fidelity model or safe baseline

versioned artifacts with reproducible validation reports

integration adapters for the firm’s simulation tools

17) Example deliverables (what the firm gets)

theory.yaml (typed AST + metadata + canonical form + equivalence class id)

theory.tex (human-readable formatted law or principle + derived equations)

principle.yaml (if variational/thermo path used: 
Ψ
,
Φ
,
𝐿
Ψ,Φ,L and derivations)

field_transform.yaml (if field discovery used: transforms/potentials + admissibility + gauge)

closure.cpp / umat.f / user_subroutine (drop-in code + Jacobians)

rom_solver/ (reduced solver with monitors and fallback logic)

validation_report.pdf (regimes, residuals, stability, limits, QoIs, runtime, certificate tier)

novelty_report.json (syntactic similarity evidence + functional novelty obligations + transform-equivalence attempts)

anomaly_report.json (baseline residuals, anomaly families, repair targets, solved/unsolved)

value_contract.md (workflow KPI targets, envelope, deployment requirements)

provenance_log/ (repro scripts, hashes, lineage graphs)

What this v3 guarantees architecturally

This version makes Maxwell-class structure discovery architecturally plausible under 24GB VRAM because it explicitly supports:

new latent state variables with non-cheating constraints (CNCC)

new operators with compression or bounded-error obligations

field discovery/potentials + gauge handling (a core Maxwell-class ingredient)

principle discovery (variational/thermo), not just PDE regression

weak-form identification with test-function coverage and adaptive separation

equivalence-class quotienting including admissible field transforms (stops fake novelty)

anomaly-driven theory repair loop (creates “why” pressure for new laws)

certificate ladder + numerical contract (deployable, solver-safe discoveries)

functional novelty obligations vs best baselines (defensible novelty, not syntactic novelty theater)

cost-aware active learning (minimize expensive solves; maximize discovery per compute)