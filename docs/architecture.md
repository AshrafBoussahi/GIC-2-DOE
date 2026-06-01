# qGridX Architecture

## The four-stage loop

qGridX implements a "cut-guided amortized sampling" heuristic — a loop over
four stages, inspired by Benders decomposition but making **no optimality
guarantees** (Correctness Rule R1).

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CUT-GUIDED AMORTIZED SAMPLING LOOP               │
│                   (HEURISTIC — not Benders decomposition)            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Stage 1: Classical preprocessing                             │   │
│  │  - Load power system → ProblemSpec                           │   │
│  │  - Build scenario tree (weather × load × N-1)               │   │
│  │  - Run cheap convex relaxation (DC-OPF) once                 │   │
│  │  - Output: Ising {h_i, J_ij}  ← R2: rescale + cap cuts      │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │ Ising                                     │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │ Stage 2: Quantum master (PCE-GQE)                            │   │
│  │  - GNN encoder: Ising graph → context                        │   │
│  │  - PCE assignment: m k-body Pauli strings, 3 families        │   │
│  │  - Transformer decoder (PCE-brickwork grammar mask):         │   │
│  │    generates M circuit token sequences                       │   │
│  │  - CUDA-Q backend: executes each circuit                     │   │
│  │  - Readout: x_i = sign(<P_i>), w_i = 1/|<P_i>|              │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │ M (bitstring, confidence) pairs           │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │ Stage 3: Readout & constraint projection                     │   │
│  │  - Confidence-weighted ILP: snap to nearest feasible plan    │   │
│  │    s.t. budget, one-hot sizing, radiality                    │   │
│  │  - Optional local bit-swap refinement                        │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │ M feasible investment plans               │
│  ┌──────────────────────▼───────────────────────────────────────┐   │
│  │ Stage 4: Classical subproblem & cut generation               │   │
│  │  - Per plan × per scenario: DC-SCOPF (or SOCP-AC)            │   │
│  │  - N-1 feasibility check                                     │   │
│  │  - Feasibility cut (infeasible) or optimality cut            │   │
│  │  - Rank by true cost; DPO update on decoder                  │   │
│  │  - Fold cuts → Ising; apply R2; repeat                       │   │
│  └──────────────────────┬───────────────────────────────────────┘   │
│                         │ Updated Ising + cut pool                  │
│           ──────────────┘ (back to Stage 1)                        │
└─────────────────────────────────────────────────────────────────────┘
                         │ after max_iters or convergence
                         ▼
               Pareto frontier of investment plans
               + Classical MISOCP baseline (separately timed)
```

---

## The three non-negotiable correctness rules

### R1 — Cut handling is heuristic

Cuts enter the Ising as soft penalty terms.  The quantum master is a trained
generative sampler, not an exact solver.  The tanh relaxation further softens
cuts.  Therefore the convergence guarantee of classical Benders decomposition
**does NOT apply**.  The loop is a cut-guided heuristic that samples diverse
high-quality candidates — never claim Benders optimality in any user-facing
text, comment, or docstring.

**In code:** every comment mentioning cuts uses the phrase
"heuristic penalty" or "cut-guided sampling".

### R2 — Coefficient rescaling + cut-pool cap

Accumulated cuts inflate the dynamic range of Ising coefficients.  PCE's tanh
loss degrades when coefficients span orders of magnitude.

**Before every Stage 2 pass:**
1. Renormalize Ising coefficients to [-1, 1] (dividing by max absolute value).
2. Keep only the K most recent cuts from the pool (configurable as
   `cuts.pool_cap_K`).

**Tested by:** `tests/test_ising.py::TestR2Rescaling`.

### R3 — Timing separation

The amortization claim of the generative master applies **only** to the
master-proposal step.  The classical subproblem is paid every iteration and
does NOT amortize.

**Every benchmark or timing output reports THREE separate numbers:**
- `master_time_s` — encoder inference + decoder sampling + readout.
- `subproblem_time_s` — SCOPF across all scenarios.
- `baseline_time_s` — the MISOCP comparator (run once, separately).

These numbers must **never** be summed into a single "quantum time".

---

## What the quantum layer does and does not do

### What it DOES

- Search the exponential discrete investment space (which buses, what sizes).
- Propose diverse candidate plans in a single forward pass (amortized inference).
- Condition on cut information without retraining (the DPO reference policy is
  frozen at initialization; only the policy model is updated).
- Produce M candidates per iteration, enabling portfolio-style evaluation.

### What it does NOT do

- Solve AC power flow (the classical subproblem handles physics).
- Guarantee feasibility (the projection ILP enforces hard constraints).
- Guarantee optimality (it is a heuristic sampler).
- Beat the classical baseline on a single static instance — its plausible edge
  is **amortized multi-scenario re-proposal** and **candidate diversity**, not
  raw speedup on any single problem instance.

### Known empirical risks

- At small qubit counts (n ≈ 5–7 for k=2, 14-bus case), PCE operates below
  the regime where it was originally validated.  The tanh sharpness α is small
  in this regime and readout fidelity must be measured empirically.
  Tagged: `# TODO(scientific-review)` in `qgridx/quantum/pce.py`.

- The DPO-through-circuit-scores loop has not been validated at the scale of
  real power systems.  The first implementation is a draft.
  Tagged: `# TODO(scientific-review)` in `qgridx/model/dpo.py`.

---

## Component interfaces and extension points

Each stage is behind an ABC; new components are registered with a decorator:

| Stage | ABC | Registry key | Built-in options |
|-------|-----|--------------|-----------------|
| Ising encoder | `IsingEncoder` | `"encoder"` | `log_binary_sizing` |
| Quantum master | `QuantumMasterBase` | `"quantum_master"` | `pce_gqe`, `random_baseline` |
| Backend | `Backend` | `"backend"` | `cpu_sim`, `gpu_sim`, `hardware` |
| Projector | `FeasibilityProjector` | `"projector"` | `confidence_ilp` |
| Subproblem | `SubproblemBase` | `"subproblem"` | `dc_scopf`, `socp_ac` |
| Baseline | — | `"baseline"` | `milp_misocp` |

See `docs/usage.md` for the step-by-step extension guide.
