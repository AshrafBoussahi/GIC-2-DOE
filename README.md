<div align="center">

# qGridX

### *Pauli Correlation Encoding is All You Need!*

**A Hybrid Quantum-Classical Codebase for Strategic Power Grid Storage Planning**

---

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Challenge](https://img.shields.io/badge/DOE_GIC_2026-Phase_2-8B5CF6)](https://aqora.io)
[![Stage](https://img.shields.io/badge/status-Phase_2_Submission-orange)]()
[![arXiv PCE](https://img.shields.io/badge/arXiv-2501.06241-b31b1b)](https://arxiv.org/abs/2501.06241)
[![arXiv cGQE](https://img.shields.io/badge/arXiv-2411.03555-b31b1b)](https://arxiv.org/abs/2411.03555)

<br/>

**Team eQoSystem** &nbsp;·&nbsp; *DOE Office of Technology Commercialization — GIC 2026*

| Achraf Boussahi | Abir Chekroun | Zakaria Lourghi |
|:---:|:---:|:---:|
| Team Lead & Quantum Expert | Quantum Expert / Commercial Lead | Software Engineer / AI & Data Science |
| [@AchrafBoussahi](https://github.com/AshrafBoussahi) | @Abeer | @ZakariaLer |

</div>

---

## What is qGridX?

**qGridX** is the Phase 2 proof-of-concept codebase for Team eQoSystem's submission to the U.S. Department of Energy Global Industry Challenge 2026 — *Quantum-Enhanced Strategic Siting of Energy Storage and Microgrids for the Era of AI and Industrial Load Expansion*.

The rapid buildout of AI data centres, semiconductor fabs, and industrial electrification is stressing power grids in ways planners have never seen before. **Where to place battery storage, and how large to make it, is a hard combinatorial problem** — one that classical tools linearize and heuristically decompose, and one that quantum computing is now capable enough to attack seriously.

We present **PCE-GQE**: the **Pauli-Correlation-Encoded Conditional Generative Quantum Eigensolver** — a novel hybrid architecture that places quantum where it belongs (combinatorial search over the exponential investment space) and keeps classical tools where they excel (AC power-flow physics and constraint enforcement). The result is a **generative model trained once that proposes circuit-encoded siting plans for any new planning scenario in a single forward pass** — no retraining, no per-instance variational optimization.

> **This repository is our Phase 2 deliverable**: complete theory, a production-quality codebase, and validated proof-of-concept results. Phase 3 is where we run at scale on real quantum hardware.

---

## The Core Innovation: Three Ideas, One Architecture

| Component | Paper | What it contributes |
|-----------|-------|---------------------|
| **Pauli Correlation Encoding (PCE)** | [Sciorilli et al., arXiv:2501.06241](https://arxiv.org/abs/2501.06241) | Maps *m* binary variables onto **n ≈ m^(1/k) qubits** via *k*-body Pauli correlations. For IEEE-14 (m=60): **only 7 qubits needed**. |
| **Conditional-GQE (cGQE)** | [Pantianagul et al., arXiv:2411.03555](https://arxiv.org/abs/2411.03555) | A Transformer decoder conditioned on a GNN representation of the problem Ising graph. Generates M valid brickwork circuits **without per-instance optimization**. |
| **DPO Preference Training** | [Rafailov et al., NeurIPS 2023](https://arxiv.org/abs/2305.18290) | Updates the generator using circuit *rankings*, not gradients through quantum hardware. **Works natively on real shot-based QPUs.** |

Combined, these produce a system that is:
- **Qubit-efficient**: 5–7 qubits for a 14-bus grid; sub-linear scaling with problem size via PCE compression
- **Amortised**: train once, propose plans for any new scenario in milliseconds
- **Hardware-agnostic**: CPU statevector → NVIDIA cuQuantum → IBM Heron r1 / QuEra Aquila, by config change
- **Honest**: the quantum layer handles combinatorics; it does not claim to solve AC power flow

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │           CUT-GUIDED AMORTISED SAMPLING LOOP  (Heuristic, R1)       │
 │                                                                     │
 │  ┌──────────────────────────────────────────────────────────────┐   │
 │  │ STAGE 1 — Classical Preprocessing                            │   │
 │  │  · Load IEEE-14/30/118 or RTS-GMLC via pandapower            │   │
 │  │  · Build scenario tree: weather × load × N-1 contingencies   │   │
 │  │  · DC-OPF sensitivity ranking prunes candidate bus set        │   │
 │  │  · Output: Ising H = Σ hᵢσᵢ + Σ Jᵢⱼσᵢσⱼ   ← R2: rescale   │   │
 │  └───────────────────────────┬──────────────────────────────────┘   │
 │                              │                                      │
 │  ┌───────────────────────────▼──────────────────────────────────┐   │
 │  │ STAGE 2 — Quantum Master: PCE-GQE            [NOVEL]         │   │
 │  │                                                              │   │
 │  │  GNN Encoder ──────────► context vector                      │   │
 │  │       │                                                      │   │
 │  │  PCE Assignment: m vars ──► n ≈ m^(1/k) qubits               │   │
 │  │       │            3 commuting families → only 3 bases       │   │
 │  │  Transformer Decoder (brickwork grammar mask)                │   │
 │  │       │                                                      │   │
 │  │  M circuits ──► CUDA-Q backend (cpu / gpu / QPU)             │   │
 │  │       │                                                      │   │
 │  │  PCE Readout:  xᵢ = sign(⟨Pᵢ⟩),   wᵢ = |⟨Pᵢ⟩|             │   │
 │  └───────────────────────────┬──────────────────────────────────┘   │
 │                              │ (bitstrings + confidence weights)    │
 │  ┌───────────────────────────▼──────────────────────────────────┐   │
 │  │ STAGE 3 — Confidence-Weighted Projection ILP                 │   │
 │  │  min Σ wᵢ|x'ᵢ − xᵢ|   s.t. budget, one-hot sizing, radiality │   │
 │  │  High-confidence bits preserved; low-confidence bits flip.   │   │
 │  └───────────────────────────┬──────────────────────────────────┘   │
 │                              │ M feasible investment plans          │
 │  ┌───────────────────────────▼──────────────────────────────────┐   │
 │  │ STAGE 4 — Classical Subproblem + Cut Generation              │   │
 │  │  · DC-SCOPF (default) or SOCP-AC per scenario in parallel    │   │
 │  │  · N-1 feasibility check → feasibility / optimality cuts     │   │
 │  │  · DPO update: prefer circuits that produced lower cost      │   │
 │  │  · Cut injection + R2 rescaling → next Ising                 │   │
 │  └───────────────────────────┬──────────────────────────────────┘   │
 │                              └──────────────────► repeat            │
 └─────────────────────────────────────────────────────────────────────┘
                               ↓  convergence / max_iters
              Best plan + Pareto frontier  vs  Classical MISOCP baseline
                         (timings always reported separately, R3)
```

> **Three non-negotiable correctness rules wired into the codebase** — R1 (cuts are heuristic Ising penalties, no Benders claim), R2 (coefficient rescaling + cut-pool cap before every quantum pass), R3 (master / subproblem / baseline timings always reported as separate numbers).

---

## Proof-of-Concept Results

All tests ran on CPU using an exact statevector simulator. Full reproducible code and hardcoded outputs are in [`notebooks/proof_of_concept.ipynb`](notebooks/proof_of_concept.ipynb).

### Test 1 — PCE-GQE vs Classical Solvers on IEEE-14

> *IEEE-14 bus system stressed by 2.6× AI/industrial load growth at buses [8, 9, 12, 13].*
> *m = 6 candidate buses, budget = 3 units, PCE: k=2, n=4 qubits, 3 brickwork layers, 60 DPO epochs.*

| Method | Cost | Grid Overload | Sites | Placement | Runtime |
|--------|------|--------------|-------|-----------|---------|
| **Brute force** *(global optimum)* | **2.3775** | 0.2487 p.u. | 3 | buses [8, 10, 13] | 0.01 s |
| Simulated annealing | 2.3775 | 0.2487 p.u. | 3 | buses [8, 10, 13] | 0.94 s |
| **PCE-GQE** *(ours)* | **2.3775** | 0.2487 p.u. | 3 | buses [8, 10, 13] | 20.12 s |

**PCE-GQE achieved 0% optimality gap**, matching the brute-force global optimum exactly.

Baseline overload *without* any storage: **1.5384 p.u.** → with the optimal plan: **0.2487 p.u.** — an **84% reduction in grid thermal violations**.

### Test 2 — Full Architecture Validation (m=60, n=7, untrained)

> *Full IEEE-14 BESS siting+sizing problem: m=60 binary variables (site-or-not + discrete sizing buckets). PCE k=2 → n=7 qubits. Model: 2.45M parameters. 32 circuits per pass.*

```
[PCE]   m=60 variables → n=7 qubits via k=2 Pauli strings
        Commuting families: Z=21, X=21, Y=18  (3 bases regardless of m)
        Pauli embedding shape: [60, 10]                              ✓

[Model] Total parameters: 2,447,010
        Forward pass time: 160.6 ms                                 ✓

[Tests] Candidates generated: 32 / 32
        All 32 solutions budget-feasible after projection:          ✓
        Energy range: [-637.03, -630.94]  variance > 0 (diversity)  ✓
        Solutions improved by local search: 17 / 32                 ✓
        DPO loss computed without NaN:                              ✓

        ✅  Architecture is correct. Next step: full training.
```

> *An untrained model is not expected to beat classical baselines — the validation confirms that every component (PCE encoding, brickwork generation, feasibility projection, DPO preference update) is functionally correct and numerically stable. Solution quality comes from training, which is Phase 3.*

---

## Repository Structure

```
qGridX/
├── README.md                       ← You are here
├── LICENSE / NOTICE                ← MIT + attribution (gqco, PCE, cGQE papers)
├── pyproject.toml                  ← pip-installable; extras: [gpu], [gurobi], [dev]
├── requirements.txt / _cpu.txt
│
├── qgridx/                         ← Core Python package
│   ├── config.py                   ← Pydantic typed config; load/validate YAML
│   ├── registry.py                 ← @register_component decorator + factory
│   ├── pipeline.py                 ← 4-stage loop orchestrator  Pipeline(cfg).run()
│   ├── cli.py                      ← qgridx run / qgridx baseline
│   ├── problems/                   ← ProblemSpec schema · pandapower loader · Ising builder
│   ├── encoding/                   ← Log-binary sizing encoder (swappable)
│   ├── quantum/                    ← PCE · brickwork ansatz · backends · readout
│   │   ├── pce.py                  ← Pauli string assignment, 3 families, tanh loss
│   │   ├── ansatz.py               ← Brickwork circuit builder (GateOp abstraction)
│   │   ├── backend.py              ← cpu_sim (numpy) · gpu_sim (CUDA-Q) · hardware stub
│   │   └── readout.py              ← xᵢ = sign(⟨Pᵢ⟩), confidence weights
│   ├── model/                      ← GNN encoder · Transformer decoder · tokenizer · DPO
│   ├── projection/                 ← Confidence-weighted ILP projector + local search
│   ├── subproblem/                 ← DC-SCOPF · SOCP-AC · cut generation
│   ├── baselines/                  ← Full MILP/MISOCP via Pyomo + HiGHS / Gurobi
│   ├── experiment/                 ← Metrics · 7 required plots · experiment runner
│   └── utils/                      ← Seeding · logging · I/O
│
├── configs/
│   ├── ieee14_demo.yaml            ← CPU demo (~6 qubits, 10 iters, minutes on laptop)
│   ├── ieee30_benchmark.yaml       ← GPU benchmark: 30-bus, 30 iters
│   └── rtsgmlc_scale.yaml          ← Scale-up: 118-bus, k=3, Gurobi
│
├── notebooks/
│   └── proof_of_concept.ipynb     ← ⭐ Validated proof-of-concept with results
│
├── examples/
│   └── run_ieee14_demo.py          ← End-to-end demo (run by user, not CI)
│
├── tests/                          ← pytest suite (fast CPU only; no GPU/Gurobi needed)
│   ├── test_schema.py              ← ProblemSpec validation + pandapower loader
│   ├── test_ising.py               ← Ising construction + R2 rescaling bounds
│   ├── test_pce.py                 ← Pauli assignment, 3 families, readout round-trip
│   ├── test_projection.py          ← ILP projection feasibility guarantees
│   ├── test_backend.py             ← CPU sim correctness + GPU/hardware stubs
│   ├── test_baseline.py            ← MISOCP on tiny instance
│   └── test_pipeline_smoke.py      ← 4-bus, 2-iter end-to-end smoke test
│
└── docs/
    ├── architecture.md             ← 4-stage loop, 3 correctness rules, honest scope
    └── usage.md                    ← Install, config reference, extension guide
```

---

## Installation

### CPU-only (development / notebook demo)

```bash
git clone https://github.com/AshrafBoussahi/GIC-2-DOE.git
cd GIC-2-DOE
pip install -r requirements_cpu.txt
pip install -e ".[dev]"
```

### GPU + CUDA-Q (Phase 3 target)

```bash
# 1. Install CUDA-Q: https://nvidia.github.io/cuda-quantum/latest/install.html
pip install -r requirements.txt
pip install -e ".[gpu,dev]"
```

### Verify

```bash
python -c "import qgridx; print('qGridX', qgridx.__version__, '✓')"
pytest tests/ -m "not slow and not gpu and not gurobi" -q
```

---

## Quick Start

```python
from qgridx.config import load_config
from qgridx.pipeline import Pipeline

cfg = load_config("configs/ieee14_demo.yaml")
result = Pipeline(cfg).run()

print(f"Best plan — build at:  {result.best_plan.build_mask.tolist()}")
print(f"Capacity per bus (MWh): {result.best_plan.size_mwh.tolist()}")
print(f"Total cost:             ${result.best_cost:,.0f}")
print(f"vs MISOCP baseline:     ${result.baseline_cost:,.0f}")
# R3 — timings always reported separately:
print(f"Master-proposal time:   {result.master_time_total_s:.2f} s")
print(f"Subproblem time:        {result.subproblem_time_total_s:.2f} s")
print(f"Baseline time:          {result.baseline_time_s:.2f} s")
```

Or via the CLI:

```bash
qgridx run      --config configs/ieee14_demo.yaml
qgridx baseline --config configs/ieee14_demo.yaml
```

---

## Why PCE? Why cGQE? Why DPO?

### The qubit problem

A 14-bus problem with 8 candidate buses and 4 size levels has m=26 binary variables. Standard QAOA/VQE needs one qubit per variable → 26 qubits. At 118-bus scale, this becomes 90+ qubits — well beyond practical NISQ reach.

**PCE solves this**: with k=2, n = ⌈√m⌉. For m=26, n=6. For m=96, n=10. Larger grids get *more* compression, not less.

### The per-instance retraining problem

Classical variational methods (QAOA, VQE) require a new optimization run for every new problem instance. As the scenario tree grows, this becomes prohibitive.

**cGQE solves this**: train on a distribution of problem instances; at inference time, condition the decoder on the new problem's Ising graph and generate circuits in milliseconds. The amortisation is the genuine quantum advantage claim.

### The hardware gradient problem

Gradients through a quantum circuit require the parameter-shift rule (2P circuit evaluations per step) — expensive and unavailable on real hardware with shot noise.

**DPO solves this**: rank circuit outputs by their true cost; preference signals flow through the decoder's *classical* token probabilities. No gradients ever pass through quantum hardware.

---

## Phase 3 Roadmap

Phase 2 establishes **theory + codebase + architectural validity**. Phase 3 runs the experiments.

| Milestone | Description | Target Platform |
|-----------|-------------|----------------|
| **Full training** | Train cGQE-PCE decoder on IEEE-14/30 scenario datasets | NVIDIA GPU (qBraid Lab) |
| **IBM Heron r1 validation** | Execute PCE circuits (5–7 qubits, depth 2–4, 3×1024 shots) with Fire Opal error mitigation | IBM Heron r1 via qBraid |
| **Scale to IEEE-118** | m≈96 → k=3 → n≈10 qubits; full DC-SCOPF + parallel scenarios | GPU + Gurobi |
| **SOCP-AC physics** | Upgrade subproblem to full SOCP-AC (PandaPower + Pyomo) | CPU / GPU |
| **Neutral-atom encoding** | Map siting graph to MIS on Rydberg array | QuEra Aquila via qBraid |
| **k-sweep benchmark** | Sweep k ∈ {2, 3, 4}: characterise compression vs quality trade-off | All platforms |
| **AI load overlay** | Add 50–500 MW synthetic data centre loads to IEEE-118 candidate buses | — |
| **Full comparison** | PCE-GQE vs QAOA vs MIS analog vs Gurobi MILP on equal footing | All platforms |

**Phase 3 qubit estimates (PCE compression):**

| Grid | m | k | n qubits | Circuit depth | Platform |
|------|---|---|----------|--------------|----------|
| IEEE-14 | 26 | 2 | 6 | 4–6 | IBM Heron r1 / IonQ Forte |
| IEEE-30 | 54 | 2 | 8 | 4–8 | IBM Heron r2 |
| IEEE-118 | 96 | 3 | 5 | 4–6 | IBM Heron r1 |
| RTS-GMLC | 120+ | 3 | 6 | 4–8 | IBM Heron r2 / IonQ Forte |

As k increases, **larger grids fit on the same or fewer qubits** — PCE compression grows with problem size.

---

## Extending the Pipeline

qGridX is designed to be extended without touching pipeline code:

```python
# Add a new quantum master (e.g. QAOA for comparison):
from qgridx.quantum.base import QuantumMasterBase
from qgridx.registry import register_component

@register_component("quantum_master", "my_qaoa")
class MyQAOAMaster(QuantumMasterBase):
    def propose(self, ising, n_samples, context=None):
        ...  # your QAOA implementation

# then in YAML: quantum_master.name: my_qaoa
```

The same pattern works for: `encoder`, `projector`, `subproblem`, `baseline`, `backend`. See [`docs/usage.md`](docs/usage.md) for the full extension guide.

---

## Stakeholder Impact

| Stakeholder | How qGridX helps |
|-------------|-----------------|
| **Transmission planners** | Explores exponentially more siting configurations than MILP in the same budget via amortised inference |
| **ISOs / RTOs** | Native scenario-awareness: N-1 and weather uncertainty condition the quantum master, not a post-hoc scenario reduction |
| **DOE / state energy offices** | Defensible quantum benchmark on public test systems; honest about present capability and scaling path |
| **AI data centre operators** | Siting decisions that explicitly account for the large-load interconnection stress |
| **Community / tribal microgrids** | Radiality constraints built into the projection ILP; resilience is a first-class objective |

---

## References

```bibtex
@article{sciorilli2025pce,
  title   = {Towards large-scale quantum optimization using a hybrid approach
             based on Pauli Correlation Encoding},
  author  = {Sciorilli, Marco and others},
  journal = {arXiv:2501.06241},
  year    = {2025}
}

@article{pantianagul2024cgqe,
  title   = {Conditional Generative Quantum Eigensolver},
  author  = {Pantianagul, Kanisorn and others},
  journal = {arXiv:2411.03555},
  year    = {2024}
}

@inproceedings{rafailov2023dpo,
  title     = {Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author    = {Rafailov, Rafael and others},
  booktitle = {NeurIPS},
  year      = {2023}
}

@article{minami2024gqe,
  title   = {Generative Quantum Eigensolver (GQE) and its application to ground state search},
  author  = {Minami, Seishi and others},
  journal = {arXiv:2401.09253},
  year    = {2024}
}
```

**Attribution**: Encoder-decoder architecture and circuit execution patterns adapted from [`gqco`](https://github.com/shunyaist/generative-quantum-combinatorial-optimization) (MIT License). See [`NOTICE`](NOTICE).

---

## License

[MIT](LICENSE) — free to use, modify, and build on with attribution.

---

<div align="center">
<sub>Built for the DOE GIC 2026 &nbsp;·&nbsp; Team eQoSystem &nbsp;·&nbsp; Phase 2 Submission</sub>
</div>
