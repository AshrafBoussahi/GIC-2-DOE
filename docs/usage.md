# qGridX Usage Guide

## Installation

### CPU-only (recommended for development)

```bash
pip install -r requirements_cpu.txt
pip install -e ".[dev]"
```

### GPU + CUDA-Q (for full performance)

Install CUDA-Q first following the official guide at  
https://nvidia.github.io/cuda-quantum/latest/install.html, then:

```bash
pip install -r requirements.txt
pip install -e ".[gpu,dev]"
```

### Gurobi (optional, for faster baseline solves)

Activate your Gurobi license, then:

```bash
pip install -e ".[gurobi]"
```

---

## 60-second quickstart

```python
from qgridx.config import load_config
from qgridx.pipeline import Pipeline

cfg = load_config("configs/ieee14_demo.yaml")
result = Pipeline(cfg).run()

print(f"Best cost (quantum): {result.best_cost:.2f}")
print(f"Baseline cost:       {result.baseline_cost:.2f}")
print(f"Master time (R3):    {result.master_time_total_s:.3f} s")
print(f"Subproblem time (R3):{result.subproblem_time_total_s:.3f} s")
```

Or via the CLI:

```bash
qgridx run --config configs/ieee14_demo.yaml
qgridx baseline --config configs/ieee14_demo.yaml
```

---

## Running the IEEE 14-bus demonstration

```bash
python examples/run_ieee14_demo.py
```

This runs the full pipeline on IEEE 14-bus, saves plots and results to
`./runs/ieee14_demo/`, and prints a summary table.

---

## ProblemSpec schema reference

`ProblemSpec` is the single public input. It can be given as a YAML/JSON dict or
built by the pandapower loader.

```yaml
problem:
  system:
    name: str                          # e.g. "ieee14"
    source: enum {pandapower_builtin, matpower_file, custom}
    path: optional[str]                # if matpower/custom
    base_mva: float                    # default 100.0

  buses: list[Bus]                     # auto-filled by pandapower loader
  lines: list[Line]                    # auto-filled by pandapower loader
  generators: list[Generator]          # auto-filled by pandapower loader

  candidates:
    bess_buses: list[int]              # candidate bus ids for BESS placement
    size_levels_mwh: list[float]       # discrete MWh sizes; index 0 MUST be 0.0
    microgrid_boundaries: list[list[int]]  # candidate islandable bus groups

  costs:
    bess_capex_per_mwh: float          # $ per MWh energy capacity
    bess_power_capex_per_mw: float     # $ per MW power capacity
    microgrid_fixed_cost: float        # $ per activated microgrid zone

  budget:
    total_capex: float                 # total $ available
    max_sites: optional[int]           # maximum number of battery sites

  scenarios:
    n_weather: int                     # number of weather realizations
    n_load: int                        # number of load levels
    n1_contingencies: enum{auto_from_n1_set, explicit}
    explicit_contingencies: optional[list[contingency]]
    weather_data: optional[path]       # if null, synthesized from seed
    seed: int                          # random seed

  encoding:
    sizing_scheme: enum{log_binary, one_hot}  # default log_binary
    mutex_penalty: float               # Ising penalty for size-bit conflicts
    budget_penalty: float              # Ising penalty for budget violations

  meta: dict                           # free-form, forward-compatible
```

### Building from pandapower (zero external files)

```python
from qgridx.problems.power_system import load_problem_spec
spec = load_problem_spec("ieee14")  # uses pandapower built-in
```

---

## Full Config reference

The top-level `Config` wraps a `ProblemSpec` and a `pipeline` block.

```yaml
pipeline:
  encoder:
    name: log_binary_sizing          # registered encoder component
    mutex_penalty: 10.0
    budget_penalty: 5.0

  quantum_master:
    name: pce_gqe                    # or: random_baseline
    pce:
      k: 2                           # k-body Pauli strings
      alpha: auto                    # "auto" or float
      beta: 0.5
      n_qubits: auto                 # "auto" derives from m and k
    ansatz:
      n_layers: auto                 # "auto" or int
      entangle_pattern: brickwork
      topology: linear
    backend:
      name: cpu_sim                  # cpu_sim | gpu_sim | hardware
      shots: 2000
    samples_per_step: 16

  model:
    encoder_gnn:
      hidden: 128
      layers: 3
    decoder:
      n_layer: 4
      n_head: 4
      n_embd: 128
      max_len: 256
    dpo:
      beta: 0.1
      lr: 0.0001
      ref_policy: frozen_init

  projection:
    name: confidence_ilp             # registered projector
    solver: highs
    local_search: true

  subproblem:
    name: dc_scopf                   # or: socp_ac
    ac_relaxation: socp_optional
    solver: highs

  cuts:
    pool_cap_K: 20                   # R2: keep at most K cuts
    rescale: max_abs                 # R2: normalization scheme

  loop:
    max_iters: 30
    convergence_tol: 0.001
    M_circuits: 32

  baseline:
    name: milp_misocp
    solver: auto                     # auto -> Gurobi if licensed, else HiGHS
    enabled: true

  experiment:
    seed: 0
    out_dir: ./runs
    save_plots: true
    log_backend: stdout              # stdout | wandb
    device: cpu                      # cpu | cuda
```

---

## Extending the pipeline

qGridX uses a simple registry: decorate your class with
`@register_component(stage, name)`, subclass the appropriate ABC, and
set the name in your YAML config.  **No changes to pipeline code are needed.**

### Step-by-step: adding a new quantum master

1. Create `my_package/my_qaoa.py`:

```python
from qgridx.quantum.base import QuantumMasterBase, QuantumResult
from qgridx.problems.ising import IsingSpec
from qgridx.registry import register_component

@register_component("quantum_master", "my_qaoa")
class MyQAOAMaster(QuantumMasterBase):
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.seed = seed

    def propose(self, ising: IsingSpec, n_samples: int, context=None):
        # ... your QAOA implementation ...
        return [...]
```

2. Import it before calling `Pipeline` (so the decorator fires):

```python
import my_package.my_qaoa  # registers "my_qaoa"
```

3. Set `quantum_master.name: my_qaoa` in your YAML config.

The same pattern works for: `encoder`, `projector`, `subproblem`, `baseline`, `backend`.

---

## Scaling up

| Config field | 14-bus demo | 30-bus benchmark | 118-bus / RTS-GMLC |
|---|---|---|---|
| `system.name` | `ieee14` | `ieee30` | `case118` |
| `candidates.bess_buses` | 8 buses | 12 buses | 15 buses |
| `quantum_master.backend.name` | `cpu_sim` | `gpu_sim` | `gpu_sim` |
| `quantum_master.pce.k` | 2 | 2 | 3 |
| `loop.max_iters` | 10 | 30 | 50 |
| `loop.M_circuits` | 16 | 32 | 64 |

Use the pre-made configs: `configs/ieee30_benchmark.yaml` and
`configs/rtsgmlc_scale.yaml`.

```bash
qgridx run --config configs/ieee30_benchmark.yaml
```

---

## Correctness rules (summary)

- **R1**: Cuts are heuristic Ising penalties — no Benders optimality is claimed.
- **R2**: Coefficients are rescaled and the cut pool is capped before every encoder pass.
- **R3**: Timing is always reported separately: master-proposal / subproblem / baseline.
