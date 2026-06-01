"""Pydantic configuration models for qGridX.

All experiment parameters are set here or via a YAML file — nothing is
hardcoded in the pipeline. Load with :func:`load_config`.
"""
from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SystemSource(str, Enum):
    pandapower_builtin = "pandapower_builtin"
    matpower_file = "matpower_file"
    custom = "custom"


class BusType(str, Enum):
    slack = "slack"
    pv = "pv"
    pq = "pq"


class N1ContingencyMode(str, Enum):
    auto_from_n1_set = "auto_from_n1_set"
    explicit = "explicit"


class SizingScheme(str, Enum):
    log_binary = "log_binary"
    one_hot = "one_hot"


class BackendName(str, Enum):
    cpu_sim = "cpu_sim"
    gpu_sim = "gpu_sim"
    hardware = "hardware"


class LogBackend(str, Enum):
    stdout = "stdout"
    wandb = "wandb"


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


# ---------------------------------------------------------------------------
# ProblemSpec sub-models
# ---------------------------------------------------------------------------

class SystemSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    source: SystemSource = SystemSource.pandapower_builtin
    path: Optional[str] = None
    base_mva: float = 100.0


class Bus(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    type: BusType = BusType.pq
    v_nominal_kv: float = 110.0
    v_min_pu: float = 0.94
    v_max_pu: float = 1.06
    load_mw: float = 0.0
    load_mvar: float = 0.0


class Line(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    from_bus: int
    to_bus: int
    r_pu: float = 0.0
    x_pu: float = 0.0
    b_pu: float = 0.0
    rate_mva: float = 100.0
    in_n1_set: bool = True


class Generator(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: int
    bus: int
    p_max_mw: float
    p_min_mw: float = 0.0
    cost_per_mwh: float = 50.0
    in_n1_set: bool = True


class Contingency(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = "line"
    element_id: int


class CandidatesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bess_buses: list[int] = Field(default_factory=list)
    size_levels_mwh: list[float] = Field(default_factory=lambda: [0.0, 25.0, 50.0, 100.0])
    microgrid_boundaries: list[list[int]] = Field(default_factory=list)

    @field_validator("size_levels_mwh")
    @classmethod
    def first_level_zero(cls, v: list[float]) -> list[float]:
        if v and v[0] != 0.0:
            raise ValueError("size_levels_mwh[0] must be 0.0 (meaning 'no battery')")
        return v


class CostsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bess_capex_per_mwh: float = 200_000.0
    bess_power_capex_per_mw: float = 150_000.0
    microgrid_fixed_cost: float = 500_000.0


class BudgetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_capex: float
    max_sites: Optional[int] = None


class ScenariosSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_weather: int = 3
    n_load: int = 2
    n1_contingencies: N1ContingencyMode = N1ContingencyMode.auto_from_n1_set
    explicit_contingencies: Optional[list[Contingency]] = None
    weather_data: Optional[str] = None
    seed: int = 0


class EncodingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sizing_scheme: SizingScheme = SizingScheme.log_binary
    mutex_penalty: float = 10.0
    budget_penalty: float = 5.0


class ProblemSpec(BaseModel):
    """Single public input to qGridX.

    Can be instantiated from a dict/YAML or built by the pandapower loader.
    """
    model_config = ConfigDict(extra="forbid")

    system: SystemSpec
    buses: list[Bus] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
    generators: list[Generator] = Field(default_factory=list)
    candidates: CandidatesSpec = Field(default_factory=CandidatesSpec)
    costs: CostsSpec = Field(default_factory=CostsSpec)
    budget: BudgetSpec = Field(default_factory=lambda: BudgetSpec(total_capex=8_000_000.0))
    scenarios: ScenariosSpec = Field(default_factory=ScenariosSpec)
    encoding: EncodingSpec = Field(default_factory=EncodingSpec)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_candidates_nonempty(self) -> "ProblemSpec":
        if not self.candidates.bess_buses:
            raise ValueError(
                "candidates.bess_buses is empty — provide at least one candidate bus."
            )
        return self

    def n_binary_variables(self) -> int:
        """Total binary variable count m derived from the encoding scheme."""
        n_buses = len(self.candidates.bess_buses)
        n_levels = len(self.candidates.size_levels_mwh)
        if self.encoding.sizing_scheme == SizingScheme.log_binary:
            sizing_bits = max(1, math.ceil(math.log2(max(n_levels, 2))))
        else:
            sizing_bits = n_levels  # one-hot
        build_bits = n_buses  # one build/no-build bit per bus
        sizing_total = n_buses * sizing_bits
        microgrid_bits = len(self.candidates.microgrid_boundaries)
        return build_bits + sizing_total + microgrid_bits


# ---------------------------------------------------------------------------
# Pipeline sub-models
# ---------------------------------------------------------------------------

class PCEConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: int = 2
    alpha: Any = "auto"  # float or "auto"
    beta: float = 0.5
    n_qubits: Any = "auto"  # int or "auto"


class AnsatzConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_layers: Any = "auto"
    entangle_pattern: str = "brickwork"
    topology: str = "linear"


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: BackendName = BackendName.cpu_sim
    shots: int = 2000


class QuantumMasterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "pce_gqe"
    pce: PCEConfig = Field(default_factory=PCEConfig)
    ansatz: AnsatzConfig = Field(default_factory=AnsatzConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)
    samples_per_step: int = 16


class EncoderGNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden: int = 128
    layers: int = 3


class DecoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    max_len: int = 256


class DPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    beta: float = 0.1
    lr: float = 1e-4
    ref_policy: str = "frozen_init"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder_gnn: EncoderGNNConfig = Field(default_factory=EncoderGNNConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    dpo: DPOConfig = Field(default_factory=DPOConfig)


class ProjectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "confidence_ilp"
    solver: str = "highs"
    local_search: bool = True


class SubproblemConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "dc_scopf"
    ac_relaxation: str = "socp_optional"
    solver: str = "highs"


class CutsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pool_cap_K: int = 20  # R2: keep at most K cuts
    rescale: str = "max_abs"  # R2: rescaling scheme


class LoopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_iters: int = 30
    convergence_tol: float = 1e-3
    M_circuits: int = 32


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "milp_misocp"
    solver: str = "auto"
    enabled: bool = True


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 0
    out_dir: str = "./runs"
    save_plots: bool = True
    log_backend: LogBackend = LogBackend.stdout
    device: Device = Device.cpu


class EncoderPipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "log_binary_sizing"
    mutex_penalty: float = 10.0
    budget_penalty: float = 5.0


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder: EncoderPipelineConfig = Field(default_factory=EncoderPipelineConfig)
    quantum_master: QuantumMasterConfig = Field(default_factory=QuantumMasterConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    subproblem: SubproblemConfig = Field(default_factory=SubproblemConfig)
    cuts: CutsConfig = Field(default_factory=CutsConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


class Config(BaseModel):
    """Top-level config object loaded from a YAML file."""
    model_config = ConfigDict(extra="forbid")
    problem: ProblemSpec
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> Config:
    """Load and validate a qGridX YAML config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated :class:`Config` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
