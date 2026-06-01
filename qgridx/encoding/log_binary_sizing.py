"""Log-binary sizing encoder — the default Ising encoder.

Variables per candidate bus:
  - 1 build/no-build bit.
  - L = ceil(log2(n_levels)) sizing bits encoding the discrete size index.
Plus one bit per microgrid boundary.
"""
from __future__ import annotations

from qgridx.config import ProblemSpec
from qgridx.encoding.base import IsingEncoder
from qgridx.problems.ising import IsingSpec, problem_spec_to_ising
from qgridx.registry import register_component


@register_component("encoder", "log_binary_sizing")
class LogBinarySizingEncoder(IsingEncoder):
    """Default encoder: log-binary sizing bits with mutex and budget penalties."""

    def __init__(self, mutex_penalty: float = 10.0, budget_penalty: float = 5.0) -> None:
        self.mutex_penalty = mutex_penalty
        self.budget_penalty = budget_penalty

    def encode(
        self,
        spec: ProblemSpec,
        cut_pool: list[dict] | None = None,
    ) -> IsingSpec:
        from qgridx.config import CutsConfig
        cuts_cfg = CutsConfig(pool_cap_K=20, rescale="max_abs")
        return problem_spec_to_ising(spec, cut_pool=cut_pool, cuts_cfg=cuts_cfg)
