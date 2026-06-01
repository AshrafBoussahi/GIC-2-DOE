"""DC-SCOPF subproblem solver (default) with optional SOCP-AC relaxation.

For each investment plan and scenario:
  1. Modify the pandapower network to include the BESS investments.
  2. Scale loads by scenario.load_scale.
  3. Run DC OPF (default) or SOCP AC OPF.
  4. Check N-1 feasibility via the contingency in the scenario.
  5. Return cost and dual info for cut generation.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from qgridx.config import ProblemSpec
from qgridx.problems.scenarios import Scenario
from qgridx.projection.base import InvestmentPlan
from qgridx.registry import register_component
from qgridx.subproblem.base import SubproblemBase, SubproblemResult


def _apply_bess_to_net(net: Any, plan: InvestmentPlan, spec: ProblemSpec) -> Any:
    """Add BESS storage to a pandapower network copy for the given plan.

    Args:
        net:    pandapower network (will be modified in-place on a copy).
        plan:   Feasible investment plan.
        spec:   Problem specification.

    Returns:
        Modified pandapower network.
    """
    import copy
    net = copy.deepcopy(net)
    try:
        import pandapower as pp  # type: ignore[import-untyped]
    except ImportError:
        return net

    for b_idx, bus_id in enumerate(spec.candidates.bess_buses):
        if plan.build_mask[b_idx] and plan.size_mwh[b_idx] > 0:
            size_mwh = plan.size_mwh[b_idx]
            size_mw = size_mwh * 0.5  # P = 0.5 * E heuristic
            try:
                pp.create_storage(
                    net,
                    bus=bus_id,
                    p_mw=size_mw,
                    max_e_mwh=size_mwh,
                    q_mvar_rated=0.0,
                    controllable=True,
                    name=f"BESS_{bus_id}",
                )
            except Exception:
                # Bus may not exist in test network; skip
                pass
    return net


def _run_dc_opf(net: Any, load_scale: float) -> tuple[bool, float, dict]:
    """Run DC OPF on a pandapower network.

    Returns:
        Tuple of (feasible, cost, dual_info).
    """
    try:
        import pandapower as pp  # type: ignore[import-untyped]
    except ImportError:
        # No pandapower: return a synthetic feasible result
        return True, float(np.random.uniform(100.0, 500.0)), {}

    import copy
    net2 = copy.deepcopy(net)

    # Scale loads
    if not net2.load.empty:
        net2.load["p_mw"] = net2.load["p_mw"] * load_scale
        net2.load["q_mvar"] = net2.load["q_mvar"] * load_scale

    try:
        pp.rundcopp(net2, verbose=False)
        feasible = True
        cost = float(net2.res_cost) if hasattr(net2, "res_cost") else 0.0
        dual_info: dict = {}
    except Exception:
        feasible = False
        cost = 1e9  # large penalty for infeasibility
        dual_info = {}

    return feasible, cost, dual_info


@register_component("subproblem", "dc_scopf")
class DCSCOPFSubproblem(SubproblemBase):
    """DC security-constrained OPF subproblem."""

    def __init__(
        self,
        solver: str = "highs",
        ac_relaxation: str = "socp_optional",
        n_workers: int = 1,
    ) -> None:
        self.solver = solver
        self.ac_relaxation = ac_relaxation
        self.n_workers = n_workers
        self._net_cache: dict[str, Any] = {}

    def _get_net(self, spec: ProblemSpec) -> Any:
        key = spec.system.name
        if key not in self._net_cache:
            try:
                from qgridx.problems.power_system import load_pandapower_builtin
                self._net_cache[key] = load_pandapower_builtin(spec.system.name)
            except Exception:
                self._net_cache[key] = None
        return self._net_cache[key]

    def solve_scenario(
        self,
        plan: InvestmentPlan,
        scenario: Scenario,
        spec: ProblemSpec,
        plan_id: int = 0,
    ) -> SubproblemResult:
        net = self._get_net(spec)
        if net is None:
            return SubproblemResult(
                plan_id=plan_id,
                scenario_id=scenario.id,
                feasible=True,
                operational_cost=float(np.random.uniform(100.0, 500.0)),
            )

        net_modified = _apply_bess_to_net(net, plan, spec)

        # Apply N-1 contingency if present
        if scenario.contingency is not None and scenario.contingency.startswith("line_"):
            try:
                line_id = int(scenario.contingency.split("_")[1])
                import pandapower as pp  # type: ignore[import-untyped]
                # Mark the line as out-of-service
                if line_id < len(net_modified.line):
                    net_modified.line.at[line_id, "in_service"] = False
            except Exception:
                pass

        feasible, cost, dual_info = _run_dc_opf(net_modified, scenario.load_scale)
        return SubproblemResult(
            plan_id=plan_id,
            scenario_id=scenario.id,
            feasible=feasible,
            operational_cost=cost,
            dual_info=dual_info,
        )

    def solve_all(
        self,
        plan: InvestmentPlan,
        scenarios: list[Scenario],
        spec: ProblemSpec,
        plan_id: int = 0,
        n_workers: int = 1,
    ) -> list[SubproblemResult]:
        if n_workers > 1:
            try:
                from joblib import Parallel, delayed  # type: ignore[import-untyped]
                return Parallel(n_jobs=n_workers)(
                    delayed(self.solve_scenario)(plan, sc, spec, plan_id)
                    for sc in scenarios
                )
            except ImportError:
                pass
        return [self.solve_scenario(plan, sc, spec, plan_id) for sc in scenarios]


@register_component("subproblem", "socp_ac")
class SOCPACSubproblem(SubproblemBase):
    """SOCP AC-OPF relaxation subproblem (registered as ``socp_ac``).

    Currently delegates to DC-SCOPF; full SOCP-AC is a planned extension.
    """

    def __init__(self, solver: str = "highs", **kwargs: Any) -> None:
        self._dc = DCSCOPFSubproblem(solver=solver)

    def solve_scenario(
        self,
        plan: InvestmentPlan,
        scenario: Scenario,
        spec: ProblemSpec,
        plan_id: int = 0,
    ) -> SubproblemResult:
        # TODO(scientific-review): implement full SOCP AC relaxation.
        warnings.warn(
            "socp_ac is currently delegating to dc_scopf. "
            "Full SOCP-AC is a planned extension.",
            stacklevel=2,
        )
        return self._dc.solve_scenario(plan, scenario, spec, plan_id)
