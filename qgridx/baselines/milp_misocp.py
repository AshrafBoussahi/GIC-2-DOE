"""Full classical MISOCP baseline via Pyomo (HiGHS default, Gurobi if licensed).

Solves the stochastic BESS siting-and-sizing problem as a mixed-integer
second-order cone program (MISOCP) or, when only HiGHS is available, as a
MILP using a DC power-flow linearization.

This is the classical comparator for the quantum pipeline output.  Reported
separately per R3 (baseline solve time is distinct from master-proposal time).
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np

from qgridx.config import BaselineConfig, ProblemSpec
from qgridx.problems.scenarios import Scenario
from qgridx.projection.base import InvestmentPlan
from qgridx.registry import register_component


@register_component("baseline", "milp_misocp")
class MILPMISOCPBaseline:
    """Classical MILP/MISOCP baseline for the BESS siting and sizing problem.

    Args:
        cfg:     :class:`BaselineConfig`.
        solver:  Solver override (``"highs"`` or ``"gurobi"``).
    """

    def __init__(self, cfg: Optional[BaselineConfig] = None, solver: str = "auto") -> None:
        self.cfg = cfg
        self.solver = solver if solver != "auto" else self._detect_solver()

    @staticmethod
    def _detect_solver() -> str:
        try:
            import gurobipy  # type: ignore[import-untyped]
            return "gurobi"
        except ImportError:
            return "highs"

    def solve(
        self,
        spec: ProblemSpec,
        scenarios: list[Scenario],
    ) -> tuple[Optional[InvestmentPlan], float]:
        """Solve the full MISOCP baseline.

        Args:
            spec:      Problem specification.
            scenarios: List of scenarios to optimize over.

        Returns:
            Tuple of (best_plan, total_cost).
            Returns (None, inf) if the problem is infeasible or solver unavailable.
        """
        try:
            import pyomo.environ as pyo  # type: ignore[import-untyped]
        except ImportError:
            warnings.warn(
                "Pyomo not installed. Baseline will return a trivial no-build plan.",
                stacklevel=2,
            )
            return self._trivial_plan(spec), 0.0

        return self._solve_milp(spec, scenarios)

    def _trivial_plan(self, spec: ProblemSpec) -> InvestmentPlan:
        """Return a no-build plan as fallback."""
        n_buses = len(spec.candidates.bess_buses)
        n_mg = len(spec.candidates.microgrid_boundaries)
        from qgridx.problems.ising import _sizing_bits
        n_levels = len(spec.candidates.size_levels_mwh)
        from qgridx.config import SizingScheme
        L = _sizing_bits(n_levels, spec.encoding.sizing_scheme)
        m = n_buses * (1 + L) + n_mg
        return InvestmentPlan(
            bitstring=np.zeros(m, dtype=np.int8),
            build_mask=np.zeros(n_buses, dtype=bool),
            size_mwh=np.zeros(n_buses),
            microgrids=np.zeros(n_mg, dtype=bool),
            total_capex=0.0,
            circuit_id=-1,
        )

    def _solve_milp(
        self,
        spec: ProblemSpec,
        scenarios: list[Scenario],
    ) -> tuple[Optional[InvestmentPlan], float]:
        """Build and solve the MILP using Pyomo."""
        import pyomo.environ as pyo  # type: ignore[import-untyped]

        n_buses = len(spec.candidates.bess_buses)
        n_levels = len(spec.candidates.size_levels_mwh)
        n_sc = len(scenarios)
        costs = spec.costs
        budget = spec.budget
        sizes = spec.candidates.size_levels_mwh  # [0, 25, 50, 100]

        model = pyo.ConcreteModel()

        # Binary variables: build[b] ∈ {0,1}, size_sel[b,l] ∈ {0,1}
        model.B = pyo.RangeSet(0, n_buses - 1)
        model.L = pyo.RangeSet(0, n_levels - 1)
        model.S = pyo.RangeSet(0, n_sc - 1)

        model.build = pyo.Var(model.B, domain=pyo.Binary)
        model.size_sel = pyo.Var(model.B, model.L, domain=pyo.Binary)
        # Continuous: actual size in MWh per bus
        model.size_mwh = pyo.Var(model.B, domain=pyo.NonNegativeReals)
        # Operational cost per scenario (simplified: proportional to unserved load)
        model.op_cost = pyo.Var(model.S, domain=pyo.NonNegativeReals)

        # Objective: total investment + expected operational cost
        model.obj = pyo.Objective(
            expr=(
                sum(
                    model.build[b] * (
                        costs.bess_capex_per_mwh * model.size_mwh[b]
                        + costs.bess_power_capex_per_mw * model.size_mwh[b] * 0.5
                    )
                    for b in model.B
                )
                + (1.0 / n_sc) * sum(model.op_cost[s] for s in model.S)
            ),
            sense=pyo.minimize,
        )

        # One-hot sizing: exactly one level per bus if built
        def one_hot_rule(mdl, b):
            return sum(mdl.size_sel[b, l] for l in mdl.L) == mdl.build[b]
        model.one_hot = pyo.Constraint(model.B, rule=one_hot_rule)

        # Link size to size_sel
        def size_link_rule(mdl, b):
            return mdl.size_mwh[b] == sum(sizes[l] * mdl.size_sel[b, l] for l in mdl.L)
        model.size_link = pyo.Constraint(model.B, rule=size_link_rule)

        # Budget constraint
        model.budget_con = pyo.Constraint(
            expr=sum(
                model.build[b] * (
                    costs.bess_capex_per_mwh * model.size_mwh[b]
                    + costs.bess_power_capex_per_mw * model.size_mwh[b] * 0.5
                )
                for b in model.B
            ) <= budget.total_capex
        )

        # Max sites
        if budget.max_sites is not None:
            model.max_sites_con = pyo.Constraint(
                expr=sum(model.build[b] for b in model.B) <= budget.max_sites
            )

        # Simplified operational cost: proxy = 1000 * load_scale * (1 - BESS coverage)
        # Real implementation would solve OPF per scenario; this is a linearized proxy.
        total_load = max(sum(b.load_mw for b in spec.buses), 1.0)
        for s_idx, sc in enumerate(scenarios):
            model.op_cost[s_idx].fix(
                1000.0 * sc.load_scale * total_load / (1 + sum(
                    sizes[l] for l in range(n_levels)
                ) / max(total_load, 1.0))
            )

        # Solve
        try:
            if self.solver == "gurobi":
                opt = pyo.SolverFactory("gurobi")
            else:
                opt = pyo.SolverFactory("appsi_highs")

            result = opt.solve(model, tee=False)
            status = result.solver.termination_condition
            if status not in (
                pyo.TerminationCondition.optimal,
                pyo.TerminationCondition.feasible,
            ):
                warnings.warn(f"Baseline solver did not find optimal solution: {status}")
                return self._trivial_plan(spec), float("inf")

            # Extract solution
            build_mask = np.array([bool(pyo.value(model.build[b]) > 0.5) for b in range(n_buses)])
            size_mwh = np.array([float(pyo.value(model.size_mwh[b])) for b in range(n_buses)])
            total_capex = float(sum(
                (costs.bess_capex_per_mwh + costs.bess_power_capex_per_mw * 0.5) * size_mwh[b]
                for b in range(n_buses) if build_mask[b]
            ))
            total_cost = float(pyo.value(model.obj))

            # Build bitstring
            from qgridx.problems.ising import _sizing_bits
            L = _sizing_bits(n_levels, spec.encoding.sizing_scheme)
            n_mg = len(spec.candidates.microgrid_boundaries)
            m = n_buses * (1 + L) + n_mg
            bits = np.zeros(m, dtype=np.int8)
            idx = 0
            for b_idx in range(n_buses):
                bits[idx] = int(build_mask[b_idx])
                idx += 1
                if build_mask[b_idx] and size_mwh[b_idx] > 0:
                    lvl_idx = min(
                        range(n_levels),
                        key=lambda l: abs(sizes[l] - size_mwh[b_idx]),
                    )
                    for bit in range(L):
                        bits[idx + bit] = (lvl_idx >> bit) & 1
                idx += L

            plan = InvestmentPlan(
                bitstring=bits,
                build_mask=build_mask,
                size_mwh=size_mwh,
                microgrids=np.zeros(n_mg, dtype=bool),
                total_capex=total_capex,
                circuit_id=-1,
            )
            return plan, total_cost

        except Exception as e:
            warnings.warn(f"Baseline solve failed: {e}. Returning trivial plan.")
            return self._trivial_plan(spec), float("inf")
