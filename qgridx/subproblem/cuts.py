"""Feasibility and optimality cut generation.

Cuts are heuristic penalty terms added to the Ising Hamiltonian (R1).
They discourage the quantum master from re-proposing infeasible or
expensive investment plans.

R1 note: These are NOT classical Benders cuts and do NOT guarantee
convergence.  They are soft penalties in the Ising objective.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from qgridx.config import CutsConfig
from qgridx.problems.ising import IsingSpec, build_variable_map
from qgridx.config import ProblemSpec
from qgridx.projection.base import InvestmentPlan
from qgridx.subproblem.base import SubproblemResult


def generate_cut(
    plan: InvestmentPlan,
    results: list[SubproblemResult],
    spec: ProblemSpec,
    cut_strength: float = 1.0,
) -> dict[str, Any]:
    """Generate a single cut from subproblem results.

    If ANY scenario is infeasible → feasibility cut (penalize this bitstring).
    If all feasible → optimality cut (penalize proportionally to cost).

    The cut is a dict with keys:
      - ``"h"``: {var_idx: penalty_coefficient}
      - ``"J"``: {(i,j): penalty_coefficient}
      - ``"type"``: "feasibility" or "optimality"
      - ``"cost"``: total true cost

    Args:
        plan:          The investment plan being evaluated.
        results:       Subproblem results for all scenarios.
        spec:          Problem specification.
        cut_strength:  Overall scale of the penalty (tune via config).

    Returns:
        Cut dictionary.
    """
    labels = build_variable_map(spec)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    any_infeasible = any(not r.feasible for r in results)
    total_cost = sum(r.operational_cost for r in results)

    cut_h: dict[int, float] = {}
    cut_J: dict[tuple[int, int], float] = {}

    # Penalize the active build bits of this plan
    build_positions = [
        label_to_idx.get(f"build_b{bus}", -1)
        for bus in spec.candidates.bess_buses
    ]

    if any_infeasible:
        # Feasibility cut: penalize co-activation of the infeasible pattern
        strength = cut_strength * 2.0
        for b_idx, pos in enumerate(build_positions):
            if pos >= 0 and plan.build_mask[b_idx]:
                cut_h[pos] = cut_h.get(pos, 0.0) + strength
        cut_type = "feasibility"
    else:
        # Optimality cut: weaker penalty proportional to cost
        if total_cost > 0:
            strength = cut_strength * min(total_cost / 1e6, 2.0)
        else:
            strength = cut_strength * 0.5
        for b_idx, pos in enumerate(build_positions):
            if pos >= 0 and plan.build_mask[b_idx]:
                cut_h[pos] = cut_h.get(pos, 0.0) + strength
        cut_type = "optimality"

    return {
        "h": cut_h,
        "J": cut_J,
        "type": cut_type,
        "cost": total_cost,
    }


def rank_plans_by_cost(
    plans: list[InvestmentPlan],
    plan_costs: list[float],
) -> list[tuple[InvestmentPlan, float]]:
    """Sort plans by total true cost (ascending).

    Args:
        plans:       List of investment plans.
        plan_costs:  Corresponding total costs (investment + operational).

    Returns:
        List of (plan, cost) tuples sorted from cheapest to most expensive.
    """
    paired = list(zip(plans, plan_costs))
    paired.sort(key=lambda x: x[1])
    return paired
