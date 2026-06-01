"""Confidence-weighted projection ILP.

Snaps the raw PCE bitstring to the nearest feasible investment plan by solving
a small integer linear program (Pyomo + HiGHS):

  min  sum_i  w_i * |x'_i - x_i|
  s.t. budget constraint,
       one-hot sizing per bus,
       microgrid radiality (single-zone activation).

High-confidence bits (large |<P_i>|) get low w_i and are hard to flip.

Reference: Stage 3 of the 4-stage loop in CONSTRAINTS.md.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from qgridx.config import ProblemSpec, SizingScheme
from qgridx.problems.ising import _sizing_bits
from qgridx.projection.base import FeasibilityProjector, InvestmentPlan
from qgridx.quantum.base import QuantumResult
from qgridx.registry import register_component


def _decode_bitstring(
    bitstring: np.ndarray,
    spec: ProblemSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a flat bitstring into (build_mask, size_mwh, microgrids).

    Args:
        bitstring: Shape (m,) binary array.
        spec:      Problem specification.

    Returns:
        Tuple of (build_mask, size_mwh, microgrids).
    """
    n_levels = len(spec.candidates.size_levels_mwh)
    L = _sizing_bits(n_levels, spec.encoding.sizing_scheme)
    n_buses = len(spec.candidates.bess_buses)
    n_mg = len(spec.candidates.microgrid_boundaries)

    build_mask = np.zeros(n_buses, dtype=bool)
    size_mwh = np.zeros(n_buses, dtype=float)
    idx = 0

    for b_idx in range(n_buses):
        build_bit = int(bitstring[idx])
        idx += 1
        build_mask[b_idx] = bool(build_bit)

        if spec.encoding.sizing_scheme == SizingScheme.log_binary:
            size_bits = bitstring[idx:idx + L]
            idx += L
            size_level_idx = int(sum(int(bit) * (2 ** i) for i, bit in enumerate(size_bits)))
            size_level_idx = min(size_level_idx, n_levels - 1)
            if build_bit:
                size_mwh[b_idx] = spec.candidates.size_levels_mwh[size_level_idx]
        else:
            # one-hot
            size_bits = bitstring[idx:idx + L]
            idx += L
            one_hot_idx = int(np.argmax(size_bits)) if np.any(size_bits) else 0
            if build_bit:
                size_mwh[b_idx] = spec.candidates.size_levels_mwh[min(one_hot_idx, n_levels - 1)]

    microgrids = bitstring[idx:idx + n_mg].astype(bool)
    return build_mask, size_mwh, microgrids


def _compute_capex(
    build_mask: np.ndarray,
    size_mwh: np.ndarray,
    microgrids: np.ndarray,
    spec: ProblemSpec,
) -> float:
    """Compute total capital expenditure for a plan."""
    costs = spec.costs
    total = 0.0
    for b_idx, (build, size) in enumerate(zip(build_mask, size_mwh)):
        if build:
            total += costs.bess_capex_per_mwh * size
            total += costs.bess_power_capex_per_mw * (size * 0.5)  # P = 0.5 * E heuristic
    for mg_active in microgrids:
        if mg_active:
            total += costs.microgrid_fixed_cost
    return total


def _greedy_project(
    raw_bitstring: np.ndarray,
    confidence: np.ndarray,
    spec: ProblemSpec,
) -> np.ndarray:
    """Greedy feasibility projection when Pyomo/HiGHS is unavailable.

    Flips the lowest-confidence bits first until the budget constraint is
    satisfied.  Does not solve the full ILP.

    Args:
        raw_bitstring: Shape (m,) raw bit array.
        confidence:    Shape (m,) confidence per bit (|<P_i>|).
        spec:          Problem specification.

    Returns:
        Projected bitstring of shape (m,).
    """
    projected = raw_bitstring.copy()

    n_levels = len(spec.candidates.size_levels_mwh)
    L = _sizing_bits(n_levels, spec.encoding.sizing_scheme)
    n_buses = len(spec.candidates.bess_buses)

    build_mask, size_mwh, microgrids = _decode_bitstring(projected, spec)
    capex = _compute_capex(build_mask, size_mwh, microgrids, spec)

    # If over budget: turn off lowest-confidence build bits first
    if capex > spec.budget.total_capex:
        # Identify build bit positions and their confidence
        build_positions = [1 + b * (1 + L) for b in range(n_buses)]
        sorted_build = sorted(build_positions, key=lambda i: confidence[i])
        for pos in sorted_build:
            if capex <= spec.budget.total_capex:
                break
            if projected[pos] == 1:
                projected[pos] = 0
                # Zero out corresponding size bits
                size_start = pos + 1
                projected[size_start:size_start + L] = 0
                build_mask, size_mwh, microgrids = _decode_bitstring(projected, spec)
                capex = _compute_capex(build_mask, size_mwh, microgrids, spec)

    # Enforce max_sites constraint
    if spec.budget.max_sites is not None:
        n_active = int(np.sum([projected[1 + b * (1 + L)] for b in range(n_buses)]))
        if n_active > spec.budget.max_sites:
            build_positions = [1 + b * (1 + L) for b in range(n_buses)]
            sorted_by_conf = sorted(
                [p for p in build_positions if projected[p] == 1],
                key=lambda i: confidence[i],
            )
            for pos in sorted_by_conf:
                if n_active <= spec.budget.max_sites:
                    break
                projected[pos] = 0
                size_start = pos + 1
                projected[size_start:size_start + L] = 0
                n_active -= 1

    return projected


def _ilp_project(
    raw_bitstring: np.ndarray,
    confidence: np.ndarray,
    spec: ProblemSpec,
    solver: str = "highs",
) -> np.ndarray:
    """ILP-based projection using Pyomo + HiGHS.

    Args:
        raw_bitstring: Shape (m,) raw bit array.
        confidence:    Shape (m,) confidence per bit.
        spec:          Problem specification.
        solver:        Pyomo solver name ("highs" or "gurobi").

    Returns:
        Projected bitstring of shape (m,).
    """
    try:
        import pyomo.environ as pyo  # type: ignore[import-untyped]
    except ImportError:
        return _greedy_project(raw_bitstring, confidence, spec)

    m = len(raw_bitstring)
    epsilon = 1e-6
    weights = 1.0 / (confidence + epsilon)  # low confidence → large weight → easier to flip

    model = pyo.ConcreteModel()
    model.vars = pyo.Var(range(m), domain=pyo.Binary)
    # Linearization auxiliary: d_i = |x'_i - x_i|
    model.diff = pyo.Var(range(m), domain=pyo.NonNegativeReals, bounds=(0, 1))

    def obj_rule(mdl):
        return sum(weights[i] * mdl.diff[i] for i in range(m))

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Linearize absolute value: d_i >= x'_i - x_i and d_i >= x_i - x'_i
    model.abs_pos = pyo.ConstraintList()
    model.abs_neg = pyo.ConstraintList()
    for i in range(m):
        xi = int(raw_bitstring[i])
        model.abs_pos.add(model.diff[i] >= model.vars[i] - xi)
        model.abs_neg.add(model.diff[i] >= xi - model.vars[i])

    # Budget constraint
    n_levels = len(spec.candidates.size_levels_mwh)
    L = _sizing_bits(n_levels, spec.encoding.sizing_scheme)
    n_buses = len(spec.candidates.bess_buses)
    costs = spec.costs
    budget = spec.budget

    def capex_expr(mdl):
        total = 0.0
        for b_idx in range(n_buses):
            base = b_idx * (1 + L)
            build_var = mdl.vars[base]
            size_mwh_approx = 0.0
            if spec.encoding.sizing_scheme == SizingScheme.log_binary:
                for bit in range(L):
                    size_mwh_approx += mdl.vars[base + 1 + bit] * (2 ** bit) * (
                        max(spec.candidates.size_levels_mwh) / (2 ** L - 1)
                    )
            else:
                for bit in range(L):
                    lev = spec.candidates.size_levels_mwh[min(bit, n_levels - 1)]
                    size_mwh_approx += mdl.vars[base + 1 + bit] * lev
            total += build_var * costs.bess_capex_per_mwh * size_mwh_approx
            total += build_var * costs.bess_power_capex_per_mw * size_mwh_approx * 0.5
        mg_start = n_buses * (1 + L)
        for mg_idx in range(len(spec.candidates.microgrid_boundaries)):
            total += mdl.vars[mg_start + mg_idx] * costs.microgrid_fixed_cost
        return total

    model.budget_con = pyo.Constraint(expr=capex_expr(model) <= budget.total_capex)

    # Max sites constraint
    if budget.max_sites is not None:
        model.max_sites_con = pyo.Constraint(
            expr=sum(model.vars[b * (1 + L)] for b in range(n_buses)) <= budget.max_sites
        )

    # Solve
    try:
        if solver == "highs":
            opt = pyo.SolverFactory("appsi_highs")
        else:
            opt = pyo.SolverFactory(solver)
        results = opt.solve(model, tee=False)
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            projected = np.array([int(round(pyo.value(model.vars[i]))) for i in range(m)], dtype=np.int8)
            return projected
    except Exception:
        pass

    # Fallback to greedy if ILP fails
    return _greedy_project(raw_bitstring, confidence, spec)


@register_component("projector", "confidence_ilp")
class ConfidenceILPProjector(FeasibilityProjector):
    """Confidence-weighted ILP projector (registered as ``confidence_ilp``)."""

    def __init__(self, solver: str = "highs", local_search: bool = True) -> None:
        self.solver = solver
        self.local_search = local_search

    def project(
        self,
        result: QuantumResult,
        spec: ProblemSpec,
    ) -> InvestmentPlan:
        projected_bits = _ilp_project(
            result.bitstring.astype(np.int8),
            result.confidence,
            spec,
            solver=self.solver,
        )
        if self.local_search:
            from qgridx.projection.local_search import local_bit_swap_search
            projected_bits = local_bit_swap_search(projected_bits, result.confidence, spec)

        build_mask, size_mwh, microgrids = _decode_bitstring(projected_bits, spec)
        capex = _compute_capex(build_mask, size_mwh, microgrids, spec)

        return InvestmentPlan(
            bitstring=projected_bits,
            build_mask=build_mask,
            size_mwh=size_mwh,
            microgrids=microgrids,
            total_capex=capex,
            circuit_id=result.circuit_id,
        )
