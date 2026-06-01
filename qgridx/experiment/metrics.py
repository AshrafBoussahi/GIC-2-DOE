"""Experiment metrics collection.

All timing metrics MUST be reported separately (R3): master-proposal time,
classical-subproblem time, and baseline solve time are distinct numbers and
must never be summed into a single "quantum time".
"""
from __future__ import annotations

import dataclasses
from typing import Optional

from qgridx.pipeline import PipelineResult


@dataclasses.dataclass
class ExperimentMetrics:
    """Computed metrics from a completed pipeline run."""
    best_cost_quantum: float
    baseline_cost: float
    cost_gap_pct: float              # (quantum - baseline) / |baseline| * 100
    feasibility_rate: float
    n_iterations: int
    # R3: timing reported separately
    master_time_total_s: float
    subproblem_time_total_s: float
    baseline_time_s: float
    # Per-iteration series (for plots)
    iter_best_costs: list[float]
    iter_mean_costs: list[float]
    iter_feasibility: list[float]
    iter_dpo_losses: list[Optional[float]]
    iter_master_times: list[float]
    iter_subproblem_times: list[float]
    iter_n_cuts: list[int]


def compute_metrics(result: PipelineResult) -> ExperimentMetrics:
    """Extract metrics from a :class:`PipelineResult`.

    Args:
        result: Completed pipeline result.

    Returns:
        :class:`ExperimentMetrics`.
    """
    baseline_cost = result.baseline_cost
    best_cost = result.best_cost
    if abs(baseline_cost) > 1e-9:
        gap = 100.0 * (best_cost - baseline_cost) / abs(baseline_cost)
    else:
        gap = float("nan")

    return ExperimentMetrics(
        best_cost_quantum=best_cost,
        baseline_cost=baseline_cost,
        cost_gap_pct=gap,
        feasibility_rate=result.feasibility_rate,
        n_iterations=len(result.iteration_log),
        master_time_total_s=result.master_time_total_s,
        subproblem_time_total_s=result.subproblem_time_total_s,
        baseline_time_s=result.baseline_time_s,
        iter_best_costs=[r.best_cost for r in result.iteration_log],
        iter_mean_costs=[r.mean_cost for r in result.iteration_log],
        iter_feasibility=[r.feasibility_rate for r in result.iteration_log],
        iter_dpo_losses=[r.dpo_loss for r in result.iteration_log],
        iter_master_times=[r.master_time_s for r in result.iteration_log],
        iter_subproblem_times=[r.subproblem_time_s for r in result.iteration_log],
        iter_n_cuts=[r.n_cuts for r in result.iteration_log],
    )


def print_summary(metrics: ExperimentMetrics) -> None:
    """Print a concise summary table to stdout."""
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Best cost (quantum pipeline): {metrics.best_cost_quantum:>12.2f}")
    print(f"  Baseline cost (MISOCP):       {metrics.baseline_cost:>12.2f}")
    print(f"  Cost gap:                     {metrics.cost_gap_pct:>11.1f}%")
    print(f"  Feasibility rate:             {metrics.feasibility_rate:>11.1%}")
    print(f"  Loop iterations:              {metrics.n_iterations:>12d}")
    print("-" * 60)
    print("  Timing (R3 — reported separately):")
    print(f"    Master-proposal total:    {metrics.master_time_total_s:>10.3f} s")
    print(f"    Subproblem total:         {metrics.subproblem_time_total_s:>10.3f} s")
    print(f"    Baseline solve:           {metrics.baseline_time_s:>10.3f} s")
    print("=" * 60)
