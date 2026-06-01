"""Experiment runner: orchestrates a complete experiment from a Config."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from qgridx.config import Config, load_config
from qgridx.experiment.metrics import ExperimentMetrics, compute_metrics, print_summary
from qgridx.experiment.plots import (
    plot_convergence,
    plot_dpo_loss,
    plot_feasibility_rate,
    plot_network_plan,
    plot_timing_breakdown,
)
from qgridx.pipeline import Pipeline, PipelineResult
from qgridx.utils.io import ensure_dir, save_json
from qgridx.utils.logging import get_logger

log = get_logger("qgridx.runner")


def run_experiment(config: Config) -> tuple[PipelineResult, ExperimentMetrics]:
    """Run the full pipeline experiment from a config object.

    Args:
        config: Validated :class:`Config`.

    Returns:
        Tuple of (PipelineResult, ExperimentMetrics).
    """
    out_dir = ensure_dir(config.pipeline.experiment.out_dir)
    save_plots = config.pipeline.experiment.save_plots

    # Run the pipeline
    pipeline = Pipeline(config)
    result = pipeline.run()
    metrics = compute_metrics(result)

    # Save results
    save_json(
        {
            "best_cost_quantum": metrics.best_cost_quantum,
            "baseline_cost": metrics.baseline_cost,
            "cost_gap_pct": metrics.cost_gap_pct,
            "feasibility_rate": metrics.feasibility_rate,
            "n_iterations": metrics.n_iterations,
            "master_time_total_s": metrics.master_time_total_s,
            "subproblem_time_total_s": metrics.subproblem_time_total_s,
            "baseline_time_s": metrics.baseline_time_s,
        },
        out_dir / "results.json",
    )

    # Write report
    _write_report(metrics, result, out_dir, config)

    # Generate plots
    if save_plots:
        _generate_plots(metrics, result, out_dir, config)

    print_summary(metrics)
    return result, metrics


def _write_report(
    metrics: ExperimentMetrics,
    result: PipelineResult,
    out_dir: Path,
    config: Config,
) -> None:
    spec = config.problem
    lines = [
        "# qGridX Experiment Report",
        "",
        f"**System:** {spec.system.name}",
        f"**Binary variables m:** {spec.n_binary_variables()}",
        f"**Scenarios:** {config.pipeline.loop.max_iters} iterations × {config.pipeline.loop.M_circuits} circuits",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best cost (quantum) | {metrics.best_cost_quantum:.2f} |",
        f"| Baseline cost (MISOCP) | {metrics.baseline_cost:.2f} |",
        f"| Cost gap | {metrics.cost_gap_pct:.1f}% |",
        f"| Feasibility rate | {metrics.feasibility_rate:.1%} |",
        f"| Iterations | {metrics.n_iterations} |",
        "",
        "## Timing (R3 — reported separately)",
        "",
        f"| Stage | Time (s) |",
        f"|-------|----------|",
        f"| Master proposal (total) | {metrics.master_time_total_s:.3f} |",
        f"| Subproblem (total) | {metrics.subproblem_time_total_s:.3f} |",
        f"| Baseline solve | {metrics.baseline_time_s:.3f} |",
        "",
        "## Notes",
        "",
        "- R1: Cuts are heuristic Ising penalties — NO Benders optimality claim.",
        "- R2: Coefficients rescaled + cut pool capped before every encoder pass.",
        "- R3: Master and subproblem times are ALWAYS reported separately.",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    log.info(f"Report written to {report_path}")


def _generate_plots(
    metrics: ExperimentMetrics,
    result: PipelineResult,
    out_dir: Path,
    config: Config,
) -> None:
    plot_convergence(
        metrics.iter_best_costs,
        metrics.iter_mean_costs,
        metrics.baseline_cost,
        out_dir,
    )
    plot_feasibility_rate(metrics.iter_feasibility, out_dir)
    plot_dpo_loss(metrics.iter_dpo_losses, out_dir)
    plot_timing_breakdown(
        metrics.master_time_total_s,
        metrics.subproblem_time_total_s,
        metrics.baseline_time_s,
        out_dir,
    )
    if result.best_plan is not None:
        plot_network_plan(result.best_plan, config.problem, out_dir)
