"""qGridX command-line interface.

Entry points:
    qgridx run    --config <path>   Run the full pipeline experiment.
    qgridx baseline --config <path> Run only the classical MISOCP baseline.
"""
from __future__ import annotations

import sys

import click

from qgridx.config import load_config
from qgridx.utils.logging import get_logger

log = get_logger("qgridx.cli")


@click.group()
@click.version_option(package_name="qGridX")
def main() -> None:
    """qGridX: hybrid quantum-classical BESS planning."""


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML config file.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def run(config: str, verbose: bool) -> None:
    """Run the full 4-stage quantum-classical pipeline."""
    import logging
    if verbose:
        log.setLevel(logging.DEBUG)

    cfg = load_config(config)
    from qgridx.experiment.runner import run_experiment
    result, metrics = run_experiment(cfg)
    sys.exit(0)


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to the YAML config file.",
)
def baseline(config: str) -> None:
    """Run only the classical MISOCP baseline (no quantum master)."""
    import time
    cfg = load_config(config)
    from qgridx.problems.scenarios import build_scenario_tree
    from qgridx.baselines.milp_misocp import MILPMISOCPBaseline

    scenarios = build_scenario_tree(cfg.problem)
    bl = MILPMISOCPBaseline(cfg=cfg.pipeline.baseline)
    t0 = time.perf_counter()
    plan, cost = bl.solve(cfg.problem, scenarios)
    elapsed = time.perf_counter() - t0

    click.echo(f"\nBaseline solve time (R3): {elapsed:.3f} s")  # R3
    click.echo(f"Baseline total cost:      {cost:.2f}")
    if plan is not None:
        click.echo(f"Build mask: {plan.build_mask}")
        click.echo(f"Size (MWh): {plan.size_mwh}")
    sys.exit(0)
