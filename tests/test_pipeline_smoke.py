"""End-to-end smoke test: 4-bus micro-instance, 2 iterations, mocked backend.

Bounded to finish in well under a minute on any CPU.
Produces a results.json and verifies the pipeline ran successfully.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qgridx.config import (
    BackendName,
    Config,
    ProblemSpec,
)


def _make_smoke_config(out_dir: str) -> Config:
    raw = {
        "problem": {
            "system": {"name": "test4bus", "source": "pandapower_builtin"},
            "candidates": {
                "bess_buses": [1, 2, 3, 4],
                "size_levels_mwh": [0, 25, 50],
                "microgrid_boundaries": [],
            },
            "budget": {"total_capex": 10_000_000.0, "max_sites": 2},
            "costs": {
                "bess_capex_per_mwh": 200_000.0,
                "bess_power_capex_per_mw": 150_000.0,
                "microgrid_fixed_cost": 500_000.0,
            },
            "scenarios": {"n_weather": 1, "n_load": 1, "seed": 0},
        },
        "pipeline": {
            "encoder": {"name": "log_binary_sizing", "mutex_penalty": 10.0, "budget_penalty": 5.0},
            "quantum_master": {
                "name": "pce_gqe",
                "pce": {"k": 2, "alpha": "auto", "beta": 0.5, "n_qubits": "auto"},
                "ansatz": {"n_layers": 2, "entangle_pattern": "brickwork", "topology": "linear"},
                "backend": {"name": "cpu_sim", "shots": 100},
                "samples_per_step": 4,
            },
            "model": {
                "encoder_gnn": {"hidden": 16, "layers": 1},
                "decoder": {"n_layer": 1, "n_head": 2, "n_embd": 16, "max_len": 32},
                "dpo": {"beta": 0.1, "lr": 0.001, "ref_policy": "frozen_init"},
            },
            "projection": {"name": "confidence_ilp", "solver": "highs", "local_search": False},
            "subproblem": {"name": "dc_scopf", "solver": "highs"},
            "cuts": {"pool_cap_K": 5, "rescale": "max_abs"},
            "loop": {"max_iters": 2, "convergence_tol": 1e-3, "M_circuits": 4},
            "baseline": {"name": "milp_misocp", "solver": "highs", "enabled": False},
            "experiment": {
                "seed": 0,
                "out_dir": out_dir,
                "save_plots": False,
                "log_backend": "stdout",
                "device": "cpu",
            },
        },
    }
    return Config.model_validate(raw)


class TestPipelineSmoke:
    def test_pipeline_runs_two_iters(self):
        """Full pipeline: 4-bus, 2 iterations, cpu_sim, no GPU/Gurobi."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_smoke_config(out_dir=tmp)
            from qgridx.pipeline import Pipeline
            pipeline = Pipeline(cfg)
            result = pipeline.run()

            # Basic sanity
            assert result is not None
            assert len(result.iteration_log) <= 2
            assert result.best_cost != float("inf") or result.iteration_log == []

    def test_results_json_created(self):
        """results.json must exist after the pipeline completes."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_smoke_config(out_dir=tmp)
            from qgridx.pipeline import Pipeline
            Pipeline(cfg).run()
            assert (Path(tmp) / "results.json").exists()

    def test_feasibility_rate_in_range(self):
        """Feasibility rate must be in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_smoke_config(out_dir=tmp)
            from qgridx.pipeline import Pipeline
            result = Pipeline(cfg).run()
            assert 0.0 <= result.feasibility_rate <= 1.0

    def test_timing_reported_separately(self):
        """R3: master_time and subproblem_time must both be non-negative."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_smoke_config(out_dir=tmp)
            from qgridx.pipeline import Pipeline
            result = Pipeline(cfg).run()
            assert result.master_time_total_s >= 0.0
            assert result.subproblem_time_total_s >= 0.0

    def test_iteration_log_populated(self):
        """Per-iteration log should have entries for each completed iteration."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_smoke_config(out_dir=tmp)
            from qgridx.pipeline import Pipeline
            result = Pipeline(cfg).run()
            assert len(result.iteration_log) >= 1
            for entry in result.iteration_log:
                assert entry.master_time_s >= 0.0
                assert entry.subproblem_time_s >= 0.0
                assert 0.0 <= entry.feasibility_rate <= 1.0
