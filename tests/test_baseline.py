"""Tests for the MISOCP classical baseline.

Uses a trivially small 2-bus instance so Pyomo/HiGHS finishes in milliseconds.
Marked @pytest.mark.slow for instances that could be slow; tiny instance is run.
"""
from __future__ import annotations

import pytest

from qgridx.baselines.milp_misocp import MILPMISOCPBaseline
from qgridx.config import ProblemSpec, BaselineConfig
from qgridx.problems.scenarios import build_scenario_tree


def _tiny_spec() -> ProblemSpec:
    return ProblemSpec.model_validate({
        "system": {"name": "test2bus", "source": "pandapower_builtin"},
        "candidates": {
            "bess_buses": [1, 2],
            "size_levels_mwh": [0, 25, 50],
            "microgrid_boundaries": [],
        },
        "budget": {"total_capex": 5_000_000.0, "max_sites": 2},
        "costs": {
            "bess_capex_per_mwh": 200_000.0,
            "bess_power_capex_per_mw": 150_000.0,
            "microgrid_fixed_cost": 500_000.0,
        },
        "scenarios": {"n_weather": 1, "n_load": 1, "seed": 0},
    })


class TestMILPMISOCPBaseline:
    def test_solve_returns_plan_and_cost(self):
        pytest.importorskip("pyomo", reason="Pyomo not installed")
        spec = _tiny_spec()
        scenarios = build_scenario_tree(spec)
        baseline = MILPMISOCPBaseline(solver="highs")
        plan, cost = baseline.solve(spec, scenarios)
        assert plan is not None
        assert cost >= 0.0

    def test_plan_satisfies_budget(self):
        pytest.importorskip("pyomo", reason="Pyomo not installed")
        spec = _tiny_spec()
        scenarios = build_scenario_tree(spec)
        baseline = MILPMISOCPBaseline(solver="highs")
        plan, cost = baseline.solve(spec, scenarios)
        assert plan is not None
        assert plan.total_capex <= spec.budget.total_capex + 1.0

    def test_plan_satisfies_max_sites(self):
        pytest.importorskip("pyomo", reason="Pyomo not installed")
        spec = _tiny_spec()
        scenarios = build_scenario_tree(spec)
        baseline = MILPMISOCPBaseline(solver="highs")
        plan, cost = baseline.solve(spec, scenarios)
        assert plan is not None
        n_built = int(plan.build_mask.sum())
        assert n_built <= 2  # max_sites=2

    def test_trivial_plan_fallback_on_no_pyomo(self, monkeypatch):
        """Without pyomo, baseline returns trivial (no-build) plan."""
        import sys
        monkeypatch.setitem(sys.modules, "pyomo", None)
        monkeypatch.setitem(sys.modules, "pyomo.environ", None)
        spec = _tiny_spec()
        scenarios = build_scenario_tree(spec)
        baseline = MILPMISOCPBaseline(solver="highs")
        plan, cost = baseline.solve(spec, scenarios)
        assert plan is not None
        assert plan.total_capex == 0.0

    @pytest.mark.slow
    def test_ieee14_baseline_feasible(self):
        """Full IEEE-14 baseline — slow, skipped by default."""
        pytest.importorskip("pandapower", reason="pandapower not installed")
        pytest.importorskip("pyomo", reason="Pyomo not installed")
        from qgridx.problems.power_system import load_problem_spec
        spec = load_problem_spec("ieee14")
        scenarios = build_scenario_tree(spec)
        baseline = MILPMISOCPBaseline(solver="highs")
        plan, cost = baseline.solve(spec, scenarios)
        assert plan is not None
        assert plan.total_capex <= spec.budget.total_capex + 1.0
