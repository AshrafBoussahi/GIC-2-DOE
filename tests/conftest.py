"""Pytest fixtures for qGridX tests."""

from __future__ import annotations

import pytest

from qgridx.problems.base import (
    BudgetSpec,
    CandidatesSpec,
    CostsSpec,
    EncodingSpec,
    ProblemSpec,
    ScenariosSpec,
    SystemSpec,
)


@pytest.fixture
def tiny_problem():
    """Create a tiny 4-bus problem for fast tests."""
    return ProblemSpec(
        system=SystemSpec(name="tiny4", source="custom", base_mva=100.0),
        buses=[
            {"id": 0, "type": "slack", "v_nominal_kv": 110.0, "v_min_pu": 0.95, "v_max_pu": 1.05, "load_mw": 0.0, "load_mvar": 0.0},
            {"id": 1, "type": "pq", "v_nominal_kv": 110.0, "v_min_pu": 0.95, "v_max_pu": 1.05, "load_mw": 10.0, "load_mvar": 3.0},
            {"id": 2, "type": "pq", "v_nominal_kv": 110.0, "v_min_pu": 0.95, "v_max_pu": 1.05, "load_mw": 15.0, "load_mvar": 4.0},
            {"id": 3, "type": "pq", "v_nominal_kv": 110.0, "v_min_pu": 0.95, "v_max_pu": 1.05, "load_mw": 20.0, "load_mvar": 5.0},
        ],
        lines=[
            {"id": 0, "from_bus": 0, "to_bus": 1, "r_pu": 0.01, "x_pu": 0.1, "rate_mva": 50.0},
            {"id": 1, "from_bus": 1, "to_bus": 2, "r_pu": 0.01, "x_pu": 0.1, "rate_mva": 50.0},
            {"id": 2, "from_bus": 2, "to_bus": 3, "r_pu": 0.01, "x_pu": 0.1, "rate_mva": 50.0},
            {"id": 3, "from_bus": 3, "to_bus": 0, "r_pu": 0.01, "x_pu": 0.1, "rate_mva": 50.0},
        ],
        generators=[
            {"id": 0, "bus": 0, "p_max_mw": 100.0, "p_min_mw": 0.0, "cost_per_mwh": 50.0},
        ],
        candidates=CandidatesSpec(
            bess_buses=[1, 2],
            size_levels_mwh=[0.0, 25.0, 50.0],
            microgrid_boundaries=[],
        ),
        costs=CostsSpec(
            bess_capex_per_mwh=200_000.0,
            bess_power_capex_per_mw=150_000.0,
            microgrid_fixed_cost=500_000.0,
        ),
        budget=BudgetSpec(total_capex=5_000_000.0, max_sites=2),
        scenarios=ScenariosSpec(n_weather=2, n_load=2, seed=42),
        encoding=EncodingSpec(sizing_scheme="log_binary", mutex_penalty=10.0, budget_penalty=5.0),
    )


@pytest.fixture
def ieee14_problem():
    """Create IEEE 14-bus problem from pandapower."""
    from qgridx.problems.power_system import load_pandapower_case
    return load_pandapower_case(
        "ieee14",
        candidate_buses=[3, 4, 5, 9, 10, 12, 13, 14],
        size_levels_mwh=[0.0, 25.0, 50.0, 100.0],
    )
