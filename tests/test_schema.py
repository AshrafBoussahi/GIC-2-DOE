"""Tests for ProblemSpec schema validation and pandapower loader.

All tests run on CPU, no GPU/Gurobi required.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from qgridx.config import (
    BudgetSpec,
    CandidatesSpec,
    CostsSpec,
    EncodingSpec,
    ProblemSpec,
    ScenariosSpec,
    SystemSpec,
)


def _minimal_spec(**overrides) -> dict:
    base = {
        "system": {"name": "ieee14", "source": "pandapower_builtin"},
        "candidates": {
            "bess_buses": [3, 4, 5],
            "size_levels_mwh": [0, 25, 50],
        },
        "budget": {"total_capex": 5_000_000.0},
    }
    base.update(overrides)
    return base


class TestProblemSpecValidation:
    def test_valid_minimal_spec(self):
        spec = ProblemSpec.model_validate(_minimal_spec())
        assert spec.system.name == "ieee14"
        assert len(spec.candidates.bess_buses) == 3

    def test_missing_bess_buses_raises(self):
        raw = _minimal_spec()
        raw["candidates"]["bess_buses"] = []
        with pytest.raises(ValidationError):
            ProblemSpec.model_validate(raw)

    def test_size_levels_first_must_be_zero(self):
        raw = _minimal_spec()
        raw["candidates"]["size_levels_mwh"] = [10, 25, 50]  # first is not 0
        with pytest.raises(ValidationError):
            ProblemSpec.model_validate(raw)

    def test_defaults_filled_in(self):
        spec = ProblemSpec.model_validate(_minimal_spec())
        assert spec.encoding.mutex_penalty == 10.0
        assert spec.encoding.budget_penalty == 5.0
        assert spec.scenarios.n_weather == 3

    def test_extra_fields_forbidden(self):
        raw = _minimal_spec()
        raw["unknown_field"] = "oops"
        with pytest.raises(ValidationError):
            ProblemSpec.model_validate(raw)

    def test_n_binary_variables_log_binary(self):
        # 3 buses, size_levels=[0,25,50] → L=1 bit, m = 3*(1+1) + 0 = 6
        spec = ProblemSpec.model_validate(_minimal_spec())
        m = spec.n_binary_variables()
        assert m == 3 * (1 + 1)  # 3 buses, L=1 sizing bit each

    def test_n_binary_variables_with_microgrids(self):
        raw = _minimal_spec()
        raw["candidates"]["microgrid_boundaries"] = [[3, 4], [5]]
        spec = ProblemSpec.model_validate(raw)
        # 3 buses, L=1 sizing bits, 2 microgrid bits
        m = spec.n_binary_variables()
        assert m == 3 * 2 + 2


class TestPandapowerLoader:
    def test_load_ieee14(self):
        pytest.importorskip("pandapower", reason="pandapower not installed")
        from qgridx.problems.power_system import load_problem_spec
        spec = load_problem_spec("ieee14")
        assert spec.system.name == "ieee14"
        assert len(spec.buses) > 0
        assert len(spec.lines) > 0
        assert len(spec.generators) > 0
        assert len(spec.candidates.bess_buses) > 0

    def test_load_unknown_raises(self):
        pytest.importorskip("pandapower", reason="pandapower not installed")
        from qgridx.problems.power_system import load_problem_spec
        with pytest.raises(ValueError, match="Unknown pandapower built-in case"):
            load_problem_spec("ieee9999")

    def test_problem_spec_valid_after_loader(self):
        pytest.importorskip("pandapower", reason="pandapower not installed")
        from qgridx.problems.power_system import load_problem_spec
        spec = load_problem_spec("ieee14")
        assert spec.budget.total_capex > 0
        assert spec.n_binary_variables() > 0
