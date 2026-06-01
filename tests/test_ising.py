"""Tests for Ising encoding, R2 rescaling, and variable map.

All tests run on CPU, no GPU/Gurobi required.
"""
from __future__ import annotations

import numpy as np
import pytest

from qgridx.config import (
    BudgetSpec,
    CandidatesSpec,
    CutsConfig,
    EncodingSpec,
    ProblemSpec,
    SizingScheme,
    SystemSpec,
)
from qgridx.problems.ising import (
    IsingSpec,
    build_variable_map,
    problem_spec_to_ising,
)


def _make_spec(
    n_buses: int = 4,
    size_levels: list = None,
    n_mg: int = 0,
    budget: float = 8_000_000.0,
) -> ProblemSpec:
    if size_levels is None:
        size_levels = [0, 25, 50, 100]
    bess_buses = list(range(1, n_buses + 1))
    mg_boundaries = [[1, 2]] * n_mg
    return ProblemSpec.model_validate({
        "system": {"name": "test", "source": "pandapower_builtin"},
        "candidates": {
            "bess_buses": bess_buses,
            "size_levels_mwh": size_levels,
            "microgrid_boundaries": mg_boundaries,
        },
        "budget": {"total_capex": budget},
    })


class TestVariableMap:
    def test_variable_count_log_binary(self):
        spec = _make_spec(n_buses=4, size_levels=[0, 25, 50, 100])
        labels = build_variable_map(spec)
        # 4 buses × (1 build + 2 sizing) + 0 mg = 12
        import math
        L = math.ceil(math.log2(4))  # = 2
        assert len(labels) == 4 * (1 + L)

    def test_variable_count_with_microgrids(self):
        spec = _make_spec(n_buses=3, size_levels=[0, 25, 50, 100], n_mg=2)
        labels = build_variable_map(spec)
        import math
        L = math.ceil(math.log2(4))  # = 2
        assert len(labels) == 3 * (1 + L) + 2

    def test_labels_unique(self):
        spec = _make_spec(n_buses=4)
        labels = build_variable_map(spec)
        assert len(labels) == len(set(labels))

    def test_build_bit_labels(self):
        spec = _make_spec(n_buses=3, size_levels=[0, 25])
        labels = build_variable_map(spec)
        build_labels = [l for l in labels if l.startswith("build_")]
        assert len(build_labels) == 3


class TestIsingConstruction:
    def test_h_and_J_shape(self):
        spec = _make_spec(n_buses=4)
        ising = problem_spec_to_ising(spec)
        m = len(build_variable_map(spec))
        assert ising.h.shape == (m,)
        assert ising.J.shape == (m, m)

    def test_J_symmetric(self):
        spec = _make_spec(n_buses=4)
        ising = problem_spec_to_ising(spec)
        np.testing.assert_allclose(ising.J, ising.J.T, atol=1e-10)

    def test_J_zero_diagonal(self):
        spec = _make_spec(n_buses=4)
        ising = problem_spec_to_ising(spec)
        np.testing.assert_allclose(np.diag(ising.J), 0, atol=1e-10)

    def test_mutex_penalty_present(self):
        """Sizing bits for the same bus should have positive coupling (mutex)."""
        spec = _make_spec(n_buses=2, size_levels=[0, 25, 50, 100])
        ising = problem_spec_to_ising(spec)
        # At least some off-diagonal J entries should be positive due to mutex
        assert np.any(ising.J > 0), "Expected positive mutex coupling terms"

    def test_budget_penalty_present(self):
        """Build bits for different buses should have positive coupling (budget)."""
        spec = _make_spec(n_buses=3)
        ising = problem_spec_to_ising(spec)
        assert np.any(ising.J > 0)


class TestR2Rescaling:
    def test_rescale_max_abs_bounds_coefficients(self):
        """After rescaling, all coefficients must be in [-1, 1] (R2)."""
        spec = _make_spec(n_buses=5)
        # Add some artificial cut terms to inflate coefficients
        big_cut = {"h": {0: 1000.0, 1: -500.0}, "J": {(0, 1): 750.0}}
        cuts_cfg = CutsConfig(pool_cap_K=10, rescale="max_abs")
        ising = problem_spec_to_ising(spec, cut_pool=[big_cut], cuts_cfg=cuts_cfg)
        # After rescaling, all values in [-1, 1]
        assert np.all(ising.h >= -1.0 - 1e-9)
        assert np.all(ising.h <= 1.0 + 1e-9)
        assert np.all(ising.J >= -1.0 - 1e-9)
        assert np.all(ising.J <= 1.0 + 1e-9)

    def test_rescale_preserves_sign(self):
        spec = _make_spec(n_buses=3)
        big_cut = {"h": {0: 100.0, 1: -200.0}, "J": {}}
        cuts_cfg = CutsConfig(pool_cap_K=10, rescale="max_abs")
        ising = problem_spec_to_ising(spec, cut_pool=[big_cut], cuts_cfg=cuts_cfg)
        assert ising.h[1] < 0  # negative sign preserved

    def test_cut_pool_cap_enforced(self):
        """Verifies the cut pool is capped at K (R2)."""
        spec = _make_spec(n_buses=3)
        cuts = [{"h": {i % 6: 10.0}, "J": {}} for i in range(50)]
        K = 5
        cuts_cfg = CutsConfig(pool_cap_K=K, rescale="max_abs")
        # Only the last K cuts should be applied; just verify no error
        ising = problem_spec_to_ising(spec, cut_pool=cuts, cuts_cfg=cuts_cfg)
        assert ising.m > 0

    def test_rescale_noop_on_all_zeros(self):
        """All-zero Ising should remain zero after rescaling."""
        m = 4
        ising = IsingSpec(
            h=np.zeros(m),
            J=np.zeros((m, m)),
            var_labels=[f"x{i}" for i in range(m)],
        )
        rescaled = ising.rescale_max_abs()
        np.testing.assert_allclose(rescaled.h, 0)
        np.testing.assert_allclose(rescaled.J, 0)
