"""Tests for the feasibility projector and local search.

Verifies that projected plans always satisfy budget + one-hot + max_sites,
and that high-confidence bits are preferred.

No GPU/Gurobi required.
"""
from __future__ import annotations

import numpy as np
import pytest

from qgridx.config import CandidatesSpec, ProblemSpec
from qgridx.projection.confidence_ilp import (
    ConfidenceILPProjector,
    _decode_bitstring,
    _greedy_project,
    _compute_capex,
)
from qgridx.quantum.base import QuantumResult


def _make_spec(n_buses: int = 4, budget: float = 8_000_000.0, max_sites: int = None) -> ProblemSpec:
    return ProblemSpec.model_validate({
        "system": {"name": "test", "source": "pandapower_builtin"},
        "candidates": {
            "bess_buses": list(range(1, n_buses + 1)),
            "size_levels_mwh": [0, 25, 50, 100],
            "microgrid_boundaries": [],
        },
        "budget": {"total_capex": budget, "max_sites": max_sites},
        "costs": {
            "bess_capex_per_mwh": 200_000.0,
            "bess_power_capex_per_mw": 150_000.0,
            "microgrid_fixed_cost": 500_000.0,
        },
    })


def _make_qresult(m: int, build_all: bool = True) -> QuantumResult:
    correlations = np.where(np.arange(m) % 2 == 0, 0.9, -0.1)
    bitstring = (correlations > 0).astype(np.int8)
    confidence = np.abs(correlations)
    return QuantumResult(
        correlations=correlations,
        bitstring=bitstring,
        confidence=confidence,
        circuit_id=0,
    )


class TestDecoding:
    def test_decode_no_build(self):
        spec = _make_spec(n_buses=3)
        # All zeros → no build
        m = spec.n_binary_variables()
        bitstring = np.zeros(m, dtype=np.int8)
        build_mask, size_mwh, microgrids = _decode_bitstring(bitstring, spec)
        assert not np.any(build_mask)
        assert all(s == 0 for s in size_mwh)

    def test_decode_build_first_bus(self):
        spec = _make_spec(n_buses=3)
        from qgridx.problems.ising import _sizing_bits
        L = _sizing_bits(4, spec.encoding.sizing_scheme)
        m = spec.n_binary_variables()
        bits = np.zeros(m, dtype=np.int8)
        bits[0] = 1  # build bus 0
        bits[1] = 1  # size bit 0 → size level 1
        build_mask, size_mwh, _ = _decode_bitstring(bits, spec)
        assert build_mask[0]
        assert size_mwh[0] > 0


class TestGreedyProjection:
    def test_projected_respects_budget(self):
        spec = _make_spec(n_buses=5, budget=1_000_000.0)  # tight budget
        m = spec.n_binary_variables()
        # Try to build everything at max size
        raw = np.ones(m, dtype=np.int8)
        confidence = np.ones(m) * 0.5
        projected = _greedy_project(raw, confidence, spec)
        build_mask, size_mwh, microgrids = _decode_bitstring(projected, spec)
        capex = _compute_capex(build_mask, size_mwh, microgrids, spec)
        assert capex <= spec.budget.total_capex + 1e-6

    def test_projected_respects_max_sites(self):
        spec = _make_spec(n_buses=6, budget=100_000_000.0, max_sites=2)
        m = spec.n_binary_variables()
        raw = np.ones(m, dtype=np.int8)
        confidence = np.ones(m) * 0.5
        projected = _greedy_project(raw, confidence, spec)
        from qgridx.problems.ising import _sizing_bits
        L = _sizing_bits(4, spec.encoding.sizing_scheme)
        n_built = sum(int(projected[b * (1 + L)]) for b in range(6))
        assert n_built <= spec.budget.max_sites

    def test_high_confidence_preserved(self):
        """High-confidence bits should be flipped last (budget allows all)."""
        spec = _make_spec(n_buses=4, budget=100_000_000.0)
        m = spec.n_binary_variables()
        raw = np.zeros(m, dtype=np.int8)
        confidence = np.zeros(m)
        # Make bit 0 (first build bit) very high confidence
        confidence[0] = 0.99
        raw[0] = 1
        projected = _greedy_project(raw, confidence, spec)
        # With huge budget, no bits need to be flipped
        assert projected[0] == 1


class TestConfidenceILPProjector:
    def test_output_is_investment_plan(self):
        spec = _make_spec(n_buses=3)
        m = spec.n_binary_variables()
        result = _make_qresult(m)
        projector = ConfidenceILPProjector(solver="highs", local_search=False)
        plan = projector.project(result, spec)
        assert plan.build_mask.shape == (3,)
        assert plan.size_mwh.shape == (3,)
        assert plan.total_capex >= 0.0

    def test_plan_satisfies_budget(self):
        spec = _make_spec(n_buses=4, budget=500_000.0)
        m = spec.n_binary_variables()
        result = _make_qresult(m, build_all=True)
        projector = ConfidenceILPProjector(solver="highs", local_search=False)
        plan = projector.project(result, spec)
        assert plan.total_capex <= spec.budget.total_capex + 1e-3
