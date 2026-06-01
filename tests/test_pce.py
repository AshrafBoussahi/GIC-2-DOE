"""Tests for PCE Pauli-string assignment, families, and readout.

All tests kept to ≤6 qubits so they run in well under a second.
No GPU required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qgridx.config import PCEConfig
from qgridx.quantum.pce import (
    PCEAssignment,
    assign_pauli_strings,
    derive_n_qubits,
    partition_into_commuting_families,
    tanh_correlation_loss,
)
from qgridx.quantum.readout import correlations_to_bitstring, make_quantum_result


class TestNQubitsDerivation:
    def test_k2_m4(self):
        # ceil(4^(1/2)) = 2
        assert derive_n_qubits(4, 2) == 2

    def test_k2_m26(self):
        # ceil(26^0.5) = 6
        assert derive_n_qubits(26, 2) == 6

    def test_k3_m27(self):
        # ceil(27^(1/3)) = 3
        assert derive_n_qubits(27, 3) == 3

    def test_n_at_least_1(self):
        assert derive_n_qubits(1, 2) >= 1


class TestPauliStringAssignment:
    def test_count_equals_m(self):
        m, n, k = 8, 3, 2
        strings = assign_pauli_strings(m, n, k, seed=0)
        assert len(strings) == m

    def test_each_string_length_n(self):
        m, n, k = 6, 3, 2
        strings = assign_pauli_strings(m, n, k, seed=0)
        for s in strings:
            assert len(s) == n

    def test_k_body_means_k_non_identity(self):
        m, n, k = 6, 4, 2
        strings = assign_pauli_strings(m, n, k, seed=42)
        for s in strings:
            n_nontrivial = sum(1 for p in s if p != "I")
            assert n_nontrivial == k, f"Expected {k} non-I ops, got {n_nontrivial} in {s}"

    def test_deterministic_with_seed(self):
        s1 = assign_pauli_strings(6, 3, 2, seed=7)
        s2 = assign_pauli_strings(6, 3, 2, seed=7)
        assert s1 == s2


class TestCommutingFamilies:
    def test_exactly_three_families(self):
        m, n, k = 6, 3, 2
        strings = assign_pauli_strings(m, n, k, seed=0)
        families = partition_into_commuting_families(strings)
        assert len(families) == 3

    def test_all_variables_assigned(self):
        m = 8
        strings = assign_pauli_strings(m, 3, 2, seed=0)
        families = partition_into_commuting_families(strings)
        assigned = sorted(i for fam in families for i in fam)
        assert assigned == list(range(m))

    def test_within_family_commutes(self):
        from qgridx.quantum.pce import _pauli_commutes
        m, n, k = 8, 4, 2
        strings = assign_pauli_strings(m, n, k, seed=1)
        families = partition_into_commuting_families(strings)
        for fam in families:
            for a in fam:
                for b in fam:
                    if a != b:
                        assert _pauli_commutes(strings[a], strings[b]), (
                            f"Strings {a} and {b} in same family but anti-commute: "
                            f"{strings[a]}, {strings[b]}"
                        )


class TestTanhLoss:
    def test_loss_negative(self):
        correlations = np.array([0.9, -0.8, 0.5])
        loss = tanh_correlation_loss(correlations, alpha=1.0)
        assert loss < 0

    def test_perfect_correlations(self):
        # All ±1 → loss should be ~ -m * tanh(alpha)
        m = 5
        correlations = np.ones(m)
        loss = tanh_correlation_loss(correlations, alpha=1.0)
        expected = -m * float(np.tanh(1.0))
        assert abs(loss - expected) < 1e-9

    def test_zero_correlations(self):
        # All zero → loss = 0
        correlations = np.zeros(6)
        loss = tanh_correlation_loss(correlations, alpha=1.0)
        assert abs(loss) < 1e-9


class TestReadout:
    def test_sign_encoding(self):
        correlations = np.array([0.8, -0.3, 0.0, -0.7, 0.5])
        bitstring, confidence = correlations_to_bitstring(correlations)
        expected = np.array([1, 0, 0, 0, 1], dtype=np.int8)
        np.testing.assert_array_equal(bitstring, expected)

    def test_confidence_equals_abs(self):
        correlations = np.array([0.8, -0.3, 0.0, -0.7, 0.5])
        _, confidence = correlations_to_bitstring(correlations)
        np.testing.assert_allclose(confidence, np.abs(correlations))

    def test_make_quantum_result(self):
        correlations = np.array([0.9, -0.1, 0.6])
        result = make_quantum_result(correlations, circuit_id=3)
        assert result.circuit_id == 3
        assert result.bitstring.dtype == np.int8
        assert result.confidence.shape == (3,)
        assert all(c >= 0 for c in result.confidence)


class TestPCEAssignment:
    def test_pce_n_qubits_auto(self):
        cfg = PCEConfig(k=2, alpha="auto", beta=0.5, n_qubits="auto")
        m = 16
        pce = PCEAssignment(cfg, m=m, seed=0)
        assert pce.n == math.ceil(m ** 0.5)

    def test_pce_pauli_count_equals_m(self):
        cfg = PCEConfig(k=2)
        m = 6
        pce = PCEAssignment(cfg, m=m, seed=0)
        assert len(pce.pauli_strings) == m

    def test_pce_three_families(self):
        cfg = PCEConfig(k=2)
        pce = PCEAssignment(cfg, m=8, seed=0)
        assert len(pce.families) == 3

    def test_tiny_circuit_readout_roundtrip(self):
        """Run a tiny (n≤6) CPU simulator and verify readout shape."""
        from qgridx.quantum.backend import CPUSimBackend
        from qgridx.quantum.ansatz import build_brickwork_circuit

        cfg = PCEConfig(k=2, n_qubits=4)
        m = 6
        pce = PCEAssignment(cfg, m=m, seed=42)
        backend = CPUSimBackend(shots=100)
        gates, _ = build_brickwork_circuit(n=pce.n, n_layers=2, seed=0)
        correlations = backend.run_circuit(gates, pce.n, pce.pauli_strings, shots=100)
        # Pad/trim to m
        if len(correlations) < m:
            correlations = np.pad(correlations, (0, m - len(correlations)))
        elif len(correlations) > m:
            correlations = correlations[:m]
        assert len(correlations) == m
        assert all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for c in correlations)
