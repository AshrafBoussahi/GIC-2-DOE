"""Tests for quantum backend implementations.

cpu_sim: runs immediately (no GPU needed).
gpu_sim: skipped if CUDA unavailable.
hardware: must raise NotImplementedError with guidance.
"""
from __future__ import annotations

import numpy as np
import pytest

from qgridx.quantum.ansatz import build_brickwork_circuit
from qgridx.quantum.backend import CPUSimBackend, GPUSimBackend, HardwareBackend
from qgridx.quantum.pce import assign_pauli_strings


def _trivial_pauli_strings(n: int) -> list[tuple[str, ...]]:
    """Return n single-qubit Z operators on successive qubits."""
    strings = []
    for q in range(n):
        p = ["I"] * n
        p[q] = "Z"
        strings.append(tuple(p))
    return strings


class TestCPUSimBackend:
    def test_runs_without_error(self):
        n = 3
        backend = CPUSimBackend(shots=100)
        gates, _ = build_brickwork_circuit(n=n, n_layers=2, seed=0)
        ps = _trivial_pauli_strings(n)
        result = backend.run_circuit(gates, n, ps, shots=100)
        assert result.shape == (n,)

    def test_correlations_in_bounds(self):
        n = 4
        backend = CPUSimBackend(shots=100)
        gates, _ = build_brickwork_circuit(n=n, n_layers=2, seed=1)
        ps = assign_pauli_strings(n, n, 1, seed=1)  # 1-body strings
        result = backend.run_circuit(gates, n, ps, shots=100)
        assert all(-1.0 - 1e-9 <= c <= 1.0 + 1e-9 for c in result)

    def test_zero_rotation_gives_known_state(self):
        """With all-zero rotations and no entanglers, state is |0...0>.
        <Z_0> on |0> = +1 (eigenvalue of Z with |0> eigenvector).
        """
        from qgridx.quantum.ansatz import GateOp
        n = 2
        backend = CPUSimBackend()
        gates = [GateOp("Ry", [0], [0.0]), GateOp("Ry", [1], [0.0])]
        ps = [("Z", "I"), ("I", "Z")]
        result = backend.run_circuit(gates, n, ps, shots=100)
        # <Z>|0> = +1 for both qubits
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-9)

    def test_pi_rotation_flips_z(self):
        """Ry(pi)|0> = |1>; <Z>|1> = -1."""
        import math
        from qgridx.quantum.ansatz import GateOp
        n = 1
        backend = CPUSimBackend()
        gates = [GateOp("Ry", [0], [math.pi])]
        ps = [("Z",)]
        result = backend.run_circuit(gates, n, ps, shots=100)
        np.testing.assert_allclose(result, [-1.0], atol=1e-6)

    def test_larger_circuit_runs(self):
        n = 6
        backend = CPUSimBackend(shots=1000)
        gates, _ = build_brickwork_circuit(n=n, n_layers=3, seed=99)
        ps = assign_pauli_strings(6, n, 2, seed=99)
        result = backend.run_circuit(gates, n, ps, shots=1000)
        assert len(result) == 6


class TestGPUSimBackend:
    @pytest.mark.gpu
    def test_gpu_sim_runs(self):
        cudaq = pytest.importorskip("cudaq", reason="CUDA-Q not installed")
        n = 3
        backend = GPUSimBackend(shots=100)
        gates, _ = build_brickwork_circuit(n=n, n_layers=2, seed=0)
        ps = _trivial_pauli_strings(n)
        result = backend.run_circuit(gates, n, ps, shots=100)
        assert result.shape == (n,)

    def test_gpu_sim_falls_back_to_cpu(self):
        """If no CUDA-Q, GPUSimBackend should warn and fall back to CPU."""
        try:
            import cudaq  # type: ignore
            pytest.skip("CUDA-Q is installed; fallback test not applicable")
        except ImportError:
            pass
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            backend = GPUSimBackend(shots=100)
            assert any("cpu_sim" in str(x.message) for x in w)
        n = 3
        gates, _ = build_brickwork_circuit(n=n, n_layers=2, seed=0)
        ps = _trivial_pauli_strings(n)
        result = backend.run_circuit(gates, n, ps, shots=100)
        assert result.shape == (n,)


class TestHardwareBackend:
    def test_raises_not_implemented(self):
        backend = HardwareBackend()
        from qgridx.quantum.ansatz import GateOp
        with pytest.raises(NotImplementedError, match="Subclass Backend"):
            backend.run_circuit([], 2, [("Z", "I")], shots=100)

    def test_error_message_has_guidance(self):
        backend = HardwareBackend()
        try:
            backend.run_circuit([], 2, [("Z", "I")], shots=100)
        except NotImplementedError as e:
            assert "register_component" in str(e) or "Subclass" in str(e)
