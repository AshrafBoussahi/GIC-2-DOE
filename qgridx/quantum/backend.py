"""Quantum backend abstraction for qGridX.

Three backends are provided:
  - ``cpu_sim``:  Pure-numpy statevector simulation (always available).
  - ``gpu_sim``:  CUDA-Q GPU simulation (requires cuda-quantum package + GPU).
  - ``hardware``: Stub that raises NotImplementedError with guidance.

All backends share the :class:`Backend` ABC, so they are swappable without
modifying pipeline code.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from qgridx.config import BackendConfig, BackendName
from qgridx.quantum.ansatz import GateOp
from qgridx.registry import register_component


class Backend(ABC):
    """Abstract quantum backend."""

    @abstractmethod
    def run_circuit(
        self,
        gates: list[GateOp],
        n_qubits: int,
        pauli_strings: list[tuple[str, ...]],
        shots: int,
    ) -> np.ndarray:
        """Execute a circuit and estimate Pauli string expectation values.

        Args:
            gates:         List of gate operations (from ansatz builder).
            n_qubits:      Number of qubits.
            pauli_strings: List of m Pauli strings (each a tuple of n labels).
            shots:         Number of measurement shots.

        Returns:
            Array of shape (m,) — estimated <P_i> for each Pauli string.
        """


# ---------------------------------------------------------------------------
# CPU statevector simulator (numpy)
# ---------------------------------------------------------------------------

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

_PAULI_MAT = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _kron_n(mats: list[np.ndarray]) -> np.ndarray:
    """Tensor product of a list of 2x2 matrices -> 2^n x 2^n."""
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _full_op(single_qubit_mat: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a single-qubit gate into the n-qubit Hilbert space."""
    ops = [_I] * n
    ops[qubit] = single_qubit_mat
    return _kron_n(ops)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _cx(n: int, ctrl: int, tgt: int) -> np.ndarray:
    """Build the 2^n x 2^n CNOT matrix."""
    dim = 2 ** n
    mat = np.eye(dim, dtype=np.complex128)
    for i in range(dim):
        bits = format(i, f"0{n}b")
        if bits[ctrl] == "1":
            j_bits = list(bits)
            j_bits[tgt] = "1" if bits[tgt] == "0" else "0"
            j = int("".join(j_bits), 2)
            mat[i, i] = 0
            mat[i, j] = 1
    return mat


def cpu_simulate_circuit(
    gates: list[GateOp],
    n: int,
    pauli_strings: list[tuple[str, ...]],
) -> np.ndarray:
    """Simulate a circuit with a numpy statevector and return <P_i> estimates.

    Args:
        gates:         List of GateOp objects.
        n:             Number of qubits.
        pauli_strings: m Pauli strings.

    Returns:
        Array of shape (m,) — expectation values in [-1, 1].
    """
    dim = 2 ** n
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0  # |0...0>

    # Apply gates
    for g in gates:
        if g.name == "Ry":
            q = g.qubits[0]
            op = _full_op(_ry(g.params[0]), q, n)
            state = op @ state
        elif g.name == "CX":
            ctrl, tgt = g.qubits
            op = _cx(n, ctrl, tgt)
            state = op @ state
        elif g.name == "H":
            q = g.qubits[0]
            op = _full_op(_H_mat, q, n)
            state = op @ state
        # Other gates can be added here

    # Compute expectation values <psi|P|psi>
    correlations = np.zeros(len(pauli_strings), dtype=np.float64)
    for i, ps in enumerate(pauli_strings):
        obs = _kron_n([_PAULI_MAT[p] for p in ps])
        correlations[i] = float(np.real(state.conj() @ (obs @ state)))

    return correlations


@register_component("backend", "cpu_sim")
class CPUSimBackend(Backend):
    """Numpy statevector CPU simulator — no GPU or CUDA-Q required."""

    def __init__(self, shots: int = 2000) -> None:
        self.shots = shots

    def run_circuit(
        self,
        gates: list[GateOp],
        n_qubits: int,
        pauli_strings: list[tuple[str, ...]],
        shots: int | None = None,
    ) -> np.ndarray:
        # shots is ignored for exact statevector simulation; kept for API compatibility
        return cpu_simulate_circuit(gates, n_qubits, pauli_strings)


# ---------------------------------------------------------------------------
# GPU simulator (CUDA-Q)
# ---------------------------------------------------------------------------

@register_component("backend", "gpu_sim")
class GPUSimBackend(Backend):
    """CUDA-Q GPU simulator backend.

    Falls back to the CPU simulator with a warning if CUDA-Q or a GPU is
    unavailable.
    """

    def __init__(self, shots: int = 4000) -> None:
        self.shots = shots
        self._cudaq_available = self._check_cudaq()
        if not self._cudaq_available:
            warnings.warn(
                "CUDA-Q (cuda-quantum) is not installed or no GPU is available. "
                "Falling back to cpu_sim backend.",
                RuntimeWarning,
                stacklevel=2,
            )

    @staticmethod
    def _check_cudaq() -> bool:
        try:
            import cudaq  # type: ignore[import-untyped]
            return True
        except ImportError:
            return False

    def run_circuit(
        self,
        gates: list[GateOp],
        n_qubits: int,
        pauli_strings: list[tuple[str, ...]],
        shots: int | None = None,
    ) -> np.ndarray:
        if not self._cudaq_available:
            return cpu_simulate_circuit(gates, n_qubits, pauli_strings)

        # TODO(scientific-review): compile GateOp list to cudaq.kernel
        # and run with cudaq.observe() or cudaq.sample().
        # This is the primary integration point for GPU execution.
        import cudaq  # type: ignore[import-untyped]
        raise NotImplementedError(
            "CUDA-Q kernel compilation from GateOp list is not yet implemented. "
            "This is a planned extension point. "
            "See qgridx/quantum/backend.py TODO for the integration recipe."
        )


# ---------------------------------------------------------------------------
# Hardware stub
# ---------------------------------------------------------------------------

@register_component("backend", "hardware")
class HardwareBackend(Backend):
    """Stub for real-hardware execution.

    Always raises :exc:`NotImplementedError` with guidance on how to
    implement a hardware backend for a specific QPU provider.
    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    def run_circuit(
        self,
        gates: list[GateOp],
        n_qubits: int,
        pauli_strings: list[tuple[str, ...]],
        shots: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Hardware backend is a stub. To implement real-hardware execution:\n"
            "  1. Subclass Backend and decorate with @register_component('backend', 'my_hw').\n"
            "  2. Compile GateOp list to the target QPU's native gate set.\n"
            "  3. Submit via the QPU provider's SDK (e.g. IonQ, IBM Quantum).\n"
            "  4. Set backend.name: my_hw in your YAML config.\n"
            "See docs/usage.md for the extension guide."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_backend(cfg: BackendConfig) -> Backend:
    """Instantiate a backend from config."""
    from qgridx.registry import build_component
    return build_component("backend", cfg.name.value, shots=cfg.shots)
