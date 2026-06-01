"""PCE readout: 3-basis correlation estimation → raw bitstring.

Given estimates <P_i> for each of the m Pauli strings (obtained from
3 measurement bases, one per commuting family), convert to a raw binary
investment decision vector and compute per-variable confidence weights.

Reference (Sciorilli et al.):
    x_i = sign(<P_i>)
    w_i = 1 / (|<P_i>| + epsilon)   (low confidence when |<P_i>| is small)
"""
from __future__ import annotations

import numpy as np

from qgridx.quantum.base import QuantumResult
from qgridx.quantum.pce import PCEAssignment


def correlations_to_bitstring(
    correlations: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert correlation estimates to a binary vector and confidence weights.

    Args:
        correlations: Array of shape (m,) — <P_i> estimates in [-1, 1].
        epsilon:      Small constant for numerical stability.

    Returns:
        Tuple of (bitstring, confidence):
            - bitstring:  int8 array of shape (m,) with values 0 or 1.
            - confidence: float64 array of shape (m,) — |<P_i>|.
    """
    bitstring = (np.sign(correlations) > 0).astype(np.int8)
    # +0 edge case: assign 0 (no-build)
    bitstring[correlations == 0.0] = 0
    confidence = np.abs(correlations)
    return bitstring, confidence


def make_quantum_result(
    correlations: np.ndarray,
    circuit_id: int,
    epsilon: float = 1e-6,
) -> QuantumResult:
    """Wrap raw correlations into a :class:`QuantumResult`.

    Args:
        correlations: Estimated <P_i> values, shape (m,).
        circuit_id:   Index of this circuit in the current iteration.
        epsilon:      Stability constant.

    Returns:
        :class:`QuantumResult` with bitstring and confidence filled in.
    """
    bitstring, confidence = correlations_to_bitstring(correlations, epsilon)
    return QuantumResult(
        correlations=correlations,
        bitstring=bitstring,
        confidence=confidence,
        circuit_id=circuit_id,
    )
