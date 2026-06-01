"""Brickwork ansatz circuit builder.

Builds a PCE-brickwork circuit as a list of abstract gate operations that can
be compiled to CUDA-Q kernels or a CPU simulator.  The brickwork structure
mitigates barren plateaus by keeping circuit depth shallow and local.

Each layer consists of:
  1. A rotation layer: Ry(theta_i) on each qubit.
  2. An entanglement layer: nearest-neighbor CX gates in an alternating pattern.

The number of layers is auto-derived from n when set to "auto".
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from qgridx.config import AnsatzConfig


@dataclasses.dataclass
class GateOp:
    """Abstract gate operation."""
    name: str           # e.g. "Ry", "CX", "H", "Rz"
    qubits: list[int]
    params: list[float] = dataclasses.field(default_factory=list)


def _n_layers_auto(n: int) -> int:
    """Heuristic: O(log n) layers, minimum 2, maximum 8."""
    return max(2, min(8, math.ceil(math.log2(max(n, 2)) + 1)))


def build_brickwork_circuit(
    n: int,
    n_layers: int | str = "auto",
    params: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[list[GateOp], np.ndarray]:
    """Build a brickwork ansatz circuit.

    Args:
        n:        Number of qubits.
        n_layers: Number of alternating rotation+entanglement layers,
                  or "auto" to derive from n.
        params:   Circuit parameters (rotation angles), shape (n_layers * n,).
                  If None, random parameters are drawn from [0, 2*pi).
        seed:     RNG seed for random parameter initialization.

    Returns:
        Tuple of (list of GateOp, params array).
    """
    if n_layers == "auto":
        n_layers = _n_layers_auto(n)
    n_layers = int(n_layers)

    n_params = n_layers * n
    if params is None:
        rng = np.random.default_rng(seed)
        params = rng.uniform(0.0, 2 * math.pi, size=n_params)
    else:
        if len(params) != n_params:
            raise ValueError(
                f"Expected {n_params} parameters, got {len(params)}."
            )
        params = np.asarray(params, dtype=np.float64)

    gates: list[GateOp] = []
    param_idx = 0

    for layer in range(n_layers):
        # Rotation layer
        for q in range(n):
            gates.append(GateOp(name="Ry", qubits=[q], params=[float(params[param_idx])]))
            param_idx += 1

        # Entanglement layer: CX on (0,1), (2,3), ... then (1,2), (3,4), ...
        parity = layer % 2
        for q in range(parity, n - 1, 2):
            gates.append(GateOp(name="CX", qubits=[q, q + 1]))

    return gates, params


def circuit_to_token_sequence(gates: list[GateOp], n: int, n_layers: int) -> list[str]:
    """Convert a gate list to a token sequence for the Transformer decoder.

    Tokens:
        ROT_qi_<angle_bucket> — rotation on qubit i at discretized angle.
        ENT_qi_qj             — CX entangler from qubit i to qubit j.
        END                   — end of circuit.

    Args:
        gates:    List of GateOp objects.
        n:        Number of qubits.
        n_layers: Number of layers (used for sequence length calculation).

    Returns:
        List of token strings.
    """
    N_ANGLE_BUCKETS = 16
    tokens: list[str] = []
    for g in gates:
        if g.name == "Ry":
            q = g.qubits[0]
            angle = g.params[0] % (2 * math.pi)
            bucket = int(angle / (2 * math.pi) * N_ANGLE_BUCKETS)
            tokens.append(f"ROT_q{q}_a{bucket}")
        elif g.name == "CX":
            q0, q1 = g.qubits
            tokens.append(f"ENT_q{q0}_q{q1}")
    tokens.append("END")
    return tokens
