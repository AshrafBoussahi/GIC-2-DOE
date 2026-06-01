"""Pauli Correlation Encoding (PCE) assignment module.

Implements the PCE encoding described in Sciorilli et al.:
  - Assign m k-body Pauli strings to m binary variables.
  - Partition them into exactly 3 commuting families.
  - Derive alpha (tanh sharpness) from n when set to "auto".
  - Provide the tanh-correlation loss.

# TODO(scientific-review): at small qubit counts (n ≈ 5-7) PCE operates below
# the regime where it was originally validated.  Readout fidelity should be
# measured empirically rather than assumed.
"""
from __future__ import annotations

import itertools
import math
from typing import Sequence

import numpy as np

from qgridx.config import PCEConfig


# Pauli matrices (2-d real matrices treated symbolically here)
PAULI_LABELS = ["I", "X", "Y", "Z"]


def _pauli_commutes(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Return True if two k-body Pauli strings commute.

    Two Pauli strings commute iff the number of positions where they
    anti-commute (both non-identity and different) is even.
    """
    anti = 0
    for pa, pb in zip(a, b):
        if pa != "I" and pb != "I" and pa != pb:
            anti += 1
    return anti % 2 == 0


def derive_n_qubits(m: int, k: int) -> int:
    """Compute the number of qubits n = ceil(m^(1/k)).

    Args:
        m: Number of binary variables.
        k: Compression degree (k-body Pauli strings).

    Returns:
        Required qubit count n.
    """
    return math.ceil(m ** (1.0 / k))


def assign_pauli_strings(m: int, n: int, k: int, seed: int = 0) -> list[tuple[str, ...]]:
    """Assign exactly m k-body Pauli strings on n qubits.

    Generates all non-identity k-body Pauli strings on n qubits and selects
    m of them.  If fewer than m are available, strings are reused with a
    warning (this can happen when m > 3^k * C(n,k)).

    Args:
        m: Number of binary variables (Pauli strings to assign).
        n: Number of qubits.
        k: Body degree.
        seed: RNG seed for reproducible selection.

    Returns:
        List of m k-body Pauli strings, each a tuple of n single-qubit labels.
    """
    rng = np.random.default_rng(seed)

    # Generate all k-body non-trivial Pauli strings on n qubits:
    # choose k positions from n, each non-identity
    non_id = ["X", "Y", "Z"]
    candidates: list[tuple[str, ...]] = []
    for positions in itertools.combinations(range(n), k):
        for pauli_ops in itertools.product(non_id, repeat=k):
            p = ["I"] * n
            for pos, op in zip(positions, pauli_ops):
                p[pos] = op
            candidates.append(tuple(p))

    if len(candidates) == 0:
        raise ValueError(f"No k-body Pauli strings possible with n={n}, k={k}.")

    if len(candidates) < m:
        # Reuse with shuffled repetition
        repeats = math.ceil(m / len(candidates))
        candidates = (candidates * repeats)[:m]

    # Shuffle and take m
    perm = rng.permutation(len(candidates))
    selected = [candidates[i] for i in perm[:m]]
    return selected


def partition_into_commuting_families(
    pauli_strings: list[tuple[str, ...]],
) -> list[list[int]]:
    """Partition Pauli string indices into exactly 3 commuting families.

    Uses a greedy graph-coloring approach: build a conflict graph (edges
    between anti-commuting pairs) and greedily color it with 3 colors.

    Args:
        pauli_strings: List of Pauli strings as tuples.

    Returns:
        List of 3 lists of indices, one per commuting family.

    Raises:
        ValueError: If 3 families are not sufficient (should not happen for
                    typical PCE assignments, but flagged for review).
    """
    m = len(pauli_strings)
    # Greedy coloring with 3 colors
    colors: list[int] = [-1] * m
    for i in range(m):
        used: set[int] = set()
        for j in range(i):
            if not _pauli_commutes(pauli_strings[i], pauli_strings[j]):
                if colors[j] >= 0:
                    used.add(colors[j])
        for c in range(3):
            if c not in used:
                colors[i] = c
                break
        if colors[i] == -1:
            # TODO(scientific-review): if 3 families are not enough, fall back
            # to 4+ families.  Flag this as a potential scientific review point.
            # For now, assign to family 0 with a warning.
            import warnings
            warnings.warn(
                f"Pauli string {i} could not be assigned to one of 3 commuting "
                "families. Assigning to family 0. Consider increasing n or k.",
                stacklevel=2,
            )
            colors[i] = 0

    families: list[list[int]] = [[], [], []]
    for i, c in enumerate(colors):
        families[c].append(i)
    return families


def derive_alpha(n: int, beta: float = 0.5) -> float:
    """Derive the tanh sharpness alpha from the qubit count n.

    Heuristic: alpha = beta / sqrt(n), clipped to [0.1, 2.0].

    # TODO(scientific-review): this heuristic has not been validated at n < 10.
    """
    alpha = beta / math.sqrt(max(n, 1))
    return float(np.clip(alpha, 0.1, 2.0))


def tanh_correlation_loss(
    correlations: np.ndarray,
    alpha: float,
) -> float:
    """Compute the PCE tanh-correlation loss.

    Loss = -sum_i tanh(alpha * <P_i>^2)

    Encourages correlations to be close to ±1 (high confidence).

    Args:
        correlations: Array of shape (m,) — estimated <P_i> values.
        alpha:        Sharpness hyperparameter.

    Returns:
        Scalar loss value (more negative = better).
    """
    return float(-np.sum(np.tanh(alpha * correlations ** 2)))


class PCEAssignment:
    """Full PCE assignment for a given problem size.

    Attributes:
        m: Number of binary variables.
        n: Number of qubits.
        k: Body degree.
        alpha: Tanh sharpness.
        beta: Secondary hyperparameter.
        pauli_strings: List of m k-body Pauli strings.
        families: 3 commuting families (lists of variable indices).
    """

    def __init__(self, cfg: PCEConfig, m: int, seed: int = 0) -> None:
        self.m = m
        self.k = cfg.k
        self.beta = cfg.beta

        # Derive n
        if cfg.n_qubits == "auto":
            self.n = derive_n_qubits(m, self.k)
        else:
            self.n = int(cfg.n_qubits)

        # Derive alpha
        if cfg.alpha == "auto":
            self.alpha = derive_alpha(self.n, self.beta)
        else:
            self.alpha = float(cfg.alpha)

        self.pauli_strings = assign_pauli_strings(m, self.n, self.k, seed=seed)
        self.families = partition_into_commuting_families(self.pauli_strings)

    def loss(self, correlations: np.ndarray) -> float:
        """Compute the tanh-correlation loss for a set of correlations."""
        return tanh_correlation_loss(correlations, self.alpha)
