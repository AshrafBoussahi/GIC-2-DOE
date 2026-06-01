"""Optional bit-swap local search within the feasible neighborhood.

After ILP projection we do a greedy local search: for each bit, try flipping
it and accept the flip if (a) feasibility is preserved and (b) the resulting
plan has lower projected cost (i.e. we move toward more-confident bits).
"""
from __future__ import annotations

import numpy as np

from qgridx.config import ProblemSpec
from qgridx.problems.ising import _sizing_bits
from qgridx.projection.confidence_ilp import _decode_bitstring, _compute_capex


def local_bit_swap_search(
    bitstring: np.ndarray,
    confidence: np.ndarray,
    spec: ProblemSpec,
    max_iters: int = 20,
) -> np.ndarray:
    """Greedy bit-swap local search for refinement.

    Iteratively flips the lowest-confidence bits if doing so keeps the plan
    feasible (budget + max_sites) and increases total confidence.

    Args:
        bitstring:  Shape (m,) projected bitstring.
        confidence: Shape (m,) per-bit confidence values.
        spec:       Problem specification.
        max_iters:  Maximum number of flip iterations.

    Returns:
        Refined bitstring of shape (m,).
    """
    best = bitstring.copy()
    best_score = float(np.sum(confidence * best))  # prefer high-confidence bits ON

    for _ in range(max_iters):
        improved = False
        # Sort candidates by ascending confidence (flip uncertain bits first)
        order = np.argsort(confidence)
        for i in order:
            candidate = best.copy()
            candidate[i] = 1 - candidate[i]
            # Check feasibility
            build_mask, size_mwh, microgrids = _decode_bitstring(candidate, spec)
            capex = _compute_capex(build_mask, size_mwh, microgrids, spec)
            if capex > spec.budget.total_capex:
                continue
            n_levels = len(spec.candidates.size_levels_mwh)
            L = _sizing_bits(n_levels, spec.encoding.sizing_scheme)
            n_buses = len(spec.candidates.bess_buses)
            if spec.budget.max_sites is not None:
                n_active = int(sum(int(candidate[b * (1 + L)]) for b in range(n_buses)))
                if n_active > spec.budget.max_sites:
                    continue
            score = float(np.sum(confidence * candidate))
            if score > best_score:
                best = candidate
                best_score = score
                improved = True
                break
        if not improved:
            break

    return best
