"""ProblemSpec → Ising (h_i, J_ij) conversion with R2 rescaling.

The Ising Hamiltonian encodes:
  - Investment objective: sum of CAPEX terms (linear h_i).
  - Budget penalty:       quadratic penalty if total CAPEX exceeds budget (R1).
  - Mutex penalty:        quadratic penalty for selecting > 1 size level per bus.
  - Cut terms:            additional h_i and J_ij from the cut pool (heuristic, R1).

Before every encoder pass R2 is enforced:
  - All coefficients are renormalized by the max-absolute-value.
  - The cut pool is capped at pool_cap_K entries.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from qgridx.config import CutsConfig, EncodingSpec, ProblemSpec, SizingScheme


@dataclasses.dataclass
class IsingSpec:
    """Ising specification used by the quantum master.

    Attributes:
        h:    1-D array of shape (m,) — linear biases.
        J:    2-D array of shape (m, m) — quadratic couplings (symmetric, zero diagonal).
        var_labels: Human-readable label for each binary variable.
        m:    Number of binary variables.
    """
    h: np.ndarray
    J: np.ndarray
    var_labels: list[str]

    @property
    def m(self) -> int:
        return len(self.h)

    def rescale_max_abs(self) -> "IsingSpec":
        """Return a new IsingSpec with coefficients normalized to [-1, 1].

        This implements R2 rescaling.  If all coefficients are zero the
        Ising is returned unchanged.
        """
        all_vals = np.concatenate([self.h.ravel(), self.J.ravel()])
        max_abs = np.max(np.abs(all_vals))
        if max_abs == 0.0:
            return self
        return IsingSpec(
            h=self.h / max_abs,
            J=self.J / max_abs,
            var_labels=self.var_labels,
        )


def _sizing_bits(n_levels: int, scheme: SizingScheme) -> int:
    if scheme == SizingScheme.log_binary:
        return max(1, math.ceil(math.log2(max(n_levels, 2))))
    return n_levels  # one-hot


def build_variable_map(spec: ProblemSpec) -> list[str]:
    """Return an ordered list of variable labels for the Ising.

    Variables are ordered as: for each candidate bus — one build bit + L
    sizing bits — then one bit per microgrid boundary.
    """
    n_levels = len(spec.candidates.size_levels_mwh)
    scheme = spec.encoding.sizing_scheme
    L = _sizing_bits(n_levels, scheme)

    labels: list[str] = []
    for bus in spec.candidates.bess_buses:
        labels.append(f"build_b{bus}")
        for bit in range(L):
            labels.append(f"size_b{bus}_bit{bit}")
    for mg_idx in range(len(spec.candidates.microgrid_boundaries)):
        labels.append(f"microgrid_mg{mg_idx}")
    return labels


def problem_spec_to_ising(
    spec: ProblemSpec,
    cut_pool: list[dict[str, Any]] | None = None,
    cuts_cfg: CutsConfig | None = None,
) -> IsingSpec:
    """Convert a :class:`ProblemSpec` (plus optional cuts) to an :class:`IsingSpec`.

    The cut pool is a list of dicts, each with keys:
      - ``"h"``: dict mapping variable index to linear penalty coefficient.
      - ``"J"``: dict mapping (i,j) tuple to quadratic penalty coefficient.

    Implements R1 (cuts as heuristic penalties) and R2 (rescaling + cap).

    Args:
        spec:      Validated problem specification.
        cut_pool:  List of cut dicts (may be None or empty).
        cuts_cfg:  CutsConfig specifying pool_cap_K and rescale scheme.

    Returns:
        :class:`IsingSpec` with rescaled coefficients.
    """
    labels = build_variable_map(spec)
    m = len(labels)
    h = np.zeros(m, dtype=np.float64)
    J = np.zeros((m, m), dtype=np.float64)

    # Helper: label → index
    label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(labels)}

    n_levels = len(spec.candidates.size_levels_mwh)
    scheme = spec.encoding.sizing_scheme
    L = _sizing_bits(n_levels, scheme)

    costs = spec.costs
    budget = spec.budget
    enc = spec.encoding

    # --- Investment objective (linear terms) ---
    for bus in spec.candidates.bess_buses:
        build_idx = label_to_idx[f"build_b{bus}"]
        # CAPEX contribution for build bit: penalize high-cost builds
        # Simplified: treat as linear cost proportional to max sizing
        max_size_mwh = max(spec.candidates.size_levels_mwh)
        capex_full = (
            costs.bess_capex_per_mwh * max_size_mwh
            + costs.bess_power_capex_per_mw * (max_size_mwh * 0.5)  # P = 0.5 * E heuristic
        )
        # Normalize to budget so coefficients are O(1)
        h[build_idx] += capex_full / budget.total_capex

    for mg_idx in range(len(spec.candidates.microgrid_boundaries)):
        mg_var_idx = label_to_idx[f"microgrid_mg{mg_idx}"]
        h[mg_var_idx] += costs.microgrid_fixed_cost / budget.total_capex

    # --- Mutex penalty: at most one size level per bus (R1: heuristic, Ising form) ---
    mu = enc.mutex_penalty
    for bus in spec.candidates.bess_buses:
        size_idxs = [label_to_idx[f"size_b{bus}_bit{bit}"] for bit in range(L)]
        # Penalize co-activation of sizing bits for log-binary encoding
        # (for one-hot, penalize all pairs being 1 simultaneously)
        for i in range(len(size_idxs)):
            for j in range(i + 1, len(size_idxs)):
                si, sj = size_idxs[i], size_idxs[j]
                J[si, sj] += mu
                J[sj, si] += mu

        # A size bit should only be active if build bit is also active
        build_idx = label_to_idx[f"build_b{bus}"]
        for si in size_idxs:
            J[build_idx, si] -= mu * 0.5  # reward correlation

    # --- Budget penalty: quadratic penalty when sum CAPEX > budget ---
    # Approximate: add positive coupling between all build bits
    bp = enc.budget_penalty
    build_idxs = [label_to_idx[f"build_b{bus}"] for bus in spec.candidates.bess_buses]
    for i in range(len(build_idxs)):
        for j in range(i + 1, len(build_idxs)):
            bi, bj = build_idxs[i], build_idxs[j]
            J[bi, bj] += bp
            J[bj, bi] += bp

    # --- Cut pool (R1: heuristic penalty terms) ---
    if cut_pool:
        # R2: cap the pool
        K = cuts_cfg.pool_cap_K if cuts_cfg else 20
        active_cuts = cut_pool[-K:]  # keep most recent K cuts
        for cut in active_cuts:
            for idx, val in cut.get("h", {}).items():
                h[int(idx)] += float(val)
            for (i, j), val in cut.get("J", {}).items():
                J[int(i), int(j)] += float(val)
                J[int(j), int(i)] += float(val)

    # --- R2: rescale ---
    ising = IsingSpec(h=h, J=J, var_labels=labels)
    if cuts_cfg and cuts_cfg.rescale == "max_abs":
        ising = ising.rescale_max_abs()
    return ising
