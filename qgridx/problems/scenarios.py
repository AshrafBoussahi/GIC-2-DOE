"""Scenario tree builder for qGridX.

Produces a flat list of :class:`Scenario` objects combining weather
realizations, load levels, and N-1 contingencies.  When no real weather
data is provided the realizations are synthesized deterministically from a
random seed (small Gaussian perturbation of base load).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import numpy as np

from qgridx.config import N1ContingencyMode, ProblemSpec


@dataclasses.dataclass
class Scenario:
    """A single operating scenario used as a subproblem instance."""
    id: int
    weather_idx: int
    load_idx: int
    contingency: Optional[str]  # e.g. "line_3" or None
    load_scale: float           # multiplicative factor on base loads
    solar_scale: float          # multiplicative factor on solar generation


def build_scenario_tree(spec: ProblemSpec) -> list[Scenario]:
    """Build the flat list of scenarios from a :class:`ProblemSpec`.

    Args:
        spec: Validated problem specification.

    Returns:
        List of :class:`Scenario` objects.
    """
    rng = np.random.default_rng(spec.scenarios.seed)
    sc = spec.scenarios

    # --- Weather realizations ---
    if sc.weather_data is not None:
        # TODO(scientific-review): implement NSRDB/WIND-style reader
        raise NotImplementedError("External weather data loading is not yet implemented.")

    # Synthesize weather: each realization is a solar_scale in (0.5, 1.0)
    solar_scales = rng.uniform(0.5, 1.0, size=sc.n_weather).tolist()

    # --- Load levels ---
    # Load levels spread around [0.8, 1.1] of base load
    load_scales = np.linspace(0.8, 1.1, sc.n_load).tolist()

    # --- N-1 contingencies ---
    contingency_labels: list[Optional[str]] = [None]  # None = base case

    if sc.n1_contingencies == N1ContingencyMode.auto_from_n1_set:
        line_ids = [ln.id for ln in spec.lines if ln.in_n1_set]
        gen_ids = [g.id for g in spec.generators if g.in_n1_set]
        for lid in line_ids:
            contingency_labels.append(f"line_{lid}")
        for gid in gen_ids:
            contingency_labels.append(f"gen_{gid}")
    elif sc.n1_contingencies == N1ContingencyMode.explicit:
        if sc.explicit_contingencies:
            for c in sc.explicit_contingencies:
                contingency_labels.append(f"{c.type}_{c.element_id}")

    # --- Cross product ---
    scenarios: list[Scenario] = []
    sid = 0
    for w_idx, sol in enumerate(solar_scales):
        for l_idx, load in enumerate(load_scales):
            for cont in contingency_labels:
                scenarios.append(Scenario(
                    id=sid,
                    weather_idx=w_idx,
                    load_idx=l_idx,
                    contingency=cont,
                    load_scale=float(load),
                    solar_scale=float(sol),
                ))
                sid += 1

    return scenarios
