"""Load pandapower / IEEE test cases into a ProblemSpec.

The primary entry point is :func:`load_problem_spec`. For pandapower built-in
cases (e.g. ``ieee14``), no external files are needed.
"""
from __future__ import annotations

import math
from typing import Any

from qgridx.config import (
    Bus,
    BudgetSpec,
    BusType,
    CandidatesSpec,
    CostsSpec,
    EncodingSpec,
    Generator,
    Line,
    ProblemSpec,
    ScenariosSpec,
    SystemSource,
    SystemSpec,
)


# Map pandapower bus types to our enum
_PP_BUS_TYPE: dict[str, BusType] = {
    "b": BusType.slack,
    "n": BusType.pq,
    "m": BusType.pv,
    # fallback
    "slack": BusType.slack,
    "pq": BusType.pq,
    "pv": BusType.pv,
}


def _pp_bus_type(pp_type: Any) -> BusType:
    return _PP_BUS_TYPE.get(str(pp_type).lower(), BusType.pq)


def _default_candidates(bus_ids: list[int]) -> CandidatesSpec:
    """Pick a sensible default candidate set from a list of PQ/PV bus ids."""
    # Exclude the slack bus; take up to 12 candidate buses
    candidates = bus_ids[:12] if len(bus_ids) > 12 else bus_ids
    return CandidatesSpec(
        bess_buses=[int(b) for b in candidates],
        size_levels_mwh=[0.0, 25.0, 50.0, 100.0],
        microgrid_boundaries=[],
    )


def load_pandapower_builtin(name: str) -> "Any":
    """Return the pandapower network for a built-in case name."""
    try:
        import pandapower.networks as ppn  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pandapower is required. Install it with: pip install pandapower"
        ) from exc

    loaders: dict[str, Any] = {
        "ieee14": ppn.case14,
        "case14": ppn.case14,
        "ieee30": ppn.case30,
        "case30": ppn.case30,
        "ieee57": ppn.case57,
        "case57": ppn.case57,
        "ieee118": ppn.case118,
        "case118": ppn.case118,
        "rtsgmlc": ppn.GBnetwork if hasattr(ppn, "GBnetwork") else None,
    }
    key = name.lower().replace("-", "").replace("_", "")
    loader = loaders.get(key)
    if loader is None:
        raise ValueError(
            f"Unknown pandapower built-in case '{name}'. "
            f"Available: {list(loaders.keys())}"
        )
    return loader()


def pandapower_to_problem_spec(
    net: "Any",
    name: str = "unknown",
    candidates: CandidatesSpec | None = None,
    costs: CostsSpec | None = None,
    budget: BudgetSpec | None = None,
    scenarios: ScenariosSpec | None = None,
    encoding: EncodingSpec | None = None,
) -> ProblemSpec:
    """Convert a pandapower network object to a :class:`ProblemSpec`.

    Sensible defaults are filled in; the caller can override any section.

    Args:
        net:        pandapower network object.
        name:       Human-readable case name.
        candidates: Optional override for candidate bus/sizing spec.
        costs:      Optional CAPEX cost override.
        budget:     Optional budget override.
        scenarios:  Optional scenario-tree override.
        encoding:   Optional encoding override.

    Returns:
        Validated :class:`ProblemSpec`.
    """
    import pandas as pd  # type: ignore[import-untyped]

    base_mva = float(getattr(net, "sn_mva", 100.0))

    # --- Buses ---
    buses: list[Bus] = []
    pq_pv_ids: list[int] = []
    for idx, row in net.bus.iterrows():
        btype_str = str(row.get("type", "n"))
        btype = _pp_bus_type(btype_str)
        vn = float(row.get("vn_kv", 110.0))
        vmin = float(row.get("min_vm_pu", 0.94))
        vmax = float(row.get("max_vm_pu", 1.06))
        # aggregate load for this bus
        load_mw = 0.0
        load_mvar = 0.0
        if not net.load.empty:
            bus_loads = net.load[net.load.bus == idx]
            load_mw = float(bus_loads["p_mw"].sum()) if "p_mw" in bus_loads else 0.0
            load_mvar = float(bus_loads["q_mvar"].sum()) if "q_mvar" in bus_loads else 0.0
        buses.append(Bus(
            id=int(idx),
            type=btype,
            v_nominal_kv=vn,
            v_min_pu=vmin,
            v_max_pu=vmax,
            load_mw=load_mw,
            load_mvar=load_mvar,
        ))
        if btype in (BusType.pq, BusType.pv):
            pq_pv_ids.append(int(idx))

    # --- Lines ---
    lines: list[Line] = []
    for idx, row in net.line.iterrows():
        r_pu = float(row.get("r_ohm_per_km", 0.0)) / base_mva
        x_pu = float(row.get("x_ohm_per_km", 0.0)) / base_mva
        b_pu = float(row.get("c_nf_per_km", 0.0)) / base_mva
        rate = float(row.get("max_i_ka", 1.0)) * vn * math.sqrt(3) if False else 100.0
        rate = float(row.get("max_loading_percent", 100.0))
        lines.append(Line(
            id=int(idx),
            from_bus=int(row["from_bus"]),
            to_bus=int(row["to_bus"]),
            r_pu=r_pu,
            x_pu=x_pu,
            b_pu=b_pu,
            rate_mva=rate,
            in_n1_set=True,
        ))

    # trafo branches
    for idx, row in net.trafo.iterrows():
        hv_bus = int(row["hv_bus"])
        lv_bus = int(row["lv_bus"])
        sn_mva = float(row.get("sn_mva", base_mva))
        vk_percent = float(row.get("vk_percent", 5.0))
        vkr_percent = float(row.get("vkr_percent", 1.0))
        x_pu = vk_percent / 100.0
        r_pu = vkr_percent / 100.0
        lines.append(Line(
            id=10000 + int(idx),
            from_bus=hv_bus,
            to_bus=lv_bus,
            r_pu=r_pu,
            x_pu=x_pu,
            b_pu=0.0,
            rate_mva=sn_mva,
            in_n1_set=True,
        ))

    # --- Generators ---
    generators: list[Generator] = []
    for idx, row in net.gen.iterrows():
        generators.append(Generator(
            id=int(idx),
            bus=int(row["bus"]),
            p_max_mw=float(row.get("max_p_mw", 100.0)),
            p_min_mw=float(row.get("min_p_mw", 0.0)),
            cost_per_mwh=50.0,
            in_n1_set=True,
        ))
    if not net.ext_grid.empty:
        for idx, row in net.ext_grid.iterrows():
            generators.append(Generator(
                id=20000 + int(idx),
                bus=int(row["bus"]),
                p_max_mw=999.0,
                p_min_mw=-999.0,
                cost_per_mwh=0.0,
                in_n1_set=False,
            ))

    # --- Defaults ---
    total_load_mw = sum(b.load_mw for b in buses)
    default_budget = BudgetSpec(
        total_capex=max(8_000_000.0, total_load_mw * 50_000),
        max_sites=None,
    )

    return ProblemSpec(
        system=SystemSpec(name=name, source=SystemSource.pandapower_builtin, base_mva=base_mva),
        buses=buses,
        lines=lines,
        generators=generators,
        candidates=candidates or _default_candidates(pq_pv_ids),
        costs=costs or CostsSpec(),
        budget=budget or default_budget,
        scenarios=scenarios or ScenariosSpec(),
        encoding=encoding or EncodingSpec(),
    )


def load_problem_spec(
    name_or_path: str,
    source: SystemSource = SystemSource.pandapower_builtin,
    **overrides: Any,
) -> ProblemSpec:
    """High-level loader: build a ProblemSpec from a case name or file path.

    Args:
        name_or_path: Case name (e.g. ``"ieee14"``) or path to a file.
        source:       Where to load from (default: pandapower built-in).
        **overrides:  Any :class:`ProblemSpec` field overrides
                      (candidates, costs, budget, scenarios, encoding).

    Returns:
        Validated :class:`ProblemSpec`.
    """
    if source == SystemSource.pandapower_builtin:
        net = load_pandapower_builtin(name_or_path)
        return pandapower_to_problem_spec(net, name=name_or_path, **overrides)
    raise NotImplementedError(
        f"Source '{source}' is not yet implemented. "
        "Supported: pandapower_builtin. Contribute via pull request."
    )
