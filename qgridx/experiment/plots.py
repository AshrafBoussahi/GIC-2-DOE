"""All required experiment plots.

Each function saves the figure to *out_dir* and displays it if
*show* is True.  All matplotlib figures are created fresh so they
are safe to call in batch mode (no lingering state).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _savefig(fig: "Any", path: Path, show: bool) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_convergence(
    iter_best: list[float],
    iter_mean: list[float],
    baseline_cost: float,
    out_dir: Path,
    show: bool = False,
) -> None:
    """Convergence curve: best/mean candidate cost vs iteration."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = range(1, len(iter_best) + 1)
    ax.plot(iters, iter_best, "b-o", ms=4, label="Best cost (quantum)")
    ax.plot(iters, iter_mean, "b--", ms=3, alpha=0.6, label="Mean cost")
    ax.axhline(baseline_cost, color="r", ls="--", label="Baseline (MISOCP)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("Convergence curve")
    ax.legend()
    _savefig(fig, out_dir / "convergence.png", show)


def plot_feasibility_rate(
    iter_feasibility: list[float],
    out_dir: Path,
    show: bool = False,
) -> None:
    """Feasibility rate vs iteration."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    iters = range(1, len(iter_feasibility) + 1)
    ax.bar(iters, [f * 100 for f in iter_feasibility], color="steelblue")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Feasibility rate (%)")
    ax.set_title("Feasibility rate of projected candidates")
    _savefig(fig, out_dir / "feasibility_rate.png", show)


def plot_dpo_loss(
    iter_dpo: list[Optional[float]],
    out_dir: Path,
    show: bool = False,
) -> None:
    """DPO training loss vs step."""
    import matplotlib.pyplot as plt
    losses = [l for l in iter_dpo if l is not None]
    if not losses:
        return
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, len(losses) + 1), losses, "g-o", ms=4)
    ax.set_xlabel("DPO step")
    ax.set_ylabel("DPO loss")
    ax.set_title("DPO training loss")
    _savefig(fig, out_dir / "dpo_loss.png", show)


def plot_pce_correlations(
    correlations_per_iter: list[list[float]],
    out_dir: Path,
    show: bool = False,
) -> None:
    """PCE diagnostic: distribution of |<P_i>| over iterations."""
    import matplotlib.pyplot as plt
    n_iters = len(correlations_per_iter)
    if n_iters == 0:
        return
    fig, axes = plt.subplots(1, min(n_iters, 5), figsize=(14, 3), sharey=True)
    if n_iters == 1:
        axes = [axes]
    for i, corrs in enumerate(correlations_per_iter[:5]):
        ax = axes[i]
        ax.hist(np.abs(corrs), bins=20, range=(0, 1), color="purple", alpha=0.7)
        ax.set_title(f"Iter {i+1}")
        ax.set_xlabel("|<P_i>|")
        if i == 0:
            ax.set_ylabel("Count")
    fig.suptitle("PCE correlation magnitude distribution (confidence sharpening)")
    _savefig(fig, out_dir / "pce_correlations.png", show)


def plot_timing_breakdown(
    master_time_total: float,
    subproblem_time_total: float,
    baseline_time: float,
    out_dir: Path,
    show: bool = False,
) -> None:
    """Timing breakdown bar chart (R3: master vs subproblem vs baseline)."""
    import matplotlib.pyplot as plt
    labels = ["Master\n(proposal)", "Subproblem\n(classical)", "Baseline\n(MISOCP)"]
    values = [master_time_total, subproblem_time_total, baseline_time]
    colors = ["royalblue", "coral", "gray"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, fmt="%.2fs", padding=3)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Timing breakdown (R3: master ≠ subproblem)")
    _savefig(fig, out_dir / "timing_breakdown.png", show)


def plot_network_plan(
    plan: "Any",
    spec: "Any",
    out_dir: Path,
    show: bool = False,
) -> None:
    """Draw the investment plan on the network graph."""
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    for bus in spec.buses:
        G.add_node(bus.id, load=bus.load_mw)
    for line in spec.lines:
        G.add_edge(line.from_bus, line.to_bus)

    # Colour nodes: red = battery site, gray = no battery
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        bus_ids = spec.candidates.bess_buses
        if node in bus_ids:
            b_idx = bus_ids.index(node)
            if plan is not None and plan.build_mask[b_idx]:
                node_colors.append("red")
                node_sizes.append(300 + plan.size_mwh[b_idx] * 3)
            else:
                node_colors.append("lightyellow")
                node_sizes.append(150)
        else:
            node_colors.append("lightgray")
            node_sizes.append(100)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        with_labels=True, font_size=7,
    )
    ax.set_title(
        f"Investment plan — {spec.system.name}\n"
        f"Red nodes = BESS sites (size ∝ node size)"
    )
    ax.axis("off")
    _savefig(fig, out_dir / "network_plan.png", show)


def plot_approximation_ratio(
    k_values: list[int],
    approx_ratios: list[float],
    out_dir: Path,
    show: bool = False,
) -> None:
    """Approximation-ratio vs compression (sweep over k)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_values, approx_ratios, "s-", color="darkgreen", ms=8)
    ax.axhline(1.0, color="r", ls="--", alpha=0.5, label="Baseline = 1.0")
    ax.set_xlabel("PCE body degree k")
    ax.set_ylabel("quantum cost / baseline cost")
    ax.set_title("Approximation ratio vs PCE compression\n(Phase-3 defensibility artifact)")
    ax.legend()
    _savefig(fig, out_dir / "approx_ratio_vs_k.png", show)
