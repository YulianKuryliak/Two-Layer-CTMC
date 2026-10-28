"""Figures for the SMPT two-scale manuscript.

Uses saved analysis outputs only (cell_summary.csv); no simulation is run.
Writes PDF/PNG files to paper/SMPT paper/Images/twoscale/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "output" / "analysis" / "smpt_twoscale_paper" / "cell_summary.csv"
OUT = ROOT / "paper" / "SMPT paper" / "Images" / "twoscale"
BETA = 0.003
SIZES = (50, 100, 250, 1000)
SIZE_COLOURS = {50: "#d73027", 100: "#fc8d59", 250: "#4575b4", 1000: "#313695"}
INFECTED, SUSCEPTIBLE, BRIDGE = "#e66101", "#cfe2f3", "#000000"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- schematic
def community_layouts(sizes=(9, 9, 9), seed=4):
    rng = np.random.default_rng(seed)
    graphs, positions = [], []
    for c, n in enumerate(sizes):
        g = nx.connected_watts_strogatz_graph(n, 4, 0.3, seed=int(rng.integers(1e6)))
        pos = nx.spring_layout(g, seed=int(rng.integers(1e6)), k=0.9)
        xy = np.array([pos[v] for v in g])
        xy = (xy - xy.mean(0)) / np.abs(xy).max() * 0.55
        graphs.append(g)
        positions.append({v: xy[i] + np.array([2.2 * c, 0.0]) for i, v in enumerate(g)})
    return graphs, positions


def draw_communities(ax, graphs, positions, infected, bridges=None, node_size=34):
    for c, (g, pos) in enumerate(zip(graphs, positions)):
        centre = np.mean(list(pos.values()), axis=0)
        ax.add_patch(plt.Circle(centre, 0.85, fc="#f7f7f7", ec="#969696", lw=0.8, zorder=0))
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#8c8c8c", width=0.6)
        colours = [INFECTED if (c, v) in infected else SUSCEPTIBLE for v in g]
        edge = [BRIDGE if bridges and (c, v) in bridges else "#525252" for v in g]
        widths = [1.6 if bridges and (c, v) in bridges else 0.5 for v in g]
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colours, edgecolors=edge,
                               linewidths=widths, node_size=node_size)
    ax.set_aspect("equal")
    ax.axis("off")


def schematic() -> None:
    graphs, positions = community_layouts()
    infected = {(0, 0), (0, 1), (0, 3), (0, 4), (0, 6), (1, 2)}
    interface = [((0, 5), (1, 7)), ((1, 4), (2, 1))]
    bridges = {end for pair in interface for end in pair}

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    ax = axes[0]
    draw_communities(ax, graphs, positions, infected, bridges)
    for (c1, v1), (c2, v2) in interface:
        p1, p2 = positions[c1][v1], positions[c2][v2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#b2182b", lw=1.6, zorder=3)
    ax.text(2.2, -1.05, "explicit edges $E_{ij}$ between bridge nodes", ha="center", fontsize=6.5, color="#b2182b")
    ax.set_title("(a) Full network $G$", pad=2)

    ax = axes[1]
    draw_communities(ax, graphs, positions, infected)
    top = 1.75
    meta = [np.array([2.2 * c, top]) for c in range(3)]
    for c, p in enumerate(meta):
        ax.add_patch(plt.Circle(p, 0.26, fc="#fee0b6", ec="#7f3b08", lw=1.0, zorder=4))
        ax.text(p[0], p[1], f"$V_{c + 1}$", ha="center", va="center", fontsize=7, zorder=5)
        ax.add_patch(FancyArrowPatch((p[0] - 0.18, 0.88), (p[0] - 0.18, top - 0.28),
                                     arrowstyle="-|>", mutation_scale=7, color="#4d4d4d", lw=0.8))
    for c in range(2):
        a, b = meta[c], meta[c + 1]
        ax.plot([a[0] + 0.26, b[0] - 0.26], [top, top], color="#7f3b08", lw=2.2, zorder=3)
        ax.text((a[0] + b[0]) / 2, top + 0.14, f"$W_{{{c + 1}{c + 2}}}$", ha="center", fontsize=7)
    ax.text(-0.25, (0.88 + top - 0.28) / 2, r"$(\bar y_i,\bar s_i)$", ha="right", va="center", fontsize=7)
    target = positions[2][5]
    ax.add_patch(FancyArrowPatch((meta[2][0] + 0.22, top - 0.22), (target[0] + 0.05, target[1] + 0.08),
                                 arrowstyle="-|>", mutation_scale=8, color="#b2182b", lw=1.0,
                                 connectionstyle="arc3,rad=-0.25", zorder=5))
    ax.text(2.2, -1.05, r"rates $\lambda_{ij}=\beta W_{ij}\bar y_i\bar s_j$; uniform import", ha="center",
            fontsize=6.5, color="#b2182b")
    ax.set_ylim(-1.15, 2.1)
    ax.set_title("(b) Two-scale simulator", pad=2)

    ax = axes[2]
    draw_communities(ax, graphs, positions, infected)
    for c in range(2):
        for u in graphs[c]:
            for v in graphs[c + 1]:
                p1, p2 = positions[c][u], positions[c + 1][v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#b2182b", lw=0.25, alpha=0.18, zorder=1)
    ax.text(2.2, -1.05, r"weight $W_{ij}/(N_iN_j)$ on every pair", ha="center", fontsize=6.5, color="#b2182b")
    ax.set_title("(c) Equivalent surrogate $\\tilde G$", pad=2)
    for a in axes:
        a.set_xlim(-1.3, 5.7)
        a.set_ylim(-1.15, 2.1)
    fig.subplots_adjust(wspace=0.02)
    save(fig, "fig_method_schematic")


# ---------------------------------------------------------------- accuracy map
def accuracy_map(summary: pd.DataFrame, *, name: str = "fig_accuracy_map") -> None:
    """Separate signed median shift and Wasserstein distance, with shared scales per metric."""
    expected = {(n, a, k) for n in SIZES for a in range(1, 7) for k in (1, 5)}
    observed = set(zip(summary.n, summary.aspl, summary.k))
    if len(summary) != len(expected) or observed != expected:
        raise ValueError("Accuracy map requires one row for each of the 48 configurations")
    fields = ["median_shift_pct", "rel_w1_pct", "rel_w1_perm_p"]
    if not np.isfinite(summary[fields].to_numpy()).all():
        raise ValueError("Accuracy map metrics must be finite")

    shift_limit = max(1.0, float(summary["median_shift_pct"].abs().max()))
    w1_limit = max(1.0, float(summary["rel_w1_pct"].max()))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), layout="constrained")
    images = []
    for row_index, k in enumerate((1, 5)):
        cell = summary[summary.k == k]
        for column, field in enumerate(("median_shift_pct", "rel_w1_pct")):
            ax = axes[row_index, column]
            signed = column == 0
            limit = shift_limit if signed else w1_limit
            grid = np.full((6, 4), np.nan)
            for _, row in cell.iterrows():
                grid[int(row.aspl) - 1, SIZES.index(int(row.n))] = row[field]
            kwargs = {"norm": TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)} if signed else {"vmin": 0, "vmax": limit}
            image = ax.imshow(grid, cmap="RdBu_r" if signed else "YlOrBr", origin="lower",
                              aspect="auto", interpolation="nearest", **kwargs)
            if row_index == 0:
                images.append(image)
            for _, row in cell.iterrows():
                i, j = int(row.aspl) - 1, SIZES.index(int(row.n))
                value = row[field]
                # The retained permutation test concerns W1, not the median shift.
                if not signed and row.rel_w1_perm_p >= 0.05:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               hatch="////", ec="#9e9e9e", lw=0, zorder=2))
                ax.text(j, i, f"{value:+.1f}" if signed else f"{value:.1f}",
                        ha="center", va="center", fontsize=10,
                        color="white" if abs(value) > 0.58 * limit else "#222222", zorder=3)
            ax.set_xticks(range(4), [str(n) for n in SIZES])
            ax.set_yticks(range(6), ["1 (clique)", "2", "3", "4", "5", "6"])
            ax.set_xlabel("community size $n$", fontsize=10)
            ax.set_ylabel("target internal ASPL", fontsize=10)
            ax.tick_params(labelsize=9)
            ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.7)
            ax.tick_params(which="minor", bottom=False, left=False)
            letter = "abcd"[2 * row_index + column]
            metric = "median shift" if signed else "Wasserstein distance"
            ax.set_title(f"({letter}) $k={k}$: {metric}", fontsize=11)
    for column, label in enumerate((r"relative median shift $b_9$ (%)", r"normalised Wasserstein distance $e_W$ (%)")):
        cbar = fig.colorbar(images[column], ax=axes[:, column], shrink=0.86, pad=0.025)
        cbar.set_label(label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
    save(fig, name)


# ---------------------------------------------------------------- descriptor
def descriptor(summary: pd.DataFrame) -> None:
    s = summary.copy()
    s["x"] = BETA * s["sigma_full"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for (n, k), row in s.groupby(["n", "k"]):
        row = row.sort_values("x")
        style = dict(color=SIZE_COLOURS[n], marker="o" if k == 1 else "^", ms=4,
                     mfc=SIZE_COLOURS[n] if k == 1 else "white", lw=0.8)
        axes[0].plot(row.x, row.rel_w1_pct, **style)
        axes[0].plot(row.x, row.rel_w1_null95_pct, ls="none", marker="_", ms=6, color="#9e9e9e")
        axes[1].plot(row.x, row.median_shift_pct, **style)
    axes[0].set_ylabel(r"$e_W$ of $T_9$ (%)")
    axes[1].set_ylabel(r"median shift $b_9$ (%)")
    axes[1].axhline(0, color="#9e9e9e", lw=0.6)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\rho=\beta\,\sigma_T$")
    axes[0].set_title("(a) distributional discrepancy")
    axes[1].set_title("(b) signed shift of the median")

    hi = s[s.aspl >= 3]
    ax = axes[2]
    for k, marker in ((1, "o"), (5, "^")):
        part = hi[hi.k == k]
        for _, row in part.iterrows():
            ax.errorbar(row.first_cross_shift_pct, row.later_cross_shift_pct,
                        xerr=[[row.first_cross_shift_pct - row.first_cross_ci_low],
                              [row.first_cross_ci_high - row.first_cross_shift_pct]],
                        fmt=marker, ms=4, color=SIZE_COLOURS[int(row.n)],
                        mfc=SIZE_COLOURS[int(row.n)] if k == 1 else "white", elinewidth=0.4, capsize=0)
    lim = (-47, 5)
    ax.plot(lim, lim, color="#9e9e9e", lw=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("first crossing: mean shift (%)")
    ax.set_ylabel("crossings 2–9: mean shift (%)")
    ax.set_title("(c) first vs later crossings, ASPL $\\geq3$")
    ax.text(-45, 1.5, "below diagonal: later\ncrossings more biased", fontsize=6, va="top", color="#636363")
    handles = [plt.Line2D([], [], color=SIZE_COLOURS[n], marker="s", ls="none", ms=5, label=f"$n={n}$") for n in SIZES]
    handles += [plt.Line2D([], [], color="k", marker="o", ls="none", ms=4, label="$k=1$"),
                plt.Line2D([], [], color="k", marker="^", mfc="white", ls="none", ms=4, label="$k=5$"),
                plt.Line2D([], [], color="#9e9e9e", marker="_", ls="none", ms=6, label="null 95%")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(w_pad=1.0)
    fig.subplots_adjust(bottom=0.27)
    save(fig, "fig_descriptor")


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    schematic()
    accuracy_map(summary)
    descriptor(summary)


if __name__ == "__main__":
    main()
