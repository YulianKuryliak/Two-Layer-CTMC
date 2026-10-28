#!/usr/bin/env python3
"""Article figure: propagation and last-community arrival for a clique chain.

Uses the saved 400-pair summary cache from the latest random-initial-node
experiment. No simulations are run here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    "l10_n1000_chain_clique_tree_augmented_aspl1to6_step1_"
    "rep400_random_initial_independent_rng"
)
CACHE = (
    ROOT / "output" / "analysis" / EXPERIMENT / "trajectory_cdf_cache"
    / "aspl1_k1.npz"
)
ANALYSIS_FIGURES = ROOT / "output" / "analysis" / "chain_cliques_n1000_k1_random_initial" / "figures"
ARTICLE_FIGURES = ROOT / "paper" / "SMPT paper" / "Images" / "chain_cliques"
STEM = "chain_cliques_n1000_k1_propagation_cdf_random_initial"
TIME_LIMIT = 6_000.0
BLUE, ORANGE = "#2878B5", "#D45A44"


def load_data():
    if not CACHE.exists():
        raise FileNotFoundError(f"Expected saved trajectory/CDF cache: {CACHE}")
    with np.load(CACHE) as data:
        result = {key: data[key].copy() for key in data.files}
    if int(result["checkpoint_count"]) != 400:
        raise RuntimeError(f"Expected 400 paired runs; cache has {result['checkpoint_count']}")
    for key in ("full_quantiles", "mm_quantiles"):
        if result[key].shape != (3, 501):
            raise RuntimeError(f"Unexpected trajectory summary shape for {key}: {result[key].shape}")
        if np.any(np.diff(result[key][1]) < 0):
            raise RuntimeError(f"Median trajectory is not nondecreasing: {key}")
    for key in ("full_arrivals", "mm_arrivals"):
        if result[key].shape != (400,) or not np.all(np.isfinite(result[key])):
            raise RuntimeError(f"Expected 400 observed last-community arrivals in {key}")
    return result


def ecdf(values: np.ndarray):
    ordered = np.sort(values)
    return np.r_[0.0, ordered], np.r_[0.0, np.arange(1, len(ordered) + 1) / len(ordered)]


def draw(data, destination: Path):
    grid = data["grid"]
    visible = grid <= TIME_LIMIT
    plot_grid = grid[visible]
    fig, (ax_curve, ax_cdf) = plt.subplots(
        1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    for key, arrivals_key, color, label in (
        ("full_quantiles", "full_arrivals", BLUE, "Full Micro"),
        ("mm_quantiles", "mm_arrivals", ORANGE, "MicroMacro"),
    ):
        q25, median, q75 = data[key]
        ax_curve.fill_between(plot_grid, q25[visible], q75[visible], color=color, alpha=0.16, linewidth=0)
        ax_curve.plot(plot_grid, median[visible], color=color, linewidth=2.0, label=label)
        x, y = ecdf(data[arrivals_key])
        ax_cdf.step(x, y, where="post", color=color, linewidth=2.0, label=label)

    ax_curve.set_title("Epidemic propagation")
    ax_curve.set_xlabel("Time")
    ax_curve.set_ylabel("Infected nodes")
    ax_curve.set_xlim(0, TIME_LIMIT)
    ax_curve.set_ylim(0, 10_300)
    ax_curve.grid(alpha=0.2)

    ax_cdf.set_title("Arrival in community 10")
    ax_cdf.set_xlabel("Time")
    ax_cdf.set_ylabel("Empirical CDF")
    ax_cdf.set_xlim(0, TIME_LIMIT)
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.grid(alpha=0.2)

    for ax in (ax_curve, ax_cdf):
        ax.spines[["top", "right"]].set_visible(False)
    ax_curve.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f"{STEM}.png", dpi=220, bbox_inches="tight")
    fig.savefig(destination / f"{STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_data()
    draw(data, ANALYSIS_FIGURES)
    draw(data, ARTICLE_FIGURES)
    print(f"Used 400 paired runs from: {CACHE.relative_to(ROOT)}")
    print(f"Saved PNG/PDF to: {ANALYSIS_FIGURES.relative_to(ROOT)}")
    print(f"Saved article copies to: {ARTICLE_FIGURES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
