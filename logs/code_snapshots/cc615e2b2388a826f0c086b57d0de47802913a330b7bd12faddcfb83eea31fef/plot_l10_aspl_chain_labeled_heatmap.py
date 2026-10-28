#!/usr/bin/env python3
"""Create an annotated version of the ASPL-chain arrival-error heatmap.

Reads the already computed layer_metrics.csv; it does not inspect checkpoints
or rerun simulations. Cell labels show whole-percent point estimates.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "l10_n1000_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
OUT = ROOT / "output" / "analysis" / EXPERIMENT
FIGURES = OUT / "figures"
ASPL = (1, 2, 3, 4, 5, 6)
K_VALUES = (1, 5)
COMMUNITIES = tuple(range(1, 10))
SOURCE = OUT / "layer_metrics.csv"
STEM = "aspl_chain_arrival_error_map_labeled"


def matrices():
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(ASPL) * len(K_VALUES) * len(COMMUNITIES):
        raise RuntimeError(f"Expected 108 ASPL × k × community cells; found {len(rows)}")
    lookup = {(int(r["target_ASPL"]), int(r["k"]), int(r["community"])): r for r in rows}
    if len(lookup) != 108 or any(int(r["both_observed"]) != 400 for r in rows):
        raise RuntimeError("Expected 400 fully observed paired arrivals in every heatmap cell")
    result = {}
    for k in K_VALUES:
        for field in ("signed_relative_mean_shift_percent", "relative_w1_percent"):
            result[k, field] = np.array([
                [float(lookup[a, k, c][field]) for c in COMMUNITIES]
                for a in ASPL
            ])
    return result


def annotate(ax, image, values, signed):
    cmap = image.get_cmap()
    norm = image.norm
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            if signed:
                label = "0.0%" if abs(value) < 0.05 else f"{value:+.1f}%"
            else:
                label = f"{value:.1f}%"
            ax.text(x, y, label, ha="center", va="center", fontsize=7.2,
                    color="white" if luminance < 0.48 else "#202020")


def main():
    data = matrices()
    shifts = [data[k, "signed_relative_mean_shift_percent"] for k in K_VALUES]
    w1s = [data[k, "relative_w1_percent"] for k in K_VALUES]
    shift_limit = max(1.0, max(float(np.max(np.abs(x))) for x in shifts))
    w1_limit = max(1.0, max(float(np.max(x)) for x in w1s))
    cmap_shift = plt.get_cmap("RdBu_r")
    cmap_w1 = plt.get_cmap("YlOrBr")

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 7.0), layout="constrained")
    left_image = right_image = None
    for row, k in enumerate(K_VALUES):
        for col, (field, title, cmap, signed, limit) in enumerate((
            ("signed_relative_mean_shift_percent", f"k = {k}  |  signed mean-arrival shift", cmap_shift, True, shift_limit),
            ("relative_w1_percent", f"k = {k}  |  relative W₁ distance", cmap_w1, False, w1_limit),
        )):
            ax = axes[row, col]
            values = data[k, field]
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit) if signed else None
            image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
                              cmap=cmap, norm=norm,
                              vmin=None if signed else 0, vmax=None if signed else limit)
            annotate(ax, image, values, signed)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(len(COMMUNITIES)), COMMUNITIES)
            ax.set_yticks(range(len(ASPL)), ASPL)
            ax.set_xlabel("Community reached")
            ax.set_ylabel("Target ASPL")
            ax.set_xticks(np.arange(-.5, len(COMMUNITIES), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(ASPL), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.7)
            ax.tick_params(which="minor", bottom=False, left=False)
            if col == 0:
                left_image = image
            else:
                right_image = image

    fig.colorbar(left_image, ax=axes[:, 0], shrink=.8, pad=.025,
                 label="100 × (mean MicroMacro − mean Full) / mean Full (%)")
    fig.colorbar(right_image, ax=axes[:, 1], shrink=.8, pad=.025,
                 label="100 × W₁ / mean Full (%)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{STEM}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved annotated heatmap (400 paired runs per cell): {FIGURES / (STEM + '.png')}")


if __name__ == "__main__":
    main()
