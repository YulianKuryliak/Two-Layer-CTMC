#!/usr/bin/env python3
"""ASPL-chain error map with relative median-arrival shift and relative W1.

Reads saved paired checkpoint arrivals once and reuses the resulting median
table on later plotting runs. No simulations are run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_l10_aspl_chain_random_initial import ASPL_VALUES, EXPECTED_PAIRS, K_VALUES, LAYERS, OUT, read_pairs

FIGURES = OUT / "figures"
ARTICLE_IMAGES = ROOT / "paper" / "SMPT paper" / "Images" / "aspl_chain"
STEM = "aspl_chain_median_arrival_error_map_labeled"
MEDIAN_TABLE = OUT / "median_layer_metrics.csv"


def load_matrices():
    if MEDIAN_TABLE.exists():
        with MEDIAN_TABLE.open(newline="", encoding="utf-8") as handle:
            cached = list(csv.DictReader(handle))
        lookup = {(int(r["target_ASPL"]), int(r["k"]), int(r["community"])): r for r in cached}
        expected = {(aspl, k, layer) for aspl in ASPL_VALUES for k in K_VALUES for layer in LAYERS}
        if set(lookup) == expected and all(int(r["paired_runs"]) == EXPECTED_PAIRS for r in cached):
            return (
                {k: np.array([[float(lookup[aspl, k, layer]["signed_relative_median_shift_percent"])
                               for layer in LAYERS] for aspl in ASPL_VALUES]) for k in K_VALUES},
                {k: np.array([[float(lookup[aspl, k, layer]["relative_w1_percent"])
                               for layer in LAYERS] for aspl in ASPL_VALUES]) for k in K_VALUES},
            )
    pairs = read_pairs()
    with (OUT / "layer_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(ASPL_VALUES) * len(K_VALUES) * len(LAYERS):
        raise RuntimeError(f"Expected 108 metric rows, found {len(rows)}")
    lookup = {(int(r["target_ASPL"]), int(r["k"]), int(r["community"])): r for r in rows}
    if len(lookup) != 108:
        raise RuntimeError("Duplicate or missing rows in layer_metrics.csv")

    medians, w1 = {}, {}
    summary_rows = []
    for k in K_VALUES:
        medians[k] = np.empty((len(ASPL_VALUES), len(LAYERS)), dtype=float)
        w1[k] = np.empty_like(medians[k])
        for ai, aspl in enumerate(ASPL_VALUES):
            cell = pairs[(aspl, k)]
            if len(cell) != EXPECTED_PAIRS:
                raise RuntimeError(f"ASPL={aspl}, k={k}: expected 400 pairs, found {len(cell)}")
            for ci, layer in enumerate(LAYERS):
                full = np.asarray([run[0][layer] for run in cell], dtype=float)
                mm = np.asarray([run[1][layer] for run in cell], dtype=float)
                if not np.all(np.isfinite(full)) or not np.all(np.isfinite(mm)):
                    raise RuntimeError(f"Censored/missing arrival at ASPL={aspl}, k={k}, community={layer}")
                median_full = float(np.median(full))
                median_mm = float(np.median(mm))
                medians[k][ai, ci] = 100.0 * (median_mm - median_full) / median_full
                row = lookup[aspl, k, layer]
                if int(row["both_observed"]) != EXPECTED_PAIRS:
                    raise RuntimeError(f"W1 cell is not fully observed: ASPL={aspl}, k={k}, community={layer}")
                w1[k][ai, ci] = float(row["relative_w1_percent"])
                summary_rows.append({
                    "target_ASPL": aspl,
                    "k": k,
                    "community": layer,
                    "paired_runs": EXPECTED_PAIRS,
                    "median_full": median_full,
                    "median_mm": median_mm,
                    "signed_relative_median_shift_percent": medians[k][ai, ci],
                    "relative_w1_percent": w1[k][ai, ci],
                })
    with MEDIAN_TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return medians, w1


def add_labels(ax, image, values, signed):
    cmap, norm = image.get_cmap(), image.norm
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            label = "0.0" if signed and abs(value) < 0.05 else f"{value:.1f}"
            ax.text(x, y, label, ha="center", va="center", fontsize=8.0,
                    color="white" if luminance < 0.48 else "#202020")


def main():
    medians, w1 = load_matrices()
    median_limit = max(1.0, max(float(np.max(np.abs(v))) for v in medians.values()))
    w1_limit = max(1.0, max(float(np.max(v)) for v in w1.values()))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.7), layout="constrained")
    left_image = right_image = None
    for row, k in enumerate(K_VALUES):
        for col, (values, title, cmap, signed, limit) in enumerate((
            (medians[k], f"k = {k}  |  signed median-arrival shift", "RdBu_r", True, median_limit),
            (w1[k], f"k = {k}  |  relative W₁ distance", "YlOrBr", False, w1_limit),
        )):
            ax = axes[row, col]
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit) if signed else None
            image = ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
                              norm=norm, vmin=None if signed else 0, vmax=None if signed else limit)
            add_labels(ax, image, values, signed)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(len(LAYERS)), LAYERS)
            ax.set_yticks(range(len(ASPL_VALUES)), ASPL_VALUES)
            ax.set_xlabel("Community reached")
            ax.set_ylabel("Target internal ASPL")
            ax.set_xticks(np.arange(-.5, len(LAYERS), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(ASPL_VALUES), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.7)
            ax.tick_params(which="minor", bottom=False, left=False)
            if col == 0:
                left_image = image
            else:
                right_image = image
    fig.colorbar(left_image, ax=axes[:, 0], shrink=.8, pad=.025,
                 label="100 × (median Tᵢ, MM − median Tᵢ, Full) / median Tᵢ, Full (%)")
    fig.colorbar(right_image, ax=axes[:, 1], shrink=.8, pad=.025,
                 label="100 × W₁ / mean Tᵢ, Full (%)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTICLE_IMAGES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{STEM}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{STEM}.pdf", bbox_inches="tight")
    fig.savefig(ARTICLE_IMAGES / f"{STEM}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved median-based heatmap (400 paired runs per cell): {ARTICLE_IMAGES / (STEM + '.png')}")


if __name__ == "__main__":
    main()
