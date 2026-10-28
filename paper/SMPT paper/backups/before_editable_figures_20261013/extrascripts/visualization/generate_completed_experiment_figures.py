#!/usr/bin/env python3
"""Generate report figures from completed CE-06--CE-10 saved summaries."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "claude_experiments"
REPORTS = ROOT / "reports" / "claude_experiments"


def save(fig, experiment: str, name: str) -> None:
    directory = REPORTS / experiment / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    if name != "regime_map.png":
        fig.tight_layout()
    fig.savefig(directory / name, dpi=220)
    plt.close(fig)


def ce06() -> None:
    data = pd.read_csv(OUT / "ce06_regime_map_full" / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    vmax = np.abs(data.delta_RMST_norm).max()
    for ax, (r, group) in zip(axes, data.groupby("r")):
        table = group.pivot(index="d", columns="k", values="delta_RMST_norm").sort_index()
        image = ax.imshow(table, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set(title=f"r={r:g}", xlabel="interface width k", xticks=range(len(table.columns)), xticklabels=table.columns)
        for y in range(len(table.index)):
            for x in range(len(table.columns)):
                ax.text(x, y, f"{table.iloc[y, x]:.2f}", ha="center", va="center", fontsize=8)
    axes[0].set(ylabel="mean internal degree d", yticks=range(len(table.index)), yticklabels=table.index)
    fig.subplots_adjust(left=.08, right=.91, bottom=.17, top=.88, wspace=.10)
    color_axis = fig.add_axes([.93, .2, .015, .62])
    fig.colorbar(image, cax=color_axis, label=r"$\Delta$RMST / $T_{end}$")
    save(fig, "ce06_regime_map", "regime_map.png")


def ce07() -> None:
    data = pd.read_csv(OUT / "ce07_calibration_floor_pilot" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for d, group in data.groupby("d"):
        group = group.sort_values("r")
        axes[0].plot(group.r, group.dRMST_uncal_norm, "o--", alpha=.55, label=f"uncal d={d:g}")
        axes[0].plot(group.r, group.dRMST_cal_norm, "o-", label=f"cal d={d:g}")
    axes[0].axhline(0, color="black", lw=.8); axes[0].set(xlabel="r", ylabel=r"$\Delta$RMST / $T_{end}$")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    for r, group in data.groupby("r"):
        axes[1].plot(group.d, group.c_star, "o-", label=f"r={r:g}")
    axes[1].set(xlabel="mean internal degree d", ylabel=r"selected calibration $c^*$")
    axes[1].legend(frameon=False)
    save(fig, "ce07_calibration_floor", "calibration_floor.png")


def ce08() -> None:
    path = OUT / "ce08_negative_control_full" / "cells" / "arrivals_clique_L3_n100_k2_r1.csv"
    data = pd.read_csv(path)
    terminal = data[data.layer == data.layer.max()]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for model, group in terminal.groupby("model"):
        values = np.sort(group.T_i.dropna().to_numpy())
        ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", label=model)
    ax.set(xlabel=r"terminal arrival $T_{last}$", ylabel="empirical CDF"); ax.legend(frameon=False)
    save(fig, "ce08_negative_control", "terminal_cdf.png")


def ce09() -> None:
    data = pd.read_csv(OUT / "ce09_boundary_isolation_full" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), sharey=True)
    for ax, (k, group) in zip(axes, data.groupby("k")):
        for mode, item in group.groupby("placement_mode"):
            item = item.sort_values("d")
            ax.plot(item.d, item.delta_RMST_norm, "o-", label=mode)
        ax.axhline(0, color="black", lw=.8); ax.set(title=f"k={k:g}", xlabel="mean internal degree d")
    axes[0].set_ylabel(r"$\Delta$RMST / $T_{end}$")
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "ce09_boundary_isolation", "placement_diagnostics.png")


def ce10() -> None:
    data = pd.read_csv(OUT / "ce10_correction_ladder_full" / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    for r, group in data.groupby("r"):
        group = group.sort_values("d")
        axes[0].plot(group.d, group.delta_RMST_off_norm, "o--", alpha=.6, label=f"off r={r:g}")
        axes[0].plot(group.d, group.delta_RMST_spectral_norm, "o-", label=f"spectral r={r:g}")
        axes[1].plot(group.d, group.gap_closure, "o-", label=f"r={r:g}")
    axes[0].axhline(0, color="black", lw=.8); axes[0].set(xlabel="mean internal degree d", ylabel=r"$\Delta$RMST / $T_{end}$")
    axes[1].axhline(0, color="black", lw=.8); axes[1].axhline(1, color="gray", lw=.8, ls="--")
    axes[1].set(xlabel="mean internal degree d", ylabel="gap closure")
    for ax in axes: ax.legend(frameon=False, fontsize=7)
    save(fig, "ce10_correction_ladder", "spectral_correction.png")


def main() -> None:
    ce06(); ce07(); ce08(); ce09(); ce10()


if __name__ == "__main__":
    main()
