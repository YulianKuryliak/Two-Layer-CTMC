#!/usr/bin/env python3
"""Create publication-oriented figures from the new independent-RNG export study.

Run analyze_single_community_export.py first.  This script only reads its
tables and writes PDF/PNG figures beneath output/analysis/single_community_export.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = REPO_ROOT / "output" / "analysis" / "single_community_export"
FIGURES = ANALYSIS / "figures"
FULL = "Full Micro"
MM = "Production MicroMacro + independent RNG"
COLORS = {FULL: "#2c7fb8", MM: "#d95f4c"}
PLACEMENT_COLORS = {"near": "#2c7fb8", "middle": "#7b3294", "far": "#d95f4c"}


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    return values, np.arange(1, len(values) + 1) / len(values)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def e1_interface(obs: pd.DataFrame, desc: pd.DataFrame) -> None:
    subset = obs[(obs.block == "E1") & (obs.topology == "random_regular") & (obs.k == 5)]
    if subset.empty:
        print("[plot] E1 random_regular k5 unavailable; skipped figure 1")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.65), constrained_layout=True)
    for placement in ("near", "middle", "far"):
        vals = subset[(subset.placement == placement) & (subset.simulator == FULL) & subset.export_observed].T_export.dropna().to_numpy(float)
        if vals.size:
            x, y = ecdf(vals)
            axes[0].step(x, y, where="post", color=PLACEMENT_COLORS[placement], label=f"Full: {placement}")
    mm = subset[(subset.placement == "near") & (subset.simulator == MM) & subset.export_observed].T_export.dropna().to_numpy(float)
    if mm.size:
        x, y = ecdf(mm)
        axes[0].step(x, y, where="post", color=COLORS[MM], linestyle="--", linewidth=2, label="MicroMacro")
    axes[0].set(xlabel="first-export time", ylabel="ECDF", title="Fixed source graph: export distributions")
    axes[0].legend(fontsize=8)

    d = desc[(desc.topology == "random_regular") & (desc.k == 5)].copy()
    full = subset[(subset.simulator == FULL) & subset.export_observed].groupby(["placement", "graph_id"], as_index=False).T_export.mean()
    full = full.merge(d[["placement", "graph_id", "D_mean"]], on=["placement", "graph_id"], how="left")
    for placement, frame in full.groupby("placement"):
        axes[1].scatter(frame.D_mean, frame.T_export, color=PLACEMENT_COLORS.get(placement, "grey"), label=placement, alpha=.8)
    axes[1].set(xlabel=r"mean graph distance $D_{mean}(s,B)$", ylabel="mean Full export time", title="Same spectrum; moved output interface")
    axes[1].legend(title="placement", fontsize=8)

    pairs = subset.pivot_table(index=["placement", "graph_id", "event_id"], columns="simulator", values="T_export", aggfunc="first").dropna()
    for placement, frame in pairs.groupby(level="placement"):
        delta = frame[MM] - frame[FULL]
        axes[2].scatter(np.full(len(delta), {"near": 0, "middle": 1, "far": 2}[placement]), delta, color=PLACEMENT_COLORS[placement], alpha=.45, s=14)
        axes[2].plot({"near": 0, "middle": 1, "far": 2}[placement], delta.mean(), "o", color="black", ms=5)
    axes[2].axhline(0, color="black", lw=.8)
    axes[2].set(xlim=(-.45, 2.45), xticks=(0, 1, 2), xticklabels=("near", "middle", "far"), ylabel=r"$T_{MM}-T_{Full}$", title="Paired signed export error")
    save(fig, "fig1_fixed_spectrum_interface_location")


def e0_e2(obs: pd.DataFrame, desc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    for side, color in zip(("path", "clique", "random_connected"), ("#2c7fb8", "#d95f4c", "#41ab5d"), strict=True):
        frame = obs[(obs.kind == "e0a") & (obs.ell == 4) & (obs.side == side) & (obs.simulator == FULL) & obs.export_observed]
        vals = frame.T_export.dropna().to_numpy(float)
        if vals.size:
            x, y = ecdf(vals); axes[0, 0].step(x, y, where="post", color=color, label=side)
    x = np.linspace(0, max(axes[0, 0].get_xlim()[1], 1), 400)
    axes[0, 0].plot(x, gamma.cdf(x, a=5, scale=1 / .003), "k--", label=r"Gamma(5, $\beta$)")
    axes[0, 0].set(xlabel="first-export time", ylabel="ECDF", title="E0a: same corridor, altered side graph")
    axes[0, 0].legend(fontsize=8)

    qframe = obs[(obs.kind == "e0b") & (obs.simulator == FULL) & obs.export_observed]
    qmeans = qframe.groupby("q").T_export.agg(["mean", "std", "count"]).reset_index()
    if not qmeans.empty:
        sem = qmeans["std"] / np.sqrt(qmeans["count"])
        axes[0, 1].errorbar(qmeans.q, qmeans["mean"], yerr=sem, marker="o", color=COLORS[FULL], capsize=3)
    axes[0, 1].set(xlabel="number of parallel paths q", ylabel="mean Full export time", title="E0b: same hop distance, more paths")

    e2d = desc[desc.kind == "e2"].copy()
    e2f = obs[(obs.kind == "e2") & (obs.simulator == FULL) & obs.export_observed].groupby(["scenario", "graph_id", "placement"], as_index=False).T_export.mean()
    e2 = e2f.merge(e2d[["scenario", "graph_id", "lambda2_normalized_laplacian", "cut_edges", "placement"]], on=["scenario", "graph_id", "placement"], how="left")
    for placement, frame in e2.groupby("placement"):
        axes[1, 0].scatter(frame.lambda2_normalized_laplacian, frame.T_export, label=placement, alpha=.75)
    axes[1, 0].set(xlabel=r"normalized Laplacian $\lambda_2$", ylabel="mean Full export time", title="E2 exploratory matched subset")
    axes[1, 0].legend(fontsize=8)

    pairs = obs[obs.kind == "e2"].pivot_table(index=["scenario", "graph_id", "event_id", "placement", "cut_edges"], columns="simulator", values="T_export", aggfunc="first").dropna()
    if not pairs.empty:
        rows = []
        for (scenario, graph_id, event_id, placement, cut), values in pairs.iterrows():
            rows.append((cut, placement, values[MM] - values[FULL]))
        error = pd.DataFrame(rows, columns=["cut", "placement", "delta"])
        for placement, frame in error.groupby("placement"):
            axes[1, 1].plot(frame.groupby("cut").delta.mean().index, frame.groupby("cut").delta.mean().values, marker="o", label=placement)
    axes[1, 1].axhline(0, color="black", lw=.8)
    axes[1, 1].set(xlabel="number of cross edges in source", ylabel=r"mean $T_{MM}-T_{Full}$", title="E2: signed error across bottlenecks")
    axes[1, 1].legend(fontsize=8)
    save(fig, "fig2_path_and_spectrum_controls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ANALYSIS / "analysis_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("valid_for_analysis"):
        raise SystemExit("Data-quality gate did not pass; do not plot.")
    obs = pd.read_csv(ANALYSIS / "export_observables.csv")
    desc = pd.read_csv(ANALYSIS / "graph_descriptors.csv")
    if not args.allow_partial:
        manifests = list(REPO_ROOT.glob("output/data/export_single_*_independent_rng_*/manifest.json"))
        incomplete = [path for path in manifests if json.loads(path.read_text()).get("completed") != json.loads(path.read_text()).get("requested_event_pairs")]
        if incomplete:
            raise SystemExit(f"{len(incomplete)} scenarios are incomplete; use --allow-partial to label/use partial data.")
    e1_interface(obs, desc)
    e0_e2(obs, desc)
    print(f"[plot] saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
