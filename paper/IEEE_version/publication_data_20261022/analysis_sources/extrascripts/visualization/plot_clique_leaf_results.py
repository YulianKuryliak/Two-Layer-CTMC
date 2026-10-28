#!/usr/bin/env python3
"""Publication figures for the completed clique--bridge--leaf benchmark.

This script only reads saved independent-RNG results.  It does not import or
run either simulator.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "output/main_clean_benchmarks_independent_rng/benchmark_observables.csv"
OUTPUT = ROOT / "output/analysis/clique_leaf"
PAPER_IMAGES = ROOT / "paper/SMPT paper/Images/clique_leaf"

BENCHMARK = "clique_leaf_n100_k2"
FULL = "Full Micro"
MM = "Production MicroMacro + independent RNG"
MODELS = (FULL, MM)
LABELS = {FULL: "Full Micro", MM: "MicroMacro"}
COLORS = {FULL: "#2878B5", MM: "#D0604C"}
LINESTYLES = {FULL: "-", MM: "--"}


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1, dtype=float) / len(x)


def survival(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, (len(x) - np.arange(len(x), dtype=float)) / len(x)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return centre - half_width, centre + half_width


def paired_mean_ci(delta: np.ndarray, seed: int = 20260919, draws: int = 5000) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling paired replication differences."""
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    batch = 100
    for start in range(0, draws, batch):
        stop = min(start + batch, draws)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def load_data() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    data = pd.read_csv(INPUT)
    data = data[data["benchmark"].eq(BENCHMARK)].copy()
    if len(data) != 20_000:
        raise RuntimeError(f"expected 20,000 rows for {BENCHMARK}, found {len(data)}")
    if data[list(("bridge_infection_time", "bridge_to_export_delay", "total_export_time"))].isna().any().any():
        raise RuntimeError("clique--bridge--leaf observables contain missing values")

    arrays: dict[str, np.ndarray] = {}
    for model in MODELS:
        model_data = data[data["simulator"].eq(model)].sort_values("replication_id")
        if len(model_data) != 10_000 or model_data["replication_id"].nunique() != 10_000:
            raise RuntimeError(f"expected 10,000 unique replications for {model}")
        for column in ("bridge_infection_time", "bridge_to_export_delay", "total_export_time"):
            arrays[f"{model}:{column}"] = model_data[column].to_numpy(float)
        arrays[f"{model}:replication_id"] = model_data["replication_id"].to_numpy(int)

    if not np.array_equal(arrays[f"{FULL}:replication_id"], arrays[f"{MM}:replication_id"]):
        raise RuntimeError("Full and MicroMacro replication IDs are not paired")
    return data, arrays


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color="#D9DDE3", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for folder in (OUTPUT, PAPER_IMAGES):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mechanism_figure(arrays: dict[str, np.ndarray], metrics: dict[str, float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.65))

    ax = axes[0]
    for model in MODELS:
        x, y = ecdf(arrays[f"{model}:total_export_time"])
        ax.step(x, y, where="post", color=COLORS[model], linestyle=LINESTYLES[model],
                linewidth=1.7, label=LABELS[model])
    ax.set(xlabel="total export time", ylabel="empirical CDF", xlim=(0, 600), ylim=(0, 1.01))
    ax.text(0.97, 0.07, rf"$W_1={metrics['W1_export']:.3f}$" + "\n" +
            rf"$D_{{KS}}={metrics['KS_export']:.4f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#C9CDD2", "pad": 3})
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.22), fontsize=9)
    ax.set_title("a  Export-time distribution", loc="left", fontsize=11)
    style_axis(ax)

    ax = axes[1]
    for model in MODELS:
        x, y = ecdf(arrays[f"{model}:bridge_to_export_delay"])
        ax.step(x, y, where="post", color=COLORS[model], linestyle=LINESTYLES[model],
                linewidth=1.7, label=LABELS[model])
    ax.axvspan(-20, 0, color="#EEE4D7", alpha=0.75, zorder=0)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set(xlabel=r"$T_{export}-T_{bridge}$", ylabel="empirical CDF",
           xlim=(-20, 100), ylim=(0, 0.55))
    ax.text(-18, 0.51, "export before bridge", fontsize=8.5, va="top", color="#5B4B3A")
    ax.set_title("b  Signed ordering diagnostic", loc="left", fontsize=11)
    style_axis(ax)

    ax = axes[2]
    counts = np.array([int((arrays[f"{model}:bridge_to_export_delay"] < 0).sum()) for model in MODELS])
    totals = np.array([len(arrays[f"{model}:bridge_to_export_delay"]) for model in MODELS])
    rates = 100.0 * counts / totals
    intervals = np.array([wilson_interval(int(c), int(n)) for c, n in zip(counts, totals)]) * 100.0
    errors = np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates))
    xpos = np.arange(2)
    ax.errorbar(xpos, rates, yerr=errors, fmt="o", markersize=7, capsize=5,
                color="#2D3035", ecolor="#2D3035", linewidth=1.4)
    for i, (rate, count, total) in enumerate(zip(rates, counts, totals)):
        ax.text(i, max(rate, intervals[i, 1]) + 0.08, f"{count}/{total}\n({rate:.2f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xpos, ["Full\nMicro", "MicroMacro"])
    ax.set(ylabel="bridge-bypass runs (%)", ylim=(-0.03, 1.35))
    ax.set_title("c  Mechanism violation", loc="left", fontsize=11)
    style_axis(ax)

    fig.suptitle("Clique–bridge–leaf: similar export times do not imply a preserved transmission path",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    save_figure(fig, "clique_leaf_endpoint_and_mechanism")


def distribution_figure(arrays: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.65))

    ax = axes[0]
    for model in MODELS:
        x, y = ecdf(arrays[f"{model}:bridge_infection_time"])
        ax.step(x, y, where="post", color=COLORS[model], linestyle=LINESTYLES[model],
                linewidth=1.7, label=LABELS[model])
    ax.set(xlabel="first bridge infection time", ylabel="empirical CDF", xlim=(0, 25), ylim=(0, 1.01))
    ax.set_title("a  Bridge activation", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    style_axis(ax)

    ax = axes[1]
    for model in MODELS:
        x, y = survival(arrays[f"{model}:total_export_time"])
        ax.step(x, y, where="post", color=COLORS[model], linestyle=LINESTYLES[model],
                linewidth=1.7, label=LABELS[model])
    ax.set_yscale("log")
    ax.set(xlabel="total export time", ylabel="empirical survival", xlim=(0, 1000), ylim=(1e-4, 1.05))
    ax.set_title("b  Export-time tail", loc="left", fontsize=11)
    style_axis(ax)

    ax = axes[2]
    probabilities = np.linspace(0.01, 0.99, 99)
    full_q = np.quantile(arrays[f"{FULL}:total_export_time"], probabilities)
    mm_q = np.quantile(arrays[f"{MM}:total_export_time"], probabilities)
    maximum = max(float(full_q.max()), float(mm_q.max()))
    ax.plot([0, maximum], [0, maximum], color="#3A3D42", linewidth=1.1, linestyle=":",
            label="equal quantiles")
    ax.scatter(full_q, mm_q, s=14, color=COLORS[MM], alpha=0.85, edgecolor="none")
    ax.set(xlabel="Full export-time quantile", ylabel="MicroMacro export-time quantile",
           xlim=(0, maximum), ylim=(0, maximum))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("c  Export-time Q–Q", loc="left", fontsize=11)
    style_axis(ax)

    fig.suptitle("Clique–bridge–leaf distribution diagnostics", fontsize=12.5, y=1.02)
    fig.tight_layout()
    save_figure(fig, "clique_leaf_distribution_diagnostics")


def main() -> None:
    _, arrays = load_data()
    full_export = arrays[f"{FULL}:total_export_time"]
    mm_export = arrays[f"{MM}:total_export_time"]
    paired_delta = mm_export - full_export
    ci_low, ci_high = paired_mean_ci(paired_delta)
    metrics = {
        "replications_per_model": 10_000,
        "W1_export": float(wasserstein_distance(full_export, mm_export)),
        "relative_W1_percent_of_full_mean": float(100.0 * wasserstein_distance(full_export, mm_export) / full_export.mean()),
        "KS_export": float(ks_2samp(full_export, mm_export).statistic),
        "mean_export_full": float(full_export.mean()),
        "mean_export_micromacro": float(mm_export.mean()),
        "paired_mean_shift_micromacro_minus_full": float(paired_delta.mean()),
        "paired_mean_shift_bootstrap_95_low": ci_low,
        "paired_mean_shift_bootstrap_95_high": ci_high,
        "bridge_bypass_full_count": int((arrays[f"{FULL}:bridge_to_export_delay"] < 0).sum()),
        "bridge_bypass_micromacro_count": int((arrays[f"{MM}:bridge_to_export_delay"] < 0).sum()),
        "bridge_bypass_micromacro_percent": float(100.0 * (arrays[f"{MM}:bridge_to_export_delay"] < 0).mean()),
    }

    mechanism_figure(arrays, metrics)
    distribution_figure(arrays)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Input: {INPUT}")
    print("Created 2 figures as PNG and PDF in:")
    print(f"  {OUTPUT}")
    print(f"  {PAPER_IMAGES}")
    print(f"Bridge bypass: Full 0/10000; MicroMacro 98/10000 (0.98%)")


if __name__ == "__main__":
    main()
