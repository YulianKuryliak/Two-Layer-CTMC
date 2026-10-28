#!/usr/bin/env python3
"""Plot the completed L=10, ASPL=1..6 chain experiment from saved checkpoints.

No simulations or network generation are performed.  Ordinary arrival-time
distances are shown only when every expected paired run reached that layer in
both simulators; otherwise the cell is masked and completion is reported.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "l10_n1000_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
DATA = ROOT / "output" / "data" / EXPERIMENT
OUT = ROOT / "output" / "analysis" / EXPERIMENT
FIGURES = OUT / "figures"
ASPL_VALUES = (1, 2, 3, 4, 5, 6)
K_VALUES = (1, 5)
LAYERS = tuple(range(1, 10))
EXPECTED_PAIRS = 400
ARRIVALS_RE = re.compile(rb'"arrivals":(\[[^\]]*\])')
NAME_RE = re.compile(r"__aspl(\d+)__k(\d+)__")
BLUE, ORANGE = "#2878B5", "#D45A44"


def read_pairs() -> dict[tuple[int, int], list[tuple[list[float | None], list[float | None]]]]:
    params = json.loads((DATA / "experiment_parameters.json").read_text(encoding="utf-8"))
    if params.get("independent_rng_per_micro_model") is not True or "random" not in str(params.get("initial_node", "")):
        raise RuntimeError("wrong RNG or initial-node policy for this figure set")
    scenarios = sorted((DATA / "scenarios").glob("*"))
    pairs: dict[tuple[int, int], list[tuple[list[float | None], list[float | None]]]] = defaultdict(list)
    for scenario in scenarios:
        match = NAME_RE.search(scenario.name)
        if match is None:
            continue
        aspl, k = map(int, match.groups())
        if (aspl, k) not in {(a, b) for a in ASPL_VALUES for b in K_VALUES}:
            continue
        checkpoints = sorted((scenario / "checkpoints").glob("*.json.gz"))
        for path in checkpoints:
            with gzip.open(path, "rb") as handle:
                raw = handle.read()
            if raw.count(b'"simulator":"Full Micro"') != 1 or raw.count(b'"simulator":"Production MicroMacro + independent RNG"') != 1:
                raise RuntimeError(f"simulator identity mismatch: {path}")
            matches = ARRIVALS_RE.findall(raw)
            if len(matches) != 2:
                raise RuntimeError(f"expected two arrival arrays: {path}")
            full, mm = (json.loads(m) for m in matches)
            if len(full) != 10 or len(mm) != 10:
                raise RuntimeError(f"expected ten communities: {path}")
            pairs[(aspl, k)].append((full, mm))
        print(f"[read] ASPL={aspl}, k={k}: {len(checkpoints)}/{EXPECTED_PAIRS} pairs", flush=True)
    expected_keys = {(a, k) for a in ASPL_VALUES for k in K_VALUES}
    if set(pairs) != expected_keys:
        raise RuntimeError(f"missing scenario cells: {sorted(expected_keys - set(pairs))}")
    return pairs


def make_metrics(pairs):
    rows = []
    for aspl in ASPL_VALUES:
        for k in K_VALUES:
            cell = pairs[(aspl, k)]
            for layer in LAYERS:
                full = np.array([p[0][layer] for p in cell if p[0][layer] is not None], dtype=float)
                mm = np.array([p[1][layer] for p in cell if p[1][layer] is not None], dtype=float)
                paired = [(p[0][layer], p[1][layer]) for p in cell if p[0][layer] is not None and p[1][layer] is not None]
                fully_observed = len(cell) == EXPECTED_PAIRS and len(paired) == EXPECTED_PAIRS
                mean_full = float(np.mean(full)) if len(full) else math.nan
                mean_mm = float(np.mean(mm)) if len(mm) else math.nan
                signed = 100 * (mean_mm - mean_full) / mean_full if fully_observed and mean_full else math.nan
                w1 = 100 * wasserstein_distance(full, mm) / mean_full if fully_observed and mean_full else math.nan
                if layer == 1:
                    d_full = np.array([p[0][1] - p[0][0] for p in cell if p[0][1] is not None], dtype=float)
                    d_mm = np.array([p[1][1] - p[1][0] for p in cell if p[1][1] is not None], dtype=float)
                else:
                    d_full = np.array([p[0][layer] - p[0][layer - 1] for p in cell if p[0][layer] is not None and p[0][layer - 1] is not None], dtype=float)
                    d_mm = np.array([p[1][layer] - p[1][layer - 1] for p in cell if p[1][layer] is not None and p[1][layer - 1] is not None], dtype=float)
                local = 100 * (float(np.mean(d_mm)) - float(np.mean(d_full))) / float(np.mean(d_full)) if fully_observed and len(d_full) == EXPECTED_PAIRS and len(d_mm) == EXPECTED_PAIRS and np.mean(d_full) else math.nan
                rows.append({
                    "target_ASPL": aspl, "k": k, "community": layer,
                    "pairs_available": len(cell), "full_observed": len(full), "mm_observed": len(mm), "both_observed": len(paired),
                    "full_completion_percent": 100 * len(full) / len(cell), "mm_completion_percent": 100 * len(mm) / len(cell),
                    "mean_full": mean_full, "mean_mm": mean_mm,
                    "signed_relative_mean_shift_percent": signed,
                    "relative_w1_percent": w1,
                    "local_delta_relative_mean_shift_percent": local,
                })
    return rows


def matrix(rows, k: int, field: str) -> np.ndarray:
    lookup = {(int(r["target_ASPL"]), int(r["k"]), int(r["community"])): r for r in rows}
    return np.array([[float(lookup[(aspl, k, layer)][field]) for layer in LAYERS] for aspl in ASPL_VALUES])


def format_heatmap(ax, values: np.ndarray, title: str, *, signed: bool, limit: float):
    cmap = plt.get_cmap("RdBu_r" if signed else "YlOrBr").copy()
    cmap.set_bad("#dedede")
    masked = np.ma.masked_invalid(values)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit) if signed else None
    image = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
                      norm=norm, vmin=None if signed else 0, vmax=None if signed else limit)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(LAYERS)), LAYERS)
    ax.set_yticks(range(len(ASPL_VALUES)), ASPL_VALUES)
    ax.set_xlabel("community reached")
    ax.set_ylabel("target internal ASPL")
    ax.set_xticks(np.arange(-.5, len(LAYERS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ASPL_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y, x in zip(*np.where(np.isnan(values))):
        ax.text(x, y, "×", ha="center", va="center", color="#333333", fontsize=14)
    return image


def save(fig, stem: str):
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_main(rows):
    shifts = [matrix(rows, k, "signed_relative_mean_shift_percent") for k in K_VALUES]
    distances = [matrix(rows, k, "relative_w1_percent") for k in K_VALUES]
    shift_limit = max(1.0, max(float(np.nanmax(np.abs(x))) for x in shifts if np.isfinite(x).any()))
    distance_limit = max(1.0, max(float(np.nanmax(x)) for x in distances if np.isfinite(x).any()))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), layout="constrained")
    for row, k in enumerate(K_VALUES):
        left = format_heatmap(axes[row, 0], shifts[row], f"k = {k}  |  signed mean-arrival shift (%)", signed=True, limit=shift_limit)
        right = format_heatmap(axes[row, 1], distances[row], f"k = {k}  |  relative W₁ (%)", signed=False, limit=distance_limit)
    fig.colorbar(left, ax=axes[:, 0], shrink=.78, pad=.02,
                 label="100 × (mean MM − mean Full) / mean Full, percent")
    fig.colorbar(right, ax=axes[:, 1], shrink=.78, pad=.02,
                 label="100 × W₁ / mean Full, percent")
    save(fig, "aspl_chain_arrival_error_map")


def ecdf_times(arrivals: list[float | None], denominator: int):
    observed = np.sort(np.array([x for x in arrivals if x is not None], dtype=float))
    if len(observed) == 0:
        return np.array([0.0]), np.array([0.0])
    return np.r_[0, observed], np.r_[0, np.arange(1, len(observed) + 1) / denominator]


def plot_final_arrivals(pairs):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.7), layout="constrained")
    for row, k in enumerate(K_VALUES):
        for col, aspl in enumerate((1, 3, 6)):
            ax = axes[row, col]
            cell = pairs[(aspl, k)]
            for index, (label, color) in enumerate((("Full Micro", BLUE), ("MicroMacro", ORANGE))):
                arrivals = [p[index][9] for p in cell]
                x, y = ecdf_times(arrivals, len(cell))
                ax.step(x, y, where="post", color=color, linewidth=1.9, label=f"{label} ({len(x)-1}/{len(cell)})")
            ax.set_title(f"k = {k}, ASPL = {aspl}")
            ax.set_xlabel("first arrival in community 9")
            ax.set_ylabel("fraction arrived by time t")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=.22)
            ax.legend(fontsize=8, loc="lower right")
    save(fig, "aspl_chain_last_arrival_ecdf")


def plot_local(rows):
    values = [matrix(rows, k, "local_delta_relative_mean_shift_percent") for k in K_VALUES]
    limit = max(1.0, max(float(np.nanmax(np.abs(x))) for x in values if np.isfinite(x).any()))
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.0), layout="constrained")
    for ax, k, data in zip(axes, K_VALUES, values):
        image = format_heatmap(ax, data, f"k = {k}  |  local transition-time shift", signed=True, limit=limit)
    fig.colorbar(image, ax=axes, shrink=.8, pad=.02, label="100 × (mean ΔMM − mean ΔFull) / mean ΔFull (%)")
    save(fig, "aspl_chain_local_transition_error")


def plot_completion(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.0), layout="constrained")
    for row, k in enumerate(K_VALUES):
        for col, (model, field) in enumerate((("Full Micro", "full_completion_percent"), ("MicroMacro", "mm_completion_percent"))):
            image = format_heatmap(axes[row, col], matrix(rows, k, field), f"k = {k}  |  {model} completion", signed=False, limit=100)
    fig.colorbar(image, ax=axes, shrink=.8, pad=.02, label="community reached by T_max (%)")
    save(fig, "aspl_chain_completion_map")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs()
    rows = make_metrics(pairs)
    with (OUT / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing = [r for r in rows if r["both_observed"] != EXPECTED_PAIRS or r["pairs_available"] != EXPECTED_PAIRS]
    print(f"[audit] {len(rows)} layer cells; {len(missing)} incomplete/censored cells", flush=True)
    if missing:
        print("[audit] ordinary bias and W1 are masked for incomplete/censored cells", flush=True)
    plot_main(rows)
    plot_final_arrivals(pairs)
    plot_local(rows)
    if missing:
        plot_completion(rows)
    print(f"Figures and metrics: {OUT}", flush=True)


if __name__ == "__main__":
    main()
