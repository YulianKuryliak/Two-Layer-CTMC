#!/usr/bin/env python3
"""Plot terminal-community arrival error across ASPL and community size.

Reads saved independent-RNG checkpoints only. A cell is assigned a metric only
when all expected paired runs have an observed arrival in community 10 for both
simulators. Incomplete/missing cells are explicitly labeled with coverage.
"""
from __future__ import annotations

import csv
import gzip
import json
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
SIZES = (50, 100, 250, 1000)
ASPL_VALUES = (1, 2, 3, 4, 5, 6)
K_VALUES = (1, 5)
EXPECTED_PAIRS = 400
DATA_TEMPLATE = "l10_n{n}_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
DATA_ROOTS = {n: ROOT / "output" / "data" / DATA_TEMPLATE.format(n=n) for n in SIZES}
OUT = ROOT / "output" / "analysis" / "l10_chain_aspl_size_heatmap"
FIGURES = OUT / "figures"
STEM = "aspl_chain_size_terminal_arrival_heatmap"
CACHE = OUT / "terminal_metrics.csv"
N1000_METRICS = ROOT / "output" / "analysis" / DATA_TEMPLATE.format(n=1000) / "median_layer_metrics.csv"
NAME_RE = re.compile(r"__aspl(\d+)__k(\d+)__")
ARRIVALS_RE = re.compile(rb'"arrivals":(\[[^\]]*\])')
FULL_NAME = b'"simulator":"Full Micro"'
MM_NAME = b'"simulator":"Production MicroMacro + independent RNG"'


def read_dataset(data_root: Path, n: int):
    """Read each saved checkpoint once and group pairs by (ASPL, k)."""
    if not data_root.exists():
        return defaultdict(list)
    params_path = data_root / "experiment_parameters.json"
    if not params_path.exists():
        raise RuntimeError(f"Missing experiment_parameters.json: {data_root}")
    params = json.loads(params_path.read_text(encoding="utf-8"))
    if params.get("independent_rng_per_micro_model") is not True:
        raise RuntimeError(f"Not an independent-RNG dataset: {data_root}")
    if "random" not in str(params.get("initial_node", "")).lower():
        raise RuntimeError(f"Initial node is not recorded as random: {data_root}")
    if int(params.get("community_size", params.get("n", n))) != n:
        raise RuntimeError(f"Community-size metadata mismatch in {data_root}")

    cells = defaultdict(list)
    for scenario in sorted((data_root / "scenarios").glob("*")):
        match = NAME_RE.search(scenario.name)
        if not match:
            continue
        aspl, k = map(int, match.groups())
        if aspl not in ASPL_VALUES or k not in K_VALUES:
            continue
        for checkpoint in sorted((scenario / "checkpoints").glob("*.json.gz")):
            with gzip.open(checkpoint, "rb") as handle:
                raw = handle.read()
            if raw.count(FULL_NAME) != 1 or raw.count(MM_NAME) != 1:
                raise RuntimeError(f"Unexpected simulator pair: {checkpoint}")
            arrival_arrays = ARRIVALS_RE.findall(raw)
            if len(arrival_arrays) != 2:
                raise RuntimeError(f"Expected two arrival arrays: {checkpoint}")
            full, mm = (json.loads(item) for item in arrival_arrays)
            if len(full) != 10 or len(mm) != 10:
                raise RuntimeError(f"Expected 10 communities: {checkpoint}")
            extended = False
            if full[9] is None:
                extension_path = scenario / "checkpoints_full_to40000" / checkpoint.name
                if extension_path.exists():
                    with gzip.open(extension_path, "rt", encoding="utf-8") as handle:
                        extension = json.load(handle)
                    original = json.loads(raw)
                    if not (
                        extension.get("source_checkpoint") == str(checkpoint.relative_to(ROOT))
                        and extension.get("task") == original.get("task")
                        and extension.get("full_event_seed") == original["event_seeds"]["full_event_seed"]
                        and extension.get("initial_node") == original["task"]["initial_node"]
                        and extension.get("original_T_max") == original["T_max"] == 30000.0
                        and extension.get("T_max") == 40000.0
                        and extension.get("pathwise_prefix_verified") is True
                        and extension.get("original_event_count") == len(original["runs"][0]["infection_events"])
                        and extension["full_run"]["simulator"] == "Full Micro"
                    ):
                        raise RuntimeError(f"Invalid Full extension: {extension_path}")
                    full = extension["full_run"]["arrivals"]
                    extended = True
            # Index 9 is the ninth post-seed community arrival (T_9), cumulative
            # from simulation start, matching community=9 in layer_metrics.csv.
            cells[aspl, k].append((full[9], mm[9], extended))
        print(f"[read] n={n}, ASPL={aspl}, k={k}: {len(cells[aspl, k])}/{EXPECTED_PAIRS} checkpoints", flush=True)
    return cells


def gather():
    shifts = {k: np.full((len(ASPL_VALUES), len(SIZES)), np.nan) for k in K_VALUES}
    w1s = {k: np.full((len(ASPL_VALUES), len(SIZES)), np.nan) for k in K_VALUES}
    coverage = {}
    cache_rows = {}
    if CACHE.exists():
        with CACHE.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (int(row["n"]), int(row["target_ASPL"]), int(row["k"]))
                cache_rows[key] = row
    n1000_rows = {}
    if N1000_METRICS.exists():
        with N1000_METRICS.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["community"]) == 9:
                    n1000_rows[(int(row["target_ASPL"]), int(row["k"]))] = row
    output_rows = []
    for ni, n in enumerate(SIZES):
        cells = None
        for ai, aspl in enumerate(ASPL_VALUES):
            for k in K_VALUES:
                if not DATA_ROOTS[n].exists():
                    coverage[aspl, n, k] = (0, 0, 0)
                    continue
                cached = cache_rows.get((n, aspl, k))
                if cached and int(cached["checkpoint_count"]) == EXPECTED_PAIRS and int(cached["paired_observed"]) == EXPECTED_PAIRS:
                    coverage[aspl, n, k] = (EXPECTED_PAIRS, EXPECTED_PAIRS, 0)
                    shifts[k][ai, ni] = float(cached["signed_median_shift_percent"])
                    w1s[k][ai, ni] = float(cached["relative_w1_percent"])
                    output_rows.append({"n": n, "target_ASPL": aspl, "k": k,
                                        "checkpoint_count": EXPECTED_PAIRS, "paired_observed": EXPECTED_PAIRS,
                                        "extended_full_to40000": cached.get("extended_full_to40000", "0"),
                                        "signed_median_shift_percent": shifts[k][ai, ni],
                                        "relative_w1_percent": w1s[k][ai, ni]})
                    continue
                if n == 1000 and (aspl, k) in n1000_rows:
                    row = n1000_rows[aspl, k]
                    if int(row["paired_runs"]) == EXPECTED_PAIRS:
                        coverage[aspl, n, k] = (EXPECTED_PAIRS, EXPECTED_PAIRS, 0)
                        shifts[k][ai, ni] = float(row["signed_relative_median_shift_percent"])
                        w1s[k][ai, ni] = float(row["relative_w1_percent"])
                        output_rows.append({"n": n, "target_ASPL": aspl, "k": k,
                                            "checkpoint_count": EXPECTED_PAIRS, "paired_observed": EXPECTED_PAIRS,
                                            "extended_full_to40000": 0,
                                            "signed_median_shift_percent": shifts[k][ai, ni],
                                            "relative_w1_percent": w1s[k][ai, ni]})
                        continue
                if cells is None:
                    cells = read_dataset(DATA_ROOTS[n], n)
                paired = cells.get((aspl, k), [])
                count = len(paired)
                finite = [(a, b) for a, b, _ in paired if a is not None and b is not None]
                extended_count = sum(extended for _, _, extended in paired)
                coverage[aspl, n, k] = (count, len(finite), len(paired) - len(finite))
                if count != EXPECTED_PAIRS or len(finite) != EXPECTED_PAIRS:
                    output_rows.append({"n": n, "target_ASPL": aspl, "k": k,
                                        "checkpoint_count": count, "paired_observed": len(finite),
                                        "extended_full_to40000": extended_count,
                                        "signed_median_shift_percent": "", "relative_w1_percent": ""})
                    continue
                full = np.asarray([a for a, _ in finite], dtype=float)
                mm = np.asarray([b for _, b in finite], dtype=float)
                median_full = float(np.median(full))
                if median_full <= 0:
                    raise RuntimeError(f"Invalid Full median at ASPL={aspl}, n={n}, k={k}")
                shifts[k][ai, ni] = 100 * (float(np.median(mm)) - median_full) / median_full
                w1s[k][ai, ni] = 100 * wasserstein_distance(full, mm) / float(np.mean(full))
                output_rows.append({"n": n, "target_ASPL": aspl, "k": k,
                                    "checkpoint_count": count, "paired_observed": len(finite),
                                    "extended_full_to40000": extended_count,
                                    "signed_median_shift_percent": shifts[k][ai, ni],
                                    "relative_w1_percent": w1s[k][ai, ni]})
                print(f"[complete] ASPL={aspl}, n={n}, k={k}: 400/400 paired arrivals")
    if output_rows:
        OUT.mkdir(parents=True, exist_ok=True)
        with CACHE.open("w", newline="", encoding="utf-8") as handle:
            fields = ["n", "target_ASPL", "k", "checkpoint_count", "paired_observed", "extended_full_to40000",
                      "signed_median_shift_percent", "relative_w1_percent"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(output_rows)
    return shifts, w1s, coverage


def draw_panel(ax, values, title, signed, limit, coverage, k, field):
    cmap = plt.get_cmap("RdBu_r" if signed else "YlOrBr").copy()
    cmap.set_bad("#e2e2e2")
    masked = np.ma.masked_invalid(values)
    kwargs = {"norm": TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)} if signed else {"vmin": 0, "vmax": limit}
    image = ax.imshow(masked, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, **kwargs)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(SIZES)), [str(x) for x in SIZES])
    ax.set_yticks(range(len(ASPL_VALUES)), [str(x) for x in ASPL_VALUES])
    ax.set_xlabel("community size n")
    ax.set_ylabel("target internal ASPL")
    ax.set_xticks(np.arange(-.5, len(SIZES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ASPL_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for ai, aspl in enumerate(ASPL_VALUES):
        for ni, n in enumerate(SIZES):
            value = values[ai, ni]
            if np.isfinite(value):
                ax.text(ni, ai, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if (abs(value) > .58 * limit if signed else value > .58 * limit) else "#222222")
            else:
                count, observed, _ = coverage[aspl, n, k]
                label = f"{observed}/{EXPECTED_PAIRS}" if DATA_ROOTS[n].exists() else "—"
                ax.text(ni, ai, label, ha="center", va="center", fontsize=7.2, color="#444444")
    return image


def main():
    shifts, w1s, coverage = gather()
    if not any(np.isfinite(x).any() for x in (*shifts.values(), *w1s.values())):
        raise RuntimeError("No complete 400-pair cells found; no metric heatmap can be drawn yet")
    shift_limit = max(1.0, max(float(np.nanmax(np.abs(x))) for x in shifts.values() if np.isfinite(x).any()))
    w1_limit = max(1.0, max(float(np.nanmax(x)) for x in w1s.values() if np.isfinite(x).any()))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0), layout="constrained")
    for row, k in enumerate(K_VALUES):
        left = draw_panel(axes[row, 0], shifts[k], f"k = {k}  |  signed median shift", True, shift_limit, coverage, k, "shift")
        right = draw_panel(axes[row, 1], w1s[k], f"k = {k}  |  relative W₁", False, w1_limit, coverage, k, "w1")
    fig.colorbar(left, ax=axes[:, 0], shrink=.82, pad=.03,
                 label="100 × (median T₉, MM − median T₉, Full) / median T₉, Full (%)")
    fig.colorbar(right, ax=axes[:, 1], shrink=.82, pad=.03,
                 label="100 × W₁ / mean T₉, Full (%)")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{STEM}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{STEM}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {FIGURES / (STEM + '.png')}")
    print("Gray cells show observed paired arrivals/400; em dash means no dataset is available yet.")


if __name__ == "__main__":
    main()
