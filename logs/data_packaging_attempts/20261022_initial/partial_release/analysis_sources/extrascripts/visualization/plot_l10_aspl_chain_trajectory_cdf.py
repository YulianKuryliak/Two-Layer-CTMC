#!/usr/bin/env python3
"""Plot SI propagation and final-community arrival CDFs for every ASPL-chain cell.

Only saved paired checkpoints are read.  Each row compares Full Micro and
Production MicroMacro with independent RNG on the same 400 network/event pairs.
The shaded trajectory band is the run-to-run IQR, not a confidence interval.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "l10_n1000_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
DATA = ROOT / "output" / "data" / EXPERIMENT
OUT = ROOT / "output" / "analysis" / EXPERIMENT
FIGURES = OUT / "figures"
CACHE = OUT / "trajectory_cdf_cache"
ASPL_VALUES = (1, 2, 3, 4, 5, 6)
K_VALUES = (1, 5)
EXPECTED_PAIRS = 400
TOTAL_NODES = 10_000
GRID_POINTS = 501
NAME_RE = re.compile(r"__aspl(\d+)__k(\d+)__")
BLUE, ORANGE = "#2878B5", "#D45A44"
SIMULATORS = ("Full Micro", "Production MicroMacro + independent RNG")


def scenario_paths() -> dict[tuple[int, int], Path]:
    params = json.loads((DATA / "experiment_parameters.json").read_text(encoding="utf-8"))
    if params.get("independent_rng_per_micro_model") is not True:
        raise RuntimeError("the saved experiment does not use independent MicroModel RNG")
    if "random" not in str(params.get("initial_node", "")):
        raise RuntimeError("the saved experiment does not use random initial nodes")
    found = {}
    for path in (DATA / "scenarios").iterdir():
        match = NAME_RE.search(path.name)
        if match:
            found[tuple(map(int, match.groups()))] = path
    expected = {(a, k) for a in ASPL_VALUES for k in K_VALUES}
    if set(found) != expected:
        raise RuntimeError(f"scenario mismatch: missing={sorted(expected - set(found))}, extra={sorted(set(found) - expected)}")
    return found


def source_files(scenario: Path) -> list[Path]:
    paths = sorted((scenario / "checkpoints").glob("*.json.gz"))
    if len(paths) != EXPECTED_PAIRS:
        raise RuntimeError(f"{scenario.name}: expected {EXPECTED_PAIRS} checkpoints, found {len(paths)}")
    return paths


def cache_path(aspl: int, k: int) -> Path:
    return CACHE / f"aspl{aspl}_k{k}.npz"


def cached_cell(paths: list[Path], aspl: int, k: int):
    target = cache_path(aspl, k)
    if not target.exists() or target.stat().st_mtime_ns < max(p.stat().st_mtime_ns for p in paths):
        return None
    with np.load(target) as saved:
        if int(saved["checkpoint_count"]) != len(paths):
            return None
        return {name: saved[name] for name in saved.files}


def read_cell(paths: list[Path], aspl: int, k: int):
    times = [[], []]
    arrivals = [[], []]
    for number, path in enumerate(paths, 1):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        runs = checkpoint.get("runs", [])
        if [run.get("simulator") for run in runs] != list(SIMULATORS):
            raise RuntimeError(f"unexpected model identity or order: {path}")
        for index, run in enumerate(runs):
            events = run["infection_events"]
            if len(events) != TOTAL_NODES or run["termination"]["stop_reason"] != "all_nodes_infected":
                raise RuntimeError(f"incomplete SI trajectory: {path} / {run['simulator']}")
            infection_times = np.fromiter((event[0] for event in events), dtype=np.float64, count=TOTAL_NODES)
            if not np.any(infection_times == 0):
                raise RuntimeError(f"missing initial infection: {path} / {run['simulator']}")
            # MicroMacro records can be appended in scheduler order rather than
            # physical-time order.  Sorting restores the exact cumulative I(t).
            infection_times.sort()
            arrival = run["arrivals"][9]
            if arrival is None or not np.isfinite(arrival):
                raise RuntimeError(f"censored final-community arrival: {path} / {run['simulator']}")
            times[index].append(infection_times)
            arrivals[index].append(float(arrival))
        if number % 100 == 0:
            print(f"  ASPL={aspl}, k={k}: read {number}/{EXPECTED_PAIRS}", flush=True)

    full_arrivals, mm_arrivals = (np.asarray(x, dtype=np.float64) for x in arrivals)
    final_event_times = np.asarray([series[-1] for group in times for series in group])
    # The same per-row window is used for both panels and both simulators.
    # Cover every T_9 and at least 99% of the final-infection times.
    horizon = 1.05 * max(float(np.max(full_arrivals)), float(np.max(mm_arrivals)),
                         float(np.quantile(final_event_times, .99)))
    grid = np.linspace(0, horizon, GRID_POINTS)
    summaries = []
    for group in times:
        counts = np.asarray([np.searchsorted(series, grid, side="right") for series in group], dtype=np.int16)
        summaries.append(np.quantile(counts, [.25, .5, .75], axis=0))
    result = {
        "checkpoint_count": np.array(len(paths)),
        "grid": grid,
        "full_quantiles": summaries[0],
        "mm_quantiles": summaries[1],
        "full_arrivals": full_arrivals,
        "mm_arrivals": mm_arrivals,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(aspl, k), **result)
    return result


def load_cells():
    cells = {}
    for (aspl, k), scenario in sorted(scenario_paths().items()):
        paths = source_files(scenario)
        cell = cached_cell(paths, aspl, k)
        if cell is None:
            print(f"[read] ASPL={aspl}, k={k}: extracting trajectories from checkpoints", flush=True)
            cell = read_cell(paths, aspl, k)
        else:
            print(f"[cache] ASPL={aspl}, k={k}: {len(paths)} paired runs", flush=True)
        cells[(aspl, k)] = cell
    return cells


def ecdf(values: np.ndarray):
    ordered = np.sort(values)
    return np.r_[0, ordered], np.r_[0, np.arange(1, len(ordered) + 1) / len(ordered)]


def draw_sheet(cells, selected: list[tuple[int, int]], stem: str):
    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(12.3, 2.4 * rows + .5),
                             squeeze=False, layout="constrained")
    for row, (aspl, k) in enumerate(selected):
        cell = cells[(aspl, k)]
        left, right = axes[row]
        grid = cell["grid"]
        for key, arr_key, color, label in (
            ("full_quantiles", "full_arrivals", BLUE, "Full Micro"),
            ("mm_quantiles", "mm_arrivals", ORANGE, "MicroMacro"),
        ):
            q25, median, q75 = cell[key]
            left.fill_between(grid, q25, q75, color=color, alpha=.15, linewidth=0)
            left.plot(grid, median, color=color, linestyle="-", linewidth=1.9, label=label)
            x, y = ecdf(cell[arr_key])
            right.step(x, y, where="post", color=color, linestyle="-",
                       linewidth=1.9, label=label)
        left.set_xlim(0, grid[-1])
        right.set_xlim(0, grid[-1])
        left.set_ylim(0, TOTAL_NODES * 1.03)
        right.set_ylim(0, 1.025)
        left.set_ylabel(f"ASPL {aspl}, k={k}\nInfected nodes")
        right.set_ylabel("CDF of $T_9$")
        left.grid(alpha=.18)
        right.grid(alpha=.18)
        left.spines[["top", "right"]].set_visible(False)
        right.spines[["top", "right"]].set_visible(False)
        if row == 0:
            left.set_title("Propagation: median and interquartile range")
            right.set_title("First arrival in the last community")
        if row == rows - 1:
            left.set_xlabel("Time")
            right.set_xlabel("Time")
    FIGURES.mkdir(parents=True, exist_ok=True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {stem}.pdf/.png", flush=True)


def main():
    cells = load_cells()
    for k in K_VALUES:
        draw_sheet(cells, [(aspl, k) for aspl in ASPL_VALUES], f"aspl_chain_trajectory_cdf_k{k}")
    draw_sheet(cells, [(aspl, k) for aspl in ASPL_VALUES for k in K_VALUES],
               "aspl_chain_trajectory_cdf_all")
    print(f"Figures: {FIGURES}", flush=True)


if __name__ == "__main__":
    main()
