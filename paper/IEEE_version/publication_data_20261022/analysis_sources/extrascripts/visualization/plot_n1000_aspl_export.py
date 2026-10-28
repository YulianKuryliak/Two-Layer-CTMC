#!/usr/bin/env python3
"""Plot the combined n=1000 random-initial ASPL 1–6 export experiment.

Reads saved paired checkpoints only. Run from the repository root:
    .venv/bin/python3 extrascripts/visualization/plot_n1000_aspl_export.py

Outputs PNG, PDF, and a metrics table in
output/analysis/n1000_random_connected_tree_augmented_aspl1to6_rep400_random_initial/figures/.
"""
from __future__ import annotations

import csv
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance


ROOT = Path(__file__).resolve().parents[2]
LEGACY_NAME = "n1000_random_connected_tree_augmented_aspl_export"
NAME = "n1000_random_connected_tree_augmented_aspl1to6_rep400_random_initial"
SOURCE = ROOT / "output" / "data" / NAME
FIGURES = ROOT / "output" / "analysis" / NAME / "figures"
FULL = "Full Micro"
MM = "Production MicroMacro + independent RNG"
COLORS = {FULL: "#2878b5", MM: "#d0604c"}
BOOTSTRAP_DRAWS = 1500
BOOTSTRAP_SEED = 20260922


def load_cells(data_dir: Path | None = None, accepted_experiments: set[str] | None = None) -> tuple[dict[tuple[float, int], list[dict]], dict]:
    data_dir = data_dir or SOURCE
    allowed = accepted_experiments or {data_dir.name}
    parameters = json.loads((data_dir / "experiment_parameters.json").read_text(encoding="utf-8"))
    cells: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for path in sorted((data_dir / "scenarios").glob("*/checkpoints/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        task = payload["task"]
        expected_experiment = parameters.get("checkpoint_experiment_by_ASPL", {}).get(
            str(float(task["target_ASPL"])))
        accepted = {expected_experiment} if expected_experiment else allowed
        if payload.get("experiment") not in accepted or not payload.get("rng_policy", {}).get("independent_rng_per_MicroModel"):
            raise ValueError(f"incompatible checkpoint: {path}")
        repair = payload.get("repair")
        if expected_experiment and expected_experiment.endswith("_full_corrected") and repair is None:
            raise ValueError(f"missing Full Micro correction: {path}")
        if repair is not None and (repair.get("corrected_initial_node") != task.get("initial_node")
                                   or not repair.get("micromacro_reused")):
            raise ValueError(f"corrected Full/MicroMacro pairing metadata mismatch: {path}")
        if parameters.get("initial_node_policy", "").find("random node") < 0:
            raise ValueError(f"dataset does not declare the shared random initial-node policy: {data_dir}")
        runs = {run["simulator"]: run for run in payload["runs"]}
        if set(runs) != {FULL, MM}:
            raise ValueError(f"expected one run per simulator: {path}")
        row = {
            "lineage": int(task["lineage_id"]),
            "interface": int(task["interface_id"]),
            "actual_aspl": float(task["actual_ASPL"]),
            "distance_to_bridge": min(task["seed_bridge_distances"]),
            "full_export": runs[FULL]["T_export"],
            "mm_export": runs[MM]["T_export"],
            "full_bridge": runs[FULL]["T_bridge_first"],
            "mm_bridge": runs[MM]["T_bridge_first"],
            "full_bypass": runs[FULL]["bypass"],
            "mm_bypass": runs[MM]["bypass"],
        }
        cells[(float(task["target_ASPL"]), int(task["k"]))].append(row)

    targets = [float(value) for value in parameters["ASPL_targets"]]
    ks = [int(value) for value in parameters["k_values"]]
    if set(cells) != {(target, k) for target in targets for k in ks}:
        raise ValueError("ASPL/k cells do not match experiment_parameters.json")
    expected_runs = int(parameters["replications_per_scenario"])
    expected_lineages = int(parameters["graph_lineages"])
    expected_interfaces = int(parameters["interfaces_per_lineage"])
    for key, rows in cells.items():
        pairs = {(row["lineage"], row["interface"]) for row in rows}
        by_lineage = defaultdict(int)
        for row in rows:
            by_lineage[row["lineage"]] += 1
        if (len(rows) != expected_runs or len(pairs) != expected_runs
                or len(by_lineage) != expected_lineages
                or set(by_lineage.values()) != {expected_interfaces}):
            raise ValueError(f"incomplete {expected_lineages} graphs × {expected_interfaces} interfaces cell: {key}")
        if any(row[field] is None for row in rows for field in
               ("full_export", "mm_export", "full_bridge", "mm_bridge")):
            raise ValueError(f"unobserved export or bridge in {key}; the plotted mean/W1 requires completed runs")
        if any(row["full_bypass"] for row in rows):
            raise ValueError(f"physical Full Micro bridge bypass in {key}")
    return cells, parameters


def merge_aspl_ranges(primary: dict, extra: dict, primary_parameters: dict,
                      extra_parameters: dict) -> tuple[dict, dict]:
    """Combine disjoint ASPL ranges with identical experiment design."""
    common = ("community_size", "beta", "gamma", "model", "tau_micro", "macro_T",
              "T_max", "k_values", "replications_per_scenario", "graph_lineages",
              "interfaces_per_lineage", "event_replicates_per_interface", "topology",
              "independent_rng_per_micro_model", "initial_node_policy")
    mismatches = [key for key in common if primary_parameters.get(key) != extra_parameters.get(key)]
    if mismatches:
        raise ValueError(f"ASPL datasets use incompatible settings: {mismatches}")
    overlap = set(primary) & set(extra)
    if overlap:
        raise ValueError(f"ASPL cells appear in both input datasets: {sorted(overlap)}")
    cells = {**extra, **primary}
    targets = sorted({target for target, _ in cells})
    ks = [int(value) for value in primary_parameters["k_values"]]
    expected = {(target, k) for target in targets for k in ks}
    if set(cells) != expected:
        raise ValueError(f"combined datasets have missing ASPL/k cells: {sorted(expected - set(cells))}")
    parameters = dict(primary_parameters)
    parameters["ASPL_targets"] = targets
    parameters["mean_degree_by_target_ASPL"] = {
        **extra_parameters["mean_degree_by_target_ASPL"],
        **primary_parameters["mean_degree_by_target_ASPL"],
    }
    parameters["ASPL_tolerance"] = {
        **extra_parameters.get("ASPL_tolerance", {}),
        **primary_parameters.get("ASPL_tolerance", {}),
    }
    return cells, parameters


def measures(rows: list[dict]) -> dict[str, float]:
    full = np.asarray([row["full_export"] for row in rows], dtype=float)
    mm = np.asarray([row["mm_export"] for row in rows], dtype=float)
    full_bridge = np.asarray([row["full_bridge"] for row in rows], dtype=float)
    mm_bridge = np.asarray([row["mm_bridge"] for row in rows], dtype=float)
    return {
        "full_export_mean": float(full.mean()),
        "mm_export_mean": float(mm.mean()),
        "signed_relative_bias_pct": float(100 * (mm.mean() - full.mean()) / full.mean()),
        "relative_w1_pct": float(100 * wasserstein_distance(full, mm) / full.mean()),
        "full_bridge_mean": float(full_bridge.mean()),
        "mm_bridge_mean": float(mm_bridge.mean()),
        "mm_bypass_pct": float(100 * np.mean([row["mm_bypass"] for row in rows])),
        "full_delay_mean": float((full - full_bridge).mean()),
        "mm_delay_mean": float((mm - mm_bridge).mean()),
    }


def summarize(cells: dict[tuple[float, int], list[dict]], parameters: dict) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    lineages = int(parameters["graph_lineages"])
    result = []
    for (target, k), rows in sorted(cells.items()):
        groups = [sorted((row for row in rows if row["lineage"] == lineage),
                         key=lambda row: row["interface"]) for lineage in range(lineages)]
        point = measures(rows)
        samples = {name: np.empty(BOOTSTRAP_DRAWS) for name in point}
        for draw in range(BOOTSTRAP_DRAWS):
            sampled = [row for lineage in rng.integers(0, lineages, size=lineages) for row in groups[lineage]]
            estimate = measures(sampled)
            for name, value in estimate.items():
                samples[name][draw] = value
        entry = {
            "target_aspl": target,
            "actual_aspl_mean": float(np.mean([group[0]["actual_aspl"] for group in groups])),
            "mean_degree": parameters["mean_degree_by_target_ASPL"][str(target)],
            "k": k,
            "network_lineages": lineages,
            "paired_runs": len(rows),
            "full_completion": 100,
            "mm_completion": 100,
        }
        for name, value in point.items():
            low, high = np.quantile(samples[name], [0.025, 0.975])
            entry[name] = value
            entry[name + "_ci_low"] = float(low)
            entry[name + "_ci_high"] = float(high)
        result.append(entry)
    return result


def series(ax, rows: list[dict], name: str, label: str, color: str, marker: str,
           *, intervals: bool = True) -> None:
    x = np.asarray([row["target_aspl"] for row in rows])
    y = np.asarray([row[name] for row in rows])
    low = np.asarray([row[name + "_ci_low"] for row in rows])
    high = np.asarray([row[name + "_ci_high"] for row in rows])
    # A percentile bootstrap interval for W1 can exclude the original-sample
    # estimate. Draw absolute bounds rather than passing negative yerr values.
    ax.plot(x, y, color=color, marker=marker, markersize=4.5,
            linewidth=1.4, label=label)
    if not intervals:
        return
    # Degenerate empirical bootstrap bounds (e.g. zero observed bypasses)
    # do not establish zero population uncertainty. Do not draw them as CIs.
    visible = high > low
    x, low, high = x[visible], low[visible], high[visible]
    ax.vlines(x, low, high, color=color, linewidth=1.1)
    ax.hlines(low, x - .035, x + .035, color=color, linewidth=1.1)
    ax.hlines(high, x - .035, x + .035, color=color, linewidth=1.1)


def setup(ax, targets, *, ylabel: str, zero: bool = False) -> None:
    # For a quarter-step ASPL grid, label every other point (0.5 spacing)
    # while retaining all data points and their connecting lines.
    if len(targets) >= 10:
        tick_targets = [target for target in targets
                        if np.isclose(target * 2, round(target * 2))]
    else:
        tick_targets = targets
    if targets[-1] not in tick_targets:
        tick_targets = [*tick_targets, targets[-1]]
    ax.set_xticks(tick_targets)
    ax.set_xlim(min(targets) - .15, max(targets) + .15)
    ax.set_xlabel("target ASPL")
    ax.set_ylabel(ylabel)
    ax.grid(color="#d8dde2", alpha=.65, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    if zero:
        ax.axhline(0, color="#34383c", linewidth=.9)


def save(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def accuracy_figure(summary: list[dict], parameters: dict) -> None:
    targets = [float(value) for value in parameters["ASPL_targets"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row_idx, k in enumerate(parameters["k_values"]):
        rows = [row for row in summary if row["k"] == k]
        ax = axes[row_idx, 0]
        series(ax, rows, "full_export_mean", "Full Micro", COLORS[FULL], "o")
        series(ax, rows, "mm_export_mean", "MicroMacro", COLORS[MM], "s")
        setup(ax, targets, ylabel="mean export time")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        axes[row_idx, 0].set_title(f"k = {k} | export time")

        ax = axes[row_idx, 1]
        series(ax, rows, "signed_relative_bias_pct", "MM − Full", COLORS[MM], "o")
        setup(ax, targets, ylabel="signed relative bias (%)", zero=True)
        ax.set_title(f"k = {k} | timing bias")

        ax = axes[row_idx, 2]
        series(ax, rows, "relative_w1_pct", "relative W₁", COLORS[MM], "o")
        setup(ax, targets, ylabel="W₁ / mean Full export time (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"k = {k} | distributional error")
    save(fig, "aspl_export_accuracy")


def mechanism_figure(summary: list[dict], parameters: dict) -> None:
    targets = [float(value) for value in parameters["ASPL_targets"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row_idx, k in enumerate(parameters["k_values"]):
        rows = [row for row in summary if row["k"] == k]
        ax = axes[row_idx, 0]
        series(ax, rows, "full_bridge_mean", "Full Micro", COLORS[FULL], "o")
        series(ax, rows, "mm_bridge_mean", "MicroMacro", COLORS[MM], "s")
        setup(ax, targets, ylabel="mean first bridge infection time")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"k = {k} | bridge activation")

        ax = axes[row_idx, 1]
        series(ax, rows, "mm_bypass_pct", "MicroMacro", COLORS[MM], "o")
        setup(ax, targets, ylabel="export before every true bridge (%)")
        ax.set_ylim(bottom=0)
        ax.text(.02, .96,
                f"Full Micro: 0/{parameters['replications_per_scenario']} at every ASPL",
                transform=ax.transAxes,
                va="top", fontsize=8)
        ax.set_title(f"k = {k} | bridge bypass")

        ax = axes[row_idx, 2]
        series(ax, rows, "full_delay_mean", "Full Micro", COLORS[FULL], "o")
        series(ax, rows, "mm_delay_mean", "MicroMacro", COLORS[MM], "s")
        setup(ax, targets, ylabel="mean export − bridge time", zero=True)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"k = {k} | signed time difference")
    save(fig, "aspl_export_mechanism")


def summary_gallery(summary: list[dict], parameters: dict) -> None:
    """Compact article figure: export time, distributional error, and bypass."""
    targets = [float(value) for value in parameters["ASPL_targets"]]
    ks = [int(value) for value in parameters["k_values"]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True,
                             sharey="col", constrained_layout=True)

    for row_idx, k in enumerate(ks):
        rows = [row for row in summary if row["k"] == k]

        ax = axes[row_idx, 0]
        series(ax, rows, "full_export_mean", "Full Micro", COLORS[FULL], "o")
        series(ax, rows, "mm_export_mean", "MicroMacro", COLORS[MM], "s")
        setup(ax, targets, ylabel="mean export time")
        ax.set_ylim(bottom=0)
        ax.set_title(f"k = {k} | export time")
        if row_idx == 0:
            ax.legend(frameon=False, fontsize=8)

        ax = axes[row_idx, 1]
        series(ax, rows, "relative_w1_pct", "relative W₁", COLORS[MM], "o", intervals=False)
        setup(ax, targets, ylabel="W₁ / mean Full export time (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"k = {k} | distributional error")

        ax = axes[row_idx, 2]
        series(ax, rows, "mm_bypass_pct", "MicroMacro", COLORS[MM], "o")
        setup(ax, targets, ylabel="export before any bridge (%)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"k = {k} | bridge bypass")

    for ax in axes[0, :]:
        ax.set_xlabel("")
    save(fig, "aspl_export_summary_gallery")


def read_summary(parameters: dict, path: Path | None = None) -> list[dict]:
    summary_path = path or FIGURES / "aspl_export_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary = []
        for row in csv.DictReader(handle):
            parsed = {key: float(value) for key, value in row.items() if value != ""}
            parsed["k"] = int(parsed["k"])
            summary.append(parsed)

    expected = {(float(target), int(k)) for target in parameters["ASPL_targets"]
                for k in parameters["k_values"]}
    actual = {(row["target_aspl"], row["k"]) for row in summary}
    if actual != expected:
        raise ValueError("summary CSV cells do not match experiment_parameters.json")
    if any(int(row["paired_runs"]) != int(parameters["replications_per_scenario"])
           or row["full_completion"] != 100 or row["mm_completion"] != 100
           for row in summary):
        raise ValueError("summary CSV has incomplete or censored cells")
    return summary


def load_aspl1_summary(data_name: str, reference_parameters: dict) -> list[dict]:
    """Load and summarize the completed ASPL=1 cells from an earlier run."""
    data_dir = ROOT / "output" / "data" / data_name
    old_parameters = json.loads((data_dir / "experiment_parameters.json").read_text(encoding="utf-8"))
    comparable_fields = ("community_size", "beta", "gamma", "model", "tau_micro", "k_values")
    mismatches = [field for field in comparable_fields
                  if old_parameters.get(field) != reference_parameters.get(field)]
    if mismatches or not old_parameters.get("independent_rng_per_micro_model"):
        raise ValueError(f"ASPL=1 source has incompatible settings: {mismatches}")
    if (old_parameters.get("topology") != "random_connected"
            or old_parameters.get("mean_degree_by_target_ASPL", {}).get("1.0") != 999.0
            or 1.0 not in [float(value) for value in old_parameters.get("ASPL_targets", [])]):
        raise ValueError("ASPL=1 source is not the complete-graph random_connected experiment")

    cells: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for path in sorted((data_dir / "scenarios").glob("*/checkpoints/*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (payload.get("experiment") != data_name
                or not payload.get("rng_policy", {}).get("independent_rng_per_MicroModel")):
            continue
        task = payload["task"]
        if float(task["target_ASPL"]) != 1.0:
            continue
        runs = {run["simulator"]: run for run in payload["runs"]}
        if set(runs) != {FULL, MM}:
            raise ValueError(f"expected paired simulators in {path}")
        row = {
            "lineage": int(task["lineage_id"]),
            "interface": int(task["interface_id"]),
            "actual_aspl": float(task["actual_ASPL"]),
            "distance_to_bridge": min(task["seed_bridge_distances"]),
            "full_export": runs[FULL]["T_export"],
            "mm_export": runs[MM]["T_export"],
            "full_bridge": runs[FULL]["T_bridge_first"],
            "mm_bridge": runs[MM]["T_bridge_first"],
            "full_bypass": runs[FULL]["bypass"],
            "mm_bypass": runs[MM]["bypass"],
        }
        if any(row[field] is None for field in
               ("full_export", "mm_export", "full_bridge", "mm_bridge")):
            raise ValueError(f"censored ASPL=1 run in {path}")
        if row["full_bypass"]:
            raise ValueError(f"physical Full Micro bridge bypass in {path}")
        cells[(1.0, int(task["k"]))].append(row)

    ks = [int(value) for value in reference_parameters["k_values"]]
    expected_runs = int(old_parameters["replications_per_scenario"])
    if set(cells) != {(1.0, k) for k in ks}:
        raise ValueError("ASPL=1 source does not contain both requested k cells")
    for key, rows in cells.items():
        pairs = {(row["lineage"], row["interface"]) for row in rows}
        by_lineage = defaultdict(int)
        for row in rows:
            by_lineage[row["lineage"]] += 1
        expected_lineages = int(old_parameters["graph_lineages"])
        expected_interfaces = int(old_parameters["interfaces_per_lineage"])
        if (len(rows) != expected_runs or len(pairs) != expected_runs
                or len(by_lineage) != expected_lineages
                or set(by_lineage.values()) != {expected_interfaces}):
            raise ValueError(f"incomplete ASPL=1 cell {key}: {len(rows)}/{expected_runs}")
    return summarize(cells, old_parameters)


def export_gallery(cells: dict, parameters: dict) -> None:
    """Old benchmark-style export densities, with common bins across cells."""
    targets = sorted(float(value) for value in parameters["ASPL_targets"])
    # The 0.25-step curve plots every target. For density panels, retain six
    # readable integer-ASPL reference cells instead of shrinking 21 columns.
    if len(targets) > 12:
        integer_targets = [target for target in targets if np.isclose(target, round(target))]
        if integer_targets:
            targets = integer_targets
    ks = sorted(int(value) for value in parameters["k_values"])
    maximum = max(row[field] for rows in cells.values() for row in rows
                  for field in ("full_export", "mm_export"))
    upper = np.ceil(maximum / 100) * 100
    bins = np.linspace(0, upper, 36)
    fig, axes = plt.subplots(len(ks), len(targets), figsize=(24, 7),
                             sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=.045, right=.995, bottom=.13, top=.82,
                        wspace=.16, hspace=.42)
    for i, k in enumerate(ks):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            rows = cells[target, k]
            for model, field, label, linestyle in (
                (FULL, "full_export", "Full Micro", "-"),
                (MM, "mm_export", "MicroMacro", "--"),
            ):
                ax.hist([row[field] for row in rows], bins=bins, density=True,
                        histtype="step", linewidth=1.6, color=COLORS[model],
                        linestyle=linestyle, label=label)
            degree = parameters["mean_degree_by_target_ASPL"][str(target)]
            ax.text(.5, 1.025,
                    f"ASPL {target:g} | d={degree:g}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
            ax.set_xlim(0, upper)
            ax.locator_params(axis="x", nbins=4)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(color="#d8dde2", alpha=.6, linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
            if i == len(ks) - 1:
                ax.set_xlabel("export time")
            if j == 0:
                ax.set_ylabel(f"k={k}\nprobability density")
    axes[0, 0].legend(frameon=False, fontsize=9)
    save(fig, "aspl_export_density_gallery")


def main() -> None:
    global NAME, SOURCE, FIGURES, LEGACY_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=NAME, help="data folder name under output/data")
    parser.add_argument("--output-name", default=None, help="analysis folder name; defaults to --name")
    parser.add_argument("--summary-gallery-only", action="store_true",
                        help="rebuild only the compact 2x3 figure from the existing summary CSV")
    parser.add_argument("--include-aspl1-from", default=None,
                        help="append completed ASPL=1 cells from another output/data experiment")
    parser.add_argument("--include-aspl-range-from", default=None,
                        help="append a disjoint ASPL range from another compatible output/data experiment")
    args = parser.parse_args()
    NAME = args.name
    SOURCE = ROOT / "output" / "data" / NAME
    FIGURES = ROOT / "output" / "analysis" / (args.output_name or NAME) / "figures"
    LEGACY_NAME = "__no_legacy__"
    if args.summary_gallery_only:
        parameters = json.loads((SOURCE / "experiment_parameters.json").read_text(encoding="utf-8"))
        source_summary = ROOT / "output" / "analysis" / NAME / "figures" / "aspl_export_summary.csv"
        summary = read_summary(parameters, source_summary)
        if args.include_aspl1_from:
            aspl1_summary = [row for row in load_aspl1_summary(
                args.include_aspl1_from, parameters) if float(row["target_aspl"]) == 1.0]
            summary = aspl1_summary + summary
            parameters = {**parameters,
                          "ASPL_targets": sorted({float(row["target_aspl"]) for row in summary})}
            FIGURES.mkdir(parents=True, exist_ok=True)
            with (FIGURES / "aspl_export_summary_with_aspl1.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            note = (
                f"ASPL=1 reused from output/data/{args.include_aspl1_from}: "
                f"{aspl1_summary[0]['paired_runs']} paired runs per k; initial node was fixed at 0.\n"
                f"ASPL>=2 from output/data/{NAME}: "
                f"{parameters['replications_per_scenario']} paired runs per k; initial node selected randomly.\n"
                "ASPL=1 is the complete graph (mean degree 999). Sample size and initial-node protocol differ.\n"
            )
            (FIGURES / "aspl1_comparison_note.txt").write_text(note, encoding="utf-8")
        summary_gallery(summary, parameters)
        print(f"Summary gallery: {FIGURES / 'aspl_export_summary_gallery.png'}")
        return
    cells, parameters = load_cells(accepted_experiments={NAME, LEGACY_NAME})
    if args.include_aspl_range_from:
        extra_name = args.include_aspl_range_from
        extra_dir = ROOT / "output" / "data" / extra_name
        extra_cells, extra_parameters = load_cells(extra_dir, {extra_name})
        cells, parameters = merge_aspl_ranges(cells, extra_cells, parameters, extra_parameters)
    summary = summarize(cells, parameters)
    accuracy_figure(summary, parameters)
    mechanism_figure(summary, parameters)
    summary_gallery(summary, parameters)
    export_gallery(cells, parameters)
    with (FIGURES / "aspl_export_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Completed: {len(summary)} ASPL × k cells; {sum(row['paired_runs'] for row in summary)} pairs")
    if args.include_aspl_range_from:
        print(f"Combined ASPL targets: {parameters['ASPL_targets']}")
    print(f"Figures and metrics: {FIGURES}")


if __name__ == "__main__":
    main()
