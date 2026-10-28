"""Required figures and source-data exports for Experiment 05."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import CORRECTED_VARIANTS, REFERENCE_VARIANT


COLORS = {"V0": "#222222", "V1": "#d55e00", "V2": "#0072b2",
          "V3": "#009e73", "V4": "#cc79a7"}
LABELS = {"V0": "Full Micro", "V1": "Baseline MicroMacro", "V2": "Lag",
          "V3": "Latent proxy", "V4": "Lag + latent"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ecdf_source(
    arrivals: pd.DataFrame, scenario: str, bootstrap_reps: int, seed: int,
) -> pd.DataFrame:
    terminal = arrivals[(arrivals.scenario_id == scenario) &
                        (arrivals.layer_index == arrivals.layer_index.max())]
    if terminal.empty:
        return pd.DataFrame(columns=["scenario_id", "variant_id", "time", "ecdf", "ci95_low", "ci95_high", "n_completed"])
    values = terminal["arrival_time"].to_numpy(float)
    grid = np.linspace(0.0, values.max(), 201)
    rng = np.random.default_rng(seed)
    replication_sets = [set(group.replication_id) for _, group in terminal.groupby("variant_id")]
    if any(ids != replication_sets[0] for ids in replication_sets[1:]):
        raise ValueError(f"unpaired replication IDs for ECDF bootstrap in {scenario}")
    replication_ids = np.array(sorted(replication_sets[0]), dtype=int)
    bootstrap_positions = rng.integers(
        0, replication_ids.size, (bootstrap_reps, replication_ids.size))
    rows: list[dict[str, object]] = []
    for variant, group in terminal.groupby("variant_id", sort=True):
        ordered = group.set_index("replication_id").loc[replication_ids]
        times = ordered.arrival_time.to_numpy(float)
        observed = ordered.observed.to_numpy(bool)
        event_matrix = observed[:, None] & (times[:, None] <= grid[None, :])
        estimate = event_matrix.mean(axis=0)
        bootstrap_curves = event_matrix[bootstrap_positions].mean(axis=1)
        low, high = np.quantile(bootstrap_curves, (0.025, 0.975), axis=0)
        rows.extend({"scenario_id": scenario, "variant_id": variant, "time": float(t),
                     "ecdf": float(y), "ci95_low": float(lo), "ci95_high": float(hi),
                     "n_completed": int(observed.sum())}
                    for t, y, lo, hi in zip(grid, estimate, low, high))
    return pd.DataFrame(rows)


def plot_ecdf(source: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    for variant, group in source.groupby("variant_id", sort=True):
        color = COLORS.get(variant)
        ax.plot(group.time, group.ecdf, color=color, label=LABELS.get(variant, variant), lw=1.8)
        ax.fill_between(group.time, group.ci95_low, group.ci95_high, color=color, alpha=.13, linewidth=0)
    ax.set(xlabel=r"Terminal arrival time $T_{last}$", ylabel="Cumulative incidence")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_layer_w1(source: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    valid = source[source.scenario_id == "baseline"]
    for variant, group in valid.groupby("variant_id", sort=True):
        ax.plot(group.layer_index, group.relative_W1_T_i_restricted_all_runs, marker="o", ms=3,
                color=COLORS.get(variant), label=LABELS.get(variant, variant))
    ax.set(xlabel="Community index $i$", ylabel=r"Relative restricted $W_1(T_i)$ (all runs)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def plot_cumulative_errors(source: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    baseline = source[source.scenario_id == "baseline"]
    for variant, group in baseline.groupby("variant_id", sort=True):
        color = COLORS.get(variant)
        ax.plot(group.layer_index, group.cumulative_error_median_all_runs, color=color,
                label=LABELS.get(variant, variant))
        ax.fill_between(group.layer_index, group.cumulative_error_q25_all_runs,
                        group.cumulative_error_q75_all_runs, color=color, alpha=.14, linewidth=0)
    ax.axhline(0, color="black", lw=.7)
    ax.set(xlabel="Community index $i$", ylabel=r"Paired error $T_i^{V_k}-T_i^{V_0}$")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def sweep_source(terminal: pd.DataFrame) -> pd.DataFrame:
    maps = {
        "k": {"k1": 1.0, "baseline": 5.0, "k10": 10.0},
        "d": {"d4": 4.0, "baseline": 6.0, "d10": 10.0},
    }
    rows: list[dict[str, object]] = []
    for parameter, mapping in maps.items():
        for scenario, x in mapping.items():
            selected = terminal[(terminal.scenario_id == scenario) & terminal.variant_id.isin(CORRECTED_VARIANTS)]
            for record in selected.to_dict("records"):
                rows.append({"parameter": parameter, "parameter_value": x,
                             "scenario_id": scenario, "variant_id": record["variant_id"],
                             "absolute_RMST_error_change_vs_V1": record.get("absolute_RMST_error_change_vs_V1"),
                             "metric_status": record.get("metric_status", "ok")})
    return pd.DataFrame(rows)


def plot_sweeps(source: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
    for ax, parameter in zip(axes, ("k", "d")):
        selected = source[source.parameter == parameter]
        for variant, group in selected.groupby("variant_id", sort=True):
            group = group.sort_values("parameter_value")
            ax.plot(group.parameter_value, group.absolute_RMST_error_change_vs_V1, marker="o",
                    color=COLORS.get(variant), label=LABELS.get(variant, variant))
        ax.set_xlabel(f"Interface width ${parameter}$" if parameter == "k" else r"Mean degree $d$")
        ax.grid(alpha=.2)
        ax.axhline(0, color="black", lw=.7)
    axes[0].set_ylabel(r"Change in absolute $\Delta$RMST vs V1")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, path)


def create_required_figures(
    arrivals: pd.DataFrame, terminal: pd.DataFrame, layers: pd.DataFrame,
    paired: pd.DataFrame, output_root: Path, report_dir: Path,
    bootstrap_reps: int = 1000, bootstrap_seed: int = 0,
) -> list[Path]:
    """Create Figures 1--4 and write every plotted value as CSV."""
    data_dir = output_root / "figures" / "data"
    figure_dir = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    ecdf = ecdf_source(arrivals, "baseline", bootstrap_reps, bootstrap_seed)
    layer_source = layers[layers.scenario_id == "baseline"].copy()
    paired_source = paired[paired.scenario_id == "baseline"].copy()
    sweeps = sweep_source(terminal)
    sources = (("figure_1_ecdf.csv", ecdf), ("figure_2_layer_w1.csv", layer_source),
               ("figure_3_paired_errors.csv", paired_source), ("figure_4_sweeps.csv", sweeps))
    for name, frame in sources:
        frame.to_csv(data_dir / name, index=False)

    paths = [figure_dir / f"figure_{index}.pdf" for index in range(1, 5)]
    plot_ecdf(ecdf, paths[0])
    plot_layer_w1(layer_source, paths[1])
    plot_cumulative_errors(paired_source, paths[2])
    plot_sweeps(sweeps, paths[3])
    return paths
