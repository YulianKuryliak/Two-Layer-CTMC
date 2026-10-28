from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MASS_COLUMNS = [
    "A_exact", "A_aggregate", "signed_hazard_mass", "absolute_hazard_mass",
    "over_hazard_mass", "under_hazard_mass", "premature_hazard_mass", "missed_hazard_mass",
]


def add_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    d = out["duration"]
    exact, aggregate = out["exact_hazard"], out["aggregate_hazard"]
    delta = aggregate - exact
    out["A_exact"] = exact * d
    out["A_aggregate"] = aggregate * d
    out["signed_hazard_mass"] = delta * d
    out["absolute_hazard_mass"] = delta.abs() * d
    out["over_hazard_mass"] = delta.clip(lower=0.0) * d
    out["under_hazard_mass"] = (-delta).clip(lower=0.0) * d
    out["premature_hazard_mass"] = aggregate.where(exact == 0.0, 0.0) * d
    out["missed_hazard_mass"] = exact.where(aggregate == 0.0, 0.0) * d
    return out


def integrate(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    enriched = add_contributions(frame)
    grouped = enriched.groupby(list(group_columns), observed=True, dropna=False)
    out = grouped[MASS_COLUMNS].sum().reset_index()
    exposure = grouped["duration"].sum().rename("exposure_duration").reset_index()
    rows = grouped.size().rename("directed_pair_interval_rows").reset_index()
    out = out.merge(exposure, on=list(group_columns)).merge(rows, on=list(group_columns))
    out["premature_fraction"] = np.where(
        out["A_aggregate"] > 0.0, out["premature_hazard_mass"] / out["A_aggregate"], np.nan
    )
    return out


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(reps, values.size), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def replication_statistics(table: pd.DataFrame, group_columns: Iterable[str], metrics: Iterable[str],
                           bootstrap_reps: int, seed: int) -> pd.DataFrame:
    group_columns = list(group_columns)
    grouped = [((), table)] if not group_columns else table.groupby(group_columns, observed=True, dropna=False)
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        prefix = dict(zip(group_columns, key))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            low, high = _bootstrap_ci(values, rng, bootstrap_reps)
            rows.append({
                **prefix, "metric": metric, "n_replications": int(group["replication_id"].nunique()),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "median": float(np.median(finite)) if finite.size else np.nan,
                "ci95_low": low, "ci95_high": high,
            })
    return pd.DataFrame(rows)


def descriptive_stratified_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    """Duration-weighted event-level descriptions; rows are explicitly not independent."""
    work = add_contributions(intervals)
    specifications = {
        "source_infected_fraction": pd.cut(work["source_infected_fraction"], np.linspace(0, 1, 11), include_lowest=True),
        "destination_susceptible_fraction": pd.cut(work["destination_susceptible_fraction"], np.linspace(0, 1, 11), include_lowest=True),
        "infected_source_boundary_nodes": work["infected_source_boundary_nodes"].astype(str),
        "susceptible_destination_boundary_nodes": work["susceptible_destination_boundary_nodes"].astype(str),
        "active_intercommunity_edge_count": work["active_intercommunity_edge_count"].astype(str),
        "active_intercommunity_edge_weight": work["active_intercommunity_edge_weight"].astype(str),
        "layer_index": work["layer_index"].astype(str),
        "time_since_source_first_infection": pd.cut(
            work["time_since_source_first_infection"], [-np.inf, 100, 250, 500, 1000, 2000, 4000, np.inf]
        ),
        "time_since_destination_first_infection": pd.cut(
            work["time_since_destination_first_infection"], [-np.inf, 100, 250, 500, 1000, 2000, 4000, np.inf]
        ),
        "epidemic_phase": work["epidemic_phase"],
    }
    rows = []
    for name, strata in specifications.items():
        local = work.assign(_stratum=strata.astype(str).replace("nan", "not_started"))
        for stratum, group in local.groupby("_stratum", observed=True):
            duration = group["duration"].sum()
            rows.append({
                "stratifier": name, "stratum": stratum,
                "directed_pair_interval_rows": len(group), "exposure_duration": duration,
                "exact_hazard_mass": group["A_exact"].sum(),
                "aggregate_hazard_mass": group["A_aggregate"].sum(),
                "signed_hazard_mass": group["signed_hazard_mass"].sum(),
                "absolute_hazard_mass": group["absolute_hazard_mass"].sum(),
                "duration_weighted_signed_error": group["signed_hazard_mass"].sum() / duration if duration else np.nan,
            })
    return pd.DataFrame(rows)


def build_summaries(intervals: pd.DataFrame, bootstrap_reps: int, seed: int) -> dict[str, pd.DataFrame]:
    replication_pair = integrate(intervals, ["replication_id", "src_community", "dst_community",
                                              "interface_index", "layer_index"])
    replication = integrate(intervals, ["replication_id"])
    replication_layer = integrate(intervals, ["replication_id", "interface_index"])
    replication_phase = integrate(intervals, ["replication_id", "epidemic_phase"])
    metrics = ["A_exact", "A_aggregate", "signed_hazard_mass", "absolute_hazard_mass",
               "over_hazard_mass", "under_hazard_mass", "premature_hazard_mass",
               "missed_hazard_mass", "premature_fraction"]
    return {
        "replication_summary": replication,
        "replication_pair": replication_pair,
        "community_pair_summary": replication_statistics(
            replication_pair, ["src_community", "dst_community", "interface_index", "layer_index"],
            metrics, bootstrap_reps, seed + 3,
        ),
        "layer_summary": replication_statistics(replication_layer, ["interface_index"], metrics,
                                                bootstrap_reps, seed + 1),
        "phase_summary": replication_statistics(replication_phase, ["epidemic_phase"], metrics,
                                                bootstrap_reps, seed + 2),
        "overall_statistics": replication_statistics(replication, [], metrics, bootstrap_reps, seed),
        "replication_layer": replication_layer,
        "replication_phase": replication_phase,
        "descriptive_stratified_summary": descriptive_stratified_summary(intervals),
    }


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_figures(intervals: pd.DataFrame, summaries: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(270020)
    nonzero = intervals[(intervals["exact_hazard"] + intervals["aggregate_hazard"]) > 0]
    sample = nonzero.iloc[rng.choice(len(nonzero), min(25000, len(nonzero)), replace=False)] if len(nonzero) else nonzero
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.scatter(sample["exact_hazard"], sample["aggregate_hazard"], s=5, alpha=.18, rasterized=True)
    limit = max(sample[["exact_hazard", "aggregate_hazard"]].max().max(), 1e-9) if len(sample) else 1.0
    ax.plot([0, limit], [0, limit], color="black", lw=1)
    ax.set(xlabel="Exact active-edge hazard", ylabel="Aggregate prevalence hazard", xlim=(0, limit), ylim=(0, limit))
    _save(fig, output_dir / "01_exact_vs_aggregate.png")

    rep = summaries["replication_summary"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2)); ax.hist(rep["signed_hazard_mass"], bins=min(12, len(rep)), color="#4477AA")
    ax.axvline(0, color="black", lw=1); ax.set(xlabel="Integrated signed hazard error", ylabel="Replications")
    _save(fig, output_dir / "02_signed_error_distribution.png")

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.boxplot([rep["over_hazard_mass"], rep["under_hazard_mass"]], tick_labels=["Over", "Under"])
    ax.set(ylabel="Integrated hazard mass")
    _save(fig, output_dir / "03_over_under_mass.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.0)); ax.bar(rep["replication_id"].astype(str), rep["premature_fraction"], color="#CC6677")
    ax.set(xlabel="Replication", ylabel="Premature fraction"); ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    _save(fig, output_dir / "04_premature_fraction.png")

    plot = intervals.copy(); plot["infected_bin"] = pd.cut(plot["source_infected_fraction"], np.linspace(0, 1, 21), include_lowest=True)
    binned = plot.groupby("infected_bin", observed=True).apply(
        lambda x: np.average(x["signed_error"], weights=x["duration"]), include_groups=False).reset_index(name="duration_weighted_error")
    centers = [item.mid for item in binned["infected_bin"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); ax.plot(centers, binned["duration_weighted_error"], marker="o", ms=3)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Source infected fraction (bin midpoint)", ylabel="Duration-weighted signed error")
    _save(fig, output_dir / "05_error_vs_source_fraction.png")

    edge = intervals.groupby("active_intercommunity_edge_count", observed=True).apply(
        lambda x: np.average(x["signed_error"], weights=x["duration"]), include_groups=False).reset_index(name="duration_weighted_error")
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); ax.plot(edge["active_intercommunity_edge_count"], edge["duration_weighted_error"], marker="o")
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Active I–S interface edges", ylabel="Duration-weighted signed error")
    _save(fig, output_dir / "06_error_vs_active_edges.png")

    layer = summaries["layer_summary"]
    layer = layer[layer["metric"] == "signed_hazard_mass"].sort_values("interface_index")
    errors = np.vstack([layer["mean"] - layer["ci95_low"], layer["ci95_high"] - layer["mean"]])
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); ax.errorbar(layer["interface_index"], layer["mean"], yerr=errors, marker="o", capsize=3)
    ax.axhline(0, color="black", lw=1); ax.set(xlabel="Chain interface index", ylabel="Mean integrated signed error (bootstrap 95% CI)")
    _save(fig, output_dir / "07_layer_integrated_error.png")

    rid = int(rep.iloc[(rep["signed_hazard_mass"] - rep["signed_hazard_mass"].median()).abs().argsort()[:1]]["replication_id"].iloc[0])
    representative_pairs = [(0, 1), (4, 5), (8, 9)]
    fig, axes = plt.subplots(len(representative_pairs), 1, figsize=(8.0, 2.5 * len(representative_pairs)), sharex=True, squeeze=False)
    for ax, pair in zip(axes[:, 0], representative_pairs):
        subset = intervals[(intervals["replication_id"] == rid) & (intervals["src_community"] == pair[0]) & (intervals["dst_community"] == pair[1])]
        ax.step(subset["time_start"], subset["exact_hazard"], where="post", label="Exact", lw=1.2)
        ax.step(subset["time_start"], subset["aggregate_hazard"], where="post", label="Aggregate", lw=1.0)
        ax.set(ylabel=f"{pair[0]}→{pair[1]}\nhazard"); ax.legend(fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel(f"Time; representative replication {rid}")
    _save(fig, output_dir / "08_representative_time_series.png")
