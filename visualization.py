import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from devtools.config import load_config, resolve_path
from devtools.visualize import (
    interp_stack_on_grid,
    plot_per_community_curves,
    plot_total_dynamics,
    representative_and_bands,
)
from sim_db import load_sir_dataframe, search_by_filters


# ========= settings =========
cfg, base_dir = load_config()
db_path = resolve_path(cfg.get("storage", {}).get("db_path", "simulations.db"), base_dir=base_dir)
GRID_POINTS = 1000  # interpolation grid resolution inside each dataset
PLOT_TOTAL_DYNAMICS = True
PLOT_PER_COMMUNITY = True
PLOT_MICROMACRO_COMPARISON = True
PLOT_METRIC_DENSITIES = True
SHOW_MEDIAN_POINTS = True
MEDIAN_POINT_STRIDE = 1
APPLY_METRIC_TIME_CUTOFF = True
METRIC_TIME_CUTOFF = float(cfg["simulation"]["T_end"])

# Quick mode: use subset of runs + smaller interpolation grid for faster runs.
QUICK_MODE = False
QUICK_MAX_RUNS_PER_DATASET = 100
QUICK_GRID_POINTS = 300
BRIDGE_DENSITY_FIRST_N_COMMUNITIES = 1  # None -> show all communities

QUERY_FIELDS = [
    "communities",
    "community_size",
    "inter_links",
    "seed",
    "macro_graph_type",
    "micro_graph_type",
    "edge_prob",
    "leaf_count",
    "leaf_degree",
    "star_leaf_attachment",
    "beta",
    "gamma",
    "model",
    "T_end",
    "initial_node",
    "dt_out",
    "tau_micro",
    "macro_T",
]


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _sanitize_time(value, max_time: float | None):
    t = _to_float_or_none(value)
    if t is None:
        return None
    if t < 0.0:
        return None
    if max_time is not None and t > max_time:
        return None
    return t


def _metric_time_limit():
    if not APPLY_METRIC_TIME_CUTOFF:
        return None
    if not np.isfinite(METRIC_TIME_CUTOFF):
        return None
    return float(METRIC_TIME_CUTOFF)


def _read_metrics_rows_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metrics", [])


def _read_metrics_rows_csv(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        expected = ["community", "t0", "t_bridge", "t_export"]
        if header != expected:
            raise ValueError(f"Unexpected metrics CSV header in {path}: {header}")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "community": parts[0],
                    "t0": parts[1] or None,
                    "t_bridge": parts[2] or None,
                    "t_export": parts[3] or None,
                }
            )
    return rows


def _load_metrics_for_simulation(run_uid: str, max_time: float | None):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT community, t0, t_bridge, t_export
            FROM community_metrics
            WHERE run_uid = ?
            ORDER BY community
            """,
            (run_uid,),
        )
        rows = cur.fetchall()
    out = {}
    for community, t0, t_bridge, t_export in rows:
        comm = int(community)
        out[comm] = {
            "t0": _sanitize_time(t0, max_time=max_time),
            "t_bridge": _sanitize_time(t_bridge, max_time=max_time),
            "t_export": _sanitize_time(t_export, max_time=max_time),
        }
    return out


def _collect_metric_density_samples(run_uids: list[str], max_time: float | None):
    by_metric: dict[str, dict[int, list[float]]] = {
        "t_bridge": {},
        "bridge_to_export": {},
        "start_to_export": {},
    }

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        for run_uid in run_uids:
            cur.execute(
                """
                SELECT community, t_bridge, t_export
                FROM community_metrics
                WHERE run_uid = ?
                """,
                (run_uid,),
            )
            for community, t_bridge_raw, t_export_raw in cur.fetchall():
                comm = int(community)
                t_bridge = _sanitize_time(t_bridge_raw, max_time=max_time)
                t_export = _sanitize_time(t_export_raw, max_time=max_time)

                if t_bridge is not None:
                    by_metric["t_bridge"].setdefault(comm, []).append(t_bridge)

                if t_bridge is not None and t_export is not None and t_export >= t_bridge:
                    by_metric["bridge_to_export"].setdefault(comm, []).append(t_export - t_bridge)

                if t_export is not None:
                    by_metric["start_to_export"].setdefault(comm, []).append(t_export)

    return by_metric


def _sorted_unique_curve_xy(curve):
    time_vals = np.asarray(curve["time"], dtype=float)
    infected_vals = np.asarray(curve["I"], dtype=float)
    order = np.argsort(time_vals)
    time_vals = time_vals[order]
    infected_vals = infected_vals[order]

    if time_vals.size <= 1:
        return time_vals, infected_vals

    unique_t, unique_idx = np.unique(time_vals, return_index=True)
    return unique_t, infected_vals[unique_idx]


def _augment_curve_with_metric_times(time_vals, infected_vals, metric_times):
    finite_metric_times = [t for t in metric_times if t is not None and np.isfinite(t)]
    if not finite_metric_times:
        return time_vals, infected_vals

    merged_times = np.unique(np.concatenate([time_vals, np.asarray(finite_metric_times, dtype=float)]))
    merged_I = np.interp(merged_times, time_vals, infected_vals)
    return merged_times, merged_I


def _require_seaborn_or_none():
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        return None
    return sns


def _plot_density_hist(ax, series: dict[str, list[float]], title: str, xlabel: str):
    finite_series = {label: np.asarray(vals, dtype=float) for label, vals in series.items() if vals}
    if not finite_series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        return

    merged = np.concatenate(list(finite_series.values()))
    if merged.size == 0:
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        return

    bins = min(40, max(8, int(np.sqrt(merged.size))))
    sns = _require_seaborn_or_none()

    if sns is not None:
        for label, vals in sorted(finite_series.items()):
            sns.histplot(
                x=vals,
                bins=bins,
                stat="density",
                element="step",
                fill=False,
                common_norm=False,
                linewidth=1.8,
                alpha=0.95,
                label=label,
                ax=ax,
            )
    else:
        for label, vals in sorted(finite_series.items()):
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.8, alpha=0.95, label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def _build_output_dir():
    net_cfg = cfg["network"]
    community_size = int(net_cfg["community_size"])
    edge_prob_str = str(net_cfg["edge_prob"]).replace(".", "p")
    folder_name = (
        f"k_{net_cfg['communities']}_"
        f"n_{community_size}_"
        f"inter_{net_cfg['inter_links']}_"
        f"macro_{net_cfg['macro_graph_type']}_"
        f"micro_{net_cfg['micro_graph_type']}_"
        f"p{edge_prob_str}"
    )
    out_dir = resolve_path(Path("plots") / folder_name, base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _dataset_filters_from_config(simulator: str):
    net = cfg.get("network", {})
    virus = cfg.get("virus", {})
    sim = cfg.get("simulation", {})
    micro = cfg.get("micro", {})
    mm = cfg.get("micromacro", {})

    filters = {"simulator": simulator}
    if "communities" in net:
        filters["communities"] = int(net["communities"])
    if "community_size" in net:
        filters["community_size"] = int(net["community_size"])
    if "inter_links" in net:
        filters["inter_links"] = int(net["inter_links"])
    if "seed" in net:
        filters["seed"] = int(net["seed"])
    if "macro_graph_type" in net:
        filters["macro_graph_type"] = str(net["macro_graph_type"])
    if "micro_graph_type" in net:
        filters["micro_graph_type"] = str(net["micro_graph_type"])
    if "edge_prob" in net:
        filters["edge_prob"] = float(net["edge_prob"])
    if "leaf_count" in net:
        filters["leaf_count"] = int(net["leaf_count"])
    if "leaf_degree" in net:
        filters["leaf_degree"] = int(net["leaf_degree"])
    if "star_leaf_attachment" in net:
        filters["star_leaf_attachment"] = str(net["star_leaf_attachment"])
    if "beta" in virus:
        filters["beta"] = float(virus["beta"])
    if "gamma" in virus:
        filters["gamma"] = float(virus["gamma"])
    if "model" in virus:
        filters["model"] = int(virus["model"])
    if "T_end" in sim:
        filters["T_end"] = float(sim["T_end"])
    if "initial_node" in sim and sim["initial_node"] is not None:
        filters["initial_node"] = int(sim["initial_node"])
    if simulator == "Micro":
        if "dt_out" in micro:
            filters["dt_out"] = float(micro["dt_out"])
    else:
        if "tau_micro" in mm:
            filters["tau_micro"] = float(mm["tau_micro"])
        if "macro_T" in mm:
            filters["macro_T"] = float(mm["macro_T"])
    return filters


def _find_ambiguous_unset_fields(simulator: str, runs: list[dict], used_filters: dict):
    if len(runs) <= 1:
        return {}
    ambiguous = {}
    for field in QUERY_FIELDS:
        if field in used_filters:
            continue
        values = {row.get(field) for row in runs}
        if len(values) > 1:
            normalized = sorted(values, key=lambda x: (x is None, str(x)))
            ambiguous[field] = normalized[:10]
    return ambiguous


def _build_dataset_summary_from_runs(runs: list[dict], *, grid_points: int):
    if not runs:
        raise FileNotFoundError("No runs found in DB for current config filters")

    raw_curves = {}
    rep_comm_curves = {}
    tmins = []
    tmaxs = []
    run_uids = []

    for row in runs:
        run_uid = str(row["run_uid"])
        run_uids.append(run_uid)
        df = load_sir_dataframe(run_uid, db_path=db_path)
        agg = (
            df.groupby("time", as_index=False)[["S", "I", "R"]]
            .sum()
            .sort_values("time")
        )
        cur = agg[["time", "I"]].drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        raw_curves[run_uid] = cur
        tmins.append(float(cur["time"].min()))
        tmaxs.append(float(cur["time"].max()))

    tmin = float(np.max(tmins))
    tmax = float(np.min(tmaxs))
    t_grid = np.linspace(tmin, tmax, grid_points)
    I_interp, I_stack = interp_stack_on_grid(raw_curves, t_grid)
    idx_rep, I_mean, I_rep, median, q1, q3, mse = representative_and_bands(I_stack)
    rep_key = list(I_interp.keys())[idx_rep]

    rep_df = load_sir_dataframe(rep_key, db_path=db_path)
    for comm_id, grp in rep_df.groupby("community"):
        rep_comm_curves[int(comm_id)] = grp.sort_values("time")[["time", "I"]].reset_index(drop=True)

    return {
        "raw": raw_curves,
        "tmin": tmin,
        "tmax": tmax,
        "t_grid": t_grid,
        "I_interp": I_interp,
        "I_stack": I_stack,
        "idx_rep": idx_rep,
        "I_mean": I_mean,
        "I_rep": I_rep,
        "median": median,
        "q1": q1,
        "q3": q3,
        "mse": mse,
        "rep_sim_id": rep_key,
        "rep_comm_curves": rep_comm_curves,
        "run_uids": run_uids,
    }


def main():
    grid_points = GRID_POINTS
    max_runs = None

    if QUICK_MODE:
        grid_points = QUICK_GRID_POINTS
        max_runs = QUICK_MAX_RUNS_PER_DATASET
        print(
            f"[Visualization] QUICK_MODE enabled: up to {QUICK_MAX_RUNS_PER_DATASET} runs per dataset, "
            f"grid_points={grid_points}"
        )

    datasets = {}
    for simulator in ("Micro", "MicroMacro"):
        filters = _dataset_filters_from_config(simulator)
        runs = search_by_filters(filters, sort="created_at DESC", limit=max_runs or 100000, db_path=db_path)
        if not runs:
            raise FileNotFoundError(f"No runs found in DB for simulator={simulator} and current config filters")

        ambiguous = _find_ambiguous_unset_fields(simulator, runs, filters)
        if ambiguous:
            print(f"[Visualization] Ambiguous query for {simulator}.")
            print("[Visualization] Unspecified parameters vary across matched runs:")
            for key, vals in ambiguous.items():
                print(f"  - {key}: {vals}")
            print("[Visualization] Add these parameters to config and rerun.")
            return
        datasets[simulator] = _build_dataset_summary_from_runs(runs, grid_points=grid_points)

    out_dir = _build_output_dir()
    max_time = _metric_time_limit()

    print(f"[Visualization] Matplotlib backend: {plt.get_backend()}")
    print(f"[Visualization] Output directory: {out_dir}")
    print(f"[Visualization] DB source: {db_path}")
    if max_time is not None:
        print(f"[Visualization] Metric time cutoff: {max_time}")

        if PLOT_TOTAL_DYNAMICS:
            base_colors = {
                "Micro": "#1f77b4",
                "MicroMacro": "#ff7f0e",
            }
            out_path = out_dir / "Micro_vs_MicroMacro.png"
            plot_total_dynamics(
                datasets=datasets,
                out_path=out_path,
                title="Micro vs MicroMacro: representative I-curves with interquartile bands",
                base_colors=base_colors,
                show_median_points=SHOW_MEDIAN_POINTS,
                median_point_stride=MEDIAN_POINT_STRIDE,
            )
            print(f"[Visualization] Saved: {out_path}")

        if PLOT_PER_COMMUNITY and "Micro" in datasets:
            d = datasets["Micro"]
            rep_sim = d["rep_sim_id"]
            comm_curves = d["rep_comm_curves"]
            out_path_comm = out_dir / "Micro_community.png"
            plot_per_community_curves(
                comm_curves=comm_curves,
                out_path=out_path_comm,
                title=f"Micro: representative simulation {rep_sim} per-community I(t)",
            )
            print(f"[Visualization] Saved: {out_path_comm}")

        if PLOT_PER_COMMUNITY and "MicroMacro" in datasets:
            d = datasets["MicroMacro"]
            rep_sim = d["rep_sim_id"]
            comm_curves = d["rep_comm_curves"]
            out_path_comm = out_dir / "MicroMacro_community.png"
            plot_per_community_curves(
                comm_curves=comm_curves,
                out_path=out_path_comm,
                title=f"MicroMacro: representative simulation {rep_sim} per-community I(t)",
            )
            print(f"[Visualization] Saved: {out_path_comm}")

        if PLOT_MICROMACRO_COMPARISON and "Micro" in datasets and "MicroMacro" in datasets:
            panel_order = ["Micro", "MicroMacro"]
            all_comms = sorted(
                set(datasets["Micro"]["rep_comm_curves"].keys())
                | set(datasets["MicroMacro"]["rep_comm_curves"].keys())
            )
            tab10 = plt.get_cmap("tab10")
            comm_to_color = {comm_id: tab10(idx % 10) for idx, comm_id in enumerate(all_comms)}
            marker_map = {
                "t0": "o",
                "t_bridge": "^",
                "t_export": "s",
            }
            marker_legend = [
                Line2D([], [], color="black", marker="o", linestyle="None", markersize=7, label="t0"),
                Line2D([], [], color="black", marker="^", linestyle="None", markersize=7, label="t_bridge"),
                Line2D([], [], color="black", marker="s", linestyle="None", markersize=7, label="t_export"),
            ]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
            for ax_idx, name in enumerate(panel_order):
                ax = axes[ax_idx]
                d = datasets[name]
                rep_sim = d["rep_sim_id"]
                comm_curves = d["rep_comm_curves"]
                comm_metrics = _load_metrics_for_simulation(run_uid=rep_sim, max_time=max_time)

                for comm_id in sorted(comm_curves.keys()):
                    color = comm_to_color[comm_id]
                    curve = comm_curves[comm_id]
                    time_vals, infected_vals = _sorted_unique_curve_xy(curve)
                    row = comm_metrics.get(comm_id, {})

                    augmented_times, augmented_I = _augment_curve_with_metric_times(
                        time_vals=time_vals,
                        infected_vals=infected_vals,
                        metric_times=[row.get("t0"), row.get("t_bridge"), row.get("t_export")],
                    )
                    ax.plot(
                        augmented_times,
                        augmented_I,
                        color=color,
                        linewidth=1.8,
                        label=f"Community {comm_id}",
                    )

                    for metric_name, marker in marker_map.items():
                        t_metric = row.get(metric_name)
                        if t_metric is None:
                            continue
                        y_metric = float(np.interp(t_metric, augmented_times, augmented_I))
                        ax.scatter(
                            [t_metric],
                            [y_metric],
                            color=color,
                            marker=marker,
                            s=48,
                            linewidths=0.5,
                            edgecolors="black",
                            zorder=5,
                        )

                ax.set_title(f"{name}: representative simulation {rep_sim}")
                ax.set_xlabel("Time")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, ncol=2, loc="upper left")

            axes[0].set_ylabel("Infected (I)")
            fig.suptitle("Per-community activation timings on representative trajectories")
            fig.legend(handles=marker_legend, loc="upper center", ncol=3, frameon=False)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

            out_path_comm = out_dir / "Micro_vs_MicroMacro_metric_panels.png"
            plt.savefig(out_path_comm, dpi=300, bbox_inches="tight")
            print(f"[Visualization] Saved: {out_path_comm}")

        if PLOT_METRIC_DENSITIES:
            density_data = {
                name: _collect_metric_density_samples(run_uids=datasets[name]["run_uids"], max_time=max_time)
                for name in datasets.keys()
            }

            metric_specs = [
                ("t_bridge", "Bridge infection time per community", "Time"),
                ("bridge_to_export", "Time from bridge infection to export", "Delta time"),
                ("start_to_export", "Time from start to export", "Time"),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, (metric_key, title, xlabel) in zip(axes, metric_specs):
                series: dict[str, list[float]] = {}
                for name, metric_map in density_data.items():
                    by_comm = metric_map.get(metric_key, {})
                    comm_ids = sorted(by_comm.keys())
                    if BRIDGE_DENSITY_FIRST_N_COMMUNITIES is not None:
                        comm_ids = comm_ids[:BRIDGE_DENSITY_FIRST_N_COMMUNITIES]
                    for comm_id in comm_ids:
                        series[f"{name} C{comm_id}"] = by_comm[comm_id]

                _plot_density_hist(ax=ax, series=series, title=title, xlabel=xlabel)

            fig.suptitle("Density-style distributions of bridge/export timings")
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            out_path_density = out_dir / "Metric_density_panels.png"
            plt.savefig(out_path_density, dpi=300, bbox_inches="tight")
            print(f"[Visualization] Saved: {out_path_density}")

        backend_name = str(plt.get_backend()).strip().lower()
        if backend_name != "agg":
            plt.show()
        else:
            print(
                "[Visualization] Interactive window disabled because backend is non-interactive "
                f"({plt.get_backend()}). Open saved PNG files from the output directory."
            )

    for name, d in datasets.items():
        print(f"[{name}] Representative simulation: {d['rep_sim_id']}")
        print(f"[{name}] Time grid range used: [{d['tmin']:.3f}, {d['tmax']:.3f}] with {grid_points} points")
        print(f"[{name}] Mean MSE across sims: {np.mean(d['mse']):.6f}, best MSE: {np.min(d['mse']):.6f}")


if __name__ == "__main__":
    main()
