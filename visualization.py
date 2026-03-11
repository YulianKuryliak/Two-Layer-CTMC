import json
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from devtools.config import load_config, resolve_path
from devtools.visualize import (
    build_datasets_summary,
    plot_per_community_curves,
    plot_total_dynamics,
)


# ========= settings =========
cfg, base_dir = load_config()
micro_out = resolve_path(cfg["micro"]["out_folder"], base_dir=base_dir)
micromacro_out = resolve_path(cfg["micromacro"]["out_folder"], base_dir=base_dir)

folders = {
    "Micro": micro_out,
    "MicroMacro": micromacro_out,
}

pattern = "*.csv"
GRID_POINTS = 1000  # interpolation grid resolution inside each dataset
PLOT_TOTAL_DYNAMICS = True
PLOT_PER_COMMUNITY = True
PLOT_MICROMACRO_COMPARISON = True
PLOT_METRIC_DENSITIES = True
SHOW_MEDIAN_POINTS = True
MEDIAN_POINT_STRIDE = 1
APPLY_METRIC_TIME_CUTOFF = True
METRIC_TIME_CUTOFF = float(cfg["simulation"]["T_end"])

# Quick mode: use subset of files + smaller interpolation grid for faster runs.
QUICK_MODE = False
QUICK_MAX_FILES_PER_DATASET = 300
QUICK_GRID_POINTS = 300
BRIDGE_DENSITY_FIRST_N_COMMUNITIES = 1  # None -> show all communities


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


def _load_metrics_for_simulation(folder: Path, sim_id: str, max_time: float | None):
    metrics_path = folder / "metrics" / f"{sim_id}_metrics.json"
    metrics_csv_path = folder / "metrics" / f"{sim_id}_metrics.csv"

    if metrics_path.exists():
        rows = _read_metrics_rows_json(metrics_path)
    elif metrics_csv_path.exists():
        rows = _read_metrics_rows_csv(metrics_csv_path)
    else:
        raise FileNotFoundError(
            f"Missing metrics file for run {sim_id} in {folder / 'metrics'} "
            f"(expected {metrics_path.name} or {metrics_csv_path.name})"
        )

    out = {}
    for row in rows:
        comm = int(row["community"])
        out[comm] = {
            "t0": _sanitize_time(row.get("t0"), max_time=max_time),
            "t_bridge": _sanitize_time(row.get("t_bridge"), max_time=max_time),
            "t_export": _sanitize_time(row.get("t_export"), max_time=max_time),
        }
    return out


def _load_all_metrics_rows(folder: Path):
    metrics_dir = folder / "metrics"
    if not metrics_dir.exists():
        return []

    json_files = sorted(metrics_dir.glob("*_metrics.json"))
    if json_files:
        rows = []
        for path in json_files:
            rows.extend(_read_metrics_rows_json(path))
        return rows

    csv_files = sorted(metrics_dir.glob("*_metrics.csv"))
    rows = []
    for path in csv_files:
        rows.extend(_read_metrics_rows_csv(path))
    return rows


def _collect_metric_density_samples(folder: Path, max_time: float | None):
    rows = _load_all_metrics_rows(folder)
    by_metric: dict[str, dict[int, list[float]]] = {
        "t_bridge": {},
        "bridge_to_export": {},
        "start_to_export": {},
    }

    for row in rows:
        comm = int(row["community"])
        t_bridge = _sanitize_time(row.get("t_bridge"), max_time=max_time)
        t_export = _sanitize_time(row.get("t_export"), max_time=max_time)

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


def _prepare_quick_mode_folders(src_folders: dict[str, Path], file_glob: str, max_files: int):
    temp_root = Path(tempfile.mkdtemp(prefix="viz_quick_"))
    quick_folders: dict[str, Path] = {}

    for name, src_folder in src_folders.items():
        dst_folder = temp_root / name
        dst_folder.mkdir(parents=True, exist_ok=True)
        quick_folders[name] = dst_folder

        csv_files = sorted(src_folder.glob(file_glob))[:max_files]
        selected_ids = set()
        for csv_path in csv_files:
            shutil.copy2(csv_path, dst_folder / csv_path.name)
            selected_ids.add(csv_path.stem)

        src_metrics = src_folder / "metrics"
        if src_metrics.exists():
            dst_metrics = dst_folder / "metrics"
            dst_metrics.mkdir(parents=True, exist_ok=True)
            for sim_id in selected_ids:
                json_path = src_metrics / f"{sim_id}_metrics.json"
                csv_path = src_metrics / f"{sim_id}_metrics.csv"
                if json_path.exists():
                    shutil.copy2(json_path, dst_metrics / json_path.name)
                elif csv_path.exists():
                    shutil.copy2(csv_path, dst_metrics / csv_path.name)

    return quick_folders, temp_root


def main():
    folders_for_run = folders
    grid_points = GRID_POINTS
    temp_quick_root: Path | None = None

    if QUICK_MODE:
        folders_for_run, temp_quick_root = _prepare_quick_mode_folders(
            src_folders=folders,
            file_glob=pattern,
            max_files=QUICK_MAX_FILES_PER_DATASET,
        )
        grid_points = QUICK_GRID_POINTS
        print(
            f"[Visualization] QUICK_MODE enabled: up to {QUICK_MAX_FILES_PER_DATASET} CSV per dataset, "
            f"grid_points={grid_points}"
        )

    include_per_community_for = ("Micro", "MicroMacro")

    try:
        datasets = build_datasets_summary(
            folders=folders_for_run,
            pattern=pattern,
            grid_points=grid_points,
            include_per_community_for=include_per_community_for,
        )

        out_dir = _build_output_dir()
        max_time = _metric_time_limit()

        print(f"[Visualization] Matplotlib backend: {plt.get_backend()}")
        print(f"[Visualization] Output directory: {out_dir}")
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
                comm_metrics = _load_metrics_for_simulation(
                    folder=folders_for_run[name],
                    sim_id=rep_sim,
                    max_time=max_time,
                )

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
                name: _collect_metric_density_samples(folder=folder, max_time=max_time)
                for name, folder in folders_for_run.items()
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
                    if metric_key == "t_bridge" and BRIDGE_DENSITY_FIRST_N_COMMUNITIES is not None:
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
    finally:
        if temp_quick_root is not None:
            shutil.rmtree(temp_quick_root, ignore_errors=True)


if __name__ == "__main__":
    main()
