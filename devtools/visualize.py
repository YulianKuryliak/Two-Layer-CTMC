from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required; install two-layer-ctmc[analysis]") from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib import cm  # type: ignore
    except ImportError as exc:
        raise ImportError("matplotlib is required; install two-layer-ctmc[analysis]") from exc
    return plt, cm


def load_per_community_curves(folder: str | Path, sim_id: str) -> Dict[int, Any]:
    pd = _require_pandas()
    folder_path = Path(folder)
    fpath = folder_path / f"{sim_id}.csv"
    df = pd.read_csv(fpath)
    required = {"community", "time", "S", "I", "R"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fpath.name} missing columns: {missing}")

    curves = {}
    for comm_id, grp in df.groupby("community"):
        curves[int(comm_id)] = grp.sort_values("time")[["time", "I"]].reset_index(drop=True)
    return curves


def infection_markers(curve: Any) -> Tuple[float | None, float | None]:
    t = curve["time"].to_numpy()
    i_vals = curve["I"].to_numpy()

    start_idx = np.argmax(i_vals > 0) if np.any(i_vals > 0) else None
    if start_idx is None or i_vals[start_idx] == 0:
        return None, None

    positive_idxs = np.where(i_vals > 0)[0]
    last_pos = positive_idxs[-1]
    end_idx = last_pos

    return float(t[start_idx]), float(t[end_idx])


def load_dataset(folder: str | Path, pattern: str = "*.csv"):
    pd = _require_pandas()
    folder_path = Path(folder)
    files = sorted(folder_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path.resolve()}")

    raw_curves = {}
    tmins, tmaxs = [], []

    for fpath in files:
        df = pd.read_csv(fpath)
        required = {"community", "time", "S", "I", "R"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{fpath.name} missing columns: {missing}")

        agg = (
            df.groupby("time", as_index=False)[["S", "I", "R"]]
            .sum()
            .sort_values("time")
        )
        cur = agg[["time", "I"]].drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        sim_id = fpath.stem
        raw_curves[sim_id] = cur

        tmins.append(cur["time"].min())
        tmaxs.append(cur["time"].max())

    return raw_curves, float(np.max(tmins)), float(np.min(tmaxs))


def interp_stack_on_grid(raw_curves: dict, t_grid: np.ndarray):
    I_interp = {}
    for sim_id, cur in raw_curves.items():
        t = cur["time"].to_numpy()
        y = cur["I"].to_numpy()
        I_interp[sim_id] = np.interp(t_grid, t, y)
    I_stack = np.vstack([I_interp[sid] for sid in I_interp.keys()])
    return I_interp, I_stack


def representative_and_bands(I_stack: np.ndarray):
    I_mean = I_stack.mean(axis=0)
    mse = np.mean((I_stack - I_mean[None, :]) ** 2, axis=1)
    idx_rep = int(np.argmin(mse))
    I_rep = I_stack[idx_rep, :]

    median = np.median(I_stack, axis=0)
    q1 = np.quantile(I_stack, 0.25, axis=0)
    q3 = np.quantile(I_stack, 0.75, axis=0)
    return idx_rep, I_mean, I_rep, median, q1, q3, mse


def build_dataset_summary(
    folder: str | Path,
    pattern: str = "*.csv",
    grid_points: int = 1000,
    include_per_community: bool = False,
):
    raw_curves, tmin, tmax = load_dataset(folder, pattern)
    t_grid = np.linspace(tmin, tmax, grid_points)
    I_interp, I_stack = interp_stack_on_grid(raw_curves, t_grid)
    idx_rep, I_mean, I_rep, median, q1, q3, mse = representative_and_bands(I_stack)
    rep_key = list(I_interp.keys())[idx_rep]

    summary = {
        "folder": Path(folder),
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
    }
    if include_per_community:
        summary["rep_comm_curves"] = load_per_community_curves(folder, rep_key)
    return summary


def build_datasets_summary(
    folders: Dict[str, str | Path],
    pattern: str = "*.csv",
    grid_points: int = 1000,
    include_per_community_for: Tuple[str, ...] = (),
):
    datasets = {}
    for name, folder in folders.items():
        include = name in include_per_community_for
        datasets[name] = build_dataset_summary(
            folder=folder,
            pattern=pattern,
            grid_points=grid_points,
            include_per_community=include,
        )
    return datasets


def plot_total_dynamics(
    datasets: dict,
    out_path: str | Path,
    title: str = "Representative I-curves with interquartile bands",
    base_colors: Dict[str, str] | None = None,
    show_median_points: bool = False,
    median_point_stride: int = 25,
):
    plt, cm = _require_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    for name, d in datasets.items():
        color = None if base_colors is None else base_colors.get(name)
        if color is None:
            colors = list(cm.tab20.colors)
            color = colors[np.random.randint(len(colors))]

        t_grid = d["t_grid"]
        plt.plot(
            t_grid,
            d["I_rep"],
            color=color,
            linewidth=2.8,
            linestyle="-",
            label=f"{name} - representative",
        )
        if show_median_points:
            median = d.get("median")
            if median is None:
                median = np.median(d["I_stack"], axis=0)
            idx = slice(None, None, max(1, int(median_point_stride)))
            plt.scatter(
                t_grid[idx],
                median[idx],
                color=color,
                s=16,
                marker="o",
                alpha=0.85,
                label=f"{name} - median",
                zorder=4,
            )
        plt.fill_between(t_grid, d["q1"], d["q3"], color=color, alpha=0.18)

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


def plot_per_community_curves(
    comm_curves: Dict[int, Any],
    out_path: str | Path,
    title: str,
    show_markers: bool = True,
):
    plt, cm = _require_matplotlib()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    cmap = cm.get_cmap("tab10")

    for idx, comm_id in enumerate(sorted(comm_curves.keys())):
        cur = comm_curves[comm_id]
        plt.plot(
            cur["time"],
            cur["I"],
            color=cmap(idx % 10),
            linewidth=1.8,
            label=f"Community {comm_id}",
        )
        if show_markers:
            start_t, end_t = infection_markers(cur)
            if start_t is not None:
                start_y = float(cur.loc[cur["time"] == start_t, "I"].iloc[0])
                plt.scatter(start_t, start_y, color=cmap(idx % 10), marker="o", s=36, zorder=5)
            if end_t is not None:
                end_y = float(cur.loc[cur["time"] == end_t, "I"].iloc[0])
                plt.scatter(end_t, end_y, color=cmap(idx % 10), marker="s", s=46, zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")


__all__ = [
    "load_per_community_curves",
    "infection_markers",
    "load_dataset",
    "interp_stack_on_grid",
    "representative_and_bands",
    "build_dataset_summary",
    "build_datasets_summary",
    "plot_total_dynamics",
    "plot_per_community_curves",
]
