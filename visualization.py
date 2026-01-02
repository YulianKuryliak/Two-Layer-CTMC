import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from pathlib import Path

# ========= settings =========
BASE_DIR = Path(__file__).resolve().parent
folders = {
    "Simulations_MicroMacro_2": BASE_DIR / "data" / "Simulations_MicroMacro_2",
    "Simulations_MicroMacro_2 copy": BASE_DIR / "data" / "Simulations_MicroMacro_2 copy",
}
pattern = "*.csv"
GRID_POINTS = 1000  # resolution of the time grid INSIDE each dataset
PLOT_TOTAL_DYNAMICS = True
PLOT_PER_COMMUNITY = True
PLOT_MICROMACRO_COMPARISON = False

# ========= helpers =========
def load_per_community_curves(folder: str, sim_id: str):
    """
    Load a single simulation CSV and return per-community I(t) curves.
    Returns dict[int -> DataFrame[time, I]] sorted by time.
    """
    folder_path = Path(folder)
    fpath = folder_path / f"{sim_id}.csv"
    df = pd.read_csv(fpath)
    required = {"community", "time", "S", "I", "R"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(fpath)} missing columns: {missing}")

    curves = {}
    for comm_id, grp in df.groupby("community"):
        curves[int(comm_id)] = grp.sort_values("time")[["time", "I"]].reset_index(drop=True)
    return curves


def infection_markers(curve: pd.DataFrame):
    """
    Given a per-community curve with columns time, I, find:
      - first time infection appears (I > 0)
      - last time infection is present (last I > 0)
    Returns (start_time, end_time) where values can be None.
    """
    t = curve["time"].to_numpy()
    i_vals = curve["I"].to_numpy()

    start_idx = np.argmax(i_vals > 0) if np.any(i_vals > 0) else None
    if start_idx is None or i_vals[start_idx] == 0:
        return None, None

    positive_idxs = np.where(i_vals > 0)[0]
    last_pos = positive_idxs[-1]
    end_idx = last_pos

    return float(t[start_idx]), float(t[end_idx])


def load_dataset(folder: str, pattern: str = "*.csv"):
    """
    Read all CSVs in folder, verify columns, aggregate over communities -> one I-curve per simulation.
    Returns:
        raw_curves: dict(sim_id -> DataFrame[time, I])
        t_min, t_max: min/max time across sims
    """
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
            raise ValueError(f"{os.path.basename(fpath)} missing columns: {missing}")

        # Sum over communities at each time -> whole-network I(t) per simulation
        agg = (df.groupby("time", as_index=False)[["S", "I", "R"]]
                 .sum()
                 .sort_values("time"))
        # keep only time & I
        cur = agg[["time", "I"]].drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
        sim_id = fpath.stem
        raw_curves[sim_id] = cur

        tmins.append(cur["time"].min())
        tmaxs.append(cur["time"].max())

    return raw_curves, float(np.max(tmins)), float(np.min(tmaxs))


def interp_stack_on_grid(raw_curves: dict, t_grid: np.ndarray):
    """
    Interpolate each simulation curve onto t_grid.
    Returns:
        I_interp: dict(sim_id -> np.array shape (K,))
        I_stack:  np.ndarray shape (N, K)
    """
    I_interp = {}
    for sim_id, cur in raw_curves.items():
        t = cur["time"].to_numpy()
        y = cur["I"].to_numpy()
        I_interp[sim_id] = np.interp(t_grid, t, y)
    I_stack = np.vstack([I_interp[sid] for sid in I_interp.keys()])
    return I_interp, I_stack


def representative_and_bands(I_stack: np.ndarray):
    """
    Given an (N_sim, K_time) stack, compute:
      - mean curve
      - index of representative (min MSE to mean)
      - interquartile envelopes (Q1 and Q3) across simulations at each time
    Returns:
      idx_rep, I_mean, I_rep, q1, q3, mse
    """
    I_mean = I_stack.mean(axis=0)  # (K,)
    # MSE to mean per sim
    mse = np.mean((I_stack - I_mean[None, :])**2, axis=1)  # (N,)
    idx_rep = int(np.argmin(mse))
    I_rep = I_stack[idx_rep, :]

    q1 = np.quantile(I_stack, 0.25, axis=0)
    q3 = np.quantile(I_stack, 0.75, axis=0)
    return idx_rep, I_mean, I_rep, q1, q3, mse


# ========= load all datasets =========
datasets = {}

for name, folder in folders.items():
    raw_curves, tmin, tmax = load_dataset(folder, pattern)
    datasets[name] = {"raw": raw_curves, "tmin": tmin, "tmax": tmax, "folder": folder}

# ========= process each dataset on its OWN grid =========
for name, d in datasets.items():
    # окремий часовий інтервал для кожного варіанта симулятора
    t_grid = np.linspace(d["tmin"], d["tmax"], GRID_POINTS)
    d["t_grid"] = t_grid

    I_interp, I_stack = interp_stack_on_grid(d["raw"], t_grid)
    idx_rep, I_mean, I_rep, q1, q3, mse = representative_and_bands(I_stack)

    # store results
    d["I_interp"] = I_interp
    d["I_stack"]  = I_stack
    d["idx_rep"]  = idx_rep
    d["I_mean"]   = I_mean
    d["I_rep"]    = I_rep
    d["q1"]       = q1
    d["q3"]       = q3
    d["mse"]      = mse
    # resolve representative sim_id
    rep_key = list(I_interp.keys())[idx_rep]
    d["rep_sim_id"] = rep_key
    if PLOT_PER_COMMUNITY and name in {"Micro", "Micro_copy_2"}:
        d["rep_comm_curves"] = load_per_community_curves(d["folder"], rep_key)

# ========= output folder =========
with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
net_cfg = cfg["network"]
edge_prob_str = str(net_cfg["edge_prob"]).replace(".", "p")
folder_name = (
    f"k_{net_cfg['communities']}_"
    f"n_{net_cfg['community_size']}_"
    f"inter_{net_cfg['inter_links']}_"
    f"macro_{net_cfg['macro_graph_type']}_"
    f"micro_{net_cfg['micro_graph_type']}_"
    f"p{edge_prob_str}"
)
out_dir = BASE_DIR / "plots" / folder_name
out_dir.mkdir(parents=True, exist_ok=True)

# ========= visualization (all datasets on one plot) =========
if PLOT_TOTAL_DYNAMICS:
    plt.figure(figsize=(11, 6))

    # choose base colors
    base_colors = {
        "Simulations_MicroMacro_2 copy":       "#1f77b4",  # blue
        "Simulations_MicroMacro_2": "#ff7f0e",  # orange
    }

    for name, d in datasets.items():
        color = base_colors.get(name, None)
        if color is None:
            # fallback if a new dataset is added
            colors = list(cm.tab20.colors)
            color = colors[np.random.randint(len(colors))]
        linestyle = "-"

        t_grid = d["t_grid"]

        # plot representative line
        plt.plot(t_grid, d["I_rep"], color=color, linewidth=2.8, linestyle=linestyle, label=f"{name} - representative")

        # interquartile bands
        plt.fill_between(t_grid, d["q1"], d["q3"],
                         color=color, alpha=0.18)

    # axes & legend
    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title("Representative I-curves with interquartile bands (all datasets, own time ranges)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    out_path = out_dir / "Micro_vs_Micro_copy_2.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")

# ========= per-community curves for each representative simulation =========
if PLOT_PER_COMMUNITY and "Micro" in datasets:
    d = datasets["Micro"]
    rep_sim = d["rep_sim_id"]
    comm_curves = d["rep_comm_curves"]

    plt.figure(figsize=(11, 6))
    cmap = cm.get_cmap("tab10")

    for idx, comm_id in enumerate(sorted(comm_curves.keys())):
        cur = comm_curves[comm_id]
        plt.plot(cur["time"], cur["I"],
                 color=cmap(idx % 10),
                 linewidth=1.8,
                 label=f"Community {comm_id}")

        start_t, end_t = infection_markers(cur)
        if start_t is not None:
            start_y = float(cur.loc[cur["time"] == start_t, "I"].iloc[0])
            plt.scatter(start_t, start_y, color=cmap(idx % 10), marker="o", s=36, zorder=5)
        if end_t is not None:
            end_y = float(cur.loc[cur["time"] == end_t, "I"].iloc[0])
            plt.scatter(end_t, end_y, color=cmap(idx % 10), marker="s", s=46, zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title(f"Micro: representative simulation {rep_sim} per-community I(t)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path_comm = out_dir / "Micro_community.png"
    plt.savefig(out_path_comm, dpi=300, bbox_inches="tight")

if PLOT_PER_COMMUNITY and "Micro_copy_2" in datasets:
    d = datasets["Micro_copy_2"]
    rep_sim = d["rep_sim_id"]
    comm_curves = d["rep_comm_curves"]

    plt.figure(figsize=(11, 6))
    cmap = cm.get_cmap("tab10")

    for idx, comm_id in enumerate(sorted(comm_curves.keys())):
        cur = comm_curves[comm_id]
        plt.plot(cur["time"], cur["I"],
                 color=cmap(idx % 10),
                 linewidth=1.8,
                 label=f"Community {comm_id}")

        start_t, end_t = infection_markers(cur)
        if start_t is not None:
            start_y = float(cur.loc[cur["time"] == start_t, "I"].iloc[0])
            plt.scatter(start_t, start_y, color=cmap(idx % 10), marker="o", s=36, zorder=5)
        if end_t is not None:
            end_y = float(cur.loc[cur["time"] == end_t, "I"].iloc[0])
            plt.scatter(end_t, end_y, color=cmap(idx % 10), marker="s", s=46, zorder=5)

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title(f"Micro_copy_2: representative simulation {rep_sim} per-community I(t)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path_comm = out_dir / "Micro_copy_2_community.png"
    plt.savefig(out_path_comm, dpi=300, bbox_inches="tight")

# ========= combined per-community plot for both MicroMacro versions =========
if PLOT_MICROMACRO_COMPARISON and "MicroMacro_hazard" in datasets and "MicroMacro_2" in datasets:
    d1 = datasets["MicroMacro_hazard"]
    d2 = datasets["MicroMacro_2"]
    comms = sorted(set(d1["rep_comm_curves"].keys()) & set(d2["rep_comm_curves"].keys()))

    plt.figure(figsize=(11, 6))
    cmap = cm.get_cmap("tab10")

    for idx, comm_id in enumerate(comms):
        c = cmap(idx % 10)
        cur1 = d1["rep_comm_curves"][comm_id]
        cur2 = d2["rep_comm_curves"][comm_id]
        plt.plot(cur1["time"], cur1["I"], color=c, linewidth=1.6, label=f"Comm {comm_id} v1")
        plt.plot(cur2["time"], cur2["I"], color=c, linewidth=1.6, linestyle="--", label=f"Comm {comm_id} v2")

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title("Representative per-community I(t): MicroMacro v1 vs v2")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path_comm = out_dir / "MicroMacro_community.png"
    plt.savefig(out_path_comm, dpi=300, bbox_inches="tight")

plt.show()

# ========= reporting =========
for name, d in datasets.items():
    print(f"[{name}] Representative simulation: {d['rep_sim_id']}")
    print(f"[{name}] Time grid range used: [{d['tmin']:.3f}, {d['tmax']:.3f}] with {GRID_POINTS} points")
    print(f"[{name}] Mean MSE across sims: {np.mean(d['mse']):.6f}, best MSE: {np.min(d['mse']):.6f}")
