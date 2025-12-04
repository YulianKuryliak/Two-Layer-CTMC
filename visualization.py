import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from matplotlib import cm

# ========= settings =========
folders = {
    "Solid": r"data\Simulations_Solid",
    # "MicroMacro": r"C:\Users\UAKuryliYu\MyFolder\PeronalProjects\ManagingEpidemicOutbreak\EpidemicOutbreakPredictor\data\Simulations_MicroMacro",
    "MicroMacro_hazard": r"data\Simulations_MicroMacro_hazard_updated",
}
pattern = "*.csv"
GRID_POINTS = 1000  # resolution of the time grid INSIDE each dataset

# ========= helpers =========
def load_dataset(folder: str, pattern: str = "*.csv"):
    """
    Read all CSVs in folder, verify columns, aggregate over communities -> one I-curve per simulation.
    Returns:
        raw_curves: dict(sim_id -> DataFrame[time, I])
        t_min, t_max: min/max time across sims
    """
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(folder)}")

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
        sim_id = os.path.splitext(os.path.basename(fpath))[0]
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
    datasets[name] = {"raw": raw_curves, "tmin": tmin, "tmax": tmax}

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

# ========= visualization (all datasets on one plot) =========
plt.figure(figsize=(11, 6))

# choose base colors
base_colors = {
    "Solid":             "#1f77b4",  # blue
    # "MicroMacro":        "#d62728",  # red
    "MicroMacro_hazard": "#2ca02c",  # green
}

for name, d in datasets.items():
    color = base_colors.get(name, None)
    if color is None:
        # fallback if a new dataset is added
        color = np.random.choice(list(cm.tab20.colors))

    t_grid = d["t_grid"]

    # plot representative line
    plt.plot(t_grid, d["I_rep"], color=color, linewidth=2.8, label=f"{name} — representative")

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

out_dir = r"plots"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "{}.png".format(datetime.now().strftime("%Y-%m-%d %H.%M")))
plt.savefig(out_path, dpi=300, bbox_inches="tight")

plt.show()

# ========= reporting =========
for name, d in datasets.items():
    print(f"[{name}] Representative simulation: {d['rep_sim_id']}")
    print(f"[{name}] Time grid range used: [{d['tmin']:.3f}, {d['tmax']:.3f}] with {GRID_POINTS} points")
    print(f"[{name}] Mean MSE across sims: {np.mean(d['mse']):.6f}, best MSE: {np.min(d['mse']):.6f}")
