import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

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
GRID_POINTS = 1000  # resolution of the time grid INSIDE each dataset
PLOT_TOTAL_DYNAMICS = True
PLOT_PER_COMMUNITY = True
PLOT_MICROMACRO_COMPARISON = True
SHOW_MEDIAN_POINTS = True
MEDIAN_POINT_STRIDE = 1

include_per_community_for = ["Micro", "MicroMacro"]

datasets = build_datasets_summary(
    folders=folders,
    pattern=pattern,
    grid_points=GRID_POINTS,
    include_per_community_for=tuple(include_per_community_for),
)

# ========= output folder =========
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
out_dir = resolve_path(Path("plots") / folder_name, base_dir=base_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ========= visualization (all datasets on one plot) =========
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

# ========= per-community curves for each representative simulation =========
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

# ========= combined per-community plot for Micro vs MicroMacro =========
if PLOT_MICROMACRO_COMPARISON and "Micro" in datasets and "MicroMacro" in datasets:
    d1 = datasets["Micro"]
    d2 = datasets["MicroMacro"]
    comms = sorted(set(d1["rep_comm_curves"].keys()) & set(d2["rep_comm_curves"].keys()))

    plt.figure(figsize=(11, 6))
    cmap = cm.get_cmap("tab10")

    for idx, comm_id in enumerate(comms):
        c = cmap(idx % 10)
        cur1 = d1["rep_comm_curves"][comm_id]
        cur2 = d2["rep_comm_curves"][comm_id]
        plt.plot(cur1["time"], cur1["I"], color=c, linewidth=1.6, label=f"Comm {comm_id} micro")
        plt.plot(cur2["time"], cur2["I"], color=c, linewidth=1.6, linestyle="--", label=f"Comm {comm_id} micromacro")

    plt.xlabel("Time")
    plt.ylabel("Infected (I)")
    plt.title("Representative per-community I(t): Micro vs MicroMacro")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    out_path_comm = out_dir / "Micro_vs_MicroMacro_community.png"
    plt.savefig(out_path_comm, dpi=300, bbox_inches="tight")

plt.show()

# ========= reporting =========
for name, d in datasets.items():
    print(f"[{name}] Representative simulation: {d['rep_sim_id']}")
    print(f"[{name}] Time grid range used: [{d['tmin']:.3f}, {d['tmax']:.3f}] with {GRID_POINTS} points")
    print(f"[{name}] Mean MSE across sims: {np.mean(d['mse']):.6f}, best MSE: {np.min(d['mse']):.6f}")
