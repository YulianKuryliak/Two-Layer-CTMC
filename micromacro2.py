import os
import csv
import math
import random
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from network import generate_two_scale_network
from MicroMacroEngine import MicroModel, MacroEngine, Orchestrator
from sim_db import log_run


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path_like: str) -> Path:
    normalized = str(path_like).replace("\\", "/")
    path = Path(normalized).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


def load_config(path: str = "config.json") -> dict:
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# Experiments (DATA EXPORT)
# ---------------------------

def _export_discrete_grid_csv(
    logs_per_comm: dict,
    k: int,
    tau_micro: float,
    T_end: float,
    csv_path: str
):
    """
    Export **only** the discrete grid t = tau, 2*tau, ..., floor(T_end/tau)*tau.
    Between grid points nothing is saved.
    Uses carry-forward of the last state at or before each grid point.
    """
    csv_path = resolve_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the grid robustly against floating point drift
    n_steps = int(math.floor(T_end / tau_micro + 1e-12))
    grid = np.array([(i + 1) * tau_micro for i in range(n_steps)], dtype=float)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['community', 'time', 'S', 'I', 'R'])

        for community in range(k):
            times = np.asarray(logs_per_comm[community]['times'], dtype=float)
            Ss    = np.asarray(logs_per_comm[community]['S'],    dtype=int)
            Is    = np.asarray(logs_per_comm[community]['I'],    dtype=int)
            Rs    = np.asarray(logs_per_comm[community]['R'],    dtype=int)

            # ensure ascending
            order = np.argsort(times)
            times, Ss, Is, Rs = times[order], Ss[order], Is[order], Rs[order]

            # For each t_out on the grid, pick the last snapshot with time <= t_out
            idxs = np.searchsorted(times, grid, side='right') - 1
            # clip (because we recorded t=0, idxs should be >= 0)
            idxs = np.clip(idxs, 0, len(times) - 1)

            for t_out, idx in zip(grid, idxs):
                s, i, r = int(Ss[idx]), int(Is[idx]), int(Rs[idx])
                # write as exact float; optionally round for neatness
                writer.writerow([community, float(t_out), s, i, r])


def run_experiments(
    runs: int = 100,
    output_dir: str = "data/Simulations_MicroMacro_hazard_updated",
    k: int = 2,
    size: int = 50,
    inter_links: int = 1,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    base_seed: int = 42,
    beta: float = 0.01,
    gamma: float = 0.0,
    tau_micro: float = 1.0,
    T_end: float = 100.0,
    macro_T: float = 1.0,
    model: int = 2,
):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- generate ONE fixed two-scale network for all runs ----
    net_seed = base_seed  # можна окремо параметризувати, але логіка проста: одна мережа
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=k,
        community_size=size,
        inter_links=inter_links,
        seed=net_seed,
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
    )

    for run_id in range(1, runs + 1):
        seed = base_seed + run_id
        random.seed(seed)
        np.random.seed(seed)

        # 2) run two-scale simulator on the same network
        orch = Orchestrator(
            W=W,
            micro_graphs=micro_graphs,
            beta_micro=beta,
            gamma=gamma,
            beta_macro=beta,
            tau_micro=tau_micro,
            T_end=T_end,
            macro_T=macro_T,
            model=model,
            full_graph=full_graph,
            layout_seed=seed
        )
        # seed a single infection (uniform susceptible) in community 0 by default
        orch.micro_models[0]._infect_node()

        print(f"\n--- Run {run_id}/{runs} starts (tau={tau_micro}) ---")
        _, _, logs_2s, _ = orch.run()
        print(f"--- Run {run_id}/{runs} ends ---\n")

        # 3) export **ONLY** discrete grid t = τ, 2τ, ..., floor(T_end/τ)*τ
        csv_path = output_dir / f"{run_id}.csv"
        _export_discrete_grid_csv(
            logs_per_comm=logs_2s,
            k=k,
            tau_micro=tau_micro,
            T_end=T_end,
            csv_path=csv_path
        )
        print(f"Run {run_id}/{runs} → {csv_path}")


if __name__ == "__main__":
    cfg = load_config()
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_common = cfg["simulation"]
    sim_cfg = cfg["micromacro"]
    sim_variant = cfg["micromacro_v2"]

    k = int(net_cfg["communities"])
    size = int(net_cfg["community_size"])
    inter_links = int(net_cfg["inter_links"])
    macro_graph_type = str(net_cfg["macro_graph_type"])
    micro_graph_type = str(net_cfg["micro_graph_type"])
    edge_prob = float(net_cfg["edge_prob"])
    base_seed = int(sim_common["base_seed"])
    beta = float(virus_cfg["beta"])
    gamma = float(virus_cfg["gamma"])
    tau_micro = float(sim_cfg["tau_micro"])
    T_end = float(sim_common["T_end"])
    macro_T = float(sim_cfg["macro_T"])
    model = int(virus_cfg["model"])
    runs = int(sim_common["n_runs"])
    output_dir = sim_variant["out_folder"]

    run_experiments(
        runs=runs,
        output_dir=output_dir,
        k=k,             # number of communities
        size=size,        # size of one community
        inter_links=inter_links,    # number of links between communities
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
        base_seed=base_seed,
        beta=beta,      #infection rate
        gamma=gamma,       #recovery rate
        tau_micro=tau_micro,   # <-- set your step here; e.g., 0.1
        T_end=T_end,     # time end
        macro_T=macro_T,       # time step
        model=model,         # 1 - SIS 2 - SIR
    )

    log_run(
        simulator="MicroMacro_v2",
        sim_version="1.0.3",
        network_params=net_cfg,
        virus_params=virus_cfg,
        sim_params={
            "n_runs": runs,
            "base_seed": base_seed,
            "T_end": T_end,
            "tau_micro": tau_micro,
            "macro_T": macro_T,
            "out_folder": output_dir,
            "k": k,
            "size": size,
            "inter_links": inter_links,
        },
        output_path=output_dir,
    )


