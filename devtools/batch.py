import csv
import random
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from devtools.config import load_config, resolve_path
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulate import normalize_model
from two_layer_ctmc.simulators import MicroMacroSimulator, MicroSimulator


def _seed_list(n_runs: int, seeds: Optional[List[int]], base_seed: Optional[int]) -> List[Optional[int]]:
    if seeds is not None:
        if len(seeds) < n_runs:
            raise ValueError("seeds must have at least n_runs entries")
        return list(seeds[:n_runs])
    if base_seed is None:
        return [None] * n_runs
    return [base_seed + i for i in range(n_runs)]


def _write_micro_csv(rows: Iterable[tuple], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["community", "time", "S", "I", "R"])
        writer.writerows(rows)


def _export_discrete_grid_csv(
    logs_per_comm: dict,
    k: int,
    tau_micro: float,
    T_end: float,
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    n_steps = int(np.floor(T_end / tau_micro + 1e-12))
    grid = np.array([(i + 1) * tau_micro for i in range(n_steps)], dtype=float)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["community", "time", "S", "I", "R"])

        for community in range(k):
            times = np.asarray(logs_per_comm[community]["times"], dtype=float)
            Ss = np.asarray(logs_per_comm[community]["S"], dtype=int)
            Is = np.asarray(logs_per_comm[community]["I"], dtype=int)
            Rs = np.asarray(logs_per_comm[community]["R"], dtype=int)

            order = np.argsort(times)
            times, Ss, Is, Rs = times[order], Ss[order], Is[order], Rs[order]

            idxs = np.searchsorted(times, grid, side="right") - 1
            idxs = np.clip(idxs, 0, len(times) - 1)

            for t_out, idx in zip(grid, idxs):
                writer.writerow([community, float(t_out), int(Ss[idx]), int(Is[idx]), int(Rs[idx])])


def run_micro_batch(
    *,
    beta: float,
    gamma: float,
    T_end: float,
    dt_out: float,
    n_runs: int,
    out_folder: str | Path,
    model: int | str = 2,
    seeds: Optional[List[int]] = None,
    base_seed: Optional[int] = None,
    n_communities: int = 2,
    community_size: int = 50,
    inter_links: int = 1,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    initial_node: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> List[Path]:
    out_folder_path = resolve_path(out_folder, base_dir=base_dir)
    model_id = normalize_model(model)

    micro_graphs, full_graph, _ = generate_two_scale_network(
        n_communities=n_communities,
        community_size=community_size,
        inter_links=inter_links,
        seed=seed,
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
    )
    comm_nodes = [list(G.nodes()) for G in micro_graphs]

    sim = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=beta,
        recovery_rate=gamma,
        model=model_id,
    )

    nodes = list(full_graph.nodes())
    output_paths: List[Path] = []
    run_seeds = _seed_list(n_runs, seeds, base_seed)
    for run_idx in range(n_runs):
        run_seed = run_seeds[run_idx]
        rng = random.Random(run_seed)
        seed_node = initial_node
        if seed_node is None and nodes:
            seed_node = rng.choice(nodes)

        result = sim.run(
            T_end=T_end,
            dt_out=dt_out,
            initial_node=seed_node,
            rng=rng,
        )
        csv_path = out_folder_path / f"{run_idx + 1}.csv"
        _write_micro_csv(result["rows"], csv_path)
        output_paths.append(csv_path)

    return output_paths


def run_micromacro_batch(
    *,
    beta_micro: float,
    gamma: float,
    tau_micro: float,
    T_end: float,
    n_runs: int,
    out_folder: str | Path,
    beta_macro: Optional[float] = None,
    macro_T: float = 1.0,
    model: int | str = 2,
    seeds: Optional[List[int]] = None,
    base_seed: Optional[int] = None,
    n_communities: int = 2,
    community_size: int = 50,
    inter_links: int = 1,
    seed: Optional[int] = None,
    macro_graph_type: str = "complete",
    micro_graph_type: str = "complete",
    edge_prob: float = 0.1,
    initial_community: int = 0,
    initial_node: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> List[Path]:
    out_folder_path = resolve_path(out_folder, base_dir=base_dir)
    model_id = normalize_model(model)
    beta_macro = beta_micro if beta_macro is None else beta_macro

    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=n_communities,
        community_size=community_size,
        inter_links=inter_links,
        seed=seed,
        macro_graph_type=macro_graph_type,
        micro_graph_type=micro_graph_type,
        edge_prob=edge_prob,
    )

    sim = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=beta_micro,
        gamma=gamma,
        beta_macro=beta_macro,
        tau_micro=tau_micro,
        T_end=T_end,
        macro_T=macro_T,
        model=model_id,
        full_graph=full_graph,
    )

    output_paths: List[Path] = []
    run_seeds = _seed_list(n_runs, seeds, base_seed)
    for run_idx in range(n_runs):
        run_seed = run_seeds[run_idx]
        result = sim.run(
            seed=run_seed,
            initial_community=initial_community,
            initial_node=initial_node,
        )
        _, _, logs, _ = result
        csv_path = out_folder_path / f"{run_idx + 1}.csv"
        _export_discrete_grid_csv(
            logs_per_comm=logs,
            k=n_communities,
            tau_micro=tau_micro,
            T_end=T_end,
            csv_path=csv_path,
        )
        output_paths.append(csv_path)

    return output_paths


def run_micro_batch_from_config(path: str | Path = "config.json") -> List[Path]:
    cfg, base_dir = load_config(path)
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_cfg = cfg["micro"]
    sim_common = cfg["simulation"]

    return run_micro_batch(
        beta=float(virus_cfg["beta"]),
        gamma=float(virus_cfg["gamma"]),
        T_end=float(sim_common["T_end"]),
        dt_out=float(sim_cfg["dt_out"]),
        n_runs=int(sim_common["n_runs"]),
        out_folder=sim_cfg["out_folder"],
        model=int(virus_cfg["model"]),
        base_seed=int(sim_common["base_seed"]),
        n_communities=int(net_cfg["communities"]),
        community_size=int(net_cfg["community_size"]),
        inter_links=int(net_cfg["inter_links"]),
        seed=int(net_cfg["seed"]),
        macro_graph_type=str(net_cfg["macro_graph_type"]),
        micro_graph_type=str(net_cfg["micro_graph_type"]),
        edge_prob=float(net_cfg["edge_prob"]),
        base_dir=base_dir,
    )


def run_micromacro_batch_from_config(
    path: str | Path = "config.json",
    variant: str = "micromacro_v2",
) -> List[Path]:
    cfg, base_dir = load_config(path)
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_common = cfg["simulation"]
    sim_cfg = cfg["micromacro"]
    variant_cfg = cfg.get(variant, {})

    return run_micromacro_batch(
        beta_micro=float(virus_cfg["beta"]),
        gamma=float(virus_cfg["gamma"]),
        tau_micro=float(sim_cfg["tau_micro"]),
        T_end=float(sim_common["T_end"]),
        n_runs=int(sim_common["n_runs"]),
        out_folder=variant_cfg.get("out_folder", "data/Simulations_MicroMacro"),
        beta_macro=float(virus_cfg["beta"]),
        macro_T=float(sim_cfg["macro_T"]),
        model=int(virus_cfg["model"]),
        base_seed=int(sim_common["base_seed"]),
        n_communities=int(net_cfg["communities"]),
        community_size=int(net_cfg["community_size"]),
        inter_links=int(net_cfg["inter_links"]),
        seed=int(net_cfg["seed"]),
        macro_graph_type=str(net_cfg["macro_graph_type"]),
        micro_graph_type=str(net_cfg["micro_graph_type"]),
        edge_prob=float(net_cfg["edge_prob"]),
        base_dir=base_dir,
    )


__all__ = [
    "run_micro_batch",
    "run_micromacro_batch",
    "run_micro_batch_from_config",
    "run_micromacro_batch_from_config",
]
