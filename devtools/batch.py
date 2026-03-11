import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def _community_size_from_config(net_cfg: dict) -> int:
    return int(net_cfg["community_size"])


def _initial_node_from_config(sim_common: dict, net_cfg: dict) -> Optional[int]:
    initial_node = sim_common.get("initial_node")
    if initial_node is not None:
        return int(initial_node)

    macro_graph_type = str(net_cfg.get("macro_graph_type", "")).strip().lower()
    if macro_graph_type in {"star", "hub_spoke", "hub-and-spoke", "hubspoke"}:
        # By default, start from node 0 for star experiments.
        return 0
    return None


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


def _bridge_nodes_by_community(full_graph, comm_nodes: List[List[int]]) -> Dict[int, set[int]]:
    node_to_comm: Dict[int, int] = {}
    for comm_id, nodes in enumerate(comm_nodes):
        for node in nodes:
            node_to_comm[int(node)] = comm_id

    bridge_nodes: Dict[int, set[int]] = {comm_id: set() for comm_id in range(len(comm_nodes))}
    for u, v in full_graph.edges():
        src_comm = node_to_comm.get(int(u))
        dst_comm = node_to_comm.get(int(v))
        if src_comm is None or dst_comm is None or src_comm == dst_comm:
            continue
        bridge_nodes[src_comm].add(int(u))
        bridge_nodes[dst_comm].add(int(v))
    return bridge_nodes


def _empty_community_metrics(n_communities: int) -> List[dict]:
    return [
        {
            "community": comm_id,
            "t0": None,
            "t_bridge": None,
            "t_export": None,
        }
        for comm_id in range(n_communities)
    ]


def _metrics_from_micro_transmissions(
    transmission_events: list[dict],
    n_communities: int,
    bridge_nodes: Dict[int, set[int]],
) -> List[dict]:
    metrics = _empty_community_metrics(n_communities)
    for event in transmission_events:
        dst_comm = event.get("dst_community")
        if dst_comm is None:
            continue
        dst_comm = int(dst_comm)
        if not (0 <= dst_comm < n_communities):
            continue

        event_time = float(event["time"])
        if metrics[dst_comm]["t0"] is None:
            metrics[dst_comm]["t0"] = event_time

        dst_node = event.get("dst_node")
        if dst_node is not None and metrics[dst_comm]["t_bridge"] is None:
            if int(dst_node) in bridge_nodes.get(dst_comm, set()):
                metrics[dst_comm]["t_bridge"] = event_time

        src_comm = event.get("src_community")
        if src_comm is None:
            continue
        src_comm = int(src_comm)
        if 0 <= src_comm < n_communities and src_comm != dst_comm and metrics[src_comm]["t_export"] is None:
            metrics[src_comm]["t_export"] = event_time

    return metrics


def _metrics_from_micromacro_events(
    event_log: list[dict],
    n_communities: int,
    bridge_nodes: Dict[int, set[int]],
) -> List[dict]:
    metrics = _empty_community_metrics(n_communities)
    for event in event_log:
        event_type = str(event.get("event_type", "")).lower()
        comm = event.get("community")
        if comm is None:
            continue
        comm = int(comm)
        if not (0 <= comm < n_communities):
            continue

        event_time = float(event["time"])
        if event_type in {"seed", "infection", "transfer"}:
            if metrics[comm]["t0"] is None:
                metrics[comm]["t0"] = event_time

            node = event.get("node")
            if node is not None and metrics[comm]["t_bridge"] is None:
                if int(node) in bridge_nodes.get(comm, set()):
                    metrics[comm]["t_bridge"] = event_time

        if event_type == "transfer":
            src_comm = event.get("src")
            if src_comm is not None:
                src_comm = int(src_comm)
                if 0 <= src_comm < n_communities and src_comm != comm and metrics[src_comm]["t_export"] is None:
                    metrics[src_comm]["t_export"] = event_time

    return metrics


def _write_metrics_files(
    *,
    simulator: str,
    sim_csv_path: Path,
    metrics: List[dict],
) -> None:
    metrics_dir = sim_csv_path.parent / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv_path = metrics_dir / f"{sim_csv_path.stem}_metrics.csv"
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["community", "t0", "t_bridge", "t_export"])
        for row in metrics:
            writer.writerow([row["community"], row["t0"], row["t_bridge"], row["t_export"]])

    metrics_json_path = metrics_dir / f"{sim_csv_path.stem}_metrics.json"
    payload = {
        "simulator": simulator,
        "run_id": sim_csv_path.stem,
        "source_csv": sim_csv_path.name,
        "metrics": metrics,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _extract_micromacro_infection_events(event_log: list[dict]) -> List[dict]:
    """
    Keep only infection-like events from MicroMacro logs:
    - micro infection events (mode=micro, event_type=infection)
    - macro transfer events (mode=macro, event_type=transfer)
    """
    out: List[dict] = []
    for event in event_log:
        mode = str(event.get("mode", "")).lower()
        event_type = str(event.get("event_type", "")).lower()
        if "time" not in event:
            continue
        time_val = float(event["time"])

        if mode == "micro" and event_type == "infection":
            dst_comm = event.get("community")
            dst_node = event.get("node")
            src_node = event.get("src")
            out.append(
                {
                    "time": time_val,
                    "mode": "micro",
                    "kind": "infection",
                    "src_community": int(dst_comm) if dst_comm is not None else None,
                    "dst_community": int(dst_comm) if dst_comm is not None else None,
                    "src_node": int(src_node) if src_node is not None else None,
                    "dst_node": int(dst_node) if dst_node is not None else None,
                }
            )
        elif mode == "macro" and event_type == "transfer":
            src_comm = event.get("src")
            dst_comm = event.get("community")
            dst_node = event.get("node")
            out.append(
                {
                    "time": time_val,
                    "mode": "macro",
                    "kind": "transfer",
                    "src_community": int(src_comm) if src_comm is not None else None,
                    "dst_community": int(dst_comm) if dst_comm is not None else None,
                    "src_node": None,
                    "dst_node": int(dst_node) if dst_node is not None else None,
                }
            )

    out.sort(key=lambda row: row["time"])
    return out


def _write_micromacro_infection_events_csv(
    *,
    sim_csv_path: Path,
    infection_events: List[dict],
) -> None:
    events_dir = sim_csv_path.parent / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    events_csv_path = events_dir / f"{sim_csv_path.stem}_infection_events.csv"
    with open(events_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time",
                "mode",
                "kind",
                "src_community",
                "dst_community",
                "src_node",
                "dst_node",
            ]
        )
        for row in infection_events:
            writer.writerow(
                [
                    float(row["time"]),
                    str(row["mode"]),
                    str(row["kind"]),
                    row["src_community"],
                    row["dst_community"],
                    row["src_node"],
                    row["dst_node"],
                ]
            )


def _print_micromacro_infection_events(run_idx: int, infection_events: List[dict]) -> None:
    if not infection_events:
        print(f"[Run {run_idx}] No infection events recorded.")
        return

    print(f"[Run {run_idx}] Infection events (time | mode | details):")
    for row in infection_events:
        t = float(row["time"])
        mode = str(row["mode"])
        if mode == "micro":
            print(
                "  "
                f"t={t:.6f} | MICRO | "
                f"community={row['dst_community']} "
                f"src_node={row['src_node']} -> dst_node={row['dst_node']}"
            )
        else:
            print(
                "  "
                f"t={t:.6f} | MACRO | "
                f"community={row['src_community']} -> community={row['dst_community']} "
                f"dst_node={row['dst_node']}"
            )


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
    leaf_count: int = 0,
    leaf_degree: int = 1,
    star_leaf_attachment: str = "random",
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
        leaf_count=leaf_count,
        leaf_degree=leaf_degree,
        star_leaf_attachment=star_leaf_attachment,
    )
    comm_nodes = [list(G.nodes()) for G in micro_graphs]

    sim = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=beta,
        recovery_rate=gamma,
        model=model_id,
    )
    bridge_nodes = _bridge_nodes_by_community(full_graph=full_graph, comm_nodes=comm_nodes)

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
        metrics = _metrics_from_micro_transmissions(
            transmission_events=result.get("transmission_events", []),
            n_communities=n_communities,
            bridge_nodes=bridge_nodes,
        )
        _write_metrics_files(simulator="Micro", sim_csv_path=csv_path, metrics=metrics)
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
    leaf_count: int = 0,
    leaf_degree: int = 1,
    star_leaf_attachment: str = "random",
    print_infection_events: bool = True,
    export_infection_events_csv: bool = True,
    verbose_steps: bool = False,
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
        leaf_count=leaf_count,
        leaf_degree=leaf_degree,
        star_leaf_attachment=star_leaf_attachment,
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
        verbose_steps=verbose_steps,
    )
    comm_nodes = [list(G.nodes()) for G in micro_graphs]
    bridge_nodes = _bridge_nodes_by_community(full_graph=full_graph, comm_nodes=comm_nodes)

    output_paths: List[Path] = []
    run_seeds = _seed_list(n_runs, seeds, base_seed)
    for run_idx in range(n_runs):
        run_seed = run_seeds[run_idx]
        result = sim.run(
            seed=run_seed,
            initial_community=initial_community,
            initial_node=initial_node,
        )
        _, _, logs, event_log = result
        csv_path = out_folder_path / f"{run_idx + 1}.csv"
        _export_discrete_grid_csv(
            logs_per_comm=logs,
            k=n_communities,
            tau_micro=tau_micro,
            T_end=T_end,
            csv_path=csv_path,
        )
        infection_events = _extract_micromacro_infection_events(event_log)
        if export_infection_events_csv:
            _write_micromacro_infection_events_csv(
                sim_csv_path=csv_path,
                infection_events=infection_events,
            )
        if print_infection_events:
            _print_micromacro_infection_events(run_idx + 1, infection_events)

        metrics = _metrics_from_micromacro_events(
            event_log=event_log,
            n_communities=n_communities,
            bridge_nodes=bridge_nodes,
        )
        _write_metrics_files(simulator="MicroMacro", sim_csv_path=csv_path, metrics=metrics)
        output_paths.append(csv_path)

    return output_paths


def run_micro_batch_from_config(path: str | Path = "config.json") -> List[Path]:
    cfg, base_dir = load_config(path)
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_cfg = cfg["micro"]
    sim_common = cfg["simulation"]
    initial_node = _initial_node_from_config(sim_common=sim_common, net_cfg=net_cfg)

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
        community_size=_community_size_from_config(net_cfg),
        inter_links=int(net_cfg["inter_links"]),
        seed=int(net_cfg["seed"]),
        macro_graph_type=str(net_cfg["macro_graph_type"]),
        micro_graph_type=str(net_cfg["micro_graph_type"]),
        edge_prob=float(net_cfg["edge_prob"]),
        leaf_count=int(net_cfg.get("leaf_count", 0)),
        leaf_degree=int(net_cfg.get("leaf_degree", 1)),
        star_leaf_attachment=str(net_cfg.get("star_leaf_attachment", "random")),
        initial_node=initial_node,
        base_dir=base_dir,
    )


def run_micromacro_batch_from_config(
    path: str | Path = "config.json",
    variant: str = "micromacro",
) -> List[Path]:
    cfg, base_dir = load_config(path)
    net_cfg = cfg["network"]
    virus_cfg = cfg["virus"]
    sim_common = cfg["simulation"]
    sim_cfg = cfg["micromacro"]
    variant_cfg = cfg.get(variant, {})
    initial_node = _initial_node_from_config(sim_common=sim_common, net_cfg=net_cfg)

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
        community_size=_community_size_from_config(net_cfg),
        inter_links=int(net_cfg["inter_links"]),
        seed=int(net_cfg["seed"]),
        macro_graph_type=str(net_cfg["macro_graph_type"]),
        micro_graph_type=str(net_cfg["micro_graph_type"]),
        edge_prob=float(net_cfg["edge_prob"]),
        leaf_count=int(net_cfg.get("leaf_count", 0)),
        leaf_degree=int(net_cfg.get("leaf_degree", 1)),
        star_leaf_attachment=str(net_cfg.get("star_leaf_attachment", "random")),
        print_infection_events=bool(sim_cfg.get("print_infection_events", True)),
        export_infection_events_csv=bool(sim_cfg.get("export_infection_events_csv", True)),
        verbose_steps=bool(sim_cfg.get("verbose_steps", False)),
        initial_node=initial_node,
        base_dir=base_dir,
    )


__all__ = [
    "run_micro_batch",
    "run_micromacro_batch",
    "run_micro_batch_from_config",
    "run_micromacro_batch_from_config",
]
