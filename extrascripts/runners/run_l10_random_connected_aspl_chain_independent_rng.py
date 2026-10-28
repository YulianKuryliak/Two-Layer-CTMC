#!/usr/bin/env python3
"""Chain ASPL experiment: Full Gillespie vs Production MicroMacro + independent RNG.

Safe stages:
  python extrascripts/runners/run_l10_random_connected_aspl_chain_independent_rng.py --topology-only
  python extrascripts/runners/run_l10_random_connected_aspl_chain_independent_rng.py --smoke-only --workers 1

The default command resumes the full collection.  It never starts smoke
automatically and it does not modify production source files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.runners.checkpoint_io import read_checkpoint, write_checkpoint
from extrascripts.runners.run_original_benchmarks_independent_rng import (
    Benchmark,
    FULL,
    MICROMACRO,
    _regression_gate,
    _run_experimental_micromacro,
)
from two_layer_ctmc.simulators.micro import MicroSimulator


# ============================ EDIT HERE ============================
L = 10
N = 1000
K_VALUES = (1, 5)
ASPL_TARGETS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
BETA = 0.003
TAU_MICRO = 1.0
T_MAX = 30_000.0
REPLICATIONS = 100
WORKERS = 2
RANDOM_INITIAL_NODE = False
# ===================================================================

# The replication grain is fixed deliberately: 20 frozen chain networks and
# five distinct interface/event replicates on each network.
GAMMA, MODEL, MACRO_T, BASE_SEED = 0.0, 2, 1.0, 20_260_919
NETWORKS = 20
REPLICATES_PER_NETWORK = REPLICATIONS // NETWORKS
MEAN_DEGREE_BY_ASPL = {2.0: 50, 2.5: 25, 3.0: 12, 3.5: 8, 4.0: 6, 4.5: 5, 5.0: 4.3}
ASPL_TOLERANCE = {2.0: 0.10, 2.5: 0.10, 3.0: 0.10, 3.5: 0.12, 4.0: 0.15, 4.5: 0.15, 5.0: 0.15}
MAX_GRAPH_ATTEMPTS = 100

EXPERIMENT = "l10_n1000_random_connected_tree_augmented_aspl_chain"
DATA_ROOT = ROOT / "output" / "data" / EXPERIMENT

if REPLICATIONS != NETWORKS * REPLICATES_PER_NETWORK:
    raise ValueError("REPLICATIONS must equal NETWORKS times REPLICATES_PER_NETWORK")


def stable_seed(*parts: object) -> int:
    """Small deterministic seed helper; this value is not an output identifier."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return value


def token(value: float | int) -> str:
    return f"{value:g}".replace(".", "p")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def experiment_parameters() -> dict[str, Any]:
    return {
        "macro_topology": "chain",
        "communities": L,
        "community_size": N,
        "k_values": list(K_VALUES),
        "target_ASPL": list(ASPL_TARGETS),
        "target_mean_degree_by_ASPL": {str(key): value for key, value in MEAN_DEGREE_BY_ASPL.items()},
        "ASPL_tolerance": {str(key): value for key, value in ASPL_TOLERANCE.items()},
        "graph_generator": "random labeled spanning tree plus random additional missing edges",
        "beta_in": BETA,
        "beta_out": BETA,
        "gamma": GAMMA,
        "model": "SI",
        "tau_micro": TAU_MICRO,
        "macro_T": MACRO_T,
        "T_max": T_MAX,
        "initial_node": 0,
        "paired_runs_per_scenario": REPLICATIONS,
        "network_realizations": NETWORKS,
        "replicates_per_network": REPLICATES_PER_NETWORK,
        "independent_rng_per_micro_model": True,
        "censoring_policy": "unreached community has null arrival time; T_max is not an arrival time",
        "output_folder": f"output/data/{EXPERIMENT}",
    }


def ensure_parameters() -> None:
    path = DATA_ROOT / "experiment_parameters.json"
    desired = experiment_parameters()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != desired:
            raise RuntimeError(f"existing parameters differ from this runner: {path}")
        return
    atomic_json(path, desired)


def exact_aspl(graph: nx.Graph) -> float:
    if graph.number_of_edges() == N * (N - 1) // 2:
        return 1.0
    total = 0
    for node in graph:
        total += sum(nx.single_source_shortest_path_length(graph, node).values())
    return total / (N * (N - 1))


def graph_path(target: float, network_id: int, community: int) -> Path:
    return DATA_ROOT / "graphs" / f"aspl{token(target)}" / f"network{network_id:03d}_community{community:02d}.json.gz"


def validate_graph(graph: nx.Graph, target: float, actual_aspl: float) -> None:
    target_degree = MEAN_DEGREE_BY_ASPL[target]
    target_edges = int(round(target_degree * N / 2))
    if set(graph) != set(range(N)) or graph.number_of_nodes() != N:
        raise RuntimeError("invalid community node labels")
    if graph.number_of_edges() != target_edges:
        raise RuntimeError("invalid community edge count")
    if graph.is_multigraph() or nx.number_of_selfloops(graph) or not nx.is_connected(graph):
        raise RuntimeError("community graph must be connected and simple")
    if not np.isclose(2.0 * graph.number_of_edges() / N, target_degree):
        raise RuntimeError("invalid community mean degree")
    if any(data.get("weight") != 1.0 for _, _, data in graph.edges(data=True)):
        raise RuntimeError("internal graph edge weights must all equal one")
    if abs(actual_aspl - target) > ASPL_TOLERANCE[target]:
        raise RuntimeError("community ASPL is outside its tolerance")


def generate_graph(target: float, network_id: int, community: int) -> tuple[nx.Graph, dict[str, Any]]:
    target_degree = MEAN_DEGREE_BY_ASPL[target]
    target_edges = int(round(target_degree * N / 2))
    if target == 1.0:
        graph = nx.complete_graph(N)
        nx.set_edge_attributes(graph, 1.0, "weight")
        validate_graph(graph, target, 1.0)
        return graph, {
            "target_ASPL": target, "ASPL": 1.0,
            "target_mean_degree": target_degree, "mean_degree": float(N - 1),
            "degree_variance": 0.0, "m": target_edges,
            "generation_attempts": 1,
            "graph_seed": stable_seed("complete_graph", N),
            "graph_type": "complete",
        }
    extra_edges = target_edges - (N - 1)
    graph_seed = stable_seed("graph", target, network_id, community)
    rng = random.Random(graph_seed)
    last_aspl = None
    for attempt in range(1, MAX_GRAPH_ATTEMPTS + 1):
        tree_seed = rng.randrange(1 << 30)
        graph = nx.random_labeled_tree(N, seed=tree_seed)
        graph.add_edges_from(rng.sample(list(nx.non_edges(graph)), extra_edges))
        actual_aspl = exact_aspl(graph)
        last_aspl = actual_aspl
        if abs(actual_aspl - target) <= ASPL_TOLERANCE[target]:
            nx.set_edge_attributes(graph, 1.0, "weight")
            validate_graph(graph, target, actual_aspl)
            degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
            return graph, {
                "target_ASPL": target,
                "ASPL": actual_aspl,
                "target_mean_degree": target_degree,
                "mean_degree": float(degrees.mean()),
                "degree_variance": float(degrees.var()),
                "m": graph.number_of_edges(),
                "generation_attempts": attempt,
                "graph_seed": graph_seed,
                "tree_seed": tree_seed,
                "extra_edges_added": extra_edges,
            }
    raise RuntimeError(f"could not generate ASPL={target:g}; last actual ASPL={last_aspl}")


def store_graph(path: Path, graph: nx.Graph, metadata: dict[str, Any]) -> None:
    payload = {"experiment": EXPERIMENT, "metadata": metadata,
               "edges": [[int(u), int(v), 1.0] for u, v in sorted(graph.edges())]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temporary, path)


def load_graph(path: Path, target: float) -> tuple[nx.Graph, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("experiment") != EXPERIMENT:
        raise RuntimeError(f"incompatible graph file: {path}")
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_weighted_edges_from(payload["edges"])
    metadata = payload["metadata"]
    validate_graph(graph, target, float(metadata["ASPL"]))
    return graph, metadata


def prepare_topology() -> dict[str, Any]:
    """Generate or reuse all frozen community graphs, then write a PASS manifest."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for target in ASPL_TARGETS:
        networks = []
        for network_id in range(NETWORKS):
            communities = []
            for community in range(L):
                path = graph_path(target, network_id, community)
                if path.exists():
                    _, metadata = load_graph(path, target)
                    source = "reused"
                else:
                    print(f"[topology ASPL={target:g}] network {network_id + 1}/{NETWORKS}, community {community + 1}/{L}", flush=True)
                    graph, metadata = generate_graph(target, network_id, community)
                    store_graph(path, graph, metadata)
                    _, metadata = load_graph(path, target)
                    source = "new"
                communities.append({"community": community, "graph_path": str(path.relative_to(ROOT)),
                                    "source": source, **metadata})
            networks.append({"network_id": network_id, "communities": communities})
        cells[str(target)] = networks
    manifest = {
        "experiment": EXPERIMENT,
        "topology": "chain with tree-augmented random-connected communities",
        "topology_gate": "PASS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cells": cells,
    }
    atomic_json(DATA_ROOT / "topology_manifest.json", manifest)
    return manifest


def interface_edges(target: float, network_id: int, replicate_id: int) -> tuple[list[list[tuple[int, int]]], list[int]]:
    """Return five nested candidate edges for every one of the nine boundaries."""
    boundaries: list[list[tuple[int, int]]] = []
    seeds = []
    for boundary in range(L - 1):
        interface_seed = stable_seed("interface", target, network_id, replicate_id, boundary)
        rng = random.Random(interface_seed)
        left = list(range(N)); right = list(range(N))
        rng.shuffle(left); rng.shuffle(right)
        boundaries.append([(boundary * N + left[i], (boundary + 1) * N + right[i]) for i in range(max(K_VALUES))])
        seeds.append(interface_seed)
    return boundaries, seeds


def event_seeds(target: float, network_id: int, replicate_id: int) -> dict[str, int]:
    return {
        "full_event_seed": stable_seed("event", "full", target, network_id, replicate_id),
        "micromacro_event_seed": stable_seed("event", "micromacro", target, network_id, replicate_id),
    }


def scenario_name(target: float, k: int) -> str:
    degree = MEAN_DEGREE_BY_ASPL[target]
    return (
        f"macro_chain__L{L}__n{N}__tree_augmented__d{token(degree)}__aspl{token(target)}"
        f"__k{k}__beta{token(BETA)}__tau{token(TAU_MICRO)}__independent_rng"
    )


def scenario_parameters(target: float, k: int) -> dict[str, Any]:
    return {
        "macro_topology": "chain",
        "communities": L,
        "community_size": N,
        "topology": "random_connected_tree_augmented",
        "target_ASPL": target,
        "target_mean_degree": MEAN_DEGREE_BY_ASPL[target],
        "ASPL_tolerance": ASPL_TOLERANCE[target],
        "k": k,
        "beta_in": BETA,
        "beta_out": BETA,
        "gamma": GAMMA,
        "model": "SI",
        "tau_micro": TAU_MICRO,
        "macro_T": MACRO_T,
        "T_max": T_MAX,
        "initial_node": 0,
        "paired_runs": REPLICATIONS,
        "replication_design": "20 chain networks x 5 interface/event replicates",
        "independent_rng_per_micro_model": True,
        "censoring_policy": "unreached community has null arrival time",
    }


def scenario_folder(task: dict[str, Any]) -> Path:
    return DATA_ROOT / "scenarios" / scenario_name(float(task["target_ASPL"]), int(task["k"]))


def checkpoint_path(task: dict[str, Any]) -> Path:
    return scenario_folder(task) / "checkpoints" / f"network{task['network_id']:03d}_replicate{task['replicate_id']:02d}.json.gz"


def tasks(topology: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    if topology.get("topology_gate") != "PASS":
        raise RuntimeError("topology gate failed")
    result = []
    targets = (ASPL_TARGETS[0], ASPL_TARGETS[-1]) if smoke else ASPL_TARGETS
    for target in targets:
        networks = topology["cells"][str(target)]
        selected_networks = networks[:1] if smoke else networks
        selected_replicates = range(1) if smoke else range(REPLICATES_PER_NETWORK)
        for network in selected_networks:
            graph_paths = [row["graph_path"] for row in network["communities"]]
            actual_aspl = [row["ASPL"] for row in network["communities"]]
            graph_seeds = [row["graph_seed"] for row in network["communities"]]
            for replicate_id in selected_replicates:
                boundaries, interface_seeds = interface_edges(target, network["network_id"], replicate_id)
                initial_node = (random.Random(stable_seed("initial_node", target, network["network_id"], replicate_id)).randrange(N)
                                if RANDOM_INITIAL_NODE else 0)
                for k in K_VALUES:
                    # JSON stores each edge as a list. Keep the in-memory task in
                    # that same form so saved checkpoints compare equal on resume.
                    edges = [list(edge) for boundary in boundaries for edge in boundary[:k]]
                    task = {
                        "target_ASPL": target,
                        "network_id": network["network_id"],
                        "replicate_id": replicate_id,
                        "k": k,
                        "graph_paths": graph_paths,
                        "actual_ASPL": actual_aspl,
                        "graph_seeds": graph_seeds,
                        "interface_edges": edges,
                        "interface_seeds": interface_seeds,
                    }
                    if RANDOM_INITIAL_NODE:
                        task["initial_node"] = initial_node
                    result.append(task)
    return result


def build_chain_network(task: dict[str, Any]) -> tuple[list[nx.Graph], nx.Graph, np.ndarray, list[list[int]]]:
    micro_graphs = []
    comm_nodes = []
    full_graph = nx.Graph()
    for community, relative_path in enumerate(task["graph_paths"]):
        local, _ = load_graph(ROOT / relative_path, float(task["target_ASPL"]))
        mapping = {node: community * N + node for node in local}
        graph = nx.relabel_nodes(local, mapping, copy=True)
        nodes = list(range(community * N, (community + 1) * N))
        micro_graphs.append(graph)
        comm_nodes.append(nodes)
        full_graph.add_nodes_from(nodes)
        full_graph.add_edges_from(graph.edges(data=True))
    full_graph.add_edges_from((u, v, {"weight": 1.0}) for u, v in task["interface_edges"])
    expected_inter_edges = (L - 1) * int(task["k"])
    if len(task["interface_edges"]) != expected_inter_edges or len(set(map(tuple, task["interface_edges"]))) != expected_inter_edges:
        raise RuntimeError("invalid chain interface")
    W = np.zeros((L, L), dtype=float)
    for community in range(L - 1):
        W[community, community + 1] = task["k"]
        W[community + 1, community] = task["k"]
    boundary_counts = [0] * L
    for left, right in task["interface_edges"]:
        boundary_counts[left // N] += 1
        boundary_counts[right // N] += 1
    full_graph.graph.update({
        "community_sizes": [N] * L,
        "alphas": [1.0] * L,
        "boundary_counts": boundary_counts,
        "mu2_values": [1.0] * L,
        "spectral_descriptors_computed": False,
    })
    return micro_graphs, full_graph, W, comm_nodes


def benchmark(task: dict[str, Any]) -> Benchmark:
    return Benchmark(
        name=scenario_name(float(task["target_ASPL"]), int(task["k"])),
        family=EXPERIMENT,
        plot_groups=(),
        replications=1,
        base_seed=BASE_SEED,
        network_seed_base=BASE_SEED,
        communities=L,
        community_size=N,
        inter_links=int(task["k"]),
        macro_graph_type="chain",
        micro_graph_type="controlled",
        edge_prob=0.0,
        beta=BETA,
        gamma=GAMMA,
        model=MODEL,
        tau_micro=TAU_MICRO,
        macro_T=MACRO_T,
        original_T_end=T_MAX,
        T_end=T_MAX,
        dt_out=T_MAX,
        initial_node=int(task.get("initial_node", 0)),
        store_trajectories=False,
    )


def full_arrivals(result: dict[str, Any]) -> list[float | None]:
    values: list[float | None] = [None] * L
    for row in result.get("community_timings", []):
        community = int(row["community"])
        value = row.get("first_infection_time")
        if value is not None and float(value) <= T_MAX:
            values[community] = float(value)
    values[0] = 0.0
    return values


def mm_arrivals(events: list[dict[str, Any]]) -> list[float | None]:
    values: list[float | None] = [None] * L
    for event in events:
        if event.get("event_type") not in {"seed", "infection", "transfer"}:
            continue
        community = event.get("community")
        event_time = float(event.get("time", float("inf")))
        if community is None or event_time > T_MAX:
            continue
        community = int(community)
        if 0 <= community < L and (values[community] is None or event_time < values[community]):
            values[community] = event_time
    values[0] = 0.0
    return values


def arrival_summary(arrivals: list[float | None]) -> dict[str, Any]:
    completion = all(value is not None for value in arrivals)
    deltas = [None]
    for community in range(1, L):
        previous, current = arrivals[community - 1], arrivals[community]
        deltas.append(current - previous if previous is not None and current is not None else None)
    return {"arrivals": arrivals, "deltas": deltas,
            "T_last": max(arrivals) if completion else None, "completion": completion}


def compact_full_events(result: dict[str, Any]) -> list[list[float | int]]:
    rows = []
    for event in result.get("transmission_events", []):
        if event.get("kind") not in {"seed", "intra", "inter"} or event.get("dst_node") is None:
            continue
        rows.append([float(event["time"]), int(event["dst_community"]), int(event["dst_node"])])
    return rows


def compact_mm_events(events: list[dict[str, Any]]) -> list[list[float | int]]:
    rows = []
    for event in events:
        if event.get("event_type") not in {"seed", "infection", "transfer"} or event.get("node") is None:
            continue
        rows.append([float(event["time"]), int(event["community"]), int(event["node"])])
    return rows


def validate_full_interfaces(result: dict[str, Any], full_graph: nx.Graph) -> bool:
    return all(
        event.get("kind") != "inter" or full_graph.has_edge(event.get("src_node"), event.get("dst_node"))
        for event in result.get("transmission_events", [])
    )


def run_pair(task: dict[str, Any]) -> dict[str, Any]:
    micro_graphs, full_graph, W, comm_nodes = build_chain_network(task)
    seeds = event_seeds(task["target_ASPL"], task["network_id"], task["replicate_id"])

    started = time.perf_counter()
    full_result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=BETA,
        recovery_rate=GAMMA,
        model=MODEL,
    ).run(
        T_end=T_MAX,
        dt_out=T_MAX,
        initial_node=int(task.get("initial_node", 0)),
        rng=random.Random(seeds["full_event_seed"]),
        stop_at_absorption=True,
    )
    full_runtime = time.perf_counter() - started
    full_events = compact_full_events(full_result)
    full_run = {
        "simulator": FULL,
        "runtime_seconds": full_runtime,
        "termination": full_result.get("run_metadata"),
        "physical_interface_valid": validate_full_interfaces(full_result, full_graph),
        "infection_events": full_events,
        **arrival_summary(full_arrivals(full_result)),
    }
    if not full_run["physical_interface_valid"]:
        raise RuntimeError("Full generated an inter-community infection outside the physical interface")
    del full_result
    gc.collect()

    started = time.perf_counter()
    mm_result, mm_metadata = _run_experimental_micromacro(
        benchmark(task), micro_graphs, full_graph, W, seeds["micromacro_event_seed"],
        independent=True, dense_logs=False, stop_at_absorption=True, return_metadata=True,
    )
    mm_runtime = time.perf_counter() - started
    mm_events = compact_mm_events(mm_result[3])
    mm_run = {
        "simulator": MICROMACRO,
        "runtime_seconds": mm_runtime,
        "termination": mm_metadata,
        "infection_events": mm_events,
        **arrival_summary(mm_arrivals(mm_result[3])),
    }
    del mm_result, micro_graphs, full_graph, W, comm_nodes
    gc.collect()
    return {
        "experiment": EXPERIMENT,
        "task": task,
        "T_max": T_MAX,
        "event_seeds": seeds,
        "rng_policy": {
            "independent_rng_per_MicroModel": True,
            "cache_policy": "discard",
            "reimport_policy": "normal",
            "clone_copies_community_rng_state": True,
            "refresh_synchronizes_clone_rng_state": True,
        },
        "runs": [full_run, mm_run],
    }


def valid_checkpoint(task: dict[str, Any]) -> bool:
    try:
        payload = read_checkpoint(checkpoint_path(task))
        runs = payload.get("runs", [])
        names = {row.get("simulator") for row in runs}
        return (
            payload.get("experiment") == EXPERIMENT
            and payload.get("task") == task
            and payload.get("T_max") == T_MAX
            and names == {FULL, MICROMACRO}
            and len(runs) == 2
            and all(len(row.get("arrivals", [])) == L and "infection_events" in row for row in runs)
            and payload.get("rng_policy", {}).get("independent_rng_per_MicroModel") is True
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def write_run_manifests(all_tasks: list[dict[str, Any]], smoke: bool) -> None:
    grouped: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for task in all_tasks:
        grouped.setdefault((float(task["target_ASPL"]), int(task["k"])), []).append(task)
    scenario_rows = []
    for (target, k), scenario_tasks in sorted(grouped.items()):
        folder = DATA_ROOT / "scenarios" / scenario_name(target, k)
        params_path = folder / "parameters.json"
        params = scenario_parameters(target, k)
        if params_path.exists() and json.loads(params_path.read_text(encoding="utf-8")) != params:
            raise RuntimeError(f"existing scenario parameters differ: {params_path}")
        if not params_path.exists():
            atomic_json(params_path, params)
        completed = sum(valid_checkpoint(task) for task in scenario_tasks)
        row = {
            "scenario": scenario_name(target, k),
            "mode": "smoke" if smoke else "full",
            "requested_pairs_in_this_run": len(scenario_tasks),
            "completed_pairs_in_this_run": completed,
            "full_design_pairs": REPLICATIONS,
            "status": "smoke_complete" if smoke and completed == len(scenario_tasks)
                      else "complete" if not smoke and completed == len(scenario_tasks)
                      else "incomplete",
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(folder / "run_manifest.json", row)
        scenario_rows.append(row)
    atomic_json(DATA_ROOT / "run_manifest.json", {
        "experiment": EXPERIMENT,
        "mode": "smoke" if smoke else "full",
        "scenarios": scenario_rows,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def run_pending(pending: list[dict[str, Any]], workers: int) -> None:
    if not pending:
        return
    done, total = 0, len(pending)
    iterator = iter(pending)
    active: dict[concurrent.futures.Future, dict[str, Any]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            while len(active) < 2 * workers:
                try:
                    task = next(iterator)
                except StopIteration:
                    break
                active[pool.submit(run_pair, task)] = task
            if not active:
                break
            future = next(concurrent.futures.as_completed(active))
            task = active.pop(future)
            write_checkpoint(checkpoint_path(task), future.result())
            done += 1
            print(
                f"[progress {done}/{total}] ASPL={task['target_ASPL']:g} k={task['k']} "
                f"network={task['network_id']} replicate={task['replicate_id']}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-regression-gate", action="store_true", help="debug only")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    ensure_parameters()
    topology = prepare_topology()
    graph_count = len(ASPL_TARGETS) * NETWORKS * L
    print(f"[topology] PASS; valid graphs={graph_count}; manifest={DATA_ROOT / 'topology_manifest.json'}", flush=True)
    if args.topology_only:
        return
    if not args.skip_regression_gate:
        passed, message = _regression_gate()
        print(f"Regression gate: {'PASS' if passed else 'FAIL'} — {message}", flush=True)
        if not passed:
            raise SystemExit(2)
    all_tasks = tasks(topology, args.smoke_only)
    pending = [task for task in all_tasks if not valid_checkpoint(task)]
    write_run_manifests(all_tasks, args.smoke_only)
    print(f"[queued] mode={'smoke' if args.smoke_only else 'full'} reused={len(all_tasks) - len(pending)}/{len(all_tasks)} new={len(pending)}", flush=True)
    run_pending(pending, args.workers)
    write_run_manifests(all_tasks, args.smoke_only)


if __name__ == "__main__":
    main()
