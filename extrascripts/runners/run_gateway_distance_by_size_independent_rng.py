#!/usr/bin/env python3
"""Test community-size effects at a fixed gateway-to-gateway distance.

The experiment holds L, d, k, beta and the absolute mean shortest-path
distance between incoming and outgoing gateway sets fixed while varying n.
Full Micro is compared with basic Production MicroMacro using one private RNG
stream per MicroModel.  Production source files are not modified.

The default command requests 100 paired replications for every size and
automatically resumes valid checkpoints::

    .venv/bin/python3 extrascripts/runners/run_gateway_distance_by_size_independent_rng.py

Use ``--smoke-only`` for the first three deterministic replications per size.
Those checkpoints are reused by the later default run.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extrascripts.runners.checkpoint_io import read_checkpoint, write_checkpoint
from extrascripts.runners.run_original_benchmarks_independent_rng import (
    Benchmark,
    FULL,
    MICROMACRO,
    _arrivals_from_events,
    _arrivals_from_full,
    _run_experimental_micromacro,
    _without_unused_spectra,
)
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulators.micro import MicroSimulator


# --------------------------- editable experiment grid ---------------------------

COMMUNITY_SIZES = (50, 100, 250, 1000)
REPLICATIONS = 100
L = 5
MEAN_INTERNAL_DEGREE = 6.0
INTER_LINKS = 5
BETA = 0.003
GAMMA = 0.0
TAU_MICRO = 1.0
MACRO_T = 1.0
T_END = 20_000.0
TARGET_D_MEAN = 2.72       # 68/25; feasible for every requested community size
DISTANCE_TOLERANCE = 0.021 # accepts the nearest 1/25 distance increment
CANDIDATES_PER_LEFT_SET = 2_000
LEFT_SET_ATTEMPTS = 12
BASE_SEED = 1_420_000

# Two pools run concurrently.  Limiting n=1000 tasks protects WSL memory.
WORKERS = 6
HEAVY_WORKERS = 2
HEAVY_SIZE_THRESHOLD = 1000

OUTPUT_ROOT = REPO_ROOT / "output" / "data"


@dataclass(frozen=True)
class Config:
    community_size: int
    requested_replications: int
    communities: int = L
    mean_internal_degree: float = MEAN_INTERNAL_DEGREE
    inter_links: int = INTER_LINKS
    beta: float = BETA
    gamma: float = GAMMA
    tau_micro: float = TAU_MICRO
    macro_T: float = MACRO_T
    T_end: float = T_END
    target_D_mean: float = TARGET_D_MEAN
    distance_tolerance: float = DISTANCE_TOLERANCE
    candidates_per_left_set: int = CANDIDATES_PER_LEFT_SET
    left_set_attempts: int = LEFT_SET_ATTEMPTS
    base_seed: int = BASE_SEED

    @property
    def edge_prob(self) -> float:
        return self.mean_internal_degree / (self.community_size - 1)

    @property
    def scenario(self) -> str:
        return (
            f"gateway_distance_size_L{self.communities}_n{self.community_size}"
            f"_d{number_token(self.mean_internal_degree)}_k{self.inter_links}"
            f"_beta{number_token(self.beta)}_Dmean{number_token(self.target_D_mean)}"
            "_independent_rng"
        )


def number_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def scientific_config(cfg: Config) -> dict[str, Any]:
    """Fingerprint fields; requested replication count is deliberately excluded."""
    data = asdict(cfg)
    data.pop("requested_replications")
    return data


def fingerprint(cfg: Config) -> str:
    encoded = json.dumps(scientific_config(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def seed_map(cfg: Config, replication_id: int) -> dict[str, int]:
    size_offset = cfg.community_size * 10_000
    return {
        "network_seed": cfg.base_seed + size_offset + replication_id,
        "gateway_seed": cfg.base_seed + 20_000_000 + size_offset + replication_id,
        "event_seed": cfg.base_seed + 40_000_000 + size_offset + replication_id,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def degree_matched_target_sets(
    graph: nx.Graph,
    nodes: Sequence[int],
    k: int,
    target: float,
    tolerance: float,
    seed: int,
    left_attempts: int,
    candidates_per_left: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Select degree-matched gateway sets with mean pair distance near target."""
    rng = random.Random(seed)
    nodes = list(nodes)
    best: tuple[float, int, list[int], list[int], float, float] | None = None
    for left_attempt in range(1, left_attempts + 1):
        # A globally uniform k-set becomes increasingly dispersed as n grows,
        # which can make a small fixed absolute separation geometrically
        # impossible.  Select the incoming interface from one local radius-2
        # neighborhood, then target the outgoing set by its mean distance to
        # that fixed interface.  This is the intended size-control design.
        anchor = rng.choice(nodes)
        local = list(nx.single_source_shortest_path_length(
            graph, anchor, cutoff=2).keys())
        if len(local) < k:
            continue
        left = sorted([anchor] + rng.sample([node for node in local if node != anchor], k - 1))
        degree_counts = Counter(int(graph.degree(node)) for node in left)
        distances = {
            source: nx.single_source_shortest_path_length(graph, source)
            for source in left
        }
        mean_distance = {
            node: float(np.mean([distances[source][node] for source in left]))
            for node in nodes
        }
        pools: dict[int, list[int]] = defaultdict(list)
        for node in nodes:
            if node not in left:
                pools[int(graph.degree(node))].append(node)
        if any(len(pools[degree]) < count for degree, count in degree_counts.items()):
            continue
        # Concentrate random search on degree-compatible nodes whose individual
        # mean distance is near the requested set-level mean.  Keep enough
        # alternatives to solve repeated-degree groups without overlap.
        for degree, count in degree_counts.items():
            ranked = sorted(pools[degree], key=lambda node: (
                abs(mean_distance[node] - target), rng.random()))
            pools[degree] = ranked[:max(24, 8 * count)]
        for candidate_index in range(1, candidates_per_left + 1):
            right: list[int] = []
            for degree, count in sorted(degree_counts.items()):
                right.extend(rng.sample(pools[degree], count))
            right.sort()
            pairwise = [distances[source][destination] for source in left for destination in right]
            d_mean = float(np.mean(pairwise))
            d_min = float(min(pairwise))
            error = abs(d_mean - target)
            overlap = len(set(left) & set(right))
            candidate = (error, overlap, left, right, d_min, d_mean)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
            if error <= tolerance and overlap == 0:
                return left, right, {
                    "D_min": d_min,
                    "D_mean": d_mean,
                    "target_error": error,
                    "overlap": overlap,
                    "left_attempt": left_attempt,
                    "candidate_index": candidate_index,
                    "left_within_degrees": sorted(int(graph.degree(node)) for node in left),
                    "right_within_degrees": sorted(int(graph.degree(node)) for node in right),
                }
    if best is None:
        raise RuntimeError("could not construct degree-matched gateway sets")
    error, overlap, left, right, d_min, d_mean = best
    raise RuntimeError(
        f"target D_mean={target:g} not reached: best={d_mean:g}, "
        f"error={error:g}, overlap={overlap}"
    )


def build_network(cfg: Config, seeds: dict[str, int]):
    with _without_unused_spectra():
        micro_graphs, full_graph, _ = generate_two_scale_network(
            n_communities=cfg.communities,
            community_size=cfg.community_size,
            inter_links=0,
            seed=seeds["network_seed"],
            macro_graph_type="chain",
            micro_graph_type="random_connected",
            edge_prob=cfg.edge_prob,
            share_topology=True,
        )
    community_nodes = [list(graph.nodes()) for graph in micro_graphs]
    incoming: list[list[int]] = [[] for _ in range(cfg.communities)]
    outgoing: list[list[int]] = [[] for _ in range(cfg.communities)]
    descriptors: list[dict[str, Any]] = []

    outgoing[0] = sorted(random.Random(seeds["gateway_seed"] + 1).sample(
        community_nodes[0], cfg.inter_links))
    incoming[-1] = sorted(random.Random(seeds["gateway_seed"] + 2).sample(
        community_nodes[-1], cfg.inter_links))
    for community in range(1, cfg.communities - 1):
        left, right, descriptor = degree_matched_target_sets(
            micro_graphs[community], community_nodes[community], cfg.inter_links,
            cfg.target_D_mean, cfg.distance_tolerance,
            seeds["gateway_seed"] + 1000 * community,
            cfg.left_set_attempts, cfg.candidates_per_left_set,
        )
        incoming[community] = left
        outgoing[community] = right
        descriptors.append({"community": community, **descriptor})

    W = np.zeros((cfg.communities, cfg.communities), dtype=float)
    for community in range(cfg.communities - 1):
        destinations = list(incoming[community + 1])
        random.Random(seeds["gateway_seed"] + 10_000 + community).shuffle(destinations)
        for source, destination in zip(outgoing[community], destinations, strict=True):
            full_graph.add_edge(source, destination, weight=1.0)
        W[community, community + 1] = W[community + 1, community] = cfg.inter_links

    boundary = [set(incoming[i]) | set(outgoing[i]) for i in range(cfg.communities)]
    full_graph.graph.update({
        "community_sizes": [cfg.community_size] * cfg.communities,
        "alphas": [1.0] * cfg.communities,
        "boundary_counts": [len(nodes) for nodes in boundary],
        "mu2_values": [1.0] * cfg.communities,
        "spectral_descriptors_computed": False,
    })
    descriptor = {
        "target_D_mean": cfg.target_D_mean,
        "actual_D_mean": float(np.mean([item["D_mean"] for item in descriptors])),
        "actual_D_min": float(np.mean([item["D_min"] for item in descriptors])),
        "max_absolute_target_error": float(max(item["target_error"] for item in descriptors)),
        "degree_matched": all(
            item["left_within_degrees"] == item["right_within_degrees"]
            for item in descriptors
        ),
        "gateway_overlap": int(sum(item["overlap"] for item in descriptors)),
        "by_community": descriptors,
    }
    return micro_graphs, full_graph, W, community_nodes, descriptor


def benchmark_for(cfg: Config) -> Benchmark:
    return Benchmark(
        name=cfg.scenario,
        family="gateway_distance_size_control",
        plot_groups=("gateway_distance_size_control",),
        replications=cfg.requested_replications,
        base_seed=cfg.base_seed,
        network_seed_base=cfg.base_seed,
        communities=cfg.communities,
        community_size=cfg.community_size,
        inter_links=cfg.inter_links,
        macro_graph_type="chain",
        micro_graph_type="random_connected",
        edge_prob=cfg.edge_prob,
        beta=cfg.beta,
        gamma=cfg.gamma,
        model=2,
        tau_micro=cfg.tau_micro,
        macro_T=cfg.macro_T,
        original_T_end=cfg.T_end,
        T_end=cfg.T_end,
        dt_out=cfg.T_end,
        initial_node=0,
        mean_internal_degree=cfg.mean_internal_degree,
        store_trajectories=False,
    )


def run_replication(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    cfg_dict, replication_id = task
    cfg = Config(**cfg_dict)
    seeds = seed_map(cfg, replication_id)
    micro_graphs, full_graph, W, community_nodes, descriptor = build_network(cfg, seeds)
    benchmark = benchmark_for(cfg)

    started = time.perf_counter()
    full_result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=community_nodes,
        infection_rate=cfg.beta,
        recovery_rate=cfg.gamma,
        model=2,
    ).run(
        T_end=cfg.T_end,
        dt_out=cfg.T_end,
        initial_node=0,
        rng=random.Random(seeds["event_seed"]),
        stop_at_absorption=True,
    )
    full_runtime = time.perf_counter() - started
    full_arrivals = _arrivals_from_full(full_result, cfg.communities, cfg.T_end)
    full_termination = full_result.get("run_metadata")
    del full_result
    gc.collect()

    started = time.perf_counter()
    mm_result, mm_termination = _run_experimental_micromacro(
        benchmark, list(micro_graphs), full_graph, W, seeds["event_seed"],
        independent=True,
        dense_logs=False,
        stop_at_absorption=True,
        return_metadata=True,
    )
    mm_runtime = time.perf_counter() - started
    mm_arrivals = _arrivals_from_events(mm_result[3], cfg.communities, cfg.T_end)
    del mm_result, micro_graphs, full_graph, W, community_nodes
    gc.collect()

    def run_row(simulator: str, arrivals: Sequence[float], runtime: float, termination: dict | None):
        t_last = max(arrivals) if all(math.isfinite(value) for value in arrivals) else None
        return {
            "simulator": simulator,
            "network_seed": seeds["network_seed"],
            "gateway_seed": seeds["gateway_seed"],
            "event_seed": seeds["event_seed"],
            "initial_node": 0,
            "completion": t_last is not None,
            "T_last": t_last,
            "runtime_seconds": runtime,
            "T_end": cfg.T_end,
            "stop_reason": None if termination is None else termination.get("stop_reason"),
            "stop_time": None if termination is None else termination.get("stop_time"),
        }

    arrivals_rows = []
    for simulator, arrivals in ((FULL, full_arrivals), (MICROMACRO, mm_arrivals)):
        arrivals_rows.extend({
            "simulator": simulator,
            "community": community,
            "first_infection_time": value if math.isfinite(value) else None,
        } for community, value in enumerate(arrivals))

    return {
        "format_version": 1,
        "experiment": "gateway_distance_size_control_independent_rng_v1",
        "config_fingerprint": fingerprint(cfg),
        "scenario": cfg.scenario,
        "config": scientific_config(cfg),
        "replication_id": replication_id,
        "seeds": seeds,
        "gateway_descriptor": descriptor,
        "rng_policy": {
            "independent_rng_per_MicroModel": True,
            "community_seed_formula": "event_seed + 10000 + 1009 * community",
            "clone_copies_community_rng_state": True,
            "refresh_synchronizes_clone_rng_state": True,
        },
        "runs": [
            run_row(FULL, full_arrivals, full_runtime, full_termination),
            run_row(MICROMACRO, mm_arrivals, mm_runtime, mm_termination),
        ],
        "community_arrivals": arrivals_rows,
    }


def valid_checkpoint(path: Path, cfg: Config, replication_id: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_checkpoint(path)
        models = {row.get("simulator") for row in payload.get("runs", [])}
        descriptor = payload["gateway_descriptor"]
        return (
            payload.get("experiment") == "gateway_distance_size_control_independent_rng_v1"
            and payload.get("config_fingerprint") == fingerprint(cfg)
            and int(payload.get("replication_id")) == replication_id
            and models == {FULL, MICROMACRO}
            and payload.get("rng_policy", {}).get("independent_rng_per_MicroModel") is True
            and descriptor.get("degree_matched") is True
            and descriptor.get("gateway_overlap") == 0
            and abs(float(descriptor["actual_D_mean"]) - cfg.target_D_mean) <= cfg.distance_tolerance
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def checkpoint_path(folder: Path, replication_id: int) -> Path:
    return folder / "checkpoints" / f"rep_{replication_id:04d}.json.gz"


def manifest(cfg: Config, completed: Sequence[int], failures: dict[int, str]) -> dict[str, Any]:
    return {
        "format": "gateway_distance_size_control_v1",
        "experiment": "gateway_distance_size_control_independent_rng_v1",
        "scenario": cfg.scenario,
        "algorithm": "Production MicroMacro with independent RNG per MicroModel",
        "complete": len(completed) == cfg.requested_replications,
        "config_fingerprint": fingerprint(cfg),
        "config": asdict(cfg),
        "requested_replications": cfg.requested_replications,
        "completed_replications": list(sorted(completed)),
        "failed_replications": {str(key): value for key, value in sorted(failures.items())},
        "seed_policy": "deterministic per size and replication; saved in every checkpoint",
        "distance_policy": {
            "estimand": "mean of all k^2 shortest-path distances between incoming and outgoing gateway sets",
            "target": cfg.target_D_mean,
            "tolerance": cfg.distance_tolerance,
            "within_degree_multiset_matched": True,
            "gateway_overlap_allowed": False,
        },
        "storage_policy": "compact arrival-only gzip checkpoints; no trajectories or event logs",
        "resume_policy": "reuse only checkpoints with matching scientific fingerprint and RNG policy",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def write_summary(folder: Path, cfg: Config, completed: Sequence[int]) -> None:
    run_rows, arrival_rows, descriptor_rows = [], [], []
    for replication_id in sorted(completed):
        payload = read_checkpoint(checkpoint_path(folder, replication_id))
        descriptor_rows.append({
            "scenario": cfg.scenario,
            "replication_id": replication_id,
            **{key: payload["gateway_descriptor"][key]
               for key in ("target_D_mean", "actual_D_mean", "actual_D_min",
                           "max_absolute_target_error", "degree_matched", "gateway_overlap")},
        })
        run_rows.extend({"scenario": cfg.scenario, "community_size": cfg.community_size,
                         "replication_id": replication_id, **row}
                        for row in payload["runs"])
        arrival_rows.extend({"scenario": cfg.scenario, "community_size": cfg.community_size,
                             "replication_id": replication_id, **row}
                            for row in payload["community_arrivals"])

    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        temporary = folder / f"{name}.tmp.{os.getpid()}"
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, folder / name)

    write_csv("runs.csv", run_rows)
    write_csv("community_arrivals.csv", arrival_rows)
    write_csv("gateway_descriptors.csv", descriptor_rows)


def topology_check(cfg: Config, replication_id: int = 1) -> dict[str, Any]:
    seeds = seed_map(cfg, replication_id)
    graphs, full, _, _, descriptor = build_network(cfg, seeds)
    result = {
        "n": cfg.community_size,
        "actual_D_mean": descriptor["actual_D_mean"],
        "max_target_error": descriptor["max_absolute_target_error"],
        "degree_matched": descriptor["degree_matched"],
        "gateway_overlap": descriptor["gateway_overlap"],
        "connected": all(nx.is_connected(graph) for graph in graphs),
        "nodes": full.number_of_nodes(),
        "edges": full.number_of_edges(),
    }
    del graphs, full
    gc.collect()
    return result


def run(configs: Sequence[Config], workers: int, heavy_workers: int, topology_only: bool) -> None:
    if topology_only:
        for cfg in configs:
            print(json.dumps(topology_check(cfg), sort_keys=True), flush=True)
        return

    state: dict[int, dict[str, Any]] = {}
    tasks_light, tasks_heavy = [], []
    for cfg in configs:
        folder = OUTPUT_ROOT / cfg.scenario
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "checkpoints").mkdir(exist_ok=True)
        completed = [rep for rep in range(1, cfg.requested_replications + 1)
                     if valid_checkpoint(checkpoint_path(folder, rep), cfg, rep)]
        pending = [rep for rep in range(1, cfg.requested_replications + 1) if rep not in completed]
        state[cfg.community_size] = {
            "cfg": cfg, "folder": folder, "completed": completed, "failures": {},
        }
        atomic_json(folder / "manifest.json", manifest(cfg, completed, {}))
        print(f"[queued] {cfg.scenario}: reused={len(completed)}/{cfg.requested_replications}; "
              f"new={len(pending)}", flush=True)
        destination = tasks_heavy if cfg.community_size >= HEAVY_SIZE_THRESHOLD else tasks_light
        destination.extend((asdict(cfg), rep) for rep in pending)

    total_pending = len(tasks_light) + len(tasks_heavy)
    if total_pending == 0:
        print("[resume] all requested checkpoints already exist", flush=True)
    else:
        light_workers = max(1, workers - heavy_workers) if tasks_light else 0
        actual_heavy_workers = min(heavy_workers, len(tasks_heavy)) if tasks_heavy else 0
        print(f"[workers] light={light_workers}, heavy={actual_heavy_workers}; "
              f"pending={total_pending}", flush=True)
        done = 0
        with ExitStack() as stack:
            pools = []
            if tasks_light:
                pools.append((stack.enter_context(concurrent.futures.ProcessPoolExecutor(
                    max_workers=light_workers)), tasks_light))
            if tasks_heavy:
                pools.append((stack.enter_context(concurrent.futures.ProcessPoolExecutor(
                    max_workers=actual_heavy_workers)), tasks_heavy))
            futures = {}
            for pool, tasks in pools:
                for task in tasks:
                    future = pool.submit(run_replication, task)
                    futures[future] = (task[0]["community_size"], task[1])
            for future in concurrent.futures.as_completed(futures):
                n, replication_id = futures[future]
                item = state[n]
                try:
                    payload = future.result()
                    write_checkpoint(checkpoint_path(item["folder"], replication_id), payload)
                    item["completed"].append(replication_id)
                    item["failures"].pop(replication_id, None)
                    status = "ok"
                except Exception as exc:
                    item["failures"][replication_id] = f"{type(exc).__name__}: {exc}"
                    status = f"FAILED {item['failures'][replication_id]}"
                done += 1
                atomic_json(item["folder"] / "manifest.json",
                            manifest(item["cfg"], item["completed"], item["failures"]))
                remaining = total_pending - done
                print(f"[progress {done}/{total_pending}] n={n}, rep={replication_id}: {status}; "
                      f"remaining={remaining}", flush=True)

    incomplete = []
    for n in sorted(state):
        item = state[n]
        cfg = item["cfg"]
        completed = sorted(set(item["completed"]))
        write_summary(item["folder"], cfg, completed)
        atomic_json(item["folder"] / "manifest.json",
                    manifest(cfg, completed, item["failures"]))
        print(f"[scenario complete] {cfg.scenario}: "
              f"{len(completed)}/{cfg.requested_replications}", flush=True)
        if len(completed) != cfg.requested_replications:
            incomplete.append((cfg.scenario, len(completed), cfg.requested_replications))
    if incomplete:
        raise SystemExit(f"incomplete scenarios: {incomplete}; rerun the same command to resume")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=REPLICATIONS)
    parser.add_argument("--smoke-only", action="store_true",
                        help="request only replications 1..3; later full runs reuse them")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(COMMUNITY_SIZES))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--heavy-workers", type=int, default=HEAVY_WORKERS)
    parser.add_argument("--topology-only", action="store_true",
                        help="construct one network per size and validate target distance; run no simulators")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    runs = 3 if args.smoke_only else args.runs
    if runs < 1:
        raise ValueError("--runs must be positive")
    if args.workers < 1 or args.heavy_workers < 1 or args.heavy_workers > args.workers:
        raise ValueError("require 1 <= --heavy-workers <= --workers")
    unknown = sorted(set(args.sizes) - set(COMMUNITY_SIZES))
    if unknown:
        raise ValueError(f"sizes not present in COMMUNITY_SIZES: {unknown}")
    configs = [Config(community_size=n, requested_replications=runs) for n in args.sizes]
    run(configs, args.workers, args.heavy_workers, args.topology_only)


if __name__ == "__main__":
    main()
