#!/usr/bin/env python3
"""Tree-augmented random-connected n=1000 ASPL export experiment.

Full Gillespie is compared with Production MicroMacro using the established
independent-RNG intervention.  The source community exports to one singleton
sink; the observable is its first infection.  This file is the sole runner
for this experiment.

Safe stages:
  python extrascripts/runners/run_n1000_aspl_export_independent_rng.py --topology-only
  python extrascripts/runners/run_n1000_aspl_export_independent_rng.py --smoke-only
The default command is the resumable full collection; do not run it before
the topology-only and smoke gates pass.

Data from this design are stored under
``output/data/n1000_random_connected_tree_augmented_aspl_export``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import gzip
import json
import os
import random
import shutil
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
    Benchmark, FULL, MICROMACRO, _regression_gate, _run_experimental_micromacro,
)
from two_layer_ctmc.simulators.micro import MicroSimulator


# ----------------------------- user settings -----------------------------
N = 1000
K_VALUES = (1, 5)
BETA = 0.003
TAU_MICRO = 1.0
T_MAX = 30_000.0
WORKERS = 2
ASPL_TARGETS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
REPLICATIONS = 200

# Each ASPL cell uses 40 independent tree-augmented connected graphs and five
# interface placements per graph: 40 x 5 = 200 paired runs.
GAMMA, MODEL, MACRO_T, BASE_SEED = 0.0, 2, 1.0, 20_260_919
GRAPH_LINEAGES, EVENT_REPLICATES_PER_INTERFACE = 40, 1
INTERFACE_REPLICATES = REPLICATIONS // GRAPH_LINEAGES
RANDOM_INITIAL_NODE = False
SMOKE_INTERFACES = 2
MEAN_DEGREE_BY_ASPL = {2.0: 50, 2.5: 25, 3.0: 12, 3.5: 8, 4.0: 6, 4.5: 5, 5.0: 4.3}
ASPL_ABS_TOLERANCE = {2.0: 0.10, 2.5: 0.10, 3.0: 0.10, 3.5: 0.12, 4.0: 0.15, 4.5: 0.15, 5.0: 0.15}
MAX_GRAPH_ATTEMPTS = 100

EXPERIMENT = "n1000_random_connected_tree_augmented_aspl_export_rep200"
LEGACY_EXPERIMENT = "n1000_random_connected_tree_augmented_aspl_export"
OUTPUT_ROOT = ROOT / "output" / "data"
ANALYSIS_ROOT = ROOT / "output" / "analysis" / EXPERIMENT
LEGACY_ROOT = OUTPUT_ROOT / LEGACY_EXPERIMENT
LEGACY_ANALYSIS_ROOT = ROOT / "output" / "analysis" / LEGACY_EXPERIMENT

if REPLICATIONS % GRAPH_LINEAGES:
    raise ValueError("REPLICATIONS must be divisible by GRAPH_LINEAGES")


def seed(*parts: object) -> int:
    """Stable private seed derivation; this is not a folder or data label."""
    text = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return value


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def root_folder() -> Path:
    return OUTPUT_ROOT / EXPERIMENT


def experiment_parameters() -> dict[str, Any]:
    """Readable settings written to experiment_parameters.json."""
    return {
        "community_size": N,
        "k_values": list(K_VALUES),
        "beta": BETA,
        "gamma": GAMMA,
        "model": "SI",
        "tau_micro": TAU_MICRO,
        "macro_T": MACRO_T,
        "T_max": T_MAX,
        "ASPL_targets": list(ASPL_TARGETS),
        "ASPL_tolerance": {str(key): value for key, value in ASPL_ABS_TOLERANCE.items()},
        "replications_per_scenario": REPLICATIONS,
        "graph_lineages": GRAPH_LINEAGES,
        "interfaces_per_lineage": INTERFACE_REPLICATES,
        "event_replicates_per_interface": EVENT_REPLICATES_PER_INTERFACE,
        "topology": "random_connected_tree_augmented",
        "mean_degree_by_target_ASPL": {
            str(target): MEAN_DEGREE_BY_ASPL[target] for target in ASPL_TARGETS
        },
        "design_note": "ASPL is varied through graph density, so mean degree changes with ASPL; a random tree guarantees connectedness",
        "independent_rng_per_micro_model": True,
        "censoring_policy": "T_max is only an upper observation bound and is never recorded as an export time",
        "output_folder": f"output/data/{EXPERIMENT}",
    }


def ensure_fixed_json(path: Path, value: dict[str, Any], label: str) -> None:
    """Write readable parameters once; refuse ambiguous mixed configurations."""
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise RuntimeError(
                f"{label} does not match the settings at the top of this runner. "
                f"Remove {path.parent} only if you intentionally want to replace that experiment."
            )
        return
    atomic_json(path, value)


def scenario_description(task: dict[str, Any]) -> dict[str, Any]:
    target_aspl = float(task["target_ASPL"])
    degree = MEAN_DEGREE_BY_ASPL[target_aspl]
    return {
        "topology": "random_connected_tree_augmented",
        "community_size": N,
        "target_ASPL": target_aspl,
        "actual_ASPL_location": f"output/analysis/{EXPERIMENT}/topology_calibration_manifest.json",
        "mean_degree": degree,
        "edge_prob": degree / (N - 1),
        "internal_edges": int(round(degree * N / 2)),
        "intercommunity_edges_k": int(task["k"]),
        "beta_in": BETA,
        "beta_out": BETA,
        "gamma": GAMMA,
        "model": "SI",
        "tau_micro": TAU_MICRO,
        "macro_T": MACRO_T,
        "T_max": T_MAX,
        "initial_node": 0,
        "paired_replications": REPLICATIONS,
        "replication_design": "40 independent graphs x 5 interface placements x 1 paired event replication",
        "interpretation": "ASPL and mean degree vary together in this tree-augmented random-connected sweep",
        "simulators": [FULL, "Production MicroMacro + independent RNG"],
        "independent_rng_per_micro_model": True,
        "base_seed": BASE_SEED,
        "censoring_policy": "unfinished export has null T_export; T_max is not an arrival time",
    }


def scenario_name(task: dict[str, Any]) -> str:
    info = scenario_description(task)
    return (
        f"topology_random_connected_tree_augmented__n{N}__d{token(info['mean_degree'])}__aspl{token(info['target_ASPL'])}"
        f"__k{task['k']}__beta{token(BETA)}__tau{token(TAU_MICRO)}__independent_rng"
    )


def scenario_folder(task: dict[str, Any]) -> Path:
    return root_folder() / "scenarios" / scenario_name(task)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def edge_rows(graph: nx.Graph) -> list[list[float]]:
    return [[int(u), int(v), float(data.get("weight", 1.0))]
            for u, v, data in sorted(graph.edges(data=True))]


def exact_aspl(graph: nx.Graph) -> float:
    if graph.number_of_edges() == N * (N - 1) // 2:
        return 1.0
    # Verify diameter two with bitsets before falling back to all-source BFS.
    # This makes the dense low-ASPL calibration cells much cheaper to check.
    all_nodes = (1 << N) - 1
    adjacency = [sum(1 << int(v) for v in graph.neighbors(node)) for node in range(N)]
    diameter_two = True
    for neighbors in adjacency:
        within_two = neighbors
        remaining = neighbors
        while remaining and within_two != all_nodes:
            bit = remaining & -remaining
            within_two |= adjacency[bit.bit_length() - 1]
            remaining ^= bit
        if within_two != all_nodes:
            diameter_two = False
            break
    if diameter_two:
        pairs = N * (N - 1) / 2
        return 2.0 - graph.number_of_edges() / pairs
    total = 0
    for node in graph:
        total += sum(nx.single_source_shortest_path_length(graph, node).values())
    return total / (N * (N - 1))


def random_connected_graph(target_aspl: float, lineage_id: int) -> tuple[nx.Graph | None, dict[str, Any]]:
    """Sample a connected random graph from a random tree plus random extra edges."""
    mean_degree = MEAN_DEGREE_BY_ASPL[target_aspl]
    edge_count = int(round(mean_degree * N / 2))
    if edge_count < N - 1:
        raise ValueError("target mean degree is too low for a connected graph")
    graph_seed = seed("tree_augmented_random_connected", target_aspl, lineage_id)
    if target_aspl == 1.0:
        graph = nx.complete_graph(N)
        nx.set_edge_attributes(graph, 1.0, "weight")
        validate_graph(graph, target_aspl, 1.0)
        return graph, {
            "accepted": True,
            "target_ASPL": target_aspl,
            "ASPL": 1.0,
            "target_mean_degree": mean_degree,
            "generation_attempts": 1,
            "graph_seed": graph_seed,
            "tree_seed": None,
            "extra_edges_added": edge_count - (N - 1),
        }
    rng = random.Random(graph_seed)
    last_aspl = None
    for attempt in range(1, MAX_GRAPH_ATTEMPTS + 1):
        tree_seed = rng.randrange(1 << 30)
        graph = nx.random_labeled_tree(N, seed=tree_seed)
        extra_edges = edge_count - (N - 1)
        if extra_edges:
            extra = rng.sample(list(nx.non_edges(graph)), extra_edges)
            graph.add_edges_from(extra)
        actual_aspl = exact_aspl(graph)
        last_aspl = actual_aspl
        if abs(actual_aspl - target_aspl) <= ASPL_ABS_TOLERANCE[target_aspl]:
            nx.set_edge_attributes(graph, 1.0, "weight")
            validate_graph(graph, target_aspl, actual_aspl)
            return graph, {
                "accepted": True,
                "target_ASPL": target_aspl,
                "ASPL": actual_aspl,
                "target_mean_degree": mean_degree,
                "generation_attempts": attempt,
                "graph_seed": graph_seed,
                "tree_seed": tree_seed,
                "extra_edges_added": extra_edges,
            }
    return None, {
        "accepted": False,
        "target_ASPL": target_aspl,
        "target_mean_degree": mean_degree,
        "generation_attempts": MAX_GRAPH_ATTEMPTS,
        "last_connected_ASPL": last_aspl,
        "graph_seed": graph_seed,
    }


def descriptors(graph: nx.Graph, *, known_aspl: float) -> dict[str, Any]:
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    return {"n": graph.number_of_nodes(), "m": graph.number_of_edges(),
            "ASPL": known_aspl, "connected": nx.is_connected(graph),
            "simple": not graph.is_multigraph(), "mean_degree": float(degrees.mean()),
            "degree_variance": float(degrees.var())}


def validate_graph(graph: nx.Graph, target_aspl: float, actual_aspl: float) -> None:
    """Reject a topology that does not satisfy the declared experiment cell."""
    expected_degree = MEAN_DEGREE_BY_ASPL[target_aspl]
    expected_edges = int(round(expected_degree * N / 2))
    if graph.number_of_nodes() != N or set(graph) != set(range(N)):
        raise RuntimeError("generated graph has an invalid node set")
    if graph.number_of_edges() != expected_edges:
        raise RuntimeError("generated graph has an invalid edge count")
    if graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise RuntimeError("generated graph is not simple")
    if not nx.is_connected(graph):
        raise RuntimeError("generated graph is not connected")
    if any(data.get("weight") != 1.0 for _, _, data in graph.edges(data=True)):
        raise RuntimeError("generated graph has a non-unit internal edge weight")
    if not np.isclose(2.0 * graph.number_of_edges() / N, expected_degree):
        raise RuntimeError("generated graph has an invalid mean degree")
    if abs(actual_aspl - target_aspl) > ASPL_ABS_TOLERANCE[target_aspl]:
        raise RuntimeError("generated graph violates its ASPL tolerance")


def store_graph(path: Path, graph: nx.Graph, descriptor: dict[str, Any]) -> None:
    payload = {"experiment": EXPERIMENT, "descriptor": descriptor,
               "edges": edge_rows(graph)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temporary, path)


def load_graph(path: Path) -> tuple[nx.Graph, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("experiment") not in {EXPERIMENT, LEGACY_EXPERIMENT}:
        raise ValueError(f"incompatible graph store: {path}")
    graph = nx.Graph(); graph.add_nodes_from(range(N))
    graph.add_weighted_edges_from(payload["edges"])
    if graph.number_of_nodes() != N or graph.number_of_edges() != payload["descriptor"]["m"]:
        raise ValueError(f"graph store is incomplete: {path}")
    return graph, payload["descriptor"]


def topology_manifest_path() -> Path:
    return ANALYSIS_ROOT / "topology_calibration_manifest.json"


def prepare_topology() -> dict[str, Any]:
    """Freeze topology before any epidemic tasks; safe to resume if valid."""
    manifest_path = topology_manifest_path()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment") != EXPERIMENT or manifest.get("topology") != "random_connected_tree_augmented":
            raise RuntimeError(f"existing data are incompatible with {EXPERIMENT}")
        if manifest.get("topology_gate") == "PASS":
            cells = manifest.get("cells", {})
            expected = set(cells) == {str(target) for target in ASPL_TARGETS}
            expected = expected and all(
                len(cells[str(target)]) == GRAPH_LINEAGES
                and all(row.get("accepted") for row in cells[str(target)])
                for target in ASPL_TARGETS
            )
            if not expected:
                raise RuntimeError("PASS topology manifest is structurally incomplete")
            for target in ASPL_TARGETS:
                for row in cells[str(target)]:
                    if abs(row["ASPL"] - target) > ASPL_ABS_TOLERANCE[target]:
                        raise RuntimeError("stored topology violates its ASPL tolerance")
                    load_graph(ROOT / row["graph_path"])
            return manifest

    legacy_manifest_path = LEGACY_ANALYSIS_ROOT / "topology_calibration_manifest.json"
    legacy_manifest = None
    if legacy_manifest_path.exists():
        candidate = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        if (candidate.get("experiment") == LEGACY_EXPERIMENT
                and candidate.get("topology_gate") == "PASS"):
            legacy_manifest = candidate

    root = root_folder() / "graphs"
    cells: dict[str, list[dict[str, Any]]] = {}
    gate = True
    for target in ASPL_TARGETS:
        rows = []
        for lineage_id in range(GRAPH_LINEAGES):
            if legacy_manifest is not None and lineage_id < 20:
                old_rows = legacy_manifest.get("cells", {}).get(str(target), [])
                if lineage_id < len(old_rows) and old_rows[lineage_id].get("accepted"):
                    old_row = dict(old_rows[lineage_id])
                    load_graph(ROOT / old_row["graph_path"])
                    rows.append(old_row)
                    print(f"[topology ASPL={target:g}] reused legacy graph {lineage_id + 1}/20", flush=True)
                    continue
            print(
                f"[topology ASPL={target:g}] graph {lineage_id + 1}/{GRAPH_LINEAGES} "
                f"(mean degree={MEAN_DEGREE_BY_ASPL[target]})",
                flush=True,
            )
            graph, generation = random_connected_graph(target, lineage_id)
            if graph is None:
                rows.append({"lineage_id": lineage_id, "graph_path": None, **generation})
                gate = False
                continue
            descriptor = descriptors(graph, known_aspl=generation["ASPL"])
            path = root / f"random_connected_tree_augmented_aspl{token(target)}" / f"graph{lineage_id:03d}.json.gz"
            store_graph(path, graph, descriptor)
            loaded, reloaded = load_graph(path)
            if reloaded["ASPL"] != descriptor["ASPL"] or loaded.number_of_edges() != descriptor["m"]:
                raise RuntimeError("graph store round-trip validation failed")
            validate_graph(loaded, target, reloaded["ASPL"])
            rows.append({
                "lineage_id": lineage_id,
                "graph_path": str(path.relative_to(ROOT)),
                **generation,
                **descriptor,
            })
        cells[str(target)] = rows
    gate = gate and all(
        len(cells[str(target)]) == GRAPH_LINEAGES
        and all(row.get("accepted") for row in cells[str(target)])
        for target in ASPL_TARGETS
    )
    manifest = {
        "experiment": EXPERIMENT,
        "topology": "random_connected_tree_augmented",
        "generator": "random labeled spanning tree plus uniformly sampled additional missing edges",
        "topology_gate": "PASS" if gate else "FAIL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ASPL_tolerance": ASPL_ABS_TOLERANCE,
        "mean_degree_by_target_ASPL": {str(key): value for key, value in MEAN_DEGREE_BY_ASPL.items()},
        "cells": cells,
        "interpretation": "ASPL is varied by changing mean degree; this is not a fixed-density causal ASPL design",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def interfaces(target_aspl: float, lineage_id: int, count: int) -> list[list[int]]:
    result = []
    for interface_id in range(count):
        nodes = list(range(1, N))
        random.Random(seed("interface", target_aspl, lineage_id, interface_id)).shuffle(nodes)
        result.append(nodes[:5])
    if len({tuple(nodes) for nodes in result}) != count:
        raise RuntimeError("duplicate interface placement; change interface seed namespace")
    return result


def task_seed(target_aspl: float, lineage_id: int, interface_id: int, event_id: int) -> dict[str, int]:
    key = ("events", target_aspl, lineage_id, interface_id, event_id)
    return {"full_event_seed": seed(*key, "full"), "micromacro_event_seed": seed(*key, "mm")}


def make_network(graph: nx.Graph, bridges: list[int]) -> tuple[list[nx.Graph], nx.Graph, np.ndarray]:
    sink = N; full = nx.Graph(); full.add_nodes_from(range(N + 1))
    full.add_edges_from((u, v, dict(data)) for u, v, data in graph.edges(data=True))
    full.add_edges_from((bridge, sink, {"weight": 1.0}) for bridge in bridges)
    full.graph.update({"community_sizes": [N, 1], "alphas": [1.0, 1.0], "boundary_counts": [len(bridges), 1],
                       "mu2_values": [1.0, 1.0], "spectral_descriptors_computed": False})
    sink_graph = nx.Graph(); sink_graph.add_node(sink)
    return [nx.Graph(graph), sink_graph], full, np.array([[0.0, len(bridges)], [len(bridges), 0.0]], dtype=float)


def benchmark(name: str, k: int, initial_node: int = 0) -> Benchmark:
    return Benchmark(name=name, family=EXPERIMENT, plot_groups=(), replications=1, base_seed=BASE_SEED,
                     network_seed_base=BASE_SEED, communities=2, community_size=N, inter_links=k, macro_graph_type="chain",
                     micro_graph_type="controlled", edge_prob=0.0, beta=BETA, gamma=GAMMA, model=MODEL,
                     tau_micro=TAU_MICRO, macro_T=MACRO_T, original_T_end=T_MAX, T_end=T_MAX, dt_out=T_MAX,
                     initial_node=initial_node, store_trajectories=False)


def infection_times_full(result: dict, initial_node: int = 0) -> dict[int, float]:
    values = {initial_node: 0.0}
    for event in result.get("transmission_events", []):
        if event.get("kind") in {"seed", "intra", "inter"} and event.get("dst_node") is not None:
            values.setdefault(int(event["dst_node"]), float(event["time"]))
    return values


def infection_times_mm(events: list[dict], initial_node: int = 0) -> dict[int, float]:
    values = {initial_node: 0.0}
    for event in events:
        if event.get("event_type") in {"seed", "infection", "transfer"} and event.get("node") is not None:
            values.setdefault(int(event["node"]), float(event["time"]))
    return values


def observables(times: dict[int, float], bridges: list[int]) -> dict[str, Any]:
    export = times.get(N); bridge_times = [times[node] for node in bridges if node in times]
    before = sum(value < export for value in bridge_times) if export is not None else None
    return {"T_export": export, "export_observed": export is not None, "T_bridge_first": min(bridge_times) if bridge_times else None,
            "bridge_infected_count_before_export": before, "bypass": bool(export is not None and before == 0),
            "I_source_before_export": sum(value < export for node, value in times.items() if node < N) if export is not None else None}


def checkpoint_path(task: dict[str, Any]) -> Path:
    return scenario_folder(task) / "checkpoints" / f"graph{task['lineage_id']:03d}_interface{task['interface_id']:02d}_event{task['event_id']:03d}.json.gz"


def legacy_checkpoint_path(task: dict[str, Any]) -> Path:
    return (LEGACY_ROOT / "scenarios" / scenario_name(task) / "checkpoints"
            / f"graph{task['lineage_id']:03d}_interface{task['interface_id']:02d}_event{task['event_id']:03d}.json.gz")


def materialize_legacy_checkpoint(task: dict[str, Any]) -> bool:
    """Place a legacy checkpoint in the new experiment folder without rerunning it."""
    destination = checkpoint_path(task)
    if destination.exists() or int(task["lineage_id"]) >= 20:
        return False
    source = legacy_checkpoint_path(task)
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def valid_checkpoint(task: dict[str, Any]) -> bool:
    path = checkpoint_path(task)
    try:
        payload = read_checkpoint(path)
        runs = payload.get("runs", [])
        simulator_names = [row.get("simulator") for row in runs]
        return (payload.get("experiment") in {EXPERIMENT, LEGACY_EXPERIMENT}
                and payload.get("task") == task and payload.get("T_max") == T_MAX
                and len(runs) == 2 and set(simulator_names) == {FULL, MICROMACRO}
                and all(isinstance(row.get("export_observed"), bool) and "termination" in row for row in runs)
                and payload.get("rng_policy", {}).get("independent_rng_per_MicroModel") is True)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def run_pair(task: dict[str, Any]) -> dict[str, Any]:
    # Public simulator APIs have no first-export callback.  Therefore this
    # safe runner evolves SI to natural absorption/T_MAX and extracts the
    # first committed sink infection afterwards; no scheduler is copied here.
    graph, _ = load_graph(ROOT / task["graph_path"])
    bridges = task["bridges"]
    initial_node = int(task.get("initial_node", 0))
    seeds = task_seed(task["target_ASPL"], task["lineage_id"], task["interface_id"], task["event_id"])
    micro_graphs, full_graph, w = make_network(graph, bridges); started = time.perf_counter()
    full_result = MicroSimulator(full_graph=full_graph, comm_nodes=[list(range(N)), [N]], infection_rate=BETA,
        recovery_rate=GAMMA, model=MODEL).run(T_end=T_MAX, dt_out=T_MAX, initial_node=initial_node,
        rng=random.Random(seeds["full_event_seed"]), stop_at_absorption=True)
    full_runtime = time.perf_counter() - started; full_obs = observables(infection_times_full(full_result, initial_node), bridges)
    full_stop = full_result.get("run_metadata"); del full_result, micro_graphs, full_graph, w; gc.collect()
    micro_graphs, full_graph, w = make_network(graph, bridges); started = time.perf_counter()
    mm_result, mm_stop = _run_experimental_micromacro(benchmark(task["family"], task["k"], initial_node), micro_graphs, full_graph, w,
        seeds["micromacro_event_seed"], independent=True, dense_logs=False, stop_at_absorption=True, return_metadata=True)
    mm_runtime = time.perf_counter() - started; mm_obs = observables(infection_times_mm(mm_result[3], initial_node), bridges)
    del mm_result, micro_graphs, full_graph, w; gc.collect()
    if full_obs["bypass"]: raise RuntimeError("Full bypassed every physical bridge")
    return {"experiment": EXPERIMENT, "task": task,
            "T_max": T_MAX, "seeds": seeds,
            "rng_policy": {"independent_rng_per_MicroModel": True, "cache_policy": "discard", "reimport_policy": "normal",
              "clone_copies_community_rng_state": True, "refresh_synchronizes_clone_rng_state": True},
            "runs": [{"simulator": FULL, "runtime_seconds": full_runtime, "termination": full_stop, **full_obs},
                     {"simulator": MICROMACRO, "runtime_seconds": mm_runtime, "termination": mm_stop, **mm_obs}]}


def tasks(manifest: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    if manifest["topology_gate"] != "PASS": raise RuntimeError("topology gate failed; no simulation tasks may be scheduled")
    result = []
    for target in ASPL_TARGETS:
        family = f"random_connected_aspl{token(target)}"
        entries = manifest["cells"][str(target)]
        selected = entries[:1] if smoke else entries
        for entry in selected:
            graph, _ = load_graph(ROOT / entry["graph_path"])
            interface_count = SMOKE_INTERFACES if smoke else INTERFACE_REPLICATES
            bridge_lists = interfaces(target, int(entry["lineage_id"]), interface_count)
            for interface_id, candidates in enumerate(bridge_lists):
                for k in ((5,) if smoke else K_VALUES):
                    bridges = sorted(candidates[:k])
                    initial_node = (seed("initial_node", target, int(entry["lineage_id"]), interface_id, 0) % N
                                    if RANDOM_INITIAL_NODE else 0)
                    distances = [nx.shortest_path_length(graph, initial_node, bridge) for bridge in bridges]
                    for event_id in range(EVENT_REPLICATES_PER_INTERFACE):
                        task = {"family": family, "target_ASPL": entry.get("target_ASPL"), "lineage_id": int(entry["lineage_id"]),
                          "interface_id": interface_id, "event_id": event_id, "k": k, "bridges": bridges,
                          "graph_path": entry["graph_path"], "actual_ASPL": entry["ASPL"],
                          "seed_bridge_distances": distances}
                        if RANDOM_INITIAL_NODE:
                            task["initial_node"] = initial_node
                        result.append(task)
    return result


def write_run_manifest(all_tasks: list[dict[str, Any]], smoke: bool) -> None:
    ensure_fixed_json(root_folder() / "experiment_parameters.json", experiment_parameters(), "experiment_parameters.json")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in all_tasks:
        grouped.setdefault(scenario_name(task), []).append(task)
    summaries = []
    for name, scenario_tasks in sorted(grouped.items()):
        folder = scenario_folder(scenario_tasks[0])
        ensure_fixed_json(folder / "parameters.json", scenario_description(scenario_tasks[0]), "scenario parameters")
        completed = sum(valid_checkpoint(task) for task in scenario_tasks)
        run_summary = {
            "scenario": name,
            "mode": "smoke" if smoke else "full",
            "requested_pairs_in_this_run": len(scenario_tasks),
            "completed_pairs_in_this_run": completed,
            "full_design_pairs": REPLICATIONS,
            "status": (
                "smoke_complete" if smoke and completed == len(scenario_tasks)
                else "complete" if not smoke and completed == len(scenario_tasks)
                else "incomplete"
            ),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(folder / "run_manifest.json", run_summary)
        summaries.append(run_summary)
    atomic_json(root_folder() / "run_manifest.json", {
        "experiment": EXPERIMENT,
        "mode": "smoke" if smoke else "full",
        "scenario_count": len(grouped),
        "scenarios": summaries,
        "independent_rng": True,
        "horizon_policy": "natural SI absorption or T_max; no padded event time",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def bounded_run(pending: list[dict[str, Any]], workers: int) -> None:
    done = 0; total = len(pending)
    def submit_group(group: list[dict[str, Any]], count: int) -> None:
        nonlocal done
        if not group: return
        iterator = iter(group); active: dict[concurrent.futures.Future, dict[str, Any]] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=count) as pool:
            while True:
                while len(active) < 2 * count:
                    try: task = next(iterator)
                    except StopIteration: break
                    active[pool.submit(run_pair, task)] = task
                if not active: break
                future = next(concurrent.futures.as_completed(active)); task = active.pop(future)
                write_checkpoint(checkpoint_path(task), future.result()); done += 1
                print(f"[progress {done}/{total}] {task['family']} k={task['k']} lineage={task['lineage_id']} interface={task['interface_id']}", flush=True)
    submit_group(pending, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-regression-gate", action="store_true", help="debug only")
    args = parser.parse_args()
    if args.workers < 1: raise ValueError("workers must be positive")
    ensure_fixed_json(root_folder() / "experiment_parameters.json", experiment_parameters(), "experiment_parameters.json")
    topology = prepare_topology()
    accepted = sum(row.get("accepted", False) for rows in topology["cells"].values() for row in rows)
    expected = len(ASPL_TARGETS) * GRAPH_LINEAGES
    print(f"[topology] {topology['topology_gate']}; accepted={accepted}/{expected}; manifest={topology_manifest_path()}", flush=True)
    if args.topology_only:
        return
    if topology["topology_gate"] != "PASS": raise SystemExit("topology gate failed; refusing simulation run")
    if not args.skip_regression_gate:
        passed, message = _regression_gate(); print(f"Regression gate: {'PASS' if passed else 'FAIL'} — {message}", flush=True)
        if not passed: raise SystemExit(2)
    all_tasks = tasks(topology, args.smoke_only)
    reused_legacy = sum(materialize_legacy_checkpoint(task) for task in all_tasks)
    pending = [task for task in all_tasks if not valid_checkpoint(task)]
    write_run_manifest(all_tasks, args.smoke_only)
    print(f"[queued] mode={'smoke' if args.smoke_only else 'full'} reused={len(all_tasks)-len(pending)}/{len(all_tasks)} "
          f"(legacy imported={reused_legacy}) new={len(pending)}", flush=True)
    bounded_run(pending, args.workers)
    write_run_manifest(all_tasks, args.smoke_only)


if __name__ == "__main__":
    main()
