#!/usr/bin/env python3
"""Controlled one-source/one-sink export experiments (E0, E1, E2).

The runner implements the design in
``paper/SMPT paper/EXPORT_PATH_SPECTRUM_EXPERIMENT_PLAN.md``.  It compares
Full Micro with the existing Production MicroMacro scheduler, changing only
MicroModel RNG ownership through the established independent-RNG helper.
Production source files are never modified.

The default command prepares/resumes the exploratory E0/E1/E2 collection:

    .venv/bin/python3 extrascripts/runners/run_single_community_export_independent_rng.py

Use ``--topology-only`` first, then ``--smoke-only`` (two graphs × two event
replicates where applicable).  Each completed pair is an atomic gzip
checkpoint under ``output/data``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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
    _regression_gate,
    _run_experimental_micromacro,
)
from extrascripts.runners.single_community_export_design import (
    INTERNAL_DEGREE,
    N,
    canonical_seed,
    descriptors,
    e0a_graph,
    e0b_graph,
    e1_placements,
    e1_source_graph,
    e2_interfaces_at_distance,
    graph_payload,
    modular_regular_cut,
)
from two_layer_ctmc.simulators.micro import MicroSimulator


# ------------------------- editable scientific configuration -------------------------

MODEL = 2                         # SI
GAMMA = 0.0
BETA_IN = 0.003
OUT_IN_RATIOS = (1.0,)
MACRO_T = 1.0
TAU_MICRO = 1.0
T_MAX = 30_000.0                  # maximum observation horizon, never padding
INITIAL_NODE = 0
BASE_SEED = 20_260_918
WORKERS = 6
HEAVY_WORKERS = 2                 # processes used for E0 clique-side jobs
GRAPH_REPLICATES = 10
EVENT_REPLICATES = 10             # 10×10 = 100 paired runs per stochastic cell
E0_EVENT_REPLICATES = 100
E2_TOPOLOGY_POOL = 30
E2_REQUIRED_MATCHED_BLOCKS = 10
BLOCKS = ("E0", "E1", "E2")
OUTPUT_ROOT = REPO_ROOT / "output" / "data"

EXPERIMENT_VERSION = "single_community_export_independent_rng_v1"


def number_token(value: float | None) -> str:
    if value is None:
        return "na"
    return f"{value:g}".replace("-", "m").replace(".", "p")


def design_hash() -> str:
    scientific = {
        "version": EXPERIMENT_VERSION, "model": MODEL, "gamma": GAMMA,
        "beta_in": BETA_IN, "out_in_ratios": OUT_IN_RATIOS,
        "macro_T": MACRO_T, "tau_micro": TAU_MICRO, "n": N,
        "internal_degree": INTERNAL_DEGREE,
    }
    encoded = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def folder_name(cell: dict[str, Any]) -> str:
    return (
        f"export_single_{cell['topology']}_n{N}_d{number_token(cell['d'])}_k{cell['k']}"
        f"_bin{number_token(BETA_IN)}_bout{number_token(cell['beta_out'])}"
        f"_tau{number_token(TAU_MICRO)}_{cell['placement']}_independent_rng_{design_hash()}"
    )


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cells_e0() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for ell in (2, 4, 8):
        for side in ("path", "clique", "random_connected"):
            cells.append({"block": "E0", "kind": "e0a", "topology": f"e0a_{side}_ell{ell}",
                          "placement": "bridge", "side": side, "ell": ell, "k": 1,
                          "d": None, "beta_out": BETA_IN})
    for q in (1, 2, 4):
        cells.append({"block": "E0", "kind": "e0b", "topology": f"e0b_q{q}_ell4",
                      "placement": "bridge", "q": q, "ell": 4, "k": 1,
                      "d": None, "beta_out": BETA_IN})
    return cells


def _cell_e1(family: str, k: int, placement: str) -> dict[str, Any]:
    return {"block": "E1", "kind": "e1", "topology": family,
            "placement": placement, "family": family, "k": k,
            "d": INTERNAL_DEGREE if family != "clique" else N - 1,
            "beta_out": BETA_IN}


def _cell_e2(cut: int, side: str) -> dict[str, Any]:
    return {"block": "E2", "kind": "e2", "topology": f"modular_regular_cut{cut}",
            "placement": side, "cut_edges": cut, "k": 5, "d": INTERNAL_DEGREE,
            "beta_out": BETA_IN}


def _build_graph(cell: dict[str, Any], graph_id: int) -> tuple[nx.Graph, int, list[int], dict[str, Any]]:
    """Reconstruct a deterministic source graph/interface for one task."""
    kind = cell["kind"]
    if kind == "e0a":
        graph, source, bridges, meta = e0a_graph(cell["side"], int(cell["ell"]), seed=canonical_seed(BASE_SEED, cell["topology"]))
    elif kind == "e0b":
        graph, source, bridges, meta = e0b_graph(int(cell["q"]), int(cell["ell"]))
    elif kind == "e1":
        if cell["family"] == "clique":
            graph = nx.complete_graph(N)
            graph = nx.Graph(graph, weight=1.0)
            source = INITIAL_NODE
            candidates = [node for node in graph if node != source]
            rng = random.Random(canonical_seed(BASE_SEED, "E1", "clique", graph_id, cell["k"]))
            bridges = sorted(rng.sample(candidates, int(cell["k"])))
            meta = {"block": "E1", "family": "clique", "graph_id": graph_id,
                    "selection_status": "reference"}
        else:
            graph, meta = e1_source_graph(cell["family"], graph_id)
            source = INITIAL_NODE
            placements, selection = e1_placements(
                graph, source, int(cell["k"]), canonical_seed(BASE_SEED, "E1-interface", cell["family"], graph_id, cell["k"])
            )
            if selection["status"] != "feasible":
                raise RuntimeError(f"E1 infeasible {cell['family']} g={graph_id} k={cell['k']}: {selection['reason']}")
            bridges = placements[cell["placement"]]
            meta = {**meta, "gateway_selection": selection}
    elif kind == "e2":
        graph, partition = modular_regular_cut(int(cell["cut_edges"]), seed=canonical_seed(BASE_SEED, "E2", graph_id))
        source = INITIAL_NODE
        # The planning pass saves selected common distance for every matched id.
        distance = int(cell["matched_distance_by_graph"][str(graph_id)])
        interfaces, selection = e2_interfaces_at_distance(graph, partition, source, distance, int(cell["k"]))
        if selection["status"] != "feasible":
            raise RuntimeError(f"E2 matched design unexpectedly infeasible: g={graph_id}, D={distance}")
        bridges = interfaces[cell["placement"]]
        meta = {"block": "E2", "graph_id": graph_id, "cut_edges": cell["cut_edges"],
                "partition": {str(node): int(side) for node, side in partition.items()},
                "gateway_selection": selection, "source_side": partition[source],
                "bridge_side": cell["placement"]}
    else:
        raise ValueError(f"unknown task kind {kind}")
    return graph, source, bridges, meta


def _e2_matched_graphs() -> tuple[list[int], dict[str, int], list[dict[str, Any]]]:
    """Find the first deterministic graph blocks with a common D across all cuts."""
    matched: list[int] = []
    distances: dict[str, int] = {}
    report: list[dict[str, Any]] = []
    for graph_id in range(E2_TOPOLOGY_POOL):
        variants = []
        for cut in (2, 4, 16):
            graph, partition = modular_regular_cut(cut, seed=canonical_seed(BASE_SEED, "E2", graph_id))
            variants.append((cut, graph, partition))
        chosen = None
        for distance in (3, 2, 4):
            if all(e2_interfaces_at_distance(graph, partition, INITIAL_NODE, distance, 5)[0]
                   for _, graph, partition in variants):
                chosen = distance
                break
        report.append({"graph_id": graph_id, "status": "matched" if chosen is not None else "unmatched",
                       "common_distance": chosen})
        if chosen is not None and len(matched) < E2_REQUIRED_MATCHED_BLOCKS:
            matched.append(graph_id)
            distances[str(graph_id)] = chosen
    return matched, distances, report


def make_plan(blocks: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = set(blocks)
    cells: list[dict[str, Any]] = []
    feasibility: dict[str, Any] = {"version": EXPERIMENT_VERSION, "design_hash": design_hash(), "blocks": sorted(selected)}
    if "E0" in selected:
        cells.extend(_cells_e0())
    if "E1" in selected:
        e1_report = []
        for family in ("random_regular", "modular_regular_cut4", "random_connected_gnm"):
            for k in (1, 5):
                feasible_ids = []
                for graph_id in range(GRAPH_REPLICATES):
                    graph, _ = e1_source_graph(family, graph_id)
                    placements, status = e1_placements(graph, INITIAL_NODE, k,
                                                       canonical_seed(BASE_SEED, "E1-interface", family, graph_id, k))
                    e1_report.append({"family": family, "k": k, "graph_id": graph_id, **status})
                    if placements:
                        feasible_ids.append(graph_id)
                # Cells carry their feasible ids so one unavailable graph does
                # not silently get replaced with a weaker interface.
                for placement in ("near", "middle", "far"):
                    cell = _cell_e1(family, k, placement)
                    cell["graph_ids"] = feasible_ids
                    cells.append(cell)
        reference = _cell_e1("clique", 5, "reference")
        reference["graph_ids"] = [0]
        cells.append(reference)
        feasibility["E1"] = e1_report
    if "E2" in selected:
        matched_ids, distances, report = _e2_matched_graphs()
        feasibility["E2"] = {"pool": report, "matched_graph_ids": matched_ids,
                               "matched_distance_by_graph": distances,
                               "required_matched_blocks": E2_REQUIRED_MATCHED_BLOCKS}
        for cut in (2, 4, 16):
            for side in ("same_side", "opposite_side"):
                cell = _cell_e2(cut, side)
                cell["graph_ids"] = matched_ids
                cell["matched_distance_by_graph"] = distances
                cells.append(cell)
    for cell in cells:
        cell["folder"] = folder_name(cell)
    feasibility["cell_count"] = len(cells)
    feasibility["feasible_cells"] = sum(bool(cell.get("graph_ids", [0])) for cell in cells)
    return cells, feasibility


def _make_network(source_graph: nx.Graph, bridges: list[int], beta_out: float) -> tuple[list[nx.Graph], nx.Graph, np.ndarray]:
    sink = N
    full = nx.Graph()
    full.add_nodes_from(range(N + 1))
    full.add_edges_from((u, v, dict(data)) for u, v, data in source_graph.edges(data=True))
    for bridge in bridges:
        full.add_edge(bridge, sink, weight=1.0)
    source = nx.Graph(source_graph)
    destination = nx.Graph()
    destination.add_node(sink)
    full.graph.update({
        "community_sizes": [N, 1], "alphas": [1.0, 1.0],
        "boundary_counts": [len(bridges), 1], "mu2_values": [1.0, 1.0],
        "spectral_descriptors_computed": False,
    })
    W = np.array([[0.0, float(len(bridges))], [float(len(bridges)), 0.0]])
    return [source, destination], full, W


def _benchmark(cell: dict[str, Any]) -> Benchmark:
    return Benchmark(
        name=cell["folder"], family="single_community_export", plot_groups=(),
        replications=1, base_seed=BASE_SEED, network_seed_base=BASE_SEED,
        communities=2, community_size=N, inter_links=int(cell["k"]), macro_graph_type="chain",
        micro_graph_type="controlled", edge_prob=0.0, beta=BETA_IN, gamma=GAMMA,
        model=MODEL, tau_micro=TAU_MICRO, macro_T=MACRO_T, original_T_end=T_MAX,
        T_end=T_MAX, dt_out=T_MAX, initial_node=INITIAL_NODE, store_trajectories=False,
    )


def _infection_times_full(result: dict, source: int) -> dict[str, float]:
    times = {str(source): 0.0}
    for event in result.get("transmission_events", []):
        if event.get("kind") in {"seed", "intra", "inter"} and event.get("dst_node") is not None:
            node, t = str(int(event["dst_node"])), float(event["time"])
            times.setdefault(node, t)
    return times


def _infection_times_mm(events: list[dict], source: int) -> dict[str, float]:
    times = {str(source): 0.0}
    for event in events:
        if event.get("event_type") in {"seed", "infection", "transfer"} and event.get("node") is not None:
            node, t = str(int(event["node"])), float(event["time"])
            times.setdefault(node, t)
    return times


def _event_observables(infection_times: dict[str, float], bridges: list[int], sink: int) -> dict[str, Any]:
    export = infection_times.get(str(sink))
    bridge_times = [infection_times[str(node)] for node in bridges if str(node) in infection_times]
    bridge_first = min(bridge_times) if bridge_times else None
    bridge_count = (sum(time < export for time in bridge_times) if export is not None else None)
    bypass = export is not None and (bridge_count == 0)
    return {
        "T_export": export, "export_observed": export is not None,
        "T_bridge_first": bridge_first, "bridge_infected_count_at_export_minus": bridge_count,
        "bypass": bypass,
        "I_source_at_export_minus": (sum(time < export for node, time in infection_times.items()
                                           if int(node) < N) if export is not None else None),
        "bridge_to_export_delay": (export - bridge_first if export is not None and bridge_first is not None else None),
    }


def _seed_map(cell: dict[str, Any], graph_id: int, event_id: int) -> dict[str, int]:
    # E1 deliberately reuses both Full and MM event seeds across the three
    # interface placements.  The tasks remain different Full networks, but
    # this makes stochastic pairing transparent and supports the MM invariant.
    placement_key = None if cell["kind"] == "e1" else cell["placement"]
    key = (EXPERIMENT_VERSION, cell["block"], cell["topology"], cell["k"],
           placement_key, graph_id, event_id)
    return {
        "topology_seed": canonical_seed(BASE_SEED, *key, "topology"),
        "gateway_seed": canonical_seed(BASE_SEED, *key, "gateway"),
        "full_event_seed": canonical_seed(BASE_SEED, *key, "full"),
        # Placement intentionally omitted for the macro seed in E1; it is the
        # regression invariant specified by the design.
        "micromacro_event_seed": canonical_seed(BASE_SEED, cell["block"], cell["topology"], cell["k"], graph_id, event_id, "mm"),
    }


def _run_one(cell: dict[str, Any], graph_id: int, event_id: int) -> dict[str, Any]:
    graph, source, bridges, design_meta = _build_graph(cell, graph_id)
    descriptor = descriptors(graph, source, bridges, family=cell["topology"], extra=design_meta)
    seeds = _seed_map(cell, graph_id, event_id)
    benchmark = _benchmark(cell)
    micro_graphs, full_graph, W = _make_network(graph, bridges, float(cell["beta_out"]))
    started = time.perf_counter()
    full_result = MicroSimulator(
        full_graph=full_graph, comm_nodes=[list(range(N)), [N]], infection_rate=BETA_IN,
        recovery_rate=GAMMA, model=MODEL,
    ).run(T_end=T_MAX, dt_out=T_MAX, initial_node=source,
          rng=random.Random(seeds["full_event_seed"]), stop_at_absorption=True)
    full_runtime = time.perf_counter() - started
    full_times = _infection_times_full(full_result, source)
    full_obs = _event_observables(full_times, bridges, N)
    full_stop = full_result.get("run_metadata")
    del full_result, full_graph, micro_graphs, W
    gc.collect()

    micro_graphs, full_graph, W = _make_network(graph, bridges, float(cell["beta_out"]))
    started = time.perf_counter()
    mm_result, mm_metadata = _run_experimental_micromacro(
        benchmark, micro_graphs, full_graph, W, seeds["micromacro_event_seed"],
        independent=True, dense_logs=False, stop_at_absorption=True, return_metadata=True,
    )
    mm_runtime = time.perf_counter() - started
    mm_events = mm_result[3]
    mm_times = _infection_times_mm(mm_events, source)
    mm_obs = _event_observables(mm_times, bridges, N)
    mm_log_hash = hashlib.sha256(stable_json(mm_events).encode()).hexdigest()
    del mm_result, micro_graphs, full_graph, W
    gc.collect()
    if full_obs["bypass"]:
        raise RuntimeError("Full Micro export bypassed every true bridge; topology/event extraction is invalid")

    return {
        "format_version": 1, "experiment": EXPERIMENT_VERSION,
        "design_hash": design_hash(), "scenario": cell["folder"], "cell": {k: v for k, v in cell.items() if k != "folder"},
        "graph_id": graph_id, "event_id": event_id, "seeds": seeds,
        "network": graph_payload(graph, source, bridges, {"descriptors": descriptor, "design": design_meta}),
        "rng_policy": {
            "independent_rng_per_MicroModel": True,
            "community_seed_formula": "micromacro_event_seed + 10000 + 1009 * community",
            "clone_copies_community_rng_state": True,
            "refresh_synchronizes_clone_rng_state": True,
            "cache_policy": "discard", "reimport_policy": "normal",
        },
        "runs": [
            {"simulator": FULL, "runtime_seconds": full_runtime, "termination": full_stop, **full_obs},
            {"simulator": MICROMACRO, "runtime_seconds": mm_runtime, "termination": mm_metadata,
             "micromacro_event_log_sha256": mm_log_hash, **mm_obs},
        ],
        "infection_times": {FULL: full_times, MICROMACRO: mm_times},
    }


def checkpoint_path(folder: Path, graph_id: int, event_id: int) -> Path:
    return folder / "checkpoints" / f"g{graph_id:04d}_e{event_id:04d}.json.gz"


def valid_checkpoint(path: Path, cell: dict[str, Any], graph_id: int, event_id: int) -> bool:
    try:
        payload = read_checkpoint(path)
        simulators = {item.get("simulator") for item in payload.get("runs", [])}
        return (payload.get("experiment") == EXPERIMENT_VERSION and payload.get("design_hash") == design_hash()
                and payload.get("scenario") == cell["folder"] and payload.get("graph_id") == graph_id
                and payload.get("event_id") == event_id and simulators == {FULL, MICROMACRO}
                and payload.get("rng_policy", {}).get("independent_rng_per_MicroModel") is True)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _run_task(task: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    cell, graph_id, event_id = task
    return _run_one(cell, graph_id, event_id)


def _manifest(cell: dict[str, Any], completed: list[tuple[int, int]], failures: dict[str, str], requested: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_VERSION, "design_hash": design_hash(), "scenario": cell["folder"],
        "cell": {k: v for k, v in cell.items() if k not in {"folder", "matched_distance_by_graph"}},
        "requested_event_pairs": requested, "completed": len(completed),
        "completed_ids": [{"graph_id": graph_id, "event_id": event_id} for graph_id, event_id in sorted(completed)],
        "failures": failures, "algorithm": "Production MicroMacro with independent RNG per MicroModel",
        "production_settings": {"alpha_mode": "off", "correction_mode": "off", "tau_micro": TAU_MICRO,
                                "macro_T": MACRO_T, "midpoint_hazard": True, "uniform_import": True},
        "horizon_policy": "run through natural SI absorption or T_MAX; store no padded trajectories",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_network_once(folder: Path, cell: dict[str, Any], graph_id: int) -> None:
    path = folder / "networks" / f"g{graph_id:04d}.json.gz"
    if path.exists():
        return
    graph, source, bridges, meta = _build_graph(cell, graph_id)
    payload = {"format_version": 1, "experiment": EXPERIMENT_VERSION, "design_hash": design_hash(),
               "graph_id": graph_id, "cell": {k: v for k, v in cell.items() if k != "folder"},
               "network": graph_payload(graph, source, bridges,
                                        {"descriptors": descriptors(graph, source, bridges, family=cell["topology"], extra=meta),
                                         "design": meta})}
    write_checkpoint(path, payload)


def _tasks_for_cell(cell: dict[str, Any], smoke_only: bool) -> list[tuple[dict[str, Any], int, int]]:
    if cell["kind"].startswith("e0"):
        graph_ids = [0]
        events = range(2 if smoke_only else E0_EVENT_REPLICATES)
    elif cell["kind"] == "e1" and cell["family"] == "clique":
        graph_ids = [0]
        events = range(2 if smoke_only else E0_EVENT_REPLICATES)
    else:
        graph_ids = cell.get("graph_ids", [])[:2 if smoke_only else None]
        events = range(2 if smoke_only else EVENT_REPLICATES)
    return [(cell, graph_id, event_id) for graph_id in graph_ids for event_id in events]


def _topology_check(cells: list[dict[str, Any]], feasibility: dict[str, Any]) -> dict[str, Any]:
    report = {"experiment": EXPERIMENT_VERSION, "design_hash": design_hash(), "feasibility": feasibility, "cells": []}
    for cell in cells:
        ids = [0] if cell["kind"].startswith("e0") else cell.get("graph_ids", [0])
        if cell["kind"] == "e1" and cell["family"] == "clique":
            ids = [0]
        for graph_id in ids[:2]:
            graph, source, bridges, meta = _build_graph(cell, graph_id)
            desc = descriptors(graph, source, bridges, family=cell["topology"], extra=meta)
            valid = nx.is_connected(graph) and source not in bridges and len(set(bridges)) == len(bridges)
            if cell["kind"] == "e2":
                valid = valid and all(degree == INTERNAL_DEGREE for _, degree in graph.degree()) and abs(desc["rho_adjacency"] - INTERNAL_DEGREE) < 1e-6
            report["cells"].append({"scenario": cell["folder"], "graph_id": graph_id, "valid": valid,
                                    "graph_hash": desc["graph_hash"], "D_histogram": desc["distance_histogram"],
                                    "lambda2_normalized": desc["lambda2_normalized_laplacian"],
                                    "rho": desc["rho_adjacency"]})
    return report


def _check_e1_invariant(states: dict[str, dict[str, Any]]) -> None:
    """Assert the prescribed placement-invariance of basic Production MM.

    Basic Production sees the source graph, n, k and W, but not which source
    nodes form the physical output interface.  Therefore all E1 placements at
    one graph/event/k must have an identical MM event-log hash and export time.
    This check includes resumed checkpoints as well as tasks just generated.
    """
    groups: dict[tuple[str, int, int, int], list[tuple[str, str, Any]]] = defaultdict(list)
    for state in states.values():
        cell = state["cell"]
        if cell["kind"] != "e1" or cell["family"] == "clique":
            continue
        for graph_id, event_id in set(state["completed"]):
            path = checkpoint_path(state["folder"], graph_id, event_id)
            if not valid_checkpoint(path, cell, graph_id, event_id):
                continue
            payload = read_checkpoint(path)
            row = next(item for item in payload["runs"] if item["simulator"] == MICROMACRO)
            groups[(cell["topology"], int(cell["k"]), graph_id, event_id)].append(
                (cell["placement"], row.get("micromacro_event_log_sha256"), row.get("T_export"))
            )
    mismatches = []
    for key, values in groups.items():
        placements = {value[0] for value in values}
        signatures = {(value[1], value[2]) for value in values}
        # A partial smoke/resume group cannot prove an invariant; it is not an
        # error.  Once all three are present their results must match exactly.
        if placements == {"near", "middle", "far"} and len(signatures) != 1:
            mismatches.append({"topology": key[0], "k": key[1], "graph_id": key[2],
                               "event_id": key[3], "values": values})
    if mismatches:
        report = OUTPUT_ROOT / "single_community_export_e1_invariant_failure.json"
        atomic_json(report, {"experiment": EXPERIMENT_VERSION, "design_hash": design_hash(),
                             "mismatches": mismatches})
        raise RuntimeError(f"E1 MicroMacro placement invariant failed; details: {report}")
    checked = sum(1 for values in groups.values() if {value[0] for value in values} == {"near", "middle", "far"})
    print(f"[E1 invariant] PASS on {checked} complete near/middle/far graph-event groups", flush=True)


def run(cells: list[dict[str, Any]], feasibility: dict[str, Any], *, smoke_only: bool, workers: int, topology_only: bool) -> None:
    if topology_only:
        report = _topology_check(cells, feasibility)
        destination = OUTPUT_ROOT / "single_community_export_design_feasibility.json"
        atomic_json(destination, report)
        total = len(report["cells"])
        valid = sum(row["valid"] for row in report["cells"])
        print(f"[topology] {valid}/{total} inspected graph/interfaces valid; report: {destination}", flush=True)
        e2 = feasibility.get("E2", {})
        if e2:
            print(f"[topology] E2 matched graph blocks: {len(e2['matched_graph_ids'])}/{E2_REQUIRED_MATCHED_BLOCKS}", flush=True)
        return

    states: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], int, int]] = []
    for cell in cells:
        folder = OUTPUT_ROOT / cell["folder"]
        tasks = _tasks_for_cell(cell, smoke_only)
        # Empty graph ids represent an explicit unavailable design, not a
        # malformed weaker substitute.
        for graph_id in sorted({task[1] for task in tasks}):
            _write_network_once(folder, cell, graph_id)
        completed = [(graph_id, event_id) for _, graph_id, event_id in tasks
                     if valid_checkpoint(checkpoint_path(folder, graph_id, event_id), cell, graph_id, event_id)]
        missing = [task for task in tasks if (task[1], task[2]) not in set(completed)]
        states[cell["folder"]] = {"cell": cell, "folder": folder, "tasks": tasks,
                                   "completed": completed, "failures": {}}
        atomic_json(folder / "manifest.json", _manifest(cell, completed, {}, len(tasks)))
        print(f"[queued] {cell['folder']}: reused={len(completed)}/{len(tasks)}; new={len(missing)}", flush=True)
        pending.extend(missing)

    if not pending:
        print("[resume] all requested checkpoints already exist", flush=True)
    else:
        print(f"[workers] processes={workers}; pending pairs={len(pending)}", flush=True)
        done = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_task, task): task for task in pending}
            for future in concurrent.futures.as_completed(futures):
                cell, graph_id, event_id = futures[future]
                state = states[cell["folder"]]
                key = f"g{graph_id:04d}_e{event_id:04d}"
                try:
                    payload = future.result()
                    write_checkpoint(checkpoint_path(state["folder"], graph_id, event_id), payload)
                    state["completed"].append((graph_id, event_id))
                    status = "ok"
                except Exception as exc:
                    state["failures"][key] = f"{type(exc).__name__}: {exc}"
                    status = f"FAILED {state['failures'][key]}"
                done += 1
                atomic_json(state["folder"] / "manifest.json",
                            _manifest(cell, state["completed"], state["failures"], len(state["tasks"])))
                print(f"[progress {done}/{len(pending)}] {cell['block']} {cell['topology']} {cell['placement']} "
                      f"g={graph_id} e={event_id}: {status}; remaining={len(pending)-done}", flush=True)

    _check_e1_invariant(states)
    incomplete = []
    for state in states.values():
        cell, folder = state["cell"], state["folder"]
        completed = sorted(set(state["completed"]))
        atomic_json(folder / "manifest.json", _manifest(cell, completed, state["failures"], len(state["tasks"])))
        print(f"[scenario complete] {cell['folder']}: {len(completed)}/{len(state['tasks'])}", flush=True)
        if len(completed) != len(state["tasks"]):
            incomplete.append(cell["folder"])
    if incomplete:
        raise SystemExit(f"incomplete scenarios: {len(incomplete)}; rerun the same command to resume")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", nargs="+", choices=("E0", "E1", "E2"), default=list(BLOCKS))
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--resume", action="store_true", help="accepted explicitly; valid checkpoints are always reused")
    parser.add_argument("--skip-regression-gate", action="store_true",
                        help="debug-only: do not run the shared-RNG production pathwise gate")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    cells, feasibility = make_plan(args.blocks)
    print(f"[plan] blocks={','.join(args.blocks)} cells={len(cells)} design={design_hash()} "
          f"mode={'topology' if args.topology_only else ('smoke' if args.smoke_only else 'full')}", flush=True)
    if "E2" in args.blocks and len(feasibility.get("E2", {}).get("matched_graph_ids", [])) < E2_REQUIRED_MATCHED_BLOCKS:
        print("[limitation] E2 has fewer than the requested matched graph blocks; only the feasible exploratory subset is scheduled.", flush=True)
    if not args.topology_only and not args.skip_regression_gate:
        passed, detail = _regression_gate()
        print(f"[regression gate] {'PASS' if passed else 'FAIL'}: {detail}", flush=True)
        if not passed:
            raise SystemExit("production shared-RNG regression gate failed; no experimental tasks were started")
    run(cells, feasibility, smoke_only=args.smoke_only, workers=args.workers, topology_only=args.topology_only)


if __name__ == "__main__":
    main()
