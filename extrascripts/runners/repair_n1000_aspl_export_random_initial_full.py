#!/usr/bin/env python3
"""Repair the ASPL 2--6 export data by rerunning only Full Micro.

The old paired checkpoints recorded a random initial node, but their Full
Micro run started at node 0. MicroMacro used the recorded node correctly.
This runner preserves each old MicroMacro row, network, interface, and event
seed, and writes corrected pairs to a separate resumable folder. The source
folder must remain available because graph paths refer to its graph files.

Full run: python extrascripts/runners/repair_n1000_aspl_export_random_initial_full.py
Small test: add --limit 1
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.runners.checkpoint_io import read_checkpoint, write_checkpoint
from extrascripts.runners import run_n1000_aspl_export_independent_rng as base
from two_layer_ctmc.simulators.micro import MicroSimulator


SOURCE_NAME = "n1000_random_connected_tree_augmented_aspl_step025_rep400_random_initial"
OUTPUT_NAME = SOURCE_NAME + "_full_corrected"
SOURCE = ROOT / "output" / "data" / SOURCE_NAME
OUTPUT = ROOT / "output" / "data" / OUTPUT_NAME
REPAIR_VERSION = "full_random_initial_v1"
EXPECTED_PAIRS = 17 * 40 * 10 * 2  # ASPL cells x graphs x interfaces x k values
DEFAULT_WORKERS = 4


def source_paths() -> list[Path]:
    parameters = json.loads((SOURCE / "experiment_parameters.json").read_text(encoding="utf-8"))
    expected_targets = [2.0 + 0.25 * index for index in range(17)]
    required = {
        "ASPL_targets": expected_targets,
        "community_size": 1000,
        "k_values": [1, 5],
        "beta": 0.003,
        "gamma": 0.0,
        "model": "SI",
        "T_max": 30000.0,
        "graph_lineages": 40,
        "interfaces_per_lineage": 10,
        "event_replicates_per_interface": 1,
        "independent_rng_per_micro_model": True,
    }
    for key, value in required.items():
        if parameters.get(key) != value:
            raise RuntimeError(f"Source parameter {key} differs from the repair design")
    paths = sorted((SOURCE / "scenarios").glob("*/checkpoints/*.json.gz"))
    if len(paths) != EXPECTED_PAIRS:
        raise RuntimeError(f"Expected {EXPECTED_PAIRS} source checkpoints, found {len(paths)}")
    return paths


def output_path(source: Path) -> Path:
    return OUTPUT / source.relative_to(SOURCE)


def source_rows(payload: dict) -> tuple[dict, dict]:
    rows = payload.get("runs", [])
    if len(rows) != 2:
        raise ValueError("Source checkpoint does not contain exactly two simulator runs")
    by_name = {row.get("simulator"): row for row in rows}
    if set(by_name) != {base.FULL, base.MICROMACRO}:
        raise ValueError("Source checkpoint has unexpected simulator names")
    return by_name[base.FULL], by_name[base.MICROMACRO]


def check_source(source: Path, payload: dict) -> tuple[dict, dict]:
    if payload.get("experiment") != SOURCE_NAME or payload.get("T_max") != base.T_MAX:
        raise ValueError(f"Unexpected experiment or horizon in {source}")
    task = payload.get("task", {})
    initial = task.get("initial_node")
    if not isinstance(initial, int) or not 0 <= initial < base.N:
        raise ValueError(f"No valid random initial node in {source}")
    if payload.get("rng_policy", {}).get("independent_rng_per_MicroModel") is not True:
        raise ValueError(f"Not an independent-RNG checkpoint: {source}")
    seeds = payload.get("seeds", {})
    if not isinstance(seeds.get("full_event_seed"), int) or not isinstance(seeds.get("micromacro_event_seed"), int):
        raise ValueError(f"Event seeds missing in {source}")
    if task.get("k") != len(task.get("bridges", [])) or task.get("k") not in (1, 5):
        raise ValueError(f"Invalid interface in {source}")
    graph_path = (ROOT / task["graph_path"]).resolve()
    if not graph_path.is_relative_to((SOURCE / "graphs").resolve()) or not graph_path.is_file():
        raise ValueError(f"Graph missing or outside source folder: {graph_path}")
    return source_rows(payload)


def load_source_graph(task: dict) -> nx.Graph:
    path = ROOT / task["graph_path"]
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        saved = json.load(handle)
    if saved.get("experiment") != SOURCE_NAME:
        raise ValueError(f"Unexpected graph provenance: {path}")
    graph = nx.Graph()
    graph.add_nodes_from(range(base.N))
    graph.add_weighted_edges_from(saved["edges"])
    if graph.number_of_nodes() != base.N or graph.number_of_edges() != saved["descriptor"]["m"]:
        raise ValueError(f"Incomplete graph: {path}")
    return graph


def run_full(payload: dict, initial_node: int) -> dict:
    task = payload["task"]
    graph = load_source_graph(task)
    _, full_graph, _ = base.make_network(graph, task["bridges"])
    started = time.perf_counter()
    result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=[list(range(base.N)), [base.N]],
        infection_rate=base.BETA,
        recovery_rate=base.GAMMA,
        model=base.MODEL,
    ).run(
        T_end=base.T_MAX,
        dt_out=base.T_MAX,
        initial_node=initial_node,
        rng=random.Random(payload["seeds"]["full_event_seed"]),
        stop_at_absorption=True,
    )
    runtime = time.perf_counter() - started
    observed = base.observables(base.infection_times_full(result, initial_node), task["bridges"])
    if observed["bypass"]:
        raise RuntimeError("Full Micro exported before a physical bridge was infected")
    return {"simulator": base.FULL, "runtime_seconds": runtime,
            "termination": result.get("run_metadata"), **observed}


def regression_gate(paths: list[Path]) -> None:
    """Verify current Full reproduces the archived node-0 simulation."""
    for path in (paths[0], paths[len(paths) // 2], paths[-1]):
        payload = read_checkpoint(path)
        old_full, _ = check_source(path, payload)
        replay = run_full(payload, 0)
        if (replay["T_export"] != old_full["T_export"]
                or replay["termination"] != old_full["termination"]):
            raise RuntimeError(f"Regression gate FAIL: archived Full is not reproduced from node 0: {path}")
    print("Regression gate: PASS (3 archived Full runs replay exactly from node 0)", flush=True)


def repaired_checkpoint(source: Path) -> dict:
    payload = read_checkpoint(source)
    old_full, micromacro = check_source(source, payload)
    initial = payload["task"]["initial_node"]
    full = old_full if initial == 0 else run_full(payload, initial)
    return {"experiment": OUTPUT_NAME, "task": payload["task"],
            "T_max": payload["T_max"], "seeds": payload["seeds"],
            "rng_policy": payload["rng_policy"],
            "repair": {"version": REPAIR_VERSION, "source_experiment": SOURCE_NAME,
                       "source_checkpoint": str(source.relative_to(ROOT)),
                       "old_full_initial_node": 0, "corrected_initial_node": initial,
                       "full_rerun": initial != 0, "micromacro_reused": True},
            "runs": [full, micromacro]}


def valid_output(source: Path) -> bool:
    destination = output_path(source)
    if not destination.exists():
        return False
    original = read_checkpoint(source)
    check_source(source, original)
    repaired = read_checkpoint(destination)
    full, mm = source_rows(repaired)
    return (repaired.get("experiment") == OUTPUT_NAME
            and repaired.get("repair", {}).get("version") == REPAIR_VERSION
            and repaired["repair"].get("source_checkpoint") == str(source.relative_to(ROOT))
            and repaired["repair"].get("corrected_initial_node") == original["task"]["initial_node"]
            and repaired.get("task") == original["task"]
            and repaired.get("seeds") == original["seeds"]
            and repaired.get("T_max") == original["T_max"]
            and repaired.get("rng_policy") == original["rng_policy"]
            and mm == source_rows(original)[1]
            and full.get("simulator") == base.FULL
            and isinstance(full.get("export_observed"), bool)
            and "termination" in full)


def write_fixed_parameters() -> None:
    source_parameters = json.loads((SOURCE / "experiment_parameters.json").read_text(encoding="utf-8"))
    parameters = dict(source_parameters)
    parameters["output_folder"] = str(OUTPUT.relative_to(ROOT))
    parameters["source_folder"] = str(SOURCE.relative_to(ROOT))
    parameters["repair"] = "Full Micro rerun from each saved random initial node; original independent-RNG MicroMacro row reused exactly"
    base.ensure_fixed_json(OUTPUT / "experiment_parameters.json", parameters, "corrected experiment parameters")
    for source_scenario in sorted((SOURCE / "scenarios").iterdir()):
        if not source_scenario.is_dir():
            continue
        scenario_parameters = json.loads((source_scenario / "parameters.json").read_text(encoding="utf-8"))
        scenario_parameters["repair"] = "Full Micro corrected; original MicroMacro reused"
        destination = OUTPUT / "scenarios" / source_scenario.name / "parameters.json"
        base.ensure_fixed_json(destination, scenario_parameters, "corrected scenario parameters")


def write_run_manifest(total: int, completed: int) -> None:
    base.atomic_json(OUTPUT / "run_manifest.json", {
        "experiment": OUTPUT_NAME,
        "source_experiment": SOURCE_NAME,
        "repair_version": REPAIR_VERSION,
        "expected_pairs": total,
        "completed_pairs": completed,
        "remaining_pairs": total - completed,
        "status": "complete" if completed == total else "incomplete",
        "source_graphs_required": True,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, help="repair at most this many pending pairs (small test)")
    parser.add_argument("--check-only", action="store_true", help="run the regression gate without writing data")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    paths = source_paths()
    regression_gate(paths)
    if args.check_only:
        print(f"Source pairs: {len(paths)}; no data written", flush=True)
        return
    pending = [path for path in paths if not valid_output(path)]
    completed = len(paths) - len(pending)
    write_fixed_parameters()
    write_run_manifest(len(paths), completed)
    print(f"[resume] reused={completed}/{len(paths)}; pending={len(pending)}", flush=True)
    if args.limit is not None:
        pending = pending[:args.limit]
    if not pending:
        return
    remaining = iter(pending)
    active: dict[concurrent.futures.Future, Path] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            while len(active) < 2 * args.workers:
                try:
                    source = next(remaining)
                except StopIteration:
                    break
                active[pool.submit(repaired_checkpoint, source)] = source
            if not active:
                break
            future = next(concurrent.futures.as_completed(active))
            source = active.pop(future)
            repaired = future.result()
            write_checkpoint(output_path(source), repaired)
            completed += 1
            if completed % 100 == 0 or completed == len(paths) or args.limit is not None:
                write_run_manifest(len(paths), completed)
                print(f"[progress] {completed}/{len(paths)}; remaining={len(paths)-completed}", flush=True)
    write_run_manifest(len(paths), completed)
    print(f"Data saved to: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
