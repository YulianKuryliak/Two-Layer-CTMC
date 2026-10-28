#!/usr/bin/env python3
"""Freeze and rerun the exact paired experiments in claude-version_merged.tex.

No simulation is performed without --run. Preparation reads historical data,
freezes inputs and sources, and checks the full 46,000-pair design. Resume is
automatic, but only checkpoints with matching input and code hashes are reused.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extrascripts.runners.checkpoint_io import read_checkpoint, write_checkpoint
from two_layer_ctmc.orchestrators.micromacro import Orchestrator
from two_layer_ctmc.simulators.micro import MicroSimulator

DEFAULT_OUTPUT = ROOT / "output/paper_rerun_residual_clock_20261007"
PAPER = ROOT / "paper/SMPT paper/claude-version_merged.tex"
VERSION = "smpt_frozen_pairs_residual_clock_v1"
FULL, MM = "Full Micro", "Production MicroMacro + independent RNG"
COUNTS = {"clique_leaf": 10000, "source_leaf": 16800, "chain": 19200}
SOURCE_LEAF_ROOTS = (
    "n1000_random_connected_tree_augmented_aspl1to1p75_step025_rep400_random_initial",
    "n1000_random_connected_tree_augmented_aspl_step025_rep400_random_initial_full_corrected",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(output, event, **fields):
    row = {"utc": utc(), "runner": VERSION, "output": str(output), "event": event, **fields}
    # Only the parent calls this; flock also protects the central log from other runners.
    for path in (output / "experiments.jsonl", ROOT / "logs/experiments.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@contextmanager
def exclusive_output(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another runner is writing this output folder") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} utc={utc()}\n")
        handle.flush()
        yield


def code_hashes():
    paths = sorted((ROOT / "two_layer_ctmc").rglob("*.py"))
    paths += [Path(__file__), ROOT / "extrascripts/runners/checkpoint_io.py",
              ROOT / "extrascripts/runners/run_original_benchmarks_independent_rng.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def source_metadata(path):
    """Read the small root metadata before runs, without decoding event arrays."""
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        result = {}
        while True:
            chunk = handle.read(65536)
            if not chunk:
                raise ValueError(f"Incomplete checkpoint metadata: {path}")
            buffer += chunk
            if not buffer:
                raise ValueError(f"Empty source checkpoint: {path}")
            if position == 0:
                position = buffer.index("{") + 1
            while True:
                while position < len(buffer) and buffer[position] in " \r\n\t,":
                    position += 1
                start = position
                try:
                    key, position = decoder.raw_decode(buffer, position)
                    while buffer[position] in " \r\n\t:":
                        position += 1
                    if key == "runs":
                        return result
                    value, position = decoder.raw_decode(buffer, position)
                except (json.JSONDecodeError, IndexError):
                    position = start
                    break
                result[key] = value


def snapshot_graph(output, relative, cache):
    if relative in cache:
        return cache[relative]
    source = ROOT / relative
    sha = digest(source)
    destination = output / "inputs/graphs" / (sha + ".json.gz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        if digest(temporary) != sha:
            raise RuntimeError(f"Graph changed while being frozen: {source}")
        os.replace(temporary, destination)
    elif digest(destination) != sha:
        raise RuntimeError(f"Corrupt frozen graph: {destination}")
    reference = {"source": relative, "sha256": sha, "snapshot": str(destination.relative_to(output))}
    cache[relative] = reference
    return reference


def prepare(output):
    current = code_hashes()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["code_sha256"] != current or manifest["paper_sha256"] != digest(PAPER):
            raise RuntimeError("Code/manuscript changed since preparation. Choose a new --output directory.")
        if manifest["python"] != sys.version or manifest["packages"] != {
                name: importlib.metadata.version(name) for name in ("numpy", "networkx")}:
            raise RuntimeError("Python/package versions changed; choose a new output directory")
        tasks = read_checkpoint(output / "tasks.json.gz")["tasks"]
        if object_hash(tasks) != manifest["tasks_sha256"]:
            raise RuntimeError("Frozen task manifest is corrupt")
        for ref in manifest["graphs"]:
            if digest(output / ref["snapshot"]) != ref["sha256"]:
                raise RuntimeError("Frozen graph input changed")
        log(output, "manifest_reused", pairs=len(tasks))
        return manifest, tasks
    log(output, "prepare_started")
    tasks, graphs = [], {}
    for series in COUNTS:
        if series == "clique_leaf":
            folder = ROOT / "output/main_clean_benchmarks_independent_rng/.checkpoints/clique_leaf_n100_k2"
            paths = sorted(folder.glob("*.json.gz"))
        elif series == "source_leaf":
            paths = sorted(path for name in SOURCE_LEAF_ROOTS for path in
                           (ROOT / "output/data" / name).glob("scenarios/*/checkpoints/*.json.gz"))
        else:
            paths = sorted(path for n in (50, 100, 250, 1000) for path in
                           (ROOT / "output/data" / f"l10_n{n}_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng").glob("scenarios/*/checkpoints/*.json.gz"))
        if len(paths) != COUNTS[series]:
            raise RuntimeError(f"{series}: expected {COUNTS[series]} original pairs, found {len(paths)}")
        for index, path in enumerate(paths):
            old = source_metadata(path)
            if series == "clique_leaf":
                cfg = old["benchmark_config"]
                replication = int(old["replication_id"])
                network_seed = cfg["network_seed_base"] + replication - 1
                event_seed = cfg["base_seed"] + replication - 1
                original_task = {"replication_id": replication, "network_seed": network_seed,
                                 "initial_node": cfg["initial_node"], "k": 2, "target_ASPL": 1.0}
                # In the saved star/clique generator, clique construction consumes
                # no random draws. Interface choices are the first RNG operation.
                bridges = random.Random(network_seed).sample(range(100), k=2)
                sizes, beta, horizon = [100, 1], .005, 1000.
                references, interfaces = [], [[node, 100] for node in bridges]
                seeds = {"full_event_seed": event_seed, "micromacro_event_seed": event_seed}
                scenario = "clique_leaf_n100_k2"
            else:
                original_task = old["task"]
                seeds = old.get("event_seeds", old.get("seeds"))
                if not seeds or not all(isinstance(seeds.get(key), int) for key in
                                         ("full_event_seed", "micromacro_event_seed")):
                    raise RuntimeError(f"Missing event seeds: {path}")
                beta, horizon = .003, float(old["T_max"])
                scenario = path.parent.parent.name
                if series == "source_leaf":
                    sizes = [1000, 1]
                    references = [snapshot_graph(output, original_task["graph_path"], graphs)]
                    interfaces = [[node, 1000] for node in original_task["bridges"]]
                else:
                    import re
                    n = int(re.search(r"__n(\d+)__", scenario)[1])
                    sizes = [n] * 10
                    references = [snapshot_graph(output, relative, graphs) for relative in original_task["graph_paths"]]
                    interfaces = original_task["interface_edges"]
            extended = path.parent.parent / "checkpoints_full_to40000" / path.name
            extension = None
            if series == "chain" and extended.exists():
                extension = {"source": str(extended.relative_to(ROOT)), "sha256": digest(extended)}
            task = {"id": f"{series}/{scenario}/{path.stem.removesuffix('.json')}", "series": series,
                    "scenario": scenario, "source_checkpoint": str(path.relative_to(ROOT)),
                    "source_sha256": digest(path), "original_task": original_task,
                    "graph_inputs": references, "community_sizes": sizes, "interface_edges": interfaces,
                    "event_seeds": seeds, "initial_node": int(original_task.get("initial_node", 0)),
                    "beta": beta, "gamma": 0., "model": 2, "tau_micro": 1., "macro_T": 1.,
                    "T_max": horizon, "full_horizon": 40000. if extension else horizon,
                    "full_extension": extension}
            task["input_sha256"] = object_hash(task)
            tasks.append(task)
            if (index + 1) % 1000 == 0:
                print(f"prepare {series}: {index + 1}/{len(paths)}", flush=True)
    cells = {}
    for task in tasks:
        key = (task["series"], task["scenario"])
        cells[key] = cells.get(key, 0) + 1
    if len([key for key in cells if key[0] == "source_leaf"]) != 42 or len([key for key in cells if key[0] == "chain"]) != 48:
        raise RuntimeError("The frozen experiment cells differ from the article design")
    if any(count != (10000 if series == "clique_leaf" else 400) for (series, _), count in cells.items()):
        raise RuntimeError("Wrong replication count in an article cell")
    if sum(task["full_extension"] is not None for task in tasks) != 2:
        raise RuntimeError("Expected exactly two historical Full horizon extensions")
    for relative, sha in current.items():
        source, target = ROOT / relative, output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(target) != sha:
            raise RuntimeError("Code changed during preparation")
    target = output / "sources" / PAPER.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAPER, target)
    manifest = {"version": VERSION, "created_utc": utc(), "status": "prepared", "expected_pairs": sum(COUNTS.values()),
                "series_pairs": COUNTS, "clique_chain_subset": "chain n1000 ASPL1 k1, reused rather than duplicated",
                "cells": [{"series": key[0], "scenario": key[1], "pairs": count} for key, count in sorted(cells.items())],
                "code_sha256": current, "code_fingerprint": object_hash(current), "paper_sha256": digest(PAPER),
                "tasks_sha256": object_hash(tasks), "graphs": list(graphs.values()),
                "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("numpy", "networkx")},
                "rng_policy": {"cache_policy": "residual_intensity_budget_v1", "independent_rng_per_MicroModel": True,
                               "seed_formula": "micromacro_event_seed + 10000 + 1009 * community", "native_clone_refresh": True},
                "full_trajectory_policy": "raw typed events and all community state change points, no resampling",
                "historical_full_extensions": [task["full_extension"] for task in tasks if task["full_extension"]]}
    write_checkpoint(output / "tasks.json.gz", {"tasks": tasks})
    atomic_json(manifest_path, manifest)
    atomic_json(output / "run_status.json", {"status": "prepared", "expected_full_pairs": sum(COUNTS.values()),
                "full_design_complete": False, "code_fingerprint": manifest["code_fingerprint"]})
    log(output, "prepare_completed", pairs=len(tasks), code_fingerprint=manifest["code_fingerprint"])
    return manifest, tasks


def network(task, output):
    sizes = task["community_sizes"]
    full = nx.Graph()
    communities, micro = [], []
    offset = 0
    for index, n in enumerate(sizes):
        nodes = list(range(offset, offset + n))
        full.add_nodes_from(nodes)
        if task["series"] == "clique_leaf" and index == 0:
            full.add_edges_from((u, v, {"weight": 1.}) for u in nodes for v in nodes if u < v)
        elif task["graph_inputs"] and index < len(task["graph_inputs"]):
            saved = read_checkpoint(output / task["graph_inputs"][index]["snapshot"])
            full.add_edges_from((int(u) + offset, int(v) + offset, {"weight": float(w)}) for u, v, w in saved["edges"])
        communities.append(nodes)
        # Preserve insertion order from the historical full graph; a node_filter
        # without a .nodes attribute makes NetworkX iterate the parent order.
        selected = set(nodes)
        micro.append(nx.subgraph_view(full, filter_node=lambda node, selected=selected: node in selected))
        offset += n
    full.add_edges_from((int(u), int(v), {"weight": 1.}) for u, v in task["interface_edges"])
    lookup = {node: c for c, nodes in enumerate(communities) for node in nodes}
    W = np.zeros((len(sizes), len(sizes)))
    boundary = [0] * len(sizes)
    for u, v in task["interface_edges"]:
        a, b = lookup[u], lookup[v]
        W[a, b] += 1
        W[b, a] += 1
        boundary[a] += 1
        boundary[b] += 1
    full.graph.update(community_sizes=sizes, alphas=[1.] * len(sizes), boundary_counts=boundary,
                      mu2_values=[1.] * len(sizes), spectral_descriptors_computed=False)
    return micro, full, W, communities


def summarize_events(raw, sizes, full, horizon, bridges):
    events = []
    for event in raw:
        if full:
            kind = event.get("kind")
            if kind not in {"seed", "intra", "inter"} or event.get("dst_node") is None:
                continue
            t, c, node = float(event["time"]), int(event["dst_community"]), int(event["dst_node"])
            macro = kind == "inter"
        else:
            if event.get("event_type") not in {"seed", "infection", "transfer"} or event.get("node") is None:
                continue
            t, c, node = float(event["time"]), int(event["community"]), int(event["node"])
            macro = event["event_type"] == "transfer"
        if t <= horizon:
            events.append((t, c, node, macro))
    events.sort(key=lambda row: row[0])
    arrivals = [None] * len(sizes)
    infected, times, repeats = set(), {}, []
    successful = 0
    for t, c, node, macro in events:
        if node in infected:
            raise RuntimeError("An SI node was infected more than once")
        if macro:
            successful += 1
            if arrivals[c] is not None and arrivals[c] < t:
                repeats.append((t, c, node))
        infected.add(node)
        times[node] = t
        if arrivals[c] is None:
            arrivals[c] = t
    terminal = arrivals[-1]
    initial = [int(n) for n in sizes]
    trajectories = {str(c): {"times": [0.], "S": [n], "I": [0], "R": [0]} for c, n in enumerate(initial)}
    counts = [0] * len(sizes)
    for t, c, _, _ in events:
        counts[c] += 1
        logrow = trajectories[str(c)]
        logrow["times"].append(t)
        logrow["S"].append(sizes[c] - counts[c])
        logrow["I"].append(counts[c])
        logrow["R"].append(0)
    result = {"infection_events": [[t, c, node] for t, c, node, _ in events], "arrivals": arrivals,
              "deltas": [None] + [None if arrivals[c] is None or arrivals[c-1] is None else arrivals[c]-arrivals[c-1] for c in range(1, len(sizes))],
              "T_last": max(arrivals) if all(t is not None for t in arrivals) else None,
              "completion": all(t is not None for t in arrivals), "change_point_trajectories": trajectories,
              "successful_imports": successful, "repeat_imports": len(repeats),
              "repeat_imports_to_endpoint": sum(t <= terminal for t, _, _ in repeats) if terminal is not None else None,
              "endpoint": "T9" if len(sizes) == 10 else "first_leaf_infection", "endpoint_time": terminal,
              "repeat_import_events": [[t, c, node] for t, c, node in repeats]}
    if len(sizes) == 2:
        before = sum(times.get(node, float("inf")) < terminal for node in bridges) if terminal is not None else None
        result.update(T_export=terminal, export_observed=terminal is not None,
                      T_bridge_first=min((times[node] for node in bridges if node in times), default=None),
                      bridge_infected_count_before_export=before, bypass=terminal is not None and before == 0,
                      I_source_before_export=sum(c == 0 and t < terminal for t, c, _, _ in events) if terminal is not None else None)
    return result


def simulate_task(task, output_string, code_fingerprint):
    output = Path(output_string)
    micro, full_graph, W, communities = network(task, output)
    seeds, sizes = task["event_seeds"], task["community_sizes"]
    bridges = [u for u, _ in task["interface_edges"]] if len(sizes) == 2 else []
    started = time.perf_counter()
    full_result = MicroSimulator(full_graph, communities, task["beta"], 0., model=2).run(
        T_end=task["full_horizon"], dt_out=task["full_horizon"], initial_node=task["initial_node"],
        rng=random.Random(seeds["full_event_seed"]), stop_at_absorption=True)
    raw_full = full_result["transmission_events"]
    full_run = {"simulator": FULL, "runtime_seconds": time.perf_counter()-started,
                "T_end": task["full_horizon"], "termination": full_result["run_metadata"], "typed_events": raw_full,
                **summarize_events(raw_full, sizes, True, task["full_horizon"], bridges)}
    full_run["physical_interface_valid"] = all(e["kind"] != "inter" or full_graph.has_edge(e["src_node"], e["dst_node"]) for e in raw_full)
    if not full_run["physical_interface_valid"]:
        raise RuntimeError("Full transmission outside frozen physical interface")
    del full_result
    gc.collect()
    started = time.perf_counter()
    event_seed = seeds["micromacro_event_seed"]
    random.seed(event_seed)
    np.random.seed(event_seed)
    orch = Orchestrator(W, micro, task["beta"], 0., task["beta"], 1., task["T_max"],
                        macro_T=1., model=2, community_sizes=sizes, alphas=[1.]*len(sizes),
                        full_graph=full_graph, dense_logs=False, stop_at_absorption=True,
                        micro_rngs=[random.Random(event_seed + 10000 + 1009*c) for c in range(len(sizes))])
    orch.micro_models[0]._infect_node(task["initial_node"])
    orch.event_log.append({"time": 0., "event_type": "seed", "mode": "seed", "community": 0,
                           "node": task["initial_node"], "src": None})
    result = orch.run()
    mm_run = {"simulator": MM, "runtime_seconds": time.perf_counter()-started, "T_end": task["T_max"],
              "termination": orch.run_metadata, "typed_events": result[3],
              **summarize_events(result[3], sizes, False, task["T_max"], bridges),
              "orchestrator_change_point_logs": result[2]}
    payload = {"version": VERSION, "task": task["original_task"], "rerun_task": task, "T_max": task["T_max"],
               "event_seeds": seeds, "input_sha256": task["input_sha256"], "code_fingerprint": code_fingerprint,
               "completed_utc": utc(), "status": "complete", "runs": [full_run, mm_run]}
    destination = output / "checkpoints" / (task["id"] + ".json.gz")
    write_checkpoint(destination, payload)
    return {"task_id": task["id"], "checkpoint": str(destination.relative_to(output)),
            "sha256": digest(destination), "full_runtime_s": full_run["runtime_seconds"],
            "mm_runtime_s": mm_run["runtime_seconds"], "mm_repeat_imports": mm_run["repeat_imports"],
            "mm_repeat_imports_to_endpoint": mm_run["repeat_imports_to_endpoint"]}


def valid_existing(output, task, fingerprint):
    path = output / "checkpoints" / (task["id"] + ".json.gz")
    if not path.exists():
        return False
    payload = read_checkpoint(path)
    if (payload.get("version") != VERSION or payload.get("input_sha256") != task["input_sha256"]
            or payload.get("code_fingerprint") != fingerprint or payload.get("status") != "complete"):
        raise RuntimeError(f"Existing checkpoint provenance differs; refusing to overwrite {path}")
    runs = payload.get("runs", [])
    if len(runs) != 2 or [run.get("simulator") for run in runs] != [FULL, MM]:
        raise RuntimeError(f"Incomplete checkpoint: {path}")
    if any(not all(key in run for key in ("typed_events", "change_point_trajectories", "termination", "repeat_imports")) for run in runs):
        raise RuntimeError(f"Required event/trajectory data missing: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", "--dry-run", action="store_true", help="freeze full design; no simulation")
    action.add_argument("--run", action="store_true", help="prepare if needed, then run or resume")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1, help="explicit user-selected concurrency; dense clique chain always serial")
    parser.add_argument("--series", choices=list(COUNTS), nargs="+", default=list(COUNTS))
    parser.add_argument("--limit", type=int, help="maximum selected pairs for a smoke run; full manifest still frozen")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    output = args.output.resolve()
    if output == ROOT / "output/data" or output.is_relative_to(ROOT / "output/data"):
        parser.error("output must be separate from historical output/data")
    with exclusive_output(output):
        try:
            manifest, tasks = prepare(output)
        except BaseException as error:
            log(output, "prepare_failed", error=repr(error), traceback=traceback.format_exc())
            raise
        if args.prepare_only:
            print(f"Prepared {len(tasks)} pairs in {output}; no simulations started.")
            return
        selected = [task for task in tasks if task["series"] in args.series]
        if args.limit:
            selected = selected[:args.limit]
        fingerprint = manifest["code_fingerprint"]
        pending, skipped = [], 0
        for task in selected:
            if valid_existing(output, task, fingerprint):
                skipped += 1
                log(output, "pair_resumed_skip", task_id=task["id"])
            else:
                pending.append(task)
        log(output, "run_started", selected_pairs=len(selected), pending_pairs=len(pending), resumed_pairs=skipped,
            workers=args.workers, series=args.series, limit=args.limit,
            code_fingerprint=fingerprint, tasks_sha256=manifest["tasks_sha256"], command=sys.argv)
        atomic_json(output / "run_status.json", {"status": "running", "started_utc": utc(),
                    "selected_pairs": len(selected), "resumed_pairs": skipped, "completed_this_invocation": 0})
        finished = 0
        current_task_id = None
        try:
            def accepted(task, result):
                nonlocal finished
                finished += 1
                log(output, "pair_completed", **result)
                atomic_json(output / "run_status.json", {"status": "running", "updated_utc": utc(),
                            "selected_pairs": len(selected), "resumed_pairs": skipped, "completed_this_invocation": finished})
                print(f"{finished}/{len(pending)} completed; {task['id']}", flush=True)
            serial = [task for task in pending if task["series"] == "chain" and task["original_task"]["target_ASPL"] == 1.]
            serial_ids = {task["id"] for task in serial}
            parallel = [task for task in pending if task["id"] not in serial_ids]
            for task in serial:
                current_task_id = task["id"]
                log(output, "pair_started", task_id=task["id"], mode="serial_dense_chain")
                accepted(task, simulate_task(task, str(output), fingerprint))
            if args.workers == 1:
                for task in parallel:
                    current_task_id = task["id"]
                    log(output, "pair_started", task_id=task["id"])
                    accepted(task, simulate_task(task, str(output), fingerprint))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                    iterator = iter(parallel)
                    active = {}
                    while True:
                        while len(active) < args.workers:
                            task = next(iterator, None)
                            if task is None:
                                break
                            log(output, "pair_started", task_id=task["id"])
                            active[pool.submit(simulate_task, task, str(output), fingerprint)] = task
                        if not active:
                            break
                        done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            task = active.pop(future)
                            current_task_id = task["id"]
                            accepted(task, future.result())
            full_scope = len(selected) == manifest["expected_pairs"]
            completion = {"status": "complete" if full_scope else "subset_complete", "finished_utc": utc(),
                          "selected_pairs": len(selected), "resumed_pairs": skipped, "completed_this_invocation": finished,
                          "expected_full_pairs": manifest["expected_pairs"], "full_design_complete": full_scope,
                          "series": args.series, "limit": args.limit, "code_fingerprint": fingerprint}
            atomic_json(output / "run_status.json", completion)
            log(output, "run_completed", **completion)
        except BaseException as error:
            failed = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "utc": utc(),
                      "error": repr(error), "task_id": current_task_id, "traceback": traceback.format_exc(),
                      "completed_this_invocation": finished, "selected_pairs": len(selected)}
            atomic_json(output / "run_status.json", failed)
            log(output, "run_failed", **failed)
            raise


if __name__ == "__main__":
    main()
