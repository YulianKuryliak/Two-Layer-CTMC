#!/usr/bin/env python3
"""Freeze the article's 19,200 chain inputs, reuse Full, and rerun only TwoScale.

No simulation without --run. No Full simulator is called. Changed source/code/
environment fingerprints require a new output folder; complete matching tasks
are skipped on resume. Historical results remain read-only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback

import numpy as np

COPY_ROOT = Path(__file__).resolve().parents[2]
ROOT = COPY_ROOT.parents[2]
sys.path.insert(0, str(COPY_ROOT))
from extrascripts.runners.checkpoint_io import read_checkpoint, write_checkpoint
from extrascripts.runners.rerun_smpt_paper import (
    atomic_json, digest, exclusive_output, log as shared_log, network,
    object_hash, snapshot_graph, source_metadata, summarize_events, utc,
)
from two_layer_ctmc.orchestrators.micromacro import Orchestrator

VERSION = "chain_twoscale_state_events_optimized_v1"
FULL = "Full Micro"
MM = "Midpoint TwoScale + residual clocks + independent RNG"
DEFAULT_OUTPUT = ROOT / "output/chain_twoscale_state_events_v1_20261008"
EXPECTED_RUNS = 19200


def log(output, event, **fields):
    shared_log(output, event, runner=VERSION, **fields)


def code_hashes():
    paths = sorted((COPY_ROOT / "two_layer_ctmc").rglob("*.py"))
    paths += [Path(__file__), COPY_ROOT / "extrascripts/runners/checkpoint_io.py",
              COPY_ROOT / "extrascripts/runners/rerun_smpt_paper.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def environment():
    return {"python": sys.version,
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "networkx")}}


def verify_reference(reference):
    path = ROOT / reference["source"]
    if digest(path) != reference["sha256"]:
        raise RuntimeError(f"Historical source changed: {path}. Choose a new output directory.")
    return path


def prepare(output):
    current = code_hashes()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != VERSION or manifest["code_sha256"] != current:
            raise RuntimeError("Runner/code changed. Choose a new --output directory.")
        if manifest["environment"] != environment():
            raise RuntimeError("Python/packages changed. Choose a new --output directory.")
        tasks = read_checkpoint(output / "tasks.json.gz")["tasks"]
        if object_hash(tasks) != manifest["tasks_sha256"]:
            raise RuntimeError("Frozen task manifest is corrupt")
        for reference in manifest["graphs"]:
            if digest(output / reference["snapshot"]) != reference["sha256"]:
                raise RuntimeError("Frozen graph input changed")
        for relative, sha in current.items():
            if digest(output / "sources" / relative) != sha:
                raise RuntimeError("Frozen source snapshot changed")
        # Validate every historical baseline, including already completed tasks.
        for task in tasks:
            verify_reference(task["source_checkpoint"])
            if task["full_extension"]:
                verify_reference(task["full_extension"])
        log(output, "manifest_reused", trajectories=len(tasks),
            code_fingerprint=manifest["code_fingerprint"])
        return manifest, tasks

    log(output, "prepare_started", simulation=False, code_fingerprint=object_hash(current))
    tasks, graphs, cells = [], {}, {}
    for n in (50, 100, 250, 1000):
        root = ROOT / "output/data" / (
            f"l10_n{n}_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng")
        paths = sorted(root.glob("scenarios/*/checkpoints/*.json.gz"))
        if len(paths) != 4800:
            raise RuntimeError(f"n={n}: expected 4,800 saved pairs; found {len(paths)}")
        for index, path in enumerate(paths):
            old = source_metadata(path)
            original = old["task"]
            seeds = old.get("event_seeds", old.get("seeds"))
            if not seeds or not all(isinstance(seeds.get(key), int) for key in
                                   ("full_event_seed", "micromacro_event_seed")):
                raise RuntimeError(f"Missing deterministic seeds: {path}")
            scenario = path.parent.parent.name
            match = re.search(r"__n(\d+)__", scenario)
            if match is None or int(match[1]) != n:
                raise RuntimeError(f"Unexpected chain scenario: {scenario}")
            aspl, k = float(original["target_ASPL"]), int(original["k"])
            if aspl not in (1., 2., 3., 4., 5., 6.) or k not in (1, 5):
                raise RuntimeError(f"Unexpected article cell: {scenario}")
            if float(old["T_max"]) != 30000.:
                raise RuntimeError(f"Unexpected MicroMacro horizon: {path}")
            refs = [snapshot_graph(output, relative, graphs) for relative in original["graph_paths"]]
            if len(refs) != 10:
                raise RuntimeError(f"Expected ten frozen internal graphs: {path}")
            extension_path = path.parent.parent / "checkpoints_full_to40000" / path.name
            extension = ({"source": str(extension_path.relative_to(ROOT)), "sha256": digest(extension_path)}
                         if extension_path.exists() else None)
            task = {"id": f"chain/{scenario}/{path.stem.removesuffix('.json')}",
                    "series": "chain", "scenario": scenario,
                    "source_checkpoint": {"source": str(path.relative_to(ROOT)), "sha256": digest(path)},
                    "original_task": original, "graph_inputs": refs, "community_sizes": [n] * 10,
                    "interface_edges": original["interface_edges"], "event_seeds": seeds,
                    "initial_node": int(original.get("initial_node", 0)),
                    "beta": .003, "gamma": 0., "model": 2, "T_max": 30000.,
                    "full_horizon": 40000. if extension else 30000., "full_extension": extension,
                    "tau_micro": 1., "macro_T": 1.,
                    "scheduler": "original_blackbox_midpoint_tau1", "macro_placement": "uniform_susceptible"}
            task["input_sha256"] = object_hash(task)
            tasks.append(task)
            cell = (n, aspl, k)
            cells[cell] = cells.get(cell, 0) + 1
            if (index + 1) % 1000 == 0:
                print(f"prepare n={n}: {index + 1}/{len(paths)}", flush=True)
    if len(tasks) != EXPECTED_RUNS or len(cells) != 48 or any(value != 400 for value in cells.values()):
        raise RuntimeError("Frozen design differs from the 48 × 400 article chain experiments")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise RuntimeError("Duplicate chain task identifiers")
    extensions = [task["full_extension"] for task in tasks if task["full_extension"]]
    if len(extensions) != 2:
        raise RuntimeError("Expected exactly two historical Full horizon extensions")
    for relative, sha in current.items():
        target = output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if digest(target) != sha:
            raise RuntimeError("Code changed during preparation")
    if code_hashes() != current:
        raise RuntimeError("Code changed while preparing; use a new output directory")
    manifest = {"version": VERSION, "created_utc": utc(), "status": "prepared",
                "expected_twoscale_runs": EXPECTED_RUNS, "new_full_runs": 0,
                "cells": [{"n": key[0], "ASPL": key[1], "k": key[2], "runs": value}
                          for key, value in sorted(cells.items())],
                "code_sha256": current, "code_fingerprint": object_hash(current),
                "tasks_sha256": object_hash(tasks), "graphs": list(graphs.values()),
                "environment": environment(), "historical_full_extensions": extensions,
                "scheduler": "original black-box midpoint, tau_micro=1, macro_T=1; finite-step approximation retained",
                "macro_hazard": "beta_out * W[i,j] * I[i]/N[i] * S[j]/N[j]",
                "macro_placement": "uniform among susceptible destination nodes",
                "rng_policy": {"micro_clock": "residual_intensity_budget_v1", "rollback": False,
                               "micro_seed": "micromacro_event_seed + 10000 + 1009 * community",
                               "macro_seed": "micromacro_event_seed (random.seed and numpy.random.seed)",
                               "streams": "private random.Random per micro model; seeded worker-local module random for macro"},
                "trajectory_policy": "all typed TwoScale events and community change points; no resampling",
                "full_policy": "unchanged historical Full row loaded after TwoScale; provenance separate in payload; no fabricated typed logs",
                "horizon_policy": "TwoScale 30000; two historical Full runs 40000; record differing censoring explicitly",
                "interpretation": "Local residual clock and RNG corrected; original midpoint tau=1 retained. Aggregate rates, placement and finite-step synchronization remain approximations relative to physical Full CTMC"}
    write_checkpoint(output / "tasks.json.gz", {"tasks": tasks})
    atomic_json(manifest_path, manifest)
    atomic_json(output / "run_status.json", {"status": "prepared", "full_design_complete": False,
                "expected_twoscale_runs": EXPECTED_RUNS, "new_full_runs": 0,
                "code_fingerprint": manifest["code_fingerprint"]})
    log(output, "prepare_completed", trajectories=len(tasks), new_full_runs=0,
        code_fingerprint=manifest["code_fingerprint"], tasks_sha256=manifest["tasks_sha256"])
    return manifest, tasks


def simulate_task(task, output_string, fingerprint):
    output = Path(output_string)
    baseline_path = verify_reference(task["source_checkpoint"])
    micro, full_graph, W, _ = network(task, output)
    seed = task["event_seeds"]["micromacro_event_seed"]
    random.seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    orch = Orchestrator(
        W, micro, task["beta"], task["gamma"], task["beta"], task["tau_micro"], task["T_max"],
        macro_T=task["macro_T"], model=task["model"], community_sizes=task["community_sizes"], alphas=[1.] * 10,
        full_graph=full_graph, dense_logs=False, stop_at_absorption=True,
        micro_rngs=[random.Random(seed + 10000 + 1009 * c) for c in range(10)])
    # Models own node states and compact neighbors; input NetworkX views and
    # their parent graph are no longer read after Orchestrator construction.
    del micro, full_graph, _
    # NetworkX caches EdgeViews that form reference cycles with their graph.
    # Explicit collection makes the release effective before stepping begins.
    gc.collect()
    orch.micro_models[0]._infect_node(task["initial_node"])
    orch.event_log.append({"time": 0., "event_type": "seed", "mode": "seed", "community": 0,
                           "node": task["initial_node"], "src": None})
    result = orch.run()
    raw = result[3]
    summary = summarize_events(raw, task["community_sizes"], False, task["T_max"], [])
    # Native pre-import flags are authoritative. Chronological SI history is
    # only a fallback when a typed event lacks that field.
    seen = set()
    repeats = []
    for event in sorted(raw, key=lambda event: float(event["time"])):
        if event.get("event_type") in {"seed", "infection", "transfer"} and event.get("node") is not None:
            community = int(event["community"])
            if event["event_type"] == "transfer":
                repeated = bool(event.get("repeat_import", community in seen))
                event.setdefault("repeat_import", repeated)
                event.setdefault("destination_previously_infected", repeated)
                if repeated and float(event["time"]) <= task["T_max"]:
                    repeats.append([float(event["time"]), community, int(event["node"])])
            seen.add(community)
    summary["repeat_import_events"] = repeats
    summary["repeat_imports"] = len(repeats)
    summary["repeat_imports_to_endpoint"] = (
        sum(row[0] <= summary["endpoint_time"] for row in repeats)
        if summary["endpoint_time"] is not None else None)
    mm_run = {"simulator": MM, "runtime_seconds": time.perf_counter() - started,
              "T_end": task["T_max"], "termination": orch.run_metadata, "typed_events": raw,
              "scheduler": task["scheduler"],
              "rng_policy": "residual budgets; no rollback; private per-community micro streams; seeded worker-local module RNG for macro",
              **summary, "orchestrator_change_point_logs": result[2]}
    # Loading creates an independent payload. No code mutates the Full row, so
    # retain it directly and avoid both deepcopy and overlap with graph setup.
    del orch, W, result
    if task["full_extension"]:
        full_run = read_checkpoint(verify_reference(task["full_extension"]))["full_run"]
        baseline_reference = task["full_extension"]
    else:
        historical = read_checkpoint(baseline_path)
        candidates = [run for run in historical["runs"] if run["simulator"] == FULL]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one saved Full row: {baseline_path}")
        full_run = candidates[0]
        baseline_reference = task["source_checkpoint"]
        del historical, candidates
    baseline_provenance = {"reused": True, "new_simulation": False, **baseline_reference,
                           "row_sha256": object_hash(full_run), "horizon": task["full_horizon"],
                           "typed_events_available": "typed_events" in full_run,
                           "missing_event_types": "Historical untyped events cannot identify macro/local channels"}
    payload = {"version": VERSION, "status": "complete", "completed_utc": utc(),
               "task": task["original_task"], "rerun_task": task, "T_max": task["T_max"],
               "event_seeds": task["event_seeds"], "input_sha256": task["input_sha256"],
               "code_fingerprint": fingerprint, "runs": [full_run, mm_run],
               "full_baseline_provenance": baseline_provenance,
               "comparison_horizons": {"Full": task["full_horizon"], "TwoScale": task["T_max"]}}
    destination = output / "checkpoints" / (task["id"] + ".json.gz")
    write_checkpoint(destination, payload)
    result_metadata = {"task_id": task["id"], "checkpoint": str(destination.relative_to(output)),
                       "sha256": digest(destination), "mm_runtime_s": mm_run["runtime_seconds"],
                       "mm_repeat_imports": mm_run["repeat_imports"],
                       "mm_repeat_imports_to_T9": mm_run["repeat_imports_to_endpoint"],
                       "full_reused": True, "full_source_sha256": baseline_reference["sha256"]}
    del raw, payload, full_run
    gc.collect()
    return result_metadata


def valid_existing(output, task, fingerprint):
    path = output / "checkpoints" / (task["id"] + ".json.gz")
    if not path.exists():
        return False
    payload = read_checkpoint(path)
    if (payload.get("version") != VERSION or payload.get("status") != "complete"
            or payload.get("input_sha256") != task["input_sha256"]
            or payload.get("code_fingerprint") != fingerprint):
        raise RuntimeError(f"Existing checkpoint provenance differs; refusing overwrite: {path}")
    rows = payload.get("runs", [])
    if len(rows) != 2 or [row.get("simulator") for row in rows] != [FULL, MM]:
        raise RuntimeError(f"Incomplete paired checkpoint: {path}")
    provenance = payload.get("full_baseline_provenance", {})
    if not provenance.get("reused"):
        raise RuntimeError(f"Missing saved Full provenance: {path}")
    reference = task["full_extension"] or task["source_checkpoint"]
    if any(provenance.get(key) != reference[key] for key in ("source", "sha256")):
        raise RuntimeError(f"Full baseline provenance differs: {path}")
    if object_hash(rows[0]) != provenance.get("row_sha256"):
        raise RuntimeError(f"Saved Full row was altered: {path}")
    if any(key not in rows[1] for key in ("typed_events", "change_point_trajectories", "termination", "repeat_imports")):
        raise RuntimeError(f"Incomplete TwoScale trajectory: {path}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", "--dry-run", action="store_true", help="freeze inputs; no simulations")
    action.add_argument("--run", action="store_true", help="run/resume only corrected chain TwoScale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1, help="explicit concurrency; dense cliques always serial")
    parser.add_argument("--limit", type=int, help="optional number of selected tasks; full manifest still frozen")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    output = args.output.resolve()
    if output.is_relative_to(ROOT / "output/data"):
        parser.error("Choose an output folder outside historical output/data")
    with exclusive_output(output):
        try:
            manifest, tasks = prepare(output)
        except BaseException as error:
            log(output, "prepare_failed", error=repr(error), traceback=traceback.format_exc())
            raise
        if args.prepare_only:
            print(f"Prepared {len(tasks)} chain TwoScale tasks; no simulations started: {output}")
            return
        selected = tasks[:args.limit] if args.limit else tasks
        fingerprint = manifest["code_fingerprint"]
        pending, skipped, finished = [], 0, 0
        current_task = None
        try:
            for task in selected:
                if valid_existing(output, task, fingerprint):
                    skipped += 1
                    log(output, "task_resumed_skip", task_id=task["id"], code_fingerprint=fingerprint)
                else:
                    pending.append(task)
            log(output, "run_started", selected_tasks=len(selected), pending_tasks=len(pending),
                resumed_tasks=skipped, workers=args.workers, limit=args.limit, new_full_runs=0,
                code_fingerprint=fingerprint, tasks_sha256=manifest["tasks_sha256"], command=sys.argv,
                parameters={"beta": .003, "gamma": 0, "L": 10, "T_max": 30000,
                            "tau_micro": 1, "macro_T": 1,
                            "scheduler": "original_blackbox_midpoint_tau1"},
                seeds="Frozen per-task event_seeds and per-community seed formula in manifest/tasks.json.gz")

            def status(state):
                return {"status": state, "updated_utc": utc(), "selected_tasks": len(selected),
                        "resumed_tasks": skipped, "completed_this_invocation": finished,
                        "expected_twoscale_runs": EXPECTED_RUNS, "new_full_runs": 0,
                        "full_design_complete": state == "complete" and len(selected) == EXPECTED_RUNS,
                        "code_fingerprint": fingerprint}

            atomic_json(output / "run_status.json", status("running"))

            def accepted(task, result):
                nonlocal finished
                finished += 1
                log(output, "task_completed", code_fingerprint=fingerprint, **result)
                atomic_json(output / "run_status.json", status("running"))
                print(f"{finished}/{len(pending)} completed; {task['id']}", flush=True)

            def started(task, mode):
                log(output, "task_started", task_id=task["id"], mode=mode, code_fingerprint=fingerprint,
                    parameters={key: task[key] for key in ("beta", "gamma", "model", "T_max", "community_sizes", "scheduler", "tau_micro", "macro_T")},
                    event_seeds=task["event_seeds"], input_sha256=task["input_sha256"], full_reused=True)

            serial = [task for task in pending if task["original_task"]["target_ASPL"] == 1.]
            sparse = [task for task in pending if task["original_task"]["target_ASPL"] != 1.]
            for task in serial:
                current_task = task["id"]
                started(task, "serial_dense_clique")
                accepted(task, simulate_task(task, str(output), fingerprint))
            if args.workers == 1:
                for task in sparse:
                    current_task = task["id"]
                    started(task, "serial")
                    accepted(task, simulate_task(task, str(output), fingerprint))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                    iterator, active = iter(sparse), {}
                    while True:
                        while len(active) < args.workers:
                            task = next(iterator, None)
                            if task is None:
                                break
                            started(task, "parallel_sparse")
                            active[pool.submit(simulate_task, task, str(output), fingerprint)] = task
                        if not active:
                            break
                        done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            task = active.pop(future)
                            current_task = task["id"]
                            accepted(task, future.result())
            completion = status("complete" if len(selected) == EXPECTED_RUNS else "subset_complete")
            atomic_json(output / "run_status.json", completion)
            log(output, "run_completed", **completion)
        except BaseException as error:
            failure = {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       "task_id": current_task, "error": repr(error), "traceback": traceback.format_exc(),
                       "completed_this_invocation": finished, "resumed_tasks": skipped,
                       "code_fingerprint": fingerprint, "new_full_runs": 0}
            atomic_json(output / "run_status.json", failure)
            log(output, "run_failed", **failure)
            raise


if __name__ == "__main__":
    main()
