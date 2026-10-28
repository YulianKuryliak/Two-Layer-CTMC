#!/usr/bin/env python3
"""Reusable exact checks for compact output, worker writes and checkpoint plots.

    python extrascripts/validation/check_checkpoint_optimizations.py

Uses saved small networks; never generates networks or touches output/data.
Temporary IO fixtures are removed on exit. No figure generation is required.
"""
from __future__ import annotations

import concurrent.futures
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
from itertools import groupby
import json
import multiprocessing
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import networkx as nx
from extrascripts.runners import checkpoint_io as storage
from extrascripts.runners import run_custom_grid_independent_rng as grid
from extrascripts.runners import run_original_benchmarks_independent_rng as base
from extrascripts.validation import check_memory_optimizations as reference
from extrascripts.visualization import plot_trajectory_runtime_by_size as plot
from two_layer_ctmc.orchestrators.micromacro import Orchestrator
from two_layer_ctmc.simulators.micro import MicroSimulator


def sample_payload(replication=1, compact=False):
    metadata = {"family": "test", "benchmark": "fixture", "replication_id": replication}
    payload = {"format_version": 3 if compact else 2, **metadata,
               "runs": [], "runtime": [], "community_arrivals": []}
    # Include negative zero, duplicate times, non-dyadic times, and a count
    # above float64's exact integer range. Counts must never pass through float.
    values = [(0, -0.0, 2**53 + 1, 1, 0), (1, 0.1, 3, 2, 0),
              (0, 0.1, 1, 4, 0), (0, 0.10000000000000002, 0, 5, 0)]
    blocks, rows = [], []
    for model in (base.FULL, base.MICROMACRO):
        blocks.append(storage.trajectory_block(model, values))
        rows.extend({**metadata, "simulator": model,
                     **dict(zip(storage.TRAJECTORY_FIELDS, row))} for row in values)
        payload["runtime"].append({**metadata, "runtime_seconds": 0.1})
        payload["runs"].append({**metadata, "simulator": model, "completion": False, "T_last": float("nan"),
                                 "network_seed": 1000 + replication, "event_seed": 2000 + replication})
        for community in (0, 1):
            payload["community_arrivals"].append(
                {**metadata, "simulator": model, "community": community,
                 "first_infection_time": 0.0 if community == 0 else float("nan")})
    payload["trajectories"] = storage.compact_trajectories(metadata, blocks) if compact else rows
    return payload


def fixture_run(task, *, compact_output=False, stop_at_absorption=False):
    """A deterministic saved-shape payload; no simulation inside IO tests."""
    if task[1] == 3:
        raise ValueError("intentional worker failure")
    return sample_payload(task[1], compact_output)


def fixture_worker(task):
    # This patch is installed inside each spawned process, not inherited RNG
    # or process state. Exercise the actual worker checkpoint/receipt code.
    with patch.object(base, "_run_task", side_effect=fixture_run):
        return grid._safe_run_task(task)


def check_storage_and_plots(directory):
    scenario = directory / "fixture"
    paths = scenario / "checkpoints"
    paths.mkdir(parents=True)
    for rep, compact in ((1, False), (2, True)):
        payload = sample_payload(rep, compact)
        if rep == 1:
            for key in ("runtime", "community_arrivals"):
                for row in payload[key]:
                    row["benchmark"] = "legacy_scenario_name"
        path = paths / f"rep_{rep:04d}.json.gz"
        storage.write_checkpoint(path, payload)
        loaded = storage.read_checkpoint(path)
        assert storage.has_trajectories(loaded)
        expected = sample_payload(rep, False)["trajectories"]
        assert reference.digest(list(storage.iter_trajectory_rows(loaded))) == reference.digest(expected)
        assert reference.digest(loaded["runs"]) == reference.digest(payload["runs"])

    # Existing CSVs must not hide newly completed pairs, even if malformed.
    (scenario / "runtime.csv").write_text("obsolete and invalid", encoding="utf-8")
    (scenario / "community_arrivals.csv").write_text("", encoding="utf-8")
    plot._PAYLOAD_CACHE.clear()
    assert set(plot.runtime_rows(scenario, "fixture").replication_id) == {1, 2}
    assert set(plot.arrival_rows(scenario, "fixture").replication_id) == {1, 2}
    times, stacks = plot.aggregate_stack(scenario, "fixture", [1, 2])
    old = pd.DataFrame(sample_payload()["trajectories"])
    for model in plot.MODELS:
        expected = old[old.simulator.eq(model)].groupby("time", sort=True)["I"].sum()
        assert np.array_equal(times, expected.index.to_numpy(float))
        assert np.array_equal(stacks[model], np.vstack([expected.to_numpy(float)] * 2))

    # A failed replacement must not damage an existing completed checkpoint.
    path = paths / "rep_0001.json.gz"
    before = path.read_bytes()
    with patch.object(Path, "replace", side_effect=OSError("intentional IO failure")):
        try:
            storage.write_checkpoint(path, sample_payload(9, True))
        except OSError:
            pass
        else:
            raise AssertionError("Expected simulated write failure")
    assert path.read_bytes() == before
    assert not list(paths.glob("*.tmp"))
    print("PASS: lossless legacy/compact IO, mixed checkpoint plotting, stale CSV ignored, atomic failure")


def check_workers(directory):
    tasks = [({"name": "fixture", "community_size": 8}, rep, 17, 19, str(directory))
             for rep in (1, 2, 3)]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        receipts = list(executor.map(fixture_worker, tasks))
    for receipt in receipts:
        assert "trajectories" not in receipt and len(json.dumps(receipt)) < 1024
        path = Path(receipt["checkpoint"])
        assert path.exists()
        payload = storage.read_checkpoint(path)
        if receipt["replication_id"] == 3:
            assert "error" in receipt and path.parent.name == "failures"
        else:
            assert "error" not in receipt and storage.has_trajectories(payload)
    assert set(grid._checkpoint_ids(directory / "fixture" / "checkpoints")) == {1, 2}
    print("PASS: two spawned workers; compact durable receipts; failed pairs excluded from reuse")


def check_resume(directory, case):
    item = replace(base.Benchmark(**case["benchmark"]), name="fixture", T_end=12.0)
    old_item = replace(item, T_end=5.0)
    scenario = directory / item.name
    scenario.mkdir(parents=True)
    complete = sample_payload(1, True)
    complete["benchmark_config"] = asdict(old_item)
    for row in complete["runs"]:
        row["completion"] = True
    censored = sample_payload(2, True)
    censored["benchmark_config"] = asdict(old_item)
    storage.write_checkpoint(scenario / "checkpoints" / "rep_0001.json.gz", complete)
    storage.write_checkpoint(scenario / "checkpoints" / "rep_0002.json.gz", censored)
    (scenario / "manifest.json").write_text(json.dumps({"benchmark_config": asdict(old_item)}))
    original_bytes = (scenario / "checkpoints" / "rep_0001.json.gz").read_bytes()
    reusable, extension = grid.inspect_reuse(scenario, item)
    assert set(reusable) == {1}
    assert extension == {"reused_completed_short": 1, "rerun_censored_short": 1}

    submitted = []
    def execute(tasks, workers, heavy_workers):
        submitted.extend(tasks)
        for task in tasks:
            assert task[0]["T_end"] == 12.0
            assert (task[2], task[3]) == (1002, 2002)
            with patch.object(base, "_run_task", side_effect=fixture_run):
                yield grid._safe_run_task(task)

    args = grid.argparse.Namespace(runs=2, workers=1, heavy_workers=None, output=directory)
    with patch.object(grid, "parse_args", return_value=args), \
         patch.object(grid, "scenarios", return_value=[item]), \
         patch.object(base, "_regression_gate", return_value=(True, "test stub; no network generation")), \
         patch.object(grid, "_partitioned_results", side_effect=execute), redirect_stdout(io.StringIO()):
        grid.main()
    assert [task[1] for task in submitted] == [2]
    assert (scenario / "checkpoints" / "rep_0001.json.gz").read_bytes() == original_bytes
    assert not list(scenario.glob("*.csv"))
    assert json.loads((scenario / "manifest.json").read_text())["benchmark_config"]["T_end"] == 12.0
    print("PASS: selective horizon extension reuses completed pair, reruns censored pair, no CSV consolidation")


def check_reporting_edges(case):
    full = reference.unpack_graph(case["full_graph"])
    graphs = [reference.unpack_graph(graph) for graph in case["micro_graphs"]]
    W = np.asarray(case["W"], dtype=float)
    # Non-integral dt and horizon exercise exact old reporting-grid endpoints.
    item = replace(base.Benchmark(**case["benchmark"]), T_end=12.05, dt_out=0.1)
    task = (asdict(item), 1, case["network_seed"], case["event_seed"])
    with patch.object(base, "_generate_network", return_value=(graphs, full, W)):
        legacy = base._run_task(task)
        compact = base._run_task(task, compact_output=True)
    compact["trajectories"] = list(storage.iter_trajectory_rows(compact))
    compact["format_version"] = legacy["format_version"]
    assert reference.digest(reference.without_runtime(legacy)) == reference.digest(reference.without_runtime(compact))

    # Multiple state changes at identical times: use the last state, exactly
    # as the dense logger does. This is an output-only synthetic fixture.
    sparse = {0: {"times": [0.0, 0.1, 0.1, 0.3], "S": [5, 4, 3, 0],
                  "I": [1, 2, 3, 6], "R": [0, 0, 0, 0]}}
    orchestrator = base.Orchestrator.__new__(base.Orchestrator)
    orchestrator.times = [0.0, 0.1, 0.1, 0.2, 0.3, 0.4]
    orchestrator.logs = deepcopy(sparse)
    orchestrator._densify_logs_from_global_times()
    item = replace(item, communities=1, T_end=0.4)
    assert base._micromacro_trajectory(item, 1, sparse) == base._micromacro_trajectory(item, 1, orchestrator.logs)
    print("PASS: non-integral reporting horizon/grid and duplicate-time resampling")


def check_horizon_replay_prefix(case):
    """A longer replay must reproduce all saved values before the old horizon."""
    benchmark = base.Benchmark(**case["benchmark"])

    def run(horizon):
        item = replace(benchmark, T_end=horizon, original_T_end=horizon)
        full = reference.unpack_graph(case["full_graph"])
        graphs = [reference.unpack_graph(graph) for graph in case["micro_graphs"]]
        with patch.object(base, "_generate_network", return_value=(
            graphs, full, np.asarray(case["W"], dtype=float)
        )):
            return base._run_task(
                (asdict(item), 1, case["network_seed"], case["event_seed"]), compact_output=True
            )

    short, long = run(6.0), run(12.0)
    short_rows = list(storage.iter_trajectory_rows(short))
    long_prefix = [row for row in storage.iter_trajectory_rows(long)
                   if float(row["time"]) <= 6.0 + 1e-12]
    assert reference.digest(short_rows) == reference.digest(long_prefix)
    print("PASS: same seeds reproduce the old exact trajectory as a longer-run prefix")


def check_natural_termination():
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 1.0, "weight")
    full = MicroSimulator(graph, [list(graph)], 1.0, 0.0, model=2).run(
        T_end=100.0, dt_out=1.0, initial_node=0, seed=17, stop_at_absorption=True
    )
    metadata = full["run_metadata"]
    assert metadata["stop_reason"] == "all_nodes_infected"
    assert metadata["stop_time"] < 100.0
    assert full["rows"][-1][1] < 100.0
    assert metadata["final_state_by_community"] == [{"community": 0, "S": 0, "I": 3, "R": 0}]

    random_graph = nx.complete_graph(3)
    nx.set_edge_attributes(random_graph, 1.0, "weight")
    mm = Orchestrator(
        W=np.zeros((1, 1)), micro_graphs=[random_graph], beta_micro=1.0,
        gamma=0.0, beta_macro=1.0, tau_micro=1.0, T_end=100.0,
        model=2, dense_logs=False, stop_at_absorption=True,
    )
    mm.micro_models[0].rng = random.Random(17)
    mm.micro_models[0]._infect_node(0)
    mm.run()
    assert mm.run_metadata["stop_reason"] == "all_nodes_infected"
    assert mm.run_metadata["stop_time"] < 100.0
    assert mm.times[-1] < 100.0
    assert mm.run_metadata["final_state_by_community"] == [{"community": 0, "S": 0, "I": 3, "R": 0}]
    print("PASS: absorbing SI stops before T_end; final state/metadata retained; no held plateau")


def check_custom_runner_termination_payload(case):
    item = replace(
        base.Benchmark(**case["benchmark"]), communities=1, community_size=3,
        beta=1.0, gamma=0.0, T_end=100.0, original_T_end=100.0,
    )
    graph = nx.complete_graph(3)
    nx.set_edge_attributes(graph, 1.0, "weight")
    with patch.object(base, "_generate_network", return_value=(
        [graph], graph, np.zeros((1, 1))
    )):
        payload = base._run_task(
            (asdict(item), 1, case["network_seed"], case["event_seed"]),
            compact_output=True, stop_at_absorption=True,
        )
    assert payload["format_version"] == 4
    assert {row["simulator"] for row in payload["termination"]} == set(plot.MODELS)
    assert all(row["stop_time"] < 100.0 for row in payload["termination"])
    for model in plot.MODELS:
        columns = storage.model_trajectory_columns(payload, model)
        assert max(columns["time"]) < 100.0
    print("PASS: custom task records compact termination metadata without T_end plateau")


def check_existing_checkpoint(path):
    """Optional real-data round trip; writes only to a temporary directory."""
    legacy = storage.read_checkpoint(path)
    if storage.is_compact(legacy.get("trajectories")):
        raise ValueError("This optional conversion test expects a legacy row checkpoint")
    rows = legacy["trajectories"]
    metadata_keys = ("family", "benchmark", "replication_id")
    metadata = {key: rows[0][key] for key in metadata_keys}
    assert all(all(row[key] == metadata[key] for key in metadata_keys) for row in rows)
    blocks = []
    for simulator, model_rows in groupby(rows, key=lambda row: row["simulator"]):
        blocks.append(storage.trajectory_block(simulator,
            (tuple(row[key] for key in storage.TRAJECTORY_FIELDS) for row in model_rows)))
    compact = {**legacy, "format_version": 3,
               "trajectories": storage.compact_trajectories(metadata, blocks)}
    with tempfile.TemporaryDirectory(prefix="ctmc-existing-checkpoint-") as tmp:
        target = Path(tmp) / "compact.json.gz"
        storage.write_checkpoint(target, compact)
        restored = storage.read_checkpoint(target)
        restored["trajectories"] = list(storage.iter_trajectory_rows(restored))
        restored["format_version"] = legacy["format_version"]
        assert reference.digest(restored) == reference.digest(legacy)
        print(f"PASS: existing checkpoint exact round trip ({len(rows)} rows); "
              f"gzip {path.stat().st_size:,} -> {target.stat().st_size:,} bytes; source untouched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-checkpoint", type=Path,
                        help="also verify a real legacy checkpoint without modifying it")
    args = parser.parse_args()
    cases = json.loads(reference.REFERENCE.read_text())["cases"]
    for case in cases:
        actual = reference.evaluate(case, shared_views=True, compact_output=True)
        differences = [key for key, expected in case["expected"].items() if actual[key] != expected]
        assert not differences, f"{case['benchmark']['name']}: changed {differences}"
    print(f"PASS: compact/sparse path matches {len(cases)} saved exact oracles (events, states, RNG, trajectories)")
    check_reporting_edges(cases[0])
    check_horizon_replay_prefix(cases[0])
    check_natural_termination()
    check_custom_runner_termination_payload(cases[0])
    with tempfile.TemporaryDirectory(prefix="ctmc-checkpoints-") as tmp:
        directory = Path(tmp)
        check_storage_and_plots(directory / "io")
        check_workers(directory / "workers")
        check_resume(directory / "resume", cases[0])
    if args.existing_checkpoint:
        check_existing_checkpoint(args.existing_checkpoint)


if __name__ == "__main__":
    main()
