#!/usr/bin/env python3
"""Bounded exact-trajectory validation of immutable baseline and optimized copy.

No paper suite is launched. Synthetic cases and two saved-input chain replays
run sequentially. Every invocation is recorded in the central experiment log.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SAVED = ROOT / "output/chain_twoscale_residual_midpoint_20261007"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(value):
    """Exact values/types, including float sign and binary representation."""
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, dict):
        return ["dict", [[freeze(k), freeze(v)] for k, v in value.items()]]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [freeze(v) for v in value]]
    return [type(value).__name__, value]


def exact_digest(value):
    return hashlib.sha256(json.dumps(freeze(value), separators=(",", ":")).encode()).hexdigest()


def assert_exact(a, b, label):
    if freeze(a) != freeze(b):
        raise AssertionError(f"Exact mismatch: {label}")


def log(output, event, **fields):
    row = {"utc": utc(), "runner": "state_events_exact_validation_v1",
           "event": event, "output": str(output), **fields}
    for path in (output / "experiments.jsonl", ROOT / "logs/experiments.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def load_package(name, variant):
    path = HERE / variant / "two_layer_ctmc"
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def model_state(model, *, snapshots=False):
    names = ("current_time", "infected_nodes", "total_infection_rate",
             "total_recovery_rate", "S_count", "I_count", "R_count",
             "_cached_wait", "_cached_event", "_pending_rate", "_pending_origin",
             "_remaining_intensity_budget")
    state = {name: getattr(model, name, None) for name in names}
    state["infected_nodes"] = list(model.infected_nodes)
    state["nodes"] = [(node, data.copy()) for node, data in model.G._node.items()]
    state["rng"] = model.rng.getstate()
    if snapshots:
        state["before"] = model._cached_rng_before
        state["after"] = model._cached_rng_after
    return state


def make_graph(kind, n, seed, *, labels=False):
    if kind == "path":
        graph = nx.path_graph(n)
    elif kind == "clique":
        graph = nx.complete_graph(n)
    elif kind == "star":
        graph = nx.star_graph(n - 1)
    elif kind == "isolated":
        graph = nx.empty_graph(n)
    else:
        graph = nx.gnp_random_graph(n, .18, seed=seed)
    rng = random.Random(seed)
    order = list(graph.nodes())
    rng.shuffle(order)
    result = nx.Graph()
    result.add_nodes_from(order)
    for u, v in graph.edges():
        result.add_edge(u, v, weight=rng.choice((0., .1, .3, 1., 1.7)))
    if labels:
        result = nx.relabel_nodes(result, {i: f"node_{i}" for i in result}, copy=True)
    return result


def make_model(package, graph, seed, gamma, model, *, fast=False):
    kwargs = {"rng": random.Random(seed)}
    if fast:
        kwargs["capture_rng_snapshots"] = False
    m = package.MicroModel(.35, gamma, model, **kwargs)
    for node in graph:
        m.add_node(node)
    for u, v, data in graph.edges(data=True):
        m.add_edge(u, v, data["weight"])
    m._infect_node(next(iter(graph)))
    return m


def micro_checks(base, opt):
    rows = []
    for kind in ("path", "clique", "sparse", "isolated"):
        for gamma, model in ((0., 2), (.2, 2), (.2, 1)):
            for seed in (7, 19, 31, 83):
                for fast in (False, True):
                    graph = make_graph(kind, 17, seed, labels=(seed == 83))
                    a = make_model(base, graph, seed, gamma, model)
                    b = make_model(opt, graph, seed, gamma, model, fast=fast)
                    history = []
                    for horizon in (.0, .05, .2, .75, 1.5, 3., 6., 12.):
                        assert_exact(a.simulate_until(horizon), b.simulate_until(horizon), "micro events")
                        assert_exact(model_state(a, snapshots=not fast),
                                     model_state(b, snapshots=not fast), "micro state")
                        if horizon in (.2, 1.5, 6.):
                            assert_exact(a.import_infection(at_time=horizon),
                                         b.import_infection(at_time=horizon), "import placement")
                            assert_exact(model_state(a, snapshots=not fast),
                                         model_state(b, snapshots=not fast), "post-import state")
                        ca, cb = a.clone(), b.clone()
                        parent_before = model_state(a, snapshots=not fast)
                        assert_exact(ca.simulate_until(horizon + .13, record_events=False),
                                     cb.simulate_until(horizon + .13, record_events=False), "probe")
                        ca.refresh_from(a)
                        cb.refresh_from(b, static_node_order=True)
                        assert_exact(model_state(ca, snapshots=not fast),
                                     model_state(cb, snapshots=not fast), "restored probe")
                        assert_exact(parent_before, model_state(a, snapshots=not fast), "probe isolation")
                        assert_exact(model_state(a, snapshots=not fast),
                                     model_state(b, snapshots=not fast), "optimized probe isolation")
                        history.append(model_state(a, snapshots=not fast))
                    rows.append({"kind": kind, "gamma": gamma, "model": model,
                                 "seed": seed, "fast_rng_path": fast,
                                 "sha256": exact_digest(history), "exact": True})
    # Standard public diagnostic API remains exact, including before/after states.
    for seed in (7, 19, 31):
        graph = make_graph("path", 10, seed)
        a, b = make_model(base, graph, seed, 0., 2), make_model(opt, graph, seed, 0., 2)
        assert_exact(a._ensure_next_event(), b._ensure_next_event(), "public 4-tuple")
    # The public refresh path is key-based even when insertion order differs.
    graph = make_graph("path", 10, 19)
    source = make_model(opt, graph, 19, .2, 1)
    reverse = nx.Graph()
    reverse.add_nodes_from(reversed(list(graph)))
    reverse.add_edges_from(graph.edges(data=True))
    target = make_model(opt, reverse, 21, .2, 1)
    target.refresh_from(source)
    assert target.G._node == source.G._node
    return rows


def engine_checks(base, opt):
    rows = []
    for cached in (False, True):
        for compact in (False, True):
            for gamma, model in ((0., 2), (.3, 2), (.3, 1)):
                for seed in (7, 19, 31):
                    models = []
                    for package in (base, opt):
                        m = package.MicroEngine(.4, gamma, model, rng=random.Random(seed),
                                                cache_neighbors=cached, compact_neighbors=compact,
                                                track_counts=True)
                        for node in (4, 0, 3, 1, 2):
                            m.add_node(node)
                        for u, v, w in ((4, 0, .1), (4, 3, .3), (4, 1, 1.),
                                        (4, 2, 0.), (4, 4, .3), (4, 0, .1), (0, 3, .2)):
                            m.add_edge(u, v, w)
                        m._infect_node(4)
                        models.append(m)
                    history = []
                    for index in range(80):
                        a, b = models
                        event_a, event_b = a.simulate_step(True), b.simulate_step(True)
                        assert_exact(event_a, event_b, "Full event")
                        assert_exact(model_state(a), model_state(b), "Full state")
                        history.append(event_a)
                        if event_a[0] == float("inf"):
                            break
                    rows.append({"cached": cached, "compact": compact, "gamma": gamma,
                                 "model": model, "seed": seed, "sha256": exact_digest(history),
                                 "exact": True})
    for seed in (7, 19, 31):
        graph = make_graph("sparse", 17, seed)
        models = []
        for package in (base, opt):
            model = package.MicroModel(.4, .3, 1, rng=random.Random(seed))
            model.bind_static_graph(graph)
            model._infect_node(next(iter(graph)))
            models.append(model)
        for horizon in (.1, 1., 3., 7.):
            a, b = models
            assert_exact(a.simulate_until(horizon), b.simulate_until(horizon), "bound graph events")
            assert_exact(model_state(a, snapshots=True), model_state(b, snapshots=True), "bound graph state")
        rows.append({"kind": "bind_static_graph", "seed": seed,
                     "sha256": exact_digest(model_state(models[0], snapshots=True)), "exact": True})
    return rows


def run_orchestrator(package, *, seed, kind, gamma, model, dense=False,
                     tau=.5, correction="off", n=14, communities=3, horizon=12.):
    graphs = []
    for c in range(communities):
        g = make_graph(kind, n, seed + c)
        graphs.append(nx.relabel_nodes(g, {node: node + c * n for node in g}, copy=True))
    W = np.zeros((communities, communities))
    if kind != "isolated":
        for c in range(communities - 1):
            W[c, c + 1] = W[c + 1, c] = 3.
    full = nx.Graph()
    full.graph.update(boundary_counts=[3] * communities, mu2_values=[.6] * communities)
    random.seed(seed)
    np.random.seed(seed)
    orch = package.Orchestrator(W, graphs, .35, gamma, .35, tau, horizon,
                                model=model, alphas=[1.2] * communities,
                                community_sizes=[n] * communities, full_graph=full,
                                dense_logs=dense, stop_at_absorption=True,
                                alpha_mode="spectral" if correction == "spectral" else "off",
                                correction_mode="off" if correction == "spectral" else correction,
                                micro_rngs=[random.Random(seed + 10000 + 1009*c)
                                            for c in range(communities)])
    orch.micro_models[0]._infect_node(next(iter(graphs[0])))
    started = time.perf_counter()
    result = orch.run()
    seconds = time.perf_counter() - started
    state = {"result": result, "termination": orch.run_metadata,
             "models": [model_state(m) for m in orch.micro_models],
             "macro_rng": random.getstate(), "numpy_rng": np.random.get_state()[1].tolist()}
    if package.__name__ == "validation_optimized":
        assert all(m._cached_rng_before is None and m._cached_rng_after is None
                   for m in orch.micro_models), "Only unused diagnostic snapshots are disabled"
    return state, seconds


def orchestrator_checks(base, opt):
    rows = []
    for kind in ("path", "clique", "sparse", "isolated"):
        for gamma, model in ((0., 2), (.2, 2), (.2, 1)):
            for seed in (7, 19, 83):
                params = dict(kind=kind, gamma=gamma, model=model, seed=seed,
                              dense=(seed == 19), tau=.25 if seed == 83 else 1.)
                a, _ = run_orchestrator(base, **params)
                b, _ = run_orchestrator(opt, **params)
                assert_exact(a, b, "complete MicroMacro trajectory")
                rows.append({**params, "sha256": exact_digest(a), "exact": True})
    for correction in ("boundary_phi", "renorm_phi", "spectral"):
        params = dict(kind="clique", gamma=.2, model=1, seed=19, correction=correction)
        a, _ = run_orchestrator(base, **params)
        b, _ = run_orchestrator(opt, **params)
        assert_exact(a, b, "corrected hazard trajectory")
        rows.append({**params, "sha256": exact_digest(a), "exact": True})
    return rows


def benchmarks(base, opt):
    rows = []
    for kind in ("midpoint", "events"):
        times = {"baseline": [], "optimized": []}
        for repeat in range(3):
            results = {}
            variants = (("baseline", base), ("optimized", opt))
            if repeat % 2:
                variants = variants[::-1]
            for name, package in variants:
                if kind == "midpoint":
                    state, seconds = run_orchestrator(package, seed=409,
                        kind="path", gamma=0., model=2, n=250, communities=10,
                        horizon=100., tau=.1)
                else:
                    graph = nx.star_graph(1499)
                    nx.set_edge_attributes(graph, 1., "weight")
                    model = make_model(package, graph, 409, 0., 2, fast=name == "optimized")
                    # make_model seeds first node; here it is the star centre.
                    start = time.perf_counter()
                    events = model.simulate_until(100000.)
                    seconds = time.perf_counter() - start
                    state = {"events": events, "state": model_state(model)}
                results[name] = state
                times[name].append(seconds)
            assert_exact(results["baseline"], results["optimized"], "benchmark trajectory")
        rows.append({"kind": kind, "seed": 409, "repeats": 3, "seconds": times,
                     "speedup_median": statistics.median(times["baseline"])/statistics.median(times["optimized"]),
                     "exact": True})
    return rows


def runner_child(variant, task_path, output, *, trace_memory=False):
    """Fresh process: package isolation plus comparable native peak RSS."""
    import resource
    import tracemalloc
    import weakref
    sys.path.insert(0, str(HERE / variant))
    from extrascripts.runners import rerun_chain_twoscale as runner
    from extrascripts.runners import rerun_smpt_paper as shared
    # The immutable baseline was copied byte-for-byte; point its data lookups
    # at the original project, without editing the copied source bytes.
    runner.ROOT = shared.ROOT = ROOT
    task = json.loads(task_path.read_text())
    graph_references = []
    def recorded_network(task, unused):
        result = shared.network(task, SAVED)
        graph_references.extend(weakref.ref(g) for g in [*result[0], result[1]])
        return result
    runner.network = recorded_network
    measurements = {}
    original_orchestrator = runner.Orchestrator
    class ObservedOrchestrator(original_orchestrator):
        def run(self):
            measurements["input_graphs_alive_at_run"] = sum(ref() is not None for ref in graph_references)
            if trace_memory:
                measurements["live_traced_bytes_at_run"] = tracemalloc.get_traced_memory()[0]
            return super().run()
    runner.Orchestrator = ObservedOrchestrator
    if trace_memory:
        tracemalloc.start()
    started = time.perf_counter()
    metadata = runner.simulate_task(task, str(output), "exact_validation_fixture")
    row = {"variant": variant, "seconds": time.perf_counter() - started,
           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "package": str(Path(sys.modules['two_layer_ctmc'].__file__).resolve()),
           "metadata": metadata, **measurements, "tracemalloc_enabled": trace_memory}
    if trace_memory:
        row["peak_traced_bytes"] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    if variant == "optimized" and row["input_graphs_alive_at_run"] != 0:
        raise AssertionError("Released input graphs still alive during optimized simulation")
    (output / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")


def replay_checks(output, fingerprint):
    import gzip
    rows = []
    fixtures = [
        ("dense", "macro_chain__L10__n50__clique__d49__aspl1__k1__beta0p003__tau1__independent_rng"),
        ("sparse", "macro_chain__L10__n50__tree_augmented__d2p6__aspl6__k1__beta0p003__tau1__independent_rng"),
    ]
    extension = json.loads((SAVED / "manifest.json").read_text())["historical_full_extensions"][0]
    extension_source = Path(extension["source"])
    fixtures.append(("extended", extension_source.parent.parent.name))
    fixtures.append(("memory_dense250", "macro_chain__L10__n250__clique__d249__aspl1__k1__beta0p003__tau1__independent_rng"))
    for kind, scenario in fixtures:
        folder = SAVED / "checkpoints/chain" / scenario
        candidates = sorted(folder.glob("*.json.gz"))
        if not candidates:
            # Use an existing n50 ASPL6 cell if its degree label differs.
            folders = sorted((SAVED / "checkpoints/chain").glob("*__n50__*__aspl6__k1__*"))
            if kind != "sparse" or not folders:
                raise RuntimeError(f"Missing saved replay fixture: {folder}")
            candidates = sorted(folders[0].glob("*.json.gz"))
        source = folder / extension_source.name if kind == "extended" else candidates[0]
        with gzip.open(source, "rt") as handle:
            saved_payload = json.load(handle)
        task = saved_payload["rerun_task"]
        if kind == "extended":
            assert task["full_extension"] is not None
        del saved_payload
        task_path = output / (kind + "_task.json")
        task_path.write_text(json.dumps(task, indent=2) + "\n")
        payloads, metrics = {}, {}
        for variant in ("baseline", "optimized"):
            target = output / "replays" / kind / variant
            target.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--runner-child",
                       variant, "--task", str(task_path), "--output", str(target)]
            if kind == "memory_dense250":
                command.append("--trace-memory")
            log(output, "child_started", code_fingerprint=fingerprint, command=command,
                variant=variant, task=task, source_checkpoint=str(source),
                source_sha256=digest(source), full_trajectory_policy="all original fields",
                resume="disabled; new validation folder")
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                log(output, "child_failed", code_fingerprint=fingerprint,
                    variant=variant, returncode=result.returncode, stderr=result.stderr)
                raise RuntimeError(result.stderr)
            metrics[variant] = json.loads((target / "metrics.json").read_text())
            checkpoint = target / metrics[variant]["metadata"]["checkpoint"]
            with gzip.open(checkpoint, "rt") as handle:
                payload = json.load(handle)
            # Only these intentionally non-scientific values differ.
            payload.pop("completed_utc")
            payload.pop("version")
            payload["runs"][1].pop("runtime_seconds")
            payloads[variant] = payload
            log(output, "child_completed", code_fingerprint=fingerprint,
                variant=variant, task_id=task["id"], metrics=metrics[variant], status="complete")
        assert_exact(payloads["baseline"], payloads["optimized"], "runner payload")
        # Compare with existing historical rerun, too, not just copied baseline.
        with gzip.open(source, "rt") as handle:
            historical = json.load(handle)
        historical.pop("completed_utc")
        historical.pop("version")
        historical["code_fingerprint"] = "exact_validation_fixture"
        historical["runs"][1].pop("runtime_seconds")
        assert_exact(historical, payloads["optimized"], "saved corrected trajectory")
        rows.append({"kind": kind, "task": task, "source": str(source),
                     "source_sha256": digest(source), "metrics": metrics,
                     "scientific_sha256": exact_digest(payloads["optimized"]), "exact": True})
    return rows


def check_snapshots():
    manifest = json.loads((HERE / "snapshot_manifest.json").read_text())
    for relative, sha in manifest["files"].items():
        for path in (ROOT / relative, HERE / "baseline" / relative):
            if digest(path) != sha:
                raise RuntimeError(f"Original/baseline source changed: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runner-child", choices=("baseline", "optimized"))
    parser.add_argument("--task", type=Path)
    parser.add_argument("--trace-memory", action="store_true")
    args = parser.parse_args()
    if args.runner_child:
        runner_child(args.runner_child, args.task, args.output, trace_memory=args.trace_memory)
        return
    output = args.output or ROOT / "output" / ("state_events_validation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = sorted(HERE.rglob("*.py"))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    fingerprint = exact_digest(hashes)
    manifest = {"created_utc": utc(), "code_fingerprint": fingerprint, "code_sha256": hashes,
                "environment": {"python": sys.version, "numpy": np.__version__, "networkx": nx.__version__},
                "parameters": {"micro": {"n": 17, "beta": .35, "seeds": [7,19,31,83],
                   "kinds": ["path","clique","sparse","isolated"],
                   "gamma_model": [[0,2],[.2,2],[.2,1]], "snapshot_modes": [True,False],
                   "horizons": [0,.05,.2,.75,1.5,3,6,12], "imports": [.2,1.5,6]},
                   "orchestrator": {"n":14,"L":3,"beta":.35,"T":12,"seeds":[7,19,83],
                    "tau": [.25,1], "dense_logs": [False,True], "corrections":["off","boundary_phi","renorm_phi","spectral"]},
                   "engine": {"seeds":[7,19,31],"steps_max":80,"beta":.4,"gamma_model":[[0,2],[.3,2],[.3,1]],
                     "cached_compact_modes":"all four combinations; self loops, duplicates, zero weights"},
                   "benchmark": {"seed":409,"repeats":3,"midpoint":{"L":10,"n":250,"T":100,"tau":.1,"beta":.35},
                     "events":{"n":1500,"kind":"star","T":100000,"beta":.35}},
                   "replays":"dense and sparse n50, extended Full n50, dense n250 with tracemalloc; exact parameters/seeds saved in *_task.json"},
                "trajectory_policy": "exact complete events, outputs, states, pending clocks and RNG states; float.hex equality",
                "resume": "disabled; every invocation requires a new output folder",
                "limitations": "bounded corpus; timings under ongoing CPU load; not the full paper suite"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(output, "run_started", command=sys.argv, code_fingerprint=fingerprint,
        parameters=manifest["parameters"], full_trajectory_policy=manifest["trajectory_policy"], resume="disabled")
    try:
        check_snapshots()
        base = load_package("validation_baseline", "baseline")
        opt = load_package("validation_optimized", "optimized")
        summary = {"micro": micro_checks(base, opt), "engine": engine_checks(base, opt),
                   "orchestrator": orchestrator_checks(base, opt)}
        print("Exact micro, Full event and MicroMacro checks passed", flush=True)
        # Run the unchanged residual-clock regression file against copied package.
        command = [sys.executable, str(HERE / "optimized/tests/test_residual_import_clock.py")]
        log(output, "regression_started", command=command, code_fingerprint=fingerprint,
            seeds="unchanged test_residual_import_clock.py", resume="disabled")
        result = subprocess.run(command, text=True, capture_output=True)
        (output / "residual_regression.txt").write_text(result.stdout + result.stderr)
        log(output, "regression_completed", code_fingerprint=fingerprint, returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(result.stderr)
        summary["benchmarks"] = benchmarks(base, opt)
        print("Bounded exact benchmarks passed", flush=True)
        summary["replays"] = replay_checks(output, fingerprint)
        check_snapshots()
        for relative, sha in hashes.items():
            if digest(ROOT / relative) != sha:
                raise RuntimeError(f"Validation source changed during run: {relative}")
        summary.update(status="complete", completed_utc=utc(), code_fingerprint=fingerprint,
                       original_sources_unchanged=True, baseline_sources_unchanged=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log(output, "run_completed", code_fingerprint=fingerprint, status="complete",
            counts={name:len(summary[name]) for name in ("micro","engine","orchestrator","benchmarks","replays")})
        print(json.dumps({"output":str(output), "status":"complete", "benchmarks":summary["benchmarks"]}, indent=2))
    except BaseException as error:
        (output / "failure.txt").write_text(traceback.format_exc())
        log(output, "run_failed", code_fingerprint=fingerprint, error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
