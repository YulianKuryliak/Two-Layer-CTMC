#!/usr/bin/env python3
"""Sequential exactness and graph-preparation checks against verified v1.

Only bounded synthetic cases and four saved-input TwoScale replays run. No
Full simulation or paper suite. Each invocation has immutable output/logs.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import importlib
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
V1_RESULTS = ROOT / "output/state_events_validation_20261008T052505348663"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checkpoint(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def log(output, event, **fields):
    row = {"utc": datetime.now(timezone.utc).isoformat(),
           "runner": "graph_build_exact_validation_v2", "event": event,
           "output": str(output), **fields}
    for path in (output / "experiments.jsonl", ROOT / "logs/experiments.jsonl"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def utilities():
    path = ROOT / "optimization/state_events_v1/validate.py"
    spec = importlib.util.spec_from_file_location("graph_validation_utilities", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse exact float/type/RNG comparators and package loader; no v1 suite
    # is invoked and no v1 source or result is modified.
    module.HERE = HERE
    return module


def legacy_builder():
    """Execute the frozen network function, with its original NX operations."""
    path = HERE / "baseline/extrascripts/runners/rerun_smpt_paper.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "network")
    scope = {"nx": nx, "np": np, "read_checkpoint": read_checkpoint}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["network"]


def topology_signature(orch):
    """Streaming exact binary hash; avoid expanding dense arrays into tuples."""
    signature = hashlib.sha256()
    for model in orch.micro_models:
        for node in model.G._node:
            row = model._neighbors[node]
            signature.update(str(node).encode() + b"\0")
            signature.update(row._nodes.tobytes())
            signature.update(row._weights.tobytes())
    return signature.hexdigest()


def make_orchestrator(package, task, topology):
    micro, full, W, _ = topology
    seed = task.get("event_seeds", {}).get("micromacro_event_seed", 19)
    random.seed(seed)
    np.random.seed(seed)
    return package.Orchestrator(W, micro, task.get("beta", .4), task.get("gamma", 0.),
        task.get("beta", .4), task.get("tau_micro", .5), task.get("T_max", 8.),
        model=task.get("model", 2), macro_T=task.get("macro_T", 1.),
        community_sizes=task["community_sizes"], alphas=[1.] * len(micro),
        full_graph=full, dense_logs=False, stop_at_absorption=True,
        micro_rngs=[random.Random(seed + 10000 + 1009*c) for c in range(len(micro))])


def run_state(orch, task, tools):
    orch.micro_models[0]._infect_node(task.get("initial_node", 0))
    result = orch.run()
    np_state = np.random.get_state()
    return {"result": result, "termination": orch.run_metadata,
            "models": [tools.model_state(m) for m in orch.micro_models],
            "macro_rng": random.getstate(),
            "numpy_rng": (np_state[0], np_state[1].tolist(), *np_state[2:])}


def write_fixture(output, name, edges, *, interfaces=None, n=7):
    path = output / "fixtures" / (name + ".json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps({"edges": edges}).encode(), mtime=0))
    ref = {"snapshot": str(path.relative_to(output)), "sha256": digest(path)}
    return {"series": "chain", "community_sizes": [n] * 3,
            "graph_inputs": [ref.copy() for _ in range(3)],
            "interface_edges": interfaces if interfaces is not None else [[0,n],[n,2*n]],
            "initial_node": 0, "beta": .4, "gamma": 0., "model": 2,
            "T_max": 8., "tau_micro": .5, "macro_T": 1.,
            "event_seeds": {"micromacro_event_seed": 19}}


def synthetic_checks(base, opt, frozen, legacy, output, tools):
    fixtures = [
        ("canonical", [[0,1,.1],[0,3,-0.],[1,2,.3],[2,2,1.7]]),
        ("reversed_duplicates", [[2,0,.1],[0,1,.7],[0,2,1.3],[1,0,.2],
                                  [2,2,.3],[2,2,1.7],[4,3,0.],[3,4,-0.]]),
        ("isolated", []),
    ]
    for seed in (7,19,31,83):
        graph = nx.gnp_random_graph(13, .4, seed=seed)
        rng = random.Random(seed)
        edges = [[u,v,rng.choice((0.,.1,.3,1.,1.7))] for u,v in graph.edges()]
        rng.shuffle(edges)
        if edges:
            u,v,_ = edges[0]
            edges.extend([[v,u,.9],[u,v,1.1]])
        fixtures.append((f"unordered_{seed}", edges))
    records = []
    for name, edges in fixtures:
        task = write_fixture(output, name, edges, n=13 if name.startswith("unordered") else 7)
        frozen.clear_topology_cache()
        old, new = legacy(task, output), frozen.prepared_chain_network(task, output)
        assert new is not None
        tools.assert_exact(old[2].tolist(), new[2].tolist(), "macro W")
        tools.assert_exact(old[1].graph, new[1].graph, "macro metadata")
        tools.assert_exact(old[3], new[3], "community node lists")
        a, b = make_orchestrator(base, task, old), make_orchestrator(opt, task, new)
        tools.assert_exact([tools.model_state(m) for m in a.micro_models],
                           [tools.model_state(m) for m in b.micro_models], "initialized node states")
        assert topology_signature(a) == topology_signature(b), name
        records.append({"kind": "topology", "name": name, "task": task,
                        "sha256": topology_signature(b), "exact": True})
        for gamma, model in ((0.,2),(.2,2),(.2,1)):
            for seed in (7,19):
                case = {**task, "gamma":gamma, "model":model,
                        "event_seeds":{"micromacro_event_seed":seed}}
                a = make_orchestrator(base, case, old)
                state_a = run_state(a, case, tools)
                b = make_orchestrator(opt, case, new)
                state_b = run_state(b, case, tools)
                tools.assert_exact(state_a, state_b, "prepared topology trajectory")
                records.append({"kind":"trajectory", "name":name, "gamma":gamma,
                                "model":model, "seed":seed,
                                "sha256":tools.exact_digest(state_b), "exact":True})
    # Macro rates count raw duplicate interfaces; physical NX edges deduplicate.
    task = write_fixture(output, "interface_duplicates", [[0,1,1.]],
                         interfaces=[[0,7],[0,7],[7,0],[7,14]])
    old, new = legacy(task, output), frozen.prepared_chain_network(task, output)
    tools.assert_exact(old[2].tolist(), new[2].tolist(), "raw interface multiplicity")
    tools.assert_exact(old[1].graph, new[1].graph, "duplicate interface boundary")
    a = make_orchestrator(base, task, old)
    state_a = run_state(a, task, tools)
    b = make_orchestrator(opt, task, new)
    tools.assert_exact(state_a, run_state(b, task, tools), "duplicate interface trajectory")
    records.append({"kind":"interface_duplicates", "exact":True})
    # Shifted arrays must not alias the template or other communities.
    prepared = frozen.prepared_chain_network(task, output)
    row = prepared[0][0]._prepared_neighbors[0]
    row._weights[0] = 9.
    independent = frozen.prepared_chain_network(task, output)
    assert independent[0][0]._prepared_neighbors[0]._weights[0] == 1.
    assert prepared[0][1]._prepared_neighbors[7]._weights[0] == 1.
    records.append({"kind":"independent_arrays", "exact":True})
    for name, edges in (("extra_node",[[0,7,1.]]), ("string_node",[["0",1,1.]]),
                        ("negative_weight",[[0,1,-1.]]), ("infinite_weight",[[0,1,float('inf')]])):
        task = write_fixture(output, name, edges)
        assert frozen.prepared_chain_network(task, output) is None
        records.append({"kind":"fallback", "name":name, "exact":True})
    task = write_fixture(output, "internal_interface", [[0,1,1.]], interfaces=[[0,1]])
    assert frozen.prepared_chain_network(task, output) is None
    records.append({"kind":"fallback", "name":"internal_interface", "exact":True})
    # Same path and a different declared SHA must not hit an old cache entry.
    task = write_fixture(output, "cache_identity", [[0,1,1.]])
    frozen.prepared_chain_network(task, output)
    bad = {**task, "graph_inputs":[{**ref,"sha256":"0"*64} for ref in task["graph_inputs"]]}
    try:
        frozen.prepared_chain_network(bad, output)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Changed declared SHA reused a cached topology")
    path = output / task["graph_inputs"][0]["snapshot"]
    original = path.read_bytes()
    try:
        path.write_bytes(gzip.compress(json.dumps({"edges":[[0,1,1.],[1,2,2.]]}).encode(),mtime=0))
        try:
            frozen.prepared_chain_network(task, output)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Changed frozen asset reused stale cached topology")
    finally:
        path.write_bytes(original)
    records.append({"kind":"cache_integrity", "exact":True})
    return records


def preparation_benchmarks(base, opt, frozen, legacy, tools):
    import gc
    task = json.loads((V1_RESULTS / "memory_dense250_task.json").read_text())
    times = {key:[] for key in ("baseline", "optimized_cold", "optimized_warm")}
    signatures = set()
    for repeat in range(3):
        sequence = ("baseline","optimized_cold","optimized_warm") if repeat % 2 == 0 else (
                    "optimized_cold","optimized_warm","baseline")
        for name in sequence:
            if name == "optimized_cold":
                frozen.clear_topology_cache()
            start = time.perf_counter()
            topology = legacy(task,SAVED) if name == "baseline" else frozen.prepared_chain_network(task,SAVED)
            built = time.perf_counter()
            orch = make_orchestrator(base if name == "baseline" else opt,task,topology)
            initialized = time.perf_counter()
            times[name].append({"network_seconds":built-start, "model_init_seconds":initialized-built,
                                "total_seconds":initialized-start})
            signatures.add(topology_signature(orch))
            del orch,topology
            gc.collect()
    assert len(signatures) == 1, "Preparation benchmark topology mismatch"
    medians = {name:statistics.median(row["total_seconds"] for row in rows) for name,rows in times.items()}
    return {"task":task, "repeats":3, "times":times, "median_seconds":medians,
            "cold_speedup":medians["baseline"]/medians["optimized_cold"],
            "warm_speedup":medians["baseline"]/medians["optimized_warm"],
            "topology_sha256":next(iter(signatures)), "exact":True}


def runner_child(variant, task_path, output, trace_memory):
    import resource
    import tracemalloc
    sys.path.insert(0,str(HERE/variant))
    from extrascripts.runners import rerun_chain_twoscale as runner
    task = json.loads(task_path.read_text())
    build = runner.network
    metrics = {}
    def observed_network(task, unused):
        started = time.perf_counter()
        result = build(task,SAVED)  # Exercise the actual copied runner fast path.
        metrics["network_seconds"] = time.perf_counter()-started
        metrics["topology_types"] = [type(g).__name__ for g in result[0]]
        return result
    runner.network = observed_network
    original_class = runner.Orchestrator
    class ObservedOrchestrator(original_class):
        def __init__(self,*args,**kwargs):
            started=time.perf_counter()
            super().__init__(*args,**kwargs)
            metrics["model_init_seconds"]=time.perf_counter()-started
        def run(self):
            if trace_memory:
                metrics["live_traced_bytes_at_run"]=tracemalloc.get_traced_memory()[0]
            started=time.perf_counter()
            result=super().run()
            metrics["simulation_seconds"]=time.perf_counter()-started
            return result
    runner.Orchestrator=ObservedOrchestrator
    if trace_memory:
        tracemalloc.start()
    started=time.perf_counter()
    result=runner.simulate_task(task,str(output),"graph_validation_fixture")
    metrics.update(variant=variant,seconds=time.perf_counter()-started,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   tracemalloc_enabled=trace_memory,metadata=result,
                   package=str(Path(sys.modules['two_layer_ctmc'].__file__).resolve()))
    if trace_memory:
        metrics["peak_traced_bytes"]=tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    if variant == "optimized":
        assert set(metrics["topology_types"]) == {"PreparedMicroTopology"}, "Fast path was not exercised"
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")


def replay_checks(output,fingerprint,tools):
    records=[]
    for kind in ("dense","sparse","extended","memory_dense250"):
        task_path=V1_RESULTS/(kind+"_task.json")
        task=json.loads(task_path.read_text())
        (output/(kind+"_task.json")).write_text(json.dumps(task,indent=2)+"\n")
        payloads,metrics={},{}
        for variant in ("baseline","optimized"):
            destination=output/"replays"/kind/variant
            destination.mkdir(parents=True)
            command=[sys.executable,str(Path(__file__).resolve()),"--runner-child",variant,
                     "--task",str(task_path),"--output",str(destination)]
            if kind == "memory_dense250": command.append("--trace-memory")
            log(output,"child_started",code_fingerprint=fingerprint,command=command,task=task,
                variant=variant,trajectory_policy="all fields, unchanged Full, original horizons",resume="disabled")
            result=subprocess.run(command,text=True,capture_output=True)
            if result.returncode:
                log(output,"child_failed",code_fingerprint=fingerprint,variant=variant,
                    returncode=result.returncode,stderr=result.stderr)
                raise RuntimeError(result.stderr)
            metrics[variant]=json.loads((destination/"metrics.json").read_text())
            checkpoint=destination/metrics[variant]["metadata"]["checkpoint"]
            payload=read_checkpoint(checkpoint)
            payload.pop("version");payload.pop("completed_utc")
            payload["runs"][1].pop("runtime_seconds")
            payloads[variant]=payload
            log(output,"child_completed",code_fingerprint=fingerprint,variant=variant,
                status="complete",metrics=metrics[variant])
        tools.assert_exact(payloads["baseline"],payloads["optimized"],"runner scientific payload")
        historical=read_checkpoint(SAVED/"checkpoints"/(task["id"]+".json.gz"))
        historical.pop("version");historical.pop("completed_utc")
        historical["code_fingerprint"]="graph_validation_fixture"
        historical["runs"][1].pop("runtime_seconds")
        tools.assert_exact(historical,payloads["optimized"],"existing corrected trajectory")
        records.append({"kind":kind,"task":task,"metrics":metrics,
                        "scientific_sha256":tools.exact_digest(payloads["optimized"]),"exact":True})
    return records


def check_snapshots():
    manifest=json.loads((HERE/"snapshot_manifest.json").read_text())
    for relative,sha in manifest["files"].items():
        for path in (ROOT/manifest["source_root"]/relative,HERE/"baseline"/relative):
            assert digest(path)==sha,f"v1/baseline changed: {path}"
    for relative,sha in manifest["external_sources"].items():
        assert digest(ROOT/relative)==sha,f"External validation source changed: {relative}"
    original=json.loads((ROOT/"optimization/state_events_v1/snapshot_manifest.json").read_text())
    for relative,sha in original["files"].items():
        assert digest(ROOT/relative)==sha,f"Original project source changed: {relative}"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--runner-child",choices=("baseline","optimized"))
    parser.add_argument("--task",type=Path)
    parser.add_argument("--trace-memory",action="store_true")
    args=parser.parse_args()
    if args.runner_child:
        runner_child(args.runner_child,args.task,args.output,args.trace_memory)
        return
    output=(args.output or ROOT/"output"/("graph_build_validation_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))).resolve()
    output.mkdir(parents=True,exist_ok=False)
    check_snapshots()
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(HERE.rglob('*.py'))}
    hashes.update(json.loads((HERE/"snapshot_manifest.json").read_text())["external_sources"])
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"code_fingerprint":fingerprint,
              "code_sha256":hashes,"environment":{"python":sys.version,"numpy":np.__version__,"networkx":nx.__version__},
              "parameters":{"synthetic":{"n":[7,13],"L":3,"beta":.4,"T":8,"tau":.5,
                  "graph_seeds":[7,19,31,83],"event_seeds":[7,19],"gamma_model":[[0,2],[.2,2],[.2,1]],
                  "cases":"canonical, unordered, duplicate/reversed, self-loop, isolated, interface duplicates, cache tampering and fallbacks"},
                  "preparation_benchmark":{"saved_task":"dense n250 L10","repeats":3,"cold_and_warm":True},
                  "replays":[json.loads((V1_RESULTS/(kind+"_task.json")).read_text())
                             for kind in ("dense","sparse","extended","memory_dense250")]},
              "trajectory_policy":"exact all events/states/pending clocks/RNG; float.hex; all runner fields except version/time/fingerprint/runtime",
              "resume":"disabled; require new directory","parallelism":"sequential",
              "limits":"bounded corpus under ongoing CPU load; no paper suite or new Full simulations"}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    log(output,"run_started",code_fingerprint=fingerprint,command=sys.argv,parameters=manifest["parameters"],
        trajectory_policy=manifest["trajectory_policy"],resume="disabled")
    try:
        tools=utilities()
        base=tools.load_package("graph_baseline","baseline")
        opt=tools.load_package("graph_optimized","optimized")
        frozen=importlib.import_module(opt.__name__+".engines.frozen_topology")
        legacy=legacy_builder()
        summary={"synthetic":synthetic_checks(base,opt,frozen,legacy,output,tools)}
        print("Exact topology, trajectory and cache checks passed",flush=True)
        summary["preparation_benchmark"]=preparation_benchmarks(base,opt,frozen,legacy,tools)
        print("Cold/warm preparation benchmarks passed",flush=True)
        command=[sys.executable,str(HERE/"optimized/tests/test_residual_import_clock.py")]
        log(output,"regression_started",code_fingerprint=fingerprint,command=command,
            seeds="unchanged residual-clock regression file",resume="disabled")
        result=subprocess.run(command,text=True,capture_output=True)
        (output/"residual_regression.txt").write_text(result.stdout+result.stderr)
        log(output,"regression_completed",code_fingerprint=fingerprint,returncode=result.returncode)
        if result.returncode: raise RuntimeError(result.stderr)
        summary["replays"]=replay_checks(output,fingerprint,tools)
        check_snapshots()
        for relative,sha in hashes.items():
            assert digest(ROOT/relative)==sha,f"Validation source changed during run: {relative}"
        summary.update(status="complete",code_fingerprint=fingerprint,
                       completed_utc=datetime.now(timezone.utc).isoformat(),original_and_v1_sources_unchanged=True)
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        log(output,"run_completed",code_fingerprint=fingerprint,status="complete",
            counts={"synthetic":len(summary["synthetic"]),"replays":len(summary["replays"])})
        print(json.dumps({"output":str(output),"status":"complete",
                          "preparation_benchmark":summary["preparation_benchmark"]},indent=2))
    except BaseException as error:
        (output/"failure.txt").write_text(traceback.format_exc())
        log(output,"run_failed",code_fingerprint=fingerprint,error=repr(error),traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
