"""Reusable exact regressions against saved pre-optimization reference hashes.

Default: load fixed networks; no network generation or benchmark output writes.
    python extrascripts/validation/check_memory_optimizations.py

The reference is an oracle, not a cache to refresh after a failing test.
Recording is an explicit maintainer operation and refuses to overwrite it:
    python extrascripts/validation/check_memory_optimizations.py --record-reference
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extrascripts.runners import run_original_benchmarks_independent_rng as runner
from two_layer_ctmc.engines.micro_engine import MicroEngine
from two_layer_ctmc.engines.micro_model import MicroModel
from two_layer_ctmc.engines.topology import NeighborList
from two_layer_ctmc.orchestrators.micro import MicroOrchestrator
from extrascripts.runners.checkpoint_io import iter_trajectory_rows

REFERENCE = ROOT / "extrascripts" / "validation" / "fixtures" / "memory_optimization_reference.json"
SOURCES = (
    "two_layer_ctmc/network.py", "two_layer_ctmc/engines/micro_engine.py",
    "two_layer_ctmc/engines/micro_model.py", "two_layer_ctmc/orchestrators/micro.py",
    "two_layer_ctmc/orchestrators/micromacro.py",
    "extrascripts/runners/run_original_benchmarks_independent_rng.py",
)


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())
    return hashlib.sha256(encoded.encode()).hexdigest()


def without_runtime(value):
    if isinstance(value, dict):
        return {k: without_runtime(v) for k, v in value.items() if k != "runtime_seconds"}
    if isinstance(value, (tuple, list)):
        return [without_runtime(v) for v in value]
    return value


def pack_graph(graph):
    return {"nodes": list(graph.nodes(data=True)), "edges": list(graph.edges(data=True)),
            "metadata": graph.graph}


def unpack_graph(data):
    graph = nx.Graph()
    graph.add_nodes_from(data["nodes"])
    graph.add_edges_from(data["edges"])
    graph.graph.update(data["metadata"])
    return graph


def evaluate(case, shared_views=False, compact_output=False):
    full = unpack_graph(case["full_graph"])
    graphs = [unpack_graph(item) for item in case["micro_graphs"]]
    if shared_views:
        views = [runner.network_module._community_view(full, list(g)) for g in graphs]
        for graph, view in zip(graphs, views):
            assert list(graph.nodes(data=True)) == list(view.nodes(data=True))
            assert list(graph.edges(data=True)) == list(view.edges(data=True))
        graphs = views
    W = np.asarray(case["W"], dtype=float)
    benchmark = runner.Benchmark(**case["benchmark"])
    before = digest([pack_graph(g) for g in [full, *graphs]])
    observed = {}
    original_full_run = runner.MicroSimulator.run
    original_mm_run = runner.Orchestrator.run

    def full_run(simulator, *args, **kwargs):
        result = original_full_run(simulator, *args, **kwargs)
        observed["full_result"] = digest(result)
        observed["full_rng"] = digest(kwargs["rng"].getstate())
        return result

    def mm_run(orchestrator):
        result = original_mm_run(orchestrator)
        if compact_output:
            # Normalize only the hash input. The runner still receives sparse
            # logs, exercising its direct resampling path below.
            sparse = orchestrator.logs
            try:
                orchestrator.logs = copy.deepcopy(sparse)
                orchestrator._densify_logs_from_global_times()
                observed["micromacro_result"] = digest(
                    (result[0], result[1], orchestrator.logs, result[3]))
            finally:
                orchestrator.logs = sparse
        else:
            observed["micromacro_result"] = digest(result)
        observed["micro_rngs"] = digest([m.rng.getstate() for m in orchestrator.micro_models])
        observed["macro_rng"] = digest(random.getstate())
        observed["micro_states"] = digest([
            [list(m.G.nodes(data=True)), m.current_time, m.infected_nodes,
             m.total_infection_rate, m.total_recovery_rate, m.count_states(),
             m._cached_wait, m._cached_event, m._cached_rng_before, m._cached_rng_after]
            for m in orchestrator.micro_models])
        return result

    with patch.object(runner, "_generate_network", return_value=(graphs, full, W)), \
         patch.object(runner.MicroSimulator, "run", full_run), \
         patch.object(runner.Orchestrator, "run", mm_run):
        result = runner._run_task((asdict(benchmark), 1, case["network_seed"], case["event_seed"]),
                                  compact_output=compact_output)
    if compact_output:
        result["trajectories"] = list(iter_trajectory_rows(result))
        result["format_version"] = runner.FORMAT_VERSION
    observed["payload"] = digest(without_runtime(result))
    assert before == digest([pack_graph(g) for g in [full, *graphs]]), "Input network mutated"
    return observed


def record():
    if REFERENCE.exists():
        raise SystemExit(f"Refusing to overwrite reference: {REFERENCE}")
    cases = []
    for topology in ("complete", "random_connected", "clique_leaf"):
        template = runner.BENCHMARKS[0 if topology == "clique_leaf" else 1]
        for model, gamma in ((2, 0.0), (2, 0.2), (1, 0.2)):
            for seed in (7, 19):
                b = replace(template, name=f"{topology}_m{model}_g{gamma}_s{seed}",
                            communities=2 if topology == "clique_leaf" else 3,
                            community_size=8, inter_links=2, beta=0.3, gamma=gamma,
                            model=model, micro_graph_type="clique_leaves" if topology == "clique_leaf" else topology,
                            edge_prob=0.5, T_end=12.0, original_T_end=12.0, dt_out=0.25,
                            replications=1, store_trajectories=True)
                graphs, full, W = runner._generate_network(b, seed + 100)
                case = {"benchmark": asdict(b), "network_seed": seed + 100, "event_seed": seed,
                        "micro_graphs": [pack_graph(g) for g in graphs],
                        "full_graph": pack_graph(full), "W": W.tolist()}
                case["expected"] = evaluate(case)
                cases.append(case)
    document = {"format_version": 1,
                "description": "Pre-memory-optimization code; independent RNG; exact hashes excluding runtime only.",
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
                "cases": cases}
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    with REFERENCE.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    print(f"Saved {len(cases)} reference cases to {REFERENCE}")


def check_storage_contracts(case):
    row = NeighborList()
    values = [(3, -0.0), (-4, 1.0), (3, 0.125), (1 << 80, 2.0), ("node", 0.5)]
    for item in values:
        row.append(item)
    assert digest(list(row)) == digest(values), "Compact conversion changed values/order"
    assert row[-1] == values[-1] and row[1:3] == values[1:3]

    graph = unpack_graph(case["full_graph"])
    nodes = [list(unpack_graph(g)) for g in case["micro_graphs"]]
    sim = MicroOrchestrator(graph, nodes, 0.3, 0.2, rng=random.Random(7))
    assert sim.sim.G._adj is graph._adj, "Full still copies adjacency"
    assert sim.sim.G._node is not graph._node, "Full shares mutable epidemic state"
    assert all(isinstance(row, NeighborList) for row in sim.sim._neighbors.values())
    sim.seed_infection(0)
    assert not graph.nodes[0].get("infected", False)
    for saved in case["micro_graphs"]:
        view = runner.network_module._community_view(graph, [n for n, _ in saved["nodes"]])
        assert nx.is_frozen(view)
        for u, v in view.edges():
            assert view[u][v] is graph[u][v], "Community edge attributes are copied"

    # Independently reproduce the old Full graph-building loop for weights,
    # self-loops and deliberately non-canonical input adjacency order.
    def legacy_bind(engine, graph):
        for node in graph:
            engine.add_node(node)
        for u, v, data in graph.edges(data=True):
            engine.add_edge(u, v, data.get("weight", 1.0))

    graph = nx.Graph()
    graph.add_nodes_from([0, 9, 1, 2, 3])
    graph.add_weighted_edges_from([(3, 9, 0.5), (2, 1, 0.125), (0, 3, 2.0),
                                   (0, 1, 0.75), (9, 2, 1.0), (1, 1, 0.5)])
    for model, gamma in ((2, 0.0), (2, 0.2), (1, 0.2)):
        for seed in (7, 19):
            with patch.object(MicroEngine, "bind_static_graph", legacy_bind):
                old = MicroOrchestrator(graph, [[0, 9], [1, 2, 3]], 0.3, gamma, model,
                                        rng=random.Random(seed))
            new = MicroOrchestrator(graph, [[0, 9], [1, 2, 3]], 0.3, gamma, model,
                                    rng=random.Random(seed))
            for node in graph:
                assert list(new.sim._neighbors_with_weights(node)) == list(old.sim._neighbors_with_weights(node))
            assert old.run(12.0, 0.25, 0) == new.run(12.0, 0.25, 0)
            assert old.rng.getstate() == new.rng.getstate()

    model = MicroModel(0.3, 0.0, rng=random.Random(7))
    for node in (0, 1):
        model.add_node(node)
    model.add_edge(0, 1, 1.0)
    with runner._rng_ownership_patch(independent=True):
        clone = model.clone()
        assert clone._neighbors is model._neighbors
        assert clone.rng is not model.rng and clone.rng.getstate() == model.rng.getstate()
        clone._infect_node(1)
        assert not model.G.nodes[1]["infected"]
        clone.refresh_from(model)
        assert not clone.G.nodes[1]["infected"]
        assert clone.rng.getstate() == model.rng.getstate()
    print("PASS: shared topology, isolated states, compact fallback, legacy Full order, clone/refresh")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-reference", action="store_true")
    parser.add_argument("--check-generator", action="store_true",
                        help="also regenerate only the small fixture networks and compare their ordered edges")
    args = parser.parse_args()
    if args.record_reference:
        record()
        return
    reference = json.loads(REFERENCE.read_text())
    for case in reference["cases"]:
        actual = evaluate(case, shared_views=True)
        differences = [key for key, value in case["expected"].items() if actual[key] != value]
        if differences:
            raise AssertionError(f"{case['benchmark']['name']}: changed {differences}")
        if args.check_generator:
            graphs, full, W = runner._generate_network(runner.Benchmark(**case["benchmark"]), case["network_seed"])
            for graph, saved in zip([full, *graphs], [case["full_graph"], *case["micro_graphs"]]):
                assert digest(list(graph.nodes(data=True))) == digest(saved["nodes"])
                assert digest(list(graph.edges(data=True))) == digest(saved["edges"])
            assert W.tolist() == case["W"]
    check_storage_contracts(reference["cases"][0])
    print(f"PASS: {len(reference['cases'])} saved networks; exact results, RNG and input immutability")


if __name__ == "__main__":
    main()
