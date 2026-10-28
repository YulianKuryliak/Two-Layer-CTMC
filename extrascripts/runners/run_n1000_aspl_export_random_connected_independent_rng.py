#!/usr/bin/env python3
"""ASPL export experiment using the original G(n,m) random_connected generator.

This is the same experiment protocol as the tree-augmented runner, with the
internal graph generator replaced by the original random_connected mechanism:
sample G(n,m), reject disconnected graphs, and accept the requested ASPL.
Stages:
  python extrascripts/runners/run_n1000_aspl_export_random_connected_independent_rng.py --topology-only
  python extrascripts/runners/run_n1000_aspl_export_random_connected_independent_rng.py --smoke-only
  python extrascripts/runners/run_n1000_aspl_export_random_connected_independent_rng.py --workers 2

Data are stored under output/data/n1000_random_connected_aspl_export_rep80_aspl1to4p5.
"""
from __future__ import annotations

import gzip
import json
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.runners import run_n1000_aspl_export_independent_rng as base


# ----------------------------- user settings -----------------------------
base.N = 1000
base.K_VALUES = (1, 5)
base.BETA = 0.003
base.TAU_MICRO = 1.0
base.T_MAX = 30_000.0
base.WORKERS = 2
# Edit this tuple to choose the ASPL points for the experiment.
# Intermediate ASPL points are included through 4.5. Existing checkpoints
# in this experiment folder are skipped automatically on resume.
base.ASPL_TARGETS = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
base.REPLICATIONS = 80
base.GRAPH_LINEAGES = 20
base.EVENT_REPLICATES_PER_INTERFACE = 1
base.INTERFACE_REPLICATES = base.REPLICATIONS // base.GRAPH_LINEAGES
base.EXPERIMENT = "n1000_random_connected_aspl_export_rep80_aspl1to4p5"
base.ANALYSIS_ROOT = ROOT / "output" / "analysis" / base.EXPERIMENT
base.LEGACY_EXPERIMENT = "__no_legacy_random_connected_reuse__"
base.LEGACY_ROOT = base.OUTPUT_ROOT / base.LEGACY_EXPERIMENT
base.LEGACY_ANALYSIS_ROOT = ROOT / "output" / "analysis" / base.LEGACY_EXPERIMENT
base.MAX_GRAPH_ATTEMPTS = 10_000  # network.py's original rejection limit
base.MEAN_DEGREE_BY_ASPL = {
    1.0: 999.0,
    2.0: 50.0,
    2.5: 25.0,
    3.0: 12.0,
    3.5: 8.0,
    4.0: 6.0,
    4.5: 5.0,
}
base.ASPL_ABS_TOLERANCE = {
    1.0: 0.0,
    2.0: 0.10,
    2.5: 0.10,
    3.0: 0.10,
    3.5: 0.12,
    4.0: 0.15,
    4.5: 0.15,
}

_base_scenario_description = base.scenario_description
_base_experiment_parameters = base.experiment_parameters


def old_random_connected_graph(target_aspl: float, lineage_id: int):
    """Original network.py-style G(n,m) rejection generator."""
    mean_degree = base.MEAN_DEGREE_BY_ASPL[target_aspl]
    edge_count = int(round(mean_degree * base.N / 2))
    if edge_count < base.N - 1:
        raise ValueError("target mean degree is too low for a connected graph")
    graph_seed = base.seed("random_connected", target_aspl, lineage_id)
    rng = random.Random(graph_seed)
    last_aspl = None
    for attempt in range(1, base.MAX_GRAPH_ATTEMPTS + 1):
        seed_try = rng.randrange(1 << 30)
        graph = nx.gnm_random_graph(base.N, edge_count, seed=seed_try)
        if not nx.is_connected(graph):
            continue
        actual_aspl = base.exact_aspl(graph)
        last_aspl = actual_aspl
        if abs(actual_aspl - target_aspl) <= base.ASPL_ABS_TOLERANCE[target_aspl]:
            nx.set_edge_attributes(graph, 1.0, "weight")
            base.validate_graph(graph, target_aspl, actual_aspl)
            return graph, {
                "accepted": True,
                "target_ASPL": target_aspl,
                "ASPL": actual_aspl,
                "target_mean_degree": mean_degree,
                "generation_attempts": attempt,
                "graph_seed": graph_seed,
                "graph_sample_seed": seed_try,
                "m": graph.number_of_edges(),
            }
    return None, {
        "accepted": False,
        "target_ASPL": target_aspl,
        "target_mean_degree": mean_degree,
        "generation_attempts": base.MAX_GRAPH_ATTEMPTS,
        "last_connected_ASPL": last_aspl,
        "graph_seed": graph_seed,
    }


def scenario_description(task):
    result = _base_scenario_description(task)
    result["topology"] = "random_connected"
    result["generator"] = "network.py G(n,m) rejection: disconnected graphs discarded"
    result["replication_design"] = "20 independent graphs x 4 interface placements x 1 paired event replication"
    result["interpretation"] = "ASPL and mean degree vary together; original random_connected generator"
    return result


def scenario_name(task):
    info = scenario_description(task)
    return (
        f"topology_random_connected__n{base.N}__d{base.token(info['mean_degree'])}"
        f"__aspl{base.token(info['target_ASPL'])}__k{task['k']}"
        f"__beta{base.token(base.BETA)}__tau{base.token(base.TAU_MICRO)}__independent_rng"
    )


def experiment_parameters():
    result = _base_experiment_parameters()
    result["topology"] = "random_connected"
    result["graph_generator"] = "network.py G(n,m) rejection: disconnected graphs discarded"
    result["output_folder"] = f"output/data/{base.EXPERIMENT}"
    return result


def lightweight_descriptors(graph: nx.Graph, *, known_aspl: float):
    """Store only metadata needed by validation and analysis.

    Diameter and average clustering are not used by either simulator or the
    current plots.  Omitting them is important for n=1000 dense graphs,
    especially the ASPL=1 complete-graph cell.
    """
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    return {
        "n": graph.number_of_nodes(),
        "m": graph.number_of_edges(),
        "ASPL": known_aspl,
        "connected": nx.is_connected(graph),
        "simple": not graph.is_multigraph(),
        "mean_degree": float(degrees.mean()),
        "degree_variance": float(degrees.var()),
    }


def prepare_topology():
    """Generate and freeze 20 old-style random_connected graphs per ASPL."""
    manifest_path = base.topology_manifest_path()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("experiment") != base.EXPERIMENT or manifest.get("topology") != "random_connected":
            raise RuntimeError(f"existing data are incompatible with {base.EXPERIMENT}")
        if manifest.get("topology_gate") == "PASS":
            for target in base.ASPL_TARGETS:
                rows = manifest.get("cells", {}).get(str(target), [])
                if len(rows) != base.GRAPH_LINEAGES or not all(row.get("accepted") for row in rows):
                    raise RuntimeError("PASS topology manifest is structurally incomplete")
                for row in rows:
                    graph, descriptor = base.load_graph(ROOT / row["graph_path"])
                    base.validate_graph(graph, target, descriptor["ASPL"])
            return manifest

    root = base.root_folder() / "graphs"
    cells = {}
    gate = True
    for target in base.ASPL_TARGETS:
        rows = []
        for lineage_id in range(base.GRAPH_LINEAGES):
            print(f"[topology ASPL={target:g}] graph {lineage_id + 1}/{base.GRAPH_LINEAGES} "
                  f"(mean degree={base.MEAN_DEGREE_BY_ASPL[target]})", flush=True)
            graph, generation = old_random_connected_graph(target, lineage_id)
            if graph is None:
                rows.append({"lineage_id": lineage_id, "graph_path": None, **generation})
                gate = False
                continue
            descriptor = base.descriptors(graph, known_aspl=generation["ASPL"])
            path = root / f"random_connected_aspl{base.token(target)}" / f"graph{lineage_id:03d}.json.gz"
            base.store_graph(path, graph, descriptor)
            loaded, reloaded = base.load_graph(path)
            base.validate_graph(loaded, target, reloaded["ASPL"])
            rows.append({"lineage_id": lineage_id, "graph_path": str(path.relative_to(ROOT)),
                         **generation, **descriptor})
        cells[str(target)] = rows
    gate = gate and all(len(cells[str(target)]) == base.GRAPH_LINEAGES and
                        all(row.get("accepted") for row in cells[str(target)])
                        for target in base.ASPL_TARGETS)
    manifest = {
        "experiment": base.EXPERIMENT,
        "topology": "random_connected",
        "generator": "network.py G(n,m) rejection: disconnected graphs discarded",
        "topology_gate": "PASS" if gate else "FAIL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ASPL_tolerance": base.ASPL_ABS_TOLERANCE,
        "mean_degree_by_target_ASPL": {str(k): v for k, v in base.MEAN_DEGREE_BY_ASPL.items()},
        "cells": cells,
        "interpretation": "ASPL is varied by changing mean degree; this is not a fixed-density causal ASPL design",
    }
    base.atomic_json(manifest_path, manifest)
    return manifest


# Patch only the protocol hooks; all simulation, checkpoint, seed and resume
# logic remains the established runner implementation.
base.random_connected_graph = old_random_connected_graph
base.descriptors = lightweight_descriptors
base.scenario_description = scenario_description
base.prepare_topology = prepare_topology
base.scenario_name = scenario_name
base.experiment_parameters = experiment_parameters


if __name__ == "__main__":
    base.main()
