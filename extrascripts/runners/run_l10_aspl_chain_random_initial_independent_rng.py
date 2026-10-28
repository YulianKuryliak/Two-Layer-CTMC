#!/usr/bin/env python3
"""L=10 chain experiment with random initial node and internal ASPL 1..6.

At ASPL=1 each community is a clique. At ASPL=2..6 communities are
tree-augmented connected graphs. Each Full/MicroMacro pair shares the same
chain network, interfaces, and deterministically sampled initial node.
The default command runs the full resumable experiment; --topology-only and
--smoke-only are available for checks.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.runners import run_l10_random_connected_aspl_chain_independent_rng as base


# ============================ EDIT HERE ============================
base.L = 10
base.N = 1000
base.K_VALUES = (1, 5)
base.ASPL_TARGETS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
base.BETA = 0.003
base.TAU_MICRO = 1.0
base.T_MAX = 30_000.0
base.NETWORKS = 40
base.REPLICATES_PER_NETWORK = 10
base.REPLICATIONS = base.NETWORKS * base.REPLICATES_PER_NETWORK
base.WORKERS = 1  # clique chains use much more memory than sparse chains
# ===================================================================

base.RANDOM_INITIAL_NODE = True
base.MEAN_DEGREE_BY_ASPL = {
    1.0: 999.0, 2.0: 50.0, 3.0: 12.0,
    4.0: 6.0, 5.0: 4.3, 6.0: 3.4,
}
base.ASPL_TOLERANCE = {target: 0.1 for target in base.ASPL_TARGETS}
base.MAX_GRAPH_ATTEMPTS = 300
base.EXPERIMENT = "l10_n1000_chain_clique_tree_augmented_aspl1to6_step1_rep400_random_initial_independent_rng"
base.DATA_ROOT = ROOT / "output" / "data" / base.EXPERIMENT

_original_graph_path = base.graph_path
_original_load_graph = base.load_graph
_original_experiment_parameters = base.experiment_parameters
_original_scenario_parameters = base.scenario_parameters
_original_scenario_name = base.scenario_name
_original_run_pending = base.run_pending


def graph_path(target: float, network_id: int, community: int) -> Path:
    if target == 1.0:
        # All complete graphs on n labeled vertices are identical; store once.
        return base.DATA_ROOT / "graphs" / "aspl1" / f"complete_n{base.N}.json.gz"
    return _original_graph_path(target, network_id, community)


@functools.lru_cache(maxsize=1)
def _cached_clique(path: Path):
    return _original_load_graph(path, 1.0)


def load_graph(path: Path, target: float):
    if target == 1.0:
        return _cached_clique(path)
    return _original_load_graph(path, target)


def experiment_parameters():
    result = _original_experiment_parameters()
    result["initial_node"] = "deterministic uniform random node per network/interface replicate, shared by both simulators"
    result["ASPL_interpretation"] = "ASPL of each internal community graph; ASPL=1 is a clique"
    result["complete_graph_unique_topologies"] = 1
    result["legacy_reuse"] = False
    result["graph_generator"] = "complete graph at ASPL=1; random labeled tree plus extra edges at ASPL=2..6"
    return result


def scenario_parameters(target: float, k: int):
    result = _original_scenario_parameters(target, k)
    result["initial_node"] = "deterministic uniform random node per network/interface replicate"
    result["replication_design"] = "40 network seeds x 10 interface/event replicates; at ASPL=1 all communities have the same clique topology"
    result["topology"] = "chain_of_cliques" if target == 1.0 else "chain_of_tree_augmented_connected_graphs"
    return result


def scenario_name(target: float, k: int) -> str:
    if target == 1.0:
        return _original_scenario_name(target, k).replace("__tree_augmented__", "__clique__")
    return _original_scenario_name(target, k)


def run_pending(pending, workers: int) -> None:
    # One dense chain can occupy several GB. Keep clique tasks serial even if
    # the user requests more workers for the sparse cells.
    clique = [task for task in pending if float(task["target_ASPL"]) == 1.0]
    sparse = [task for task in pending if float(task["target_ASPL"]) != 1.0]
    if clique:
        print(f"[clique] {len(clique)} pairs, 1 worker (memory limit)", flush=True)
        _original_run_pending(clique, 1)
    if sparse:
        print(f"[other ASPL] {len(sparse)} pairs, {workers} workers", flush=True)
        _original_run_pending(sparse, workers)


base.graph_path = graph_path
base.load_graph = load_graph
base.experiment_parameters = experiment_parameters
base.scenario_parameters = scenario_parameters
base.scenario_name = scenario_name
base.run_pending = run_pending


if __name__ == "__main__":
    base.main()
