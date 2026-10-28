#!/usr/bin/env python3
"""Tree-augmented n=1000 export experiment for ASPL 1.00 through 1.75.

This reuses the established independent-RNG tree-augmented runner. Results
are resumable and stored in
``output/data/n1000_random_connected_tree_augmented_aspl1to1p75_step025_rep400_random_initial``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.runners import run_n1000_aspl_export_independent_rng as base
base.__doc__ = __doc__


# ----------------------------- user settings -----------------------------
base.N = 1000
base.K_VALUES = (1, 5)
base.BETA = 0.003
base.TAU_MICRO = 1.0
base.T_MAX = 30_000.0
base.WORKERS = 4

# Short dense-graph range: ASPL = 1.00, 1.25, 1.50, 1.75.
base.ASPL_TARGETS = (1.0, 1.25, 1.5, 1.75)

# Dense random graphs have ASPL close to 2 - mean_degree/(n-1).
# The topology gate checks every generated graph against the target.
base.MEAN_DEGREE_BY_ASPL = {
    1.0: 999.0,
    1.25: 749.0,
    1.5: 500.0,
    1.75: 250.0,
}
base.ASPL_ABS_TOLERANCE = {1.0: 0.0, 1.25: 0.10, 1.5: 0.10, 1.75: 0.10}

base.GRAPH_LINEAGES = 40
base.EVENT_REPLICATES_PER_INTERFACE = 1
base.INTERFACE_REPLICATES = 10
base.REPLICATIONS = base.GRAPH_LINEAGES * base.INTERFACE_REPLICATES
base.RANDOM_INITIAL_NODE = True
base.EXPERIMENT = "n1000_random_connected_tree_augmented_aspl1to1p75_step025_rep400_random_initial"
base.ANALYSIS_ROOT = ROOT / "output" / "analysis" / base.EXPERIMENT

# The initial-infection protocol changed, so no old topology or checkpoint
# data may be reused in this experiment.
base.LEGACY_EXPERIMENT = "__no_legacy_reuse_random_initial__"
base.LEGACY_ROOT = base.OUTPUT_ROOT / base.LEGACY_EXPERIMENT
base.LEGACY_ANALYSIS_ROOT = ROOT / "output" / "analysis" / base.LEGACY_EXPERIMENT

_base_scenario_description = base.scenario_description
_base_experiment_parameters = base.experiment_parameters


def experiment_parameters():
    result = _base_experiment_parameters()
    result["initial_node_policy"] = "deterministic random node per graph lineage and interface; shared by Full and MicroMacro"
    result["legacy_reuse"] = False
    return result


def scenario_description(task):
    result = _base_scenario_description(task)
    result["initial_node"] = "deterministic random node per graph/interface"
    result["replication_design"] = "40 independent graphs x 10 interface placements x 1 paired event replication"
    result["ASPL_tolerance"] = {str(target): 0.1 for target in base.ASPL_TARGETS}
    return result


base.scenario_description = scenario_description
base.experiment_parameters = experiment_parameters


if __name__ == "__main__":
    base.main()
