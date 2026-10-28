#!/usr/bin/env python3
"""Expose the two compatible random-initial ASPL ranges as one dataset.

The assembled directory uses relative symlinks, so the original datasets must
remain in output/data. No simulations, checkpoints, or graphs are copied.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output" / "data"
LOW = "n1000_random_connected_tree_augmented_aspl1to1p75_step025_rep400_random_initial"
HIGH = "n1000_random_connected_tree_augmented_aspl_step025_rep400_random_initial_full_corrected"
COMBINED = "n1000_random_connected_tree_augmented_aspl1to6_rep400_random_initial"
DESTINATION = DATA / COMBINED


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Existing metadata differs; refusing to overwrite: {path}")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def link_directory(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting link: {destination}")
        return
    if destination.exists():
        raise RuntimeError(f"Existing non-link directory: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


def main() -> None:
    low = read_json(DATA / LOW / "experiment_parameters.json")
    high = read_json(DATA / HIGH / "experiment_parameters.json")
    common = (
        "community_size", "k_values", "beta", "gamma", "model", "tau_micro",
        "macro_T", "T_max", "graph_lineages", "interfaces_per_lineage",
        "event_replicates_per_interface", "replications_per_scenario", "topology",
        "independent_rng_per_micro_model", "initial_node_policy",
    )
    mismatches = [key for key in common if low.get(key) != high.get(key)]
    if mismatches:
        raise RuntimeError(f"Incompatible input parameters: {mismatches}")
    low_targets = [float(value) for value in low["ASPL_targets"]]
    high_targets = [float(value) for value in high["ASPL_targets"]]
    targets = sorted(low_targets + high_targets)
    if len(targets) != len(set(targets)) or targets != [1.0 + 0.25 * i for i in range(21)]:
        raise RuntimeError("Expected disjoint ASPL targets 1.00 through 6.00 at step 0.25")
    high_run = read_json(DATA / HIGH / "run_manifest.json")
    if high_run.get("status") != "complete" or high_run.get("completed_pairs") != 13600:
        raise RuntimeError("The corrected high-ASPL dataset is not complete")
    low_run = read_json(DATA / LOW / "run_manifest.json")
    if len(low_run.get("scenarios", [])) != 8 or any(
        entry.get("status") != "complete" or entry.get("full_design_pairs") != 400
        for entry in low_run["scenarios"]
    ):
        raise RuntimeError("The low-ASPL dataset is not complete")

    high_graph_source = DATA / high["source_folder"].split("/")[-1]
    if not high_graph_source.is_dir():
        raise FileNotFoundError(f"Corrected checkpoints require source graphs: {high_graph_source}")
    ranges = ((LOW, DATA / LOW, low_targets), (HIGH, DATA / HIGH, high_targets))
    scenarios = DESTINATION / "scenarios"
    graphs = DESTINATION / "graphs"
    scenarios.mkdir(parents=True, exist_ok=True)
    graphs.mkdir(parents=True, exist_ok=True)
    linked_scenarios = linked_graph_groups = 0
    for name, folder, source_targets in ranges:
        expected_scenarios = len(source_targets) * len(low["k_values"])
        source_scenarios = list((folder / "scenarios").iterdir())
        if len(source_scenarios) != expected_scenarios:
            raise RuntimeError(f"Wrong scenario count in {name}: {len(source_scenarios)}")
        for source in source_scenarios:
            link_directory(source, scenarios / source.name)
            linked_scenarios += 1
        graph_folder = DATA / LOW if name == LOW else high_graph_source
        source_groups = list((graph_folder / "graphs").iterdir())
        if len(source_groups) != len(source_targets):
            raise RuntimeError(f"Wrong graph-group count in {name}: {len(source_groups)}")
        for source in source_groups:
            link_directory(source, graphs / source.name)
            linked_graph_groups += 1

    parameters = dict(low)
    parameters.update({
        "ASPL_targets": targets,
        "ASPL_tolerance": {**low["ASPL_tolerance"], **high["ASPL_tolerance"]},
        "mean_degree_by_target_ASPL": {
            **low["mean_degree_by_target_ASPL"], **high["mean_degree_by_target_ASPL"]
        },
        "checkpoint_experiment_by_ASPL": {
            **{str(value): LOW for value in low_targets},
            **{str(value): HIGH for value in high_targets},
        },
        "output_folder": f"output/data/{COMBINED}",
        "assembly_policy": "relative symlinks; source datasets and original high-ASPL graphs must remain available",
    })
    stable_json(DESTINATION / "experiment_parameters.json", parameters)
    stable_json(DESTINATION / "assembly_manifest.json", {
        "combined_experiment": COMBINED,
        "ASPL_1_to_1p75": LOW,
        "ASPL_2_to_6": HIGH,
        "high_ASPL_graph_source": high_graph_source.name,
        "expected_pairs": 16800,
        "scenario_links": linked_scenarios,
        "graph_group_links": linked_graph_groups,
        "note": "ASPL 2–6 uses corrected Full Micro and unchanged independent-RNG MicroMacro",
    })
    print(f"Assembled {linked_scenarios} scenarios and {linked_graph_groups} graph groups in {DESTINATION}")
    print("The source folders must remain available; no simulations were run.")


if __name__ == "__main__":
    main()
