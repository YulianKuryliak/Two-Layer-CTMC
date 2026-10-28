"""Oracle source-bridge and deployable degree-proxy hazard ceiling.

Full Micro and aggregate/uniform baseline arrivals are reused from the paired
Experiment 01 dataset.  Only the two new uniform-placement hazard variants are
simulated, which avoids repeating identical expensive arms.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.analysis.missing_experiment_utils import (
    arrival_rows, load_completed_json, sha256_file, summarize_arrivals,
    write_json_atomic,
)
from two_layer_ctmc.experiments.hazard_placement_ablation.config import (
    ExperimentConfig, ModelVariant, stream_seeds,
)
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import (
    HazardPlacementSimulator, INFECTED,
)
from two_layer_ctmc.network import generate_two_scale_network


class OracleSourceBridgeSimulator(HazardPlacementSimulator):
    """Source bridge availability with uniform susceptible target placement."""

    def exact_pair_hazards(self, state):
        counts = self._community_counts(state)
        hazards = {}
        for pair, edges in self.index.directed_edges.items():
            source_weight = sum(weight for source, _, weight in edges
                                if state[source] == INFECTED)
            target = pair[1]
            susceptible_fraction = counts[target][0] / len(self.index.communities[target])
            hazards[pair] = self.config.beta_out * source_weight * susceptible_fraction
        return hazards


class DegreeProxySimulator(HazardPlacementSimulator):
    """Within-degree-weighted source prevalence with uniform target placement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.within_degree = {node: 0.0 for node in self.index.node_to_community}
        for u, v, weight in self.index.internal_edges:
            self.within_degree[u] += weight
            self.within_degree[v] += weight

    def exact_pair_hazards(self, state):
        counts = self._community_counts(state)
        q_by_community = {}
        for community, nodes in enumerate(self.index.communities):
            denominator = sum(self.within_degree[node] for node in nodes)
            numerator = sum(self.within_degree[node] for node in nodes
                            if state[node] == INFECTED)
            q_by_community[community] = numerator / denominator if denominator else 0.0
        hazards = {}
        for pair, weight_sum in self.index.W.items():
            target = pair[1]
            susceptible_fraction = counts[target][0] / len(self.index.communities[target])
            hazards[pair] = (self.config.beta_out * weight_sum
                             * q_by_community[pair[0]] * susceptible_fraction)
        return hazards


def run_one(task: tuple[int, int, int, int, float]) -> dict:
    replication_id, network_seed, initial_node, base_seed, horizon = task
    micro_graphs, full_graph, _ = generate_two_scale_network(
        n_communities=10, community_size=100, inter_links=5,
        seed=network_seed, macro_graph_type="chain",
        micro_graph_type="random_connected", edge_prob=6.0 / 99.0,
    )
    communities = [list(graph.nodes()) for graph in micro_graphs]
    config = ExperimentConfig(
        experiment_name="oracle_bridge_ceiling", L=10, n=100,
        mean_internal_degree=6.0, k=5, beta_in=0.003, beta_out=0.003,
        gamma=0.0, T_end=horizon, base_seed=base_seed, initial_node=initial_node,
    )
    variant = ModelVariant.ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT
    rows = []
    for name, simulator_type in (
        ("MM_oracle_source_bridge", OracleSourceBridgeSimulator),
        ("MM_degree_proxy", DegreeProxySimulator),
    ):
        simulator = simulator_type(
            full_graph, communities, config, variant,
            stream_seeds(config, replication_id, variant),
            record_events=False, record_diagnostics=False, record_trajectories=False,
        )
        result = simulator.run(initial_node)
        rows.extend(arrival_rows(
            replication_id, "oracle_bridge_ceiling", name, result["arrivals"], horizon,
            network_seed=network_seed,
        ))
    return {"replication_id": replication_id, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source", type=Path,
                        default=Path("output/ablation_hazard_placement/final_100"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/missing_experiments/oracle_bridge_ceiling"))
    args = parser.parse_args()
    n_runs = 3 if args.grid == "smoke" else 100
    base_seed, horizon = 1_410_000, 4000.0
    target = args.output_dir / args.grid
    run_dir = target / "replications"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_arrivals = pd.read_csv(args.source / "arrival_times.csv")
    source_runs = pd.read_csv(args.source / "runs.csv")
    source_names = {
        "FULL_MICRO_REFERENCE": "FullMicro",
        "AGGREGATE_HAZARD_UNIFORM_PLACEMENT": "MM_uniform",
    }
    archived = source_arrivals[
        source_arrivals.replication_id.between(1, n_runs)
        & source_arrivals.model_variant.isin(source_names)
    ].copy()
    archived["model"] = archived.model_variant.map(source_names)
    archived["scenario"] = "oracle_bridge_ceiling"
    archived["layer"] = archived.layer_index.astype(int)
    archived["observed"] = archived.T_i.notna()
    archived["restricted_T_i"] = archived.T_i.fillna(horizon)
    reference_runs = source_runs[source_runs.model_variant == "FULL_MICRO_REFERENCE"].set_index("replication_id")
    manifest_path = target / "manifest.json"
    manifest = {
        "experiment": "oracle_source_bridge_correction_ceiling",
        "grid": args.grid, "n_runs": n_runs, "T_end": horizon,
        "source": str(args.source), "source_reuse": ["FullMicro", "MM_uniform"],
        "source_sha256": {
            "arrival_times.csv": sha256_file(args.source / "arrival_times.csv"),
            "runs.csv": sha256_file(args.source / "runs.csv"),
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "models": ["FullMicro", "MM_uniform", "MM_oracle_source_bridge", "MM_degree_proxy"],
        "oracle_formula": "beta*sum_cross_edge_weight_from_infected_sources*(S_j/N_j)",
        "proxy_formula": "beta*W_ij*degree_weighted_source_prevalence*(S_j/N_j)",
        "placement": "uniform susceptible in every MicroMacro arm",
        "pairing": "shared archived network and initial node; deterministic variant streams",
        "full_trajectory_policy": "not saved; compact arrival records only",
        "complete": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(manifest_path, manifest)
    pending = [rep for rep in range(1, n_runs + 1)
               if not (args.resume and (run_dir / f"rep_{rep:04d}.json").exists())]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_one, (
                rep, int(reference_runs.loc[rep, "network_seed"]),
                int(reference_runs.loc[rep, "initial_node"]), base_seed, horizon)): rep
            for rep in pending
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            payload = future.result()
            write_json_atomic(run_dir / f"rep_{payload['replication_id']:04d}.json", payload)
            print(f"[oracle-bridge:{args.grid}] {done}/{len(pending)} new replications", flush=True)
    rows = archived[["replication_id", "scenario", "model", "layer", "T_i",
                     "observed", "restricted_T_i"]].to_dict("records")
    rows.extend(load_completed_json(run_dir))
    expected = n_runs * 4 * 10
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} arrival rows, found {len(rows)}")
    pd.DataFrame(rows).to_csv(target / "arrivals.csv", index=False)
    summary = summarize_arrivals(rows, "FullMicro", horizon, 2000, base_seed)
    baseline = abs(float(summary.loc[summary.model == "MM_uniform", "delta_RMST"].iloc[0]))
    summary["gap_closure_vs_MM_uniform"] = summary.delta_RMST.map(
        lambda value: (baseline - abs(float(value))) / baseline if baseline > 1e-12 else float("nan"))
    summary.to_csv(target / "summary.csv", index=False)
    manifest["complete"] = True
    manifest["completed_replications"] = n_runs
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
