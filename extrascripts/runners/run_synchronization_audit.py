"""Production midpoint vs event-driven synchronization audit.

The network and initial condition are matched within replication.  Production
MicroMacro is evaluated at three synchronization steps; the event-driven
aggregate/uniform scheduler is the numerical reference for the same aggregate
hazard model.  Outputs contain arrivals only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extrascripts.analysis.missing_experiment_utils import (
    arrival_rows, load_completed_json, run_production_mm, sha256_file,
    summarize_arrivals, write_json_atomic,
)
from two_layer_ctmc.network import generate_two_scale_network


TAUS = (0.25, 0.5, 1.0)


def run_one(task: tuple[int, int, int, float]) -> dict:
    replication_id, network_seed, base_seed, horizon = task
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=10, community_size=100, inter_links=5,
        seed=network_seed, macro_graph_type="chain",
        micro_graph_type="random_connected", edge_prob=6.0 / 99.0,
    )
    communities = [list(graph.nodes()) for graph in micro_graphs]
    initial_node = int(communities[0][0])
    rows = []
    for tau in TAUS:
        seed = base_seed + 200_000 + replication_id - 1
        rows.extend(arrival_rows(
            replication_id, "sync_audit", f"ProductionMM_tau{tau:g}",
            run_production_mm(W, full_graph, micro_graphs, 0.003, horizon, tau,
                              initial_node, seed),
            horizon, tau_micro=tau, network_seed=network_seed,
        ))

    return {"replication_id": replication_id, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/missing_experiments/synchronization_audit"))
    parser.add_argument("--source", type=Path,
                        default=Path("output/ablation_hazard_placement/final_100"))
    args = parser.parse_args()
    n_runs = 3 if args.grid == "smoke" else 100
    base_seed, horizon = 1_310_000, 4000.0
    run_dir = args.output_dir / args.grid / "replications"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.grid / "manifest.json"
    manifest = {
        "experiment": "production_vs_event_driven_synchronization_audit",
        "grid": args.grid, "n_runs": n_runs, "base_seed": base_seed,
        "T_end": horizon, "tau_micro": list(TAUS),
        "models": ["FullMicro", "EventDrivenAggregate",
                   *[f"ProductionMM_tau{tau:g}" for tau in TAUS]],
        "source": str(args.source),
        "source_reuse": ["FullMicro", "EventDrivenAggregate"],
        "source_sha256": {
            "arrival_times.csv": sha256_file(args.source / "arrival_times.csv"),
            "runs.csv": sha256_file(args.source / "runs.csv"),
        },
        "runner_sha256": sha256_file(Path(__file__)),
        "pairing": "shared network and initial node; deterministic model streams",
        "full_trajectory_policy": "not saved; compact arrival records only",
        "complete": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(manifest_path, manifest)
    source_arrivals = pd.read_csv(args.source / "arrival_times.csv")
    source_runs = pd.read_csv(args.source / "runs.csv")
    source_names = {
        "FULL_MICRO_REFERENCE": "FullMicro",
        "AGGREGATE_HAZARD_UNIFORM_PLACEMENT": "EventDrivenAggregate",
    }
    archived = source_arrivals[
        source_arrivals.replication_id.between(1, n_runs)
        & source_arrivals.model_variant.isin(source_names)
    ].copy()
    archived["model"] = archived.model_variant.map(source_names)
    archived["scenario"] = "sync_audit"
    archived["layer"] = archived.layer_index.astype(int)
    archived["observed"] = archived.T_i.notna()
    archived["restricted_T_i"] = archived.T_i.fillna(horizon)
    network_seeds = (source_runs[source_runs.model_variant == "FULL_MICRO_REFERENCE"]
                     .set_index("replication_id").network_seed.astype(int).to_dict())
    pending = [rep for rep in range(1, n_runs + 1)
               if not (args.resume and (run_dir / f"rep_{rep:04d}.json").exists())]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (rep, network_seeds[rep], base_seed, horizon)): rep
                   for rep in pending}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            payload = future.result()
            write_json_atomic(run_dir / f"rep_{payload['replication_id']:04d}.json", payload)
            print(f"[sync:{args.grid}] {done}/{len(pending)} new replications", flush=True)
    rows = archived[["replication_id", "scenario", "model", "layer", "T_i",
                     "observed", "restricted_T_i"]].to_dict("records")
    rows.extend(load_completed_json(run_dir))
    expected = n_runs * (2 + len(TAUS)) * 10
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} arrival rows, found {len(rows)}")
    pd.DataFrame(rows).to_csv(args.output_dir / args.grid / "arrivals.csv", index=False)
    summaries = []
    for reference in ("FullMicro", "EventDrivenAggregate"):
        summary = summarize_arrivals(rows, reference, horizon, 1000, base_seed + len(reference))
        summary.insert(0, "reference", reference)
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output_dir / args.grid / "summary.csv", index=False)
    manifest["complete"] = True
    manifest["completed_replications"] = n_runs
    manifest["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
