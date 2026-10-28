"""Small paired topology pilot; preserves individual results and explicit seeds.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for modest BLAS overhead.
No production sources or algorithms are modified.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from extrascripts.runners import run_original_benchmarks_independent_rng as runner


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Validate a common T=4000 with target Full median=500 at k=5")
    args = parser.parse_args()
    if args.compact:
        args.validate = True
    target = 500 if args.compact else 1500
    out = runner.REPO_ROOT / "output/topology_calibration_pilot"
    out.mkdir(parents=True, exist_ok=True)
    passed, reason = runner._regression_gate()
    print(f"Regression gate: {passed}: {reason}", flush=True)
    if not passed:
        raise RuntimeError(reason)
    base = runner.BENCHMARKS[1]
    calibrated = {}
    if args.validate:
        initial = json.loads((out / "results.json").read_text())
        for degree in (4, 10, "complete"):
            times = [r["T_last"] for p in initial if p["benchmark_config"]["inter_links"] == 5 and p["benchmark_config"]["name"].startswith(f"d{degree}_") for r in p["runs"] if r["simulator"] == runner.FULL and math.isfinite(r["T_last"])]
            if len(times) != 3:
                raise RuntimeError("Calibration requires three uncensored Full arrivals")
            calibrated[str(degree)] = round(0.003 * statistics.median(times) / target, 6)
    results = []
    phase = "compact" if args.compact else "validation" if args.validate else "screen"
    jobs = [(d, k, r) for d in (4, 10, "complete") for k in ((1, 20) if args.validate else (1, 5, 20)) for r in range(3)]
    for index, (degree, k, rep) in enumerate(jobs, 1):
        beta = calibrated[str(degree)] if args.validate else 0.003
        b = replace(base, name=f"d{degree}_k{k}_{phase}", family="topology_pilot", communities=10, community_size=100, inter_links=k, micro_graph_type="complete" if degree == "complete" else "random_connected", edge_prob=1.0 if degree == "complete" else degree / 99, mean_internal_degree=None if degree == "complete" else degree, beta=beta, gamma=0, model=2, tau_micro=1, macro_T=1, T_end=8000, original_T_end=8000, dt_out=10, initial_node=0, store_trajectories=False, replications=3)
        if args.compact:
            b = replace(b, T_end=4000, original_T_end=4000)
        path = out / f"{b.name}_{rep}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            if payload["benchmark_config"] != json.loads(json.dumps(asdict(b))):
                raise RuntimeError(f"Checkpoint config mismatch: {path}")
        else:
            print(f"[{index}/{len(jobs)}] {b.name} beta={beta} rep={rep} running", flush=True)
            offset = 10000 if args.validate else 0
            payload = runner._run_task((asdict(b), rep, 731000 + offset + rep, 831000 + offset + rep))
            validation = payload["network_validation"]
            if not (validation["topology_valid"] and validation["micro_graphs_connected"] and validation["initial_infection_valid"]):
                raise RuntimeError(f"Invalid pilot network: {validation}")
            path.write_text(json.dumps(payload, indent=2))
        results.append(payload)
        print(f"[{index}/{len(jobs)}] " + " | ".join(f"{r['simulator']}: T_last={r['T_last']:.1f}, {r['runtime_seconds']:.1f}s" for r in payload["runs"]), flush=True)
    (out / (f"{phase}.json" if args.validate else "results.json")).write_text(json.dumps(results, indent=2))
    rows = []
    for payload in results:
        for run in payload["runs"]:
            row = dict(run, beta=payload["benchmark_config"]["beta"])
            for horizon in (2500, 4000, 8000):
                row[f"arrival_by_{horizon}"] = (math.isfinite(run["T_last"]) and run["T_last"] <= horizon) if horizon <= run["T_end"] or math.isfinite(run["T_last"]) else None
            rows.append(row)
    with (out / f"{phase}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if calibrated:
        (out / f"{phase}_beta.json").write_text(json.dumps(dict(target_full_median_k5=target, beta=calibrated, note="Full SI time-scaling estimate; validation seeds differ from screening; compact repeats validation seeds with changed beta/horizon; fixed tau=1 in actual paired validation."), indent=2))


if __name__ == "__main__":
    main()
