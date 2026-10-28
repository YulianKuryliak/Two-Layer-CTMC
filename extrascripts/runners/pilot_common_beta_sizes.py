"""Bounded common-beta pilot, paired on one network per size/topology/seed.

Default: one screening seed per cell, with per-cell checkpoints and budgets.
This is a feasibility pilot, not an accuracy or tail-probability estimate.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import csv
import hashlib
import json
import math
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

from extrascripts.runners import run_original_benchmarks_independent_rng as r

OUT = r.REPO_ROOT / "output/common_beta_sizes_pilot"


def save(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temp.replace(path)


def timeout_handler(*_):
    raise TimeoutError("Stage wall-time budget exceeded")


def worker(n, degree, rep):
    directory = OUT / f"n{n}_d{degree}_r{rep}"
    directory.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, timeout_handler)
    b = replace(r.BENCHMARKS[1], name=directory.name, family="common_beta_sizes", communities=10,
                community_size=n, inter_links=1, micro_graph_type="complete" if degree == "complete" else "random_connected",
                edge_prob=1 if degree == "complete" else int(degree)/(n-1), mean_internal_degree=None if degree == "complete" else int(degree),
                gamma=0, model=2, beta=0.003, tau_micro=1, macro_T=1, T_end=4000, original_T_end=4000,
                dt_out=1, initial_node=0, store_trajectories=False, replications=1)
    network_seed, event_seed = 951000 + rep, 961000 + rep
    meta = dict(config=asdict(b), network_seed=network_seed, event_seed=event_seed,
                status="generating_network", network_budget_seconds=45, simulation_budget_seconds=120)
    save(directory / "status.json", meta)
    started = time.perf_counter()
    try:
        signal.alarm(45)
        graphs, full, W = r._generate_network(b, network_seed)
        signal.alarm(0)
        meta["generation_seconds"] = time.perf_counter() - started
        meta["network_validation"] = r._validate_network(b, graphs, full, W)
        if not meta["network_validation"]["topology_valid"] or not meta["network_validation"]["micro_graphs_connected"]:
            raise RuntimeError("Network validation failed")
        meta["actual_degrees"] = [2*g.number_of_edges()/g.number_of_nodes() for g in graphs]
    except Exception as exc:
        signal.alarm(0)
        meta.update(status="network_failed", reason=f"{type(exc).__name__}: {exc}", generation_seconds=time.perf_counter()-started)
        save(directory / "status.json", meta)
        print(f"{directory.name}: {meta['reason']}", flush=True)
        return
    schedule = [(beta, 1.0, sim) for beta in (0.003, 0.01, 0.03) for sim in (r.FULL, r.MICROMACRO)]
    schedule += [(0.01, 0.3, r.MICROMACRO), (0.03, 0.1, r.MICROMACRO)]
    for beta, tau, simulator in schedule:
        label = "full" if simulator == r.FULL else "mm"
        path = directory / f"{label}_b{beta}_tau{tau}.json"
        if path.exists():
            continue
        meta.update(status="simulating", current=dict(beta=beta, tau=tau, simulator=simulator))
        save(directory / "status.json", meta)
        conf = replace(b, beta=beta, tau_micro=tau)
        row = dict(n=n, degree=degree, k=1, replication=rep, simulator=simulator,
                   beta=beta, tau=tau, macro_T=1, T_end=4000, network_seed=network_seed, event_seed=event_seed)
        started = time.perf_counter()
        try:
            signal.alarm(120)
            if simulator == r.FULL:
                result = r.MicroSimulator(full_graph=full, comm_nodes=[list(g.nodes()) for g in graphs],
                    infection_rate=beta, recovery_rate=0, model=2).run(T_end=4000, dt_out=1,
                    initial_node=0, rng=random.Random(event_seed))
                arrivals = r._arrivals_from_full(result, 10, 4000)
            else:
                result = r._run_experimental_micromacro(conf, graphs, full, W, event_seed, independent=True)
                arrivals = r._arrivals_from_events(result[3], 10, 4000)
            signal.alarm(0)
            row.update(status="ok", arrivals=[v if math.isfinite(v) else None for v in arrivals],
                       completion=all(math.isfinite(v) for v in arrivals),
                       T_last=max(arrivals) if all(math.isfinite(v) for v in arrivals) else None)
            del result
        except Exception as exc:
            signal.alarm(0)
            row.update(status="failed", reason=f"{type(exc).__name__}: {exc}", completion=None, T_last=None)
        row["runtime_seconds"] = time.perf_counter()-started
        save(path, row)
        print(f"{directory.name}: {label} beta={beta} tau={tau} {row['status']} T_last={row['T_last']} runtime={row['runtime_seconds']:.1f}s", flush=True)
    meta.update(status="finished")
    save(directory / "status.json", meta)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cell", nargs=2, metavar=("N", "DEGREE"))
    parser.add_argument("--rep", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cell:
        worker(int(args.cell[0]), args.cell[1], args.rep)
        return
    passed, reason = r._regression_gate()
    print(f"Regression gate: {passed}: {reason}", flush=True)
    if not passed:
        raise RuntimeError(reason)
    cells = [(n, d) for n in (50,100,250,1000) for d in (4,10,"complete")] + [(1000,6)]
    sources = [Path(__file__), Path(r.__file__)] + list((r.REPO_ROOT / "two_layer_ctmc").rglob("*.py"))
    manifest = dict(cells=cells, screening_replication=args.rep, rates=[0.003,0.01,0.03], horizon=4000,
                    regression=dict(passed=passed, reason=reason), cell_budget_seconds=300,
                    source_sha256={str(p.relative_to(r.REPO_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    note="One screening replication per cell. Production algorithms unchanged. No full trajectories; arrivals saved. tau control holds beta*tau, not macro_T. Network failures are not epidemic censoring.")
    manifest_path = OUT / f"manifest_r{args.rep}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("Sources changed since pilot manifest; use a fresh output directory")
    save(manifest_path, manifest)
    for index, (n, d) in enumerate(cells, 1):
        status_path = OUT / f"n{n}_d{d}_r{args.rep}" / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())["status"] in ("finished", "network_failed", "cell_timeout"):
            print(f"[{index}/{len(cells)}] reuse {status_path.parent.name}", flush=True)
            continue
        print(f"[{index}/{len(cells)}] n={n}, d={d}, k=1", flush=True)
        try:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--cell", str(n), str(d), "--rep", str(args.rep)], timeout=300, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            meta = json.loads(status_path.read_text()) if status_path.exists() else {}
            meta.update(status="cell_timeout" if isinstance(exc, subprocess.TimeoutExpired) else "process_failed", reason=str(exc))
            status_path.parent.mkdir(parents=True, exist_ok=True)
            save(status_path, meta)
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("n*/[fm]*_b*.json"))]
    fields = sorted({k for row in rows for k in row if k != "arrivals"})
    with (OUT / "runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} simulation records to {OUT}", flush=True)


if __name__ == "__main__":
    main()
