#!/usr/bin/env python3
"""Simple paired sweep: Full Micro vs Production MicroMacro with independent RNG.

Edit the arrays in the first section, then run:
    python extrascripts/runners/run_simple_topology_sweep.py

Re-run the same command after interruption: completed replications are skipped.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic

from extrascripts.runners import run_original_benchmarks_independent_rng as base


# ======================= EDIT THESE PARAMETERS =======================
COMMUNITIES = 10
SIZES = [50, 100, 250, 1000]                 # nodes in each community
DEGREES = [8, 10, 20, "complete"]            # random_connected mean degree, or "complete"
K_VALUES = [1, 5, 10, 20]                    # inter-community links per adjacent pair
BETAS = [0.003]                               # infection rates to compare
REPLICATIONS = 100                            # paired networks per (n, d, k, beta)

T_END = 10_000.0
TAU_MICRO = 1.0                               # fixed Production MicroMacro sync interval
WORKERS = 1                                   # this simple runner is deliberately sequential
BASE_SEED = 20260909
OUTPUT_DIR = Path("output/data")
# =====================================================================


def scenario_id(n: int, d: int | str, k: int, beta: float) -> str:
    beta_text = str(beta).replace(".", "p")
    topology = "chain_cliques" if d == "complete" else "chain_random_connected"
    degree = "dcomplete" if d == "complete" else f"d{d}"
    return f"{topology}_L{COMMUNITIES}_n{n}_{degree}_k{k}_beta{beta_text}_independent_rng"


def seed_for(scenario: str, replication: int, kind: str) -> int:
    text = f"{BASE_SEED}:{scenario}:{replication}:{kind}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big") % (2**31)


def write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=True)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def benchmark(n: int, d: int | str, k: int, beta: float):
    if d != "complete" and (not isinstance(d, int) or d < 2 or d >= n):
        raise ValueError(f"d={d} must be an integer in [2, n-1], or 'complete'")
    return replace(
        base.BENCHMARKS[1],
        name=scenario_id(n, d, k, beta),
        family="simple_topology_sweep",
        communities=COMMUNITIES,
        community_size=n,
        inter_links=k,
        macro_graph_type="chain",
        micro_graph_type="complete" if d == "complete" else "random_connected",
        edge_prob=1.0 if d == "complete" else d / (n - 1),
        mean_internal_degree=None if d == "complete" else d,
        beta=beta,
        gamma=0.0,
        model=2,                         # SI
        tau_micro=TAU_MICRO,
        macro_T=1.0,
        T_end=T_END,
        original_T_end=T_END,
        dt_out=1.0,
        initial_node=0,
        store_trajectories=False,
        replications=REPLICATIONS,
    )


def all_scenarios():
    return [benchmark(n, d, k, beta)
            for n in SIZES for d in DEGREES for k in K_VALUES for beta in BETAS]


def rebuild_csv(output: Path, scenarios) -> None:
    runs, arrivals, failures = [], [], []
    for item in scenarios:
        directory = output / item.name / "checkpoints"
        for replication in range(REPLICATIONS):
            path = directory / f"rep_{replication:03d}.json.gz"
            if not path.exists():
                continue
            payload = read_checkpoint(path)
            if "error" in payload:
                failures.append(payload)
                continue
            runs.extend(payload["runs"])
            arrivals.extend(payload["community_arrivals"])
    for filename, rows in (("per_replication.csv", runs), ("arrival_by_community.csv", arrivals), ("failures.csv", failures)):
        path = output / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    if COMMUNITIES < 2 or REPLICATIONS < 1 or T_END <= 0 or TAU_MICRO <= 0:
        raise ValueError("COMMUNITIES, REPLICATIONS, T_END and TAU_MICRO must be positive")
    if WORKERS != 1:
        print("WORKERS is currently informational: this deliberately simple runner runs sequentially.")
    scenarios = all_scenarios()
    if len({item.name for item in scenarios}) != len(scenarios):
        raise ValueError("Duplicate scenario: check SIZES, DEGREES, K_VALUES and BETAS")

    output = OUTPUT_DIR.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "simple_topology_sweep_manifest.json"
    configuration = {
        "communities": COMMUNITIES, "sizes": SIZES, "degrees": DEGREES,
        "k_values": K_VALUES, "betas": BETAS, "replications": REPLICATIONS,
        "T_end": T_END, "tau_micro": TAU_MICRO, "model": "SI", "gamma": 0,
        "initial_node": 0, "base_seed": BASE_SEED,
        "simulators": [base.FULL, base.MICROMACRO],
        "note": "Every pair uses the same generated network. MicroMacro uses private RNG per MicroModel.",
    }
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != configuration:
        raise ValueError("Existing output uses different settings. Change OUTPUT_DIR or remove that output deliberately.")
    manifest.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    passed, detail = base._regression_gate()
    print(f"Regression gate: {'PASS' if passed else 'FAIL'} ({detail})", flush=True)
    if not passed:
        raise RuntimeError(detail)
    total = len(scenarios) * REPLICATIONS
    completed = 0
    started = monotonic()
    print(f"Scenarios: {len(scenarios)}; paired replications: {total}; workers: 1", flush=True)
    for item in scenarios:
        directory = output / "checkpoints" / item.name
        directory.mkdir(parents=True, exist_ok=True)
        for replication in range(REPLICATIONS):
            path = directory / f"rep_{replication:03d}.json.gz"
            if path.exists():
                completed += 1
                continue
            network_seed = seed_for(item.name, replication, "network")
            event_seed = seed_for(item.name, replication, "events")
            print(f"[{completed + 1}/{total}] {item.name} rep={replication}", flush=True)
            try:
                payload = base._run_task((asdict(item), replication, network_seed, event_seed))
            except Exception as error:
                payload = {"scenario": item.name, "replication_id": replication,
                           "network_seed": network_seed, "event_seed": event_seed,
                           "error": f"{type(error).__name__}: {error}"}
                print(f"  FAILED: {payload['error']}", flush=True)
            write_checkpoint(path, payload)
            completed += 1
            elapsed = monotonic() - started
            remaining = total - completed
            eta = elapsed / completed * remaining if completed else math.nan
            print(f"  done={completed}/{total}; remaining={remaining}; elapsed={elapsed/60:.1f} min; ETA={eta/60:.1f} min", flush=True)
    rebuild_csv(output, scenarios)
    print(f"Done. Data saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
