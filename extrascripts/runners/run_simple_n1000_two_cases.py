#!/usr/bin/env python3
"""Edit parameters below. Run with --resume to skip smoke tests.

Uses run_original_benchmarks_independent_rng.py, including trajectory output.
"""
from pathlib import Path
import json
import subprocess
import sys


# ======================= EDIT THESE PARAMETERS =======================
COMMUNITIES = 10
SIZES = [1000]                  # nodes per community
DEGREES = ["complete", 10]     # complete or random_connected mean degree
K_VALUES = [5]                  # links per adjacent pair of communities
BETAS = [0.003]
REPLICATIONS = 10
T_END = 4000.0
TAU_MICRO = 1.0
DT_OUT = 1.0
INITIAL_NODE = 0
WORKERS = 1
BASE_SEED = 20260910
OUTPUT_DIR = "output/data"
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_RUNNER = ROOT / "extrascripts/runners/run_original_benchmarks_independent_rng.py"


def main() -> None:
    scenarios = []
    for n in SIZES:
        for d in DEGREES:
            for k in K_VALUES:
                for beta in BETAS:
                    graph = "complete" if d == "complete" else "random_connected"
                    degree_tag = "" if d == "complete" else f"_d{d}"
                    beta_tag = f"{beta:g}".removeprefix("0.")
                    seed = BASE_SEED + 2000 * len(scenarios)
                    scenarios.append({
                        "name": f"chain_{graph}_L{COMMUNITIES}_n{n}{degree_tag}_k{k}_beta{beta_tag}_independent_rng",
                        "family": f"n{n}_{graph}{degree_tag}",
                        "community_size": n, "inter_links": k, "beta": beta,
                        "micro_graph_type": graph,
                        "edge_prob": 1.0 if d == "complete" else d / (n - 1),
                        "mean_internal_degree": None if d == "complete" else d,
                        "base_seed": seed, "network_seed_base": seed + 1000,
                    })
    output = ROOT / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    config = output / "n1000_two_cases_runner_config.json"
    config.write_text(json.dumps({
        "replications_default": REPLICATIONS,
        "defaults": {
            "communities": COMMUNITIES, "macro_graph_type": "chain",
            "gamma": 0.0, "model": 2, "macro_T": 1.0,
            "tau_micro": TAU_MICRO, "T_end": T_END, "dt_out": DT_OUT,
            "initial_node": INITIAL_NODE, "store_trajectories": True,
        },
        "scenarios": scenarios,
    }, indent=2), encoding="utf-8")
    command = [
        sys.executable,
        str(ORIGINAL_RUNNER),
        "--config", str(config),
        "--output-dir", str(output),
        "--workers", str(WORKERS),
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
