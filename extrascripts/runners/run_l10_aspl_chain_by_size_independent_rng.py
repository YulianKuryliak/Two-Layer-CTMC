#!/usr/bin/env python3
"""Collect the L=10 ASPL-chain comparison for n=50, 100, and 250.

The n=1000 experiment has its own runner and is never selected here.  This
runner uses the same random-initial-node, independent-RNG, 40-network x
10-replicate protocol, with a separate resumable output folder per size.

Default full run:
    .venv/bin/python3 extrascripts/runners/run_l10_aspl_chain_by_size_independent_rng.py --workers 2

To run only one size, add ``--sizes 50`` (or 100 or 250).  The underlying
runner also accepts ``--topology-only`` and ``--smoke-only``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing this wrapper installs the established clique handling, random
# initial-node policy, and private RNG intervention on the base runner.
from extrascripts.runners import run_l10_aspl_chain_random_initial_independent_rng as configured

base = configured.base


# The degree values are realizable edge counts (2m/n).  They were calibrated
# with the same tree-plus-random-edges generator as the n=1000 experiment.
# The generator still checks every realized graph against target ASPL ±0.1.
DEGREE_BY_SIZE_AND_ASPL = {
    50: {1.0: 49.0, 2.0: 9.0, 3.0: 3.8, 4.0: 2.72, 5.0: 2.36, 6.0: 2.08},
    100: {1.0: 99.0, 2.0: 13.0, 3.0: 5.0, 4.0: 3.3, 5.0: 2.66, 6.0: 2.36},
    250: {1.0: 249.0, 2.0: 23.0, 3.0: 7.2, 4.0: 4.2, 5.0: 3.2, 6.0: 2.744},
}
SIZES = tuple(DEGREE_BY_SIZE_AND_ASPL)


def configure_size(size: int) -> None:
    base.N = size
    base.MEAN_DEGREE_BY_ASPL = DEGREE_BY_SIZE_AND_ASPL[size].copy()
    base.ASPL_TARGETS = tuple(sorted(base.MEAN_DEGREE_BY_ASPL))
    base.ASPL_TOLERANCE = {target: 0.1 for target in base.ASPL_TARGETS}
    base.L = 10
    base.K_VALUES = (1, 5)
    base.BETA = 0.003
    base.TAU_MICRO = 1.0
    base.T_MAX = 30_000.0
    base.NETWORKS = 40
    base.REPLICATES_PER_NETWORK = 10
    base.REPLICATIONS = 400
    base.WORKERS = 2
    base.RANDOM_INITIAL_NODE = True
    base.MAX_GRAPH_ATTEMPTS = 300
    base.EXPERIMENT = (
        f"l10_n{size}_chain_clique_tree_augmented_aspl1to6_step1_"
        "rep400_random_initial_independent_rng"
    )
    base.DATA_ROOT = ROOT / "output" / "data" / base.EXPERIMENT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES),
                        help="community sizes to collect; default: 50 100 250")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-regression-gate", action="store_true", help="debug only")
    args = parser.parse_args()
    if len(set(args.sizes)) != len(args.sizes) or any(size not in SIZES for size in args.sizes):
        parser.error("--sizes must contain distinct values from 50, 100, 250; n=1000 is separate")
    if args.workers < 1:
        parser.error("--workers must be positive")
    original_argv = sys.argv
    try:
        child_argv = [original_argv[0], "--workers", str(args.workers)]
        if args.topology_only:
            child_argv.append("--topology-only")
        if args.smoke_only:
            child_argv.append("--smoke-only")
        if args.skip_regression_gate:
            child_argv.append("--skip-regression-gate")
        sys.argv = child_argv
        for index, size in enumerate(args.sizes, 1):
            configure_size(size)
            print(f"[size {index}/{len(args.sizes)}] n={size}; output={base.DATA_ROOT}", flush=True)
            base.main()
            print(f"[size complete] n={size}", flush=True)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
