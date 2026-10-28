"""Check uniform import placement and residual waits in the actual local engine."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from two_layer_ctmc.engines.micro_model import MicroModel
from extrascripts.runners.rerun_smpt_paper import code_hashes, object_hash, log, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=710000)
    parser.add_argument("--output", type=Path, default=ROOT / "output/validation/residual_import_clock_20261007")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("Choose a new output folder; previous diagnostic results are immutable")
    import_time = math.log(4 / 3)
    source = code_hashes()
    import hashlib
    source[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    context = dict(experiment="residual_import_clock", attempts=args.runs,
                   seed_start=args.seed, seed_end=args.seed + args.runs - 1,
                   beta=1., gamma=0., graph="0--1--2", initial_node=0,
                   import_time=import_time, code_sha256=source,
                   code_fingerprint=object_hash(source), expected_node2_probability=.5,
                   expected_mean_wait=.75)
    log(args.output, "diagnostic_started", **context)
    try:
        targets, waits = [], []
        for seed in range(args.seed, args.seed + args.runs):
            model = MicroModel(1, 0, rng=random.Random(seed))
            for node in range(3):
                model.add_node(node)
            model.add_edge(0, 1, 1)
            model.add_edge(1, 2, 1)
            model._infect_node(0)
            if model.simulate_until(import_time)[3]:
                continue
            old_wait = model._cached_wait
            target = model.import_infection(at_time=import_time)
            wait = model._ensure_next_event()[0]
            expected_wait = (old_wait - import_time) / model.total_infection_rate
            if not math.isclose(wait, expected_wait, rel_tol=1e-12, abs_tol=1e-12):
                raise AssertionError("Residual budget was not preserved")
            targets.append(target)
            waits.append(wait)
        count = len(waits)
        if count < 2:
            raise ValueError("Not enough conditioned samples")
        fraction = float(np.mean(np.asarray(targets) == 2))
        mean = float(np.mean(waits))
        se = float(np.std(waits, ddof=1) / math.sqrt(count))
        checks = {"uniform_placement_within_5se": abs(fraction-.5) <= 5*math.sqrt(.25/count),
                  "correct_wait_mean_within_5se": abs(mean-.75) <= 5*se,
                  "residual_formula_all_samples": True}
        result = {**context, "completion": "finished", "conditioned_runs": count,
                  "fraction_node2": fraction, "mean_wait": mean,
                  "mean_wait_ci95": [mean-1.96*se, mean+1.96*se], "checks": checks,
                  "scope": "Isolated native community engine with independent exogenous import; does not validate finite-tau midpoint macro scheduler."}
        atomic_json(args.output / "manifest.json", result)
        log(args.output, "diagnostic_completed", **result)
        print(json.dumps({key: result[key] for key in ("conditioned_runs", "fraction_node2", "mean_wait", "mean_wait_ci95", "checks")}, indent=2))
        if not all(checks.values()):
            raise AssertionError("Statistical diagnostic outside predeclared 5-SE bounds")
    except BaseException as error:
        log(args.output, "diagnostic_failed", error=repr(error), **context)
        raise


if __name__ == "__main__":
    main()
