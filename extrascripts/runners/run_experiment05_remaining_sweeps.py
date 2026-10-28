"""Resume Experiment 05 and complete the five-scenario grid at a rational N.

This runner deliberately stops after simulation, gates, and summary aggregation.
Report regeneration is deferred until the saved outputs have been inspected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from two_layer_ctmc.experiments.experiment_05_spectral_corrections.config import (
    ALL_VARIANTS,
    SCENARIOS,
)
from two_layer_ctmc.experiments.experiment_05_spectral_corrections.runner import (
    run_experiment,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete Experiment 05 k/d sweeps without rerunning saved replications."
    )
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--master-seed", type=int, default=270000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output" / "experiment_05_spectral_corrections",
    )
    parser.add_argument("--resume", action="store_true", required=True)
    return parser.parse_args()


def completed_counts(output_root: Path, target: int) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for scenario_id in SCENARIOS:
        counts[scenario_id] = {}
        for variant in ALL_VARIANTS:
            checkpoint = output_root / "checkpoints" / f"{scenario_id}_{variant.value}.json"
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            replication_ids = {int(value) for value in payload["replications"]}
            counts[scenario_id][variant.value] = sum(
                replication_id in replication_ids
                for replication_id in range(1, target + 1)
            )
    return counts


def main() -> None:
    args = parse_args()
    if args.replications < 100:
        raise ValueError("Experiment 05 sweeps require at least 100 paired replications")

    manifest = run_experiment(
        stage="final",
        scenario="all",
        replications=args.replications,
        resume=True,
        validation_approved=True,
        master_seed=args.master_seed,
        output_root=args.output_root,
        repo_root=ROOT,
    )
    counts = completed_counts(args.output_root, args.replications)
    expected = args.replications
    complete = all(
        count == expected
        for scenario_counts in counts.values()
        for count in scenario_counts.values()
    )
    completion = {
        "experiment": "experiment_05_spectral_corrections",
        "purpose": "complete k and d robustness sweeps",
        "target_paired_replications_per_scenario": expected,
        "scenarios": list(SCENARIOS),
        "variant_counts_within_target_range": counts,
        "complete": complete,
        "manifest_complete": bool(manifest.get("complete")),
        "report_regeneration_deferred": True,
    }
    destination = args.output_root / "sweep_100_completion.json"
    destination.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    if not complete:
        raise RuntimeError(f"incomplete target grid; inspect {destination}")


if __name__ == "__main__":
    main()
