from __future__ import annotations

import argparse
from pathlib import Path

from .config import derive_seed
from .runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 05 spectral corrections")
    parser.add_argument("--stage", choices=("smoke", "validation", "final"), required=True)
    parser.add_argument(
        "--scenario", choices=("baseline", "k1", "k10", "d4", "d10", "all"), required=True
    )
    parser.add_argument("--replications", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validation-approved", action="store_true")
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/experiment_05_spectral_corrections")
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest = run_experiment(
        stage=args.stage,
        scenario=args.scenario,
        replications=args.replications,
        resume=args.resume,
        validation_approved=args.validation_approved,
        master_seed=args.master_seed,
        output_root=args.output_root,
        repo_root=repo_root,
    )
    from .report import compile_report, render_report
    tex_path = render_report(
        output_root=args.output_root,
        report_dir=repo_root / "reports" / "experiment_05_spectral_corrections",
        stage=args.stage,
        horizon=4000.0,
        bootstrap_reps=1000,
        bootstrap_seed=derive_seed(args.master_seed, 0, "bootstrap_rng", "ANALYSIS"),
        max_replication_id=args.replications,
    )
    compile_report(tex_path)
    print(f"Experiment 05 {args.stage} complete: {manifest['complete']}")


if __name__ == "__main__":
    main()
