from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import List

import networkx as nx
import numpy as np
import pandas as pd

from ...network import generate_two_scale_network
from ..hazard_placement_ablation.config import (
    ExperimentConfig as Experiment01Config,
    ModelVariant,
    shared_initialization_seed,
    shared_network_seed,
    stream_seeds,
)
from .analysis import build_summaries, create_figures
from .config import AuditConfig, STAGE_RUNS
from .replay import replay_replication


FULL = ModelVariant.FULL_MICRO_REFERENCE.value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_config(config: AuditConfig, source: Path) -> dict:
    manifest_path = source / "experiment_config.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Experiment 01 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = config.to_dict()
    for key in ("L", "n", "mean_internal_degree", "k", "beta_in", "beta_out", "gamma", "T_end", "base_seed", "initial_node"):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"Experiment 01 mismatch for {key}: {manifest.get(key)} != {expected[key]}")
    if not manifest.get("complete") or int(manifest.get("n_runs", 0)) < 100:
        raise ValueError("canonical Experiment 01 final_100 is not complete")
    return manifest


def _load_source_replication(source: Path, replication_id: int) -> tuple[pd.Series, pd.DataFrame, dict]:
    run_dir = source / "raw" / f"run_{replication_id:04d}"
    if not (run_dir / "complete.json").exists():
        raise FileNotFoundError(f"missing Experiment 01 checkpoint {run_dir}")
    runs = pd.read_csv(run_dir / "runs.csv")
    selected = runs[runs["model_variant"] == FULL]
    if len(selected) != 1:
        raise ValueError(f"replication {replication_id}: expected one Full Micro run row")
    events = pd.read_csv(run_dir / "event_log.csv")
    events = events[events["model_variant"] == FULL].copy()
    if events.empty:
        raise ValueError(f"replication {replication_id}: no Full Micro events")
    provenance = {
        "source_run_file_sha256": _sha256(run_dir / "runs.csv"),
        "source_event_file_sha256": _sha256(run_dir / "event_log.csv"),
        "source_checkpoint_sha256": _sha256(run_dir / "complete.json"),
    }
    return selected.iloc[0], events, provenance


def _expected_seed_checks(config: AuditConfig, replication_id: int, run: pd.Series) -> None:
    exp01 = Experiment01Config()
    expected = {
        "network_seed": shared_network_seed(exp01, replication_id),
        "initialization_seed": shared_initialization_seed(exp01, replication_id),
        "seed_micro_event_rng": stream_seeds(exp01, replication_id, ModelVariant.FULL_MICRO_REFERENCE)["micro_event_rng"],
        "initial_node": config.initial_node,
    }
    for column, value in expected.items():
        if int(run[column]) != int(value):
            raise AssertionError(f"replication {replication_id}: Experiment 01 identifier mismatch for {column}")


def _write_checkpoint(run_dir: Path, intervals: pd.DataFrame, metadata: dict) -> None:
    temporary = run_dir.with_name(f".{run_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    intervals.to_csv(temporary / "intervals.csv.gz", index=False, compression="gzip")
    (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (temporary / "complete.json").write_text(json.dumps({"replication_id": metadata["replication_id"], "complete": True}, indent=2), encoding="utf-8")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    temporary.replace(run_dir)


def _read_checkpoints(raw_dir: Path, n_runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    intervals, metadata = [], []
    for replication_id in range(1, n_runs + 1):
        run_dir = raw_dir / f"run_{replication_id:04d}"
        if not (run_dir / "complete.json").exists():
            raise RuntimeError(f"missing Experiment 02 checkpoint {run_dir}")
        intervals.append(pd.read_csv(run_dir / "intervals.csv.gz"))
        metadata.append(json.loads((run_dir / "metadata.json").read_text(encoding="utf-8")))
    return pd.concat(intervals, ignore_index=True), pd.DataFrame(metadata)


def _validate_saved_community_trajectory(source: Path, replication_id: int, config: AuditConfig) -> int:
    """Independently compare event-derived counts with Experiment 01 regular-grid counts."""
    run_dir = source / "raw" / f"run_{replication_id:04d}"
    events = pd.read_csv(run_dir / "event_log.csv")
    events = events[events["model_variant"] == FULL]
    trajectories = pd.read_csv(run_dir / "trajectories.csv")
    trajectories = trajectories[trajectories["model_variant"] == FULL]
    infection_times: dict[int, float] = {}
    for event in events.sort_values("time", kind="stable").itertuples(index=False):
        if event.kind in {"seed", "intra", "inter"} and float(event.time) <= config.T_end:
            infection_times[int(event.dst_node)] = float(event.time)
    checks = 0
    for row in trajectories.itertuples(index=False):
        community = int(row.community)
        lower, upper = community * config.n, (community + 1) * config.n
        infected = sum(lower <= node < upper and time <= float(row.time) for node, time in infection_times.items())
        if infected != int(row.I) or config.n - infected != int(row.S) or int(row.R) != 0:
            raise AssertionError(
                f"replication {replication_id}: replay/grid count mismatch at t={row.time}, community={community}"
            )
        checks += 1
    return checks


def _validate_cross_stage_determinism(current_raw: Path, reference_raw: Path, n_runs: int) -> int:
    if not reference_raw.exists():
        raise FileNotFoundError(f"missing reference stage for deterministic check: {reference_raw}")
    checked = 0
    for replication_id in range(1, n_runs + 1):
        first = pd.read_csv(reference_raw / f"run_{replication_id:04d}" / "intervals.csv.gz")
        second = pd.read_csv(current_raw / f"run_{replication_id:04d}" / "intervals.csv.gz")
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        checked += len(first)
    return checked


def run_stage(config: AuditConfig, stage: str, output_root: Path, resume: bool = False) -> Path:
    n_runs = STAGE_RUNS[stage]
    source = config.source_dir
    source_manifest = _validate_source_config(config, source)
    stage_dir = output_root / f"{stage}_{n_runs}"
    raw_dir = stage_dir / "raw"
    summaries_dir = stage_dir / "summaries"
    figures_dir = stage_dir / "figures"
    checkpoints_dir = output_root / "checkpoints" / f"{stage}_{n_runs}"
    config_dir = output_root / "config"
    for directory in (raw_dir, summaries_dir, figures_dir, checkpoints_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    existing = list(raw_dir.glob("run_*/complete.json"))
    if existing and not resume:
        raise FileExistsError(f"{stage_dir} has checkpoints; pass --resume")
    manifest = {
        **config.to_dict(), "stage": stage, "n_runs": n_runs, "complete": False,
        "source_experiment": str(source), "replay_source": "Experiment 01 FULL_MICRO_REFERENCE transmission events",
        "full_micro_rerun": False, "interval_convention": "post-event state on [t_r,t_{r+1}), clipped at T_end",
        "raw_format": "gzip-compressed CSV (Parquet engine unavailable in development environment)",
    }
    (config_dir / "audit_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (config_dir / "source_experiment_01_manifest.json").write_text(json.dumps(source_manifest, indent=2), encoding="utf-8")
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for replication_id in range(1, n_runs + 1):
        run_dir = raw_dir / f"run_{replication_id:04d}"
        if (run_dir / "complete.json").exists():
            print(f"[{stage}] replay {replication_id}/{n_runs} already complete", flush=True)
            continue
        run, events, provenance = _load_source_replication(source, replication_id)
        _expected_seed_checks(config, replication_id, run)
        network_seed = int(run["network_seed"])
        micro_graphs, full_graph, W = generate_two_scale_network(
            n_communities=config.L, community_size=config.n, inter_links=config.k,
            seed=network_seed, macro_graph_type="chain", micro_graph_type="random_connected",
            edge_prob=config.edge_prob,
        )
        communities = [list(graph.nodes()) for graph in micro_graphs]
        replay, metadata = replay_replication(
            events, full_graph, W, communities, config, replication_id, network_seed,
            int(run["initialization_seed"]), int(run["initial_node"]),
        )
        metadata.update(provenance)
        metadata["source_seed_micro_event_rng"] = int(run["seed_micro_event_rng"])
        metadata["networkx_version"] = nx.__version__
        _write_checkpoint(run_dir, replay, metadata)
        shutil.copy2(run_dir / "complete.json", checkpoints_dir / f"run_{replication_id:04d}.json")
        print(f"[{stage}] replay {replication_id}/{n_runs}: {len(replay):,} directed-pair rows", flush=True)

    intervals, replay_metadata = _read_checkpoints(raw_dir, n_runs)
    expected_rows = int(replay_metadata["directed_pair_interval_rows"].sum())
    if len(intervals) != expected_rows:
        raise AssertionError(f"aggregate row mismatch: {len(intervals)} != {expected_rows}")
    if intervals.isna().any()[["exact_hazard", "aggregate_hazard", "signed_error", "absolute_error"]].any():
        raise AssertionError("NaN in required hazard columns")
    numeric_required = intervals[["duration", "exact_hazard", "aggregate_hazard", "signed_error", "absolute_error"]].to_numpy()
    if not np.isfinite(numeric_required).all() or (intervals["duration"] <= 0).any():
        raise AssertionError("invalid numeric replay output")
    trajectory_count_checks = sum(
        _validate_saved_community_trajectory(source, replication_id, config)
        for replication_id in range(1, n_runs + 1)
    )
    if stage == "validation":
        deterministic_rows_checked = _validate_cross_stage_determinism(
            raw_dir, output_root / "smoke_5" / "raw", STAGE_RUNS["smoke"]
        )
    elif stage == "final":
        deterministic_rows_checked = _validate_cross_stage_determinism(
            raw_dir, output_root / "validation_20" / "raw", STAGE_RUNS["validation"]
        )
    else:
        deterministic_rows_checked = 0
    intervals.to_csv(raw_dir / "intervals.csv.gz", index=False, compression="gzip")
    replay_metadata.to_csv(summaries_dir / "replay_metadata.csv", index=False)
    summaries = build_summaries(intervals, config.bootstrap_reps, config.bootstrap_seed)
    for name, table in summaries.items():
        table.to_csv(summaries_dir / f"{name}.csv", index=False)
    create_figures(intervals, summaries, figures_dir)
    manifest.update({
        "complete": True, "directed_pair_interval_rows": len(intervals),
        "positive_duration_state_intervals": int(replay_metadata["positive_duration_intervals"].sum()),
        "events_beyond_horizon_ignored": int(replay_metadata["events_beyond_horizon_ignored"].sum()),
        "all_semantic_checks_passed": True,
        "experiment_01_trajectory_count_checks": trajectory_count_checks,
        "cross_stage_deterministic_rows_checked": deterministic_rows_checked,
        "deterministic_replay_passed": stage == "smoke" or deterministic_rows_checked > 0,
    })
    (stage_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (stage_dir / "commands.txt").write_text(
        f"{sys.executable} extrascripts/runners/run_experiment_02_state_replay_hazard_audit.py --stage {stage} --resume\n"
        f"{sys.executable} extrascripts/analysis/analyze_experiment_02_state_replay_hazard_audit.py --stage {stage}\n",
        encoding="utf-8",
    )
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "networkx": nx.__version__}
    (stage_dir / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    (stage_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps({"current_stage": stage, "latest": str(stage_dir), **manifest}, indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(_root_readme(config), encoding="utf-8")
    return stage_dir


def _readme(manifest: dict) -> str:
    return f"""# Experiment 02 — State-Replay Hazard Audit ({manifest['stage']})

This stage replays exact Full Micro infection events from Experiment 01; Full Micro was not rerun. Because `gamma=0`, the seed plus every node-level infection exactly determines each microscopic state. Events after `T_end` are ignored before applying them. Networks are deterministically regenerated from the Experiment 01 network seeds and archived with stable identifiers; this relies on the current generator and NetworkX version because Experiment 01 did not archive edge lists.

The aggregate formula is evaluated only on Full Micro states. No aggregate trajectory is simulated. Raw files are separate from Experiment 01. The interval table is gzip-compressed CSV because no Parquet backend was installed in the development environment.

Phase definition: `not_started` for I/N=0; `early` for 0<I/N<=0.10; `middle` for 0.10<I/N<=0.50; and `late` for I/N>0.50. For monotone SI dynamics these fixed, cross-replication-comparable bands represent establishment, growth, and saturation.

Directed-pair interval rows: {manifest.get('directed_pair_interval_rows', 'pending')}. Events beyond horizon ignored: {manifest.get('events_beyond_horizon_ignored', 'pending')}.
"""


def _root_readme(config: AuditConfig) -> str:
    return f"""# Experiment 02 — State-Replay Hazard Audit

Isolated output namespace. Experiment 01 is read-only input and is never mixed into these raw tables.

- Source: `{config.source_experiment_dir}` Full Micro event logs.
- Reconstruction: exact for this `gamma=0` baseline; no Full Micro rerun.
- Final 100 is gated by `--audit-approved` and has not been launched during development.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Experiment 01 Full Micro states and audit hazard formulas")
    parser.add_argument("--stage", choices=tuple(STAGE_RUNS), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/experiment_02_state_replay_hazard_audit"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-approved", action="store_true")
    args = parser.parse_args()
    if args.stage == "final" and not args.audit_approved:
        parser.error("final stage requires explicit --audit-approved after validation review")
    run_stage(AuditConfig(), args.stage, args.output_root, resume=args.resume)


if __name__ == "__main__":
    main()
