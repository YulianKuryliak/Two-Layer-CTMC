import argparse
import json
from pathlib import Path
import random
import shutil
import time
from typing import Dict, List, Sequence

import pandas as pd

from ...network import generate_two_scale_network
from ...simulators.micro import MicroSimulator
from .analysis import delay_metrics, layer_metrics, summarize
from .config import (
    ExperimentConfig,
    ModelVariant,
    shared_initialization_seed,
    shared_network_seed,
    stream_seeds,
)
from .simulator import HazardPlacementSimulator


ALL_VARIANTS = list(ModelVariant)

RAW_TABLES = {
    "runs": "runs.csv",
    "arrivals": "arrival_times.csv",
    "delays": "local_delays.csv",
    "hazards": "hazard_diagnostics.csv",
    "events": "event_log.csv",
    "trajectories": "trajectories.csv",
}


def run_experiment(
    config: ExperimentConfig, n_runs: int, output_dir: Path, stage: str, resume: bool = False
) -> Path:
    if config.beta_in != config.beta_out:
        raise ValueError("Full Micro uses one edge rate; this ablation currently requires beta_in == beta_out")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    existing = sorted(raw_dir.glob("run_*/complete.json"))
    if existing and not resume:
        raise FileExistsError(f"{output_dir} contains completed runs; pass --resume")
    manifest = {
        **config.to_dict(),
        "stage": stage,
        "n_runs": n_runs,
        "model_variants": [variant.value for variant in ALL_VARIANTS],
        "scheduler": "exact competing micro and macro exponential clocks",
        "pairing": "shared network and initialization; variant-specific event streams",
        "complete": False,
    }
    manifest_path = output_dir / "experiment_config.json"
    if manifest_path.exists() and resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = ("L", "n", "k", "mean_internal_degree", "beta_in", "beta_out", "gamma", "T_end", "base_seed")
        if any(previous.get(key) != manifest.get(key) for key in comparable):
            raise ValueError("resume configuration differs from existing experiment")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for replication_id in range(1, n_runs + 1):
        run_dir = raw_dir / f"run_{replication_id:04d}"
        if (run_dir / "complete.json").exists():
            print(f"[{stage}] replication {replication_id}/{n_runs} already complete", flush=True)
            continue
        frames: Dict[str, List[dict]] = {key: [] for key in RAW_TABLES}
        network_seed = shared_network_seed(config, replication_id)
        micro_graphs, full_graph, _ = generate_two_scale_network(
            n_communities=config.L,
            community_size=config.n,
            inter_links=config.k,
            seed=network_seed,
            macro_graph_type="chain",
            micro_graph_type="random_connected",
            edge_prob=config.edge_prob,
        )
        communities = [list(graph.nodes()) for graph in micro_graphs]
        initialization_seed = shared_initialization_seed(config, replication_id)
        initial_node = config.initial_node
        if initial_node is None:
            initial_node = random.Random(initialization_seed).choice(communities[config.initial_community])

        full_seeds = stream_seeds(config, replication_id, ModelVariant.FULL_MICRO_REFERENCE)
        started = time.perf_counter()
        full_result = MicroSimulator(
            full_graph=full_graph,
            comm_nodes=communities,
            infection_rate=config.beta_in,
            recovery_rate=config.gamma,
            model=2,
        ).run(
            T_end=config.T_end,
            dt_out=config.tau_micro,
            initial_node=initial_node,
            rng=random.Random(full_seeds["micro_event_rng"]),
        )
        runtime = time.perf_counter() - started
        arrivals = [item.get("first_infection_time") for item in full_result["community_timings"]]
        inter_events = [event for event in full_result["transmission_events"] if event.get("kind") == "inter"]
        seen = {config.initial_community}
        repeated = 0
        for event in inter_events:
            destination = int(event["dst_community"])
            repeated += destination in seen
            seen.add(destination)
        counters = {
            "micro_events": sum(event.get("kind") == "intra" for event in full_result["transmission_events"]),
            "macro_events": 0,
            "imports": len(inter_events),
            "repeated_imports": repeated,
            "premature_hazard_opportunities": 0,
            "premature_hazard_pair_duration": 0.0,
            "undefined_exact_placement_events": 0,
            "undefined_exact_placement_opportunities": 0,
            "null_events": 0,
        }
        _append_run_outputs(
            frames, config, replication_id, ModelVariant.FULL_MICRO_REFERENCE, network_seed,
            initialization_seed, int(initial_node), full_seeds, arrivals, runtime, counters,
            events=full_result["transmission_events"], trajectories=full_result["rows"],
        )

        for variant in ALL_VARIANTS[1:]:
            seeds = stream_seeds(config, replication_id, variant)
            simulator = HazardPlacementSimulator(full_graph, communities, config, variant, seeds)
            started = time.perf_counter()
            result = simulator.run(int(initial_node))
            runtime = time.perf_counter() - started
            _append_run_outputs(
                frames, config, replication_id, variant, network_seed, initialization_seed,
                int(initial_node), seeds, result["arrivals"], runtime, result["counters"],
                events=result["event_log"], trajectories=result["trajectories"],
                diagnostics=result["hazard_diagnostics"],
            )
        _write_replication(run_dir, frames, replication_id)
        print(f"[{stage}] replication {replication_id}/{n_runs}", flush=True)

    missing = [i for i in range(1, n_runs + 1) if not (raw_dir / f"run_{i:04d}" / "complete.json").exists()]
    if missing:
        raise RuntimeError(f"incomplete replications: {missing}")
    _aggregate_replications(raw_dir, output_dir, n_runs)
    runs = pd.read_csv(output_dir / "runs.csv")
    arrivals_df = pd.read_csv(output_dir / "arrival_times.csv")
    delays_df = pd.read_csv(output_dir / "local_delays.csv")

    summary, intervals = summarize(
        arrivals_df, runs, config.T_end, config.bootstrap_reps,
        stream_seeds(config, 0, ModelVariant.FULL_MICRO_REFERENCE)["bootstrap_rng"],
    )
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    intervals.to_csv(output_dir / "bootstrap_intervals.csv", index=False)
    layer_metrics(arrivals_df).to_csv(output_dir / "layer_metrics.csv", index=False)
    delay_metrics(delays_df).to_csv(output_dir / "local_delay_metrics.csv", index=False)
    manifest["complete"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(_readme(manifest, summary), encoding="utf-8")
    return output_dir


def _write_replication(run_dir: Path, frames: Dict[str, List[dict]], replication_id: int) -> None:
    temporary = run_dir.with_name(f".{run_dir.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for key, filename in RAW_TABLES.items():
        pd.DataFrame(frames[key]).to_csv(temporary / filename, index=False)
    (temporary / "complete.json").write_text(
        json.dumps({"replication_id": replication_id, "complete": True}, indent=2), encoding="utf-8"
    )
    if run_dir.exists():
        shutil.rmtree(run_dir)
    temporary.replace(run_dir)


def _aggregate_replications(raw_dir: Path, output_dir: Path, n_runs: int) -> None:
    for filename in RAW_TABLES.values():
        destination = output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as target:
            for replication_id in range(1, n_runs + 1):
                source_path = raw_dir / f"run_{replication_id:04d}" / filename
                with source_path.open("rb") as source:
                    if replication_id > 1:
                        source.readline()
                    shutil.copyfileobj(source, target)
        temporary.replace(destination)


def _append_run_outputs(
    frames, config, replication_id, variant, network_seed, initialization_seed, initial_node,
    seeds, arrivals, runtime, counters, events, trajectories, diagnostics=(),
):
    t_last = arrivals[-1] if arrivals and arrivals[-1] is not None else None
    hazard_signed = sum(float(row["signed_hazard_error_contribution"]) for row in diagnostics)
    hazard_absolute = sum(float(row["absolute_hazard_error_contribution"]) for row in diagnostics)
    frames["runs"].append({
        "replication_id": replication_id,
        "model_variant": variant.value,
        "network_seed": network_seed,
        "initialization_seed": initialization_seed,
        "initial_node": initial_node,
        "completion": t_last is not None,
        "T_last": t_last,
        "censored_T_last": t_last if t_last is not None else config.T_end,
        "runtime_seconds": runtime,
        **counters,
        "signed_hazard_error": hazard_signed,
        "absolute_hazard_error": hazard_absolute,
        **{f"seed_{key}": value for key, value in seeds.items()},
    })
    for layer, value in enumerate(arrivals):
        frames["arrivals"].append({
            "replication_id": replication_id, "model_variant": variant.value,
            "layer_index": layer, "T_i": value,
        })
        if layer + 1 < len(arrivals):
            following = arrivals[layer + 1]
            frames["delays"].append({
                "replication_id": replication_id, "model_variant": variant.value,
                "layer_index": layer,
                "Delta_T_i": following - value if value is not None and following is not None else None,
            })
    for row in diagnostics:
        frames["hazards"].append({"replication_id": replication_id, "model_variant": variant.value, **row})
    for row in events:
        frames["events"].append({"replication_id": replication_id, "model_variant": variant.value, **row})
    for row in trajectories:
        if isinstance(row, dict):
            payload = row
        else:
            community, time_value, s, i, r = row
            payload = {"community": community, "time": time_value, "S": s, "I": i, "R": r}
        frames["trajectories"].append({"replication_id": replication_id, "model_variant": variant.value, **payload})


def _readme(manifest: dict, summary: pd.DataFrame) -> str:
    approval_flag = " --oracle-approved" if manifest["stage"] == "final" else ""
    compact_results = summary[[
        "model_variant", "completion_probability", "relative_W1_T_last_vs_micro", "delta_RMST_vs_micro"
    ]].to_csv(index=False)
    return f"""# Hazard-placement ablation: {manifest['stage']}

## Modes

- `FULL_MICRO_REFERENCE`: production full-network Gillespie reference.
- `ORACLE_EXACT_HAZARD_EXACT_PLACEMENT`: exact active-edge hazard and exact edge-conditioned target placement.
- `ORACLE_EXACT_HAZARD_UNIFORM_PLACEMENT`: exact active-edge hazard and uniform susceptible placement.
- `AGGREGATE_HAZARD_BOUNDARY_PLACEMENT`: aggregate prevalence hazard and static edge-weighted susceptible boundary placement.
- `AGGREGATE_HAZARD_UNIFORM_PLACEMENT`: aggregate prevalence hazard and uniform susceptible placement.

The mathematically problematic aggregate-hazard/exact-placement cell is deliberately absent. When aggregate hazard is positive but exact active-edge hazard is zero, no microscopic transmission event exists and the exact edge-conditioned target distribution is undefined. No uniform fallback is used. The boundary-weighted mode is a separate approximation and gates a directed hazard to zero when no susceptible boundary target exists.

## Parameters and seeds

`L={manifest['L']}`, `n={manifest['n']}`, `d={manifest['mean_internal_degree']}`, `k={manifest['k']}`, `beta_in={manifest['beta_in']}`, `beta_out={manifest['beta_out']}`, `gamma={manifest['gamma']}`, `T_end={manifest['T_end']}`, runs={manifest['n_runs']}.

Every replication shares the generated network, inter-community edges, initialization seed, and initial node across modes. Event, macro-budget, pair-selection, and placement streams are deterministic and variant-specific. Pairing therefore controls network and initialization variability but is not event-by-event common-random-number coupling.

## Scheduler limitation

The isolated ablation uses exact competition between micro and macro exponential clocks. This removes midpoint quadrature as a third confounder. The aggregate/uniform mode reproduces the current model's hazard and placement semantics, but not the production MicroMacro midpoint scheduler. Production simulators are unchanged.

## Reproduction

```bash
.venv/bin/python extrascripts/runners/run_hazard_placement_ablation.py --stage {manifest['stage']} --n-runs {manifest['n_runs']}{approval_flag} --resume
```

Interpretation is blocked until the oracle exact/exact mode is statistically consistent with Full Micro.

## Preliminary results

```csv
{compact_results.strip()}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "validation", "final"), required=True)
    parser.add_argument("--n-runs", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/ablation_hazard_placement"))
    parser.add_argument("--oracle-approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "final" and not args.oracle_approved:
        parser.error("final stage requires explicit --oracle-approved after validation review")
    config = ExperimentConfig()
    run_experiment(
        config, args.n_runs, args.output_root / f"{args.stage}_{args.n_runs}", args.stage, resume=args.resume
    )


if __name__ == "__main__":
    main()
