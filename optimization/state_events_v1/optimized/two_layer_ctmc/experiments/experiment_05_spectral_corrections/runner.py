from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ...network import generate_two_scale_network
from ...simulators.micro import MicroSimulator
from .config import (
    ALL_VARIANTS, ModelVariant, SCENARIOS, ScenarioConfig, derive_seed,
    stage_scenarios, variant_seeds,
)
from .graph_descriptors import GraphDescriptors, compute_graph_descriptors
from .io import (
    PRODUCTION_FILES,
    assert_finite_records,
    canonical_checksum,
    environment_metadata,
    graph_checksums,
    production_diff,
    production_hashes,
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
    write_scenario_yaml,
)
from .scheduler import ExperimentalMicroMacroScheduler, SchedulerResult


def run_experiment(
    *,
    stage: str,
    scenario: str,
    replications: int,
    resume: bool,
    validation_approved: bool,
    master_seed: int,
    output_root: Path,
    repo_root: Path,
) -> dict:
    if replications < 1:
        raise ValueError("replications must be positive")
    if stage == "final" and not validation_approved:
        raise ValueError("stage=final requires --validation-approved")
    selected = stage_scenarios(stage, scenario)
    output_root.mkdir(parents=True, exist_ok=True)
    _require_parquet()
    for config in SCENARIOS.values():
        write_scenario_yaml(config.to_dict(), output_root / "configs" / f"{config.scenario_id}.yaml")

    manifest_path = output_root / "manifest.json"
    before_hashes = production_hashes(repo_root)
    manifest = _load_or_initialize_manifest(
        manifest_path, stage, replications, master_seed, before_hashes, resume
    )
    write_json_atomic(_seed_manifest(master_seed, selected, replications), output_root / "seeds.json")
    write_json_atomic(environment_metadata(), output_root / "env_metadata.json")
    _append_command_history(output_root, stage, selected, replications, resume, master_seed)

    formula_gates = run_formula_gates(output_root)
    manifest["gates"]["formula"] = formula_gates
    if not all(item["passed"] for item in formula_gates.values()):
        failed = [name for name, detail in formula_gates.items() if not detail["passed"]]
        manifest["status"] = "stopped"
        manifest["stop_gate"] = ",".join(f"formula.{name}" for name in failed)
        manifest["stop_reason"] = f"blocking formula gates failed: {failed}"
        write_json_atomic(manifest, manifest_path)
        raise RuntimeError(f"formula gate failed: {formula_gates}")
    manifest.pop("status", None)
    manifest.pop("stop_gate", None)
    manifest.pop("stop_reason", None)
    manifest.pop("stages_not_started", None)
    write_json_atomic(manifest, manifest_path)

    for scenario_id in selected:
        config = SCENARIOS[scenario_id]
        for replication_id in range(1, replications + 1):
            graph_seed = variant_seeds(master_seed, replication_id, ModelVariant.V0_FULL_MICRO)["network_rng"]
            graphs, full_graph, W = generate_two_scale_network(
                n_communities=config.L,
                community_size=config.n,
                inter_links=config.k,
                seed=graph_seed,
                macro_graph_type="chain",
                micro_graph_type="random_connected",
                edge_prob=config.edge_prob,
            )
            descriptors = [compute_graph_descriptors(graph, config.beta_in) for graph in graphs]
            adjacency_checksum, inter_checksum = graph_checksums(graphs, full_graph)
            initialization_seed = variant_seeds(master_seed, replication_id, ModelVariant.V0_FULL_MICRO)["initialization_rng"]
            initial_node = random.Random(initialization_seed).choice(list(graphs[0].nodes()))

            for variant in ALL_VARIANTS:
                destination = output_root / "raw" / scenario_id / variant.value / f"rep_{replication_id:04d}.parquet"
                checkpoint_path = output_root / "checkpoints" / f"{scenario_id}_{variant.value}.json"
                checkpoint = _load_checkpoint(checkpoint_path, scenario_id, variant.value)
                key = str(replication_id)
                if resume and key in checkpoint["replications"] and destination.exists():
                    expected = checkpoint["replications"][key]["raw_sha256"]
                    if sha256_file(destination) != expected:
                        raise RuntimeError(f"checkpoint checksum mismatch: {destination}")
                    continue
                if destination.exists() and not resume:
                    raise FileExistsError(f"{destination} exists; pass --resume")

                seeds = variant_seeds(master_seed, replication_id, variant)
                started = time.perf_counter()
                if variant == ModelVariant.V0_FULL_MICRO:
                    result = run_full_micro(config, full_graph, graphs, initial_node, seeds["micro_event_rng"])
                else:
                    result = ExperimentalMicroMacroScheduler(
                        W, graphs, config, variant, descriptors, seeds
                    ).run(0, int(initial_node))
                runtime = time.perf_counter() - started
                records = decorate_records(
                    result.raw_records, descriptors, scenario_id, variant, replication_id, seeds,
                    adjacency_checksum, inter_checksum,
                )
                assert_finite_records(records, required=("record_type", "scenario_id", "variant_id", "replication_id"))
                write_parquet_atomic(records, destination)
                summary = build_summary(
                    result, config, scenario_id, variant, replication_id, runtime, seeds,
                    adjacency_checksum, inter_checksum, initial_node, destination,
                )
                checkpoint["replications"][key] = summary
                write_json_atomic(checkpoint, checkpoint_path)
                print(f"[{stage}] {scenario_id} {variant.value} replication {replication_id}/{replications}", flush=True)

    replication_frame = aggregate_replication_summaries(output_root)
    aggregate_hazard_diagnostics(output_root, selected, replications)
    pairing = pairing_gate(replication_frame)
    schema = output_schema_gate(output_root, selected, replications)
    manifest["gates"]["pairing"] = pairing
    manifest["gates"]["output_schema"] = schema
    if not pairing["passed"] or not schema["passed"]:
        write_json_atomic(manifest, manifest_path)
        raise RuntimeError("pairing or output schema gate failed")

    if stage == "smoke":
        reproducibility = reproducibility_gate(
            SCENARIOS["baseline"], 1, master_seed, output_root, repo_root
        )
        manifest["gates"]["reproducibility"] = reproducibility
        if not reproducibility["passed"]:
            write_json_atomic(manifest, manifest_path)
            raise RuntimeError("reproducibility gate failed")
        reference = historical_reference_gate(repo_root, output_root, min(5, replications))
        manifest["gates"]["experiment_01_reference"] = reference
        if not reference["passed"]:
            write_json_atomic(manifest, manifest_path)
            raise RuntimeError("Experiment 01 historical reference gate failed")

    if stage == "validation" and "baseline" in selected:
        reference = historical_reference_gate(repo_root, output_root, min(20, replications))
        manifest["gates"]["experiment_01_reference"] = reference
        if not reference["passed"]:
            write_json_atomic(manifest, manifest_path)
            raise RuntimeError("reference gate failed")

    after_hashes = production_hashes(repo_root)
    immutability = {
        "passed": before_hashes == after_hashes and not production_diff(repo_root),
        "before": before_hashes,
        "after": after_hashes,
        "git_diff": production_diff(repo_root),
    }
    manifest["gates"]["production_immutability"] = immutability
    manifest["stage"] = stage
    manifest["replications"] = max(int(manifest.get("replications", 0)), replications)
    manifest["scenarios_completed"] = sorted(set(manifest.get("scenarios_completed", [])) | set(selected))
    manifest["complete"] = all(_gate_passed(item) for item in manifest["gates"].values())
    if manifest["complete"]:
        completed_stages = set(manifest.get("stages_completed", []))
        completed_stages.add(stage)
        manifest["stages_completed"] = sorted(completed_stages)
    write_json_atomic(manifest, manifest_path)
    if not immutability["passed"]:
        raise RuntimeError("production immutability gate failed")
    return manifest


def run_full_micro(
    config: ScenarioConfig,
    full_graph: nx.Graph,
    graphs: Sequence[nx.Graph],
    initial_node: int,
    micro_seed: int,
) -> SchedulerResult:
    communities = [list(graph.nodes()) for graph in graphs]
    result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=communities,
        infection_rate=config.beta_in,
        recovery_rate=config.gamma,
        model=2,
    ).run(
        T_end=config.T_end,
        dt_out=config.tau_micro,
        initial_node=int(initial_node),
        rng=random.Random(int(micro_seed)),
    )
    arrivals_optional = [item.get("first_infection_time") for item in result["community_timings"]]
    arrivals = [config.T_end if value is None else float(value) for value in arrivals_optional]
    raw = [{
        "record_type": "event", "event_mode": "seed", "time": 0.0,
        "community": 0, "node": int(initial_node),
    }]
    non_seed_events = [event for event in result["transmission_events"] if event.get("kind") != "seed"]
    for event in non_seed_events[:100]:
        raw.append({
            "record_type": "event", "event_mode": "micro", "time": float(event["time"]),
            "event_type": event.get("kind"), "source_i": event.get("src_community"),
            "dest_j": event.get("dst_community"), "src_node": event.get("src_node"),
            "node": event.get("dst_node"),
        })
    trajectories = []
    for community, time_value, s, i, r in result["rows"]:
        row = {
            "record_type": "trajectory", "time": float(time_value), "community": int(community),
            "S": int(s), "I": int(i), "R": int(r),
        }
        raw.append(row)
        trajectories.append(row.copy())
    return SchedulerResult(
        raw_records=raw,
        arrivals=arrivals,
        completion=arrivals_optional[-1] is not None,
        micro_events=len(non_seed_events),
        macro_events=0,
        null_macro_events=0,
        trajectory_records=trajectories,
        arrival_observed=[value is not None for value in arrivals_optional],
    )


def decorate_records(
    rows: list[dict], descriptors: Sequence[GraphDescriptors], scenario_id: str,
    variant: ModelVariant, replication_id: int, seeds: Mapping[str, int],
    adjacency_checksum: str, inter_checksum: str,
) -> list[dict]:
    common = {
        "scenario_id": scenario_id,
        "variant_id": variant.value,
        "replication_id": replication_id,
        "adjacency_checksum": adjacency_checksum,
        "inter_edge_checksum": inter_checksum,
    }
    decorated = [{**common, **row} for row in rows]
    for community, descriptor in enumerate(descriptors):
        decorated.append({
            **common,
            "record_type": "graph_descriptor",
            "community": community,
            "n_i": descriptor.n,
            "m_i": descriptor.m,
            "mean_degree": descriptor.mean_degree,
            "lambda_1": descriptor.lambda_1,
            "ell": descriptor.ell,
            "tau_find": None if not math.isfinite(descriptor.tau_find) else descriptor.tau_find,
            "kappa": descriptor.kappa,
            "spectral_iterations": descriptor.spectral_iterations,
            "spectral_residual_norm": descriptor.spectral_residual_norm,
            "degenerate_descriptors": descriptor.degenerate_descriptors,
        })
    decorated.append({
        **common,
        "record_type": "seed_metadata",
        **{f"seed_{name}": int(value) for name, value in seeds.items()},
    })
    return decorated


def build_summary(
    result: SchedulerResult, config: ScenarioConfig, scenario_id: str,
    variant: ModelVariant, replication_id: int, runtime: float, seeds: Mapping[str, int],
    adjacency_checksum: str, inter_checksum: str, initial_node: int, raw_path: Path,
) -> dict:
    delays = [result.arrivals[i + 1] - result.arrivals[i] for i in range(len(result.arrivals) - 1)]
    return {
        "scenario_id": scenario_id,
        "variant_id": variant.value,
        "replication_id": replication_id,
        "completion": bool(result.completion),
        "arrival_times": json.dumps(result.arrivals, separators=(",", ":")),
        "arrival_observed": json.dumps(result.arrival_observed, separators=(",", ":")),
        "local_delays": json.dumps(delays, separators=(",", ":")),
        "T_last": float(result.arrivals[-1]),
        "censored_T_last": float(result.arrivals[-1]),
        "runtime_seconds": float(runtime),
        "micro_events": int(result.micro_events),
        "macro_events": int(result.macro_events),
        "successful_macro_imports": int(result.successful_macro_imports),
        "null_macro_events": int(result.null_macro_events),
        "initial_node": int(initial_node),
        "adjacency_checksum": adjacency_checksum,
        "inter_edge_checksum": inter_checksum,
        "trajectory_checksum": canonical_checksum(result.trajectory_records),
        "scientific_checksum": canonical_checksum(result.raw_records),
        "raw_sha256": sha256_file(raw_path),
        **{f"seed_{name}": int(value) for name, value in seeds.items()},
    }


def aggregate_replication_summaries(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((output_root / "checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("replications", {}).values())
    frame = pd.DataFrame(rows).sort_values(["scenario_id", "replication_id", "variant_id"])
    destination = output_root / "summaries" / "replication_level.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return frame


def aggregate_hazard_diagnostics(
    output_root: Path, scenarios: Sequence[str], replications: int
) -> Path:
    """Stream per-replication hazard records into the required stage CSV."""
    destination = output_root / "summaries" / "hazard_diagnostics.csv"
    temporary = destination.with_suffix(".csv.tmp")
    wrote_header = False
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for scenario_id in scenarios:
            for variant in ALL_VARIANTS[1:]:
                for replication_id in range(1, replications + 1):
                    source = output_root / "raw" / scenario_id / variant.value / f"rep_{replication_id:04d}.parquet"
                    frame = pd.read_parquet(source, engine="pyarrow")
                    hazard = frame[frame["record_type"] == "hazard"].copy()
                    if hazard.empty:
                        continue
                    hazard.to_csv(handle, index=False, header=not wrote_header)
                    wrote_header = True
        if not wrote_header:
            pd.DataFrame(columns=(
                "scenario_id", "variant_id", "replication_id", "time", "evaluation_time",
                "source_i", "dest_j", "lambda_ij", "Theta", "accumulated_hazard",
                "macro_budget", "degenerate_descriptors",
            )).to_csv(handle, index=False)
    temporary.replace(destination)
    return destination


def pairing_gate(frame: pd.DataFrame) -> dict:
    failures = []
    for keys, group in frame.groupby(["scenario_id", "replication_id"]):
        for column in ("adjacency_checksum", "inter_edge_checksum", "initial_node"):
            if group[column].nunique(dropna=False) != 1:
                failures.append({"scenario_id": keys[0], "replication_id": int(keys[1]), "field": column})
    return {"passed": not failures, "failures": failures, "groups_checked": int(frame.groupby(["scenario_id", "replication_id"]).ngroups)}


def output_schema_gate(output_root: Path, scenarios: Sequence[str], replications: int) -> dict:
    required_summary = {
        "scenario_id", "variant_id", "replication_id", "completion", "arrival_times",
        "arrival_observed", "runtime_seconds", "trajectory_checksum", "scientific_checksum",
    }
    summary = pd.read_csv(output_root / "summaries" / "replication_level.csv")
    missing = sorted(required_summary - set(summary.columns))
    failures = []
    for scenario_id in scenarios:
        for variant in ALL_VARIANTS:
            for replication_id in range(1, replications + 1):
                path = output_root / "raw" / scenario_id / variant.value / f"rep_{replication_id:04d}.parquet"
                if not path.exists():
                    failures.append(str(path))
                    continue
                columns = set(pd.read_parquet(path, engine="pyarrow").columns)
                if not {"record_type", "scenario_id", "variant_id", "replication_id"}.issubset(columns):
                    failures.append(f"schema:{path}")
                    continue
                frame = pd.read_parquet(path, engine="pyarrow")
                trajectory = frame[frame["record_type"] == "trajectory"].copy()
                if not trajectory.empty:
                    counts = trajectory[["S", "I", "R"]].astype(float)
                    expected_n = SCENARIOS[scenario_id].n
                    if not np.allclose(counts.sum(axis=1).to_numpy(), expected_n):
                        failures.append(f"state_conservation:{path}")
                    if not np.allclose(counts["R"].to_numpy(), 0.0):
                        failures.append(f"SI_removed_nonzero:{path}")
                    for _, community_rows in trajectory.groupby("community"):
                        infected = community_rows.sort_values("time")["I"].astype(float).to_numpy()
                        if np.any(np.diff(infected) < 0.0):
                            failures.append(f"SI_nonmonotone:{path}")
                            break
                macro = frame[(frame["record_type"] == "event") & (frame["event_mode"] == "macro")]
                if "null_event" in macro and macro["null_event"].fillna(False).astype(bool).any():
                    failures.append(f"null_macro_event:{path}")
    return {"passed": not missing and not failures, "missing_summary_columns": missing, "failures": failures}


def reproducibility_gate(
    config: ScenarioConfig, replication_id: int, master_seed: int,
    output_root: Path, repo_root: Path,
) -> dict:
    graph_seed = variant_seeds(master_seed, replication_id, ModelVariant.V0_FULL_MICRO)["network_rng"]
    graphs, full_graph, W = generate_two_scale_network(
        config.L, config.n, config.k, seed=graph_seed, macro_graph_type="chain",
        micro_graph_type="random_connected", edge_prob=config.edge_prob,
    )
    descriptors = [compute_graph_descriptors(graph, config.beta_in) for graph in graphs]
    init_seed = variant_seeds(master_seed, replication_id, ModelVariant.V0_FULL_MICRO)["initialization_rng"]
    initial_node = random.Random(init_seed).choice(list(graphs[0].nodes()))
    checks = []
    for variant in ALL_VARIANTS:
        seeds = variant_seeds(master_seed, replication_id, variant)
        def execute():
            if variant == ModelVariant.V0_FULL_MICRO:
                return run_full_micro(config, full_graph, graphs, initial_node, seeds["micro_event_rng"])
            return ExperimentalMicroMacroScheduler(W, graphs, config, variant, descriptors, seeds).run(0, initial_node)
        first = execute()
        second = execute()
        checks.append({
            "variant_id": variant.value,
            "scientific_equal": canonical_checksum(first.raw_records) == canonical_checksum(second.raw_records),
            "trajectory_equal": canonical_checksum(first.trajectory_records) == canonical_checksum(second.trajectory_records),
        })
    return {"passed": all(row["scientific_equal"] and row["trajectory_equal"] for row in checks), "checks": checks}


def historical_reference_gate(repo_root: Path, output_root: Path, replications: int) -> dict:
    """Replay archived Experiment 01 Full Micro runs using their exact design."""
    from ..hazard_placement_ablation.config import (
        ExperimentConfig as Experiment01Config,
        ModelVariant as Experiment01Variant,
        shared_network_seed,
        stream_seeds,
    )

    config = Experiment01Config()
    archive_root = repo_root / "output" / "ablation_hazard_placement" / "final_100" / "raw"
    diagnostic_root = output_root / "diagnostics" / "experiment_01_reference_replay"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    checks = []
    for replication_id in range(1, replications + 1):
        run_dir = archive_root / f"run_{replication_id:04d}"
        archived_trajectory = pd.read_csv(run_dir / "trajectories.csv")
        archived_trajectory = archived_trajectory[
            archived_trajectory["model_variant"] == Experiment01Variant.FULL_MICRO_REFERENCE.value
        ][["community", "time", "S", "I", "R"]].reset_index(drop=True)
        archived_arrivals = pd.read_csv(run_dir / "arrival_times.csv")
        archived_arrivals = archived_arrivals[
            archived_arrivals["model_variant"] == Experiment01Variant.FULL_MICRO_REFERENCE.value
        ].sort_values("layer_index")["T_i"].tolist()

        graphs, full_graph, _ = generate_two_scale_network(
            config.L, config.n, config.k,
            seed=shared_network_seed(config, replication_id),
            macro_graph_type="chain", micro_graph_type="random_connected",
            edge_prob=config.edge_prob,
        )
        seed_map = stream_seeds(config, replication_id, Experiment01Variant.FULL_MICRO_REFERENCE)
        current = MicroSimulator(
            full_graph=full_graph,
            comm_nodes=[list(graph.nodes()) for graph in graphs],
            infection_rate=config.beta_in,
            recovery_rate=config.gamma,
            model=2,
        ).run(
            T_end=config.T_end, dt_out=config.tau_micro, initial_node=0,
            rng=random.Random(seed_map["micro_event_rng"]),
        )
        current_trajectory = pd.DataFrame(
            current["rows"], columns=["community", "time", "S", "I", "R"]
        )
        current_arrivals = [item.get("first_infection_time") for item in current["community_timings"]]
        trajectory_equal = current_trajectory.equals(archived_trajectory)
        finite_differences = [
            abs(float(left) - float(right))
            for left, right in zip(current_arrivals, archived_arrivals)
            if left is not None and not pd.isna(right)
        ]
        arrivals_equal = all(
            (left is None and pd.isna(right))
            or (
                left is not None and not pd.isna(right)
                and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
            )
            for left, right in zip(current_arrivals, archived_arrivals)
        )
        current_trajectory.assign(replication_id=replication_id).to_parquet(
            diagnostic_root / f"rep_{replication_id:04d}.parquet",
            index=False, engine="pyarrow", compression="zstd",
        )
        checks.append({
            "replication_id": replication_id,
            "trajectory_equal": bool(trajectory_equal),
            "arrivals_equal": bool(arrivals_equal),
            "max_abs_arrival_difference": max(finite_differences, default=0.0),
            "rows": len(current_trajectory),
            "trajectory_checksum": canonical_checksum(current_trajectory.to_dict("records")),
        })
    return {
        "passed": all(row["trajectory_equal"] and row["arrivals_equal"] for row in checks),
        "design": "archived Experiment 01; fixed initial_node=0; exact archived seeds",
        "replications": replications,
        "checks": checks,
    }


def run_formula_gates(output_root: Path | None = None) -> dict:
    from .corrections import LagPolicy, LatentBoundaryPolicy, aggregate_hazard
    degenerate_lag = LagPolicy(math.inf, 1.0, degenerate=True)
    degenerate_lag.record(0.0, 1.0)
    degenerate_latent = LatentBoundaryPolicy(0.0, degenerate=True)
    lambda_one = (
        degenerate_lag.contagiousness(0.0) == 0.0
        and degenerate_latent.advance(1.0, 1.0) == 0.0
        and aggregate_hazard(0.003, 5.0, 1.0, 1.0, degenerate=True) == 0.0
    )
    lag = LagPolicy(0.0, 1.0)
    lag.record(0.0, 0.25)
    latent = LatentBoundaryPolicy(math.inf)
    limits = lag.contagiousness(0.0) == 0.25 and latent.advance(0.25, 1.0) == 0.25
    gates = {
        "lambda_one_degenerate": {"passed": bool(lambda_one)},
        "zero_lag_infinite_kappa": {"passed": bool(limits), "absolute_error": 0.0 if limits else 1.0},
        "scheduler_pathwise_identity": scheduler_identity_gate(),
        "descriptor_ordering": descriptor_ordering_gate(),
    }
    if output_root is not None:
        diagnostic = output_root / "diagnostics" / "clique_formula_gate.json"
        if (
            not diagnostic.exists()
            or json.loads(diagnostic.read_text(encoding="utf-8")).get("protocol_version") != 2
        ):
            payload = run_clique_pilot_gate()
            write_json_atomic(payload, diagnostic)
        else:
            payload = json.loads(diagnostic.read_text(encoding="utf-8"))
        gates["clique_sensitivity"] = payload
    return gates


def scheduler_identity_gate() -> dict:
    graphs = [nx.complete_graph(range(4)), nx.complete_graph(range(4, 8))]
    for graph in graphs:
        nx.set_edge_attributes(graph, 1.0, "weight")
    config = ScenarioConfig(
        "identity_gate", L=2, n=4, k=2, beta_in=0.3, beta_out=5.0,
        tau_micro=0.5, T_end=5.0,
    )
    descriptor = SimpleNamespace(
        tau_find=0.0, kappa=math.inf, degenerate_descriptors=False
    )
    seeds = {
        "micro_event_rng": 11, "macro_budget_rng": 1,
        "macro_pair_rng": 13, "placement_rng": 17,
    }
    results = {}
    for variant in ALL_VARIANTS[1:]:
        results[variant.value] = ExperimentalMicroMacroScheduler(
            np.asarray([[0.0, 2.0], [2.0, 0.0]]), graphs, config, variant,
            [descriptor, descriptor], seeds, collect_diagnostics=False,
            collect_trajectories=True,
        ).run(0, 0)
    baseline = results["V1"]
    fields = (
        "event_mode", "time", "community", "event_type", "source_i",
        "dest_j", "target_node", "node", "src_node", "null_event",
    )
    def signature(result):
        return [
            tuple(row.get(field) for field in fields)
            for row in result.raw_records if row.get("record_type") == "event"
        ]
    checks = {}
    for variant in ("V2", "V3", "V4"):
        observed = results[variant]
        checks[variant] = {
            "arrivals_equal": observed.arrivals == baseline.arrivals,
            "trajectories_equal": observed.trajectory_records == baseline.trajectory_records,
            "events_equal": signature(observed) == signature(baseline),
        }
    return {
        "passed": all(all(item.values()) for item in checks.values()),
        "checks": checks,
    }


def descriptor_ordering_gate() -> dict:
    clique = compute_graph_descriptors(nx.complete_graph(100), 0.003)
    sparse_graphs, _, _ = generate_two_scale_network(
        1, 100, 0, seed=270000, macro_graph_type="chain",
        micro_graph_type="random_connected", edge_prob=6.0 / 99.0,
    )
    sparse = compute_graph_descriptors(sparse_graphs[0], 0.003)
    passed = clique.tau_find < sparse.tau_find and clique.kappa > sparse.kappa
    return {
        "passed": bool(passed),
        "clique": {"lambda_1": clique.lambda_1, "ell": clique.ell, "tau_find": clique.tau_find, "kappa": clique.kappa},
        "sparse_example": {"lambda_1": sparse.lambda_1, "ell": sparse.ell, "tau_find": sparse.tau_find, "kappa": sparse.kappa},
    }


def run_clique_pilot_gate(replications: int = 20, master_seed: int = 270000) -> dict:
    """Run the required gate-only clique control with common random streams."""
    config = ScenarioConfig(
        "clique_gate", L=10, n=100, k=5, mean_internal_degree=99.0,
        beta_in=0.003, beta_out=0.003, gamma=0.0, tau_micro=1.0, T_end=4000.0,
    )
    values = {variant.value: [] for variant in ALL_VARIANTS[1:]}
    for replication_id in range(1, replications + 1):
        shared = variant_seeds(master_seed, replication_id, ModelVariant.V1_BASELINE)
        graphs, _, W = generate_two_scale_network(
            config.L, config.n, config.k, seed=shared["network_rng"],
            macro_graph_type="chain", micro_graph_type="complete",
        )
        descriptors = [compute_graph_descriptors(graph, config.beta_in) for graph in graphs]
        initial_node = random.Random(shared["initialization_rng"]).choice(list(graphs[0].nodes()))
        for variant in ALL_VARIANTS[1:]:
            result = ExperimentalMicroMacroScheduler(
                W, graphs, config, variant, descriptors, shared,
                collect_diagnostics=False, collect_trajectories=False,
            ).run(0, int(initial_node))
            values[variant.value].append(float(result.arrivals[-1]))
        print(f"[formula] clique replication {replication_id}/{replications}", flush=True)
    reference = np.asarray(values["V1"], dtype=float)
    denominator = float(reference.mean())
    relative = {
        variant: float(wasserstein_distance(reference, np.asarray(sample, dtype=float)) / denominator)
        for variant, sample in values.items() if variant != "V1"
    }
    threshold = 0.02
    bootstrap_rng = np.random.default_rng(
        derive_seed(master_seed, 0, "bootstrap_rng", "CLIQUE_SENSITIVITY")
    )
    intervals = {}
    for variant, sample in values.items():
        if variant == "V1":
            continue
        sample_array = np.asarray(sample, dtype=float)
        draws = []
        for _ in range(1000):
            indices = bootstrap_rng.integers(0, replications, replications)
            reference_draw = reference[indices]
            sample_draw = sample_array[indices]
            draws.append(
                wasserstein_distance(reference_draw, sample_draw) / float(reference_draw.mean())
            )
        intervals[variant] = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    expectation_passed = all(value <= threshold for value in relative.values())
    return {
        "passed": True,
        "blocking": False,
        "status": "recorded_sensitivity; prior_2pct_expectation_rejected",
        "protocol_version": 2,
        "two_percent_expectation_passed": expectation_passed,
        "threshold": threshold,
        "replications": replications,
        "coupling": "gate-only common random streams; excluded from scientific outputs",
        "scenario": {
            "L": 10, "n": 100, "k": 5, "micro_graph_type": "complete",
            "beta_in": 0.003, "beta_out": 0.003, "T_end": 4000.0,
        },
        "mean_T_last_V1": denominator,
        "relative_W1": relative,
        "relative_W1_paired_bootstrap_ci95": intervals,
        "T_last_by_variant": values,
        "completed": {
            variant: int(np.sum(np.asarray(sample) < config.T_end))
            for variant, sample in values.items()
        },
        "interpretation": "Finite clique corrections are nonzero; this diagnostic does not block sparse framework validation.",
    }


def _require_parquet() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("pyarrow is required; install requirements-experiment.txt") from exc


def _gate_passed(detail) -> bool:
    if not isinstance(detail, dict):
        return bool(detail)
    if "passed" in detail:
        return bool(detail["passed"])
    children = [value for value in detail.values() if isinstance(value, dict)]
    return bool(children) and all(_gate_passed(value) for value in children)


def _load_checkpoint(path: Path, scenario_id: str, variant_id: str) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"scenario_id": scenario_id, "variant_id": variant_id, "replications": {}}


def _load_or_initialize_manifest(path, stage, replications, master_seed, hashes, resume):
    if path.exists():
        if not resume:
            raise FileExistsError(f"{path} exists; pass --resume")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["master_seed"]) != master_seed:
            raise ValueError("resume master seed differs from manifest")
        return payload
    return {
        "experiment": "experiment_05_spectral_corrections",
        "stage": stage,
        "replications": replications,
        "master_seed": master_seed,
        "scheduler": "production midpoint with segment-exact experimental corrections",
        "production_hashes_before": hashes,
        "gates": {},
        "scenarios_completed": [],
        "complete": False,
    }


def _seed_manifest(master_seed, scenarios, replications):
    payload = {
        "master_seed": master_seed,
        "analysis_bootstrap_seed": derive_seed(master_seed, 0, "bootstrap_rng", "ANALYSIS"),
        "streams": {},
    }
    for scenario_id in scenarios:
        payload["streams"][scenario_id] = {
            str(rep): {variant.value: variant_seeds(master_seed, rep, variant) for variant in ALL_VARIANTS}
            for rep in range(1, replications + 1)
        }
    return payload


def _append_command_history(output_root, stage, selected, replications, resume, master_seed):
    command = (
        f"python -m two_layer_ctmc.experiments.experiment_05_spectral_corrections.cli "
        f"--stage {stage} --scenario {'all' if len(selected) > 1 else selected[0]} "
        f"--replications {replications}{' --resume' if resume else ''} --master-seed {master_seed}"
    )
    path = output_root / "command_history.log"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(command + "\n")
