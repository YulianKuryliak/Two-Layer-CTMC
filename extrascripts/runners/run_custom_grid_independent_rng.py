#!/usr/bin/env python3
"""Generate a user-defined Full-vs-Production MicroMacro parameter grid.

Edit the configuration block below and run:

    python extrascripts/runners/run_custom_grid_independent_rng.py

This script only generates simulation data.  It does not create figures.
Completed replication checkpoints are reused automatically, so rerunning the
command acts as resume. Workers atomically save numeric-column checkpoints;
no consolidated CSVs are produced. Plotters must read checkpoints, not old CSVs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import monotonic

# When executed as ``python extrascripts/runners/<script>.py``, Python puts
# only the script directory on sys.path; add the repository root explicitly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extrascripts.runners import run_original_benchmarks_independent_rng as base
from extrascripts.runners.checkpoint_io import has_trajectories, read_checkpoint, write_checkpoint


# ============================ EDIT HERE ============================
COMMUNITIES = 10
BETA_VALUES = [0.001] #0.001, 0.003
K_VALUES = [1, 5, 10]
# Use integers for random_connected mean degree and "complete" for cliques.
D_VALUES = [6, 10, "complete"]
COMMUNITY_SIZES = [50, 100, 250, 1000]
TOPOLOGY_TYPES = ["chain_cliques", "chain_random_connected"]

T_END = 30000.0                # maximum observation horizon for this grid
# When T_END is raised, keep an old paired replication only if both models had
# already reached every community.  Censored pairs are reproducibly rerun with
# their original deterministic seeds at the new horizon.
REUSE_COMPLETED_PAIRS_ON_HORIZON_EXTENSION = True
TAU_MICRO = 1.0
DT_OUT = 1.0
INITIAL_NODE = 0
REPLICATIONS = 100              # paired replications per grid cell
WORKERS = 12                    # total simulation workers
HEAVY_WORKERS = 4              # workers reserved for community_size >= 1000
BASE_SEED = 20260911
# All new independent-RNG scenarios live directly below this data root.
OUTPUT_DIR = Path("output/data")
# ===================================================================


def _text(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def scenario_name(topology: str, n: int, d: int | str, k: int, beta: float) -> str:
    topology_text = {
        "chain_cliques": "chain_cliques",
        "chain_random_connected": "chain_random_connected",
    }.get(topology, topology)
    return (
        f"{topology_text}_L{COMMUNITIES}_n{n}_d{_text(d)}"
        f"_k{k}_beta{_text(beta).replace('.', 'p')}_independent_rng"
    )


def seed_for(scenario: str, replication: int, kind: str) -> int:
    raw = f"{BASE_SEED}:{scenario}:{replication}:{kind}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31)


def make_benchmark(topology: str, n: int, d: int | str, k: int, beta: float):
    if topology not in TOPOLOGY_TYPES:
        raise ValueError(f"unsupported topology: {topology}")
    if n < 2:
        raise ValueError("community sizes must be >= 2")
    if k < 1:
        raise ValueError("k must be >= 1")
    if topology == "chain_cliques":
        micro_type = "complete"
        edge_prob = 1.0
        mean_degree = None
    else:
        if d == "complete":
            raise ValueError("chain_random_connected requires a numeric d")
        if not isinstance(d, (int, float)) or not (1 < float(d) < n):
            raise ValueError(f"random_connected d must satisfy 1 < d < n; got d={d}, n={n}")
        micro_type = "random_connected"
        mean_degree = float(d)
        edge_prob = float(d) / (n - 1)
    return replace(
        base.BENCHMARKS[1],
        name=scenario_name(topology, n, d, k, beta),
        family="custom_grid",
        plot_groups=("custom_grid",),
        replications=REPLICATIONS,
        communities=COMMUNITIES,
        community_size=n,
        inter_links=k,
        macro_graph_type="chain",
        micro_graph_type=micro_type,
        edge_prob=edge_prob,
        mean_internal_degree=mean_degree,
        beta=float(beta),
        gamma=0.0,
        model=2,                  # SI: production model 2 with gamma=0
        tau_micro=TAU_MICRO,
        macro_T=1.0,
        original_T_end=T_END,
        T_end=T_END,
        dt_out=DT_OUT,
        initial_node=INITIAL_NODE,
        store_trajectories=True,
    )


def scenarios():
    return [
        make_benchmark(topology, n, d, k, beta)
        for topology in TOPOLOGY_TYPES
        for n in COMMUNITY_SIZES
        for d in D_VALUES
        for k in K_VALUES
        for beta in BETA_VALUES
        if topology == "chain_cliques" and d == "complete"
        or topology == "chain_random_connected" and d != "complete"
    ]


def _checkpoint_ids(directory: Path) -> dict[int, Path]:
    result = {}
    for path in directory.glob("rep_*.json.gz"):
        try:
            result[int(path.stem.split("_")[-1].split(".")[0])] = path
        except ValueError:
            continue
    return result


def _same_number(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def _compatible_config(existing: dict, requested: object) -> bool:
    # edge_prob is deliberately omitted: for complete graphs it is unused,
    # while random_connected is identified by mean_internal_degree.
    for key in (
        "communities", "community_size", "inter_links", "macro_graph_type",
        "micro_graph_type", "beta", "gamma", "model", "tau_micro",
        "macro_T", "dt_out", "initial_node", "mean_internal_degree",
    ):
        if key not in existing or not _same_number(existing.get(key), getattr(requested, key)):
            return False
    return True


def _paired_completion(payload: dict) -> bool:
    """True only when both paired simulators reached all communities."""
    runs = payload.get("runs", [])
    completion = {row.get("simulator"): bool(row.get("completion")) for row in runs}
    return all(completion.get(model) is True for model in base.SIMULATORS)


def _stored_seeds(path: Path, fallback: tuple[int, int]) -> tuple[int, int]:
    """Reuse the exact seeds recorded in a prior partial checkpoint."""
    try:
        payload = read_checkpoint(path)
        pairs = {
            (int(row["network_seed"]), int(row["event_seed"]))
            for row in payload.get("runs", [])
            if row.get("network_seed") is not None and row.get("event_seed") is not None
        }
        if len(pairs) == 1:
            return next(iter(pairs))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return fallback


def _checkpoint_horizon(payload: dict) -> float | None:
    config = payload.get("benchmark_config", {})
    try:
        return float(config["T_end"])
    except (KeyError, TypeError, ValueError):
        return None


def inspect_reuse(scenario_output: Path, item) -> tuple[dict[int, Path], dict[str, int]]:
    """Return reusable pairs; selectively extend only prior censored pairs.

    A completed pair stores all arrival quantities needed by this experiment,
    regardless of a later, longer observation horizon. A censored pair is
    rerun from the same seeds, never continued from a partial engine state.
    """
    checkpoints = scenario_output / "checkpoints"
    ids = _checkpoint_ids(checkpoints) if checkpoints.exists() else {}
    if not ids:
        return {}, {"reused_completed_short": 0, "rerun_censored_short": 0}
    manifest_path = scenario_output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    reusable: dict[int, Path] = {}
    stats = {"reused_completed_short": 0, "rerun_censored_short": 0}
    for replication, path in sorted(ids.items()):
        payload = read_checkpoint(path)
        config = payload.get("benchmark_config") or manifest.get("benchmark_config", {})
        if not _compatible_config(config, item):
            raise RuntimeError(
                f"Existing checkpoint {path} has incompatible parameters; "
                "choose a different scenario name or archive it manually."
            )
        horizon = _checkpoint_horizon(payload)
        usable_trajectory = not item.store_trajectories or has_trajectories(payload)
        if horizon is not None and horizon >= float(item.T_end) - 1e-12 and usable_trajectory:
            reusable[replication] = path
        elif (
            horizon is not None
            and horizon < float(item.T_end) - 1e-12
            and usable_trajectory
            and REUSE_COMPLETED_PAIRS_ON_HORIZON_EXTENSION
            and _paired_completion(payload)
        ):
            reusable[replication] = path
            stats["reused_completed_short"] += 1
        elif horizon is not None and horizon < float(item.T_end) - 1e-12:
            stats["rerun_censored_short"] += 1
    if stats["reused_completed_short"] or stats["rerun_censored_short"]:
        print(
            f"[extend] {item.name}: reuse completed-before-old-horizon="
            f"{stats['reused_completed_short']}; rerun censored/partial="
            f"{stats['rerun_censored_short']} at T_end={item.T_end:g}", flush=True,
        )
    return reusable, stats


def _safe_run_task(task: tuple[dict, int, int, int, str]) -> dict:
    """Write in the worker; IPC returns only a small durable-completion receipt."""
    benchmark_dict, replication, network_seed, event_seed, output = task
    receipt = {"benchmark": benchmark_dict["name"], "replication_id": replication}
    scenario_output = Path(output) / benchmark_dict["name"]
    try:
        payload = base._run_task(
            task[:4], compact_output=True, stop_at_absorption=True
        )
    except Exception as error:
        payload = {
            "benchmark": benchmark_dict["name"],
            "replication_id": replication,
            "network_seed": network_seed,
            "event_seed": event_seed,
            "error": f"{type(error).__name__}: {error}",
        }
        receipt["error"] = payload["error"]
    # Failed attempts are diagnostic data, never reusable successful pairs.
    directory = "failures" if "error" in payload else "checkpoints"
    path = scenario_output / directory / f"rep_{replication:04d}.json.gz"
    write_checkpoint(path, payload)
    receipt["checkpoint"] = str(path)
    return receipt


def _executor_results(tasks, executor, workers: int):
    """Yield completed results promptly while bounding queued payloads."""
    iterator = iter(tasks)
    pending = set()
    for _ in range(max(1, 2 * workers)):
        try:
            pending.add(executor.submit(_safe_run_task, next(iterator)))
        except StopIteration:
            break
    while pending:
        completed, pending = concurrent.futures.wait(
            pending, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in completed:
            yield future.result()
            try:
                pending.add(executor.submit(_safe_run_task, next(iterator)))
            except StopIteration:
                pass


def _parallel_results(tasks: list[tuple[dict, int, int, int, str]], workers: int):
    """Compatibility wrapper for bounded parallel execution."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        yield from _executor_results(tasks, executor, workers)


def _task_is_heavy(task: tuple[dict, int, int, int, str]) -> bool:
    """Classify large-community tasks that need a separate memory cap."""
    return int(task[0].get("community_size", 0)) >= 1000


def _partitioned_results(
    tasks: list[tuple[dict, int, int, int, str]], workers: int, heavy_workers: int
):
    """Run large and small tasks concurrently using separate worker pools.

    Heavy workers are recycled after every replication.  This avoids retaining
    NetworkX allocator arenas after a 1000-node clique task.
    """
    if not tasks:
        return
    if workers == 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, max_tasks_per_child=1
        ) as executor:
            yield from _executor_results(tasks, executor, 1)
        return

    heavy_workers = max(1, min(int(heavy_workers), workers - 1))
    light_workers = workers - heavy_workers
    pools = {
        "heavy": (
            concurrent.futures.ProcessPoolExecutor(
                max_workers=heavy_workers, max_tasks_per_child=1
            ),
            iter(task for task in tasks if _task_is_heavy(task)),
            heavy_workers,
        ),
        "light": (
            concurrent.futures.ProcessPoolExecutor(max_workers=light_workers),
            iter(task for task in tasks if not _task_is_heavy(task)),
            light_workers,
        ),
    }
    pending: dict[concurrent.futures.Future, str] = {}

    def fill(pool_name: str) -> None:
        executor, iterator, capacity = pools[pool_name]
        current = sum(name == pool_name for name in pending.values())
        for _ in range(max(1, 2 * capacity) - current):
            try:
                task = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(_safe_run_task, task)] = pool_name

    try:
        fill("heavy")
        fill("light")
        while pending:
            completed, _ = concurrent.futures.wait(
                tuple(pending), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                pool_name = pending.pop(future)
                yield future.result()
                fill(pool_name)
    finally:
        for executor, _, _ in pools.values():
            executor.shutdown(wait=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=None, help="override replications per cell")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--heavy-workers", type=int, default=None,
        help="override HEAVY_WORKERS from the configuration block",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs is not None and args.runs < 1:
        raise ValueError("--runs must be positive")
    workers = WORKERS if args.workers is None else args.workers
    if workers < 1:
        raise ValueError("--workers must be positive")
    heavy_workers = HEAVY_WORKERS if args.heavy_workers is None else args.heavy_workers
    if workers > 1 and not (1 <= heavy_workers < workers):
        raise ValueError("--heavy-workers must be between 1 and workers-1")
    if workers == 1:
        heavy_workers = 1
    items = scenarios()
    if not items:
        raise ValueError("the parameter arrays produce no scenarios")
    if len({item.name for item in items}) != len(items):
        raise ValueError("duplicate scenario names; check the parameter arrays")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs_per_cell = args.runs if args.runs is not None else REPLICATIONS
    manifest = {
        "script": "extrascripts/runners/run_custom_grid_independent_rng.py",
        "communities": COMMUNITIES,
        "beta_values": BETA_VALUES,
        "k_values": K_VALUES,
        "d_values": D_VALUES,
        "community_sizes": COMMUNITY_SIZES,
        "topology_types": TOPOLOGY_TYPES,
        "T_end": T_END,
        "tau_micro": TAU_MICRO,
        "dt_out": DT_OUT,
        "initial_node": INITIAL_NODE,
        "replications": runs_per_cell,
        "workers": workers,
        "heavy_workers": heavy_workers,
        "base_seed": BASE_SEED,
        "simulators": [base.FULL, base.MICROMACRO],
        "scenarios": [asdict(item) for item in items],
        "seed_policy": "sha256(base_seed, scenario_name, replication, kind)",
        "independent_rng": True,
        "checkpoint_format": "gzip_json_v3_numeric_columns_v1 (legacy v2 also readable)",
        "consolidation": False,
        "checkpoint_writer": "worker; atomic rename after fsync; small IPC receipt",
        "trajectory_policy": "all original per-community reporting points, no rounding or downsampling",
        "micromacro_log_policy": "sample original change-point logs directly; midpoint event lists omitted",
        "termination_policy": (
            "Stop at an absorbing/no-event state or T_end, whichever occurs first; "
            "do not materialize held trajectory rows to T_end."
        ),
        "note": "One network is generated and reused for Full and Production per replication.",
    }
    manifest_path = output / "grid_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        # The same scenario identity may be requested with a new horizon or
        # replication count. Keep the prior grid manifest and record the new
        # request separately instead of blocking reusable scenarios.
        latest = output / "grid_manifest_latest.json"
        latest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    passed, detail = base._regression_gate()
    print(f"Regression gate: {'PASS' if passed else 'FAIL'} ({detail})", flush=True)
    if not passed:
        raise RuntimeError(detail)

    total = len(items) * runs_per_cell
    done = 0
    started = monotonic()
    print(
        f"Grid cells: {len(items)}; paired replications: {total}; "
        f"workers: {workers} (heavy={heavy_workers}, light={max(0, workers-heavy_workers)})",
        flush=True,
    )
    pending = []
    reuse_counts = {}
    for scenario_index, item in enumerate(items, 1):
        scenario_output = output / item.name
        directory = scenario_output / "checkpoints"
        scenario_output.mkdir(parents=True, exist_ok=True)
        reusable, extension = inspect_reuse(scenario_output, item)
        reusable_count = sum(
            rep in reusable and reusable[rep].exists()
            for rep in range(1, runs_per_cell + 1)
        )
        reuse_counts[item.name] = reusable_count
        scenario_pending = []
        for replication in range(1, runs_per_cell + 1):
            if replication in reusable and reusable[replication].exists():
                continue
            fallback_seeds = (
                seed_for(item.name, replication, "network"),
                seed_for(item.name, replication, "events"),
            )
            old_checkpoint = directory / f"rep_{replication:04d}.json.gz"
            network_seed, event_seed = _stored_seeds(old_checkpoint, fallback_seeds)
            scenario_pending.append(
                (asdict(item), replication, network_seed, event_seed, str(output))
            )
        pending.extend(scenario_pending)
        scenario_manifest = scenario_output / "manifest.json"
        scenario_metadata = {
            "format": "independent_rng_scenario_v1",
            "scenario": item.name,
            "algorithm": "Production MicroMacro with independent RNG per MicroModel",
            "benchmark_config": asdict(item),
            "requested_replications": runs_per_cell,
            "requested_T_end": T_END,
            "horizon_extension_policy": (
                "When requested T_end is longer, completed pairs may retain their "
                "shorter exact trajectories; only prior censored/partial pairs are "
                "rerun from deterministic seeds at requested T_end."
            ),
            "reuse_completed_pairs_on_horizon_extension": REUSE_COMPLETED_PAIRS_ON_HORIZON_EXTENSION,
            "seed_extension_policy": (
                "Partial checkpoints reuse their recorded network_seed and event_seed; "
                "deterministic derivation is only a fallback when old seeds are absent."
            ),
            "consolidation": False,
            "data_source": "checkpoints (ignore any historical consolidated CSVs)",
            "checkpoint_format": "numeric_columns_v1; mixed legacy checkpoints supported",
            "reuse_policy": "existing independent-RNG checkpoints are reused when parameters match and stored T_end >= requested T_end; seeds are not required to match",
            "note": "T_end and replication count are manifest fields, not folder-name fields.",
        }
        if scenario_manifest.exists():
            try:
                scenario_metadata["previous_manifest"] = json.loads(
                    scenario_manifest.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                pass
        scenario_manifest.write_text(json.dumps(scenario_metadata, indent=2), encoding="utf-8")
        directory.mkdir(parents=True, exist_ok=True)
        print(
            f"[queued {scenario_index}/{len(items)}] {item.name}: "
            f"reused={reusable_count}/{runs_per_cell}; "
            f"new={runs_per_cell - reusable_count}; "
            f"extended={extension['rerun_censored_short']}", flush=True
        )

    done = sum(reuse_counts.values())
    print(
        f"[run] New simulations required: {len(pending)}; reused: {done}; "
        f"workers={workers} (heavy={heavy_workers}, light={max(0, workers-heavy_workers)})",
        flush=True,
    )
    completed_by = dict(reuse_counts)
    failures_by = {}
    active_reported = set()

    def consume(payload: dict) -> None:
        nonlocal done
        benchmark_name = payload["benchmark"]
        if benchmark_name not in active_reported:
            active_reported.add(benchmark_name)
            print(f"[running] {benchmark_name}", flush=True)
        done += 1
        if "error" in payload:
            failures_by[benchmark_name] = failures_by.get(benchmark_name, 0) + 1
        completed_by[benchmark_name] = completed_by.get(benchmark_name, 0) + 1
        if completed_by[benchmark_name] >= runs_per_cell:
            status = "scenario finished with failures" if failures_by.get(benchmark_name) else "scenario complete"
            print(f"[{status}] {benchmark_name}", flush=True)
        elapsed = monotonic() - started
        eta = elapsed / max(1, done) * (total - done)
        if "error" in payload:
            print(f"  FAILED {benchmark_name} rep={payload['replication_id']}: {payload['error']}", flush=True)
        print(f"  progress {done}/{total}; remaining {total-done}; ETA {eta/60:.1f} min", flush=True)

    for receipt in _partitioned_results(pending, workers, heavy_workers):
        consume(receipt)
    print("[saved] All worker checkpoints written; CSV consolidation disabled.", flush=True)
    if failures_by:
        print(f"Failed pairs: {sum(failures_by.values())}; rerun to retry (details in failures/).", flush=True)
    print(f"Data saved to: {output}", flush=True)


if __name__ == "__main__":
    main()
