#!/usr/bin/env python3
"""Gateway-distance experiment on paired chain networks.

Only the inter-community endpoints change between ``near``, ``random``, and
``far``.  Every replication reuses exactly the same connected ER community
graphs in all three modes.  Results are written one replication at a time, so
an interrupted invocation can safely be resumed.

Examples
--------
Smoke test (3 paired network replications)::

    .venv/bin/python extrascripts/runners/run_gateway_distance_sweep.py --grid smoke --workers 2

Full experiment (100 paired network replications)::

    .venv/bin/python extrascripts/runners/run_gateway_distance_sweep.py --grid full --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulators import MicroMacroSimulator, MicroSimulator


MODES = ("near", "random", "far")
MODELS = ("Micro", "MicroMacro")


@dataclass(frozen=True)
class Config:
    grid: str
    n_runs: int
    L: int = 5
    n: int = 100
    d: float = 6.0
    k: int = 5
    beta: float = 0.003
    gamma: float = 0.0
    T_end: float = 2000.0
    tau_micro: float = 1.0
    macro_T: float = 1.0
    model: str = "SI"
    base_seed: int = 731_000
    initial_community: int = 0
    initial_node: int = 0

    @property
    def edge_prob(self) -> float:
        return self.d / (self.n - 1)


def _atomic_json(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file, then atomically replace."""
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fingerprint(cfg: Config) -> str:
    encoded = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _seed_map(cfg: Config, replication_id: int) -> dict[str, int]:
    # Widely separated streams make the provenance explicit and stable.
    offset = replication_id - 1
    return {
        "network": cfg.base_seed + offset,
        "gateway": cfg.base_seed + 100_000 + offset,
        "micro_events": cfg.base_seed + 200_000 + offset,
        "micromacro_events": cfg.base_seed + 300_000 + offset,
    }


def _choose_gateways(
    graph: nx.Graph, nodes: Sequence[int], k: int, mode: str, seed: int
) -> tuple[list[int], list[int], float, float]:
    """Choose left gateways once and right gateways by graph-distance mode."""
    if k > len(nodes):
        raise ValueError("k cannot exceed community size")
    rng_left = random.Random(seed)
    left = sorted(rng_left.sample(list(nodes), k))
    distances = dict(nx.multi_source_dijkstra_path_length(graph, left, weight=None))
    # Match the exact within-degree multiset of the fixed import set in all
    # three modes.  Distance is therefore changed without a degree confound.
    right: list[int] = []
    tie_rng = random.Random(seed + {"random": 11, "near": 23, "far": 37}[mode])
    for degree in sorted(graph.degree(node) for node in left):
        candidates = [node for node in nodes
                      if node not in right and graph.degree(node) == degree]
        if not candidates:
            raise RuntimeError("cannot match gateway within-degree multiset")
        if mode == "random":
            chosen = tie_rng.choice(candidates)
        else:
            ranked = sorted(candidates, key=lambda node: (distances[node], tie_rng.random()))
            chosen = ranked[0] if mode == "near" else ranked[-1]
        right.append(chosen)

    pairwise = [nx.shortest_path_length(graph, u, v) for u in left for v in right]
    return left, sorted(right), float(min(pairwise)), float(np.mean(pairwise))


def _build_mode_network(
    micro_graphs: Sequence[nx.Graph], mode: str, gateway_seed: int, k: int
) -> tuple[nx.Graph, np.ndarray, dict[str, Any]]:
    """Compose one chain while leaving all within-community edges untouched."""
    L = len(micro_graphs)
    comm_nodes = [sorted(graph.nodes()) for graph in micro_graphs]
    left_sets: list[list[int]] = [[] for _ in range(L)]
    right_sets: list[list[int]] = [[] for _ in range(L)]
    d_min: list[float] = []
    d_mean: list[float] = []
    by_community: list[dict[str, float | int]] = []

    # Chain endpoints have only one boundary.  Keep those endpoints identical
    # across modes; only intermediate left/right separation is manipulated.
    right_sets[0] = sorted(random.Random(gateway_seed + 1).sample(comm_nodes[0], k))
    left_sets[-1] = sorted(random.Random(gateway_seed + 2).sample(comm_nodes[-1], k))
    for community in range(1, L - 1):
        left, right, local_min, local_mean = _choose_gateways(
            micro_graphs[community], comm_nodes[community], k, mode,
            gateway_seed + 1000 * community,
        )
        left_sets[community] = left
        right_sets[community] = right
        d_min.append(local_min)
        d_mean.append(local_mean)
        by_community.append({
            "community": community,
            "D_min": local_min,
            "D_mean": local_mean,
            "left_within_degrees": sorted(int(micro_graphs[community].degree(node)) for node in left),
            "right_within_degrees": sorted(int(micro_graphs[community].degree(node)) for node in right),
        })

    full_graph = nx.Graph()
    for graph in micro_graphs:
        full_graph.add_nodes_from(graph.nodes(data=True))
        full_graph.add_edges_from(graph.edges(data=True))
    for community in range(L - 1):
        left_side = list(right_sets[community])
        right_side = list(left_sets[community + 1])
        # Pairing is deterministic and does not alter either gateway set.
        random.Random(gateway_seed + 10_000 + community).shuffle(right_side)
        full_graph.add_edges_from(
            (u, v, {"weight": 1.0}) for u, v in zip(left_side, right_side)
        )

    W = np.zeros((L, L), dtype=float)
    for community in range(L - 1):
        W[community, community + 1] = W[community + 1, community] = float(k)
    full_graph.graph.update({
        "community_sizes": [len(nodes) for nodes in comm_nodes],
        # Baseline production MicroMacro does not consume gateway identities.
        "gateway_mode": mode,
    })
    return full_graph, W, {
        "D_min": float(np.mean(d_min)),
        "D_mean": float(np.mean(d_mean)),
        "intermediate_communities": L - 2,
        "by_community": by_community,
        "degree_matched": all(
            item["left_within_degrees"] == item["right_within_degrees"]
            for item in by_community),
    }


def _micro_arrivals(result: dict[str, Any], L: int) -> list[Optional[float]]:
    arrivals: list[Optional[float]] = [None] * L
    for row in result.get("community_timings", []):
        community = int(row["community"])
        value = row.get("first_infection_time")
        if 0 <= community < L and value is not None and math.isfinite(float(value)):
            arrivals[community] = float(value)
    return arrivals


def _micromacro_arrivals(events: Iterable[dict[str, Any]], L: int) -> list[Optional[float]]:
    arrivals: list[Optional[float]] = [None] * L
    for event in sorted(events, key=lambda item: float(item.get("time", math.inf))):
        if str(event.get("event_type", "")).lower() not in {"seed", "infection", "transfer"}:
            continue
        community = event.get("community")
        if community is None:
            continue
        community = int(community)
        value = float(event["time"])
        if 0 <= community < L and arrivals[community] is None and math.isfinite(value):
            arrivals[community] = value
    return arrivals


def _run_replication(task: tuple[Config, int]) -> dict[str, Any]:
    cfg, replication_id = task
    seeds = _seed_map(cfg, replication_id)
    micro_graphs, _, _ = generate_two_scale_network(
        n_communities=cfg.L,
        community_size=cfg.n,
        inter_links=0,
        seed=seeds["network"],
        macro_graph_type="chain",
        micro_graph_type="random_connected",
        edge_prob=cfg.edge_prob,
    )
    comm_nodes = [sorted(graph.nodes()) for graph in micro_graphs]
    rows: list[dict[str, Any]] = []
    descriptors: dict[str, dict[str, Any]] = {}
    micromacro_arrivals: list[Optional[float]] | None = None
    micromacro_runtime: float | None = None

    for mode in MODES:
        full_graph, W, descriptor = _build_mode_network(
            micro_graphs, mode, seeds["gateway"], cfg.k
        )
        descriptors[mode] = descriptor

        started = time.perf_counter()
        micro_result = MicroSimulator(
            full_graph=full_graph,
            comm_nodes=comm_nodes,
            infection_rate=cfg.beta,
            recovery_rate=cfg.gamma,
            model=2,  # SI is SIR with gamma=0 in the production API.
        ).run(
            T_end=cfg.T_end,
            dt_out=cfg.tau_micro,
            initial_node=cfg.initial_node,
            rng=random.Random(seeds["micro_events"]),
        )
        rows.append({
            "mode": mode,
            "model": "Micro",
            "runtime_seconds": time.perf_counter() - started,
            "arrivals": _micro_arrivals(micro_result, cfg.L),
        })

        if micromacro_arrivals is None:
            started = time.perf_counter()
            _, _, _, event_log = MicroMacroSimulator(
                W=W,
                micro_graphs=list(micro_graphs),
                beta_micro=cfg.beta,
                gamma=cfg.gamma,
                beta_macro=cfg.beta,
                tau_micro=cfg.tau_micro,
                T_end=cfg.T_end,
                macro_T=cfg.macro_T,
                model=2,
                full_graph=full_graph,
            ).run(
                seed=seeds["micromacro_events"],
                initial_community=cfg.initial_community,
                initial_node=cfg.initial_node,
            )
            micromacro_runtime = time.perf_counter() - started
            micromacro_arrivals = _micromacro_arrivals(event_log, cfg.L)
        rows.append({
            "mode": mode,
            "model": "MicroMacro",
            "runtime_seconds": micromacro_runtime if mode == MODES[0] else 0.0,
            "reused_across_modes": mode != MODES[0],
            "arrivals": micromacro_arrivals,
        })

    return {
        "schema_version": 1,
        "config_fingerprint": _fingerprint(cfg),
        "replication_id": replication_id,
        "seeds": seeds,
        "gateway_descriptors": descriptors,
        "results": rows,
    }


def _valid_replication(path: Path, cfg: Config, replication_id: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pairs = {(row["mode"], row["model"]) for row in payload["results"]}
        return (
            payload.get("config_fingerprint") == _fingerprint(cfg)
            and payload.get("replication_id") == replication_id
            and pairs == {(mode, model) for mode in MODES for model in MODELS}
            and all(len(row["arrivals"]) == cfg.L for row in payload["results"])
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _empirical_w1(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size or left.size == 0:
        return float("nan")
    return float(np.mean(np.abs(np.sort(left) - np.sort(right))))


def _summarize(replications: Sequence[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        descriptor_min = [rep["gateway_descriptors"][mode]["D_min"] for rep in replications]
        descriptor_mean = [rep["gateway_descriptors"][mode]["D_mean"] for rep in replications]
        indexed = {
            (rep["replication_id"], result["model"]): result
            for rep in replications for result in rep["results"] if result["mode"] == mode
        }
        ids = sorted(rep["replication_id"] for rep in replications)
        for layer in range(cfg.L):
            micro_raw = [indexed[(rep_id, "Micro")]["arrivals"][layer] for rep_id in ids]
            mm_raw = [indexed[(rep_id, "MicroMacro")]["arrivals"][layer] for rep_id in ids]
            micro = np.asarray([cfg.T_end if value is None else value for value in micro_raw], dtype=float)
            mm = np.asarray([cfg.T_end if value is None else value for value in mm_raw], dtype=float)
            rows.append({
                "mode": mode,
                "layer": layer,
                "n_replications": len(ids),
                "D_min_mean": float(np.mean(descriptor_min)),
                "D_mean_mean": float(np.mean(descriptor_mean)),
                "micro_completion": sum(value is not None for value in micro_raw) / len(ids),
                "micromacro_completion": sum(value is not None for value in mm_raw) / len(ids),
                "delta_completion": (
                    sum(value is not None for value in mm_raw)
                    - sum(value is not None for value in micro_raw)
                ) / len(ids),
                "delta_RMST": float(np.mean(mm - micro)),
                "restricted_W1": _empirical_w1(micro, mm),
                "restricted_relative_W1": (
                    _empirical_w1(micro, mm) / float(np.mean(micro))
                    if float(np.mean(micro)) > 0 else None
                ),
            })
    return rows


def _write_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _summarize_contrasts(replications: Sequence[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    errors: dict[str, list[float]] = {mode: [] for mode in MODES}
    for replication in sorted(replications, key=lambda item: item["replication_id"]):
        indexed = {(row["mode"], row["model"]): row for row in replication["results"]}
        for mode in MODES:
            micro = indexed[(mode, "Micro")]["arrivals"][-1]
            mm = indexed[(mode, "MicroMacro")]["arrivals"][-1]
            errors[mode].append((cfg.T_end if mm is None else mm)
                                - (cfg.T_end if micro is None else micro))
    rng = np.random.default_rng(cfg.base_seed + 900_000)
    rows = []
    for left, right in (("near", "random"), ("near", "far"), ("random", "far")):
        values = np.asarray(errors[right]) - np.asarray(errors[left])
        estimates = np.empty(2000)
        for index in range(len(estimates)):
            draw = rng.integers(0, len(values), len(values))
            estimates[index] = values[draw].mean()
        low, high = np.quantile(estimates, (0.025, 0.975))
        rows.append({
            "contrast": f"error_{right}_minus_{left}",
            "n_network_blocks": len(values),
            "estimate": float(values.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "bootstrap_unit": "paired network block",
        })
    return rows


def _manifest(cfg: Config, completed: Sequence[int], failures: dict[int, str]) -> dict[str, Any]:
    expected = list(range(1, cfg.n_runs + 1))
    return {
        "schema_version": 1,
        "experiment": "gateway_distance_near_random_far",
        "complete": list(completed) == expected,
        "config_fingerprint": _fingerprint(cfg),
        "parameters": asdict(cfg),
        "seed_policy": {
            "formula": "base_seed + replication_id - 1, with stream offsets 0/100000/200000/300000",
            "base_seed": cfg.base_seed,
            "network_paired_across_modes": True,
        },
        "placement_policy": {
            "modes": list(MODES),
            "invariant": "internal graphs and k edges per adjacent community pair",
            "descriptors": "mean across intermediate communities of pairwise gateway D_min and D_mean",
            "within_degree_multiset_matched": True,
        },
        "expected_replications": expected,
        "completed_replications": list(completed),
        "failed_replications": {str(key): value for key, value in sorted(failures.items())},
        "compact_output": "arrival times only; no trajectories or event logs",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run(cfg: Config, output_dir: Path, workers: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reps_dir = output_dir / "replications"
    reps_dir.mkdir(exist_ok=True)
    failures: dict[int, str] = {}
    completed = [
        rep_id for rep_id in range(1, cfg.n_runs + 1)
        if _valid_replication(reps_dir / f"rep_{rep_id:03d}.json", cfg, rep_id)
    ]
    pending = [rep_id for rep_id in range(1, cfg.n_runs + 1) if rep_id not in completed]
    _atomic_json(output_dir / "manifest.json", _manifest(cfg, sorted(completed), failures))
    print(f"gateway-distance {cfg.grid}: {len(completed)} resumed, {len(pending)} pending", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_replication, (cfg, rep_id)): rep_id for rep_id in pending}
        for future in concurrent.futures.as_completed(futures):
            rep_id = futures[future]
            try:
                payload = future.result()
                _atomic_json(reps_dir / f"rep_{rep_id:03d}.json", payload)
                completed.append(rep_id)
                failures.pop(rep_id, None)
                print(f"  completed {len(completed)}/{cfg.n_runs} (rep {rep_id})", flush=True)
            except Exception as exc:  # Preserve other completed reps and continue.
                failures[rep_id] = f"{type(exc).__name__}: {exc}"
                print(f"  FAILED rep {rep_id}: {failures[rep_id]}", file=sys.stderr, flush=True)
            _atomic_json(output_dir / "manifest.json", _manifest(cfg, sorted(completed), failures))

    payloads = [
        json.loads((reps_dir / f"rep_{rep_id:03d}.json").read_text(encoding="utf-8"))
        for rep_id in sorted(completed)
    ]
    if payloads:
        _write_summary(output_dir / "summary.csv", _summarize(payloads, cfg))
        _write_summary(output_dir / "contrasts.csv", _summarize_contrasts(payloads, cfg))
    _atomic_json(output_dir / "manifest.json", _manifest(cfg, sorted(completed), failures))
    if len(completed) != cfg.n_runs:
        raise SystemExit(f"incomplete: {len(completed)}/{cfg.n_runs}; rerun the same command to resume")
    print(f"complete: {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe near/random/far gateway-distance sweep on paired connected-ER chains."
    )
    parser.add_argument("--grid", choices=("smoke", "full"), default="smoke",
                        help="smoke=3 replications; full=100 replications")
    parser.add_argument("--workers", type=int, default=2, help="parallel replication workers")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: output/gateway_distance_<grid>")
    parser.add_argument("--base-seed", type=int, default=731_000)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    cfg = Config(grid=args.grid, n_runs=3 if args.grid == "smoke" else 100,
                 base_seed=args.base_seed)
    output_dir = args.output_dir or REPO_ROOT / "output" / f"gateway_distance_{args.grid}"
    run(cfg, output_dir.resolve(), args.workers)


if __name__ == "__main__":
    main()
