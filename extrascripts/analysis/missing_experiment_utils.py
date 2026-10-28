"""Small shared helpers for the remaining approximation-error experiments."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from two_layer_ctmc.simulators.micro import MicroSimulator
from two_layer_ctmc.simulators.micromacro import MicroMacroSimulator


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def micro_arrivals(result: dict, communities: int) -> list[float | None]:
    arrivals: list[float | None] = [None] * communities
    for row in result.get("community_timings", []):
        index = int(row["community"])
        value = row.get("first_infection_time")
        if 0 <= index < communities and value is not None:
            arrivals[index] = float(value)
    return arrivals


def mm_arrivals(event_log: Sequence[dict], communities: int) -> list[float | None]:
    arrivals: list[float | None] = [None] * communities
    for event in sorted(event_log, key=lambda item: float(item.get("time", math.inf))):
        if str(event.get("event_type", "")).lower() not in {"seed", "infection", "transfer"}:
            continue
        community = event.get("community")
        if community is None:
            continue
        index = int(community)
        if 0 <= index < communities and arrivals[index] is None:
            arrivals[index] = float(event["time"])
    return arrivals


def run_full_micro(full_graph, micro_graphs, beta: float, horizon: float,
                   initial_node: int, seed: int) -> list[float | None]:
    result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=[list(graph.nodes()) for graph in micro_graphs],
        infection_rate=beta,
        recovery_rate=0.0,
        model=2,
    ).run(T_end=horizon, dt_out=1.0, initial_node=initial_node,
          rng=random.Random(seed))
    return micro_arrivals(result, len(micro_graphs))


def run_production_mm(W, full_graph, micro_graphs, beta: float, horizon: float,
                      tau: float, initial_node: int, seed: int) -> list[float | None]:
    result = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=beta,
        gamma=0.0,
        beta_macro=beta,
        tau_micro=tau,
        T_end=horizon,
        macro_T=1.0,
        model=2,
        full_graph=full_graph,
    ).run(seed=seed, initial_community=0, initial_node=initial_node)
    return mm_arrivals(result[3], len(micro_graphs))


def arrival_rows(replication_id: int, scenario: str, model: str,
                 arrivals: Sequence[float | None], horizon: float, **metadata) -> list[dict]:
    return [{
        "replication_id": replication_id,
        "scenario": scenario,
        "model": model,
        "layer": layer,
        "T_i": value,
        "observed": value is not None,
        "restricted_T_i": float(value) if value is not None else float(horizon),
        **metadata,
    } for layer, value in enumerate(arrivals)]


def summarize_arrivals(rows: Iterable[dict], reference: str, horizon: float,
                       bootstrap_reps: int = 1000, seed: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    terminal = frame[frame.layer == frame.layer.max()].copy()
    ref = terminal[terminal.model == reference].set_index("replication_id")
    if ref.empty:
        raise ValueError(f"missing reference model {reference}")
    rng = np.random.default_rng(seed)
    output = []
    for model, group in terminal.groupby("model", sort=True):
        sample = group.set_index("replication_id").reindex(ref.index)
        if sample.restricted_T_i.isna().any():
            raise ValueError(f"unpaired terminal rows for {model}")
        ref_t = ref.restricted_T_i.to_numpy(float)
        model_t = sample.restricted_T_i.to_numpy(float)
        diff = model_t - ref_t
        w1 = float(wasserstein_distance(ref_t, model_t))
        boot = np.empty(bootstrap_reps, dtype=float)
        for index in range(bootstrap_reps):
            draw = rng.integers(0, len(diff), len(diff))
            boot[index] = diff[draw].mean()
        low, high = np.quantile(boot, (0.025, 0.975))
        output.append({
            "model": model,
            "n_runs": len(sample),
            "completion": float(sample.observed.mean()),
            "reference_completion": float(ref.observed.mean()),
            "delta_completion": float(sample.observed.mean() - ref.observed.mean()),
            "RMST": float(model_t.mean()),
            "delta_RMST": float(diff.mean()),
            "delta_RMST_ci95_low": float(low),
            "delta_RMST_ci95_high": float(high),
            "restricted_W1": w1,
            "restricted_rel_W1": w1 / float(ref_t.mean()) if ref_t.mean() else math.nan,
            "horizon": float(horizon),
        })
    return pd.DataFrame(output)


def load_completed_json(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows
