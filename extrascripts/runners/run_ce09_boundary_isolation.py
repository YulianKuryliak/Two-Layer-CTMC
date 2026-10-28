"""
CE-09: Boundary isolation — quantify error from WHERE imports are placed.

Мета: виміряти вклад МІСЦЯ розміщення імпортів (uniform vs boundary-aware)
у похибку MicroMacro як функцію структури границь спільнот (d, k параметри).

Placement variants:
  - UNIFORM_PLACEMENT: uniform random susceptible (current MicroMacro default)
  - BOUNDARY_PLACEMENT: degree-weighted boundary nodes (PARAMETERS-ONLY surrogate)
  - ORACLE_EXACT_PLACEMENT: exact edge-conditioned target (ORACLE upper bound)

Усі варіанти використовують AGGREGATE_HAZARD (реалістична макро-модель), щоб
ізолювати саме ефект placement. Oracle точний hazard не є реалістичним, тому
не використовується (це вже було вивчено в CE-01 ablation).

Сценарії: варіюємо d (mean degree) і k (inter-community links) для зміни
boundary fraction — sparse/few bridges → менше boundary nodes.

Запуск (smoke test 3 reps):
    .venv/bin/python extrascripts/runners/run_ce09_boundary_isolation.py --grid smoke --workers 4

Пілотний запуск (20 reps):
    .venv/bin/python extrascripts/runners/run_ce09_boundary_isolation.py --grid pilot --workers 4

Повний запуск (40 reps):
    .venv/bin/python extrascripts/runners/run_ce09_boundary_isolation.py --grid full --workers 8

Resume:
    .venv/bin/python extrascripts/runners/run_ce09_boundary_isolation.py --grid full --workers 8 --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulators.micro import MicroSimulator
from two_layer_ctmc.experiments.hazard_placement_ablation.config import (
    HazardMode,
    ModelVariant,
    PlacementMode,
    VARIANT_MODES,
    STREAM_NAMES,
)
from two_layer_ctmc.experiments.hazard_placement_ablation.simulator import HazardPlacementSimulator


# ============================================================================
# Scenario definition
# ============================================================================

@dataclass(frozen=True)
class ScenarioConfig:
    """CE-09 scenario: network structure + placement policy."""
    name: str
    L: int
    n: int
    k: int
    d: float
    beta_in: float
    beta_out: float
    gamma: float
    r: float
    T_end: float
    placement_mode: str  # "uniform", "boundary", "oracle_exact"
    tau_micro: float = 1.0
    base_seed: int = 900000
    n_runs: int = 40
    initial_community: int = 0
    initial_node: Optional[int] = 0

    @property
    def edge_prob(self) -> float:
        return self.d / (self.n - 1)


def adaptive_T_end(n: int, d: float, beta_in: float, beta_out: float, L: int, safety: float = 1.6) -> float:
    """Адаптивний горизонт для порівнюваної цензури (див. CE-06)."""
    within = 0.5 * math.log(max(n, 2)) / (beta_in * max(d - 1.0, 1.0))
    crossing = 1.0 / beta_out
    raw = safety * L * (within + crossing)
    if raw <= 0:
        return 1000.0
    mag = 10 ** math.floor(math.log10(raw))
    return float(math.ceil(raw / mag * 10) / 10 * mag)


def build_grid(name: str) -> List[ScenarioConfig]:
    """Побудувати сітку сценаріїв для CE-09."""
    beta_in = 0.003
    r = 1.0  # CE-09 фокусується на placement, не на rate ratio
    L, n = 10, 100

    if name == "smoke":
        dk_pairs = [(6.0, 5), (12.0, 2)]
        n_runs = 3
        base = 900000
    elif name == "pilot":
        dk_pairs = [(4.0, 5), (6.0, 5), (12.0, 5), (6.0, 2)]
        n_runs = 20
        base = 910000
    elif name == "full":
        dk_pairs = [(4.0, 2), (4.0, 5), (6.0, 2), (6.0, 5), (12.0, 2), (12.0, 5)]
        n_runs = 40
        base = 920000
    else:
        raise ValueError(f"unknown grid: {name}")

    # Placement modes: uniform (baseline), boundary (params-only), oracle_exact (upper bound)
    placement_modes = ["uniform", "boundary", "oracle_exact"]

    cells: List[ScenarioConfig] = []
    idx = 0
    for d, k in dk_pairs:
        for pmode in placement_modes:
            beta_out = beta_in * r
            T_end = adaptive_T_end(n, d, beta_in, beta_out, L)
            cname = f"d{d:g}_k{k}_{pmode}"
            cells.append(ScenarioConfig(
                name=cname, L=L, n=n, k=k, d=d,
                beta_in=beta_in, beta_out=beta_out, gamma=0.0, r=r,
                T_end=T_end, placement_mode=pmode, n_runs=n_runs,
                base_seed=base + idx * 1000,
            ))
            idx += 1
    return cells


# ============================================================================
# Boundary descriptor: повторюємо з CE-06 compute_descriptors
# ============================================================================

def compute_boundary_count(micro_graphs, full_graph, comm_nodes) -> float:
    """Середня кількість boundary nodes на спільноту (параметр/статистика)."""
    import networkx as nx
    comm_of = {}
    for c, nodes in enumerate(comm_nodes):
        for u in nodes:
            comm_of[u] = c
    boundary_nodes = {c: set() for c in range(len(comm_nodes))}
    for u, v in full_graph.edges():
        cu, cv = comm_of.get(u), comm_of.get(v)
        if cu is not None and cv is not None and cu != cv:
            boundary_nodes[cu].add(u)
            boundary_nodes[cv].add(v)
    counts = [len(boundary_nodes[c]) for c in range(len(comm_nodes))]
    return float(np.mean(counts)) if counts else float("nan")


# ============================================================================
# Arrivals extraction
# ============================================================================

def _arrival_times_micro(result: dict, L: int) -> List[Optional[float]]:
    out = [None] * L
    for row in result.get("community_timings", []):
        c = int(row["community"])
        if 0 <= c < L and row.get("first_infection_time") is not None:
            out[c] = float(row["first_infection_time"])
    return out


def _arrival_times_from_ablation(arrivals: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Hazard ablation simulator повертає arrivals безпосередньо."""
    return [float(t) if t is not None and math.isfinite(t) else None for t in arrivals]


# ============================================================================
# Single replication runner
# ============================================================================

def _run_one(args: Tuple) -> List[dict]:
    """
    Запустити одну реплікацію: Full Micro (reference) + ablation variant (placement).
    
    Повертає список рядків для arrivals.csv (по одному на layer × model).
    """
    cfg, rep_id = args
    network_seed = cfg.base_seed + rep_id - 1
    init_seed = cfg.base_seed + 50000 + rep_id - 1
    event_seed = cfg.base_seed + 100000 + rep_id - 1

    # Генерувати мережу
    micro_graphs, full_graph, _ = generate_two_scale_network(
        n_communities=cfg.L,
        community_size=cfg.n,
        inter_links=cfg.k,
        seed=network_seed,
        macro_graph_type="chain",
        micro_graph_type="random_connected",
        edge_prob=cfg.edge_prob,
    )
    comm_nodes = [list(g.nodes()) for g in micro_graphs]
    boundary_count = compute_boundary_count(micro_graphs, full_graph, comm_nodes)

    # Ініціалізація
    initial_node = (
        int(cfg.initial_node) if cfg.initial_node is not None
        else int(random.Random(init_seed).choice(comm_nodes[cfg.initial_community]))
    )

    rows: List[dict] = []

    # --- Full Micro reference ---
    t0 = time.perf_counter()
    micro_result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=cfg.beta_in,
        recovery_rate=cfg.gamma,
        model=2,  # SIR (gamma=0 → SI)
    ).run(
        T_end=cfg.T_end,
        dt_out=cfg.tau_micro,
        initial_node=initial_node,
        rng=random.Random(event_seed),
    )
    micro_rt = time.perf_counter() - t0
    micro_arrivals = _arrival_times_micro(micro_result, cfg.L)

    for layer, t_val in enumerate(micro_arrivals):
        rows.append({
            "scenario": cfg.name,
            "replication_id": rep_id,
            "placement_mode": cfg.placement_mode,
            "model": "Micro",
            "layer": layer,
            "T_i": t_val,
            "runtime": micro_rt,
            "network_seed": network_seed,
            "boundary_count_mean": boundary_count,
        })

    # --- Ablation variant (placement mode) ---
    # Використовуємо HazardPlacementSimulator з AGGREGATE_HAZARD + placement mode
    # ORACLE_EXACT_PLACEMENT = exact edge-conditioned (oracle upper bound)
    # BOUNDARY_WEIGHTED = degree-weighted boundary (parameters-only surrogate)
    # UNIFORM_SUSCEPTIBLE = uniform (baseline)
    if cfg.placement_mode == "oracle_exact":
        placement_enum = PlacementMode.EXACT_EDGE_CONDITIONED
    elif cfg.placement_mode == "boundary":
        placement_enum = PlacementMode.BOUNDARY_WEIGHTED
    else:
        placement_enum = PlacementMode.UNIFORM_SUSCEPTIBLE

    # Створюємо pseudo-variant для ablation simulator (hazard=AGGREGATE, placement=...)
    # Використовуємо AGGREGATE_HAZARD_UNIFORM_PLACEMENT як шаблон конфігу
    from two_layer_ctmc.experiments.hazard_placement_ablation.config import ExperimentConfig

    # Створюємо ablation config зі сценарію
    ablation_config = ExperimentConfig(
        L=cfg.L,
        n=cfg.n,
        mean_internal_degree=cfg.d,
        k=cfg.k,
        beta_in=cfg.beta_in,
        beta_out=cfg.beta_out,
        gamma=cfg.gamma,
        tau_micro=cfg.tau_micro,
        T_end=cfg.T_end,
        base_seed=cfg.base_seed,
        initial_community=cfg.initial_community,
        initial_node=cfg.initial_node,
    )

    # Створюємо stream seeds для ablation simulator
    stream_seed_map = {
        "network_rng": network_seed,
        "initialization_rng": init_seed,
        "micro_event_rng": event_seed,
        "macro_budget_rng": event_seed + 1,
        "macro_pair_rng": event_seed + 2,
        "placement_rng": event_seed + 3,
        "bootstrap_rng": event_seed + 4,
    }

    # Маніпулюємо VARIANT_MODES для тимчасового варіанту
    # (не змінюємо код в two_layer_ctmc/, просто передаємо моки)
    # Створюємо тимчасовий ModelVariant для цього запуску
    # TRICK: використовуємо існуючий enum, але підміняємо logic в simulator
    # Насправді, HazardPlacementSimulator приймає variant і шукає в VARIANT_MODES
    # Безпечніше: використовувати існуючі варіанти прямо
    if cfg.placement_mode == "oracle_exact":
        # EXACT_EDGE_CONDITIONED + AGGREGATE_HAZARD — це не існуюча комбінація
        # але ORACLE_EXACT_HAZARD_EXACT_PLACEMENT використовує EXACT_ACTIVE_EDGE + EXACT_EDGE_CONDITIONED
        # Для consistency ми хочемо AGGREGATE_HAZARD + EXACT_EDGE_CONDITIONED
        # PROBLEM: VARIANT_MODES не має цієї комбінації
        # SOLUTION: використаємо ORACLE_EXACT_HAZARD_EXACT_PLACEMENT для oracle upper bound
        # (хоча це змішує hazard і placement effects — але це oracle anyway)
        variant_key = ModelVariant.ORACLE_EXACT_HAZARD_EXACT_PLACEMENT
    elif cfg.placement_mode == "boundary":
        variant_key = ModelVariant.AGGREGATE_HAZARD_BOUNDARY_PLACEMENT
    else:
        variant_key = ModelVariant.AGGREGATE_HAZARD_UNIFORM_PLACEMENT

    # ВАЖЛИВО: для CE-09 ми хочемо ізолювати placement effect при AGGREGATE_HAZARD
    # але oracle_exact технічно змішує hazard+placement (oracle на обох)
    # Коментар в output позначить цей нюанс

    t0 = time.perf_counter()
    simulator = HazardPlacementSimulator(
        full_graph=full_graph,
        communities=comm_nodes,
        config=ablation_config,
        variant=variant_key,
        stream_seed_map=stream_seed_map,
    )
    result = simulator.run(int(initial_node))
    variant_rt = time.perf_counter() - t0
    variant_arrivals = _arrival_times_from_ablation(result["arrivals"])

    model_label = f"Ablation_{cfg.placement_mode}"
    for layer, t_val in enumerate(variant_arrivals):
        rows.append({
            "scenario": cfg.name,
            "replication_id": rep_id,
            "placement_mode": cfg.placement_mode,
            "model": model_label,
            "layer": layer,
            "T_i": t_val,
            "runtime": variant_rt,
            "network_seed": network_seed,
            "boundary_count_mean": boundary_count,
        })

    return rows


# ============================================================================
# Metrics computation (повторює CE-06 logic)
# ============================================================================

def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        return float("nan")


def compute_metrics(arrivals_rows: List[dict], T_end: float, bootstrap_reps: int = 1000, seed: int = 42) -> dict:
    """Обчислити метрики порівняння ablation vs Micro (як CE-06)."""
    df = pd.DataFrame(arrivals_rows)
    rng = np.random.default_rng(seed)
    last_layer = df["layer"].max()
    terminal = df[df["layer"] == last_layer].copy()

    micro_t = terminal[terminal["model"] == "Micro"].set_index("replication_id")["T_i"].to_dict()
    # Ablation model name contains placement mode
    ablation_label = [m for m in terminal["model"].unique() if m.startswith("Ablation_")]
    if not ablation_label:
        raise ValueError("No ablation model found in arrivals")
    ablation_label = ablation_label[0]
    ablation_t = terminal[terminal["model"] == ablation_label].set_index("replication_id")["T_i"].to_dict()

    all_ids = sorted(set(micro_t) | set(ablation_t))
    ref = np.array([micro_t.get(i) for i in all_ids], dtype=float)
    var = np.array([ablation_t.get(i) for i in all_ids], dtype=float)

    censored_ref = np.where(np.isfinite(ref), ref, T_end)
    censored_var = np.where(np.isfinite(var), var, T_end)
    delta_rmst = float(np.mean(censored_var - censored_ref))

    ref_obs = ref[np.isfinite(ref)]
    var_obs = var[np.isfinite(var)]

    diffs = censored_var - censored_ref
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(bootstrap_reps)])
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])

    w1 = _w1(ref_obs, var_obs)
    mean_ref = float(np.mean(ref_obs)) if ref_obs.size > 0 else float("nan")
    iqr_ref = float(np.subtract(*np.percentile(ref_obs, [75, 25]))) if ref_obs.size >= 2 else float("nan")
    rel_w1 = w1 / mean_ref if math.isfinite(mean_ref) and mean_ref > 0 else float("nan")
    w1_over_iqr = w1 / iqr_ref if math.isfinite(iqr_ref) and iqr_ref > 0 else float("nan")

    p_compl_micro = float(np.isfinite(ref).mean())
    p_compl_ablation = float(np.isfinite(var).mean())

    return {
        "p_completion": p_compl_ablation,
        "micro_completion": p_compl_micro,
        "delta_completion": p_compl_ablation - p_compl_micro,
        "micro_mean_T_last": mean_ref,
        "ablation_mean_T_last": float(np.mean(var_obs)) if var_obs.size else float("nan"),
        "W1": w1,
        "rel_W1": rel_w1,
        "W1_over_IQR": w1_over_iqr,
        "delta_RMST": delta_rmst,
        "delta_RMST_norm": delta_rmst / T_end if T_end > 0 else float("nan"),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


# ============================================================================
# Main experiment runner
# ============================================================================

def _cell_filename(name: str) -> str:
    return name.replace("=", "").replace(".", "p").replace(" ", "_")


def run_experiment(cells: List[ScenarioConfig], out_dir: Path, n_workers: int = 4, resume: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    manifest = {
        "experiment": "CE-09 boundary isolation",
        "grid_cells": len(cells),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cells": [],
        "note_oracle_placement": (
            "oracle_exact uses ORACLE_EXACT_HAZARD_EXACT_PLACEMENT which mixes oracle hazard + placement. "
            "This is an upper bound on placement gains but not a pure placement-only ablation. "
            "boundary mode uses AGGREGATE_HAZARD_BOUNDARY_PLACEMENT (params-only). "
            "uniform mode uses AGGREGATE_HAZARD_UNIFORM_PLACEMENT (current MicroMacro default)."
        )
    }

    for ci, cell in enumerate(cells, 1):
        cell_arr_path = cells_dir / f"arrivals_{_cell_filename(cell.name)}.csv"
        if resume and cell_arr_path.exists():
            print(f"[{ci}/{len(cells)}] skip existing {cell.name}", flush=True)
            scenario_rows = pd.read_csv(cell_arr_path).to_dict("records")
        else:
            print(f"\n[{ci}/{len(cells)}] === {cell.name}  (T_end={cell.T_end:g}, n_runs={cell.n_runs}) ===", flush=True)
            tasks = [(cell, rep) for rep in range(1, cell.n_runs + 1)]
            scenario_rows = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_one, task): task[1] for task in tasks}
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    rep_id = futures[future]
                    try:
                        scenario_rows.extend(future.result())
                    except Exception as exc:
                        print(f"  rep {rep_id} failed: {exc}", flush=True)
                        import traceback
                        traceback.print_exc()
                    if i % 20 == 0 or i == cell.n_runs:
                        print(f"  {i}/{cell.n_runs} done", flush=True)
            pd.DataFrame(scenario_rows).to_csv(cell_arr_path, index=False)

        metrics = compute_metrics(scenario_rows, cell.T_end)
        # Boundary count (mean across reps)
        boundary_vals = [r["boundary_count_mean"] for r in scenario_rows if "boundary_count_mean" in r]
        boundary_mean = float(np.mean([b for b in boundary_vals if math.isfinite(b)])) if boundary_vals else float("nan")

        row = {
            "scenario": cell.name,
            "placement_mode": cell.placement_mode,
            "d": cell.d,
            "k": cell.k,
            "L": cell.L,
            "n": cell.n,
            "r": cell.r,
            "beta_in": cell.beta_in,
            "beta_out": cell.beta_out,
            "T_end": cell.T_end,
            "n_runs": cell.n_runs,
            "boundary_count_mean": boundary_mean,
            **metrics,
        }
        summary_rows.append(row)
        manifest["cells"].append({
            "name": cell.name,
            "placement_mode": cell.placement_mode,
            "n_runs": cell.n_runs,
            "p_completion": metrics["p_completion"],
            "micro_completion": metrics["micro_completion"],
        })
        print(f"  dRMST={metrics['delta_RMST']:.1f} (norm={metrics['delta_RMST_norm']:.4f}) "
              f"rel_W1={metrics['rel_W1']:.4f} p_compl={metrics['p_completion']:.3f} "
              f"boundary_count={boundary_mean:.1f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n=== CE-09 Complete ===\nResults: {out_dir}/summary.csv", flush=True)
    cols = ["scenario", "placement_mode", "d", "k", "boundary_count_mean",
            "p_completion", "delta_RMST_norm", "rel_W1", "W1_over_IQR"]
    print("\n" + summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="CE-09 boundary isolation: placement error vs boundary structure")
    parser.add_argument("--grid", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs per cell")
    parser.add_argument("--resume", action="store_true", help="Skip cells with existing arrivals CSV")
    args = parser.parse_args()

    cells = build_grid(args.grid)
    if args.n_runs is not None:
        from dataclasses import replace
        cells = [replace(c, n_runs=args.n_runs) for c in cells]

    out_dir = REPO_ROOT / "output" / "claude_experiments" / f"ce09_boundary_isolation_{args.grid}"
    run_experiment(cells, out_dir, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
