"""
CE-07: Calibration floor — розділити структурну похибку (irreducible) від
калібраційної похибки (reducible by tuning beta_out).

Метод: для кожного сценарію, запускаємо Full Micro (baseline arrivals), потім
MicroMacro з сіткою калібраційних множників c = {0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0}
(застосованих до beta_out: beta_out_eff = c * beta_out). Знаходимо c*, що
мінімізує |mean(T_MicroMacro) - mean(T_Micro)| (або KM-RMST різницю).

Записуємо:
- c*: оптимальний калібраційний множник
- dRMST_uncal: ΔRMST при c=1 (некалібрований)
- dRMST_cal: ΔRMST при c* (калібрований residual = structural floor)
- structural_fraction: |dRMST_cal| / |dRMST_uncal|
- rel_W1 для обох

ОБМЕЖЕННЯ: beta_out — ПАРАМЕТР (не точна структура), тому це дозволено як
параметричне налаштування. Симулятор не використовує точних позицій вузлів.

Запуск (smoke test):
    .venv/bin/python extrascripts/runners/run_ce07_calibration_floor.py --grid smoke --workers 4

Pilot:
    .venv/bin/python extrascripts/runners/run_ce07_calibration_floor.py --grid pilot --workers 4

Full:
    .venv/bin/python extrascripts/runners/run_ce07_calibration_floor.py --grid full --workers 8 --resume
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulate import normalize_model
from two_layer_ctmc.simulators import MicroMacroSimulator, MicroSimulator


# ---------------------------------------------------------------------------
# Cell (scenario) definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellConfig:
    name: str
    L: int
    n: int
    k: int
    d: float
    beta_in: float
    beta_out: float
    gamma: float
    T_end: float
    r: float
    model: str = "SIR"  # gamma=0 realizes SI dynamics
    tau_micro: float = 1.0
    macro_T: float = 1.0
    base_seed: int = 700000
    n_runs: int = 40
    calibration_factors: Tuple[float, ...] = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)
    initial_community: int = 0
    initial_node: Optional[int] = 0

    @property
    def edge_prob(self) -> float:
        return self.d / (self.n - 1)


def adaptive_T_end(n: int, d: float, beta_in: float, beta_out: float, L: int, safety: float = 1.6) -> float:
    """Rough per-cell horizon so that censoring stays roughly comparable.

    per-hop time ~ within-community find time + inter-community crossing time.
    within find ~ 0.5 * log(n) / (beta_in * (d - 1))   (branching, halved to
    correct the tree-approx overestimate on dense communities)
    crossing   ~ 1 / beta_out
    T_end = safety * L * (within + crossing), rounded up to a clean number.
    """
    within = 0.5 * math.log(max(n, 2)) / (beta_in * max(d - 1.0, 1.0))
    crossing = 1.0 / beta_out
    raw = safety * L * (within + crossing)
    # round up to 2 significant figures for reproducibility / clean logs
    if raw <= 0:
        return 1000.0
    mag = 10 ** math.floor(math.log10(raw))
    return float(math.ceil(raw / mag * 10) / 10 * mag)


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def build_grid(name: str) -> List[CellConfig]:
    beta_in = 0.003
    L, n = 10, 100

    if name == "smoke":
        r_values = [1.0]
        d_values = [6.0]
        k_values = [5]
        n_runs = 4
        calibration_factors = (0.7, 1.0, 1.5)
        base = 700000
    elif name == "pilot":
        r_values = [0.5, 1.0, 2.0]
        d_values = [4.0, 6.0, 12.0]
        k_values = [5]
        n_runs = 40
        calibration_factors = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)
        base = 710000
    elif name == "full":
        # 3 x 4 x 3 = 36 cells
        r_values = [0.5, 1.0, 2.0]
        d_values = [4.0, 6.0, 10.0, 20.0]
        k_values = [2, 5, 10]
        n_runs = 60
        calibration_factors = (0.3, 0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0)
        base = 720000
    else:
        raise ValueError(f"unknown grid: {name}")

    cells: List[CellConfig] = []
    idx = 0
    for r in r_values:
        for d in d_values:
            for k in k_values:
                beta_out = beta_in * r
                T_end = adaptive_T_end(n, d, beta_in, beta_out, L)
                cname = f"r{r:g}_d{d:g}_k{k}"
                cells.append(CellConfig(
                    name=cname, L=L, n=n, k=k, d=d,
                    beta_in=beta_in, beta_out=beta_out, gamma=0.0,
                    T_end=T_end, r=r, n_runs=n_runs,
                    calibration_factors=calibration_factors,
                    base_seed=base + idx * 1000,
                ))
                idx += 1
    return cells


# ---------------------------------------------------------------------------
# Arrival extraction (mirrors CE-06)
# ---------------------------------------------------------------------------

def _arrival_times_micro(result: dict, L: int) -> List[float]:
    out = [float("nan")] * L
    for row in result.get("community_timings", []):
        c = int(row["community"])
        if 0 <= c < L and row.get("first_infection_time") is not None:
            out[c] = float(row["first_infection_time"])
    return out


def _arrival_times_mm(event_log: Sequence[dict], L: int) -> List[float]:
    out = [float("nan")] * L
    for event in sorted(event_log, key=lambda e: float(e.get("time", math.inf))):
        ev = str(event.get("event_type", "")).lower()
        if ev not in {"seed", "infection", "transfer"}:
            continue
        c = event.get("community")
        if c is None:
            continue
        c = int(c)
        if 0 <= c < L and not math.isfinite(out[c]):
            out[c] = float(event["time"])
    return out


# ---------------------------------------------------------------------------
# Single-replication runner
# ---------------------------------------------------------------------------

def _run_one(args: Tuple) -> List[dict]:
    """Run Full Micro once, then MicroMacro with all calibration factors.
    
    Returns rows for arrivals CSV: each row = scenario, replication_id, model,
    calibration_factor, layer, T_i.
    """
    cfg, rep_id = args
    network_seed = cfg.base_seed + rep_id - 1
    run_seed = cfg.base_seed + 50000 + rep_id - 1

    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=cfg.L,
        community_size=cfg.n,
        inter_links=cfg.k,
        seed=network_seed,
        macro_graph_type="chain",
        micro_graph_type="random_connected",
        edge_prob=cfg.edge_prob,
    )
    comm_nodes = [list(g.nodes()) for g in micro_graphs]
    initial_node = (
        int(cfg.initial_node) if cfg.initial_node is not None
        else int(random.Random(run_seed).choice(comm_nodes[cfg.initial_community]))
    )
    model_id = normalize_model(cfg.model)

    rows: List[dict] = []

    # --- Full Micro (baseline) ---
    t0 = time.perf_counter()
    micro_result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=cfg.beta_in,
        recovery_rate=cfg.gamma,
        model=model_id,
    ).run(T_end=cfg.T_end, dt_out=cfg.tau_micro, initial_node=initial_node, rng=random.Random(run_seed))
    micro_rt = time.perf_counter() - t0
    micro_arrivals = _arrival_times_micro(micro_result, cfg.L)

    for layer, t in enumerate(micro_arrivals):
        rows.append({
            "scenario": cfg.name,
            "replication_id": rep_id,
            "model": "Micro",
            "calibration_factor": 1.0,  # nominal
            "layer": layer,
            "T_i": t if math.isfinite(t) else None,
            "runtime": micro_rt,
            "network_seed": network_seed,
        })

    # --- MicroMacro for each calibration factor ---
    for c_factor in cfg.calibration_factors:
        beta_out_eff = cfg.beta_out * c_factor
        t0 = time.perf_counter()
        _, _, _, event_log = MicroMacroSimulator(
            W=W,
            micro_graphs=micro_graphs,
            beta_micro=cfg.beta_in,
            gamma=cfg.gamma,
            beta_macro=beta_out_eff,
            tau_micro=cfg.tau_micro,
            T_end=cfg.T_end,
            macro_T=cfg.macro_T,
            model=model_id,
            full_graph=full_graph,
        ).run(seed=run_seed, initial_community=cfg.initial_community, initial_node=initial_node)
        mm_rt = time.perf_counter() - t0
        mm_arrivals = _arrival_times_mm(event_log, cfg.L)

        for layer, t in enumerate(mm_arrivals):
            rows.append({
                "scenario": cfg.name,
                "replication_id": rep_id,
                "model": "MicroMacro",
                "calibration_factor": c_factor,
                "layer": layer,
                "T_i": t if math.isfinite(t) else None,
                "runtime": mm_rt,
                "network_seed": network_seed,
            })

    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except Exception:
        return float("nan")


def compute_metrics_for_calibration(arrivals_rows: List[dict], T_end: float, 
                                    bootstrap_reps: int = 1000, seed: int = 42) -> dict:
    """Compute calibration metrics: find c* that minimizes |mean(T_MM) - mean(T_Micro)|.
    
    Returns:
        dict with c_star, dRMST_uncal, dRMST_uncal_norm, dRMST_cal, dRMST_cal_norm,
        rel_W1_uncal, rel_W1_cal, structural_fraction, micro_completion, mm_completion_uncal,
        mm_completion_cal.
    """
    import pandas as pd
    df = pd.DataFrame(arrivals_rows)
    rng = np.random.default_rng(seed)
    last_layer = df["layer"].max()
    terminal = df[df["layer"] == last_layer].copy()

    # Extract Full Micro arrivals
    micro_df = terminal[terminal["model"] == "Micro"]
    micro_t = micro_df.set_index("replication_id")["T_i"].to_dict()
    all_ids = sorted(micro_t.keys())
    ref = np.array([micro_t.get(i) for i in all_ids], dtype=float)
    ref_obs = ref[np.isfinite(ref)]
    mean_ref = float(np.mean(ref_obs)) if ref_obs.size > 0 else float("nan")
    micro_completion = float(np.isfinite(ref).mean())

    # For each calibration factor, compute metrics
    calibration_factors = sorted(terminal[terminal["model"] == "MicroMacro"]["calibration_factor"].unique())
    
    results_by_c = {}
    for c in calibration_factors:
        mm_df = terminal[(terminal["model"] == "MicroMacro") & (terminal["calibration_factor"] == c)]
        mm_t = mm_df.set_index("replication_id")["T_i"].to_dict()
        var = np.array([mm_t.get(i) for i in all_ids], dtype=float)
        var_obs = var[np.isfinite(var)]
        
        # Censored RMST
        censored_ref = np.where(np.isfinite(ref), ref, T_end)
        censored_var = np.where(np.isfinite(var), var, T_end)
        delta_rmst = float(np.mean(censored_var - censored_ref))
        
        # Bootstrap CI
        diffs = censored_var - censored_ref
        boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(bootstrap_reps)])
        ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
        
        # Wasserstein
        w1 = _w1(ref_obs, var_obs)
        rel_w1 = w1 / mean_ref if math.isfinite(mean_ref) and mean_ref > 0 else float("nan")
        
        mm_completion = float(np.isfinite(var).mean())
        mean_var = float(np.mean(var_obs)) if var_obs.size else float("nan")
        
        results_by_c[c] = {
            "delta_rmst": delta_rmst,
            "delta_rmst_norm": delta_rmst / T_end if T_end > 0 else float("nan"),
            "rel_w1": rel_w1,
            "w1": w1,
            "mm_completion": mm_completion,
            "mean_var": mean_var,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    
    # Find c* that minimizes |mean(T_MM) - mean(T_Micro)|
    best_c = None
    best_diff = float("inf")
    for c, res in results_by_c.items():
        if math.isfinite(res["mean_var"]) and math.isfinite(mean_ref):
            diff = abs(res["mean_var"] - mean_ref)
            if diff < best_diff:
                best_diff = diff
                best_c = c
    
    # If no valid c* found, use c=1.0 as fallback
    if best_c is None:
        best_c = 1.0
    
    # Uncalibrated (c=1.0)
    uncal = results_by_c.get(1.0, {})
    dRMST_uncal = uncal.get("delta_rmst", float("nan"))
    dRMST_uncal_norm = uncal.get("delta_rmst_norm", float("nan"))
    rel_W1_uncal = uncal.get("rel_w1", float("nan"))
    mm_completion_uncal = uncal.get("mm_completion", float("nan"))
    
    # Calibrated (c*)
    cal = results_by_c.get(best_c, {})
    dRMST_cal = cal.get("delta_rmst", float("nan"))
    dRMST_cal_norm = cal.get("delta_rmst_norm", float("nan"))
    rel_W1_cal = cal.get("rel_w1", float("nan"))
    mm_completion_cal = cal.get("mm_completion", float("nan"))
    
    # Structural fraction
    if math.isfinite(dRMST_uncal) and abs(dRMST_uncal) > 1e-9:
        structural_fraction = abs(dRMST_cal) / abs(dRMST_uncal)
    else:
        structural_fraction = float("nan")
    
    return {
        "c_star": best_c,
        "dRMST_uncal": dRMST_uncal,
        "dRMST_uncal_norm": dRMST_uncal_norm,
        "dRMST_cal": dRMST_cal,
        "dRMST_cal_norm": dRMST_cal_norm,
        "rel_W1_uncal": rel_W1_uncal,
        "rel_W1_cal": rel_W1_cal,
        "structural_fraction": structural_fraction,
        "micro_completion": micro_completion,
        "mm_completion_uncal": mm_completion_uncal,
        "mm_completion_cal": mm_completion_cal,
        "micro_mean_T_last": mean_ref,
        "mm_mean_T_last_uncal": uncal.get("mean_var", float("nan")),
        "mm_mean_T_last_cal": cal.get("mean_var", float("nan")),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _cell_filename(name: str) -> str:
    return name.replace("=", "").replace(".", "p").replace(" ", "_")


def run_experiment(cells: List[CellConfig], out_dir: Path, n_workers: int = 4, resume: bool = False) -> None:
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    manifest = {"grid_cells": len(cells), "started": time.strftime("%Y-%m-%d %H:%M:%S"), "cells": []}

    for ci, cell in enumerate(cells, 1):
        cell_arr_path = cells_dir / f"arrivals_{_cell_filename(cell.name)}.csv"
        if resume and cell_arr_path.exists():
            print(f"[{ci}/{len(cells)}] skip existing {cell.name}", flush=True)
            scenario_rows = pd.read_csv(cell_arr_path).to_dict("records")
        else:
            print(f"\n[{ci}/{len(cells)}] === {cell.name}  (T_end={cell.T_end:g}, n_runs={cell.n_runs}, "
                  f"c_factors={len(cell.calibration_factors)}) ===", flush=True)
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
                    if i % 20 == 0:
                        print(f"  {i}/{cell.n_runs} done", flush=True)
            pd.DataFrame(scenario_rows).to_csv(cell_arr_path, index=False)

        metrics = compute_metrics_for_calibration(scenario_rows, cell.T_end)
        row = {
            "scenario": cell.name, "r": cell.r, "d": cell.d, "k": cell.k,
            "L": cell.L, "n": cell.n, "beta_in": cell.beta_in, "beta_out": cell.beta_out,
            "T_end": cell.T_end, "n_runs": cell.n_runs,
            **metrics,
        }
        summary_rows.append(row)
        manifest["cells"].append({
            "name": cell.name, "n_runs": cell.n_runs,
            "c_star": metrics["c_star"],
            "micro_completion": metrics["micro_completion"],
            "mm_completion_uncal": metrics["mm_completion_uncal"],
            "mm_completion_cal": metrics["mm_completion_cal"],
        })
        print(f"  c*={metrics['c_star']:.2f} | "
              f"dRMST_uncal={metrics['dRMST_uncal']:.1f} (norm={metrics['dRMST_uncal_norm']:.4f}) | "
              f"dRMST_cal={metrics['dRMST_cal']:.1f} (norm={metrics['dRMST_cal_norm']:.4f}) | "
              f"struct_frac={metrics['structural_fraction']:.3f} | "
              f"rel_W1: {metrics['rel_W1_uncal']:.4f} -> {metrics['rel_W1_cal']:.4f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Results saved to {out_dir}", flush=True)
    cols = ["scenario", "r", "d", "k", "c_star", "micro_completion", "mm_completion_uncal", "mm_completion_cal",
            "dRMST_uncal_norm", "dRMST_cal_norm", "structural_fraction", "rel_W1_uncal", "rel_W1_cal"]
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="CE-07 calibration floor: structural vs calibration error")
    parser.add_argument("--grid", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs per cell")
    parser.add_argument("--resume", action="store_true", help="Skip cells with existing arrivals CSV")
    args = parser.parse_args()

    cells = build_grid(args.grid)
    if args.n_runs is not None:
        from dataclasses import replace
        cells = [replace(c, n_runs=args.n_runs) for c in cells]

    out_dir = REPO_ROOT / "output" / "claude_experiments" / f"ce07_calibration_floor_{args.grid}"
    run_experiment(cells, out_dir, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
