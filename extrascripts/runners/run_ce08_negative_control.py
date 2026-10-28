"""
CE-08: Negative control — clique–bridge–leaf з повними метриками.

Мета: quantify ΔRMST, W1, KS, Δp з CI у мінімальній confounder-free топології
(короткий chain of cliques, k=2, perfect mixing всередині) — де теорія передбачає
малу/відсутню помилку. Це falsifiability baseline для paper §3.4.

Топологія:
- L=3 (короткий ланцюг для мінімальної акумуляції)
- n=100 (стандартний розмір спільноти)
- k=2 (мінімальний bridge — найменший можливий boundary)
- micro_graph_type="complete" (кожна спільнота = клика K_100)
- beta_in=0.003, r=1.0 (beta_out=0.003) — збалансований режим
- gamma=0 (SI через SIR model)

Метрики (розширення CE-06):
- ΔRMST + bootstrap 95% CI, ΔRMST_norm
- W1 (на завершених), rel_W1, W1/IQR(Micro)
- KS statistic + p-value (scipy.stats.ks_2samp)
- Δp (completion difference)
- Paired seeds (network_seed, run_seed) як у CE-06

Запуск (smoke test):
    .venv/bin/python extrascripts/runners/run_ce08_negative_control.py --grid smoke --workers 2

Повний запуск:
    .venv/bin/python extrascripts/runners/run_ce08_negative_control.py --grid full --workers 8

Resume:
    .venv/bin/python extrascripts/runners/run_ce08_negative_control.py --grid full --workers 8 --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import iqr, ks_2samp, wasserstein_distance

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
    beta_in: float
    beta_out: float
    gamma: float
    T_end: float
    r: float
    model: str = "SIR"  # gamma=0 realizes SI dynamics
    tau_micro: float = 1.0
    macro_T: float = 1.0
    base_seed: int = 800000
    n_runs: int = 200
    initial_community: int = 0
    initial_node: Optional[int] = 0


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def build_grid(name: str) -> List[CellConfig]:
    """
    Negative control: clique-bridge-leaf — perfect mixing (complete graphs),
    minimal boundary (k=2), balanced rates (r=1.0), short chain (L=3).
    
    Error should be small/absent here. We need n_runs=200 to bound the
    near-null effect with sufficient power.
    """
    beta_in = 0.003
    r = 1.0
    beta_out = beta_in * r
    L = 3
    n = 100
    k = 2
    gamma = 0.0  # SI dynamics

    # Rough T_end: within-clique find is very fast (complete graph),
    # crossing time ~ 1/beta_out. For L=3, 2 hops, T ~ 1200.
    # Add safety margin.
    T_end = 1500.0

    if name == "smoke":
        n_runs = 3
        base = 800000
    elif name == "full":
        n_runs = 200  # Large N to detect/bound near-null effects
        base = 800000
    else:
        raise ValueError(f"unknown grid: {name}")

    cells: List[CellConfig] = [
        CellConfig(
            name=f"clique_L{L}_n{n}_k{k}_r{r:g}",
            L=L, n=n, k=k,
            beta_in=beta_in, beta_out=beta_out, gamma=gamma,
            T_end=T_end, r=r, n_runs=n_runs, base_seed=base,
        )
    ]
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
    cfg, rep_id = args
    network_seed = cfg.base_seed + rep_id - 1
    run_seed = cfg.base_seed + 50000 + rep_id - 1

    # Generate clique-bridge network: complete graphs, chain, k=2
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=cfg.L,
        community_size=cfg.n,
        inter_links=cfg.k,
        seed=network_seed,
        macro_graph_type="chain",
        micro_graph_type="complete",  # Perfect mixing: K_n cliques
    )
    comm_nodes = [list(g.nodes()) for g in micro_graphs]
    initial_node = (
        int(cfg.initial_node) if cfg.initial_node is not None
        else int(random.Random(run_seed).choice(comm_nodes[cfg.initial_community]))
    )
    model_id = normalize_model(cfg.model)

    rows: List[dict] = []

    # --- Full Micro ---
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
            "scenario": cfg.name, "replication_id": rep_id, "model": "Micro",
            "layer": layer, "T_i": t if math.isfinite(t) else None,
            "runtime": micro_rt, "network_seed": network_seed,
        })

    # --- MicroMacro ---
    t0 = time.perf_counter()
    _, _, _, event_log = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=cfg.beta_in,
        gamma=cfg.gamma,
        beta_macro=cfg.beta_out,
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
            "scenario": cfg.name, "replication_id": rep_id, "model": "MicroMacro",
            "layer": layer, "T_i": t if math.isfinite(t) else None,
            "runtime": mm_rt, "network_seed": network_seed,
        })

    return rows


# ---------------------------------------------------------------------------
# Statistics (extended from CE-06 with KS test)
# ---------------------------------------------------------------------------

def _w1(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        return float(wasserstein_distance(a, b))
    except Exception:
        return float("nan")


def compute_metrics(arrivals_rows: List[dict], T_end: float, bootstrap_reps: int = 1000, seed: int = 42) -> dict:
    import pandas as pd
    from scipy.stats import iqr, ks_2samp
    df = pd.DataFrame(arrivals_rows)
    rng = np.random.default_rng(seed)
    last_layer = df["layer"].max()
    terminal = df[df["layer"] == last_layer].copy()

    micro_t = terminal[terminal["model"] == "Micro"].set_index("replication_id")["T_i"].to_dict()
    mm_t = terminal[terminal["model"] == "MicroMacro"].set_index("replication_id")["T_i"].to_dict()

    all_ids = sorted(set(micro_t) | set(mm_t))
    ref = np.array([micro_t.get(i) for i in all_ids], dtype=float)
    var = np.array([mm_t.get(i) for i in all_ids], dtype=float)

    # Completion rates
    micro_completion = float(np.isfinite(ref).mean())
    mm_completion = float(np.isfinite(var).mean())

    # Completed observations
    ref_obs = ref[np.isfinite(ref)]
    var_obs = var[np.isfinite(var)]

    # RMST with censoring at T_end
    censored_ref = np.where(np.isfinite(ref), ref, T_end)
    censored_var = np.where(np.isfinite(var), var, T_end)
    delta_rmst = float(np.mean(censored_var - censored_ref))

    # Bootstrap CI for ΔRMST
    diffs = censored_var - censored_ref
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(bootstrap_reps)])
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])

    # W1 and relative W1
    w1 = _w1(ref_obs, var_obs)
    mean_ref = float(np.mean(ref_obs)) if ref_obs.size > 0 else float("nan")
    rel_w1 = w1 / mean_ref if math.isfinite(mean_ref) and mean_ref > 0 else float("nan")

    # W1 normalized by IQR (alternative scale normalization)
    iqr_ref = float(iqr(ref_obs)) if ref_obs.size >= 2 else float("nan")
    w1_over_iqr = w1 / iqr_ref if math.isfinite(iqr_ref) and iqr_ref > 0 else float("nan")

    # KS test (only on completed arrivals)
    if ref_obs.size >= 2 and var_obs.size >= 2:
        ks_stat, ks_pval = ks_2samp(ref_obs, var_obs)
    else:
        ks_stat, ks_pval = float("nan"), float("nan")

    return {
        "p_Micro": micro_completion,
        "p_MM": mm_completion,
        "delta_p": mm_completion - micro_completion,
        "micro_mean_T_last": mean_ref,
        "mm_mean_T_last": float(np.mean(var_obs)) if var_obs.size else float("nan"),
        "W1": w1,
        "rel_W1": rel_w1,
        "W1_over_IQR": w1_over_iqr,
        "delta_RMST": delta_rmst,
        "delta_RMST_norm": delta_rmst / T_end if T_end > 0 else float("nan"),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "KS_statistic": float(ks_stat),
        "KS_pvalue": float(ks_pval),
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
                    if i % 20 == 0:
                        print(f"  {i}/{cell.n_runs} done", flush=True)
            pd.DataFrame(scenario_rows).to_csv(cell_arr_path, index=False)

        metrics = compute_metrics(scenario_rows, cell.T_end)
        row = {
            "scenario": cell.name,
            "r": cell.r,
            "k": cell.k,
            "L": cell.L,
            "n": cell.n,
            "beta_in": cell.beta_in,
            "beta_out": cell.beta_out,
            "T_end": cell.T_end,
            "n_runs": cell.n_runs,
            **metrics,
        }
        summary_rows.append(row)
        manifest["cells"].append({
            "name": cell.name,
            "n_runs": cell.n_runs,
            "p_Micro": metrics["p_Micro"],
            "p_MM": metrics["p_MM"],
        })
        print(f"  ΔRMST={metrics['delta_RMST']:.2f} (norm={metrics['delta_RMST_norm']:.5f}) "
              f"W1={metrics['W1']:.2f} rel_W1={metrics['rel_W1']:.5f} "
              f"KS={metrics['KS_statistic']:.4f} (p={metrics['KS_pvalue']:.4f}) "
              f"Δp={metrics['delta_p']:+.3f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Results saved to {out_dir}", flush=True)
    cols = ["scenario", "r", "k", "L", "p_Micro", "p_MM",
            "delta_RMST_norm", "rel_W1", "KS_statistic", "KS_pvalue", "delta_p"]
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="CE-08 negative control: clique-bridge-leaf quantitative baseline")
    parser.add_argument("--grid", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs")
    parser.add_argument("--resume", action="store_true", help="Skip existing cells")
    args = parser.parse_args()

    cells = build_grid(args.grid)
    if args.n_runs is not None:
        from dataclasses import replace
        cells = [replace(c, n_runs=args.n_runs) for c in cells]

    out_dir = REPO_ROOT / "output" / "claude_experiments" / f"ce08_negative_control_{args.grid}"
    run_experiment(cells, out_dir, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
