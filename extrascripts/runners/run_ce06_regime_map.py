"""
CE-06: Regime map — де саме і наскільки MicroMacro відхиляється від Full Micro
як функція ПАРАМЕТРІВ мережі (не точної структури).

Мета: побудувати датасет (network parameters -> error metrics) для подальшого
аналізу того, які параметри (спектральна щілина lambda2, середній степінь d,
boundary-структура, r = beta_out/beta_in) керують похибкою і як її зменшити.

Обмеження: обидва рівні знають лише ПАРАМЕТРИ мережі (W, розміри, степеневі та
спектральні статистики), не точні звʼязки/позиції вузлів. Дескриптори нижче
записуються лише для ОФЛАЙН-аналізу предикторів, симулятор їх не використовує.

Виправлення конфаунда CE-03: первинні метрики — горизонто-нормалізовані
(rel_W1, delta_RMST / T_end, delta_completion). Rate completion зберігається,
щоб позначати недорозвʼязані комірки.

Запуск (smoke test):
    .venv/bin/python extrascripts/runners/run_ce06_regime_map.py --grid smoke --workers 4

Повний запуск:
    .venv/bin/python extrascripts/runners/run_ce06_regime_map.py --grid full --workers 8

Resume (пропустити вже пораховані комірки):
    .venv/bin/python extrascripts/runners/run_ce06_regime_map.py --grid full --workers 8 --resume
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
    base_seed: int = 600000
    n_runs: int = 60
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
        r_values = [1.0, 2.0]
        d_values = [6.0]
        k_values = [5]
        n_runs = 3
        base = 600000
    elif name == "pilot":
        r_values = [0.5, 1.0, 2.0]
        d_values = [4.0, 6.0, 12.0]
        k_values = [5]
        n_runs = 30
        base = 610000
    elif name == "full":
        # 3 x 4 x 3 = 36 cells around the CE-03 sign-change region
        r_values = [0.5, 1.0, 2.0]
        d_values = [4.0, 6.0, 10.0, 20.0]
        k_values = [2, 5, 10]
        n_runs = 60
        base = 620000
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
                    T_end=T_end, r=r, n_runs=n_runs, base_seed=base + idx * 1000,
                ))
                idx += 1
    return cells


# ---------------------------------------------------------------------------
# Network descriptors (predictors for offline analysis only)
# ---------------------------------------------------------------------------

def _normalized_laplacian_eigs(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1)
    with np.errstate(divide="ignore"):
        dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    n = adj.shape[0]
    lap = np.eye(n) - (dinv_sqrt[:, None] * adj * dinv_sqrt[None, :])
    eig = np.linalg.eigvalsh((lap + lap.T) * 0.5)
    return np.sort(eig)


def compute_descriptors(micro_graphs, full_graph, comm_nodes, beta_in: float) -> Dict[str, float]:
    """Aggregate structural predictors across communities for one realization.

    All quantities are network-level statistics (means over communities), the
    kind a macro model could in principle estimate from parameters — not
    node-level identities. Recorded only for offline predictor analysis.
    """
    import networkx as nx

    lam2_norm: List[float] = []       # algebraic connectivity (2nd smallest of L_norm)
    lam2_tilde: List[float] = []      # 2nd largest of L_norm (legacy alpha convention; not used for the correction)
    spec_radius: List[float] = []     # adjacency spectral radius (lambda1)
    mean_deg: List[float] = []
    tau_find: List[float] = []

    for g in micro_graphs:
        nodes = list(g.nodes())
        A = nx.to_numpy_array(g, nodelist=nodes, weight="weight", dtype=float)
        nn = A.shape[0]
        if nn >= 2:
            eig = _normalized_laplacian_eigs(A)
            lam2_norm.append(float(eig[1]))
            lam2_tilde.append(float(eig[-2]))
        m = g.number_of_edges()
        md = 2.0 * m / nn if nn > 0 else 0.0
        mean_deg.append(md)
        # adjacency spectral radius via eigen (communities are small)
        sr = float(np.max(np.abs(np.linalg.eigvalsh((A + A.T) * 0.5)))) if nn > 0 else 0.0
        spec_radius.append(sr)
        if beta_in > 0 and sr > 1.0 and nn >= 2:
            tau_find.append(math.log(nn) / (beta_in * (sr - 1.0)))
        else:
            tau_find.append(float("nan"))

    # Boundary structure: nodes with cross-community edges, and their
    # within-community degree (controls how fast a bridge node gets infected).
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

    b_counts: List[int] = []
    b_within_deg: List[float] = []
    for c, g in enumerate(micro_graphs):
        bset = boundary_nodes[c]
        b_counts.append(len(bset))
        for u in bset:
            if g.has_node(u):
                b_within_deg.append(float(g.degree(u)))

    def _nanmean(xs):
        arr = np.array([x for x in xs if x is not None], dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    _lam2n = _nanmean(lam2_norm)
    return {
        "lambda2_norm": _lam2n,
        "lambda2_tilde": _nanmean(lam2_tilde),
        # alpha applied by the corrected spectral macro hazard: damping-only
        # exponent alpha = clamp(1/mu2, 1, 6) with mu2 = Fiedler value (lam2_norm).
        "alpha_spectral": (max(1.0, min(1.0 / _lam2n, 6.0)) if _lam2n > 0 else float("nan")),
        "spectral_radius": _nanmean(spec_radius),
        "mean_degree": _nanmean(mean_deg),
        "tau_find": _nanmean(tau_find),
        "boundary_count_mean": float(np.mean(b_counts)) if b_counts else float("nan"),
        "boundary_within_deg_mean": _nanmean(b_within_deg),
    }


# ---------------------------------------------------------------------------
# Arrival extraction (mirrors run_ce03_ce04.py)
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

    desc = compute_descriptors(micro_graphs, full_graph, comm_nodes, cfg.beta_in)

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
        rows.append({"scenario": cfg.name, "replication_id": rep_id, "model": "Micro",
                     "layer": layer, "T_i": t if math.isfinite(t) else None,
                     "runtime": micro_rt, "network_seed": network_seed, **desc})

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
        rows.append({"scenario": cfg.name, "replication_id": rep_id, "model": "MicroMacro",
                     "layer": layer, "T_i": t if math.isfinite(t) else None,
                     "runtime": mm_rt, "network_seed": network_seed, **desc})

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


def compute_metrics(arrivals_rows: List[dict], T_end: float, bootstrap_reps: int = 1000, seed: int = 42) -> dict:
    import pandas as pd
    df = pd.DataFrame(arrivals_rows)
    rng = np.random.default_rng(seed)
    last_layer = df["layer"].max()
    terminal = df[df["layer"] == last_layer].copy()

    micro_t = terminal[terminal["model"] == "Micro"].set_index("replication_id")["T_i"].to_dict()
    mm_t = terminal[terminal["model"] == "MicroMacro"].set_index("replication_id")["T_i"].to_dict()

    all_ids = sorted(set(micro_t) | set(mm_t))
    ref = np.array([micro_t.get(i) for i in all_ids], dtype=float)
    var = np.array([mm_t.get(i) for i in all_ids], dtype=float)

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
    rel_w1 = w1 / mean_ref if math.isfinite(mean_ref) and mean_ref > 0 else float("nan")

    micro_completion = float(np.isfinite(ref).mean())
    mm_completion = float(np.isfinite(var).mean())

    return {
        "micro_completion": micro_completion,
        "mm_completion": mm_completion,
        "delta_completion": mm_completion - micro_completion,
        "micro_mean_T_last": mean_ref,
        "mm_mean_T_last": float(np.mean(var_obs)) if var_obs.size else float("nan"),
        "W1": w1,
        "rel_W1": rel_w1,
        "delta_RMST": delta_rmst,
        "delta_RMST_norm": delta_rmst / T_end if T_end > 0 else float("nan"),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def _mean_descriptors(rows: List[dict]) -> Dict[str, float]:
    keys = ["lambda2_norm", "lambda2_tilde", "alpha_spectral", "spectral_radius",
            "mean_degree", "tau_find", "boundary_count_mean", "boundary_within_deg_mean"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([r.get(k, float("nan")) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[k] = float(vals.mean()) if vals.size else float("nan")
    return out


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
        desc = _mean_descriptors(scenario_rows)
        row = {
            "scenario": cell.name, "r": cell.r, "d": cell.d, "k": cell.k,
            "L": cell.L, "n": cell.n, "beta_in": cell.beta_in, "beta_out": cell.beta_out,
            "T_end": cell.T_end, "n_runs": cell.n_runs,
            **metrics, **desc,
        }
        summary_rows.append(row)
        manifest["cells"].append({"name": cell.name, "n_runs": cell.n_runs,
                                  "micro_completion": metrics["micro_completion"],
                                  "mm_completion": metrics["mm_completion"]})
        print(f"  dRMST={metrics['delta_RMST']:.1f} (norm={metrics['delta_RMST_norm']:.4f}) "
              f"rel_W1={metrics['rel_W1']:.4f} d_compl={metrics['delta_completion']:+.3f} "
              f"lam2={desc['lambda2_norm']:.4f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Results saved to {out_dir}", flush=True)
    cols = ["scenario", "r", "d", "k", "micro_completion", "mm_completion",
            "delta_RMST_norm", "rel_W1", "delta_completion", "lambda2_norm"]
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="CE-06 regime map: parameter -> error dataset")
    parser.add_argument("--grid", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs per cell")
    parser.add_argument("--resume", action="store_true", help="Skip cells with existing arrivals CSV")
    args = parser.parse_args()

    cells = build_grid(args.grid)
    if args.n_runs is not None:
        from dataclasses import replace
        cells = [replace(c, n_runs=args.n_runs) for c in cells]

    out_dir = REPO_ROOT / "output" / "claude_experiments" / f"ce06_regime_map_{args.grid}"
    run_experiment(cells, out_dir, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
