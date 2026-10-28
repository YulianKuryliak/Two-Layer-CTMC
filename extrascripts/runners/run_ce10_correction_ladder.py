#!/usr/bin/env python3
"""CE-10: Correction ladder — activate the parameters-only spectral alpha.

For each scenario we run Full Micro once (ground truth) and MicroMacro TWICE on
identical PAIRED seeds:
  * alpha_mode="off"       -> legacy hazard, infectivity = I/N            (baseline)
  * alpha_mode="spectral"  -> infectivity = (I/N) ** alpha_i, alpha_i = clamp(1/mu2, 1, 6)

alpha_i is a PARAMETERS-ONLY quantity (Fiedler value = 2nd-smallest eigenvalue of each community's
normalized Laplacian) — it uses NO exact cross-community structure. We measure how
much of the |ΔRMST| gap the correction closes:

    gap_closure = (|ΔRMST_off| - |ΔRMST_spectral|) / |ΔRMST_off|

Positive gap_closure => the spectral correction moves MicroMacro toward Full Micro.

Reuses the CE-06 runner helpers (CellConfig, adaptive_T_end, descriptors, arrival
extraction, compute_metrics). No changes to two_layer_ctmc/ beyond the opt-in
alpha_mode already plumbed into MacroEngine/Orchestrator/MicroMacroSimulator.

Usage:
  python extrascripts/runners/run_ce10_correction_ladder.py --grid smoke|pilot|full --workers N [--resume]
Output: output/claude_experiments/ce10_correction_ladder_{grid}/
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CE06_PATH = ROOT / "extrascripts" / "run_ce06_regime_map.py"

# Import the CE-06 runner module by file path (extrascripts is not a package).
# Register in sys.modules BEFORE exec so its @dataclass definitions resolve
# cls.__module__ (dataclasses looks the module up in sys.modules).
_spec = importlib.util.spec_from_file_location("run_ce06_regime_map", CE06_PATH)
ce06 = importlib.util.module_from_spec(_spec)
sys.modules["run_ce06_regime_map"] = ce06
_spec.loader.exec_module(ce06)  # type: ignore[union-attr]

CellConfig = ce06.CellConfig
adaptive_T_end = ce06.adaptive_T_end
compute_descriptors = ce06.compute_descriptors
compute_metrics = ce06.compute_metrics
_mean_descriptors = ce06._mean_descriptors
_cell_filename = ce06._cell_filename
_arrival_times_micro = ce06._arrival_times_micro
_arrival_times_mm = ce06._arrival_times_mm
generate_two_scale_network = ce06.generate_two_scale_network
normalize_model = ce06.normalize_model
MicroSimulator = ce06.MicroSimulator
MicroMacroSimulator = ce06.MicroMacroSimulator


# ---------------------------------------------------------------------------
# Grid — focus on the premature-arrival regime where the correction should help
# ---------------------------------------------------------------------------

def build_grid(name: str) -> List[CellConfig]:
    beta_in = 0.003
    L, n = 10, 100

    if name == "smoke":
        r_values = [2.0]
        d_values = [4.0]
        k_values = [5]
        n_runs = 3
        base = 800000
    elif name == "pilot":
        r_values = [1.0, 2.0]
        d_values = [4.0, 6.0, 12.0]
        k_values = [5]
        n_runs = 30
        base = 810000
    elif name == "full":
        r_values = [0.5, 1.0, 2.0]
        d_values = [4.0, 6.0, 10.0, 20.0]
        k_values = [5]
        n_runs = 60
        base = 820000
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
# Single-replication runner: Full Micro once + MicroMacro (off) + MicroMacro (spectral)
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

    # --- Full Micro (shared ground truth) ---
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
                     "variant": "both", "layer": layer,
                     "T_i": t if math.isfinite(t) else None,
                     "runtime": micro_rt, "network_seed": network_seed, **desc})

    # --- MicroMacro in both alpha modes on identical paired seeds ---
    for variant in ("off", "spectral"):
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
            alpha_mode=variant,
        ).run(seed=run_seed, initial_community=cfg.initial_community, initial_node=initial_node)
        mm_rt = time.perf_counter() - t0
        mm_arrivals = _arrival_times_mm(event_log, cfg.L)
        for layer, t in enumerate(mm_arrivals):
            rows.append({"scenario": cfg.name, "replication_id": rep_id, "model": "MicroMacro",
                         "variant": variant, "layer": layer,
                         "T_i": t if math.isfinite(t) else None,
                         "runtime": mm_rt, "network_seed": network_seed, **desc})

    return rows


def _metrics_for_variant(scenario_rows: List[dict], variant: str, T_end: float) -> dict:
    """Run compute_metrics on Micro rows + MicroMacro rows of one alpha variant."""
    subset = [r for r in scenario_rows
              if r["model"] == "Micro" or r.get("variant") == variant]
    return compute_metrics(subset, T_end)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(cells: List[CellConfig], out_dir: Path, n_workers: int = 4, resume: bool = False) -> None:
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[dict] = []
    manifest = {"grid_cells": len(cells), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "correction": "spectral alpha_i = clamp(1/mu2, 1, 6), mu2=Fiedler (parameters-only)",
                "cells": []}

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

        m_off = _metrics_for_variant(scenario_rows, "off", cell.T_end)
        m_spec = _metrics_for_variant(scenario_rows, "spectral", cell.T_end)
        desc = _mean_descriptors(scenario_rows)

        d_off = m_off["delta_RMST"]
        d_spec = m_spec["delta_RMST"]
        gap_closure = ((abs(d_off) - abs(d_spec)) / abs(d_off)) if abs(d_off) > 0 else float("nan")

        row = {
            "scenario": cell.name, "r": cell.r, "d": cell.d, "k": cell.k,
            "L": cell.L, "n": cell.n, "beta_in": cell.beta_in, "beta_out": cell.beta_out,
            "T_end": cell.T_end, "n_runs": cell.n_runs,
            "delta_RMST_off": d_off, "delta_RMST_off_norm": m_off["delta_RMST_norm"],
            "ci_low_off": m_off["ci_low"], "ci_high_off": m_off["ci_high"],
            "rel_W1_off": m_off["rel_W1"], "mm_completion_off": m_off["mm_completion"],
            "delta_RMST_spectral": d_spec, "delta_RMST_spectral_norm": m_spec["delta_RMST_norm"],
            "ci_low_spectral": m_spec["ci_low"], "ci_high_spectral": m_spec["ci_high"],
            "rel_W1_spectral": m_spec["rel_W1"], "mm_completion_spectral": m_spec["mm_completion"],
            "micro_completion": m_off["micro_completion"],
            "gap_closure": gap_closure,
            "alpha_spectral_mean": desc.get("alpha_spectral", float("nan")),
            **desc,
        }
        summary_rows.append(row)
        manifest["cells"].append({"name": cell.name, "n_runs": cell.n_runs,
                                  "gap_closure": gap_closure})
        print(f"  dRMST off={d_off:.1f} (norm={m_off['delta_RMST_norm']:.4f}) | "
              f"spectral={d_spec:.1f} (norm={m_spec['delta_RMST_norm']:.4f}) | "
              f"gap_closure={gap_closure:+.3f} | alpha_mean={desc.get('alpha_spectral', float('nan')):.3f}",
              flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nDone. Results saved to {out_dir}", flush=True)
    cols = ["scenario", "r", "d", "k", "delta_RMST_off_norm", "delta_RMST_spectral_norm",
            "gap_closure", "rel_W1_off", "rel_W1_spectral", "alpha_spectral_mean"]
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="CE-10 correction ladder (spectral alpha)")
    ap.add_argument("--grid", default="smoke", choices=["smoke", "pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-runs", type=int, default=None, help="override per-cell n_runs")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cells = build_grid(args.grid)
    if args.n_runs is not None:
        cells = [CellConfig(**{**asdict(c), "n_runs": args.n_runs}) for c in cells]

    out_dir = ROOT / "output" / "claude_experiments" / f"ce10_correction_ladder_{args.grid}"
    run_experiment(cells, out_dir, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
