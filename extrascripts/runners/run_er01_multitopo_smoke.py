#!/usr/bin/env python3
"""ER-01: Multi-topology smoke validation of parameters-only error-reduction corrections.

Validates candidate C2 (boundary_phi) and C12 (renorm_phi) across a topology matrix:
- macro_graph_type in {chain, complete, star}
- micro_graph_type in {complete, random_connected, clique_leaves}

For each topology cell, runs Full Micro once + MicroMacro in THREE correction modes
on PAIRED seeds:
  * correction_mode="off"           -> baseline (no correction)
  * correction_mode="boundary_phi"  -> C2 boundary-availability multiplier
  * correction_mode="renorm_phi"    -> C12 renormalized rate phi = mu2/(mu2+r)

Computes per-topology gap_closure = (|off| - |corrected|) / |off| to assess
worst-case (minimax) generalization.

Usage:
  python extrascripts/runners/run_er01_multitopo_smoke.py --grid smoke --workers 2 [--resume]
Output: output/error_reduction_research/er01_multitopo_smoke_{grid}/
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CE06_PATH = ROOT / "extrascripts" / "run_ce06_regime_map.py"

# Import CE-06 helpers (reuse network builder, metrics, descriptors)
_spec = importlib.util.spec_from_file_location("run_ce06_regime_map", CE06_PATH)
ce06 = importlib.util.module_from_spec(_spec)
sys.modules["run_ce06_regime_map"] = ce06
_spec.loader.exec_module(ce06)  # type: ignore[union-attr]

generate_two_scale_network = ce06.generate_two_scale_network
normalize_model = ce06.normalize_model
MicroSimulator = ce06.MicroSimulator
MicroMacroSimulator = ce06.MicroMacroSimulator
adaptive_T_end = ce06.adaptive_T_end
compute_descriptors = ce06.compute_descriptors
compute_metrics = ce06.compute_metrics
_mean_descriptors = ce06._mean_descriptors
_arrival_times_micro = ce06._arrival_times_micro
_arrival_times_mm = ce06._arrival_times_mm


# ---------------------------------------------------------------------------
# Topology Cell Configuration
# ---------------------------------------------------------------------------

@dataclass
class TopoCell:
    name: str
    macro_graph_type: str  # chain, complete, star
    micro_graph_type: str  # complete, random_connected, clique_leaves
    L: int
    n: int
    k: int
    d: float
    beta_in: float
    beta_out: float
    gamma: float
    T_end: float
    r: float
    n_runs: int
    base_seed: int
    tau_micro: float = 0.1
    macro_T: float = 1.0
    initial_community: int = 0
    initial_node: Optional[int] = None
    model: str = "SIR"
    edge_prob: float = 0.0
    leaf_count: int = 0
    leaf_degree: int = 0


def build_grid(name: str) -> List[TopoCell]:
    """Build topology matrix: macro x micro types + a few (r, d) points."""
    beta_in = 0.003
    L, n = 10, 100
    
    if name == "smoke":
        # Minimal smoke: 2 r, 1 d, subset of topologies, few runs
        r_values = [1.0, 2.0]
        d_values = [6.0]
        k = 5
        n_runs = 8
        base = 900000
    elif name == "pilot":
        r_values = [0.5, 1.0, 2.0]
        d_values = [4.0, 6.0, 12.0]
        k = 5
        n_runs = 20
        base = 910000
    else:
        raise ValueError(f"unknown grid: {name}")
    
    # Topology matrix
    macro_types = ["chain", "complete"]  # skip star for smoke (single-node leaves complicate)
    micro_types = ["complete", "random_connected", "clique_leaves"]
    
    cells: List[TopoCell] = []
    idx = 0
    for macro_type in macro_types:
        for micro_type in micro_types:
            # Skip invalid combos
            if macro_type == "star" and micro_type == "clique_leaves":
                # star forces single-node leaves -> clique_leaves doesn't apply cleanly
                continue
            
            for r in r_values:
                for d in d_values:
                    beta_out = beta_in * r
                    T_end = adaptive_T_end(n, d, beta_in, beta_out, L)
                    
                    # Compute edge_prob for random_connected/clique_leaves
                    if micro_type in ("random_connected", "complete"):
                        edge_prob = d / (n - 1) if n > 1 else 0.1
                    else:
                        edge_prob = 0.1
                    
                    # For clique_leaves: clique of size n, leaves = 0 (just the clique for smoke)
                    leaf_count = 0
                    leaf_degree = 0
                    
                    cname = f"{macro_type}_{micro_type}_r{r:g}_d{d:g}"
                    cells.append(TopoCell(
                        name=cname,
                        macro_graph_type=macro_type,
                        micro_graph_type=micro_type,
                        L=L, n=n, k=k, d=d,
                        beta_in=beta_in, beta_out=beta_out, gamma=0.0,
                        T_end=T_end, r=r, n_runs=n_runs, base_seed=base + idx * 1000,
                        edge_prob=edge_prob,
                        leaf_count=leaf_count,
                        leaf_degree=leaf_degree,
                    ))
                    idx += 1
    return cells


# ---------------------------------------------------------------------------
# Single-replication runner: Micro + MicroMacro (off, boundary_phi, renorm_phi)
# ---------------------------------------------------------------------------

def _run_one(args: Tuple) -> List[dict]:
    cfg, rep_id = args
    network_seed = cfg.base_seed + rep_id - 1
    run_seed = cfg.base_seed + 50000 + rep_id - 1
    
    # Generate network
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=cfg.L,
        community_size=cfg.n,
        inter_links=cfg.k,
        seed=network_seed,
        macro_graph_type=cfg.macro_graph_type,
        micro_graph_type=cfg.micro_graph_type,
        edge_prob=cfg.edge_prob,
        leaf_count=cfg.leaf_count,
        leaf_degree=cfg.leaf_degree,
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
    
    # --- MicroMacro in three correction modes on identical paired seeds ---
    for variant in ("off", "boundary_phi", "renorm_phi"):
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
            correction_mode=variant,
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
    """Run compute_metrics on Micro rows + MicroMacro rows of one correction variant."""
    subset = [r for r in scenario_rows
              if r["model"] == "Micro" or r.get("variant") == variant]
    return compute_metrics(subset, T_end)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _cell_filename(name: str) -> str:
    return name.replace("=", "").replace(".", "p").replace(" ", "_")


def run_experiment(cells: List[TopoCell], out_dir: Path, n_workers: int = 2, resume: bool = False) -> None:
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
                    if i % 10 == 0:
                        print(f"  {i}/{cell.n_runs} done", flush=True)
            pd.DataFrame(scenario_rows).to_csv(cell_arr_path, index=False)
        
        # Compute metrics for each variant
        metrics_off = _metrics_for_variant(scenario_rows, "off", cell.T_end)
        metrics_boundary = _metrics_for_variant(scenario_rows, "boundary_phi", cell.T_end)
        metrics_renorm = _metrics_for_variant(scenario_rows, "renorm_phi", cell.T_end)
        desc = _mean_descriptors(scenario_rows)
        
        # Compute gap_closure for each correction
        drmst_off = abs(metrics_off["delta_RMST_norm"])
        gap_closure_boundary = (drmst_off - abs(metrics_boundary["delta_RMST_norm"])) / drmst_off if drmst_off > 0 else 0.0
        gap_closure_renorm = (drmst_off - abs(metrics_renorm["delta_RMST_norm"])) / drmst_off if drmst_off > 0 else 0.0
        
        row = {
            "scenario": cell.name,
            "macro_type": cell.macro_graph_type,
            "micro_type": cell.micro_graph_type,
            "r": cell.r, "d": cell.d, "k": cell.k,
            "L": cell.L, "n": cell.n,
            "beta_in": cell.beta_in, "beta_out": cell.beta_out,
            "T_end": cell.T_end, "n_runs": cell.n_runs,
            # Baseline (off)
            "off_dRMST_norm": metrics_off["delta_RMST_norm"],
            "off_rel_W1": metrics_off["rel_W1"],
            "off_delta_completion": metrics_off["delta_completion"],
            # C2 (boundary_phi)
            "boundary_phi_dRMST_norm": metrics_boundary["delta_RMST_norm"],
            "boundary_phi_rel_W1": metrics_boundary["rel_W1"],
            "boundary_phi_gap_closure": gap_closure_boundary,
            # C12 (renorm_phi)
            "renorm_phi_dRMST_norm": metrics_renorm["delta_RMST_norm"],
            "renorm_phi_rel_W1": metrics_renorm["rel_W1"],
            "renorm_phi_gap_closure": gap_closure_renorm,
            **desc,
        }
        summary_rows.append(row)
        manifest["cells"].append({"name": cell.name, "n_runs": cell.n_runs})
        print(f"  off: dRMST_norm={metrics_off['delta_RMST_norm']:.4f} rel_W1={metrics_off['rel_W1']:.4f}", flush=True)
        print(f"  boundary_phi: dRMST_norm={metrics_boundary['delta_RMST_norm']:.4f} gap_closure={gap_closure_boundary:+.3f}", flush=True)
        print(f"  renorm_phi: dRMST_norm={metrics_renorm['delta_RMST_norm']:.4f} gap_closure={gap_closure_renorm:+.3f}", flush=True)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps([asdict(c) for c in cells], indent=2), encoding="utf-8")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    print(f"\nDone. Results saved to {out_dir}", flush=True)
    cols = ["scenario", "macro_type", "micro_type", "r", "d",
            "off_dRMST_norm", "boundary_phi_gap_closure", "renorm_phi_gap_closure",
            "lambda2_norm"]
    print(summary_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="ER-01: Multi-topology smoke validation of error corrections")
    parser.add_argument("--grid", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Skip cells with existing arrivals CSV")
    args = parser.parse_args()
    
    cells = build_grid(args.grid)
    out_dir = ROOT / "output" / "error_reduction_research" / f"er01_multitopo_smoke_{args.grid}"
    run_experiment(cells, out_dir, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
