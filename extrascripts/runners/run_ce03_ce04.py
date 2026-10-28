"""
CE-03: Beta-ratio sweep — як відношення beta_out / beta_in впливає на похибку MicroMacro.
CE-04: SIR validation — як gamma змінює величину і знак зміщення.

Запуск:
    .venv/bin/python extrascripts/runners/run_ce03_ce04.py --experiment ce03 --workers 4
    .venv/bin/python extrascripts/runners/run_ce03_ce04.py --experiment ce04 --workers 4
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulate import normalize_model
from two_layer_ctmc.simulators import MicroMacroSimulator, MicroSimulator


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    L: int
    n: int
    k: int
    beta_in: float
    beta_out: float
    gamma: float
    T_end: float
    d: float = 6.0
    model: str = "SIR"
    tau_micro: float = 1.0
    macro_T: float = 1.0
    base_seed: int = 380000
    n_runs: int = 100
    initial_community: int = 0
    initial_node: Optional[int] = 0

    @property
    def edge_prob(self) -> float:
        return self.d / (self.n - 1)


# CE-03: vary beta_out/beta_in at fixed topology
CE03_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(name="r=0.1",  L=10, n=100, k=5, beta_in=0.003, beta_out=0.0003, gamma=0.0, T_end=20000.0, base_seed=380010),
    ScenarioConfig(name="r=0.3",  L=10, n=100, k=5, beta_in=0.003, beta_out=0.0009, gamma=0.0, T_end=10000.0, base_seed=380020),
    ScenarioConfig(name="r=1.0",  L=10, n=100, k=5, beta_in=0.003, beta_out=0.003,  gamma=0.0, T_end=4000.0,  base_seed=380030),
    ScenarioConfig(name="r=2.0",  L=10, n=100, k=5, beta_in=0.003, beta_out=0.006,  gamma=0.0, T_end=3000.0,  base_seed=380040),
    ScenarioConfig(name="r=5.0",  L=10, n=100, k=5, beta_in=0.003, beta_out=0.015,  gamma=0.0, T_end=2000.0,  base_seed=380050),
]

# CE-04: vary gamma at fixed rates (SIR)
# R0 ~ beta_in * (n-1) * d / gamma ... approximate for SIS/network
# Here we use SI/SIR on sparse random connected graph with beta=0.003
CE04_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(name="SI (gamma=0)",         L=10, n=100, k=5, beta_in=0.003, beta_out=0.003, gamma=0.0,    T_end=4000.0, base_seed=390010),
    ScenarioConfig(name="SIR gamma=0.0005",     L=10, n=100, k=5, beta_in=0.003, beta_out=0.003, gamma=0.0005, T_end=4000.0, base_seed=390020),
    ScenarioConfig(name="SIR gamma=0.001",      L=10, n=100, k=5, beta_in=0.003, beta_out=0.003, gamma=0.001,  T_end=4000.0, base_seed=390030),
    ScenarioConfig(name="SIR gamma=0.003",      L=10, n=100, k=5, beta_in=0.003, beta_out=0.003, gamma=0.003,  T_end=5000.0, base_seed=390040),
]


# ---------------------------------------------------------------------------
# Single-replication runner
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


def _run_one(args: Tuple) -> List[dict]:
    scenario, rep_id = args
    cfg: ScenarioConfig = scenario
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
    initial_node = int(cfg.initial_node) if cfg.initial_node is not None else int(random.Random(run_seed).choice(comm_nodes[cfg.initial_community]))
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
        rows.append({"scenario": cfg.name, "replication_id": rep_id, "model": "Micro",
                     "layer": layer, "T_i": t if math.isfinite(t) else None,
                     "runtime": micro_rt, "network_seed": network_seed})

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
                     "runtime": mm_rt, "network_seed": network_seed})

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


def compute_metrics(arrivals_rows: List[dict], T_end: float, n_runs: int, bootstrap_reps: int = 1000, seed: int = 42) -> dict:
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

    # Bootstrap CI for RMST shift
    diffs = censored_var - censored_ref
    boots = np.array([rng.choice(diffs, len(diffs), replace=True).mean() for _ in range(bootstrap_reps)])
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])

    w1 = _w1(ref_obs, var_obs)
    mean_ref = float(np.mean(ref_obs)) if ref_obs.size > 0 else float("nan")
    rel_w1 = w1 / mean_ref if math.isfinite(mean_ref) and mean_ref > 0 else float("nan")

    # Per-layer metrics
    layer_rows = []
    for layer in sorted(df["layer"].unique()):
        m_layer = df[(df["layer"] == layer) & (df["model"] == "Micro")].set_index("replication_id")["T_i"].to_dict()
        mm_layer = df[(df["layer"] == layer) & (df["model"] == "MicroMacro")].set_index("replication_id")["T_i"].to_dict()
        ids = sorted(set(m_layer) | set(mm_layer))
        rt = np.array([m_layer.get(i) for i in ids], dtype=float)
        vt = np.array([mm_layer.get(i) for i in ids], dtype=float)
        paired = np.array([vt[i] - rt[i] for i in range(len(ids)) if math.isfinite(rt[i]) and math.isfinite(vt[i])])
        cr = np.where(np.isfinite(rt), rt, T_end)
        cv = np.where(np.isfinite(vt), vt, T_end)
        layer_rows.append({
            "layer": int(layer),
            "micro_completion": float(np.isfinite(rt).mean()),
            "mm_completion": float(np.isfinite(vt).mean()),
            "delta_RMST": float(np.mean(cv - cr)),
            "mean_shift": float(np.mean(paired)) if paired.size > 0 else float("nan"),
            "median_shift": float(np.median(paired)) if paired.size > 0 else float("nan"),
        })

    return {
        "micro_completion": float(np.isfinite(ref).mean()),
        "mm_completion": float(np.isfinite(var).mean()),
        "micro_mean_T_last": float(np.mean(ref_obs)) if ref_obs.size else float("nan"),
        "mm_mean_T_last": float(np.mean(var_obs)) if var_obs.size else float("nan"),
        "W1": w1,
        "rel_W1": rel_w1,
        "delta_RMST": delta_rmst,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "layer_metrics": layer_rows,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(scenarios: List[ScenarioConfig], out_dir: Path, n_workers: int = 4) -> None:
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    summary_rows: List[dict] = []

    for scenario in scenarios:
        print(f"\n=== Scenario: {scenario.name} ===", flush=True)
        tasks = [(scenario, rep) for rep in range(1, scenario.n_runs + 1)]
        scenario_rows: List[dict] = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_run_one, task): task[1] for task in tasks}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                rep_id = futures[future]
                try:
                    result = future.result()
                    scenario_rows.extend(result)
                    all_rows.extend(result)
                except Exception as exc:
                    print(f"  rep {rep_id} failed: {exc}", flush=True)
                if i % 20 == 0:
                    print(f"  {i}/{scenario.n_runs} done", flush=True)

        print(f"  Computing metrics...", flush=True)
        metrics = compute_metrics(scenario_rows, scenario.T_end, scenario.n_runs)
        summary_rows.append({
            "scenario": scenario.name,
            "beta_in": scenario.beta_in,
            "beta_out": scenario.beta_out,
            "gamma": scenario.gamma,
            "ratio": scenario.beta_out / scenario.beta_in,
            **{k: v for k, v in metrics.items() if k != "layer_metrics"},
        })
        print(f"  delta_RMST={metrics['delta_RMST']:.1f} CI=[{metrics['ci_low']:.1f}, {metrics['ci_high']:.1f}]  rel_W1={metrics['rel_W1']:.4f}", flush=True)

        # Save per-scenario layer metrics
        layer_df = pd.DataFrame(metrics["layer_metrics"])
        layer_df.insert(0, "scenario", scenario.name)
        layer_df.to_csv(out_dir / f"layer_metrics_{scenario.name.replace('=','').replace('.','p').replace(' ','_')}.csv", index=False)

    # Save all arrivals
    pd.DataFrame(all_rows).to_csv(out_dir / "arrivals.csv", index=False)
    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    # Save config
    config_list = [asdict(s) for s in scenarios]
    (out_dir / "config.json").write_text(json.dumps(config_list, indent=2), encoding="utf-8")

    print(f"\nDone. Results saved to {out_dir}", flush=True)
    print(summary_df[["scenario", "micro_completion", "mm_completion", "delta_RMST", "ci_low", "ci_high", "rel_W1"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CE-03 or CE-04 chain sweep experiments")
    parser.add_argument("--experiment", choices=["ce03", "ce04"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs per scenario")
    args = parser.parse_args()

    if args.experiment == "ce03":
        scenarios = CE03_SCENARIOS
        out_dir = REPO_ROOT / "output" / "claude_experiments" / "ce03_beta_ratio"
    else:
        scenarios = CE04_SCENARIOS
        out_dir = REPO_ROOT / "output" / "claude_experiments" / "ce04_sir_validation"

    if args.n_runs is not None:
        from dataclasses import replace
        scenarios = [replace(s, n_runs=args.n_runs) for s in scenarios]

    run_experiment(scenarios, out_dir, n_workers=args.workers)


if __name__ == "__main__":
    main()
