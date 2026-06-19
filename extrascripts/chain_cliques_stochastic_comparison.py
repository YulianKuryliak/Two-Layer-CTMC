import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp, wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulators import MicroMacroSimulator, MicroSimulator


def _format_float_for_id(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def scenario_id(*, L: int, n: int, k: int, macro_graph_type: str, micro_graph_type: str, edge_prob: float) -> str:
    return (
        f"k_{L}_n_{n}_inter_{k}_macro_{macro_graph_type}_"
        f"micro_{micro_graph_type}_p{_format_float_for_id(edge_prob)}"
    )


def first_arrivals_from_micro(result: dict, L: int) -> List[float]:
    arrivals = [math.nan] * L
    for row in result.get("community_timings", []):
        comm = int(row["community"])
        if 0 <= comm < L:
            value = row.get("first_infection_time")
            arrivals[comm] = float(value) if value is not None else math.nan
    return arrivals


def first_arrivals_from_micromacro(event_log: list[dict], L: int) -> List[float]:
    arrivals = [math.nan] * L
    for event in sorted(event_log, key=lambda row: float(row.get("time", math.inf))):
        event_type = str(event.get("event_type", "")).lower()
        if event_type not in {"seed", "infection", "transfer"}:
            continue
        comm = event.get("community")
        if comm is None:
            continue
        comm = int(comm)
        if 0 <= comm < L and math.isnan(arrivals[comm]):
            arrivals[comm] = float(event["time"])
    return arrivals


def delays_from_arrivals(arrivals: List[float]) -> List[float]:
    delays = []
    for idx in range(len(arrivals) - 1):
        left = arrivals[idx]
        right = arrivals[idx + 1]
        if math.isnan(left) or math.isnan(right):
            delays.append(math.nan)
        else:
            delays.append(float(right - left))
    return delays


def run_one_replication(args: argparse.Namespace, replication_id: int, run_seed: int, network_seed: int) -> List[dict]:
    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=args.L,
        community_size=args.n,
        inter_links=args.k,
        seed=network_seed,
        macro_graph_type=args.macro_graph_type,
        micro_graph_type=args.micro_graph_type,
        edge_prob=args.edge_prob,
    )
    comm_nodes = [list(graph.nodes()) for graph in micro_graphs]
    initial_node = args.initial_node

    rows: List[dict] = []

    micro = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=args.beta,
        recovery_rate=args.gamma,
        model=args.model,
    )
    micro_rng = random.Random(run_seed)
    t0 = time.perf_counter()
    micro_result = micro.run(
        T_end=args.T_end,
        dt_out=args.dt_out,
        initial_node=initial_node,
        rng=micro_rng,
    )
    micro_runtime = time.perf_counter() - t0
    micro_arrivals = first_arrivals_from_micro(micro_result, args.L)
    rows.append(
        build_replication_row(
            model_name="Micro",
            replication_id=replication_id,
            run_seed=run_seed,
            network_seed=network_seed,
            runtime_seconds=micro_runtime,
            arrivals=micro_arrivals,
        )
    )

    micromacro = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=args.beta,
        gamma=args.gamma,
        beta_macro=args.beta_macro,
        tau_micro=args.tau_micro,
        T_end=args.T_end,
        macro_T=args.macro_T,
        model=args.model,
        full_graph=full_graph,
        verbose_steps=False,
    )
    t0 = time.perf_counter()
    _, _, _, event_log = micromacro.run(
        seed=run_seed,
        initial_community=args.initial_community,
        initial_node=initial_node,
    )
    mm_runtime = time.perf_counter() - t0
    mm_arrivals = first_arrivals_from_micromacro(event_log, args.L)
    rows.append(
        build_replication_row(
            model_name="MicroMacro",
            replication_id=replication_id,
            run_seed=run_seed,
            network_seed=network_seed,
            runtime_seconds=mm_runtime,
            arrivals=mm_arrivals,
        )
    )

    return rows


def build_replication_row(
    *,
    model_name: str,
    replication_id: int,
    run_seed: int,
    network_seed: int,
    runtime_seconds: float,
    arrivals: List[float],
) -> dict:
    delays = delays_from_arrivals(arrivals)
    row = {
        "model": model_name,
        "replication_id": int(replication_id),
        "run_seed": int(run_seed),
        "network_seed": int(network_seed),
        "runtime_seconds": float(runtime_seconds),
        "reached_all": bool(not math.isnan(arrivals[-1])),
        "T_last": float(arrivals[-1]) if not math.isnan(arrivals[-1]) else math.nan,
    }
    for idx, value in enumerate(arrivals):
        row[f"T_{idx}"] = value
    for idx, value in enumerate(delays):
        row[f"DeltaT_{idx}"] = value
    return row


def finite_pair(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    left_values = left.dropna().to_numpy(dtype=float)
    right_values = right.dropna().to_numpy(dtype=float)
    return left_values, right_values


def distribution_metrics(micro_values: np.ndarray, mm_values: np.ndarray, *, n_expected: int) -> dict:
    micro_reached_fraction = float(len(micro_values) / n_expected) if n_expected else math.nan
    mm_reached_fraction = float(len(mm_values) / n_expected) if n_expected else math.nan
    if len(micro_values) == 0 or len(mm_values) == 0:
        return {
            "n_micro": len(micro_values),
            "n_micromacro": len(mm_values),
            "micro_reached_fraction": micro_reached_fraction,
            "micromacro_reached_fraction": mm_reached_fraction,
            "W1": math.nan,
            "D_KS": math.nan,
            "delta_mean": math.nan,
            "delta_q50": math.nan,
            "delta_q90": math.nan,
            "micro_mean": math.nan,
            "micromacro_mean": math.nan,
            "micro_q50": math.nan,
            "micromacro_q50": math.nan,
            "micro_q90": math.nan,
            "micromacro_q90": math.nan,
        }
    ks = ks_2samp(micro_values, mm_values, alternative="two-sided", mode="auto")
    micro_mean = float(np.mean(micro_values))
    mm_mean = float(np.mean(mm_values))
    micro_q50 = float(np.quantile(micro_values, 0.5))
    mm_q50 = float(np.quantile(mm_values, 0.5))
    micro_q90 = float(np.quantile(micro_values, 0.9))
    mm_q90 = float(np.quantile(mm_values, 0.9))
    return {
        "n_micro": int(len(micro_values)),
        "n_micromacro": int(len(mm_values)),
        "micro_reached_fraction": micro_reached_fraction,
        "micromacro_reached_fraction": mm_reached_fraction,
        "W1": float(wasserstein_distance(micro_values, mm_values)),
        "D_KS": float(ks.statistic),
        "delta_mean": float(mm_mean - micro_mean),
        "delta_q50": float(mm_q50 - micro_q50),
        "delta_q90": float(mm_q90 - micro_q90),
        "micro_mean": micro_mean,
        "micromacro_mean": mm_mean,
        "micro_q50": micro_q50,
        "micromacro_q50": mm_q50,
        "micro_q90": micro_q90,
        "micromacro_q90": mm_q90,
    }


def paired_error_metrics(per_rep: pd.DataFrame, column: str) -> dict:
    pivot = per_rep.pivot(index="replication_id", columns="model", values=column)
    if "Micro" not in pivot or "MicroMacro" not in pivot:
        return {"paired_n": 0, "paired_bias": math.nan, "paired_MAE": math.nan, "paired_RMSE": math.nan}
    diff = pivot["MicroMacro"] - pivot["Micro"]
    diff = diff.dropna().to_numpy(dtype=float)
    if len(diff) == 0:
        return {"paired_n": 0, "paired_bias": math.nan, "paired_MAE": math.nan, "paired_RMSE": math.nan}
    return {
        "paired_n": int(len(diff)),
        "paired_bias": float(np.mean(diff)),
        "paired_MAE": float(np.mean(np.abs(diff))),
        "paired_RMSE": float(np.sqrt(np.mean(diff * diff))),
    }


def build_metric_tables(per_rep: pd.DataFrame, L: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    micro_rows = per_rep[per_rep["model"] == "Micro"]
    mm_rows = per_rep[per_rep["model"] == "MicroMacro"]
    n_expected = int(min(len(micro_rows), len(mm_rows)))

    global_rows = []
    micro_values, mm_values = finite_pair(micro_rows["T_last"], mm_rows["T_last"])
    global_rows.append(
        {
            "observable": "T_last",
            **distribution_metrics(micro_values, mm_values, n_expected=n_expected),
            **paired_error_metrics(per_rep, "T_last"),
        }
    )
    global_df = pd.DataFrame(global_rows)

    arrival_rows = []
    for idx in range(L):
        column = f"T_{idx}"
        micro_values, mm_values = finite_pair(micro_rows[column], mm_rows[column])
        arrival_rows.append(
            {
                "layer": idx,
                "observable": column,
                **distribution_metrics(micro_values, mm_values, n_expected=n_expected),
                **paired_error_metrics(per_rep, column),
            }
        )
    arrival_df = pd.DataFrame(arrival_rows)

    delay_rows = []
    for idx in range(L - 1):
        column = f"DeltaT_{idx}"
        micro_values, mm_values = finite_pair(micro_rows[column], mm_rows[column])
        delay_rows.append(
            {
                "transition": f"{idx}->{idx + 1}",
                "observable": column,
                **distribution_metrics(micro_values, mm_values, n_expected=n_expected),
                **paired_error_metrics(per_rep, column),
            }
        )
    delay_df = pd.DataFrame(delay_rows)

    runtime_df = (
        per_rep.groupby("model")["runtime_seconds"]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
        .rename(columns={"count": "n_runs", "mean": "mean_seconds", "median": "median_seconds", "sum": "total_seconds"})
    )
    totals = runtime_df.set_index("model")["total_seconds"].to_dict()
    if "Micro" in totals and "MicroMacro" in totals and totals["MicroMacro"] > 0:
        speedup = totals["Micro"] / totals["MicroMacro"]
    else:
        speedup = math.nan
    runtime_df["speedup_total_micro_over_micromacro"] = speedup
    return global_df, arrival_df, delay_df, runtime_df


def write_long_tables(per_rep: pd.DataFrame, L: int, out_dir: Path) -> None:
    arrival_records = []
    delay_records = []
    for _, row in per_rep.iterrows():
        base = {
            "model": row["model"],
            "replication_id": int(row["replication_id"]),
            "run_seed": int(row["run_seed"]),
            "network_seed": int(row["network_seed"]),
        }
        for idx in range(L):
            arrival_records.append({**base, "layer": idx, "arrival_time": row[f"T_{idx}"]})
        for idx in range(L - 1):
            delay_records.append({**base, "transition": f"{idx}->{idx + 1}", "delay": row[f"DeltaT_{idx}"]})
    pd.DataFrame(arrival_records).to_csv(out_dir / "arrival_times_long.csv", index=False)
    pd.DataFrame(delay_records).to_csv(out_dir / "delays_long.csv", index=False)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1, dtype=float) / len(xs)
    return xs, ys


def plot_cdf(per_rep: pd.DataFrame, column: str, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"Micro": "#2ca02c", "MicroMacro": "#1f77b4"}
    for model_name in ["Micro", "MicroMacro"]:
        values = per_rep.loc[per_rep["model"] == model_name, column].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        xs, ys = ecdf(values)
        ax.step(xs, ys, where="post", label=f"{model_name} (n={len(values)})", color=colors[model_name], linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical CDF")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_density_values(ax, values: np.ndarray, *, label: str, color: str) -> None:
    if len(values) == 0:
        return
    if len(values) >= 2 and float(np.std(values)) > 0.0:
        xs = np.linspace(float(np.min(values)), float(np.max(values)), 256)
        ys = gaussian_kde(values)(xs)
        ax.plot(xs, ys, label=label, color=color, linewidth=2)
        ax.fill_between(xs, ys, color=color, alpha=0.16)
        return
    ax.hist(values, bins=1, density=True, alpha=0.25, color=color, label=label)


def plot_density(per_rep: pd.DataFrame, column: str, title: str, xlabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"Micro": "#2ca02c", "MicroMacro": "#1f77b4"}
    for model_name in ["Micro", "MicroMacro"]:
        values = per_rep.loc[per_rep["model"] == model_name, column].dropna().to_numpy(dtype=float)
        _plot_density_values(ax, values, label=f"{model_name} (n={len(values)})", color=colors[model_name])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delay_cdfs(per_rep: pd.DataFrame, delay_indices: List[int], out_path: Path) -> None:
    ncols = len(delay_indices)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2), squeeze=False)
    colors = {"Micro": "#2ca02c", "MicroMacro": "#1f77b4"}
    for ax, idx in zip(axes[0], delay_indices):
        column = f"DeltaT_{idx}"
        for model_name in ["Micro", "MicroMacro"]:
            values = per_rep.loc[per_rep["model"] == model_name, column].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            xs, ys = ecdf(values)
            ax.step(xs, ys, where="post", label=model_name, color=colors[model_name], linewidth=2)
        ax.set_title(rf"$\Delta T_{idx}$")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_delay_densities(per_rep: pd.DataFrame, delay_indices: List[int], out_path: Path) -> None:
    ncols = len(delay_indices)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.2), squeeze=False)
    colors = {"Micro": "#2ca02c", "MicroMacro": "#1f77b4"}
    for ax, idx in zip(axes[0], delay_indices):
        column = f"DeltaT_{idx}"
        for model_name in ["Micro", "MicroMacro"]:
            values = per_rep.loc[per_rep["model"] == model_name, column].dropna().to_numpy(dtype=float)
            _plot_density_values(ax, values, label=model_name, color=colors[model_name])
        ax.set_title(rf"$\Delta T_{idx}$")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_error_growth(arrival_metrics: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    x = arrival_metrics["layer"].to_numpy(dtype=int)
    panels = [
        ("W1", r"$W_1(T_i)$"),
        ("delta_mean", r"$\Delta\mu(T_i)$"),
        ("delta_q50", r"$\Delta q_{0.5}(T_i)$"),
    ]
    for ax, (column, label) in zip(axes, panels):
        ax.plot(x, arrival_metrics[column].to_numpy(dtype=float), marker="o", linewidth=2, color="#333333")
        ax.axhline(0.0, color="#999999", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Community index i")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def git_value(args: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def write_parameters(args: argparse.Namespace, out_dir: Path, run_label: str) -> None:
    status = git_value(["status", "--short"])
    params = {
        "run_label": run_label,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo_commit": git_value(["rev-parse", "HEAD"]),
        "dirty_worktree": bool(status),
        "L": args.L,
        "n": args.n,
        "k_inter_links": args.k,
        "beta": args.beta,
        "beta_macro": args.beta_macro,
        "gamma": args.gamma,
        "model": args.model,
        "tau_micro": args.tau_micro,
        "macro_T": args.macro_T,
        "dt_out": args.dt_out,
        "T_end": args.T_end,
        "n_replications": args.n_replications,
        "base_seed": args.base_seed,
        "network_seed_base": args.network_seed_base,
        "initial_community": args.initial_community,
        "initial_node": args.initial_node,
        "macro_graph_type": args.macro_graph_type,
        "micro_graph_type": args.micro_graph_type,
        "edge_prob": args.edge_prob,
    }
    with open(out_dir / "experiment_metadata.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    with open(out_dir / "parameters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for key, value in params.items():
            writer.writerow([key, value])


def write_latex_table(global_metrics: pd.DataFrame, delay_metrics: pd.DataFrame, out_path: Path) -> None:
    selected = pd.concat(
        [
            global_metrics.assign(row_label="T_last"),
            delay_metrics[delay_metrics["observable"].isin(["DeltaT_0", f"DeltaT_{len(delay_metrics) - 1}"])].assign(
                row_label=lambda df: df["observable"]
            ),
        ],
        ignore_index=True,
    )
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Observable & $W_1$ & $D_{\mathrm{KS}}$ & $\Delta\mu$ & $\Delta q_{0.5}$ & $\Delta q_{0.9}$ \\",
        r"\hline",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"{row['row_label']} & {row['W1']:.3f} & {row['D_KS']:.3f} & "
            f"{row['delta_mean']:.3f} & {row['delta_q50']:.3f} & {row['delta_q90']:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stochastic Micro vs MicroMacro chain-of-cliques comparison.")
    parser.add_argument("--L", type=int, default=7, help="Number of clique communities.")
    parser.add_argument("--n", type=int, default=100, help="Nodes per clique.")
    parser.add_argument("--k", type=int, default=1, help="Inter-community links between neighboring cliques.")
    parser.add_argument("--beta", type=float, default=0.005, help="Micro infection rate.")
    parser.add_argument("--beta-macro", type=float, default=None, help="Macro infection rate; defaults to beta.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Recovery rate.")
    parser.add_argument("--model", type=int, default=2, help="Local epidemic model, 2=SIR/SI with gamma=0.")
    parser.add_argument("--tau-micro", type=float, default=1.0, help="MicroMacro synchronization step.")
    parser.add_argument("--macro-T", type=float, default=1.0, help="Macro rate multiplier T.")
    parser.add_argument("--dt-out", type=float, default=1.0, help="Micro output grid step.")
    parser.add_argument("--T-end", type=float, default=2500.0, help="Simulation horizon.")
    parser.add_argument("--n-replications", type=int, default=100, help="Monte Carlo replications.")
    parser.add_argument("--base-seed", type=int, default=42000, help="Base stochastic run seed.")
    parser.add_argument("--network-seed-base", type=int, default=52000, help="Base network seed.")
    parser.add_argument("--initial-community", type=int, default=0)
    parser.add_argument("--initial-node", type=int, default=0)
    parser.add_argument("--macro-graph-type", default="chain")
    parser.add_argument("--micro-graph-type", default="complete")
    parser.add_argument("--edge-prob", type=float, default=0.031)
    parser.add_argument("--run-label", default=None, help="Output label. Defaults to timestamped run label.")
    parser.add_argument("--data-root", default=str(Path("data") / "Simulations_ChainCliques"))
    parser.add_argument("--plot-root", default="plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.beta_macro = args.beta if args.beta_macro is None else args.beta_macro
    if args.n_replications < 1:
        raise ValueError("n_replications must be positive")
    if args.L < 2:
        raise ValueError("L must be at least 2")

    sid = scenario_id(
        L=args.L,
        n=args.n,
        k=args.k,
        macro_graph_type=args.macro_graph_type,
        micro_graph_type=args.micro_graph_type,
        edge_prob=args.edge_prob,
    )
    run_label = args.run_label or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    data_dir = REPO_ROOT / args.data_root / sid / run_label
    plot_dir = REPO_ROOT / args.plot_root / sid
    data_dir.mkdir(parents=True, exist_ok=False)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    rep_records = []
    experiment_start = time.perf_counter()
    for rep_idx in range(args.n_replications):
        replication_id = rep_idx + 1
        run_seed = args.base_seed + rep_idx
        network_seed = args.network_seed_base + rep_idx
        rep_records.append({"replication_id": replication_id, "run_seed": run_seed, "network_seed": network_seed})
        rows.extend(run_one_replication(args, replication_id, run_seed, network_seed))
        if replication_id == 1 or replication_id % max(1, args.n_replications // 10) == 0:
            print(f"completed {replication_id}/{args.n_replications} replications")

    elapsed = time.perf_counter() - experiment_start
    per_rep = pd.DataFrame(rows).sort_values(["replication_id", "model"])
    replications = pd.DataFrame(rep_records)
    write_parameters(args, data_dir, run_label)
    replications.to_csv(data_dir / "replications.csv", index=False)
    per_rep.to_csv(data_dir / "per_replication_arrivals.csv", index=False)
    write_long_tables(per_rep, args.L, data_dir)

    global_metrics, arrival_metrics, delay_metrics, runtime_summary = build_metric_tables(per_rep, args.L)
    global_metrics.to_csv(data_dir / "global_arrival_metrics.csv", index=False)
    arrival_metrics.to_csv(data_dir / "arrival_layer_metrics.csv", index=False)
    delay_metrics.to_csv(data_dir / "delay_metrics.csv", index=False)
    runtime_summary.to_csv(data_dir / "runtime_summary.csv", index=False)
    write_latex_table(global_metrics, delay_metrics, data_dir / "paper_metrics_table.tex")

    plot_cdf(
        per_rep,
        "T_last",
        r"First arrival in last community: $T_{\mathrm{last}}$",
        r"$T_{\mathrm{last}}$",
        plot_dir / f"{run_label}_tlast_cdf.png",
    )
    plot_density(
        per_rep,
        "T_last",
        r"Density of first arrival in last community: $T_{\mathrm{last}}$",
        r"$T_{\mathrm{last}}$",
        plot_dir / f"{run_label}_tlast_density.png",
    )
    delay_indices = sorted(set([0, max(0, (args.L - 2) // 2), args.L - 2]))
    plot_delay_cdfs(per_rep, delay_indices, plot_dir / f"{run_label}_selected_delay_cdfs.png")
    plot_delay_densities(per_rep, delay_indices, plot_dir / f"{run_label}_selected_delay_densities.png")
    plot_error_growth(arrival_metrics, plot_dir / f"{run_label}_arrival_error_growth.png")

    summary = {
        "data_dir": str(data_dir.relative_to(REPO_ROOT)),
        "plot_dir": str(plot_dir.relative_to(REPO_ROOT)),
        "run_label": run_label,
        "scenario_id": sid,
        "elapsed_seconds": elapsed,
        "plot_files": [
            str((plot_dir / f"{run_label}_tlast_cdf.png").relative_to(REPO_ROOT)),
            str((plot_dir / f"{run_label}_tlast_density.png").relative_to(REPO_ROOT)),
            str((plot_dir / f"{run_label}_selected_delay_cdfs.png").relative_to(REPO_ROOT)),
            str((plot_dir / f"{run_label}_selected_delay_densities.png").relative_to(REPO_ROOT)),
            str((plot_dir / f"{run_label}_arrival_error_growth.png").relative_to(REPO_ROOT)),
        ],
    }
    with open(data_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
