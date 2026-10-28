import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.simulate import normalize_model
from two_layer_ctmc.simulators import MicroMacroSimulator, MicroSimulator


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required; install the analysis dependencies") from exc
    return pd


def _require_matplotlib():
    try:
        cache_dir = REPO_ROOT / ".matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("matplotlib is required; install the analysis dependencies") from exc
    return plt


@dataclass(frozen=True)
class ExperimentConfig:
    L: int = 5
    n: int = 100
    k: int = 1
    beta: float = 0.005
    gamma: float = 0.0
    model: str = "SIR"
    tau_micro: float = 1.0
    macro_T: float = 1.0
    T_end: float = 1000.0
    n_runs: int = 100
    base_seed: int = 42
    network_seed: int = 42
    regenerate_network_each_run: bool = True
    macro_graph_type: str = "chain"
    micro_graph_type: str = "complete"
    edge_prob: float = 0.031
    initial_community: int = 0
    initial_node: Optional[int] = 0

    @property
    def scenario_id(self) -> str:
        beta_id = _format_float(self.beta)
        return f"chain_L{self.L}_n{self.n}_k{self.k}_beta{beta_id}"


def _format_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _plot_dir(cfg: ExperimentConfig) -> Path:
    p = _format_float(cfg.edge_prob)
    folder = (
        f"k_{cfg.L}_n_{cfg.n}_inter_{cfg.k}_"
        f"macro_{cfg.macro_graph_type}_micro_{cfg.micro_graph_type}_p{p}"
    )
    return REPO_ROOT / "plots" / folder


def _data_dir(cfg: ExperimentConfig, run_tag: str) -> Path:
    return REPO_ROOT / "data" / "chain_cliques_arrival_comparison" / f"{cfg.scenario_id}_{run_tag}"


def _trajectory_dir(cfg: ExperimentConfig, run_tag: str) -> Path:
    beta_id = _format_float(cfg.beta)
    folder = (
        f"k_{cfg.L}_n_{cfg.n}_inter_{cfg.k}_"
        f"macro_{cfg.macro_graph_type}_micro_{cfg.micro_graph_type}_beta{beta_id}_{run_tag}"
    )
    return REPO_ROOT / "data" / "Simulations_ChainCliques_v2" / folder


def _finite(values: Iterable[object]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _w1_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance  # type: ignore

        return float(wasserstein_distance(a, b))
    except Exception:
        xa = np.sort(a)
        xb = np.sort(b)
        if xa.size == xb.size:
            return float(np.mean(np.abs(xa - xb)))
        grid = np.unique(np.concatenate([xa, xb]))
        if grid.size <= 1:
            return 0.0
        fa = np.searchsorted(xa, grid, side="right") / float(xa.size)
        fb = np.searchsorted(xb, grid, side="right") / float(xb.size)
        return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * np.diff(grid)))


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    try:
        from scipy.stats import ks_2samp  # type: ignore

        return float(ks_2samp(a, b, method="auto").statistic)
    except Exception:
        grid = np.unique(np.concatenate([a, b]))
        fa = np.searchsorted(np.sort(a), grid, side="right") / float(a.size)
        fb = np.searchsorted(np.sort(b), grid, side="right") / float(b.size)
        return float(np.max(np.abs(fa - fb)))


def _q(values: np.ndarray, prob: float) -> float:
    return float(np.quantile(values, prob)) if values.size else float("nan")


def _empirical_cdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def _arrival_times_micro(result: dict, L: int) -> List[float]:
    out = [float("nan")] * L
    for row in result.get("community_timings", []):
        community = int(row["community"])
        if 0 <= community < L and row.get("first_infection_time") is not None:
            out[community] = float(row["first_infection_time"])
    return out


def _arrival_times_micromacro(event_log: Sequence[dict], L: int) -> List[float]:
    out = [float("nan")] * L
    for event in sorted(event_log, key=lambda item: float(item.get("time", math.inf))):
        event_type = str(event.get("event_type", "")).lower()
        if event_type not in {"seed", "infection", "transfer"}:
            continue
        community = event.get("community")
        if community is None:
            continue
        community = int(community)
        if 0 <= community < L and not math.isfinite(out[community]):
            out[community] = float(event["time"])
    return out


def _delays(arrivals: Sequence[float]) -> List[float]:
    out: List[float] = []
    for left, right in zip(arrivals[:-1], arrivals[1:]):
        if math.isfinite(left) and math.isfinite(right):
            out.append(float(right - left))
        else:
            out.append(float("nan"))
    return out


def _choose_initial_node(cfg: ExperimentConfig, comm_nodes: Sequence[Sequence[int]], run_seed: int) -> Optional[int]:
    if cfg.initial_node is not None:
        return int(cfg.initial_node)
    if not (0 <= cfg.initial_community < len(comm_nodes)):
        return None
    nodes = list(comm_nodes[cfg.initial_community])
    if not nodes:
        return None
    return int(random.Random(run_seed).choice(nodes))


def _run_one_replication(
    cfg: ExperimentConfig,
    replication_id: int,
    trajectory_root: Optional[Path] = None,
) -> Tuple[List[dict], List[dict]]:
    run_seed = cfg.base_seed + replication_id - 1
    network_seed = cfg.network_seed + replication_id - 1 if cfg.regenerate_network_each_run else cfg.network_seed

    micro_graphs, full_graph, W = generate_two_scale_network(
        n_communities=cfg.L,
        community_size=cfg.n,
        inter_links=cfg.k,
        seed=network_seed,
        macro_graph_type=cfg.macro_graph_type,
        micro_graph_type=cfg.micro_graph_type,
        edge_prob=cfg.edge_prob,
    )
    comm_nodes = [list(g.nodes()) for g in micro_graphs]
    initial_node = _choose_initial_node(cfg, comm_nodes, run_seed)
    model_id = normalize_model(cfg.model)

    metadata_rows: List[dict] = []
    arrival_rows: List[dict] = []

    micro_sim = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=comm_nodes,
        infection_rate=cfg.beta,
        recovery_rate=cfg.gamma,
        model=model_id,
    )
    t0 = time.perf_counter()
    micro_result = micro_sim.run(
        T_end=cfg.T_end,
        dt_out=cfg.tau_micro,
        initial_node=initial_node,
        rng=random.Random(run_seed),
    )
    micro_runtime = time.perf_counter() - t0
    arrivals = _arrival_times_micro(micro_result, cfg.L)
    arrival_rows.extend(_arrival_records(cfg, "Micro", replication_id, run_seed, network_seed, initial_node, arrivals))
    metadata_rows.append(
        _metadata_record(cfg, "Micro", replication_id, run_seed, network_seed, initial_node, micro_runtime)
    )
    if trajectory_root is not None:
        _write_micro_trajectory(trajectory_root, replication_id, cfg, micro_result["rows"], run_seed, network_seed, initial_node)

    mm_sim = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=cfg.beta,
        gamma=cfg.gamma,
        beta_macro=cfg.beta,
        tau_micro=cfg.tau_micro,
        T_end=cfg.T_end,
        macro_T=cfg.macro_T,
        model=model_id,
        full_graph=full_graph,
    )
    t0 = time.perf_counter()
    _, _, logs_per_comm, event_log = mm_sim.run(
        seed=run_seed,
        initial_community=cfg.initial_community,
        initial_node=initial_node,
    )
    mm_runtime = time.perf_counter() - t0
    arrivals = _arrival_times_micromacro(event_log, cfg.L)
    arrival_rows.extend(
        _arrival_records(cfg, "MicroMacro", replication_id, run_seed, network_seed, initial_node, arrivals)
    )
    metadata_rows.append(
        _metadata_record(cfg, "MicroMacro", replication_id, run_seed, network_seed, initial_node, mm_runtime)
    )
    if trajectory_root is not None:
        _write_micromacro_trajectory(trajectory_root, replication_id, cfg, logs_per_comm, run_seed, network_seed, initial_node)
    return metadata_rows, arrival_rows


def _write_metadata_json(rep_dir: Path, cfg: ExperimentConfig, replication_id: int, run_seed: int, network_seed: int, initial_node: Optional[int]) -> None:
    payload = {
        **asdict(cfg),
        "scenario_id": cfg.scenario_id,
        "replication_id": replication_id,
        "run_seed": run_seed,
        "network_seed": network_seed,
        "initial_node": initial_node,
    }
    (rep_dir / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_micro_trajectory(
    trajectory_root: Path,
    replication_id: int,
    cfg: ExperimentConfig,
    rows: Sequence[tuple],
    run_seed: int,
    network_seed: int,
    initial_node: Optional[int],
) -> None:
    pd = _require_pandas()
    rep_dir = trajectory_root / f"run_{replication_id - 1:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=["community", "time", "S", "I", "R"])
    out.insert(0, "model", "Micro")
    out.insert(0, "run_id", replication_id - 1)
    out.to_csv(rep_dir / "Micro.csv", index=False)
    _write_metadata_json(rep_dir, cfg, replication_id, run_seed, network_seed, initial_node)


def _write_micromacro_trajectory(
    trajectory_root: Path,
    replication_id: int,
    cfg: ExperimentConfig,
    logs_per_comm: Dict[int, dict],
    run_seed: int,
    network_seed: int,
    initial_node: Optional[int],
) -> None:
    pd = _require_pandas()
    rep_dir = trajectory_root / f"run_{replication_id - 1:03d}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for community, logs in sorted(logs_per_comm.items()):
        for time_value, s_value, i_value, r_value in zip(logs["times"], logs["S"], logs["I"], logs["R"]):
            rows.append(
                {
                    "run_id": replication_id - 1,
                    "model": "MicroMacro",
                    "community": int(community),
                    "time": float(time_value),
                    "S": int(s_value),
                    "I": int(i_value),
                    "R": int(r_value),
                }
            )
    pd.DataFrame(rows).to_csv(rep_dir / "MicroMacro.csv", index=False)
    _write_metadata_json(rep_dir, cfg, replication_id, run_seed, network_seed, initial_node)


def _metadata_record(
    cfg: ExperimentConfig,
    model_name: str,
    replication_id: int,
    run_seed: int,
    network_seed: int,
    initial_node: Optional[int],
    runtime_seconds: float,
) -> dict:
    return {
        "scenario_id": cfg.scenario_id,
        "model": model_name,
        "replication_id": replication_id,
        "run_seed": run_seed,
        "network_seed": network_seed,
        "initial_community": cfg.initial_community,
        "initial_node": initial_node,
        "runtime_seconds": runtime_seconds,
        "L": cfg.L,
        "n": cfg.n,
        "k": cfg.k,
        "beta": cfg.beta,
        "gamma": cfg.gamma,
        "process_model": cfg.model,
        "T_end": cfg.T_end,
        "tau_micro": cfg.tau_micro,
        "macro_T": cfg.macro_T,
        "macro_graph_type": cfg.macro_graph_type,
        "micro_graph_type": cfg.micro_graph_type,
        "edge_prob": cfg.edge_prob,
    }


def _arrival_records(
    cfg: ExperimentConfig,
    model_name: str,
    replication_id: int,
    run_seed: int,
    network_seed: int,
    initial_node: Optional[int],
    arrivals: Sequence[float],
) -> List[dict]:
    censored_arrivals = [
        float(t_i) if math.isfinite(float(t_i)) and float(t_i) <= cfg.T_end else float("nan")
        for t_i in arrivals
    ]
    delays = _delays(censored_arrivals)
    t_last = float(censored_arrivals[-1]) if censored_arrivals else float("nan")
    reached_last = math.isfinite(t_last)
    rows: List[dict] = []
    for layer, t_i in enumerate(censored_arrivals):
        delta_t = delays[layer] if layer < len(delays) else float("nan")
        rows.append(
            {
                "scenario_id": cfg.scenario_id,
                "model": model_name,
                "replication_id": replication_id,
                "run_seed": run_seed,
                "network_seed": network_seed,
                "initial_node": initial_node,
                "layer_index": layer,
                "T_i": t_i,
                "Delta_T_i": delta_t,
                "T_last": t_last,
                "reached_i": math.isfinite(t_i),
                "reached_next": math.isfinite(delta_t),
                "reached_last": reached_last,
            }
        )
    return rows


def _wide_table(arrivals_df):
    pd = _require_pandas()
    base_cols = ["scenario_id", "model", "replication_id", "run_seed", "network_seed", "initial_node"]
    t_wide = arrivals_df.pivot_table(index=base_cols, columns="layer_index", values="T_i", aggfunc="first")
    t_wide = t_wide.rename(columns={c: f"T_{int(c)}" for c in t_wide.columns})
    d_wide = arrivals_df.pivot_table(index=base_cols, columns="layer_index", values="Delta_T_i", aggfunc="first")
    d_wide = d_wide.rename(columns={c: f"Delta_T_{int(c)}" for c in d_wide.columns})
    out = pd.concat([t_wide, d_wide], axis=1).reset_index()
    t_cols = sorted([c for c in out.columns if str(c).startswith("T_")], key=lambda x: int(str(x).split("_")[1]))
    last_col = t_cols[-1]
    out["T_last"] = out[last_col]
    out["complete"] = np.isfinite(out["T_last"].to_numpy(dtype=float))
    return out


def _summary_tables(arrivals_df, wide_df, cfg: ExperimentConfig):
    pd = _require_pandas()
    summary_rows: List[dict] = []
    comparison_rows: List[dict] = []
    paired_rows: List[dict] = []

    for model_name, mdf in arrivals_df.groupby("model"):
        for layer in range(cfg.L):
            vals = _finite(mdf.loc[mdf["layer_index"] == layer, "T_i"])
            summary_rows.append(_summary_record(cfg, model_name, "T_i", layer, vals, int(mdf["replication_id"].nunique())))
        for delay_idx in range(cfg.L - 1):
            vals = _finite(mdf.loc[mdf["layer_index"] == delay_idx, "Delta_T_i"])
            summary_rows.append(
                _summary_record(cfg, model_name, "Delta_T_i", delay_idx, vals, int(mdf["replication_id"].nunique()))
            )
        tlast = _finite(wide_df.loc[wide_df["model"] == model_name, "T_last"])
        summary_rows.append(_summary_record(cfg, model_name, "T_last", cfg.L - 1, tlast, int(mdf["replication_id"].nunique())))

    for layer in range(cfg.L):
        comparison_rows.append(_comparison_record(cfg, "T_i", layer, arrivals_df, "T_i"))
    for delay_idx in range(cfg.L - 1):
        comparison_rows.append(_comparison_record(cfg, "Delta_T_i", delay_idx, arrivals_df, "Delta_T_i"))
    comparison_rows.append(_comparison_from_wide(cfg, "T_last", cfg.L - 1, wide_df, "T_last"))

    micro = wide_df[wide_df["model"] == "Micro"].set_index("replication_id")
    mm = wide_df[wide_df["model"] == "MicroMacro"].set_index("replication_id")
    for rep in sorted(set(micro.index) & set(mm.index)):
        mrow = micro.loc[rep]
        arow = mm.loc[rep]
        if getattr(mrow, "ndim", 1) > 1:
            mrow = mrow.iloc[0]
        if getattr(arow, "ndim", 1) > 1:
            arow = arow.iloc[0]
        for layer in range(cfg.L):
            err_t = _paired_diff(arow.get(f"T_{layer}"), mrow.get(f"T_{layer}"))
            err_d = _paired_diff(arow.get(f"Delta_T_{layer}"), mrow.get(f"Delta_T_{layer}")) if layer < cfg.L - 1 else float("nan")
            paired_rows.append(
                {
                    "scenario_id": cfg.scenario_id,
                    "replication_id": int(rep),
                    "run_seed": int(mrow["run_seed"]),
                    "network_seed": int(mrow["network_seed"]),
                    "layer_index": layer,
                    "error_T_i": err_t,
                    "error_Delta_T_i": err_d,
                    "error_T_last": _paired_diff(arow.get("T_last"), mrow.get("T_last")),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows), pd.DataFrame(paired_rows)


def _runtime_summary(metadata_df):
    summary = (
        metadata_df.groupby("model")["runtime_seconds"]
        .agg(n_runs="count", total_runtime_seconds="sum", mean_runtime_seconds="mean", median_runtime_seconds="median")
        .reset_index()
    )
    values = {row["model"]: float(row["total_runtime_seconds"]) for _, row in summary.iterrows()}
    micro = values.get("Micro", float("nan"))
    micromacro = values.get("MicroMacro", float("nan"))
    speedup = micro / micromacro if math.isfinite(micro) and math.isfinite(micromacro) and micromacro > 0.0 else float("nan")
    summary["speedup_micro_over_micromacro"] = speedup
    return summary


def _summary_record(cfg: ExperimentConfig, model_name: str, variable: str, layer: int, vals: np.ndarray, n_total: int) -> dict:
    return {
        "scenario_id": cfg.scenario_id,
        "model": model_name,
        "variable": variable,
        "layer_index": layer,
        "n_total": n_total,
        "n_observed": int(vals.size),
        "p_observed": float(vals.size / n_total) if n_total else float("nan"),
        "mean": float(np.mean(vals)) if vals.size else float("nan"),
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
        "q10": _q(vals, 0.10),
        "q25": _q(vals, 0.25),
        "median": _q(vals, 0.50),
        "q75": _q(vals, 0.75),
        "q90": _q(vals, 0.90),
        "min": float(np.min(vals)) if vals.size else float("nan"),
        "max": float(np.max(vals)) if vals.size else float("nan"),
    }


def _comparison_record(cfg: ExperimentConfig, variable: str, layer: int, arrivals_df, column: str) -> dict:
    micro = _finite(arrivals_df.loc[(arrivals_df["model"] == "Micro") & (arrivals_df["layer_index"] == layer), column])
    mm = _finite(arrivals_df.loc[(arrivals_df["model"] == "MicroMacro") & (arrivals_df["layer_index"] == layer), column])
    return _comparison_values(cfg, variable, layer, micro, mm)


def _comparison_from_wide(cfg: ExperimentConfig, variable: str, layer: int, wide_df, column: str) -> dict:
    micro = _finite(wide_df.loc[wide_df["model"] == "Micro", column])
    mm = _finite(wide_df.loc[wide_df["model"] == "MicroMacro", column])
    return _comparison_values(cfg, variable, layer, micro, mm)


def _comparison_values(cfg: ExperimentConfig, variable: str, layer: int, micro: np.ndarray, mm: np.ndarray) -> dict:
    w1 = _w1_distance(micro, mm)
    micro_mean = float(np.mean(micro)) if micro.size else float("nan")
    mean_shift = float(np.mean(mm) - np.mean(micro)) if micro.size and mm.size else float("nan")
    return {
        "scenario_id": cfg.scenario_id,
        "variable": variable,
        "layer_index": layer,
        "n_micro": int(micro.size),
        "n_micromacro": int(mm.size),
        "wasserstein_w1": w1,
        "relative_w1": w1 / micro_mean if math.isfinite(w1) and math.isfinite(micro_mean) and micro_mean != 0.0 else float("nan"),
        "ks_distance": _ks_distance(micro, mm),
        "micro_mean": micro_mean,
        "micromacro_mean": float(np.mean(mm)) if mm.size else float("nan"),
        "mean_shift": mean_shift,
        "relative_mean_shift": mean_shift / micro_mean if math.isfinite(mean_shift) and math.isfinite(micro_mean) and micro_mean != 0.0 else float("nan"),
        "median_shift": _q(mm, 0.50) - _q(micro, 0.50) if micro.size and mm.size else float("nan"),
        "q90_shift": _q(mm, 0.90) - _q(micro, 0.90) if micro.size and mm.size else float("nan"),
    }


def _paired_diff(left: object, right: object) -> float:
    try:
        left_f = float(left)
        right_f = float(right)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(left_f) or not math.isfinite(right_f):
        return float("nan")
    return left_f - right_f


def _write_cdf_points(wide_df, out_dir: Path) -> None:
    pd = _require_pandas()
    rows: List[dict] = []
    for model_name, mdf in wide_df.groupby("model"):
        vals = _finite(mdf["T_last"])
        if vals.size == 0:
            continue
        xs, ys = _empirical_cdf(vals)
        rows.extend({"model": model_name, "variable": "T_last", "x": float(x), "cdf": float(y)} for x, y in zip(xs, ys))
    pd.DataFrame(rows).to_csv(out_dir / "tlast_cdf_points.csv", index=False)


def _write_density_histograms(arrivals_df, wide_df, selected_deltas: Sequence[int], bins: int, out_dir: Path) -> None:
    pd = _require_pandas()
    rows: List[dict] = []
    for model_name, mdf in wide_df.groupby("model"):
        vals = _finite(mdf["T_last"])
        rows.extend(_hist_rows(model_name, "T_last", None, vals, bins))
    for delay_idx in selected_deltas:
        for model_name, mdf in arrivals_df[arrivals_df["layer_index"] == delay_idx].groupby("model"):
            vals = _finite(mdf["Delta_T_i"])
            rows.extend(_hist_rows(model_name, "Delta_T_i", delay_idx, vals, bins))
    pd.DataFrame(rows).to_csv(out_dir / "density_histograms.csv", index=False)


def _hist_rows(model_name: str, variable: str, layer: Optional[int], vals: np.ndarray, bins: int) -> List[dict]:
    if vals.size == 0:
        return []
    hist, edges = np.histogram(vals, bins=max(2, bins), density=True)
    return [
        {
            "model": model_name,
            "variable": variable,
            "layer_index": layer,
            "bin_left": float(edges[i]),
            "bin_right": float(edges[i + 1]),
            "density": float(hist[i]),
        }
        for i in range(hist.size)
    ]


def _plot_tlast_cdf(wide_df, plot_dir: Path, run_tag: str) -> Path:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Micro": "#2f6f9f", "MicroMacro": "#c95f36"}
    for model_name in ("Micro", "MicroMacro"):
        vals = _finite(wide_df.loc[wide_df["model"] == model_name, "T_last"])
        if vals.size == 0:
            continue
        x_grid = np.linspace(float(np.min(vals)), float(np.max(vals)), 240)
        lo, hi = _bootstrap_ecdf_band(vals, x_grid)
        ax.fill_between(x_grid, lo, hi, color=colors[model_name], alpha=0.16, linewidth=0)
        xs, ys = _empirical_cdf(vals)
        ax.step(xs, ys, where="post", label=model_name, color=colors[model_name], linewidth=2.0)
    ax.set_xlabel(r"$T_{\mathrm{last}}$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Arrival time of the last community")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = plot_dir / f"chain_cliques_tlast_cdf_{run_tag}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _bootstrap_ecdf_band(vals: np.ndarray, x_grid: np.ndarray, n_boot: int = 500, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    if vals.size <= 1:
        ecdf = np.searchsorted(np.sort(vals), x_grid, side="right") / max(1, vals.size)
        return ecdf, ecdf
    rng = np.random.default_rng(12345)
    sorted_boot = np.sort(rng.choice(vals, size=(n_boot, vals.size), replace=True), axis=1)
    cdfs = np.vstack([np.searchsorted(sample, x_grid, side="right") for sample in sorted_boot]) / float(vals.size)
    return np.quantile(cdfs, alpha / 2.0, axis=0), np.quantile(cdfs, 1.0 - alpha / 2.0, axis=0)


def _plot_selected_delays(arrivals_df, selected_deltas: Sequence[int], plot_dir: Path, run_tag: str, bins: int) -> Path:
    plt = _require_matplotlib()
    n_panels = max(1, len(selected_deltas))
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.0), squeeze=False)
    colors = {"Micro": "#2f6f9f", "MicroMacro": "#c95f36"}
    for ax, delay_idx in zip(axes[0], selected_deltas):
        for model_name in ("Micro", "MicroMacro"):
            vals = _finite(
                arrivals_df.loc[
                    (arrivals_df["model"] == model_name) & (arrivals_df["layer_index"] == delay_idx),
                    "Delta_T_i",
                ]
            )
            if vals.size == 0:
                continue
            ax.hist(vals, bins=max(2, bins), density=True, histtype="step", linewidth=2.0, color=colors[model_name], label=model_name)
        ax.set_title(rf"$\Delta T_{delay_idx}$")
        ax.set_xlabel("Delay")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
    axes[0][0].legend()
    fig.tight_layout()
    out = plot_dir / f"chain_cliques_selected_delay_density_{run_tag}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _plot_error_growth(comparison_df, paired_df, plot_dir: Path, run_tag: str) -> Path:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2))

    t_metrics = comparison_df[comparison_df["variable"] == "T_i"].sort_values("layer_index")
    axes[0].plot(t_metrics["layer_index"], t_metrics["wasserstein_w1"], marker="o", color="#2f6f9f")
    axes[0].set_title(r"$W_1(T_i)$")
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Time")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_metrics["layer_index"], t_metrics["relative_w1"], marker="o", color="#2f6f9f")
    axes[1].set_title(r"Relative $W_1(T_i)$")
    axes[1].set_xlabel("Layer index")
    axes[1].set_ylabel(r"$W_1 / E[T_i^{Micro}]$")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_metrics["layer_index"], t_metrics["mean_shift"], marker="o", color="#c95f36")
    axes[2].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[2].set_title(r"Mean shift in $T_i$")
    axes[2].set_xlabel("Layer index")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t_metrics["layer_index"], t_metrics["median_shift"], marker="o", label=r"$\Delta q_{0.5}$", color="#3b7d4a")
    axes[3].plot(t_metrics["layer_index"], t_metrics["q90_shift"], marker="s", label=r"$\Delta q_{0.9}$", color="#7f5aa2")
    axes[3].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[3].set_title("Quantile shifts")
    axes[3].set_xlabel("Layer index")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()

    if not paired_df.empty:
        paired_summary = (
            paired_df.groupby("layer_index")["error_T_i"]
            .agg(mean_bias="mean", mae=lambda x: np.nanmean(np.abs(x)), rmse=lambda x: math.sqrt(np.nanmean(np.square(x))))
            .reset_index()
        )
        paired_summary.to_csv(plot_dir / f"chain_cliques_paired_error_summary_{run_tag}.csv", index=False)

    fig.tight_layout()
    out = plot_dir / f"chain_cliques_error_growth_{run_tag}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _plot_paired_arrival_error(paired_df, plot_dir: Path, run_tag: str) -> Optional[Path]:
    pd = _require_pandas()
    plt = _require_matplotlib()
    rows = []
    if paired_df.empty:
        return None
    for layer, grp in paired_df.groupby("layer_index"):
        vals = _finite(grp["error_T_i"])
        if vals.size == 0:
            continue
        rows.append(
            {
                "layer_index": int(layer),
                "n": int(vals.size),
                "q25": _q(vals, 0.25),
                "median": _q(vals, 0.50),
                "q75": _q(vals, 0.75),
                "q10": _q(vals, 0.10),
                "q90": _q(vals, 0.90),
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("layer_index")
    df.to_csv(plot_dir / f"chain_cliques_paired_arrival_error_{run_tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    x = df["layer_index"].to_numpy(dtype=float)
    ax.fill_between(x, df["q25"], df["q75"], color="#c95f36", alpha=0.24, label="25-75%")
    ax.plot(x, df["median"], marker="o", color="#c95f36", linewidth=2.0, label="median")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$T_i^{MicroMacro} - T_i^{Micro}$")
    ax.set_title("Paired arrival-time error from simulation start")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = plot_dir / f"chain_cliques_paired_arrival_error_{run_tag}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _aggregate_total_curve(df):
    return df.groupby("time", as_index=False)[["I"]].sum().sort_values("time")


def _load_trajectory_csvs(trajectory_root: Path, model_name: str):
    pd = _require_pandas()
    curves = {}
    for path in sorted(trajectory_root.glob(f"run_*/{model_name}.csv")):
        rep_id = int(path.parent.name.split("_")[1])
        df = pd.read_csv(path)
        curves[rep_id] = _aggregate_total_curve(df)
    return curves


def _trajectory_stack(curves: Dict[int, object], t_grid: np.ndarray) -> np.ndarray:
    stack = []
    for curve in curves.values():
        times = curve["time"].to_numpy(dtype=float)
        values = curve["I"].to_numpy(dtype=float)
        stack.append(np.interp(t_grid, times, values, left=values[0], right=values[-1]))
    return np.vstack(stack) if stack else np.empty((0, t_grid.size))


def _plot_total_trajectory_bands(trajectory_root: Path, plot_dir: Path, run_tag: str, cfg: ExperimentConfig) -> Optional[Path]:
    plt = _require_matplotlib()
    curves = {name: _load_trajectory_csvs(trajectory_root, name) for name in ("Micro", "MicroMacro")}
    if not curves["Micro"] or not curves["MicroMacro"]:
        return None
    t_grid = np.arange(0.0, cfg.T_end + cfg.tau_micro, cfg.tau_micro)
    colors = {"Micro": "#2f6f9f", "MicroMacro": "#c95f36"}
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for model_name in ("Micro", "MicroMacro"):
        stack = _trajectory_stack(curves[model_name], t_grid)
        if stack.size == 0:
            continue
        q25 = np.quantile(stack, 0.25, axis=0)
        median = np.quantile(stack, 0.50, axis=0)
        q75 = np.quantile(stack, 0.75, axis=0)
        color = colors[model_name]
        ax.fill_between(t_grid, q25, q75, color=color, alpha=0.22, linewidth=0, label=f"{model_name} 25-75%")
        ax.plot(t_grid, median, color=color, linewidth=2.4, label=f"{model_name} median")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Total infected $I_{tot}(t)$")
    ax.set_title("Total infected over time")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = plot_dir / f"chain_cliques_total_infected_iqr_{run_tag}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _representative_replication(wide_df) -> Optional[int]:
    micro = wide_df[(wide_df["model"] == "Micro") & np.isfinite(wide_df["T_last"])].copy()
    mm = wide_df[(wide_df["model"] == "MicroMacro") & np.isfinite(wide_df["T_last"])].copy()
    common = sorted(set(micro["replication_id"]) & set(mm["replication_id"]))
    if not common:
        return None
    micro = micro[micro["replication_id"].isin(common)].copy()
    if micro.empty:
        return None
    median_t = float(np.median(micro["T_last"].to_numpy(dtype=float)))
    micro["abs_to_median"] = np.abs(micro["T_last"].to_numpy(dtype=float) - median_t)
    return int(micro.sort_values(["abs_to_median", "replication_id"]).iloc[0]["replication_id"]) - 1


def _plot_per_community_pair(trajectory_root: Path, plot_dir: Path, run_tag: str, rep_zero: int) -> Optional[Path]:
    pd = _require_pandas()
    plt = _require_matplotlib()
    paths = {
        "Micro": trajectory_root / f"run_{rep_zero:03d}" / "Micro.csv",
        "MicroMacro": trajectory_root / f"run_{rep_zero:03d}" / "MicroMacro.csv",
    }
    if not all(path.exists() for path in paths.values()):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, model_name in zip(axes, ("Micro", "MicroMacro")):
        df = pd.read_csv(paths[model_name])
        for community, grp in df.groupby("community"):
            grp = grp.sort_values("time")
            ax.plot(grp["time"], grp["I"], linewidth=1.2, label=f"{int(community)}")
        ax.set_title(model_name)
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel(r"Community infected $I_i(t)$")
    axes[1].legend(title="Community", fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"Per-community infected curves, paired run {rep_zero:03d}")
    fig.tight_layout()
    out = plot_dir / f"chain_cliques_per_community_run_{rep_zero:03d}_{run_tag}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def _default_selected_deltas(L: int) -> List[int]:
    candidates = [0, max(0, (L - 2) // 2), max(0, L - 2)]
    return sorted(set(c for c in candidates if 0 <= c <= L - 2))


def run_experiment(
    cfg: ExperimentConfig,
    run_tag: str,
    selected_deltas: Sequence[int],
    bins: int,
    save_trajectories: bool = False,
) -> dict:
    pd = _require_pandas()
    out_dir = _data_dir(cfg, run_tag)
    plot_dir = _plot_dir(cfg)
    trajectory_root = _trajectory_dir(cfg, run_tag) if save_trajectories else None
    out_dir.mkdir(parents=True, exist_ok=False)
    plot_dir.mkdir(parents=True, exist_ok=True)
    if trajectory_root is not None:
        trajectory_root.mkdir(parents=True, exist_ok=False)

    metadata_rows: List[dict] = []
    arrival_rows: List[dict] = []
    t_start = time.perf_counter()
    for replication_id in range(1, cfg.n_runs + 1):
        rep_meta, rep_arrivals = _run_one_replication(cfg, replication_id, trajectory_root=trajectory_root)
        metadata_rows.extend(rep_meta)
        arrival_rows.extend(rep_arrivals)
        if replication_id == 1 or replication_id % max(1, min(25, cfg.n_runs // 4 or 1)) == 0:
            print(f"[chain-arrivals] completed replication {replication_id}/{cfg.n_runs}", flush=True)

    total_runtime = time.perf_counter() - t_start
    metadata_df = pd.DataFrame(metadata_rows)
    arrivals_df = pd.DataFrame(arrival_rows)
    wide_df = _wide_table(arrivals_df)
    summary_df, comparison_df, paired_df = _summary_tables(arrivals_df, wide_df, cfg)
    runtime_df = _runtime_summary(metadata_df)

    parameters = {
        **asdict(cfg),
        "scenario_id": cfg.scenario_id,
        "run_tag": run_tag,
        "selected_deltas": list(selected_deltas),
        "total_wall_runtime_seconds": total_runtime,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (out_dir / "parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    metadata_df.to_csv(out_dir / "replication_metadata.csv", index=False)
    arrivals_df.to_csv(out_dir / "per_replication_arrivals.csv", index=False)
    wide_df.to_csv(out_dir / "arrival_times_wide.csv", index=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    comparison_df.to_csv(out_dir / "model_comparison_metrics.csv", index=False)
    paired_df.to_csv(out_dir / "paired_error_growth.csv", index=False)
    runtime_df.to_csv(out_dir / "runtime_summary.csv", index=False)
    _write_cdf_points(wide_df, out_dir)
    _write_density_histograms(arrivals_df, wide_df, selected_deltas, bins, out_dir)

    plot_paths = [
        _plot_tlast_cdf(wide_df, plot_dir, run_tag),
        _plot_selected_delays(arrivals_df, selected_deltas, plot_dir, run_tag, bins),
        _plot_error_growth(comparison_df, paired_df, plot_dir, run_tag),
    ]
    paired_error_plot = _plot_paired_arrival_error(paired_df, plot_dir, run_tag)
    if paired_error_plot is not None:
        plot_paths.append(paired_error_plot)
    if trajectory_root is not None:
        summary_dir = trajectory_root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(summary_dir / "replication_metadata.csv", index=False)
        arrivals_df.to_csv(summary_dir / "arrival_metrics.csv", index=False)
        comparison_df.to_csv(summary_dir / "comparison_metrics.csv", index=False)
        runtime_df.to_csv(summary_dir / "runtime_summary.csv", index=False)
        total_plot = _plot_total_trajectory_bands(trajectory_root, plot_dir, run_tag, cfg)
        rep_zero = _representative_replication(wide_df)
        community_plot = _plot_per_community_pair(trajectory_root, plot_dir, run_tag, rep_zero) if rep_zero is not None else None
        plot_paths.extend(path for path in (total_plot, community_plot) if path is not None)
    (out_dir / "plot_paths.json").write_text(
        json.dumps({"plots": [str(path.relative_to(REPO_ROOT)) for path in plot_paths]}, indent=2),
        encoding="utf-8",
    )

    return {
        "output_dir": str(out_dir),
        "plot_dir": str(plot_dir),
        "trajectory_dir": str(trajectory_root) if trajectory_root is not None else None,
        "plot_paths": [str(path) for path in plot_paths],
        "total_runtime_seconds": total_runtime,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stochastic Micro vs MicroMacro arrival-time comparison on a chain of cliques.")
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--model", default="SIR")
    parser.add_argument("--tau-micro", type=float, default=1.0)
    parser.add_argument("--macro-T", type=float, default=1.0)
    parser.add_argument("--T-end", type=float, default=1000.0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--edge-prob", type=float, default=0.031)
    parser.add_argument("--macro-graph-type", default="chain")
    parser.add_argument("--micro-graph-type", default="complete")
    parser.add_argument("--initial-community", type=int, default=0)
    parser.add_argument("--initial-node", type=str, default="0")
    parser.add_argument("--fixed-network", action="store_true", help="Use one shared network seed instead of network_seed + replication_id - 1.")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--selected-deltas", nargs="*", type=int, default=None)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--save-trajectories", action="store_true", help="Save run_XXX/Micro.csv and run_XXX/MicroMacro.csv files.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    initial_node = None if str(args.initial_node).strip().lower() in {"none", "null", ""} else int(args.initial_node)
    cfg = ExperimentConfig(
        L=args.L,
        n=args.n,
        k=args.k,
        beta=args.beta,
        gamma=args.gamma,
        model=args.model,
        tau_micro=args.tau_micro,
        macro_T=args.macro_T,
        T_end=args.T_end,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        network_seed=args.network_seed,
        regenerate_network_each_run=not args.fixed_network,
        macro_graph_type=args.macro_graph_type,
        micro_graph_type=args.micro_graph_type,
        edge_prob=args.edge_prob,
        initial_community=args.initial_community,
        initial_node=initial_node,
    )
    run_tag = args.run_tag or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    selected_deltas = args.selected_deltas if args.selected_deltas is not None else _default_selected_deltas(cfg.L)
    result = run_experiment(cfg, run_tag, selected_deltas, args.bins, save_trajectories=args.save_trajectories)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
