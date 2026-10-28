#!/usr/bin/env python3
"""Compare shared and per-community RNG routing in Production MicroMacro.

The default invocation runs 100 paired replications for two six-community
chain topologies and writes all outputs to output/rng_topology_comparison_l6.
No production source files are modified; the independent-RNG behavior is
installed only inside a short-lived context manager in each worker process.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.engines.micro_model import MicroModel
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.orchestrators.micromacro import Orchestrator
from two_layer_ctmc.simulators.micro import MicroSimulator
from two_layer_ctmc.simulators.micromacro import MicroMacroSimulator


L = 6
COMMUNITY_SIZE = 100
INTER_LINKS = 5
BETA_IN = 0.003
BETA_OUT = 0.003
GAMMA = 0.0
MODEL = 2  # SI is SIR with gamma=0 in the production implementation.
TAU_MICRO = 1.0
MACRO_T = 1.0
INITIAL_NODE = 0
RANDOM_CONNECTED_EDGE_PROB = 6.0 / (COMMUNITY_SIZE - 1)
MODEL_ORDER = ("Full Micro", "Production shared RNG", "Production independent RNG")
COLORS = {
    "Full Micro": "#2ca02c",
    "Production shared RNG": "#1f77b4",
    "Production independent RNG": "#d62728",
}


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    micro_graph_type: str
    initial_horizon: float
    retry_horizon: float | None = None
    edge_prob: float = RANDOM_CONNECTED_EDGE_PROB


SCENARIOS = (
    Scenario("clique", "chain of cliques", "complete", 4000.0),
    Scenario(
        "random_connected",
        "chain of random_connected graphs",
        "random_connected",
        8000.0,
        retry_horizon=12000.0,
    ),
)


def _copy_rng(rng: Any) -> Any:
    if rng is random:
        return random
    copied = random.Random()
    copied.setstate(rng.getstate())
    return copied


def _patched_clone(self: MicroModel) -> MicroModel:
    """Factorial-runner clone logic, with RNG handling selected by context."""
    independent = bool(getattr(_patched_clone, "independent", False))
    clone_rng = _copy_rng(self.rng) if independent else self.rng
    model = MicroModel(self.infection_rate, self.recovery_rate, self.model, rng=clone_rng)
    model.current_time = self.current_time
    model.G = nx.Graph()
    model.G.add_nodes_from(self.G.nodes(data=True))
    model.infected_nodes = list(self.infected_nodes)
    model.total_infection_rate = self.total_infection_rate
    model.total_recovery_rate = self.total_recovery_rate
    model.S_count = self.S_count
    model.I_count = self.I_count
    model.R_count = self.R_count
    model._neighbors = self._neighbors
    model._invalidate_cache()  # original cache behavior: discard
    return model


def _patched_refresh(self: MicroModel, src: MicroModel) -> None:
    independent = bool(getattr(_patched_clone, "independent", False))
    self.current_time = src.current_time
    self.infected_nodes = list(src.infected_nodes)
    self.total_infection_rate = src.total_infection_rate
    self.total_recovery_rate = src.total_recovery_rate
    self.S_count = src.S_count
    self.I_count = src.I_count
    self.R_count = src.R_count
    self._neighbors = src._neighbors
    for node in self.G.nodes():
        dst = self.G.nodes[node]
        source = src.G.nodes[node]
        dst["infected"] = source["infected"]
        dst["recovered"] = source["recovered"]
        dst["sum_of_weights_i"] = source["sum_of_weights_i"]
    if independent:
        self.rng.setstate(src.rng.getstate())
    self._invalidate_cache()  # original cache behavior: discard


@contextmanager
def _rng_patch(independent: bool) -> Iterator[None]:
    """Temporarily install only the RNG-routing mechanism from the factorial runner."""
    original_clone = MicroModel.clone
    original_refresh = MicroModel.refresh_from
    _patched_clone.independent = independent  # type: ignore[attr-defined]
    MicroModel.clone = _patched_clone  # type: ignore[method-assign]
    MicroModel.refresh_from = _patched_refresh  # type: ignore[method-assign]
    try:
        yield
    finally:
        MicroModel.clone = original_clone  # type: ignore[method-assign]
        MicroModel.refresh_from = original_refresh  # type: ignore[method-assign]


def _build_network(scenario: Scenario, network_seed: int):
    return generate_two_scale_network(
        n_communities=L,
        community_size=COMMUNITY_SIZE,
        inter_links=INTER_LINKS,
        seed=network_seed,
        macro_graph_type="chain",
        micro_graph_type=scenario.micro_graph_type,
        edge_prob=scenario.edge_prob,
    )


def _run_production_shared(
    micro_graphs: list[nx.Graph], full_graph: nx.Graph, W: np.ndarray, seed: int, horizon: float
):
    simulator = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=BETA_IN,
        gamma=GAMMA,
        beta_macro=BETA_OUT,
        tau_micro=TAU_MICRO,
        T_end=horizon,
        macro_T=MACRO_T,
        model=MODEL,
        full_graph=full_graph,
    )
    return simulator.run(seed=seed, initial_community=0, initial_node=INITIAL_NODE)


def _run_experimental_micromacro(
    micro_graphs: list[nx.Graph],
    full_graph: nx.Graph,
    W: np.ndarray,
    seed: int,
    horizon: float,
    *,
    independent: bool,
):
    """Production orchestration with only per-MicroModel RNG routing changed."""
    random.seed(seed)
    np.random.seed(seed)
    with _rng_patch(independent=independent):
        orchestrator = Orchestrator(
            W=W,
            micro_graphs=micro_graphs,
            beta_micro=BETA_IN,
            gamma=GAMMA,
            beta_macro=BETA_OUT,
            tau_micro=TAU_MICRO,
            T_end=horizon,
            macro_T=MACRO_T,
            model=MODEL,
            community_sizes=[g.number_of_nodes() for g in micro_graphs],
            alphas=full_graph.graph.get("alphas"),
            full_graph=full_graph,
        )
        if independent:
            for community, model in enumerate(orchestrator.micro_models):
                model.rng = random.Random(seed + 10_000 + 1009 * community)
        orchestrator.micro_models[0]._infect_node(INITIAL_NODE)
        orchestrator.event_log.append(
            {
                "time": 0.0,
                "wait_time": 0.0,
                "event_type": "seed",
                "mode": "seed",
                "community": 0,
                "node": INITIAL_NODE,
                "src": None,
            }
        )
        return orchestrator.run()


def _regression_network(seed: int):
    return generate_two_scale_network(
        n_communities=3,
        community_size=12,
        inter_links=2,
        seed=seed,
        macro_graph_type="chain",
        micro_graph_type="complete",
        edge_prob=0.2,
    )


def regression_gate(control_seeds: Sequence[int] = (7101, 7102, 7103)) -> tuple[bool, str]:
    """Require exact pathwise equality between wrapper-shared and Production."""
    global BETA_IN, BETA_OUT
    saved_beta_in, saved_beta_out = BETA_IN, BETA_OUT
    # Higher rates keep this small gate short while exercising macro transfers.
    BETA_IN = BETA_OUT = 0.03
    try:
        for seed in control_seeds:
            micro_graphs, full_graph, W = _regression_network(seed + 100)
            production = _run_production_shared(micro_graphs, full_graph, W, seed, 80.0)
            experimental = _run_experimental_micromacro(
                micro_graphs, full_graph, W, seed, 80.0, independent=False
            )
            if production != experimental:
                names = ("times", "I_total", "logs", "event_log")
                mismatches = [name for name, left, right in zip(names, production, experimental) if left != right]
                return False, f"seed={seed}; mismatched components: {', '.join(mismatches)}"
        return True, f"exact pathwise match for seeds {list(control_seeds)}"
    finally:
        BETA_IN, BETA_OUT = saved_beta_in, saved_beta_out


def _arrivals_from_full(result: dict[str, Any], horizon: float) -> list[float]:
    arrivals = [math.nan] * L
    for timing in result.get("community_timings", []):
        community = int(timing["community"])
        value = timing.get("first_infection_time")
        if value is not None and float(value) <= horizon:
            arrivals[community] = float(value)
    arrivals[0] = 0.0
    return arrivals


def _arrivals_from_events(events: Sequence[dict[str, Any]], horizon: float) -> list[float]:
    arrivals = [math.nan] * L
    for event in sorted(events, key=lambda item: float(item.get("time", math.inf))):
        if str(event.get("event_type", "")).lower() not in {"seed", "infection", "transfer"}:
            continue
        community = event.get("community")
        event_time = float(event.get("time", math.inf))
        if community is None or event_time > horizon:
            continue
        community = int(community)
        if 0 <= community < L and not math.isfinite(arrivals[community]):
            arrivals[community] = event_time
    arrivals[0] = 0.0
    return arrivals


def _record(
    scenario: Scenario,
    replication: int,
    network_seed: int,
    run_seed: int,
    horizon: float,
    model: str,
    arrivals: Sequence[float],
    runtime: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "topology": scenario.key,
        "scenario": scenario.label,
        "model": model,
        "replication": replication,
        "network_seed": network_seed,
        "run_seed": run_seed,
        "initial_node": INITIAL_NODE,
        "T_end": horizon,
        "completion": bool(math.isfinite(arrivals[-1])),
        "runtime": runtime,
    }
    for community in range(1, L):
        row[f"T_{community}"] = arrivals[community]
        left, right = arrivals[community - 1], arrivals[community]
        row[f"delta_{community}"] = right - left if math.isfinite(left) and math.isfinite(right) else math.nan
    row["T_last"] = arrivals[-1]
    return row


def _run_replication(task: tuple[dict[str, Any], int, int, int, float]) -> list[dict[str, Any]]:
    scenario_dict, replication, network_seed, run_seed, horizon = task
    scenario = Scenario(**scenario_dict)
    micro_graphs, full_graph, W = _build_network(scenario, network_seed)
    comm_nodes = [list(graph.nodes()) for graph in micro_graphs]
    rows: list[dict[str, Any]] = []

    started = time.perf_counter()
    full = MicroSimulator(full_graph, comm_nodes, BETA_IN, GAMMA, model=MODEL).run(
        T_end=horizon,
        dt_out=TAU_MICRO,
        initial_node=INITIAL_NODE,
        rng=random.Random(run_seed),
    )
    rows.append(
        _record(
            scenario,
            replication,
            network_seed,
            run_seed,
            horizon,
            MODEL_ORDER[0],
            _arrivals_from_full(full, horizon),
            time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    shared = _run_production_shared(micro_graphs, full_graph, W, run_seed, horizon)
    rows.append(
        _record(
            scenario,
            replication,
            network_seed,
            run_seed,
            horizon,
            MODEL_ORDER[1],
            _arrivals_from_events(shared[3], horizon),
            time.perf_counter() - started,
        )
    )

    started = time.perf_counter()
    independent = _run_experimental_micromacro(
        micro_graphs, full_graph, W, run_seed, horizon, independent=True
    )
    rows.append(
        _record(
            scenario,
            replication,
            network_seed,
            run_seed,
            horizon,
            MODEL_ORDER[2],
            _arrivals_from_events(independent[3], horizon),
            time.perf_counter() - started,
        )
    )
    return rows


def run_scenario(
    scenario: Scenario,
    replications: int,
    workers: int,
    base_seed: int,
    network_seed_base: int,
    horizon: float,
) -> pd.DataFrame:
    tasks = [
        (asdict(scenario), rep, network_seed_base + rep - 1, base_seed + rep - 1, horizon)
        for rep in range(1, replications + 1)
    ]
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if workers == 1:
        for completed, task in enumerate(tasks, 1):
            rows.extend(_run_replication(task))
            if completed == 1 or completed % max(1, replications // 10) == 0:
                print(f"  {scenario.key}: {completed}/{replications} replications", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_replication, task) for task in tasks]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                rows.extend(future.result())
                if completed == 1 or completed % max(1, replications // 10) == 0:
                    elapsed = time.perf_counter() - started
                    print(f"  {scenario.key}: {completed}/{replications} replications ({elapsed:.1f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame["_model_order"] = frame["model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    return frame.sort_values(["replication", "_model_order"]).drop(columns="_model_order").reset_index(drop=True)


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return math.nan
    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except ImportError:
        grid = np.unique(np.concatenate([a, b]))
        if len(grid) <= 1:
            return 0.0
        sa, sb = np.sort(a), np.sort(b)
        fa = np.searchsorted(sa, grid, side="right") / len(sa)
        fb = np.searchsorted(sb, grid, side="right") / len(sb)
        return float(np.sum(np.abs(fa[:-1] - fb[:-1]) * np.diff(grid)))


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return math.nan
    try:
        from scipy.stats import ks_2samp

        return float(ks_2samp(a, b, method="auto").statistic)
    except ImportError:
        grid = np.unique(np.concatenate([a, b]))
        sa, sb = np.sort(a), np.sort(b)
        fa = np.searchsorted(sa, grid, side="right") / len(sa)
        fb = np.searchsorted(sb, grid, side="right") / len(sb)
        return float(np.max(np.abs(fa - fb)))


def make_summary(per_replication: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_config in SCENARIOS:
        for model in MODEL_ORDER:
            group = per_replication[
                (per_replication["topology"] == scenario_config.key) & (per_replication["model"] == model)
            ]
            values = _finite(group["T_last"])
            q1 = float(np.quantile(values, 0.25)) if len(values) else math.nan
            q3 = float(np.quantile(values, 0.75)) if len(values) else math.nan
            rows.append(
                {
                    "topology": scenario_config.key,
                    "scenario": scenario_config.label,
                    "model": model,
                    "N": len(group),
                    "N_completed": len(values),
                    "completion_rate": float(group["completion"].mean()),
                    "median_T_last": float(np.median(values)) if len(values) else math.nan,
                    "IQR": q3 - q1,
                    "mean_T_last": float(np.mean(values)) if len(values) else math.nan,
                    "T_end": float(group["T_end"].iloc[0]),
                    "mean_runtime": float(group["runtime"].mean()),
                }
            )
    return pd.DataFrame(rows)


def make_arrival_long(per_replication: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in per_replication.to_dict("records"):
        base = {key: record[key] for key in (
            "topology", "scenario", "model", "replication", "network_seed", "run_seed", "T_end", "completion"
        )}
        for community in range(L):
            arrival = 0.0 if community == 0 else record[f"T_{community}"]
            delta = math.nan if community == 0 else record[f"delta_{community}"]
            rows.append({**base, "community": community, "arrival_time": arrival, "delta_i": delta})
    return pd.DataFrame(rows)


def make_distances(per_replication: pd.DataFrame) -> pd.DataFrame:
    rows = []
    observables = [f"T_{community}" for community in range(1, L)] + [
        f"delta_{community}" for community in range(1, L)
    ] + ["T_last"]
    comparisons = (
        (MODEL_ORDER[0], MODEL_ORDER[1], "shared vs Full"),
        (MODEL_ORDER[0], MODEL_ORDER[2], "independent vs Full"),
        (MODEL_ORDER[1], MODEL_ORDER[2], "shared vs independent"),
    )
    for topology, topology_df in per_replication.groupby("topology", sort=False):
        for observable in observables:
            for model_a, model_b, comparison in comparisons:
                a = _finite(topology_df.loc[topology_df["model"] == model_a, observable])
                b = _finite(topology_df.loc[topology_df["model"] == model_b, observable])
                rows.append(
                    {
                        "topology": topology,
                        "observable": observable,
                        "comparison": comparison,
                        "model_a": model_a,
                        "model_b": model_b,
                        "N_a_observed": len(a),
                        "N_b_observed": len(b),
                        "wasserstein_1": _wasserstein(a, b),
                        "ks_distance": _ks(a, b),
                        "censoring_policy": "observed arrivals only; horizon is never imputed",
                    }
                )
    return pd.DataFrame(rows)


def _configure_plotting():
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1, dtype=float) / len(x)


def plot_ecdf(per_replication: pd.DataFrame, topology: str, path: Path) -> None:
    plt = _configure_plotting()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    frame = per_replication[per_replication["topology"] == topology]
    for model in MODEL_ORDER:
        values = _finite(frame.loc[frame["model"] == model, "T_last"])
        if len(values):
            x, y = _ecdf(values)
            ax.step(x, y, where="post", label=f"{model} (n={len(values)})", color=COLORS[model], linewidth=2)
    ax.set(title=f"T_last ECDF — {topology}", xlabel="T_last", ylabel="Empirical CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_density(per_replication: pd.DataFrame, topology: str, path: Path) -> None:
    plt = _configure_plotting()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    frame = per_replication[per_replication["topology"] == topology]
    for model in MODEL_ORDER:
        values = _finite(frame.loc[frame["model"] == model, "T_last"])
        if not len(values):
            continue
        try:
            from scipy.stats import gaussian_kde

            if len(values) > 1 and float(np.std(values)) > 0:
                x = np.linspace(float(values.min()), float(values.max()), 300)
                y = gaussian_kde(values)(x)
                ax.plot(x, y, label=f"{model} (n={len(values)})", color=COLORS[model], linewidth=2)
            else:
                ax.hist(values, bins=1, density=True, alpha=0.25, label=model, color=COLORS[model])
        except (ImportError, np.linalg.LinAlgError):
            ax.hist(values, bins="auto", density=True, histtype="step", label=model, color=COLORS[model])
    ax.set(title=f"T_last density — {topology}", xlabel="T_last", ylabel="Density")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_arrival_by_layer(per_replication: pd.DataFrame, topology: str, path: Path) -> None:
    plt = _configure_plotting()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    frame = per_replication[per_replication["topology"] == topology]
    x = np.arange(L)
    for model in MODEL_ORDER:
        group = frame[frame["model"] == model]
        medians, q1s, q3s = [], [], []
        for community in range(L):
            values = np.zeros(len(group)) if community == 0 else _finite(group[f"T_{community}"])
            medians.append(float(np.median(values)) if len(values) else math.nan)
            q1s.append(float(np.quantile(values, 0.25)) if len(values) else math.nan)
            q3s.append(float(np.quantile(values, 0.75)) if len(values) else math.nan)
        medians_array = np.asarray(medians)
        ax.plot(x, medians_array, marker="o", label=model, color=COLORS[model], linewidth=2)
        ax.fill_between(x, q1s, q3s, color=COLORS[model], alpha=0.12)
    ax.set(title=f"Arrival accumulation — {topology}", xlabel="Community", ylabel="Arrival time (median; IQR band)")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _tlast_distance(distances: pd.DataFrame, topology: str, comparison: str) -> pd.Series:
    match = distances[
        (distances["topology"] == topology)
        & (distances["observable"] == "T_last")
        & (distances["comparison"] == comparison)
    ]
    return match.iloc[0]


def write_report(
    path: Path,
    per_replication: pd.DataFrame,
    summary: pd.DataFrame,
    distances: pd.DataFrame,
    regression_detail: str,
) -> None:
    lines = [
        "# RNG topology comparison (L=6)",
        "",
        f"Regression gate: **PASS** — {regression_detail}.",
        "",
        "All arrival summaries and distances use observed arrivals only. Censored runs remain missing; `T_end` is not substituted as an arrival time. The random-connected scenario was rerun at 12000 if any initial 8000-horizon run was censored.",
        "",
        "## T_last summary",
        "",
        "| topology | model | N | completion | median | IQR | mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['topology']} | {row['model']} | {row['N']} | {row['completion_rate']:.3f} | "
            f"{_fmt(row['median_T_last'])} | {_fmt(row['IQR'])} | {_fmt(row['mean_T_last'])} |"
        )

    lines.extend(["", "## Answers", ""])
    answer_rows = []
    for topology in (scenario.key for scenario in SCENARIOS):
        rng = _tlast_distance(distances, topology, "shared vs independent")
        shared = _tlast_distance(distances, topology, "shared vs Full")
        independent = _tlast_distance(distances, topology, "independent vs Full")
        answer_rows.append((topology, rng, shared, independent))
    rng_text = "; ".join(
        f"{top}: W1={_fmt(rng['wasserstein_1'])}, KS={_fmt(rng['ks_distance'])}" for top, rng, _, _ in answer_rows
    )
    lines.append(f"1. **How much shared RNG changes MicroMacro.** Shared-to-independent T_last distances are {rng_text}.")

    closer_parts = []
    for topology, _, shared, independent in answer_rows:
        sw1, iw1 = float(shared["wasserstein_1"]), float(independent["wasserstein_1"])
        if math.isfinite(sw1) and math.isfinite(iw1):
            verdict = "closer" if iw1 < sw1 else "farther" if iw1 > sw1 else "unchanged"
            closer_parts.append(f"{topology}: {verdict} (W1 {sw1:.3f} → {iw1:.3f})")
        else:
            closer_parts.append(f"{topology}: insufficient completed arrivals")
    lines.append(f"2. **Whether independent RNG brings Production closer to Full.** {'; '.join(closer_parts)}.")

    finite_effects = [(top, float(rng["wasserstein_1"])) for top, rng, _, _ in answer_rows if math.isfinite(float(rng["wasserstein_1"]))]
    if len(finite_effects) == 2:
        larger = max(finite_effects, key=lambda item: item[1])
        lines.append(
            f"3. **Topology dependence.** The larger shared-to-independent T_last W1 occurs for `{larger[0]}` ({larger[1]:.3f}); compare this with the full per-layer table in `distribution_distances.csv`."
        )
    else:
        lines.append("3. **Topology dependence.** Insufficient completed T_last samples for a two-topology comparison.")

    onset_parts = []
    for topology in (scenario.key for scenario in SCENARIOS):
        layer_rows = distances[
            (distances["topology"] == topology)
            & distances["observable"].isin([f"T_{i}" for i in range(1, L)])
            & (distances["comparison"] == "shared vs independent")
        ].copy()
        layer_rows["community"] = layer_rows["observable"].str.split("_").str[-1].astype(int)
        layer_rows = layer_rows.sort_values("community")
        values = layer_rows["wasserstein_1"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any() or float(np.nanmax(values)) <= 0:
            onset_parts.append(f"{topology}: no measurable onset")
            continue
        threshold = 0.2 * float(np.nanmax(values))
        onset = int(layer_rows.loc[finite & (values >= threshold), "community"].iloc[0])
        maximum = int(layer_rows.loc[layer_rows["wasserstein_1"].idxmax(), "community"])
        onset_parts.append(f"{topology}: starts at community {onset}, maximum at {maximum}")
    lines.append(
        "4. **Where the difference starts accumulating.** Using the first community whose shared-vs-independent arrival W1 reaches 20% of its topology-specific maximum: "
        + "; ".join(onset_parts)
        + ". This is a descriptive threshold, not a significance test."
    )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- W1 and KS are computed on observed values only. If completion is below 1, the corresponding distribution comparison is conditional on completion and may be selection-biased.",
            "- The experiment isolates RNG stream routing while retaining production discard-cache, normal re-import, macro scheduler, midpoint integration, aggregate hazard, and uniform placement behavior.",
            "- Network and stochastic seeds are paired by replication across all three models.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_final_table(summary: pd.DataFrame, distances: pd.DataFrame) -> None:
    print("\ntopology | model | completion | median T_last | IQR | W1 vs Full | KS vs Full")
    print("-" * 102)
    for row in summary.to_dict("records"):
        if row["model"] == MODEL_ORDER[0]:
            w1 = ks = 0.0
        else:
            comparison = "shared vs Full" if row["model"] == MODEL_ORDER[1] else "independent vs Full"
            metric = _tlast_distance(distances, row["topology"], comparison)
            w1, ks = metric["wasserstein_1"], metric["ks_distance"]
        print(
            f"{row['topology']} | {row['model']} | {row['completion_rate']:.3f} | "
            f"{_fmt(row['median_T_last'])} | {_fmt(row['IQR'])} | {_fmt(w1)} | {_fmt(ks)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--base-seed", type=int, default=1_610_000)
    parser.add_argument("--network-seed-base", type=int, default=1_710_000)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output" / "rng_topology_comparison_l6")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replications < 1:
        raise ValueError("--replications must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    passed, detail = regression_gate()
    print(f"Regression gate: {'PASS' if passed else 'FAIL'}")
    print("Scenario 1: chain of cliques")
    print("Scenario 2: chain of random_connected graphs")
    print(f"Replications: {args.replications}")
    if not passed:
        raise RuntimeError(f"Regression gate failed; experiment stopped: {detail}")

    scenario_frames = []
    for index, scenario in enumerate(SCENARIOS):
        network_seed_base = args.network_seed_base + index * 100_000
        run_seed_base = args.base_seed + index * 100_000
        print(f"Running {scenario.label} at T_end={scenario.initial_horizon:g}", flush=True)
        frame = run_scenario(
            scenario,
            args.replications,
            args.workers,
            run_seed_base,
            network_seed_base,
            scenario.initial_horizon,
        )
        if scenario.retry_horizon is not None and not bool(frame["completion"].all()):
            censored = int((~frame["completion"]).sum())
            print(
                f"  detected {censored} censored model-runs; repeating complete scenario at T_end={scenario.retry_horizon:g}",
                flush=True,
            )
            frame = run_scenario(
                scenario,
                args.replications,
                args.workers,
                run_seed_base,
                network_seed_base,
                scenario.retry_horizon,
            )
        scenario_frames.append(frame)

    per_replication = pd.concat(scenario_frames, ignore_index=True)
    arrival_by_community = make_arrival_long(per_replication)
    summary = make_summary(per_replication)
    distances = make_distances(per_replication)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_replication.to_csv(args.output_dir / "per_replication.csv", index=False)
    arrival_by_community.to_csv(args.output_dir / "arrival_by_community.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    distances.to_csv(args.output_dir / "distribution_distances.csv", index=False)
    for scenario in SCENARIOS:
        plot_ecdf(per_replication, scenario.key, args.output_dir / f"ecdf_{scenario.key}.png")
        plot_density(per_replication, scenario.key, args.output_dir / f"density_{scenario.key}.png")
        plot_arrival_by_layer(
            per_replication, scenario.key, args.output_dir / f"arrival_by_layer_{scenario.key}.png"
        )
    write_report(args.output_dir / "REPORT.md", per_replication, summary, distances, detail)
    print_final_table(summary, distances)
    print(f"\nOutputs: {args.output_dir}")


if __name__ == "__main__":
    main()
