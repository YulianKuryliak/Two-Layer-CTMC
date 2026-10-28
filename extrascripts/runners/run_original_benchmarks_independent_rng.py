#!/usr/bin/env python3
"""Run every plotted configuration in main_clean's three benchmark families.

The production simulator is not edited.  This runner temporarily replaces only
MicroModel.clone/refresh_from inside a worker process, following
error_source_decomposition/runners/run_production_factorial.py.
Network generation skips unused alpha/mu2 spectra for the basic Production mode.

The scenario matrix is the union of the configurations needed by the figures in
paper/main.tex.  Repeated baseline cells are simulated once and reused by the
interface, length, and size summaries.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import gzip
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from two_layer_ctmc.engines.micro_model import MicroModel
import two_layer_ctmc.network as network_module
from two_layer_ctmc.network import generate_two_scale_network
from two_layer_ctmc.orchestrators.micromacro import Orchestrator
from two_layer_ctmc.simulators.micro import MicroSimulator
from two_layer_ctmc.simulators.micromacro import MicroMacroSimulator
from extrascripts.runners.checkpoint_io import compact_trajectories, trajectory_block


FULL = "Full Micro"
MICROMACRO = "Production MicroMacro + independent RNG"
SIMULATORS = (FULL, MICROMACRO)
FORMAT_VERSION = 2


@dataclass(frozen=True)
class Benchmark:
    name: str
    family: str
    plot_groups: tuple[str, ...]
    replications: int
    base_seed: int
    network_seed_base: int
    communities: int
    community_size: int
    inter_links: int
    macro_graph_type: str
    micro_graph_type: str
    edge_prob: float
    beta: float
    gamma: float
    model: int
    tau_micro: float
    macro_T: float
    original_T_end: float
    T_end: float
    dt_out: float
    initial_node: int
    mean_internal_degree: float | None = None
    store_trajectories: bool = True
    star_leaf_attachment: str = "random"
    leaf_count: int = 0
    leaf_degree: int = 1


# Only the three benchmark families named as benchmarks in main_clean are here.
# There are 16 unique simulation configurations.  The L10/n100/k5 baselines
# occur in several plotted sweeps but must not be rerun as duplicate cells.
BENCHMARKS = (
    Benchmark(
        name="clique_leaf_n100_k2",
        family="clique_leaf",
        plot_groups=("metric_density_panels",),
        replications=10_000,
        base_seed=42,
        network_seed_base=42,
        communities=2,
        community_size=100,
        inter_links=2,
        macro_graph_type="star",
        micro_graph_type="clique_leaves",
        edge_prob=0.031,
        beta=0.005,
        gamma=0.0,
        model=2,
        tau_micro=1.0,
        macro_T=1.0,
        original_T_end=1000.0,
        T_end=1000.0,
        dt_out=1.0,
        initial_node=0,
        store_trajectories=False,
    ),
    Benchmark(
        name="chain_cliques_L10_n100_k5_baseline",
        family="chain_of_cliques",
        plot_groups=("baseline", "interface_k5", "length_L10", "size_n100"),
        replications=100,
        base_seed=2000,
        network_seed_base=2000,
        communities=10,
        community_size=100,
        inter_links=5,
        macro_graph_type="chain",
        micro_graph_type="complete",
        edge_prob=0.031,
        beta=0.001,
        gamma=0.0,
        model=2,
        tau_micro=1.0,
        macro_T=1.0,
        original_T_end=2500.0,
        T_end=2500.0,
        dt_out=1.0,
        initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L10_n100_k1",
        family="chain_of_cliques",
        plot_groups=("interface_k1",),
        replications=100,
        base_seed=3000,
        network_seed_base=3000,
        communities=10,
        community_size=100,
        inter_links=1,
        macro_graph_type="chain",
        micro_graph_type="complete",
        edge_prob=0.031,
        beta=0.001,
        gamma=0.0,
        model=2,
        tau_micro=1.0,
        macro_T=1.0,
        original_T_end=2500.0,
        T_end=2500.0,
        dt_out=1.0,
        initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L10_n100_k1_dynamics_T5000",
        family="chain_of_cliques",
        plot_groups=("interface_k1_extended_dynamics",),
        replications=100,
        base_seed=9500,
        network_seed_base=9500,
        communities=10, community_size=100, inter_links=1,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=5000.0, T_end=5000.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L10_n100_k10",
        family="chain_of_cliques", plot_groups=("interface_k10",),
        replications=100, base_seed=9000, network_seed_base=9000,
        communities=10, community_size=100, inter_links=10,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=2500.0, T_end=2500.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L10_n100_k20",
        family="chain_of_cliques", plot_groups=("interface_k20",),
        replications=100, base_seed=4000, network_seed_base=4000,
        communities=10, community_size=100, inter_links=20,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=2500.0, T_end=2500.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L5_n100_k5",
        family="chain_of_cliques", plot_groups=("length_L5",),
        replications=100, base_seed=5000, network_seed_base=5000,
        communities=5, community_size=100, inter_links=5,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=1500.0, T_end=1500.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L15_n100_k5",
        family="chain_of_cliques", plot_groups=("length_L15",),
        replications=100, base_seed=6000, network_seed_base=6000,
        communities=15, community_size=100, inter_links=5,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L10_n50_k5",
        family="chain_of_cliques", plot_groups=("size_n50",),
        replications=100, base_seed=7000, network_seed_base=7000,
        communities=10, community_size=50, inter_links=5,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=2500.0, T_end=2500.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_cliques_L10_n200_k5",
        family="chain_of_cliques", plot_groups=("size_n200",),
        replications=100, base_seed=8000, network_seed_base=8000,
        communities=10, community_size=200, inter_links=5,
        macro_graph_type="chain", micro_graph_type="complete", edge_prob=0.031,
        beta=0.001, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=2500.0, T_end=2500.0, dt_out=1.0, initial_node=0,
    ),
    Benchmark(
        name="chain_random_connected_k5_d6_baseline",
        family="chain_of_random_graphs",
        plot_groups=("baseline", "interface_k5", "density_d6"),
        replications=100, base_seed=8000, network_seed_base=18000,
        communities=10, community_size=100, inter_links=5,
        macro_graph_type="chain", micro_graph_type="random_connected", edge_prob=6 / 99,
        beta=0.003, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
        mean_internal_degree=6.0,
    ),
    Benchmark(
        name="chain_random_connected_k1_d6",
        family="chain_of_random_graphs", plot_groups=("interface_k1",),
        replications=100, base_seed=8500, network_seed_base=18500,
        communities=10, community_size=100, inter_links=1,
        macro_graph_type="chain", micro_graph_type="random_connected", edge_prob=6 / 99,
        beta=0.003, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
        mean_internal_degree=6.0,
    ),
    Benchmark(
        name="chain_random_connected_k10_d6",
        family="chain_of_random_graphs", plot_groups=("interface_k10",),
        replications=100, base_seed=8510, network_seed_base=18510,
        communities=10, community_size=100, inter_links=10,
        macro_graph_type="chain", micro_graph_type="random_connected", edge_prob=6 / 99,
        beta=0.003, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
        mean_internal_degree=6.0,
    ),
    Benchmark(
        name="chain_random_connected_k20_d6",
        family="chain_of_random_graphs", plot_groups=("interface_k20",),
        replications=100, base_seed=8520, network_seed_base=18520,
        communities=10, community_size=100, inter_links=20,
        macro_graph_type="chain", micro_graph_type="random_connected", edge_prob=6 / 99,
        beta=0.003, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
        mean_internal_degree=6.0,
    ),
    Benchmark(
        name="chain_random_connected_k5_d4",
        family="chain_of_random_graphs", plot_groups=("density_d4",),
        replications=100, base_seed=8600, network_seed_base=18600,
        communities=10, community_size=100, inter_links=5,
        macro_graph_type="chain", micro_graph_type="random_connected", edge_prob=4 / 99,
        beta=0.003, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
        mean_internal_degree=4.0,
    ),
    Benchmark(
        name="chain_random_connected_k5_d10",
        family="chain_of_random_graphs", plot_groups=("density_d10",),
        replications=100, base_seed=8610, network_seed_base=18610,
        communities=10, community_size=100, inter_links=5,
        macro_graph_type="chain", micro_graph_type="random_connected", edge_prob=10 / 99,
        beta=0.003, gamma=0.0, model=2, tau_micro=1.0, macro_T=1.0,
        original_T_end=4000.0, T_end=4000.0, dt_out=1.0, initial_node=0,
        mean_internal_degree=10.0,
    ),
)


def _load_benchmarks(config_path: Path | None) -> list[Benchmark]:
    """Load user scenarios, or return the built-in main_clean matrix."""
    if config_path is None:
        return list(BENCHMARKS)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {config_path}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("scenarios"), list):
        raise ValueError("config must be a JSON object containing a 'scenarios' list")

    defaults = {
        "family": "custom",
        "plot_groups": (),
        "replications": int(document.get("replications_default", 100)),
        "base_seed": 1000,
        "network_seed_base": None,
        "macro_graph_type": "chain",
        "micro_graph_type": "complete",
        "edge_prob": 0.1,
        "beta": 0.003,
        "gamma": 0.0,
        "model": 2,
        "tau_micro": 1.0,
        "macro_T": 1.0,
        "original_T_end": None,
        "T_end": 1000.0,
        "dt_out": 1.0,
        "initial_node": 0,
        "mean_internal_degree": None,
        "store_trajectories": True,
        "star_leaf_attachment": "random",
        "leaf_count": 0,
        "leaf_degree": 1,
        "communities": 5,
        "community_size": 100,
        "inter_links": 5,
    }
    defaults.update(document.get("defaults", {}))
    valid = set(Benchmark.__dataclass_fields__)
    scenarios: list[Benchmark] = []
    for index, raw in enumerate(document["scenarios"], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"scenario {index} must be a JSON object")
        values = dict(defaults)
        values.update(raw)
        unknown = set(values) - valid
        if unknown:
            raise ValueError(f"scenario {index} has unknown fields: {sorted(unknown)}")
        if not values.get("name"):
            raise ValueError(f"scenario {index} requires a unique 'name'")
        if values.get("network_seed_base") is None:
            values["network_seed_base"] = int(values["base_seed"]) + 100_000
        if values.get("original_T_end") is None:
            values["original_T_end"] = values["T_end"]
        if values.get("mean_internal_degree") is not None and "edge_prob" not in raw:
            values["edge_prob"] = float(values["mean_internal_degree"]) / max(
                1, int(values["community_size"]) - 1
            )
        values["plot_groups"] = tuple(values.get("plot_groups", ()))
        scenarios.append(Benchmark(**values))
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")
    if not scenarios:
        raise ValueError("config scenarios list is empty")
    return scenarios


RUN_FIELDS = (
    "family",
    "benchmark",
    "replication_id",
    "simulator",
    "network_seed",
    "event_seed",
    "initial_node",
    "completion",
    "T_last",
    "runtime_seconds",
    "T_end",
)
ARRIVAL_FIELDS = (
    "family",
    "benchmark",
    "replication_id",
    "simulator",
    "community",
    "first_infection_time",
)
TRAJECTORY_FIELDS = (
    "family",
    "benchmark",
    "replication_id",
    "simulator",
    "time",
    "community",
    "S",
    "I",
    "R",
)
RUNTIME_FIELDS = (
    "family",
    "benchmark",
    "replication_id",
    "simulator",
    "network_seed",
    "event_seed",
    "runtime_seconds",
)
SMOKE_FIELDS = (
    "family",
    "benchmark",
    "attempt",
    "replication_id",
    "simulator",
    "network_seed",
    "event_seed",
    "T_end",
    "completion",
    "propagated",
    "runtime_seconds",
    "topology_valid",
    "initial_infection_valid",
    "micro_graphs_connected",
    "connectivity_required",
    "projected_100_run_wall_seconds",
    "runtime_practical",
)
OBSERVABLE_FIELDS = (
    "family",
    "benchmark",
    "replication_id",
    "simulator",
    "bridge_infection_time",
    "bridge_to_export_delay",
    "total_export_time",
)


def _copy_rng(rng: Any) -> Any:
    if rng is random:
        return random
    copied = random.Random()
    copied.setstate(rng.getstate())
    return copied


def _patched_clone(self: MicroModel) -> MicroModel:
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
    model._invalidate_cache()  # production policy: discard sampled probe cache
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
        destination = self.G.nodes[node]
        source = src.G.nodes[node]
        destination["infected"] = source["infected"]
        destination["recovered"] = source["recovered"]
        destination["sum_of_weights_i"] = source["sum_of_weights_i"]
    if independent:
        self.rng.setstate(src.rng.getstate())
    self._invalidate_cache()  # production policy: discard sampled probe cache


@contextmanager
def _rng_ownership_patch(independent: bool) -> Iterator[None]:
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


def _boundary_counts_without_spectrum(micro_graphs, full_graph, communities_nodes):
    community_of = {node: c for c, nodes in enumerate(communities_nodes) for node in nodes}
    boundary = [set() for _ in micro_graphs]
    for u, v in full_graph.edges():
        cu, cv = community_of[u], community_of[v]
        if cu != cv:
            boundary[cu].add(u)
            boundary[cv].add(v)
    # Counts are measured; mu2 values are neutral placeholders, never used here.
    return [len(nodes) for nodes in boundary], [1.0] * len(micro_graphs)


@contextmanager
def _without_unused_spectra():
    # This runner only uses basic Production (both corrections off).
    for cls in (Orchestrator, MicroMacroSimulator):
        signature = inspect.signature(cls)
        if any(signature.parameters[key].default != "off" for key in ("alpha_mode", "correction_mode")):
            raise RuntimeError("Skipping spectra requires alpha_mode=off and correction_mode=off")
    original_alpha = network_module._compute_micro_alphas
    original_boundary = network_module._compute_boundary_descriptors
    network_module._compute_micro_alphas = lambda graphs: [1.0] * len(graphs)
    network_module._compute_boundary_descriptors = _boundary_counts_without_spectrum
    try:
        yield
    finally:
        network_module._compute_micro_alphas = original_alpha
        network_module._compute_boundary_descriptors = original_boundary


def _generate_network(benchmark: Benchmark, network_seed: int):
    # Process-local preprocessing shortcut: graph sampling and RNG draws unchanged.
    with _without_unused_spectra():
        graphs, full_graph, W = generate_two_scale_network(
            n_communities=benchmark.communities,
            community_size=benchmark.community_size,
            inter_links=benchmark.inter_links,
            seed=network_seed,
            macro_graph_type=benchmark.macro_graph_type,
            micro_graph_type=benchmark.micro_graph_type,
            edge_prob=benchmark.edge_prob,
            leaf_count=benchmark.leaf_count,
            leaf_degree=benchmark.leaf_degree,
            star_leaf_attachment=benchmark.star_leaf_attachment,
            share_topology=True,
        )
    full_graph.graph["spectral_descriptors_computed"] = False
    return graphs, full_graph, W


def _run_production_shared(
    benchmark: Benchmark,
    micro_graphs: list[nx.Graph],
    full_graph: nx.Graph,
    W: np.ndarray,
    event_seed: int,
):
    simulator = MicroMacroSimulator(
        W=W,
        micro_graphs=micro_graphs,
        beta_micro=benchmark.beta,
        gamma=benchmark.gamma,
        beta_macro=benchmark.beta,
        tau_micro=benchmark.tau_micro,
        T_end=benchmark.T_end,
        macro_T=benchmark.macro_T,
        model=benchmark.model,
        full_graph=full_graph,
    )
    return simulator.run(seed=event_seed, initial_community=0, initial_node=benchmark.initial_node)


def _run_experimental_micromacro(
    benchmark: Benchmark,
    micro_graphs: list[nx.Graph],
    full_graph: nx.Graph,
    W: np.ndarray,
    event_seed: int,
    *,
    independent: bool,
    dense_logs: bool = True,
    stop_at_absorption: bool = False,
    return_metadata: bool = False,
):
    random.seed(event_seed)
    np.random.seed(event_seed)
    with _rng_ownership_patch(independent=independent):
        orchestrator = Orchestrator(
            W=W,
            micro_graphs=micro_graphs,
            beta_micro=benchmark.beta,
            gamma=benchmark.gamma,
            beta_macro=benchmark.beta,
            tau_micro=benchmark.tau_micro,
            T_end=benchmark.T_end,
            macro_T=benchmark.macro_T,
            model=benchmark.model,
            community_sizes=[graph.number_of_nodes() for graph in micro_graphs],
            alphas=full_graph.graph.get("alphas"),
            full_graph=full_graph,
            dense_logs=dense_logs,
            stop_at_absorption=stop_at_absorption,
        )
        if independent:
            for community, model in enumerate(orchestrator.micro_models):
                model.rng = random.Random(event_seed + 10_000 + 1009 * community)
        orchestrator.micro_models[0]._infect_node(benchmark.initial_node)
        orchestrator.event_log.append(
            {
                "time": 0.0,
                "wait_time": 0.0,
                "event_type": "seed",
                "mode": "seed",
                "community": 0,
                "node": benchmark.initial_node,
                "src": None,
            }
        )
        result = orchestrator.run()
        return (result, orchestrator.run_metadata) if return_metadata else result


def _regression_gate() -> tuple[bool, str]:
    control = Benchmark(
        name="regression_control",
        family="regression_control",
        plot_groups=(),
        replications=1,
        base_seed=7101,
        network_seed_base=7201,
        communities=3,
        community_size=12,
        inter_links=2,
        macro_graph_type="chain",
        micro_graph_type="complete",
        edge_prob=0.2,
        beta=0.03,
        gamma=0.0,
        model=2,
        tau_micro=1.0,
        macro_T=1.0,
        original_T_end=80.0,
        T_end=80.0,
        dt_out=1.0,
        initial_node=0,
    )
    seeds = (7101, 7102, 7103)
    for seed in seeds:
        graphs, full_graph, W = _generate_network(control, seed + 100)
        production = _run_production_shared(control, graphs, full_graph, W, seed)
        experimental = _run_experimental_micromacro(
            control, graphs, full_graph, W, seed, independent=False
        )
        if production != experimental:
            components = ("times", "I_total", "logs", "event_log")
            mismatches = [
                name
                for name, left, right in zip(components, production, experimental)
                if left != right
            ]
            return False, f"seed={seed}; mismatch in {', '.join(mismatches)}"
    return True, f"exact pathwise match on seeds {list(seeds)}"


def _validate_network(
    benchmark: Benchmark,
    micro_graphs: list[nx.Graph],
    full_graph: nx.Graph,
    W: np.ndarray,
) -> dict[str, Any]:
    expected_sizes = (
        [benchmark.community_size, 1]
        if benchmark.family == "clique_leaf"
        else [benchmark.community_size] * benchmark.communities
    )
    actual_sizes = [graph.number_of_nodes() for graph in micro_graphs]
    expected_pairs = (
        {(0, 1)}
        if benchmark.family == "clique_leaf"
        else (
            {(i, j) for i in range(benchmark.communities) for j in range(i + 1, benchmark.communities)}
            if benchmark.macro_graph_type == "complete"
            else {(i, i + 1) for i in range(benchmark.communities - 1)}
        )
    )
    actual_pairs = {
        (i, j)
        for i in range(W.shape[0])
        for j in range(i + 1, W.shape[1])
        if W[i, j] > 0
    }
    pair_weights_valid = all(W[i, j] == benchmark.inter_links for i, j in expected_pairs)
    all_edge_weights_one = all(float(data.get("weight", 1.0)) == 1.0 for _, _, data in full_graph.edges(data=True))
    if benchmark.family == "clique_leaf":
        internal_valid = micro_graphs[0].number_of_edges() == benchmark.community_size * (benchmark.community_size - 1) // 2
        bridge_nodes = {
            u if u < benchmark.community_size else v
            for u, v in full_graph.edges()
            if (u < benchmark.community_size) != (v < benchmark.community_size)
        }
        special_valid = len(bridge_nodes) == benchmark.inter_links
    elif benchmark.micro_graph_type == "complete":
        target_edges = benchmark.community_size * (benchmark.community_size - 1) // 2
        internal_valid = all(graph.number_of_edges() == target_edges for graph in micro_graphs)
        special_valid = True
    else:
        internal_valid = True
        special_valid = True
    connected = [nx.is_connected(graph) if graph.number_of_nodes() else False for graph in micro_graphs]
    topology_valid = all(
        (
            actual_sizes == expected_sizes,
            actual_pairs == expected_pairs,
            pair_weights_valid,
            all_edge_weights_one,
            internal_valid,
            special_valid,
            benchmark.initial_node in micro_graphs[0],
        )
    )
    return {
        "topology_valid": topology_valid,
        "actual_community_sizes": actual_sizes,
        "expected_macro_pairs": sorted(expected_pairs),
        "actual_macro_pairs": sorted(actual_pairs),
        "all_edge_weights_one": all_edge_weights_one,
        "micro_graphs_connected": all(connected),
        "connectivity_by_community": connected,
        "connectivity_required": benchmark.micro_graph_type != "random",
        "spectral_descriptors_computed": full_graph.graph.get("spectral_descriptors_computed", True),
        "initial_infection_valid": benchmark.initial_node in micro_graphs[0],
    }


def _arrivals_from_full(result: dict[str, Any], communities: int, horizon: float) -> list[float]:
    arrivals = [math.nan] * communities
    for timing in result.get("community_timings", []):
        community = int(timing["community"])
        value = timing.get("first_infection_time")
        if value is not None and float(value) <= horizon:
            arrivals[community] = float(value)
    arrivals[0] = 0.0
    return arrivals


def _arrivals_from_events(
    events: Sequence[dict[str, Any]], communities: int, horizon: float
) -> list[float]:
    arrivals = [math.nan] * communities
    # Production event times are finite: their minima need no O(E log E) sort.
    # Preserve the historical sort/first-finite behavior for malformed or
    # non-finite times (notably NaN, which has no total sorting order).
    finite_times = all(math.isfinite(float(event.get("time", math.inf))) for event in events)
    ordered_events = (
        events if finite_times
        else sorted(events, key=lambda item: float(item.get("time", math.inf)))
    )
    for event in ordered_events:
        if str(event.get("event_type", "")).lower() not in {"seed", "infection", "transfer"}:
            continue
        community = event.get("community")
        event_time = float(event.get("time", math.inf))
        if community is None or event_time > horizon:
            continue
        community = int(community)
        if 0 <= community < communities and (
            not math.isfinite(arrivals[community])
            or (finite_times and event_time < arrivals[community])
        ):
            arrivals[community] = event_time
    arrivals[0] = 0.0
    return arrivals


def _t_last(arrivals: Sequence[float]) -> float:
    return max(arrivals) if all(math.isfinite(value) for value in arrivals) else math.nan


def _full_trajectory(
    benchmark: Benchmark, replication_id: int, rows: Sequence[Sequence[Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "family": benchmark.family,
            "benchmark": benchmark.name,
            "replication_id": replication_id,
            "simulator": FULL,
            "time": float(time_value),
            "community": int(community),
            "S": int(S),
            "I": int(I),
            "R": int(R),
        }
        for community, time_value, S, I, R in _full_trajectory_values(benchmark, rows)
    ]


def _full_trajectory_values(benchmark: Benchmark, rows):
    for community, time_value, S, I, R in rows:
        if float(time_value) <= benchmark.T_end + 1e-9:
            yield int(community), float(time_value), int(S), int(I), int(R)


def _micromacro_trajectory_values(
    benchmark: Benchmark, logs, trajectory_end: float | None = None
):
    # Keep exactly the old reporting grid and rightmost-state-at-time rule.
    end = benchmark.T_end if trajectory_end is None else min(benchmark.T_end, trajectory_end)
    grid = np.arange(0.0, end + 0.5 * benchmark.dt_out, benchmark.dt_out)
    for community in range(benchmark.communities):
        log = logs[community]
        times = np.asarray(log["times"], dtype=float)
        indices = np.maximum(0, np.searchsorted(times, grid, side="right") - 1)
        for time_value, index in zip(grid, indices):
            yield (community, float(time_value), int(log["S"][index]),
                   int(log["I"][index]), int(log["R"][index]))


def _micromacro_trajectory(
    benchmark: Benchmark, replication_id: int, logs: dict[int, dict[str, list[Any]]],
    trajectory_end: float | None = None,
) -> list[dict[str, Any]]:
    return [
        {"family": benchmark.family, "benchmark": benchmark.name,
         "replication_id": replication_id, "simulator": MICROMACRO,
         "time": time_value, "community": community, "S": S, "I": I, "R": R}
        for community, time_value, S, I, R in _micromacro_trajectory_values(
            benchmark, logs, trajectory_end
        )
    ]


def _result_rows(
    benchmark: Benchmark,
    replication_id: int,
    network_seed: int,
    event_seed: int,
    simulator: str,
    arrivals: Sequence[float],
    runtime_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    last = _t_last(arrivals)
    run = {
        "family": benchmark.family,
        "benchmark": benchmark.name,
        "replication_id": replication_id,
        "simulator": simulator,
        "network_seed": network_seed,
        "event_seed": event_seed,
        "initial_node": benchmark.initial_node,
        "completion": math.isfinite(last),
        "T_last": last,
        "runtime_seconds": runtime_seconds,
        "T_end": benchmark.T_end,
    }
    arrival_rows = [
        {
            "family": benchmark.family,
            "benchmark": benchmark.name,
            "replication_id": replication_id,
            "simulator": simulator,
            "community": community,
            "first_infection_time": value,
        }
        for community, value in enumerate(arrivals)
    ]
    runtime = {key: run[key] for key in RUNTIME_FIELDS}
    return run, arrival_rows, runtime


def _clique_leaf_observable(
    benchmark: Benchmark,
    replication_id: int,
    simulator: str,
    bridge_time: float,
    export_time: float,
) -> dict[str, Any]:
    delay = (
        export_time - bridge_time
        if math.isfinite(bridge_time) and math.isfinite(export_time)
        else math.nan
    )
    return {
        "family": benchmark.family,
        "benchmark": benchmark.name,
        "replication_id": replication_id,
        "simulator": simulator,
        "bridge_infection_time": bridge_time,
        "bridge_to_export_delay": delay,
        "total_export_time": export_time,
    }


def _run_task(
    task: tuple[dict[str, Any], int, int, int], *, compact_output: bool = False,
    stop_at_absorption: bool = False,
) -> dict[str, Any]:
    benchmark_dict, replication_id, network_seed, event_seed = task
    benchmark = Benchmark(**benchmark_dict)
    micro_graphs, full_graph, W = _generate_network(benchmark, network_seed)
    validation = _validate_network(benchmark, micro_graphs, full_graph, W)
    if not validation["topology_valid"]:
        raise RuntimeError(f"network validation failed for {benchmark.name} replication {replication_id}: {validation}")
    community_nodes = [list(graph.nodes()) for graph in micro_graphs]

    started = time.perf_counter()
    full_result = MicroSimulator(
        full_graph=full_graph,
        comm_nodes=community_nodes,
        infection_rate=benchmark.beta,
        recovery_rate=benchmark.gamma,
        model=benchmark.model,
    ).run(
        T_end=benchmark.T_end,
        dt_out=benchmark.dt_out,
        initial_node=benchmark.initial_node,
        rng=random.Random(event_seed),
        stop_at_absorption=stop_at_absorption,
    )
    full_runtime = time.perf_counter() - started
    full_arrivals = _arrivals_from_full(full_result, benchmark.communities, benchmark.T_end)
    full_run, full_arrival_rows, full_runtime_row = _result_rows(
        benchmark, replication_id, network_seed, event_seed, FULL, full_arrivals, full_runtime
    )
    full_metadata = full_result.get("run_metadata") if stop_at_absorption else None
    # Retain only data needed in the final payload while MicroMacro is active.
    full_rows = full_result["rows"] if benchmark.store_trajectories else ()
    full_block = None
    if compact_output and benchmark.store_trajectories:
        full_block = trajectory_block(FULL, _full_trajectory_values(benchmark, full_rows))
        full_rows = ()
    full_bridge = math.nan
    bridge_nodes = set()
    if benchmark.family == "clique_leaf":
        source_nodes = set(community_nodes[0])
        bridge_nodes = {
            u if u in source_nodes else v
            for u, v in full_graph.edges()
            if (u in source_nodes) != (v in source_nodes)
        }
        for timing in full_result.get("community_timings", []):
            if int(timing["community"]) == 0:
                value = timing.get("first_bridge_infection_time")
                if value is not None:
                    full_bridge = float(value)
                break
    del full_result
    gc.collect()  # Release expired engine/view cycles before the next engine.

    started = time.perf_counter()
    mm_output = _run_experimental_micromacro(
        benchmark, micro_graphs, full_graph, W, event_seed, independent=True,
        dense_logs=not compact_output,
        stop_at_absorption=stop_at_absorption,
        return_metadata=stop_at_absorption,
    )
    if stop_at_absorption:
        mm_result, mm_metadata = mm_output
    else:
        mm_result, mm_metadata = mm_output, None
    mm_runtime = time.perf_counter() - started
    mm_arrivals = _arrivals_from_events(mm_result[3], benchmark.communities, benchmark.T_end)
    mm_run, mm_arrival_rows, mm_runtime_row = _result_rows(
        benchmark, replication_id, network_seed, event_seed, MICROMACRO, mm_arrivals, mm_runtime
    )
    del micro_graphs, full_graph, W, community_nodes
    gc.collect()  # Networks are not part of the checkpoint or trajectory output.

    observables: list[dict[str, Any]] = []
    if benchmark.family == "clique_leaf":
        mm_bridge_candidates = [
            float(event["time"])
            for event in mm_result[3]
            if str(event.get("event_type", "")).lower() in {"seed", "infection"}
            and event.get("node") in bridge_nodes
            and float(event.get("time", math.inf)) <= benchmark.T_end
        ]
        mm_bridge = min(mm_bridge_candidates, default=math.nan)
        observables = [
            _clique_leaf_observable(
                benchmark, replication_id, FULL, full_bridge, full_arrivals[1]
            ),
            _clique_leaf_observable(
                benchmark, replication_id, MICROMACRO, mm_bridge, mm_arrivals[1]
            ),
        ]

    trajectories = []
    if benchmark.store_trajectories:
        mm_trajectory_end = (
            float(mm_metadata["stop_time"])
            if stop_at_absorption and mm_metadata is not None else None
        )
        if compact_output:
            trajectories = compact_trajectories(
                {"family": benchmark.family, "benchmark": benchmark.name,
                 "replication_id": replication_id},
                [full_block, trajectory_block(
                    MICROMACRO, _micromacro_trajectory_values(
                        benchmark, mm_result[2], mm_trajectory_end
                    ))],
            )
        else:
            trajectories = _full_trajectory(benchmark, replication_id, full_rows)
            trajectories.extend(_micromacro_trajectory(
                benchmark, replication_id, mm_result[2], mm_trajectory_end
            ))
        del full_rows

    payload = {
        "format_version": 4 if compact_output and stop_at_absorption else (3 if compact_output else FORMAT_VERSION),
        "benchmark": benchmark.name,
        "replication_id": replication_id,
        "benchmark_config": asdict(benchmark),
        "runs": [full_run, mm_run],
        "community_arrivals": full_arrival_rows + mm_arrival_rows,
        "trajectories": trajectories,
        "benchmark_observables": observables,
        "runtime": [full_runtime_row, mm_runtime_row],
        "network_validation": validation,
    }
    if stop_at_absorption:
        payload["termination"] = [
            {"simulator": FULL, **(full_metadata or {})},
            {"simulator": MICROMACRO, **(mm_metadata or {})},
        ]
    return payload


def _run_count(benchmark: Benchmark, runs_override: int | None) -> int:
    return benchmark.replications if runs_override is None else runs_override


def _tasks(
    benchmarks: Sequence[Benchmark], runs_override: int | None
) -> list[tuple[dict[str, Any], int, int, int]]:
    return [
        (
            asdict(benchmark),
            replication_id,
            benchmark.network_seed_base + replication_id - 1,
            benchmark.base_seed + replication_id - 1,
        )
        for benchmark in benchmarks
        for replication_id in range(1, _run_count(benchmark, runs_override) + 1)
    ]


def _execute(
    tasks: Sequence[tuple[dict[str, Any], int, int, int]], workers: int
) -> Iterator[dict[str, Any]]:
    if workers == 1:
        for task in tasks:
            yield _run_task(task)
        return
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        task_iterator = iter(tasks)
        pending = set()
        # Bound queued work AND retained results.  A list of every submitted
        # Future otherwise keeps all trajectory payloads alive until the end.
        for _ in range(2 * workers):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.add(executor.submit(_run_task, task))
        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            while done:
                future = done.pop()
                yield future.result()
                del future
                try:
                    task = next(task_iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(_run_task, task))


def _format_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _report_due(
    completed_since_start: int,
    progress_every: int,
    last_report_time: float,
    progress_seconds: float,
) -> bool:
    return (
        completed_since_start % progress_every == 0
        or time.monotonic() - last_report_time >= progress_seconds
    )


def _run_smoke(
    benchmarks: Sequence[Benchmark],
    smoke_runs: int,
    workers: int,
    runs_override: int | None,
) -> tuple[list[Benchmark], list[dict[str, Any]], list[dict[str, Any]]]:
    # Validate one generated network for every plotted configuration, including
    # all sparse degrees and interface widths, without paying for 16 smoke runs.
    for benchmark in benchmarks:
        graphs, full_graph, W = _generate_network(benchmark, benchmark.network_seed_base)
        validation = _validate_network(benchmark, graphs, full_graph, W)
        if not validation["topology_valid"]:
            raise RuntimeError(f"preflight topology validation failed for {benchmark.name}: {validation}")
        del graphs, full_graph, W
    gc.collect()  # Do not keep the last preflight network in the parent/pool.
    print(f"[setup] Topology preflight: PASS ({len(benchmarks)}/{len(benchmarks)} configurations)", flush=True)

    # One representative configuration per named benchmark family is sufficient
    # to exercise both simulators and the family-specific data extraction.
    families = list(dict.fromkeys(item.family for item in benchmarks))
    smoke_benchmarks = []
    for family in families:
        family_items = [item for item in benchmarks if item.family == family]
        smoke_benchmarks.append(
            next(
                (item for item in family_items if "baseline" in item.plot_groups),
                family_items[0],
            )
        )
    smoke_rows: list[dict[str, Any]] = []
    for benchmark in smoke_benchmarks:
        print(
            f"[smoke] {benchmark.family}: running {smoke_runs} paired replications...",
            flush=True,
        )
        family_rows = []
        total_runtime = 0.0
        completed_pairs = 0
        for payload in _execute(_tasks([benchmark], smoke_runs), workers):
            completed_pairs += all(run["completion"] for run in payload["runs"])
            validation = payload["network_validation"]
            for run in payload["runs"]:
                total_runtime += float(run["runtime_seconds"])
                propagated = sum(
                    math.isfinite(float(row["first_infection_time"]))
                    for row in payload["community_arrivals"]
                    if row["simulator"] == run["simulator"]
                ) > 1
                family_rows.append(
                    {
                        "family": benchmark.family,
                        "benchmark": benchmark.name,
                        "attempt": 1,
                        "replication_id": run["replication_id"],
                        "simulator": run["simulator"],
                        "network_seed": run["network_seed"],
                        "event_seed": run["event_seed"],
                        "T_end": benchmark.T_end,
                        "completion": run["completion"],
                        "propagated": propagated,
                        "runtime_seconds": run["runtime_seconds"],
                        "topology_valid": validation["topology_valid"],
                        "initial_infection_valid": validation["initial_infection_valid"],
                        "micro_graphs_connected": validation["micro_graphs_connected"],
                        "connectivity_required": validation["connectivity_required"],
                    }
                )
            del payload
        projected = (
            total_runtime / smoke_runs
            * _run_count(benchmark, runs_override) / max(1, workers)
        )
        for row in family_rows:
            row["projected_100_run_wall_seconds"] = projected
            row["runtime_practical"] = projected <= 6 * 3600
        smoke_rows.extend(family_rows)
        family_passed = all(
            bool(row["propagated"])
            and bool(row["topology_valid"])
            and bool(row["initial_infection_valid"])
            for row in family_rows
        )
        print(
            f"[smoke] {benchmark.family}: {'PASS' if family_passed else 'FAIL'}; "
            f"both completed in {completed_pairs}/{smoke_runs} pairs "
            f"(censoring is allowed)",
            flush=True,
        )
    gates_pass = all(
        bool(row["propagated"])
        and bool(row["topology_valid"])
        and bool(row["initial_infection_valid"])
        for row in smoke_rows
    )
    if not gates_pass:
        raise RuntimeError("smoke gate failed; fixed article horizons were not altered")
    # Censoring is an intended outcome in main_clean, especially for k=1.
    return list(benchmarks), smoke_rows, []


def _checkpoint_path(output_dir: Path, benchmark: str, replication_id: int) -> Path:
    return output_dir / benchmark / "checkpoints" / f"rep_{replication_id:04d}.json.gz"


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=True, separators=(",", ":"))
    temporary.replace(path)


def _read_checkpoint(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"unsupported checkpoint format: {path}")
    return payload


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None


def _manifest(
    runs_override: int | None,
    smoke_runs: int,
    workers: int,
    benchmarks: Sequence[Benchmark],
    regression_detail: str,
    adjustments: Sequence[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    source_files = [
        "extrascripts/runners/run_original_benchmarks_independent_rng.py",
        "two_layer_ctmc/network.py",
        "two_layer_ctmc/simulators/micro.py",
        "two_layer_ctmc/simulators/micromacro.py",
        "two_layer_ctmc/orchestrators/micro.py",
        "two_layer_ctmc/orchestrators/micromacro.py",
        "two_layer_ctmc/engines/micro_model.py",
        "two_layer_ctmc/engines/micro_engine.py",
        "two_layer_ctmc/engines/topology.py",
        "two_layer_ctmc/engines/macro_engine.py",
        "error_source_decomposition/runners/run_production_factorial.py",
    ]
    return {
        "format_version": FORMAT_VERSION,
        "status": status,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "runs_override": runs_override,
        "scenario_count": len(benchmarks),
        "paired_replications_total": sum(_run_count(item, runs_override) for item in benchmarks),
        "smoke_replications": smoke_runs,
        "workers": workers,
        "network_preprocessing": {
            "spectral_descriptors_computed": False,
            "alphas_and_mu2": "neutral placeholders (1), unused by basic Production",
            "boundary_counts": "measured from the actual graph",
            "graph_sampling_unchanged": True,
        },
        "memory_optimizations": {
            "community_graphs": "read-only views sharing full_graph edges",
            "full_engine": "shared edge view, private node states, legacy-order compact neighbors",
            "micro_neighbors": "int64/float64 arrays; Python scalar iteration; generic-node fallback",
            "intermediate_data": "release obsolete Full logs, networks and preflight graphs before next stage",
            "regression_reference": "extrascripts/validation/fixtures/memory_optimization_reference.json",
        },
        "seed_policy": {
            "network_seed": "scenario.network_seed_base + replication_id - 1",
            "event_seed": "scenario.base_seed + replication_id - 1",
            "independent_micro_seed": "event_seed + 10000 + 1009 * community",
        },
        "historical_source_evidence": {
            "main_clean": {
                "source": "paper/main.tex",
                "evidence": "three benchmark families; all interface, length, size, and sparse-density figure cells",
            },
            "clique_leaf_config": {
                "source": "git commit 038460c:config.json plus recovered main-old blob",
                "note": "article fixes n=100,k=2; historical config supplies beta=0.005,gamma=0,model=2,tau=1,macro_T=1,T_end=1000,initial_node=0 and unit reporting",
            },
            "chain_of_cliques_config": {
                "source": "data/chain_cliques_arrival_comparison/*/parameters.json and paper/main.tex",
                "note": "calibrated beta=0.001 matrix; fixed scenario-specific horizons and archived seed bases",
            },
            "chain_of_random_graphs_config": {
                "source": "data/chain_cliques_arrival_comparison/*/parameters.json and paper/main.tex",
                "note": "L=10 connected exact-edge ER graphs, beta=0.003, T_end=4000, archived seed bases",
            },
            "random_graph_interpretation": {
                "selected": "random_connected",
                "connectedness_enforced": True,
                "evidence": "paper/main.tex and archived scenario manifests explicitly require connected exact-edge ER graphs",
            },
        },
        "scenario_matrix": [asdict(benchmark) for benchmark in benchmarks],
        "baseline_reuse": "identical L10/n100/k5 baselines are simulated once and tagged for every applicable plot group",
        "smoke_adjustments": list(adjustments),
        "trajectory_policy": "all chain scenarios: per-community S/I/R on the same regular dt_out grid for both simulators; clique-leaf: timing observables only because its article figure uses distributions, not trajectories",
        "epidemic_model": "SI represented by production model=2 (SIR) with gamma=0",
        "production_algorithm_validation": {
            "regression_gate": "PASS",
            "regression_detail": regression_detail,
            "aggregate_macro_hazard_unchanged": True,
            "uniform_import_placement_unchanged": True,
            "production_tau_scheduler_unchanged": True,
            "midpoint_hazard_integration_unchanged": True,
            "repeated_imports_unchanged": True,
            "cache_policy": "production discard",
            "independent_rng_per_MicroModel": True,
            "patched_methods_only": ["MicroModel.clone", "MicroModel.refresh_from"],
        },
        "source_sha256": {
            relative: _sha256(REPO_ROOT / relative) for relative in source_files
        },
    }


def _write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    validation = manifest["production_algorithm_validation"]
    adjustments = manifest["smoke_adjustments"]
    source = manifest.get("scenario_source", "built-in main_clean matrix")
    lines = [
        "# Original article benchmarks with independent RNG",
        "",
        f"Scenario source: `{source}`. This run contains {manifest['scenario_count']} configurations and {manifest['paired_replications_total']} paired replications.",
        "",
        "The built-in configuration reproduces the three benchmark families from `main_clean`; a user JSON supplied with `--config` may define arbitrary families and topologies.",
        "",
        "## Production-algorithm validation",
        "",
        f"- Regression gate: {validation['regression_gate']} ({validation['regression_detail']})",
        "- Aggregate macro hazard unchanged: yes",
        "- Uniform import placement unchanged: yes",
        "- Production tau scheduler unchanged: yes",
        "- Midpoint hazard integration unchanged: yes",
        "- Repeated imports unchanged: yes",
        "- Cache policy unchanged: production discard",
        "- Independent RNG per MicroModel: enabled",
        "",
        "The only simulator intervention is RNG ownership in `MicroModel.clone()` and `refresh_from()`.",
        "Network preprocessing skips unused spectral decompositions; alpha/mu2 metadata are neutral placeholders. Actual boundary counts and graph sampling are retained.",
        "",
        "## Historical interpretation",
        "",
        "For `random_connected`, `edge_prob` is passed to the existing exact-edge connected generator. If omitted and `mean_internal_degree` is supplied, the runner derives `edge_prob = mean_internal_degree/(community_size-1)`.",
        "",
        "## Trajectories",
        "",
        "`trajectories.csv` stores per-community S/I/R on the same regular `dt_out` grid for both simulators for every chain scenario. The 10,000-run clique-leaf density benchmark is stored in `benchmark_observables.csv` and does not duplicate unused trajectories.",
        "",
        "## Progress and resume",
        "",
        "The runner prints setup and smoke-test status, restored checkpoint counts, per-scenario completion, overall done/remaining counts, elapsed time, and an approximate ETA. Defaults are an update every 25 paired replications or after 30 seconds; use `--progress-every` and `--progress-seconds` to change this. Resume an interrupted run with `--resume`.",
        "",
        "## Smoke adjustments",
        "",
    ]
    if adjustments:
        for adjustment in adjustments:
            lines.append(
                f"- `{adjustment['benchmark']}`: {adjustment['parameter']} {adjustment['original']} -> {adjustment['final']} ({adjustment['reason']})."
            )
    else:
        lines.append("- None. Article horizons are fixed; censoring never triggers automatic horizon changes.")
    lines.extend(
        [
            "",
            "Censored arrivals are empty values. A horizon is never stored as a synthetic arrival time.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _consolidate(
    output_dir: Path,
    benchmarks: Sequence[Benchmark],
    runs_override: int | None,
    progress_every: int,
    progress_seconds: float,
) -> list[dict[str, Any]]:
    specifications = {
        "runs.csv": (RUN_FIELDS, "runs"),
        "community_arrivals.csv": (ARRIVAL_FIELDS, "community_arrivals"),
        "trajectories.csv": (TRAJECTORY_FIELDS, "trajectories"),
        "benchmark_observables.csv": (OBSERVABLE_FIELDS, "benchmark_observables"),
        "runtime.csv": (RUNTIME_FIELDS, "runtime"),
    }
    handles: dict[str, Any] = {}
    writers: dict[str, csv.DictWriter] = {}
    temporary_paths: dict[str, Path] = {}
    run_rows: list[dict[str, Any]] = []
    total = sum(_run_count(benchmark, runs_override) for benchmark in benchmarks)
    processed = 0
    consolidation_started = time.monotonic()
    last_report_time = consolidation_started
    print(f"[consolidate] Writing CSV files from {total} checkpoints...", flush=True)
    try:
        for filename, (fields, _) in specifications.items():
            final_path = output_dir / filename
            temporary = final_path.with_suffix(final_path.suffix + ".tmp")
            handle = temporary.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            handles[filename] = handle
            writers[filename] = writer
            temporary_paths[filename] = temporary

        # Stream one compressed checkpoint at a time.  The full matrix contains
        # about 100 million trajectory rows, so collecting payloads in memory is
        # not safe during final consolidation.
        for benchmark in benchmarks:
            for replication_id in range(1, _run_count(benchmark, runs_override) + 1):
                payload = _read_checkpoint(
                    _checkpoint_path(output_dir, benchmark.name, replication_id)
                )
                for filename, (_, payload_key) in specifications.items():
                    rows = payload.get(payload_key, [])
                    writers[filename].writerows(rows)
                    if payload_key == "runs":
                        run_rows.extend(rows)
                processed += 1
                now = time.monotonic()
                if (
                    processed == total
                    or _report_due(processed, progress_every, last_report_time, progress_seconds)
                ):
                    elapsed = now - consolidation_started
                    eta = elapsed / processed * (total - processed) if processed else math.nan
                    print(
                        f"[consolidate] {processed}/{total} checkpoints "
                        f"({100.0 * processed / total:.1f}%); "
                        f"remaining={total - processed}; "
                        f"elapsed={_format_duration(elapsed)}; ETA={_format_duration(eta)}",
                        flush=True,
                    )
                    last_report_time = now
    finally:
        for handle in handles.values():
            handle.close()
    for filename, temporary in temporary_paths.items():
        temporary.replace(output_dir / filename)
    return run_rows


def _print_summary(
    run_rows: Sequence[dict[str, Any]],
    benchmarks: Sequence[Benchmark],
    runs_override: int | None,
    output_dir: Path,
) -> None:
    print("benchmark                                      Full completed   MicroMacro completed")
    for benchmark in benchmarks:
        runs = _run_count(benchmark, runs_override)
        full_count = sum(row["completion"] for row in run_rows if row["benchmark"] == benchmark.name and row["simulator"] == FULL)
        mm_count = sum(row["completion"] for row in run_rows if row["benchmark"] == benchmark.name and row["simulator"] == MICROMACRO)
        print(f"{benchmark.name:<46} {full_count}/{runs:<12} {mm_count}/{runs}")
    print(f"\nData saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON scenario configuration; omit to use the built-in main_clean matrix",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="override replications for every configuration (default: article counts)",
    )
    parser.add_argument("--smoke-runs", type=int, default=3, choices=range(3, 6))
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print run/consolidation progress after this many replications",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=30.0,
        help="also print progress after this many seconds without an update",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs is not None and args.runs < 1:
        raise ValueError("--runs must be positive")
    if args.resume and args.smoke_only:
        raise ValueError("--resume and --smoke-only cannot be used together")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.progress_seconds <= 0:
        raise ValueError("--progress-seconds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = _load_benchmarks(args.config)

    planned_total = sum(_run_count(item, args.runs) for item in benchmarks)
    print(
        f"[setup] {len(set(item.family for item in benchmarks))} benchmark families; "
        f"{len(benchmarks)} unique configurations; "
        f"{planned_total} paired replications; workers={args.workers}",
        flush=True,
    )
    print("[setup] Running shared-RNG pathwise regression gate...", flush=True)
    regression_passed, regression_detail = _regression_gate()
    if not regression_passed:
        raise RuntimeError(f"independent-RNG validation failed: {regression_detail}")
    print(f"[setup] Regression gate: PASS ({regression_detail})", flush=True)

    if args.resume:
        print("[smoke] Resume mode: smoke tests skipped.", flush=True)
        final_benchmarks = list(benchmarks)
        adjustments: list[dict[str, Any]] = []
    else:
        final_benchmarks, smoke_rows, adjustments = _run_smoke(
            benchmarks, args.smoke_runs, args.workers, args.runs
        )
        _write_csv(args.output_dir / "smoke_test_results.csv", SMOKE_FIELDS, smoke_rows)
    status = "smoke_complete" if args.smoke_only else "running"
    manifest = _manifest(
        args.runs,
        args.smoke_runs,
        args.workers,
        final_benchmarks,
        regression_detail,
        adjustments,
        status,
    )
    manifest["scenario_source"] = (
        str(args.config.resolve()) if args.config is not None else "built-in main_clean matrix"
    )
    manifest["smoke_skipped_on_resume"] = bool(args.resume)
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_readme(args.output_dir, manifest)
    if args.smoke_only:
        print("Smoke test: PASS")
        print(f"Data saved to: {args.output_dir}")
        return

    existing = [
        path
        for benchmark in final_benchmarks
        for path in (args.output_dir / benchmark.name / "checkpoints").glob("rep_*.json.gz")
    ]
    if existing and not args.resume:
        raise RuntimeError("current-matrix checkpoints already exist; use --resume or choose a different --output-dir")

    pending = [
        task
        for task in _tasks(final_benchmarks, args.runs)
        if not _checkpoint_path(args.output_dir, task[0]["name"], task[1]).exists()
    ]
    restored = planned_total - len(pending)
    planned_by_scenario = {
        benchmark.name: _run_count(benchmark, args.runs)
        for benchmark in final_benchmarks
    }
    pending_by_scenario = {name: 0 for name in planned_by_scenario}
    for task in pending:
        pending_by_scenario[task[0]["name"]] += 1
    done_by_scenario = {
        name: planned_by_scenario[name] - pending_by_scenario[name]
        for name in planned_by_scenario
    }
    print(
        f"[run] Checkpoints available: {restored}/{planned_total}; "
        f"remaining={len(pending)}",
        flush=True,
    )
    if restored:
        for name, done in done_by_scenario.items():
            if done:
                print(
                    f"[resume] {name}: {done}/{planned_by_scenario[name]} done; "
                    f"remaining={planned_by_scenario[name] - done}",
                    flush=True,
                )
    run_started = time.monotonic()
    last_report_time = run_started
    completed_since_start = 0
    for payload in _execute(pending, args.workers):
        path = _checkpoint_path(args.output_dir, payload["benchmark"], payload["replication_id"])
        _write_checkpoint(path, payload)
        completed_since_start += 1
        scenario = payload["benchmark"]
        done_by_scenario[scenario] += 1
        overall_completed = restored + completed_since_start
        now = time.monotonic()
        if done_by_scenario[scenario] == planned_by_scenario[scenario]:
            print(
                f"[scenario] COMPLETE {scenario}: "
                f"{planned_by_scenario[scenario]}/{planned_by_scenario[scenario]}; "
                f"overall remaining={planned_total - overall_completed}",
                flush=True,
            )
        if (
            overall_completed == planned_total
            or _report_due(
                completed_since_start,
                args.progress_every,
                last_report_time,
                args.progress_seconds,
            )
        ):
            elapsed = now - run_started
            eta = (
                elapsed / completed_since_start * (planned_total - overall_completed)
                if completed_since_start
                else math.nan
            )
            print(
                f"[run] {overall_completed}/{planned_total} paired replications "
                f"({100.0 * overall_completed / planned_total:.1f}%); "
                f"remaining={planned_total - overall_completed}; "
                f"scenario={scenario} "
                f"{done_by_scenario[scenario]}/{planned_by_scenario[scenario]}; "
                f"last_rep={payload['replication_id']}; "
                f"elapsed={_format_duration(elapsed)}; ETA={_format_duration(eta)}",
                flush=True,
            )
            last_report_time = now

    if not pending:
        print("[run] No simulations pending; rebuilding consolidated CSV files.", flush=True)
    run_rows = _consolidate(
        args.output_dir,
        final_benchmarks,
        args.runs,
        args.progress_every,
        args.progress_seconds,
    )
    manifest["status"] = "complete"
    manifest["updated_at_utc"] = datetime.now(UTC).isoformat()
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_readme(args.output_dir, manifest)
    _print_summary(run_rows, final_benchmarks, args.runs, args.output_dir)


if __name__ == "__main__":
    main()
