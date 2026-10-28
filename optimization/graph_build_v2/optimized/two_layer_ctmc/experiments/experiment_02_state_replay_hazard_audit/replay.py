from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from ...engines.macro_engine import MacroEngine
from ...engines.micro_engine import MicroEngine
from .config import AuditConfig


Pair = Tuple[int, int]
SUSCEPTIBLE = 0
INFECTED = 1


@dataclass(frozen=True)
class InterfaceIndex:
    communities: Tuple[Tuple[int, ...], ...]
    node_to_community: Mapping[int, int]
    directed_edges: Mapping[Pair, Tuple[Tuple[int, int, float], ...]]
    source_boundary: Mapping[Pair, frozenset[int]]
    destination_boundary: Mapping[Pair, frozenset[int]]
    W: Mapping[Pair, float]

    @classmethod
    def from_graph(cls, graph: nx.Graph, communities: Sequence[Sequence[int]]) -> "InterfaceIndex":
        frozen = tuple(tuple(int(n) for n in nodes) for nodes in communities)
        node_to_community = {node: ci for ci, nodes in enumerate(frozen) for node in nodes}
        directed: Dict[Pair, List[Tuple[int, int, float]]] = {}
        for u0, v0, data in graph.edges(data=True):
            u, v = int(u0), int(v0)
            cu, cv = node_to_community[u], node_to_community[v]
            if cu == cv:
                continue
            weight = float(data.get("weight", 1.0))
            directed.setdefault((cu, cv), []).append((u, v, weight))
            directed.setdefault((cv, cu), []).append((v, u, weight))
        directed_frozen = {pair: tuple(edges) for pair, edges in sorted(directed.items())}
        return cls(
            communities=frozen,
            node_to_community=node_to_community,
            directed_edges=directed_frozen,
            source_boundary={pair: frozenset(u for u, _, _ in edges) for pair, edges in directed_frozen.items()},
            destination_boundary={pair: frozenset(v for _, v, _ in edges) for pair, edges in directed_frozen.items()},
            W={pair: float(sum(w for _, _, w in edges)) for pair, edges in directed_frozen.items()},
        )


def stable_graph_id(graph: nx.Graph, index: InterfaceIndex) -> str:
    payload = {
        "nodes": sorted(int(n) for n in graph.nodes()),
        "edges": sorted((min(int(u), int(v)), max(int(u), int(v)), float(d.get("weight", 1.0)))
                        for u, v, d in graph.edges(data=True)),
        "communities": index.communities,
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def stable_initial_state_id(nodes: Iterable[int], infected_node: int) -> str:
    payload = {"nodes": sorted(int(n) for n in nodes), "infected": [int(infected_node)], "recovered": []}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def stable_events_id(events: pd.DataFrame) -> str:
    cols = ["time", "kind", "src_node", "dst_node", "src_community", "dst_community"]
    normalized = events[cols].fillna("").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(normalized.encode()).hexdigest()


def exact_pair_hazard(state: Mapping[int, int], edges: Sequence[Tuple[int, int, float]], beta_out: float) -> float:
    return beta_out * sum(w for u, v, w in edges if state[u] == INFECTED and state[v] == SUSCEPTIBLE)


def aggregate_pair_hazard(I_source: int, S_destination: int, n_source: int, n_destination: int,
                          weight_sum: float, beta_macro: float, macro_T: float = 1.0) -> float:
    return beta_macro * macro_T * weight_sum * (I_source / n_source) * (S_destination / n_destination)


def classify_state(exact: float, aggregate: float, abs_tol: float, rel_tol: float) -> str:
    exact_zero = exact == 0.0
    aggregate_zero = aggregate == 0.0
    if exact_zero and aggregate_zero:
        return "both_zero"
    if exact_zero:
        return "premature_positive"
    if aggregate_zero:
        return "missed_positive"
    if math.isclose(exact, aggregate, abs_tol=abs_tol, rel_tol=rel_tol):
        return "approximate_match"
    return "overestimation" if aggregate > exact else "underestimation"


def epidemic_phase(infected_fraction: float, config: AuditConfig) -> str:
    if infected_fraction == 0.0:
        return "not_started"
    if infected_fraction <= config.phase_early_upper:
        return "early"
    if infected_fraction <= config.phase_middle_upper:
        return "middle"
    return "late"


def _counts(state: Mapping[int, int], index: InterfaceIndex) -> Tuple[List[int], List[int]]:
    infected = [sum(state[node] == INFECTED for node in nodes) for nodes in index.communities]
    susceptible = [len(nodes) - infected[ci] for ci, nodes in enumerate(index.communities)]
    return infected, susceptible


def _production_engines(graph: nx.Graph, W: np.ndarray, config: AuditConfig) -> Tuple[MicroEngine, MacroEngine]:
    micro = MicroEngine(config.beta_in, config.gamma, model=2, cache_neighbors=True)
    for node in graph.nodes():
        micro.add_node(int(node))
    for u, v, data in graph.edges(data=True):
        micro.add_edge(int(u), int(v), float(data.get("weight", 1.0)))
    macro = MacroEngine(W, config.beta_macro, config.macro_T, [config.n] * config.L)
    return micro, macro


def replay_replication(
    events: pd.DataFrame,
    graph: nx.Graph,
    W_matrix: np.ndarray,
    communities: Sequence[Sequence[int]],
    config: AuditConfig,
    replication_id: int,
    network_seed: int,
    initialization_seed: int,
    initial_node: int,
) -> Tuple[pd.DataFrame, dict]:
    index = InterfaceIndex.from_graph(graph, communities)
    expected_pairs = {(i, i + 1) for i in range(config.L - 1)} | {(i + 1, i) for i in range(config.L - 1)}
    if set(index.directed_edges) != expected_pairs:
        raise ValueError(f"unexpected directed interface set: {set(index.directed_edges) ^ expected_pairs}")
    if config.gamma != 0.0:
        raise ValueError("exact replay from Experiment 01 infection-only logs is guaranteed only for gamma=0")
    events = events.sort_values("time", kind="stable").reset_index(drop=True)
    seeds = events[events["kind"] == "seed"]
    if len(seeds) != 1 or float(seeds.iloc[0]["time"]) != 0.0 or int(seeds.iloc[0]["dst_node"]) != initial_node:
        raise ValueError(f"replication {replication_id}: invalid seed event")
    infections = events[events["kind"].isin(["intra", "inter"])].copy()
    if len(infections) + 1 != len(events):
        unknown = sorted(set(events["kind"]) - {"seed", "intra", "inter"})
        raise ValueError(f"replication {replication_id}: unsupported events {unknown}")
    beyond_horizon = int((infections["time"] > config.T_end).sum())
    infections = infections[infections["time"] <= config.T_end]

    state = {int(node): SUSCEPTIBLE for node in graph.nodes()}
    state[initial_node] = INFECTED
    first_infection = [math.nan] * config.L
    first_infection[index.node_to_community[initial_node]] = 0.0
    production_micro, production_macro = _production_engines(graph, W_matrix, config)
    production_micro._infect_node(initial_node)

    interval_rows: List[dict] = []
    semantic_checks = 0
    max_micro_rate_difference = 0.0
    max_macro_formula_difference = 0.0
    current_time = 0.0

    def append_interval(end_time: float, interval_id: int) -> None:
        nonlocal semantic_checks, max_micro_rate_difference, max_macro_formula_difference
        duration = float(end_time - current_time)
        if duration < -1e-12:
            raise ValueError("event times are not monotone")
        if duration <= 0.0:
            return
        I_counts, S_counts = _counts(state, index)
        brute_total = config.beta_in * sum(
            float(data.get("weight", 1.0))
            for u, v, data in graph.edges(data=True)
            if state[int(u)] != state[int(v)]
        )
        micro_difference = abs(brute_total - production_micro.total_infection_rate)
        max_micro_rate_difference = max(max_micro_rate_difference, micro_difference)
        if micro_difference > 1e-10:
            raise AssertionError(f"production MicroEngine rate mismatch: {micro_difference}")
        production_macro.update_hazards(I_counts, S_counts)
        for pair, edges in index.directed_edges.items():
            source, destination = pair
            exact = exact_pair_hazard(state, edges, config.beta_out)
            aggregate = aggregate_pair_hazard(
                I_counts[source], S_counts[destination], len(index.communities[source]),
                len(index.communities[destination]), index.W[pair], config.beta_macro, config.macro_T,
            )
            macro_difference = abs(aggregate - float(production_macro.hazards[source, destination]))
            max_macro_formula_difference = max(max_macro_formula_difference, macro_difference)
            if macro_difference > 1e-14:
                raise AssertionError(f"production MacroEngine formula mismatch: {macro_difference}")
            active = [(u, v, w) for u, v, w in edges if state[u] == INFECTED and state[v] == SUSCEPTIBLE]
            signed = aggregate - exact
            denom = aggregate + exact
            src_first, dst_first = first_infection[source], first_infection[destination]
            interval_rows.append({
                "replication_id": replication_id, "interval_id": interval_id,
                "time_start": current_time, "time_end": end_time, "duration": duration,
                "src_community": source, "dst_community": destination,
                "interface_index": min(source, destination), "layer_index": destination,
                "network_seed": network_seed, "initialization_seed": initialization_seed,
                "initial_node": initial_node,
                "exact_hazard": exact, "aggregate_hazard": aggregate,
                "signed_error": signed, "absolute_error": abs(signed),
                "symmetric_relative_error": (2.0 * abs(signed) / denom) if denom > 0 else math.nan,
                "state_category": classify_state(exact, aggregate, config.match_abs_tolerance, config.match_rel_tolerance),
                "source_infected_fraction": I_counts[source] / len(index.communities[source]),
                "destination_susceptible_fraction": S_counts[destination] / len(index.communities[destination]),
                "infected_source_boundary_nodes": sum(state[u] == INFECTED for u in index.source_boundary[pair]),
                "susceptible_destination_boundary_nodes": sum(state[v] == SUSCEPTIBLE for v in index.destination_boundary[pair]),
                "active_intercommunity_edge_count": len(active),
                "active_intercommunity_edge_weight": sum(w for _, _, w in active),
                "time_since_source_first_infection": current_time - src_first if math.isfinite(src_first) else math.nan,
                "time_since_destination_first_infection": current_time - dst_first if math.isfinite(dst_first) else math.nan,
                "epidemic_phase": epidemic_phase(I_counts[source] / len(index.communities[source]), config),
            })
            semantic_checks += 1

    for event_number, event in enumerate(infections.itertuples(index=False), start=1):
        event_time = float(event.time)
        append_interval(event_time, event_number - 1)
        dst = int(event.dst_node)
        src = int(event.src_node)
        if state.get(src) != INFECTED or state.get(dst) != SUSCEPTIBLE or not graph.has_edge(src, dst):
            raise AssertionError(f"replication {replication_id}: event {event_number} cannot occur in replay state")
        source_community = index.node_to_community[src]
        destination_community = index.node_to_community[dst]
        expected_kind = "intra" if source_community == destination_community else "inter"
        if event.kind != expected_kind or int(event.src_community) != source_community or int(event.dst_community) != destination_community:
            raise AssertionError(f"replication {replication_id}: event metadata mismatch")
        state[dst] = INFECTED
        production_micro._infect_node(dst)
        if not math.isfinite(first_infection[destination_community]):
            first_infection[destination_community] = event_time
        current_time = event_time
    append_interval(config.T_end, len(infections))
    if not interval_rows:
        raise AssertionError("replay produced no positive-duration intervals")
    frame = pd.DataFrame(interval_rows)
    if not np.isfinite(frame.drop(columns=["symmetric_relative_error", "time_since_source_first_infection",
                                           "time_since_destination_first_infection"])
                       .select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("non-finite required replay quantity")
    metadata = {
        "replication_id": replication_id,
        "network_seed": network_seed,
        "initialization_seed": initialization_seed,
        "initial_node": initial_node,
        "network_id": stable_graph_id(graph, index),
        "initial_state_id": stable_initial_state_id(graph.nodes(), initial_node),
        "source_events_id": stable_events_id(events),
        "source_event_rows": int(len(events)),
        "events_replayed_through_horizon": int(len(infections)),
        "events_beyond_horizon_ignored": beyond_horizon,
        "positive_duration_intervals": int(frame["interval_id"].nunique()),
        "directed_pair_interval_rows": int(len(frame)),
        "semantic_checks": semantic_checks,
        "max_micro_rate_difference": max_micro_rate_difference,
        "max_macro_formula_difference": max_macro_formula_difference,
    }
    return frame, metadata
