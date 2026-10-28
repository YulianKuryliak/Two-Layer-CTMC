from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from .config import ExperimentConfig, HazardMode, ModelVariant, PlacementMode, VARIANT_MODES


SUSCEPTIBLE = 0
INFECTED = 1
REMOVED = 2
Pair = Tuple[int, int]
Edge = Tuple[int, int, float]


def _weighted_choice(items: Sequence, weights: Sequence[float], rng: random.Random):
    total = float(sum(weights))
    if not items or total <= 0.0:
        return None
    threshold = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += float(weight)
        if cumulative > threshold:
            return item
    return items[-1]


@dataclass(frozen=True)
class InterfaceIndex:
    node_to_community: Mapping[int, int]
    internal_edges: Tuple[Edge, ...]
    directed_edges: Mapping[Pair, Tuple[Edge, ...]]
    static_boundary_weights: Mapping[Pair, Mapping[int, float]]
    W: Mapping[Pair, float]
    communities: Tuple[Tuple[int, ...], ...]

    @classmethod
    def from_graph(cls, full_graph: nx.Graph, communities: Sequence[Sequence[int]]) -> "InterfaceIndex":
        node_to_community = {
            int(node): community
            for community, nodes in enumerate(communities)
            for node in nodes
        }
        internal: List[Edge] = []
        directed: Dict[Pair, List[Edge]] = {}
        for u_raw, v_raw, data in full_graph.edges(data=True):
            u, v = int(u_raw), int(v_raw)
            weight = float(data.get("weight", 1.0))
            cu, cv = node_to_community[u], node_to_community[v]
            if cu == cv:
                internal.append((u, v, weight))
                continue
            directed.setdefault((cu, cv), []).append((u, v, weight))
            directed.setdefault((cv, cu), []).append((v, u, weight))

        boundary: Dict[Pair, Dict[int, float]] = {}
        W: Dict[Pair, float] = {}
        for pair, edges in directed.items():
            target_weights: Dict[int, float] = {}
            for _, target, weight in edges:
                target_weights[target] = target_weights.get(target, 0.0) + weight
            boundary[pair] = target_weights
            W[pair] = sum(weight for _, _, weight in edges)
        return cls(
            node_to_community=node_to_community,
            internal_edges=tuple(internal),
            directed_edges={pair: tuple(edges) for pair, edges in directed.items()},
            static_boundary_weights=boundary,
            W=W,
            communities=tuple(tuple(int(n) for n in nodes) for nodes in communities),
        )


class HazardPlacementSimulator:
    """Exact event-competition scheduler for the isolated hazard/placement ablation."""

    def __init__(
        self,
        full_graph: nx.Graph,
        communities: Sequence[Sequence[int]],
        config: ExperimentConfig,
        variant: ModelVariant,
        stream_seed_map: Mapping[str, int],
        record_events: bool = True,
        record_diagnostics: bool = True,
        record_trajectories: bool = True,
    ):
        if variant == ModelVariant.FULL_MICRO_REFERENCE:
            raise ValueError("Full Micro remains a separate production reference")
        self.config = config
        self.variant = variant
        self.hazard_mode, self.placement_mode = VARIANT_MODES[variant]
        self.index = InterfaceIndex.from_graph(full_graph, communities)
        self.micro_rng = random.Random(stream_seed_map["micro_event_rng"])
        self.macro_budget_rng = random.Random(stream_seed_map["macro_budget_rng"])
        self.macro_pair_rng = random.Random(stream_seed_map["macro_pair_rng"])
        self.placement_rng = random.Random(stream_seed_map["placement_rng"])
        self.record_events = bool(record_events)
        self.record_diagnostics = bool(record_diagnostics)
        self.record_trajectories = bool(record_trajectories)

    def exact_pair_hazards(self, state: Mapping[int, int]) -> Dict[Pair, float]:
        return {
            pair: self.config.beta_out * sum(
                weight
                for source, target, weight in edges
                if state[source] == INFECTED and state[target] == SUSCEPTIBLE
            )
            for pair, edges in self.index.directed_edges.items()
        }

    def aggregate_pair_hazards(self, state: Mapping[int, int]) -> Dict[Pair, float]:
        counts = self._community_counts(state)
        hazards: Dict[Pair, float] = {}
        for pair, weight_sum in self.index.W.items():
            source, target = pair
            susceptible_boundary = self._susceptible_boundary_weights(pair, state)
            if self.placement_mode == PlacementMode.BOUNDARY_WEIGHTED and not susceptible_boundary:
                hazards[pair] = 0.0
                continue
            n_source = len(self.index.communities[source])
            n_target = len(self.index.communities[target])
            hazards[pair] = (
                self.config.beta_out
                * weight_sum
                * counts[source][1] / n_source
                * counts[target][0] / n_target
            )
        return hazards

    def exact_target_weights(self, pair: Pair, state: Mapping[int, int]) -> Dict[int, float]:
        weights: Dict[int, float] = {}
        for source, target, weight in self.index.directed_edges.get(pair, ()):
            if state[source] == INFECTED and state[target] == SUSCEPTIBLE:
                weights[target] = weights.get(target, 0.0) + weight
        return weights

    def _susceptible_boundary_weights(self, pair: Pair, state: Mapping[int, int]) -> Dict[int, float]:
        return {
            node: weight
            for node, weight in self.index.static_boundary_weights.get(pair, {}).items()
            if state[node] == SUSCEPTIBLE
        }

    def _community_counts(self, state: Mapping[int, int]) -> List[Tuple[int, int, int]]:
        output = []
        for nodes in self.index.communities:
            output.append(tuple(sum(state[node] == status for node in nodes) for status in (SUSCEPTIBLE, INFECTED, REMOVED)))
        return output

    def _micro_channels(self, state: Mapping[int, int]):
        channels = []
        weights = []
        for u, v, edge_weight in self.index.internal_edges:
            if state[u] == INFECTED and state[v] == SUSCEPTIBLE:
                channels.append(("infection", u, v))
                weights.append(self.config.beta_in * edge_weight)
            elif state[v] == INFECTED and state[u] == SUSCEPTIBLE:
                channels.append(("infection", v, u))
                weights.append(self.config.beta_in * edge_weight)
        if self.config.gamma > 0.0:
            for node, status in state.items():
                if status == INFECTED:
                    channels.append(("recovery", node, None))
                    weights.append(self.config.gamma)
        return channels, weights

    def _select_target(self, pair: Pair, state: Mapping[int, int]) -> Tuple[Optional[int], Optional[int], int]:
        destination = pair[1]
        boundary_susceptible = self._susceptible_boundary_weights(pair, state)
        if self.placement_mode == PlacementMode.EXACT_EDGE_CONDITIONED:
            target_weights = self.exact_target_weights(pair, state)
        elif self.placement_mode == PlacementMode.BOUNDARY_WEIGHTED:
            target_weights = boundary_susceptible
        else:
            target_weights = {
                node: 1.0
                for node in self.index.communities[destination]
                if state[node] == SUSCEPTIBLE
            }
        target = _weighted_choice(list(target_weights), list(target_weights.values()), self.placement_rng)
        source_node = None
        if target is not None and self.placement_mode == PlacementMode.EXACT_EDGE_CONDITIONED:
            source_edges = [
                (source, weight)
                for source, edge_target, weight in self.index.directed_edges[pair]
                if edge_target == target and state[source] == INFECTED
            ]
            source_node = _weighted_choice(
                [source for source, _ in source_edges],
                [weight for _, weight in source_edges],
                self.placement_rng,
            )
        return target, source_node, len(boundary_susceptible)

    def run(self, initial_node: int) -> dict:
        state = {node: SUSCEPTIBLE for node in self.index.node_to_community}
        state[int(initial_node)] = INFECTED
        t = 0.0
        arrivals: List[Optional[float]] = [None] * len(self.index.communities)
        arrivals[self.index.node_to_community[int(initial_node)]] = 0.0
        event_log = []
        diagnostics = []
        trajectories = []
        counters = {
            "micro_events": 0,
            "macro_events": 0,
            "imports": 0,
            "repeated_imports": 0,
            "premature_hazard_opportunities": 0,
            "premature_hazard_pair_duration": 0.0,
            "undefined_exact_placement_events": 0,
            "undefined_exact_placement_opportunities": 0,
            "null_events": 0,
        }
        if self.record_trajectories:
            self._append_trajectory(trajectories, t, state)

        while t < self.config.T_end:
            micro_channels, micro_weights = self._micro_channels(state)
            micro_total = float(sum(micro_weights))
            exact_hazards = self.exact_pair_hazards(state)
            aggregate_hazards = self.aggregate_pair_hazards(state)
            selected_hazards = exact_hazards if self.hazard_mode == HazardMode.EXACT_ACTIVE_EDGE else aggregate_hazards
            macro_total = float(sum(selected_hazards.values()))

            dt_micro = self.micro_rng.expovariate(micro_total) if micro_total > 0.0 else math.inf
            dt_macro = self.macro_budget_rng.expovariate(macro_total) if macro_total > 0.0 else math.inf
            dt = min(dt_micro, dt_macro, self.config.T_end - t)
            self._append_diagnostics(diagnostics, t, dt, exact_hazards, aggregate_hazards, state, counters)
            if not math.isfinite(min(dt_micro, dt_macro)) or t + min(dt_micro, dt_macro) > self.config.T_end:
                t = self.config.T_end
                break
            t += min(dt_micro, dt_macro)

            if dt_micro < dt_macro:
                event = _weighted_choice(micro_channels, micro_weights, self.micro_rng)
                if event is None:
                    counters["null_events"] += 1
                    continue
                event_type, source, target = event
                if event_type == "infection":
                    state[target] = INFECTED
                    community = self.index.node_to_community[target]
                    if arrivals[community] is None:
                        arrivals[community] = t
                else:
                    state[source] = REMOVED
                    community = self.index.node_to_community[source]
                counters["micro_events"] += 1
                if self.record_events:
                    event_log.append({
                        "time": t, "event_type": event_type, "src_community": community,
                        "dst_community": community, "src_node": source, "dst_node": target,
                        "susceptible_boundary_nodes": None, "exact_placement_defined": None,
                    })
            else:
                pairs = list(selected_hazards)
                pair = _weighted_choice(pairs, [selected_hazards[p] for p in pairs], self.macro_pair_rng)
                counters["macro_events"] += 1
                if pair is None:
                    counters["null_events"] += 1
                    continue
                exact_defined = exact_hazards.get(pair, 0.0) > 0.0
                if not exact_defined:
                    counters["undefined_exact_placement_events"] += 1
                target, source_node, boundary_count = self._select_target(pair, state)
                if target is None:
                    counters["null_events"] += 1
                    if self.record_events:
                        event_log.append({
                            "time": t, "event_type": "null_import", "src_community": pair[0],
                            "dst_community": pair[1], "src_node": source_node, "dst_node": None,
                            "susceptible_boundary_nodes": boundary_count, "exact_placement_defined": exact_defined,
                        })
                    continue
                if arrivals[pair[1]] is not None:
                    counters["repeated_imports"] += 1
                state[target] = INFECTED
                if arrivals[pair[1]] is None:
                    arrivals[pair[1]] = t
                counters["imports"] += 1
                if self.record_events:
                    event_log.append({
                        "time": t, "event_type": "import", "src_community": pair[0],
                        "dst_community": pair[1], "src_node": source_node, "dst_node": target,
                        "susceptible_boundary_nodes": boundary_count, "exact_placement_defined": exact_defined,
                    })
            if self.record_trajectories:
                self._append_trajectory(trajectories, t, state)

        return {
            "variant": self.variant.value,
            "arrivals": arrivals,
            "event_log": event_log,
            "hazard_diagnostics": diagnostics,
            "trajectories": trajectories,
            "counters": counters,
        }

    def _append_trajectory(self, rows: List[dict], time: float, state: Mapping[int, int]) -> None:
        for community, (s, i, r) in enumerate(self._community_counts(state)):
            rows.append({"time": time, "community": community, "S": s, "I": i, "R": r})

    def _append_diagnostics(
        self,
        rows: List[dict],
        time: float,
        duration: float,
        exact: Mapping[Pair, float],
        aggregate: Mapping[Pair, float],
        state: Mapping[int, int],
        counters: dict,
    ) -> None:
        for pair in sorted(self.index.directed_edges):
            exact_value = exact.get(pair, 0.0)
            aggregate_value = aggregate.get(pair, 0.0)
            premature = aggregate_value > 0.0 and exact_value <= 0.0
            boundary_count = len(self._susceptible_boundary_weights(pair, state))
            if premature:
                counters["premature_hazard_opportunities"] += 1
                counters["premature_hazard_pair_duration"] += duration
                counters["undefined_exact_placement_opportunities"] += 1
            if self.record_diagnostics:
                rows.append({
                    "time": time,
                    "duration": duration,
                    "src_community": pair[0],
                    "dst_community": pair[1],
                    "aggregate_hazard": aggregate_value,
                    "exact_hazard": exact_value,
                    "signed_hazard_error_contribution": (aggregate_value - exact_value) * duration,
                    "absolute_hazard_error_contribution": abs(aggregate_value - exact_value) * duration,
                    "aggregate_positive_exact_zero": premature,
                    "exact_placement_defined": exact_value > 0.0,
                    "susceptible_boundary_nodes": boundary_count,
                })
